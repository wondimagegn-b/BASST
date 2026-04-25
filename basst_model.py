import os
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from timm.models.vision_transformer import Attention, Block as DefaultBlock


def token2feature(tokens: torch.Tensor) -> torch.Tensor:
    b, n, d = tokens.shape
    h = w = int(n ** 0.5)
    x = tokens.permute(0, 2, 1).contiguous().view(b, d, h, w)
    return x


def feature2token(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    tokens = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
    return tokens


class DynamicWeightingModule4(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.alpha_s = nn.Parameter(torch.tensor(float(init)))
        self.alpha_l = nn.Parameter(torch.tensor(float(init)))
        self.alpha_a = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor, s: torch.Tensor, l: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return x + self.alpha_s * s + self.alpha_l * l + self.alpha_a * a


class SelectiveScan1D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.to_delta = nn.Linear(d_model, d_model)
        self.to_B = nn.Linear(d_model, d_model)
        self.to_C = nn.Linear(d_model, d_model)
        self.to_gate = nn.Linear(d_model, d_model)
        self.softplus = nn.Softplus()

        for layer in [self.to_delta, self.to_B, self.to_C, self.to_gate]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @torch.cuda.amp.autocast(enabled=False)
    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.float()
        a = -self.softplus(self.A_log).to(x.device).float()

        h = torch.zeros(batch_size, dim, device=x.device, dtype=torch.float32)
        out = torch.empty(batch_size, seq_len, dim, device=x.device, dtype=torch.float32)

        for t in range(seq_len):
            x_t = x[:, t, :]
            delta = self.softplus(self.to_delta(x_t)).clamp_(min=1e-4, max=3.0)
            b_t = torch.tanh(self.to_B(x_t)) * 1.5
            c_t = torch.tanh(self.to_C(x_t)) * 1.5
            g_t = torch.sigmoid(self.to_gate(x_t))

            z = torch.clamp(delta * a, min=-20.0, max=2.0)
            transition = torch.exp(z)
            h = transition * h + (delta * b_t) * x_t
            out[:, t, :] = g_t * (c_t * h)

        return out

    def forward(self, x: torch.Tensor, bidirectional: bool = False, fuse: str = "sum") -> torch.Tensor:
        y_fwd = self._scan(x)
        if not bidirectional:
            return y_fwd

        y_bwd = self._scan(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])

        if fuse == "sum":
            return y_fwd + y_bwd
        return torch.cat([y_fwd, y_bwd], dim=-1)


class TemporalSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_frames: int,
        ssm_dim: Optional[int] = None,
        bidirectional: bool = False,
        fuse: str = "sum",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_dim = d_model
        self.ssm_dim = ssm_dim or max(64, d_model // 4)

        self.pre = nn.Linear(self.in_dim, self.ssm_dim)
        self.core = SelectiveScan1D(self.ssm_dim)
        self.post = nn.Linear(self.ssm_dim, self.in_dim)

        nn.init.xavier_uniform_(self.pre.weight)
        nn.init.zeros_(self.pre.bias)
        nn.init.xavier_uniform_(self.post.weight)
        nn.init.zeros_(self.post.bias)

        self.bidirectional = bidirectional
        self.fuse = fuse

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        bt, n, d = x_tokens.shape
        t = self.num_frames
        if bt % t != 0:
            raise ValueError(f"Input length {bt} is not divisible by num_frames={t}.")
        b = bt // t

        x_seq = rearrange(x_tokens, "(b t) n d -> (b n) t d", b=b, t=t).contiguous()

        with torch.cuda.amp.autocast(enabled=False):
            x_red = self.pre(x_seq.float())

        y_red = self.core(x_red, bidirectional=self.bidirectional, fuse=self.fuse)

        with torch.cuda.amp.autocast(enabled=False):
            y_seq = self.post(y_red.float())

        y_tokens = rearrange(y_seq, "(b n) t d -> (b t) n d", b=b, t=t, n=n).contiguous()
        return y_tokens


class SpatioTemporalSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_frames: int,
        grid_size: Tuple[int, int] = None,
        temporal_fuse: str = "sum",
        ssm_ratio: float = 0.25,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__()
        ssm_dim = max(64, int(d_model * ssm_ratio))
        self.temporal = TemporalSSM(
            d_model=d_model,
            num_frames=num_frames,
            ssm_dim=ssm_dim,
            bidirectional=bidirectional,
            fuse=temporal_fuse,
        )
        self.grid_size = grid_size

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        return x_tokens + self.temporal(x_tokens)


class TMAdapter(nn.Module):
    def __init__(
        self,
        d_features: int,
        num_frames: int,
        ratio: float = 0.25,
        bidirectional: bool = False,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        ssm_dim = max(64, int(d_features * ratio))
        self.temporal = TemporalSSM(
            d_model=d_features,
            num_frames=num_frames,
            ssm_dim=ssm_dim,
            bidirectional=bidirectional,
            fuse="sum",
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.temporal(x)


class Prompt_block(nn.Module):
    def __init__(self, inplanes: int, hide_channel: Optional[int], num_frames: int, ratio: float = 0.25):
        super().__init__()
        self.num_frames = num_frames
        self.branch_dim = inplanes
        hid = hide_channel or (inplanes // 2)

        self.red_frame = nn.Conv2d(self.branch_dim, hid, kernel_size=1, bias=False)
        self.red_aux = nn.Conv2d(self.branch_dim, hid, kernel_size=1, bias=False)

        self.mix = nn.Sequential(
            nn.Conv2d(2 * hid, hid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hid, self.branch_dim, kernel_size=1, bias=True),
        )

        self.norm_f = nn.GroupNorm(1, hid)
        self.norm_a = nn.GroupNorm(1, hid)

        self.TMA = TMAdapter(self.branch_dim, num_frames=num_frames, ratio=ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        b, ctot, h, w = x.shape
        if ctot != 2 * self.branch_dim:
            raise ValueError(
                f"Prompt_block expects {2 * self.branch_dim} channels, but got {ctot}."
            )

        x_frame = x[:, :self.branch_dim, :, :]
        x_aux = x[:, self.branch_dim:, :, :]

        x2_tokens = x_frame.flatten(2).transpose(1, 2)
        x2 = self.TMA(x2_tokens)

        f = self.norm_f(self.red_frame(x_frame))
        a = self.norm_a(self.red_aux(x_aux))
        fused = torch.cat([f, a], dim=1)
        x0 = self.mix(fused)

        x0_tokens = x0.flatten(2).transpose(1, 2)
        x0_tma = self.TMA(x0_tokens)

        return x0, x2, x0_tma


class Prompt_PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=14,
        patch_size=16,
        in_chans=128,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // 1, img_size[1] // 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        _assert(h == self.img_size[0], f"Input height ({h}) doesn't match model ({self.img_size[0]}).")
        _assert(w == self.img_size[1], f"Input width ({w}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class I3DHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
    ):
        super().__init__()
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        nn.init.trunc_normal_(self.fc_cls.weight, std=0.02)
        nn.init.zeros_(self.fc_cls.bias)

    def forward(self, x: torch.Tensor):
        if self.spatial_type == "avg":
            x = x.mean(dim=[2, 3, 4])
        else:
            x = x[:, :, 0, 0, 0]

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score, x


class BASST(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        in_chans_l: int = 128,
        num_frames: int = 16,
        num_classes: int = 2,
        prompt_type: str = "deep",
        global_pool: str = "token",
        hidden_dim: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        adapter_scale: float = 0.25,
        head_dropout_ratio: float = 0.5,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values=None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"

        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.prompt_type = prompt_type

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_embed_prompt = Prompt_PatchEmbed(
            img_size=img_size // patch_size,
            patch_size=patch_size,
            in_chans=in_chans_l,
            embed_dim=embed_dim,
            norm_layer=None,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens

        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02, requires_grad=False)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        if self.prompt_type in ["shallow", "deep"]:
            block_nums = depth if self.prompt_type == "deep" else 1

            self.prompt_blocks = nn.ModuleList(
                [
                    Prompt_block(
                        inplanes=embed_dim,
                        hide_channel=hidden_dim,
                        num_frames=num_frames,
                        ratio=adapter_scale,
                    )
                    for _ in range(block_nums)
                ]
            )

            self.prompt_norms = nn.ModuleList([norm_layer(embed_dim) for _ in range(block_nums)])
            self.AttentionFusion = nn.ModuleList([DynamicWeightingModule4() for _ in range(depth)])
        else:
            self.prompt_blocks = nn.ModuleList()
            self.prompt_norms = nn.ModuleList()
            self.AttentionFusion = nn.ModuleList()

        self.full_tma_layers = nn.ModuleList(
            [
                SpatioTemporalSSM(
                    d_model=embed_dim,
                    num_frames=num_frames,
                    grid_size=(img_size // patch_size, img_size // patch_size),
                    temporal_fuse="sum",
                    ssm_ratio=adapter_scale,
                    bidirectional=False,
                )
                for _ in range(depth)
            ]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                DefaultBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.ln_post = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        self.head = I3DHead(
            num_classes=num_classes,
            in_channels=embed_dim,
            dropout_ratio=head_dropout_ratio,
        ) if num_classes > 0 else nn.Identity()

        self.weak_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.aux_mixer = nn.Identity()

        if weight_init != "skip":
            self.apply(self._init_weights)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_token_t, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.temporal_embedding, std=0.02)

        self.height = img_size // patch_size
        self.width = img_size // patch_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "temporal_embedding", "cls_token", "dist_token"}

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = I3DHead(
            num_classes=num_classes,
            in_channels=self.embed_dim,
            dropout_ratio=0.5,
        ) if num_classes > 0 else nn.Identity()
        self.weak_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, d: torch.Tensor):
        x = self.patch_embed(x)
        d = self.patch_embed_prompt(d)

        prompt_tokens = d

        if self.prompt_type in ["shallow", "deep"]:
            y = self.prompt_norms[0](x)
            x_feat = token2feature(y)

            d_feat = token2feature(self.prompt_norms[0](d))
            x_feat_cat = torch.cat([x_feat, d_feat], dim=1)

            s0_feat, l0_tok, a0_tok = self.prompt_blocks[0](x_feat_cat)
            s0_tok = feature2token(s0_feat)
            prompt_tokens = s0_tok

            x = self.AttentionFusion[0](x, s0_tok, l0_tok, a0_tok)
        else:
            x = x + d

        x = self._pos_embed(x)

        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        x = self.norm_pre(x)
        x = self.full_tma_layers[0](x)

        for i, blk in enumerate(self.blocks):
            if i >= 1 and self.prompt_type == "deep":
                x_ori = x
                x_norm_i = self.prompt_norms[i - 1](x)
                x_feat_i = token2feature(x_norm_i[:, 1:])

                prompt_feat_i = token2feature(self.prompt_norms[0](prompt_tokens))
                x_feat_cat = torch.cat([x_feat_i, prompt_feat_i], dim=1)

                si_feat, li_tok, ai_tok = self.prompt_blocks[i](x_feat_cat)
                si_tok = feature2token(si_feat)
                prompt_tokens = si_tok

                x_no_cls = self.AttentionFusion[i](x_ori[:, 1:], si_tok, li_tok, ai_tok)
                x = torch.cat([x_ori[:, :1], x_no_cls], dim=1)
                x = self.full_tma_layers[i](x)

            x = blk(x)

        x = self.ln_post(x)
        return x

    def forward_head(self, x: torch.Tensor):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        return x

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        b, c, t, h, w = x.shape
        if t != self.num_frames:
            raise ValueError(f"Input video must have {self.num_frames} frames, but got {t}.")

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.forward_features(x, d)
        x = self.forward_head(x)

        x = rearrange(x, "(b t) c -> b c t", b=b, t=t)
        x = x.unsqueeze(-1).unsqueeze(-1)
        score, feat = self.head(x)
        return score, feat, None


def load_pretrained_finetune(model: BASST, pretrained_path: str):
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = dict(state_dict)

    for key in list(state_dict.keys()):
        if key.startswith("landmark."):
            state_dict.pop(key)

    if "norm.weight" in state_dict and "ln_post.weight" not in state_dict:
        state_dict["ln_post.weight"] = state_dict.pop("norm.weight")
    if "norm.bias" in state_dict and "ln_post.bias" not in state_dict:
        state_dict["ln_post.bias"] = state_dict.pop("norm.bias")

    classifier_keys = [
        "head.weight",
        "head.bias",
        "head.fc_cls.weight",
        "head.fc_cls.bias",
        "weak_head.weight",
        "weak_head.bias",
    ]
    for key in classifier_keys:
        if key in state_dict:
            state_dict.pop(key)

    if "patch_embed_prompt.proj.weight" in state_dict:
        ckpt_weight = state_dict["patch_embed_prompt.proj.weight"]
        if tuple(ckpt_weight.shape) != tuple(model.patch_embed_prompt.proj.weight.shape):
            state_dict.pop("patch_embed_prompt.proj.weight", None)
            state_dict.pop("patch_embed_prompt.proj.bias", None)

    model.load_state_dict(state_dict, strict=False)


def build_model(args):
    model = BASST(
        img_size=getattr(args, "input_size", 224),
        patch_size=getattr(args, "patch_size", 16),
        in_chans=getattr(args, "in_channels", 3),
        in_chans_l=getattr(args, "aux_feature_dim", 128),
        num_frames=getattr(args, "num_frames", 16),
        num_classes=getattr(args, "nb_classes", 2),
        prompt_type=getattr(args, "prompt_type", "deep"),
        global_pool=getattr(args, "global_pool", "token"),
        hidden_dim=getattr(args, "prompt_hidden_dim", 8),
        embed_dim=getattr(args, "embed_dim", 768),
        depth=getattr(args, "depth", 12),
        adapter_scale=getattr(args, "scale_factor", 0.25),
        head_dropout_ratio=getattr(args, "head_dropout", getattr(args, "dropout_ratio", 0.5)),
        num_heads=getattr(args, "num_heads", 12),
        mlp_ratio=getattr(args, "mlp_ratio", 4.0),
        qkv_bias=getattr(args, "qkv_bias", True),
        drop_rate=getattr(args, "dropout", 0.0),
        attn_drop_rate=getattr(args, "attn_drop_rate", 0.0),
        drop_path_rate=getattr(args, "drop_path_rate", 0.0),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    finetune_path = getattr(args, "finetune_path", "")
    if not finetune_path:
        finetune_path = getattr(args, "finetune", "")

    if finetune_path:
        load_pretrained_finetune(model, finetune_path)

    return model