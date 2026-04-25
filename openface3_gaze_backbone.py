# openface3_gaze_backbone.py
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenFace3GazeBackbone(nn.Module):
    """
    Deep feature extractor from OpenFace-3.0's multitask backbone.

    Input:  x  -> (BT, 3, H, W)  RGB float in [0,1] or [0,255]
    Output: (BT, 128, 14, 14)    deep features (NOT gaze angles)
    """
    def __init__(self,
                 repo_root: str,
                 device: str = 'cuda',
                 weights_path: str = None,  # defaults to <repo_root>/weights/MTL_backbone.pth
                 out_ch: int = 128,
                 out_spa: int = 14,
                 input_size: int = 224):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_ch = out_ch
        self.out_spa = out_spa
        self.input_size = input_size

        # ---------- Ensure we can import OpenFace-3.0/model as a package ----------
        if not os.path.isdir(repo_root):
            raise FileNotFoundError(f"repo_root not found: {repo_root}")

        model_dir = os.path.join(repo_root, "model")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Expected directory not found: {model_dir}")

        init_file = os.path.join(model_dir, "__init__.py")
        if not os.path.exists(init_file):
            # Auto-create the package marker so relative imports inside MLT.py work
            os.makedirs(model_dir, exist_ok=True)
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# package init for OpenFace-3.0/model\n")

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            from model.MLT import MLT  # relative imports inside MLT.py will now resolve
        except Exception as e:
            raise ImportError(
                "Could not import model.MLT. Confirm repo_root points to your OpenFace-3.0 folder."
            ) from e

        # ---------- Build model + load weights ----------
        self.mlt = MLT(base_model_name='tf_efficientnet_b0_ns', expr_classes=8, au_numbers=8)

        ckpt_path = weights_path or os.path.join(repo_root, 'weights', 'MTL_backbone.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Weights not found at: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Handle possible checkpoint formats
        state = ckpt.get('state_dict', ckpt)
        def _strip_prefix(d, pfxs=('module.', 'model.')):
            out = {}
            for k, v in d.items():
                for p in pfxs:
                    if k.startswith(p):
                        k = k[len(p):]
                out[k] = v
            return out
        state = _strip_prefix(state)

        missing, unexpected = self.mlt.load_state_dict(state, strict=False)
        if hasattr(missing, "__len__") and len(missing) > 0:
            print(f"[OpenFace3GazeBackbone] Missing keys (ok if only heads differ): {len(missing)}")
        if hasattr(unexpected, "__len__") and len(unexpected) > 0:
            print(f"[OpenFace3GazeBackbone] Unexpected keys: {len(unexpected)}")

        # ---------- Use the EfficientNet trunk (timm model) ----------
        self.backbone = self.mlt.base_model   # timm model
        self.backbone.eval()

        # Try to infer channels of spatial feature map (EfficientNet-B0 typically 1280)
        C = None
        if hasattr(self.backbone, 'feature_info'):
            try:
                C = int(self.backbone.feature_info.channels()[-1])
            except Exception:
                C = None
        if C is None and hasattr(self.backbone, 'num_features'):
            C = int(self.backbone.num_features)
        if C is None:
            # last-resort: run a tiny dummy forward to get C
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
                dummy = (dummy - torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)) / \
                        torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
                feat = self.backbone.forward_features(dummy)
                C = int(feat.shape[1])

        # Project to 128 channels and resize to 14x14
        self.proj = nn.Conv2d(C, self.out_ch, kernel_size=1, bias=False)

        # ImageNet normalization for timm EfficientNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # Freeze everything by default
        for p in self.parameters():
            p.requires_grad = False

        self.to(self.device).eval()

    @torch.no_grad()
    def _preproc(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BT,3,H,W) in [0,1] or [0,255]
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=True)
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        x = self._preproc(x)

        # Spatial feature map before classification heads
        feat = self.backbone.forward_features(x)   # (BT, C, h, w) e.g., (BT, 1280, 7, 7)

        y = self.proj(feat)                        # (BT, 128, h, w)
        if y.shape[2] != self.out_spa or y.shape[3] != self.out_spa:
            y = F.interpolate(y, size=(self.out_spa, self.out_spa),
                               mode='bilinear', align_corners=False)
        return y
