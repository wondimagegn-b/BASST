import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from openface3_gaze_backbone import OpenFace3GazeBackbone


class OpenFaceFeatureExtractor(nn.Module):
    def __init__(
        self,
        repo_root: str,
        weights_path: str = None,
        device: str = "cuda",
        input_size: int = 224,
        out_channels: int = 128,
        out_size: int = 14,
    ):
        super().__init__()

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device_name = device
        self.input_size = input_size

        self.backbone = OpenFace3GazeBackbone(
            repo_root=repo_root,
            device=device,
            weights_path=weights_path,
            out_ch=out_channels,
            out_spa=out_size,
            input_size=input_size,
        ).to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(
                f"Expected input shape [B, C, T, H, W], got {tuple(videos.shape)}"
            )

        b, c, t, h, w = videos.shape
        frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        if frames.shape[-2:] != (self.input_size, self.input_size):
            frames = F.interpolate(
                frames,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        features = self.backbone(frames)
        return features


def build_auxiliary_extractor(args):
    repo_root = getattr(args, "openface_repo_root", None)
    if repo_root is None:
        repo_root = os.path.join(os.path.dirname(__file__), "OpenFace-3.0")

    return OpenFaceFeatureExtractor(
        repo_root=repo_root,
        weights_path=getattr(args, "openface_weights_path", None),
        device=getattr(args, "device", "cuda"),
        input_size=getattr(args, "input_size", 224),
        out_channels=getattr(args, "aux_feature_dim", 128),
        out_size=getattr(args, "aux_feature_size", 14),
    )