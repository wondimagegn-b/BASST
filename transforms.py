import math
import random
import numbers
from typing import List, Sequence, Tuple, Union, Optional

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from rand_augment import rand_augment_transform


ImageLike = Union[np.ndarray, Image.Image]


# =========================================================
# Basic clip helpers
# =========================================================
def _is_tensor_clip(clip: torch.Tensor) -> bool:
    return torch.is_tensor(clip) and clip.ndim == 4


def _get_resize_sizes(im_h: int, im_w: int, size: int) -> Tuple[int, int]:
    if im_w < im_h:
        out_w = size
        out_h = int(size * im_h / im_w)
    else:
        out_h = size
        out_w = int(size * im_w / im_h)
    return out_h, out_w


def crop_clip(
    clip: List[ImageLike],
    top: int,
    left: int,
    height: int,
    width: int,
) -> List[ImageLike]:
    if isinstance(clip[0], np.ndarray):
        return [img[top:top + height, left:left + width, :] for img in clip]
    if isinstance(clip[0], Image.Image):
        return [img.crop((left, top, left + width, top + height)) for img in clip]
    raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")


def resize_clip(
    clip: List[ImageLike],
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
) -> List[ImageLike]:
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, _ = clip[0].shape
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = _get_resize_sizes(im_h, im_w, int(size))
            size = (new_w, new_h)
        else:
            size = (size[1], size[0]) if len(size) == 2 else size

        np_inter = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        return [cv2.resize(img, size, interpolation=np_inter) for img in clip]

    if isinstance(clip[0], Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = _get_resize_sizes(im_h, im_w, int(size))
            size = (new_w, new_h)
        else:
            size = (size[1], size[0]) if len(size) == 2 else size

        pil_inter = Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
        return [img.resize(size, pil_inter) for img in clip]

    raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")


# =========================================================
# Compose-style transforms for list-of-frames
# =========================================================
class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class Resize:
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str = "bilinear"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip: List[ImageLike]) -> List[ImageLike]:
        return resize_clip(clip, self.size, interpolation=self.interpolation)


class CenterCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = size  # (h, w)

    def __call__(self, clip: List[ImageLike]) -> List[ImageLike]:
        crop_h, crop_w = self.size

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")

        if crop_w > im_w or crop_h > im_h:
            raise ValueError(
                f"Crop size {(crop_h, crop_w)} must be <= image size {(im_h, im_w)}"
            )

        top = int(round((im_h - crop_h) / 2.0))
        left = int(round((im_w - crop_w) / 2.0))
        return crop_clip(clip, top, left, crop_h, crop_w)


class ClipToTensor:
    """
    Convert a list of frames into a float tensor of shape (C, T, H, W),
    scaled to [0, 1] when div_255=True.
    """
    def __init__(self, div_255: bool = True):
        self.div_255 = div_255

    def __call__(self, clip: List[ImageLike]) -> torch.Tensor:
        if len(clip) == 0:
            raise ValueError("Clip is empty")

        frames = []
        for img in clip:
            if isinstance(img, Image.Image):
                img = np.array(img, copy=True)
            elif not isinstance(img, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(img)}")

            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)

            if img.shape[-1] != 3:
                raise ValueError(f"Expected 3-channel image, got shape {img.shape}")

            tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
            if self.div_255:
                tensor = tensor / 255.0
            frames.append(tensor)

        clip_tensor = torch.stack(frames, dim=1)  # C,T,H,W
        return clip_tensor


class Normalize:
    """
    Normalize a tensor clip of shape (C, T, H, W).
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if not _is_tensor_clip(clip):
            raise TypeError("Expected torch tensor clip with shape (C, T, H, W)")

        mean = torch.as_tensor(self.mean, dtype=clip.dtype, device=clip.device)[:, None, None, None]
        std = torch.as_tensor(self.std, dtype=clip.dtype, device=clip.device)[:, None, None, None]
        return (clip - mean) / std


# =========================================================
# Augmentation used by training path
# =========================================================
def create_random_augment(
    input_size: Tuple[int, int],
    auto_augment: Optional[str],
    interpolation: str = "bilinear",
):
    """
    Return a callable that applies RandAugment to a list of PIL images.
    """
    if not auto_augment:
        return lambda clip: clip

    aa_params = {
        "translate_const": int(min(input_size) * 0.45),
        "img_mean": (124, 116, 104),
        "interpolation": _pil_interp(interpolation),
    }
    return rand_augment_transform(auto_augment, aa_params)


def _pil_interp(method: str):
    if method == "bicubic":
        return Image.BICUBIC
    if method == "lanczos":
        return Image.LANCZOS
    if method == "hamming":
        return Image.HAMMING
    return Image.BILINEAR


# =========================================================
# Tensor normalization used in training _aug_frame path
# Input shape: (T, H, W, C)
# =========================================================
def tensor_normalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    return (tensor - mean) / std


# =========================================================
# Spatial sampling utilities
# Input/Output shape: (C, T, H, W)
# =========================================================
def random_short_side_scale_jitter(
    images: torch.Tensor,
    min_size: int,
    max_size: int,
    inverse_uniform_sampling: bool = False,
) -> Tuple[torch.Tensor, None]:
    if inverse_uniform_sampling:
        size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]

    if (width <= height and width == size) or (height <= width and height == size):
        return images, None

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    out = torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return out, None


def random_crop(images: torch.Tensor, size: int) -> Tuple[torch.Tensor, None]:
    if images.shape[2] == size and images.shape[3] == size:
        return images, None

    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0 if height <= size else int(np.random.randint(0, height - size))
    x_offset = 0 if width <= size else int(np.random.randint(0, width - size))

    cropped = images[:, :, y_offset:y_offset + size, x_offset:x_offset + size]
    return cropped, None


def horizontal_flip(prob: float, images: torch.Tensor) -> Tuple[torch.Tensor, None]:
    if np.random.uniform() < prob:
        images = images.flip((-1,))
    return images, None


def uniform_crop(images: torch.Tensor, size: int, spatial_idx: int) -> Tuple[torch.Tensor, None]:
    assert spatial_idx in [0, 1, 2]

    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size

    cropped = images[:, :, y_offset:y_offset + size, x_offset:x_offset + size]
    return cropped, None


def random_resized_crop(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    scale: Sequence[float],
    ratio: Sequence[float],
) -> torch.Tensor:
    """
    Minimal clean version for the training path.
    Input images: (C, T, H, W)
    """
    _, _, height, width = images.shape
    area = height * width

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)
            cropped = images[:, :, top:top + h, left:left + w]
            return torch.nn.functional.interpolate(
                cropped,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

    # fallback to center crop after resize
    min_side = min(height, width)
    scale_factor = max(target_height / min_side, target_width / min_side)
    resized = torch.nn.functional.interpolate(
        images,
        size=(int(round(height * scale_factor)), int(round(width * scale_factor))),
        mode="bilinear",
        align_corners=False,
    )
    cropped, _ = uniform_crop(resized, min(target_height, target_width), spatial_idx=1)
    if cropped.shape[2] != target_height or cropped.shape[3] != target_width:
        cropped = torch.nn.functional.interpolate(
            cropped,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return cropped


def spatial_sampling(
    frames: torch.Tensor,
    spatial_idx: int = -1,
    min_scale: int = 256,
    max_scale: int = 320,
    crop_size: int = 224,
    random_horizontal_flip: bool = True,
    inverse_uniform_sampling: bool = False,
    aspect_ratio: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    motion_shift: bool = False,
) -> torch.Tensor:
    """
    frames: (C, T, H, W)
    """
    assert spatial_idx in [-1, 0, 1, 2]

    if motion_shift:
        raise NotImplementedError("motion_shift=True is not needed in the cleaned DFEW pipeline")

    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            frames = random_resized_crop(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )

        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)

    return frames