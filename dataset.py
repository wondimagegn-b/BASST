import glob
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

from transforms import (
    Compose,
    Resize,
    CenterCrop,
    ClipToTensor,
    Normalize,
    create_random_augment,
    tensor_normalize,
    spatial_sampling,
)
from random_erasing import RandomErasing


def build_dataset(is_train: bool, test_mode: bool, args):
    if is_train:
        mode = "train"
        anno_path = args.train_label_path
    elif test_mode:
        mode = "test"
        anno_path = args.test_label_path
    else:
        mode = "validation"
        anno_path = args.test_label_path

    dataset = ASDVideoDataset(
        anno_path=anno_path,
        data_path=args.data_path,
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        args=args,
        file_ext=getattr(args, "file_ext", "jpg"),
    )
    nb_classes = int(args.nb_classes)

    return dataset, nb_classes


class FrameFolderReader:
    def __init__(self, clip_dir: str, file_ext: str = "jpg"):
        self.frames = sorted(glob.glob(os.path.join(clip_dir, f"*.{file_ext}")))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Image.Image:
        frame_path = self.frames[index]
        with open(frame_path, "rb") as f:
            return Image.open(f).convert("RGB")

    def load(self, indices: List[int]) -> List[Image.Image]:
        if len(self.frames) == 0:
            return []

        max_idx = len(self.frames) - 1
        out = []
        for idx in indices:
            idx = int(np.clip(idx, 0, max_idx))
            out.append(self[idx])
        return out


class ASDVideoDataset(Dataset):
    def __init__(
        self,
        anno_path: str,
        data_path: str,
        mode: str = "train",
        clip_len: int = 16,
        frame_sample_rate: int = 1,
        crop_size: int = 224,
        short_side_size: int = 224,
        test_num_segment: int = 2,
        test_num_crop: int = 2,
        args=None,
        file_ext: str = "jpg",
    ):
        super().__init__()
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args
        self.file_ext = file_ext

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.use_augmentation = self.mode == "train" and getattr(self.args, "augment", True)
        self.use_random_erasing = self.mode == "train" and getattr(self.args, "reprob", 0.0) > 0

        table = pd.read_csv(self.anno_path, delimiter=",", dtype={0: str})
        self.samples = list(table.iloc[:, 0].values)
        self.labels = [int(x) for x in table.iloc[:, 1].values]

        if self.mode == "train":
            self.train_resize = Compose([
                Resize(size=(self.short_side_size, self.short_side_size), interpolation="bilinear"),
            ])

        elif self.mode == "validation":
            self.eval_transform = Compose([
                Resize(size=(self.short_side_size, self.short_side_size), interpolation="bilinear"),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        elif self.mode == "test":
            self.test_resize = Compose([
                Resize(size=(self.short_side_size, self.short_side_size), interpolation="bilinear"),
            ])
            self.test_transform = Compose([
                ClipToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

            self.test_items = []
            for temporal_idx in range(self.test_num_segment):
                for spatial_idx in range(self.test_num_crop):
                    for sample, label in zip(self.samples, self.labels):
                        self.test_items.append((sample, label, temporal_idx, spatial_idx))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self):
        return len(self.test_items) if self.mode == "test" else len(self.samples)

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        if self.mode == "train":
            sample = self.samples[index]
            frames = self.load_clip(sample)

            while len(frames) == 0:
                warnings.warn(f"Failed to load clip: {sample}. Resampling.")
                index = np.random.randint(len(self.samples))
                sample = self.samples[index]
                frames = self.load_clip(sample)

            frames = self.train_resize(frames)

            if getattr(self.args, "num_sample", 1) > 1:
                clips, labels, indices = [], [], []
                for _ in range(self.args.num_sample):
                    clips.append(self._augment_frames(frames))
                    labels.append(self.labels[index])
                    indices.append(index)
                return clips, labels, indices, {}

            clip = self._augment_frames(frames)
            return clip, self.labels[index], index, {}

        if self.mode == "validation":
            sample = self.samples[index]
            frames = self.load_clip(sample)

            while len(frames) == 0:
                warnings.warn(f"Failed to load clip: {sample}. Resampling.")
                index = np.random.randint(len(self.samples))
                sample = self.samples[index]
                frames = self.load_clip(sample)

            clip = self.eval_transform(frames)
            sample_id = self._sample_id(sample)
            return clip, self.labels[index], sample_id

        sample, label, temporal_idx, spatial_idx = self.test_items[index]
        frames = self.load_clip(sample)

        while len(frames) == 0:
            warnings.warn(
                f"Failed to load clip: {sample} "
                f"(temporal_idx={temporal_idx}, spatial_idx={spatial_idx}). Resampling."
            )
            index = np.random.randint(len(self.test_items))
            sample, label, temporal_idx, spatial_idx = self.test_items[index]
            frames = self.load_clip(sample)

        frames = self.test_resize(frames)
        frames_np = np.stack([np.array(img) for img in frames], axis=0)

        spatial_denom = max(self.test_num_crop - 1, 1)
        temporal_denom = max(self.test_num_segment - 1, 1)

        spatial_span = max(max(frames_np.shape[1], frames_np.shape[2]) - self.short_side_size, 0)
        temporal_span = max(frames_np.shape[0] - self.clip_len, 0)

        spatial_step = spatial_span / spatial_denom if spatial_denom > 0 else 0.0
        temporal_step = temporal_span / temporal_denom if temporal_denom > 0 else 0.0

        temporal_start = int(round(temporal_idx * temporal_step))
        spatial_start = int(round(spatial_idx * spatial_step))

        if frames_np.shape[1] >= frames_np.shape[2]:
            frames_np = frames_np[
                temporal_start:temporal_start + self.clip_len,
                spatial_start:spatial_start + self.short_side_size,
                :,
                :
            ]
        else:
            frames_np = frames_np[
                temporal_start:temporal_start + self.clip_len,
                :,
                spatial_start:spatial_start + self.short_side_size,
                :
            ]

        if frames_np.shape[0] < self.clip_len:
            pad_count = self.clip_len - frames_np.shape[0]
            last_frame = frames_np[-1:]
            frames_np = np.concatenate([frames_np] + [last_frame] * pad_count, axis=0)

        frames = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames_np]
        clip = self.test_transform(frames)
        sample_id = self._sample_id(sample)

        return clip, label, sample_id, temporal_idx, spatial_idx

    def _augment_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        if self.use_augmentation:
            augment = create_random_augment(
                input_size=(self.crop_size, self.crop_size),
                auto_augment=getattr(self.args, "aa", None),
                interpolation=getattr(self.args, "train_interpolation", "bicubic"),
            )
            frames = augment(frames)

        frames = [tv_transforms.ToTensor()(img) for img in frames]
        frames = torch.stack(frames, dim=0)            # T, C, H, W
        frames = frames.permute(0, 2, 3, 1)           # T, H, W, C
        frames = tensor_normalize(frames, self.mean, self.std)
        frames = frames.permute(3, 0, 1, 2)           # C, T, H, W

        frames = spatial_sampling(
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=[0.75, 1.3333],
            scale=[0.08, 1.0],
            motion_shift=False,
        )

        if self.use_random_erasing:
            eraser = RandomErasing(
                probability=self.args.reprob,
                mode=self.args.remode,
                max_count=self.args.recount,
                num_splits=self.args.recount,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)       # T, C, H, W
            frames = eraser(frames)
            frames = frames.permute(1, 0, 2, 3)       # C, T, H, W

        return frames

    def load_clip(self, sample: str, sample_rate_scale: int = 1) -> List[Image.Image]:
        clip_path = os.path.normpath(os.path.join(self.data_path, sample))

        if os.path.isfile(clip_path):
            with open(clip_path, "rb") as f:
                img = Image.open(f).convert("RGB")
            return [img.copy() for _ in range(self.clip_len)]

        if not os.path.isdir(clip_path):
            return []

        reader = FrameFolderReader(clip_path, file_ext=self.file_ext)
        if len(reader) == 0:
            return []

        if self.mode == "test":
            indices = [x for x in range(0, len(reader), self.frame_sample_rate)]
            while len(indices) < self.clip_len:
                indices.append(indices[-1])
            return reader.load(indices)

        converted_len = int(self.clip_len * self.frame_sample_rate)
        total_frames = len(reader)

        if total_frames <= converted_len:
            if total_frames // self.frame_sample_rate > 0:
                indices = np.linspace(0, total_frames, num=total_frames // self.frame_sample_rate)
            else:
                indices = np.array([0])

            if len(indices) < self.clip_len:
                pad = np.ones(self.clip_len - len(indices)) * total_frames
                indices = np.concatenate((indices, pad))

            indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
        else:
            end_idx = np.random.randint(converted_len, total_frames)
            start_idx = end_idx - converted_len
            indices = np.linspace(start_idx, end_idx, num=self.clip_len)
            indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

        indices = indices[::int(sample_rate_scale)]
        return reader.load(indices.tolist())

    @staticmethod
    def _sample_id(sample: str) -> str:
        base = os.path.basename(os.path.normpath(sample))
        name, _ = os.path.splitext(base)
        return name if name else base