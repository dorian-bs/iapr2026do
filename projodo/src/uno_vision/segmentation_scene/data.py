"""Datasets and file collection helpers for scene-segmentation training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from uno_vision.image_ops import letterbox_pil
from uno_vision.paths import GAME_SNAPSHOTS_DIR


IMG_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def collect_scene_pairs(
    scenes_dir: Path = GAME_SNAPSHOTS_DIR,
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    image_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> list[tuple[str, str]]:
    """Collect aligned scene image-mask pairs from generated game snapshots."""

    images_dir = scenes_dir / image_subdir
    masks_dir = scenes_dir / mask_subdir
    if not images_dir.is_dir() or not masks_dir.is_dir():
        return []

    allowed = {suffix.lower() for suffix in image_suffixes}
    pairs: list[tuple[str, str]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in allowed:
            continue
        mask_path = masks_dir / image_path.name
        if mask_path.is_file():
            pairs.append((str(image_path), str(mask_path)))
    return pairs


def preprocess_scene_pair(
    image_path: str | Path,
    mask_path: str | Path,
    image_size: int = IMG_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one scene/mask pair and map it to normalized tensors."""

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = letterbox_pil(image, image_size, fill=255, interpolation=Image.BILINEAR)
    mask = letterbox_pil(mask, image_size, fill=0, interpolation=Image.NEAREST)

    image_t = TF.to_tensor(image)
    image_t = TF.normalize(image_t, mean=IMAGE_MEAN, std=IMAGE_STD)

    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    mask_arr = (mask_arr > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
    return image_t, mask_t


class SceneSegDataset(Dataset):
    """Torch dataset for synthetic game snapshots and binary scene masks."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        augment: bool = False,
        image_size: int = IMG_SIZE,
        preload_to_ram: bool = True,
        verbose: bool = False,
    ):
        self.augment = augment
        self.image_size = image_size
        self.preload_to_ram = preload_to_ram
        self.verbose = verbose
        self.pairs = list(pairs)

        self.data: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.preload_to_ram:
            self.data = []
            loaded_pairs: list[tuple[str, str]] = []
            if self.verbose:
                print(f"  Pre-loading {len(self.pairs)} scene pairs into RAM...", flush=True)
            for index, (image_path, mask_path) in enumerate(self.pairs):
                try:
                    self.data.append(
                        preprocess_scene_pair(
                            image_path=image_path,
                            mask_path=mask_path,
                            image_size=self.image_size,
                        )
                    )
                    loaded_pairs.append((image_path, mask_path))
                except Exception as exc:
                    if self.verbose:
                        print(f"  [skip] {exc}")
                if self.verbose and (index + 1) % 100 == 0:
                    print(f"  {index + 1}/{len(self.pairs)} loaded", flush=True)
            self.pairs = loaded_pairs
            if self.verbose:
                print(f"  Done. {len(self.pairs)} pairs ready.", flush=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            image_t, mask_t = preprocess_scene_pair(
                image_path=self.pairs[idx][0],
                mask_path=self.pairs[idx][1],
                image_size=self.image_size,
            )
        else:
            image_t, mask_t = self.data[idx]

        if self.augment:
            if random.random() < 0.5:
                image_t = TF.hflip(image_t)
                mask_t = TF.hflip(mask_t)
            if random.random() < 0.5:
                image_t = TF.vflip(image_t)
                mask_t = TF.vflip(mask_t)
        return image_t, mask_t
