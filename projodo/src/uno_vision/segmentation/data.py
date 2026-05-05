"""Datasets and file collection helpers for training the card segmenter."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from uno_vision.image_ops import letterbox_pil
from uno_vision.paths import AUGMENTATIONS_DIR, REFERENCE_CARDS_DIR


IMG_SIZE = 256
DEFAULT_REFERENCE_TAGS = ("L1000765", "L1000766", "L1000767", "L1000768")


def load_aligned_mask(mask_path: str | Path, img_h: int, img_w: int) -> np.ndarray:
    """Crop a closed component mask to its foreground box and align it to an image crop."""

    closed_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if closed_mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    if np.count_nonzero(closed_mask) == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(closed_mask)
    if w <= 0 or h <= 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    mask_crop = closed_mask[y:y + h, x:x + w]
    if mask_crop.size == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    if mask_crop.shape[:2] != (img_h, img_w):
        mask_crop = cv2.resize(mask_crop, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return mask_crop


def collect_segmentation_pairs(
    reference_cards_dir: Path = REFERENCE_CARDS_DIR,
    augmentations_dir: Path = AUGMENTATIONS_DIR,
    tags: tuple[str, ...] = DEFAULT_REFERENCE_TAGS,
) -> list[tuple[str, str]]:
    """Collect image-mask pairs from reference crops and generated augmentations."""

    pairs: list[tuple[str, str]] = []
    for tag in tags:
        crops_dir = reference_cards_dir / tag / "crops"
        masks_dir = reference_cards_dir / tag / "masks"
        if not crops_dir.is_dir() or not masks_dir.is_dir():
            continue
        for crop_path in sorted(crops_dir.glob("crop_*.jpg")):
            idx = crop_path.stem.replace("crop_", "")
            mask_path = masks_dir / f"closed_component_{idx}.jpg"
            if mask_path.is_file():
                pairs.append((str(crop_path), str(mask_path)))

    aug_dir = augmentations_dir / "images"
    aug_masks_dir = augmentations_dir / "masks"
    if aug_dir.is_dir() and aug_masks_dir.is_dir():
        for image_path in sorted(aug_dir.glob("*.jpg")):
            mask_path = aug_masks_dir / image_path.name
            if mask_path.is_file():
                pairs.append((str(image_path), str(mask_path)))
    return pairs


class SegDataset(Dataset):
    """Torch dataset that returns normalized card crops with binary foreground masks."""

    def __init__(self, pairs: list[tuple[str, str]], augment: bool = False, image_size: int = IMG_SIZE):
        self.pairs = pairs
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        img_h, img_w = img_array.shape[:2]
        mask_aligned = load_aligned_mask(mask_path, img_h, img_w)
        mask = Image.fromarray(mask_aligned)

        # Letterboxing keeps cards geometrically comparable while avoiding stretch artifacts.
        img = letterbox_pil(img, self.image_size, fill=255, interpolation=Image.BILINEAR)
        mask = letterbox_pil(mask, self.image_size, fill=0, interpolation=Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_t, mask_t
