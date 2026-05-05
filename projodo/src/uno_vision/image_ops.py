"""Image resizing helpers that preserve aspect ratio for model inputs and masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to map a square letterboxed mask back to the original crop."""

    orig_w: int
    orig_h: int
    new_w: int
    new_h: int
    left: int
    top: int
    target: int


def letterbox_bgr(img_bgr: np.ndarray, size: int = 128, fill: int = 128) -> np.ndarray:
    """Resize a BGR image into a square canvas without distorting card shape."""

    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def letterbox_pil(
    img: Image.Image,
    target_size: int,
    fill: int = 255,
    interpolation: int = Image.BILINEAR,
) -> Image.Image:
    """Resize a PIL image into a square canvas while keeping its aspect ratio."""

    import torchvision.transforms.functional as TF

    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), interpolation)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(resized, padding, fill=fill)


def letterbox_pil_with_meta(
    img: Image.Image,
    target_size: int,
    fill: int = 255,
    interpolation: int = Image.BILINEAR,
) -> tuple[Image.Image, LetterboxMeta]:
    """Letterbox a PIL image and return the padding metadata for inverse mapping."""

    import torchvision.transforms.functional as TF

    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), interpolation)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top
    img_lb = TF.pad(resized, (left, top, right, bottom), fill=fill)
    return img_lb, LetterboxMeta(w, h, new_w, new_h, left, top, target_size)


def unletterbox_mask(mask_lb: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Remove letterbox padding and resize a predicted mask back to crop coordinates."""

    core = mask_lb[meta.top:meta.top + meta.new_h, meta.left:meta.left + meta.new_w]
    return cv2.resize(core, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)
