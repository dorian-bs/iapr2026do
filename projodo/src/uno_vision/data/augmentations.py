"""Generate synthetic card crop and mask augmentations from allowed reference assets."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import skimage.util

from uno_vision.paths import AUGMENTATIONS_DIR, REFERENCE_CARDS_DIR


DEFAULT_SOURCE_TAGS = ("L1000765", "L1000766", "L1000767", "L1000768")


def load_reference_labels(reference_csv_path: Path = REFERENCE_CARDS_DIR / "labels.csv") -> dict[str, str]:
    """Read reference crop labels keyed by generated image id."""

    labels: dict[str, str] = {}
    with reference_csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"].strip()
            card = row["card"].strip()
            if image_id:
                labels[image_id] = card
    return labels


def load_aligned_mask(closed_dir: Path, crop_idx: int, target_h: int, target_w: int) -> np.ndarray:
    """Load a closed mask, crop it to foreground, and resize it to match the crop."""

    mask_path = closed_dir / f"closed_component_{crop_idx}.jpg"
    closed_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if closed_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    x, y, w, h = cv2.boundingRect(closed_mask)
    mask_crop = closed_mask[y:y + h, x:x + w]
    if mask_crop.shape[:2] != (target_h, target_w):
        mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask_crop > 127


def make_pattern_variants(h: int, w: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Create simple procedural backgrounds for foreground-preserving augmentations."""

    rng = rng or np.random.default_rng()
    patterns: dict[str, np.ndarray] = {}
    stripe_w = 12
    stripe = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(0, w, stripe_w * 2):
        stripe[:, c:c + stripe_w] = (0, 0, 255)
        stripe[:, c + stripe_w:c + 2 * stripe_w] = (255, 0, 0)
    patterns["pattern_stripes"] = stripe

    block = 48
    rects = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            rects[r:r + block, c:c + block] = rng.integers(0, 256, size=3, dtype=np.uint8)
    patterns["pattern_big_rectangles"] = rects

    tile = 24
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            checker[r:r + tile, c:c + tile] = (20, 220, 120) if ((r // tile) + (c // tile)) % 2 == 0 else (220, 20, 180)
    patterns["pattern_checkerboard"] = checker
    patterns["pattern_noise"] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return patterns


def apply_pattern_background(img_bgr: np.ndarray, mask_fg: np.ndarray, pattern_bgr: np.ndarray) -> np.ndarray:
    """Composite a card foreground over a generated BGR background pattern."""

    out = pattern_bgr.copy()
    out[mask_fg] = img_bgr[mask_fg]
    return out


def crop_image_augmentations(
    img_bgr: np.ndarray,
    mask_fg: np.ndarray | None,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Create geometric, photometric, occlusion, and background variants of one crop."""

    h, w = img_bgr.shape[:2]
    aug_images: list[tuple[str, np.ndarray]] = []
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    aug_images.append(("rot_45", cv2.warpAffine(img_bgr, matrix, (w, h))))
    aug_images.append(("flip_vertical", cv2.flip(img_bgr, 0)))
    aug_images.append(("flip_horizontal", cv2.flip(img_bgr, 1)))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for i in range(3):
        shift = int(rng.choice([-12, -8, -4, 4, 8, 12]))
        hue_img = hsv.copy()
        hue_img[:, :, 0] = (hue_img[:, :, 0].astype(np.int16) + shift) % 180
        aug_images.append((f"hue_{shift}_{i + 1}", cv2.cvtColor(hue_img, cv2.COLOR_HSV2BGR)))

    aug_images.append(("dimmed", cv2.convertScaleAbs(img_bgr, alpha=0.5, beta=0)))
    aug_images.append(("less_contrast", cv2.convertScaleAbs(img_bgr, alpha=0.6, beta=110)))
    aug_images.append(("more_contrast", cv2.convertScaleAbs(img_bgr, alpha=1.5, beta=0)))
    noisy = skimage.util.random_noise(img_bgr / 255.0, mode="speckle", var=0.08)
    aug_images.append(("speckle_noise", np.clip(noisy * 255.0, 0, 255).astype(np.uint8)))

    squeezed_img = cv2.resize(img_bgr, (max(1, w // 2), h))
    squeezed_full = np.ones_like(img_bgr) * 255
    x_offset = (w - squeezed_img.shape[1]) // 2
    squeezed_full[:, x_offset:x_offset + squeezed_img.shape[1]] = squeezed_img
    aug_images.append(("squeezed", squeezed_full))

    stretched_wide = cv2.resize(img_bgr, (w * 2, h))
    start_x = (stretched_wide.shape[1] - w) // 2
    aug_images.append(("stretched", stretched_wide[:, start_x:start_x + w]))

    for rect_i, (smin, smax) in enumerate([(0.12, 0.25), (0.30, 0.50), (0.60, 0.78)], start=1):
        cover = img_bgr.copy()
        rect_w = int(rng.uniform(smin, smax) * w)
        rect_h = int(rng.uniform(smin, smax) * h)
        x1 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y1 = int(rng.integers(0, max(1, h - rect_h + 1)))
        cv2.rectangle(cover, (x1, y1), (min(w, x1 + rect_w), min(h, y1 + rect_h)), (255, 255, 255), -1)
        aug_images.append((f"white_rectangle_{rect_i}", cover))

    if mask_fg is not None:
        # Pattern backgrounds exercise the segmenter without introducing external images.
        for name, pattern in make_pattern_variants(h, w, rng).items():
            aug_images.append((name, apply_pattern_background(img_bgr, mask_fg, pattern)))

    scale = 0.7
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    white_canvas = np.ones_like(img_bgr) * 255
    x_offset = (w - small_w) // 2
    y_offset = (h - small_h) // 2
    white_canvas[y_offset:y_offset + small_h, x_offset:x_offset + small_w] = small_img
    aug_images.append(("small_centered", white_canvas))
    return aug_images


def generate_augmentation_images(
    source_tags: tuple[str, ...] = DEFAULT_SOURCE_TAGS,
    reference_cards_dir: Path = REFERENCE_CARDS_DIR,
    augmentations_dir: Path = AUGMENTATIONS_DIR,
    seed: int = 42,
) -> int:
    """Generate augmented crop images and their labels CSV."""

    rng = np.random.default_rng(seed)
    aug_dir = augmentations_dir / "images"
    aug_dir.mkdir(parents=True, exist_ok=True)
    ref_labels = load_reference_labels(reference_cards_dir / "labels.csv")
    aug_rows: list[list[str]] = []

    for source_tag in source_tags:
        crops_dir = reference_cards_dir / source_tag / "crops"
        masks_dir = reference_cards_dir / source_tag / "masks"
        if not crops_dir.is_dir():
            continue
        for crop_file in sorted(crops_dir.glob("crop_*.jpg")):
            crop_idx = int(crop_file.stem.split("_")[1])
            base_id = f"{source_tag}_crop_{crop_idx}"
            card_label = ref_labels.get(base_id)
            if card_label is None:
                continue
            img_bgr = cv2.imread(str(crop_file), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            try:
                mask_fg = load_aligned_mask(masks_dir, crop_idx, h, w)
            except FileNotFoundError:
                mask_fg = None
            for aug_i, (_, aug_img) in enumerate(crop_image_augmentations(img_bgr, mask_fg, rng), start=1):
                image_id = f"{base_id}_aug{aug_i}"
                cv2.imwrite(str(aug_dir / f"{image_id}.jpg"), aug_img)
                aug_rows.append([image_id, card_label])

    labels_path = augmentations_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "card"])
        writer.writerows(aug_rows)
    return len(aug_rows)


def load_binary_mask(closed_dir: Path, crop_idx: int, target_h: int, target_w: int) -> np.ndarray:
    """Load a closed component mask and align it to the corresponding crop canvas."""

    aligned_bool = load_aligned_mask(closed_dir, crop_idx, target_h, target_w)
    return aligned_bool.astype(np.uint8) * 255


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a binary mask with nearest-neighbor interpolation to preserve labels."""

    h, w = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def mask_augmentations(mask: np.ndarray, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Mirror image augmentations for masks so segmentation pairs stay aligned."""

    h, w = mask.shape[:2]
    aug_masks: list[tuple[str, np.ndarray]] = [("rot_45", rotate_mask(mask, 45))]
    aug_masks.append(("flip_vertical", cv2.flip(mask, 0)))
    aug_masks.append(("flip_horizontal", cv2.flip(mask, 1)))
    # Consume the same RNG draws as image hue shifts so later random occlusions align.
    for i in range(3):
        _ = int(rng.choice([-12, -8, -4, 4, 8, 12]))
        aug_masks.append((f"hue_copy_{i + 1}", mask.copy()))
    aug_masks.extend([
        ("dimmed_copy", mask.copy()),
        ("less_contrast_copy", mask.copy()),
        ("more_contrast_copy", mask.copy()),
        ("speckle_noise_copy", mask.copy()),
    ])

    squeezed = cv2.resize(mask, (max(1, w // 2), h), interpolation=cv2.INTER_NEAREST)
    squeezed_full = np.zeros_like(mask)
    x_offset = (w - squeezed.shape[1]) // 2
    squeezed_full[:, x_offset:x_offset + squeezed.shape[1]] = squeezed
    aug_masks.append(("squeezed", squeezed_full))

    stretched_wide = cv2.resize(mask, (w * 2, h), interpolation=cv2.INTER_NEAREST)
    start_x = (stretched_wide.shape[1] - w) // 2
    aug_masks.append(("stretched", stretched_wide[:, start_x:start_x + w]))

    for rect_i, (smin, smax) in enumerate([(0.12, 0.25), (0.30, 0.50), (0.60, 0.78)], start=1):
        cover = mask.copy()
        rect_w = int(rng.uniform(smin, smax) * w)
        rect_h = int(rng.uniform(smin, smax) * h)
        x1 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y1 = int(rng.integers(0, max(1, h - rect_h + 1)))
        cv2.rectangle(cover, (x1, y1), (min(w, x1 + rect_w), min(h, y1 + rect_h)), 0, -1)
        aug_masks.append((f"white_rectangle_{rect_i}", cover))

    for name in ("pattern_stripes", "pattern_big_rectangles", "pattern_checkerboard", "pattern_noise"):
        aug_masks.append((name, mask.copy()))

    scale = 0.7
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    small_full = np.zeros_like(mask)
    x_offset = (w - small_w) // 2
    y_offset = (h - small_h) // 2
    small_full[y_offset:y_offset + small_h, x_offset:x_offset + small_w] = small
    aug_masks.append(("small_centered", small_full))
    return aug_masks


def generate_augmentation_masks(
    source_tags: tuple[str, ...] = DEFAULT_SOURCE_TAGS,
    reference_cards_dir: Path = REFERENCE_CARDS_DIR,
    augmentations_dir: Path = AUGMENTATIONS_DIR,
    seed: int = 42,
) -> int:
    """Generate augmented binary masks that correspond to the augmented crop images."""

    rng = np.random.default_rng(seed)
    aug_masks_dir = augmentations_dir / "masks"
    aug_masks_dir.mkdir(parents=True, exist_ok=True)
    ref_labels = load_reference_labels(reference_cards_dir / "labels.csv")
    saved_masks = 0
    for source_tag in source_tags:
        closed_dir = reference_cards_dir / source_tag / "masks"
        crops_dir = reference_cards_dir / source_tag / "crops"
        if not closed_dir.is_dir():
            continue
        for closed_file in sorted(closed_dir.glob("closed_component_*.jpg")):
            crop_idx = int(closed_file.stem.split("_")[-1])
            base_id = f"{source_tag}_crop_{crop_idx}"
            if base_id not in ref_labels:
                continue
            crop_path = crops_dir / f"crop_{crop_idx}.jpg"
            crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop_img is None:
                continue
            crop_h, crop_w = crop_img.shape[:2]
            mask = load_binary_mask(closed_dir, crop_idx, crop_h, crop_w)
            for aug_i, (_, aug_mask) in enumerate(mask_augmentations(mask, rng), start=1):
                image_id = f"{base_id}_aug{aug_i}"
                cv2.imwrite(str(aug_masks_dir / f"{image_id}.jpg"), aug_mask)
                saved_masks += 1
    return saved_masks


def generate_augmentations(seed: int = 42) -> tuple[int, int]:
    """Generate both image and mask augmentations with the same random seed."""

    image_count = generate_augmentation_images(seed=seed)
    mask_count = generate_augmentation_masks(seed=seed)
    return image_count, mask_count
