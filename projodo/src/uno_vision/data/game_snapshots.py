"""Synthetic game-scene generation utilities built from reference card crops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uno_vision.data.augmentations import DEFAULT_SOURCE_TAGS, load_aligned_mask, load_reference_labels
from uno_vision.paths import GAME_SNAPSHOTS_DIR, REFERENCE_CARDS_DIR

CardPlacement = dict[str, int | str]


@dataclass(frozen=True)
class SnapshotGenerationResult:
    """Filesystem locations written when snapshot generation finishes."""

    images_dir: Path
    masks_dir: Path
    labels_path: Path
    scenes_generated: int


def load_reference_card_pools(
    source_tags: tuple[str, ...] = DEFAULT_SOURCE_TAGS,
    reference_cards_dir: Path = REFERENCE_CARDS_DIR,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Load card crop images and aligned binary masks keyed by card label."""

    labels_by_id = load_reference_labels(reference_cards_dir / "labels.csv")
    card_images: dict[str, list[np.ndarray]] = {}
    card_masks: dict[str, list[np.ndarray]] = {}

    for source_tag in source_tags:
        crops_dir = reference_cards_dir / source_tag / "crops"
        masks_dir = reference_cards_dir / source_tag / "masks"
        if not crops_dir.is_dir():
            continue

        for crop_file in sorted(crops_dir.glob("crop_*.jpg")):
            parts = crop_file.stem.split("_")
            if len(parts) < 2:
                continue
            crop_idx = int(parts[1])
            base_id = f"{source_tag}_crop_{crop_idx}"
            card_label = labels_by_id.get(base_id)
            if card_label is None:
                continue

            img_bgr = cv2.imread(str(crop_file), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            try:
                mask_fg = load_aligned_mask(masks_dir, crop_idx, h, w)
            except FileNotFoundError:
                continue
            mask_u8 = mask_fg.astype(np.uint8) * 255

            card_images.setdefault(card_label, []).append(img_bgr)
            card_masks.setdefault(card_label, []).append(mask_u8)

    return card_images, card_masks


def add_right_angle_rotations(
    card_images: dict[str, list[np.ndarray]],
    card_masks: dict[str, list[np.ndarray]],
) -> int:
    """Append +/-90 degree variants for each loaded card crop/mask pair."""

    added = 0
    for label in list(card_images.keys()):
        originals_img = list(card_images.get(label, []))
        originals_mask = list(card_masks.get(label, []))

        for img, mask in zip(originals_img, originals_mask):
            for rot_flag in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                card_images[label].append(cv2.rotate(img, rot_flag))
                card_masks[label].append(cv2.rotate(mask, rot_flag))
                added += 1
    return added


def _bg_solid(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return np.full((h, w, 3), 255, dtype=np.uint8)
    if choice == 1:
        base = int(rng.integers(220, 256))
        offset = rng.integers(-15, 16, size=3)
        color = np.clip([base, base, base] + offset, 0, 255).astype(np.uint8)
        return np.full((h, w, 3), color, dtype=np.uint8)
    color = rng.integers(80, 220, size=3, dtype=np.uint8)
    return np.full((h, w, 3), color, dtype=np.uint8)


def _bg_gradient(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    color_a = rng.integers(50, 256, size=3).astype(np.float32)
    color_b = rng.integers(50, 256, size=3).astype(np.float32)
    out = np.zeros((h, w, 3), dtype=np.float32)
    if rng.random() < 0.5:
        for x in range(w):
            out[:, x] = color_a + (color_b - color_a) * x / max(1, w - 1)
    else:
        for y in range(h):
            out[y, :] = color_a + (color_b - color_a) * y / max(1, h - 1)
    return out.clip(0, 255).astype(np.uint8)


def _bg_checkerboard(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    tile = int(rng.integers(30, 80))
    color_a = rng.integers(30, 256, size=3, dtype=np.uint8)
    color_b = rng.integers(30, 256, size=3, dtype=np.uint8)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            out[r:r + tile, c:c + tile] = color_a if ((r // tile + c // tile) % 2 == 0) else color_b
    return out


def _bg_stripes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    stripe = int(rng.integers(15, 60))
    color_a = rng.integers(30, 256, size=3, dtype=np.uint8)
    color_b = rng.integers(30, 256, size=3, dtype=np.uint8)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if rng.random() < 0.5:
        for c in range(0, w, stripe * 2):
            out[:, c:c + stripe] = color_a
            out[:, c + stripe:c + stripe * 2] = color_b
    else:
        for r in range(0, h, stripe * 2):
            out[r:r + stripe, :] = color_a
            out[r + stripe:r + stripe * 2, :] = color_b
    return out


def _bg_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


_BG_GENERATORS = (_bg_solid, _bg_gradient, _bg_checkerboard, _bg_stripes, _bg_noise)


def random_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Return one random procedural BGR background image."""

    generator = _BG_GENERATORS[int(rng.integers(0, len(_BG_GENERATORS)))]
    return generator(h, w, rng)


def _convex_hull_footprint(mask: np.ndarray) -> np.ndarray:
    """Create a filled convex hull footprint for stable card compositing."""

    footprint = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return footprint
    all_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(all_points)
    cv2.fillPoly(footprint, [hull], 255)
    return footprint


def make_scene(
    card_images: dict[str, list[np.ndarray]],
    card_masks: dict[str, list[np.ndarray]],
    rng: np.random.Generator,
    scene_w: int = 1280,
    scene_h: int = 720,
    n_cards_range: tuple[int, int] = (12, 20),
    card_scale_range: tuple[float, float] = (0.10, 0.25),
) -> tuple[np.ndarray, np.ndarray, list[CardPlacement]]:
    """Compose one synthetic scene and return (image, mask, placement metadata)."""

    if not card_images:
        raise ValueError("card_images is empty; load reference pools before scene generation")

    canvas = random_background(scene_h, scene_w, rng)
    mask_canvas = np.zeros((scene_h, scene_w), dtype=np.uint8)

    labels = list(card_images.keys())
    n_cards = int(rng.integers(*n_cards_range))
    placements: list[CardPlacement] = []

    for _ in range(n_cards):
        label = labels[int(rng.integers(0, len(labels)))]
        image_pool = card_images.get(label, [])
        mask_pool = card_masks.get(label, [])
        n_variants = min(len(image_pool), len(mask_pool))
        if n_variants == 0:
            continue

        card_idx = int(rng.integers(0, n_variants))
        src_img = image_pool[card_idx]
        src_mask = mask_pool[card_idx]

        target_h = int(rng.uniform(*card_scale_range) * scene_h)
        scale = target_h / max(1, src_img.shape[0])
        target_w = max(1, int(src_img.shape[1] * scale))

        img_resized = cv2.resize(src_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(src_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        angle = float(rng.uniform(0, 360))
        rot_mat = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
        img_rot = cv2.warpAffine(
            img_resized,
            rot_mat,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        mask_rot = cv2.warpAffine(
            mask_resized,
            rot_mat,
            (target_w, target_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        footprint = _convex_hull_footprint(mask_rot)
        if np.count_nonzero(footprint) == 0:
            continue

        crop_h, crop_w = img_rot.shape[:2]
        center_x = rng.normal(loc=scene_w * 0.50, scale=scene_w * 0.20)
        center_y = rng.normal(loc=scene_h * 0.50, scale=scene_h * 0.20)
        x1 = int(center_x - crop_w / 2)
        y1 = int(center_y - crop_h / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(scene_w, x2)
        dst_y2 = min(scene_h, y2)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            continue

        img_patch = img_rot[src_y1:src_y2, src_x1:src_x2]
        footprint_patch = footprint[src_y1:src_y2, src_x1:src_x2]
        card_pixels = footprint_patch > 127
        if not np.any(card_pixels):
            continue

        scene_patch = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
        scene_patch[card_pixels] = img_patch[card_pixels]

        mask_patch = mask_canvas[dst_y1:dst_y2, dst_x1:dst_x2]
        mask_patch[card_pixels] = 255

        placements.append({
            "label": label,
            "card_idx": card_idx,
            "x1": dst_x1,
            "y1": dst_y1,
            "x2": dst_x2,
            "y2": dst_y2,
        })

    return canvas, mask_canvas, placements


def generate_game_snapshots(
    card_images: dict[str, list[np.ndarray]],
    card_masks: dict[str, list[np.ndarray]],
    save_dir: Path = GAME_SNAPSHOTS_DIR,
    n_scenes: int = 500,
    scene_w: int = 1280,
    scene_h: int = 720,
    seed: int = 100,
    n_cards_range: tuple[int, int] = (12, 20),
    card_scale_range: tuple[float, float] = (0.10, 0.25),
) -> SnapshotGenerationResult:
    """Generate synthetic scenes, masks, and labels metadata on disk."""

    save_dir.mkdir(parents=True, exist_ok=True)
    images_dir = save_dir / "images"
    masks_dir = save_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels_meta: list[dict[str, object]] = []

    for i in range(n_scenes):
        scene_bgr, mask_canvas, placements = make_scene(
            card_images,
            card_masks,
            rng,
            scene_w=scene_w,
            scene_h=scene_h,
            n_cards_range=n_cards_range,
            card_scale_range=card_scale_range,
        )
        scene_name = f"scene_{i:04d}"
        cv2.imwrite(str(images_dir / f"{scene_name}.jpg"), scene_bgr)
        cv2.imwrite(str(masks_dir / f"{scene_name}.jpg"), mask_canvas)
        labels_meta.append({"scene": scene_name, "placements": placements})

    labels_path = save_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as handle:
        json.dump(labels_meta, handle, indent=2)

    return SnapshotGenerationResult(
        images_dir=images_dir,
        masks_dir=masks_dir,
        labels_path=labels_path,
        scenes_generated=n_scenes,
    )


__all__ = [
    "SnapshotGenerationResult",
    "add_right_angle_rotations",
    "generate_game_snapshots",
    "load_reference_card_pools",
    "make_scene",
    "random_background",
]
