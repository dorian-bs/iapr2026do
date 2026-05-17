"""Report helpers for the generated training data and augmentation stage."""
from __future__ import annotations

import csv
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PLAYERS = ("p1", "p2", "p3", "p4")

PLAYER_LAYOUTS = {
    "p1": {"center": (0.50, 0.84), "axis": (1.0, 0.0), "inward": (0.0, -1.0), "angle": 0.0},
    "p2": {"center": (0.84, 0.50), "axis": (0.0, 1.0), "inward": (-1.0, 0.0), "angle": -90.0},
    "p3": {"center": (0.50, 0.16), "axis": (1.0, 0.0), "inward": (0.0, 1.0), "angle": 180.0},
    "p4": {"center": (0.16, 0.50), "axis": (0.0, 1.0), "inward": (1.0, 0.0), "angle": 90.0},
}

TOKEN_STYLE_DIRS = {
    "white_rectangle": "white_rectangle",
    "yellow_round": "yellow_round",
}


@dataclass
class CreateAugmentedDataConfig:
    """Configuration used by the original augmented-data notebook."""

    seed: int = 67

    n_aug_per_reference: int = 1000
    aug_card_canvas: tuple[int, int] = (256, 256)
    aug_card_height_range: tuple[int, int] = (140, 220)
    aug_card_angle_range_deg: tuple[float, float] = (-25.0, 25.0)
    aug_card_shift_fraction: float = 0.12
    aug_card_image_format: str = "jpg"
    aug_card_jpeg_quality: int = 80
    aug_card_generation_workers: int = 0
    aug_card_generation_backend: str = "thread"
    aug_card_generation_chunk_size: int = 32

    n_scenes: int = 12288
    scene_width: int = 1280
    scene_height: int = 720
    min_cards_per_player: int = 0
    max_cards_per_player: int = 6
    mask_gap_pixels: int = 2
    min_visible_card_area: int = 90
    clear_output_dirs: bool = True

    player_card_height_fraction_range: tuple[float, float] = (0.20, 0.22)
    center_card_height_fraction_range: tuple[float, float] = (0.20, 0.22)
    player_slot_spacing_factor_range: tuple[float, float] = (0.09, 1.50)
    player_hand_center_jitter_fraction_range: tuple[float, float] = (0.006, 0.020)
    player_slot_offset_jitter_fraction_range: tuple[float, float] = (0.05, 0.20)
    player_inward_jitter_fraction_range: tuple[float, float] = (0.008, 0.030)
    player_rotation_jitter_deg_range: tuple[float, float] = (0.5, 45.0)
    center_position_fraction_range_x: tuple[float, float] = (0.40, 0.70)
    center_position_fraction_range_y: tuple[float, float] = (0.40, 0.70)
    center_angle_range_deg: tuple[float, float] = (-70.0, 70.0)
    scene_image_format: str = "jpg"
    scene_jpeg_quality: int = 80
    scene_generation_workers: int = 0
    scene_generation_backend: str = "thread"
    scene_generation_chunk_size: int = 16

    scene_shadow_enabled: bool = True
    scene_shadow_offset_fraction_range: tuple[float, float] = (0.006, 0.020)
    scene_shadow_blur_fraction_range: tuple[float, float] = (0.003, 0.010)
    scene_shadow_opacity_range: tuple[float, float] = (0.14, 0.30)

    token_inward_distance_fraction_range: tuple[float, float] = (0.0, 0.05)
    token_lateral_distance_fraction_range: tuple[float, float] = (0.23, 0.33)
    token_side_mode: str = "right"
    token_center_jitter_fraction: float = 0.008
    token_clamp_margin_fraction: float = 0.02

    preview_saved_sample_count: int = 4
    preview_card_sample_count: int = 12


@dataclass(frozen=True)
class CardAsset:
    image_id: str
    label: str
    image_bgr: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class TokenAsset:
    image_bgr: np.ndarray
    mask: np.ndarray
    source_path: Path | None


@dataclass(frozen=True)
class AugmentationPaths:
    project_root: Path
    training_data: Path
    labels_dir: Path
    reference_csv: Path
    ref_cards_dir: Path
    funky_bg_path: Path
    white_bg_path: Path
    aug_cards_dir: Path
    aug_masks_dir: Path
    aug_csv_path: Path
    scenes_img_dir: Path
    scenes_mask_dir: Path
    scenes_labels_path: Path
    token_asset_dir: Path


def resolve_augmentation_paths(project_root: Path) -> AugmentationPaths:
    """Resolve inputs and outputs for full generated-data reproduction."""
    project_root = Path(project_root).resolve()
    training_data = project_root / "training_data"
    reference_data = training_data / "reference_data"
    augmented_data = training_data / "augmented_data"
    labels_dir = training_data / "object_labels"
    return AugmentationPaths(
        project_root=project_root,
        training_data=training_data,
        labels_dir=labels_dir,
        reference_csv=labels_dir / "reference_cards.csv",
        ref_cards_dir=reference_data / "card_crops",
        funky_bg_path=reference_data / "backgrounds" / "textured_table_background.jpg",
        white_bg_path=reference_data / "backgrounds" / "plain_white_background.jpg",
        aug_cards_dir=augmented_data / "card_images",
        aug_masks_dir=augmented_data / "card_masks",
        aug_csv_path=labels_dir / "augmented_cards.csv",
        scenes_img_dir=augmented_data / "scene_images",
        scenes_mask_dir=augmented_data / "scene_masks",
        scenes_labels_path=labels_dir / "augmented_scenes.json",
        token_asset_dir=reference_data / "token_crops",
    )


def _normalize_image_format(value: str) -> str:
    fmt = str(value).lower().strip().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"jpg", "png"}:
        raise ValueError(f"Unsupported image format {value!r}; use 'jpg' or 'png'.")
    return fmt


def _jpeg_quality(value: int) -> int:
    return max(1, min(100, int(value)))


def _write_image(path: Path, image_bgr: np.ndarray, fmt: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_quality(jpeg_quality)] if fmt == "jpg" else []
    ok = cv2.imwrite(str(path), image_bgr, params)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def _natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _read_reference_labels(reference_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with reference_csv.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row.get("image_id", "")).strip()
            label = str(row.get("card", "")).strip()
            if image_id and label:
                labels[image_id] = label
    return labels


def load_reference_card_assets(paths: AugmentationPaths) -> list[CardAsset]:
    labels = _read_reference_labels(paths.reference_csv)
    assets: list[CardAsset] = []
    for crop_path in sorted(paths.ref_cards_dir.glob("*.png"), key=_natural_sort_key):
        image_id = crop_path.stem
        label = labels.get(image_id)
        if label is None:
            continue
        rgba = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3:
            continue
        image_bgr = rgba[:, :, :3]
        mask = np.where(rgba[:, :, 3] > 0, 255, 0).astype(np.uint8) if rgba.shape[2] >= 4 else np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
        assets.append(CardAsset(image_id=image_id, label=label, image_bgr=image_bgr, mask=mask))
    if not assets:
        raise RuntimeError("No labelled RGBA reference crops found. Check training_data/reference_data/card_crops and object_labels/reference_cards.csv.")
    return assets


def _ensure_output_dirs(paths: AugmentationPaths) -> None:
    for output_dir in (
        paths.aug_cards_dir,
        paths.aug_masks_dir,
        paths.labels_dir,
        paths.scenes_img_dir,
        paths.scenes_mask_dir,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)


def _clear_generated_outputs(paths: AugmentationPaths) -> None:
    for directory, patterns in (
        (paths.aug_cards_dir, ("augmented_card_*.png", "augmented_card_*.jpg", "augmented_card_*.jpeg")),
        (paths.aug_masks_dir, ("augmented_card_*.png", "augmented_card_*.jpg", "augmented_card_*.jpeg")),
        (paths.scenes_img_dir, ("augmented_scene_*.png", "augmented_scene_*.jpg", "augmented_scene_*.jpeg")),
        (paths.scenes_mask_dir, ("augmented_scene_*.png",)),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for output_file in directory.glob(pattern):
                if output_file.is_file():
                    output_file.unlink()
    for label_path in (paths.aug_csv_path, paths.scenes_labels_path):
        if label_path.is_file():
            label_path.unlink()


def _resolve_worker_count(requested: int, total: int) -> int:
    if requested < 0:
        raise ValueError("worker count must be >= 0")
    if total <= 1:
        return 1
    if requested == 0:
        return min(8, max(1, os.cpu_count() or 1), total)
    return min(int(requested), total)


def sample_cover_crop(source_bgr: np.ndarray, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    source_h, source_w = source_bgr.shape[:2]
    cover_scale = max(width / source_w, height / source_h) * float(rng.uniform(1.0, 1.16))
    resized_w = max(width, int(round(source_w * cover_scale)))
    resized_h = max(height, int(round(source_h * cover_scale)))
    resized = cv2.resize(source_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        resized = cv2.flip(resized, 1)
    x0 = int(rng.integers(0, max(1, resized_w - width + 1)))
    y0 = int(rng.integers(0, max(1, resized_h - height + 1)))
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _fallback_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.indices((height, width))
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = 80 + 55 * np.sin(xx / 45.0)
    base[..., 1] = 120 + 70 * np.sin((xx + yy) / 80.0)
    base[..., 2] = 125 + 65 * np.cos(yy / 55.0)
    base += rng.normal(0, 10, size=base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def make_background(paths: AugmentationPaths, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    candidates = [paths.funky_bg_path, paths.white_bg_path]
    rng.shuffle(candidates)
    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            background = sample_cover_crop(image, height, width, rng)
            alpha = float(rng.uniform(0.90, 1.10))
            beta = float(rng.uniform(-10.0, 10.0))
            return cv2.convertScaleAbs(background, alpha=alpha, beta=beta)
    return _fallback_background(height, width, rng)


def transform_card(
    asset: CardAsset,
    target_height: float,
    angle_degrees: float,
    rng: np.random.Generator,
    apply_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_h, source_w = asset.image_bgr.shape[:2]
    scale = float(target_height) / max(1, source_h)
    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))
    image = cv2.resize(asset.image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(asset.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(asset.mask.astype(np.float32) / 255.0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    image = cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.88, 1.14)), beta=float(rng.uniform(-16, 16)))
    noise_std = float(rng.uniform(0.0, 5.5))
    if noise_std > 0.1:
        image = np.clip(image.astype(np.float32) + rng.normal(0.0, noise_std, image.shape), 0, 255).astype(np.uint8)

    if not apply_rotation or abs(float(angle_degrees)) < 1e-6:
        return image, np.where(mask > 127, 255, 0).astype(np.uint8), np.clip(alpha, 0.0, 1.0)

    center = (new_w / 2.0, new_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    rot_w = int(round(new_h * abs_sin + new_w * abs_cos))
    rot_h = int(round(new_h * abs_cos + new_w * abs_sin))
    matrix[0, 2] += rot_w / 2.0 - center[0]
    matrix[1, 2] += rot_h / 2.0 - center[1]
    image = cv2.warpAffine(image, matrix, (rot_w, rot_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mask = cv2.warpAffine(mask, matrix, (rot_w, rot_h), flags=cv2.INTER_NEAREST, borderValue=0)
    alpha = cv2.warpAffine(alpha, matrix, (rot_w, rot_h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return image, np.where(mask > 127, 255, 0).astype(np.uint8), np.clip(alpha, 0.0, 1.0)


def paste_patch(
    scene_bgr: np.ndarray,
    instance_map: np.ndarray,
    patch_bgr: np.ndarray,
    patch_mask: np.ndarray,
    center_xy: tuple[float, float],
    patch_alpha: np.ndarray | None = None,
    instance_id: int | None = None,
) -> tuple[int, int, int, int] | None:
    scene_h, scene_w = scene_bgr.shape[:2]
    patch_h, patch_w = patch_bgr.shape[:2]
    left = int(round(center_xy[0] - patch_w / 2.0))
    top = int(round(center_xy[1] - patch_h / 2.0))
    right = left + patch_w
    bottom = top + patch_h
    dst_l, dst_t = max(0, left), max(0, top)
    dst_r, dst_b = min(scene_w, right), min(scene_h, bottom)
    if dst_r <= dst_l or dst_b <= dst_t:
        return None

    src_l, src_t = dst_l - left, dst_t - top
    src_r, src_b = src_l + (dst_r - dst_l), src_t + (dst_b - dst_t)
    raw_mask = patch_mask[src_t:src_b, src_l:src_r]
    mask_bool = raw_mask > 127
    if not np.any(mask_bool):
        return None
    alpha = raw_mask.astype(np.float32) / 255.0 if patch_alpha is None else patch_alpha[src_t:src_b, src_l:src_r].astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)
    roi = scene_bgr[dst_t:dst_b, dst_l:dst_r]
    patch = patch_bgr[src_t:src_b, src_l:src_r]
    roi[:] = np.clip(roi.astype(np.float32) * (1.0 - alpha[:, :, None]) + patch.astype(np.float32) * alpha[:, :, None], 0, 255).astype(np.uint8)
    if instance_id is not None:
        instance_map[dst_t:dst_b, dst_l:dst_r][mask_bool] = instance_id
    yy, xx = np.where(mask_bool)
    return int(dst_l + xx.min()), int(dst_t + yy.min()), int(dst_l + xx.max() + 1), int(dst_t + yy.max() + 1)


def _shadow_alpha(card_alpha: np.ndarray, sigma: float, opacity: float) -> np.ndarray:
    shadow = cv2.GaussianBlur(np.clip(card_alpha, 0.0, 1.0), (0, 0), sigmaX=max(0.1, sigma), sigmaY=max(0.1, sigma))
    return np.clip(shadow * float(opacity), 0.0, 1.0)


def _paste_shadow(
    scene_bgr: np.ndarray,
    instance_map: np.ndarray,
    card_mask: np.ndarray,
    card_alpha: np.ndarray,
    center_xy: tuple[float, float],
    cfg: CreateAugmentedDataConfig,
    rng: np.random.Generator,
) -> None:
    if not cfg.scene_shadow_enabled:
        return
    offset = float(rng.uniform(*cfg.scene_shadow_offset_fraction_range) * cfg.scene_height)
    angle = float(rng.uniform(0, 2 * math.pi))
    center = (center_xy[0] + math.cos(angle) * offset, center_xy[1] + math.sin(angle) * offset)
    sigma = float(rng.uniform(*cfg.scene_shadow_blur_fraction_range) * cfg.scene_height)
    opacity = float(rng.uniform(*cfg.scene_shadow_opacity_range))
    shadow = np.zeros((*card_mask.shape, 3), dtype=np.uint8)
    shadow_alpha = _shadow_alpha(card_alpha, sigma=sigma, opacity=opacity)
    paste_patch(scene_bgr, instance_map, shadow, card_mask, center, patch_alpha=shadow_alpha, instance_id=None)


def separated_binary_mask(instance_map: np.ndarray, gap_pixels: int, min_visible_card_area: int) -> np.ndarray:
    binary = np.zeros(instance_map.shape, dtype=np.uint8)
    kernel = None
    if gap_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(gap_pixels) + 1, 2 * int(gap_pixels) + 1))
    for instance_id in [int(value) for value in np.unique(instance_map) if int(value) > 0]:
        mask = np.where(instance_map == instance_id, 255, 0).astype(np.uint8)
        if np.count_nonzero(mask) < min_visible_card_area:
            continue
        if kernel is not None:
            mask = cv2.erode(mask, kernel, iterations=1)
        if np.count_nonzero(mask) >= min_visible_card_area:
            binary[mask > 0] = 255
    return binary


def _sample_card(card_assets: list[CardAsset], rng: np.random.Generator) -> CardAsset:
    return card_assets[int(rng.integers(0, len(card_assets)))]


def _clamped_center(center_xy: np.ndarray, scene_width: int, scene_height: int) -> tuple[float, float]:
    margin_x = scene_width * 0.07
    margin_y = scene_height * 0.07
    return (
        float(np.clip(center_xy[0], margin_x, scene_width - margin_x)),
        float(np.clip(center_xy[1], margin_y, scene_height - margin_y)),
    )


def _player_card_placements(
    player_name: str,
    card_count: int,
    card_assets: list[CardAsset],
    cfg: CreateAugmentedDataConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if card_count <= 0:
        return []
    layout = PLAYER_LAYOUTS[player_name]
    base_center = np.array([layout["center"][0] * cfg.scene_width, layout["center"][1] * cfg.scene_height], dtype=np.float32)
    axis = np.array(layout["axis"], dtype=np.float32)
    inward = np.array(layout["inward"], dtype=np.float32)
    hand_jitter = float(rng.uniform(*cfg.player_hand_center_jitter_fraction_range))
    base_center += np.array([
        float(rng.normal(0, cfg.scene_width * hand_jitter)),
        float(rng.normal(0, cfg.scene_height * hand_jitter)),
    ], dtype=np.float32)

    target_height = float(rng.uniform(*cfg.player_card_height_fraction_range) * cfg.scene_height)
    slot_spacing = target_height * 0.64 * float(rng.uniform(*cfg.player_slot_spacing_factor_range))
    offsets = (np.arange(card_count, dtype=np.float32) - (card_count - 1) / 2.0) * slot_spacing
    placements: list[dict[str, Any]] = []
    for offset in offsets:
        offset_jitter = float(rng.normal(0, slot_spacing * float(rng.uniform(*cfg.player_slot_offset_jitter_fraction_range))))
        inward_jitter = float(rng.normal(0, cfg.scene_height * float(rng.uniform(*cfg.player_inward_jitter_fraction_range))))
        center = base_center + axis * (float(offset) + offset_jitter) + inward * inward_jitter
        placements.append({
            "asset": _sample_card(card_assets, rng),
            "center": _clamped_center(center, cfg.scene_width, cfg.scene_height),
            "angle": float(layout["angle"] + rng.normal(0, float(rng.uniform(*cfg.player_rotation_jitter_deg_range)))),
            "target_height": target_height * float(rng.uniform(0.90, 1.10)),
            "region": player_name,
        })
    return placements


def _center_card_placement(card_assets: list[CardAsset], cfg: CreateAugmentedDataConfig, rng: np.random.Generator) -> dict[str, Any]:
    center = np.array([
        float(rng.uniform(*cfg.center_position_fraction_range_x) * cfg.scene_width),
        float(rng.uniform(*cfg.center_position_fraction_range_y) * cfg.scene_height),
    ], dtype=np.float32)
    return {
        "asset": _sample_card(card_assets, rng),
        "center": _clamped_center(center, cfg.scene_width, cfg.scene_height),
        "angle": float(rng.uniform(*cfg.center_angle_range_deg)),
        "target_height": float(rng.uniform(*cfg.center_card_height_fraction_range) * cfg.scene_height),
        "region": "center",
    }


def _load_token_asset(paths: AugmentationPaths, player_name: str, rng: np.random.Generator) -> TokenAsset:
    candidates: list[Path] = []
    for style_dir in TOKEN_STYLE_DIRS.values():
        candidates.extend(sorted((paths.token_asset_dir / style_dir).glob(f"{player_name}.*")))
    if candidates:
        path = candidates[int(rng.integers(0, len(candidates)))]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 3 and image.shape[2] == 4:
                return TokenAsset(image[:, :, :3], np.where(image[:, :, 3] > 0, 255, 0).astype(np.uint8), path)
            image_bgr = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            return TokenAsset(image_bgr, np.where(gray < 245, 255, 0).astype(np.uint8), path)

    radius = 26
    canvas = np.zeros((radius * 2 + 4, radius * 2 + 4, 3), dtype=np.uint8)
    canvas[:] = (0, 220, 245)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (radius + 2, radius + 2), radius, 255, thickness=-1)
    canvas[mask == 0] = 0
    return TokenAsset(canvas, mask, None)


def _token_center(player_name: str, cfg: CreateAugmentedDataConfig, rng: np.random.Generator) -> tuple[float, float]:
    layout = PLAYER_LAYOUTS[player_name]
    center = np.array([layout["center"][0] * cfg.scene_width, layout["center"][1] * cfg.scene_height], dtype=np.float32)
    axis = np.array(layout["axis"], dtype=np.float32)
    inward = np.array(layout["inward"], dtype=np.float32)
    side = cfg.token_side_mode.strip().lower()
    side_sign = -1.0 if side == "left" else 1.0
    if side == "random":
        side_sign = -1.0 if rng.random() < 0.5 else 1.0
    if side == "center":
        side_sign = 0.0
    center += axis * side_sign * float(rng.uniform(*cfg.token_lateral_distance_fraction_range) * cfg.scene_width)
    center += inward * float(rng.uniform(*cfg.token_inward_distance_fraction_range) * cfg.scene_height)
    center += rng.normal(0, cfg.token_center_jitter_fraction * cfg.scene_height, size=2).astype(np.float32)
    margin = cfg.token_clamp_margin_fraction
    return (
        float(np.clip(center[0], cfg.scene_width * margin, cfg.scene_width * (1 - margin))),
        float(np.clip(center[1], cfg.scene_height * margin, cfg.scene_height * (1 - margin))),
    )


def render_augmented_card(asset: CardAsset, rng: np.random.Generator, cfg: CreateAugmentedDataConfig, paths: AugmentationPaths) -> tuple[np.ndarray, np.ndarray]:
    canvas_h, canvas_w = int(cfg.aug_card_canvas[1]), int(cfg.aug_card_canvas[0])
    background = make_background(paths, canvas_h, canvas_w, rng)
    image, mask, alpha = transform_card(
        asset,
        target_height=float(rng.uniform(*cfg.aug_card_height_range)),
        angle_degrees=float(rng.uniform(*cfg.aug_card_angle_range_deg)),
        rng=rng,
        apply_rotation=True,
    )
    instance_map = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    shift = cfg.aug_card_shift_fraction * min(canvas_h, canvas_w)
    center = (canvas_w / 2.0 + float(rng.normal(0, shift)), canvas_h / 2.0 + float(rng.normal(0, shift)))
    paste_patch(background, instance_map, image, mask, center, patch_alpha=alpha, instance_id=1)
    out_mask = np.where(instance_map > 0, 255, 0).astype(np.uint8)
    return background, out_mask


def compose_augmented_scene(
    card_assets: list[CardAsset],
    rng: np.random.Generator,
    cfg: CreateAugmentedDataConfig,
    paths: AugmentationPaths,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scene = make_background(paths, cfg.scene_height, cfg.scene_width, rng)
    instance_map = np.zeros((cfg.scene_height, cfg.scene_width), dtype=np.int32)
    active_player = PLAYERS[int(rng.integers(0, len(PLAYERS)))]
    player_counts = {
        player: int(rng.integers(cfg.min_cards_per_player, cfg.max_cards_per_player + 1))
        for player in PLAYERS
    }
    placements = [_center_card_placement(card_assets, cfg, rng)]
    for player in PLAYERS:
        placements.extend(_player_card_placements(player, player_counts[player], card_assets, cfg, rng))
    rng.shuffle(placements)

    card_rows: list[dict[str, Any]] = []
    for instance_id, placement in enumerate(placements, start=1):
        asset: CardAsset = placement["asset"]
        patch, patch_mask, patch_alpha = transform_card(
            asset,
            target_height=float(placement["target_height"]),
            angle_degrees=float(placement["angle"]),
            rng=rng,
            apply_rotation=True,
        )
        center = tuple(placement["center"])
        _paste_shadow(scene, instance_map, patch_mask, patch_alpha, center, cfg, rng)
        bbox = paste_patch(scene, instance_map, patch, patch_mask, center, patch_alpha=patch_alpha, instance_id=instance_id)
        if bbox is None:
            continue
        card_rows.append({
            "label": asset.label,
            "source_image_id": asset.image_id,
            "bbox": [int(value) for value in bbox],
            "region": placement["region"],
            "angle": float(placement["angle"]),
        })

    token = _load_token_asset(paths, active_player, rng)
    token_scale = float(rng.uniform(0.045, 0.070) * cfg.scene_height) / max(1, token.image_bgr.shape[0])
    token_w = max(1, int(round(token.image_bgr.shape[1] * token_scale)))
    token_h = max(1, int(round(token.image_bgr.shape[0] * token_scale)))
    token_img = cv2.resize(token.image_bgr, (token_w, token_h), interpolation=cv2.INTER_AREA)
    token_mask = cv2.resize(token.mask, (token_w, token_h), interpolation=cv2.INTER_NEAREST)
    token_bbox = paste_patch(scene, instance_map, token_img, token_mask, _token_center(active_player, cfg, rng), instance_id=None)

    scene_mask = separated_binary_mask(instance_map, cfg.mask_gap_pixels, cfg.min_visible_card_area)
    metadata = {
        "active_player": active_player,
        "player_card_counts": player_counts,
        "cards": card_rows,
        "token_bbox": None if token_bbox is None else [int(value) for value in token_bbox],
    }
    return scene, scene_mask, metadata


def initialize_create_augmented_data_pipeline(
    cfg: CreateAugmentedDataConfig | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare the full augmentation pipeline used by the report notebook."""
    cfg = cfg or CreateAugmentedDataConfig()
    paths = resolve_augmentation_paths(Path.cwd() if project_root is None else Path(project_root))
    _ensure_output_dirs(paths)
    if cfg.clear_output_dirs:
        _clear_generated_outputs(paths)
    card_assets = load_reference_card_assets(paths)
    card_fmt = _normalize_image_format(cfg.aug_card_image_format)
    scene_fmt = _normalize_image_format(cfg.scene_image_format)
    print(f"Project root: {paths.project_root}")
    print(f"Reference assets: {len(card_assets)} crops from {paths.ref_cards_dir}")
    print(f"Augmented-card output: {paths.aug_cards_dir} (*.{card_fmt})")
    print(f"Scene output: {paths.scenes_img_dir} (*.{scene_fmt})")
    return {
        "cfg": cfg,
        "paths": paths,
        "card_assets": card_assets,
        "card_fmt": card_fmt,
        "scene_fmt": scene_fmt,
        "aug_rows": [],
        "scene_metadata": [],
        "preview_scene_bgr": None,
        "preview_mask": None,
        "preview_metadata": None,
    }


def _card_generation_task(args: tuple[int, int, int, CardAsset, CreateAugmentedDataConfig, AugmentationPaths, str]) -> dict[str, str]:
    global_index, reference_index, aug_index, asset, cfg, paths, card_fmt = args
    rng = np.random.default_rng(cfg.seed + 10_000 + global_index)
    image_id = f"augmented_card_{global_index:05d}"
    image_bgr, mask = render_augmented_card(asset, rng, cfg, paths)
    image_path = paths.aug_cards_dir / f"{image_id}.{card_fmt}"
    mask_path = paths.aug_masks_dir / f"{image_id}.png"
    _write_image(image_path, image_bgr, fmt=card_fmt, jpeg_quality=cfg.aug_card_jpeg_quality)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to write mask: {mask_path}")
    return {
        "image_id": image_id,
        "card": asset.label,
        "image_path": str(image_path.relative_to(paths.project_root)),
        "mask_path": str(mask_path.relative_to(paths.project_root)),
        "source_image_id": asset.image_id,
        "reference_index": str(reference_index),
        "augmentation_index": str(aug_index),
    }


def run_card_generation(state: dict[str, Any]) -> dict[str, Any]:
    """Generate every single-card crop and mask from the reference assets."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: AugmentationPaths = state["paths"]
    assets: list[CardAsset] = state["card_assets"]
    card_fmt: str = state["card_fmt"]
    tasks = [
        (ref_idx * cfg.n_aug_per_reference + aug_idx, ref_idx, aug_idx, asset, cfg, paths, card_fmt)
        for ref_idx, asset in enumerate(assets)
        for aug_idx in range(int(cfg.n_aug_per_reference))
    ]
    workers = _resolve_worker_count(int(cfg.aug_card_generation_workers), len(tasks))
    print(f"Generating {len(tasks):,} augmented card crops with {workers} worker(s)...")
    if workers == 1:
        rows = [_card_generation_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_card_generation_task, tasks, chunksize=max(1, int(cfg.aug_card_generation_chunk_size))))
    rows.sort(key=lambda row: row["image_id"])
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    with paths.aug_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "card", "image_path", "mask_path", "source_image_id", "reference_index", "augmentation_index"])
        writer.writeheader()
        writer.writerows(rows)
    state["aug_rows"] = rows
    print(f"Wrote {len(rows):,} augmented-card rows -> {paths.aug_csv_path}")
    return state


def run_scene_preview(state: dict[str, Any]) -> dict[str, Any]:
    """Render one unsaved scene for quick notebook inspection."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: AugmentationPaths = state["paths"]
    scene_bgr, mask, metadata = compose_augmented_scene(state["card_assets"], np.random.default_rng(cfg.seed + 999), cfg, paths)
    state["preview_scene_bgr"] = scene_bgr
    state["preview_mask"] = mask
    state["preview_metadata"] = metadata
    print(f"Preview scene: active={metadata['active_player']} cards={len(metadata['cards'])}")
    return state


def _scene_generation_task(args: tuple[int, CreateAugmentedDataConfig, AugmentationPaths, list[CardAsset], str]) -> dict[str, Any]:
    scene_index, cfg, paths, card_assets, scene_fmt = args
    rng = np.random.default_rng(cfg.seed + 200_000 + scene_index)
    scene_bgr, scene_mask, metadata = compose_augmented_scene(card_assets, rng, cfg, paths)
    scene_name = f"augmented_scene_{scene_index:05d}"
    image_path = paths.scenes_img_dir / f"{scene_name}.{scene_fmt}"
    mask_path = paths.scenes_mask_dir / f"{scene_name}.png"
    _write_image(image_path, scene_bgr, fmt=scene_fmt, jpeg_quality=cfg.scene_jpeg_quality)
    if not cv2.imwrite(str(mask_path), scene_mask):
        raise OSError(f"Failed to write scene mask: {mask_path}")
    return {
        "scene": scene_name,
        "image_path": str(image_path.relative_to(paths.project_root)),
        "mask_path": str(mask_path.relative_to(paths.project_root)),
        **metadata,
    }


def run_scene_generation(state: dict[str, Any]) -> dict[str, Any]:
    """Generate synthetic tabletop scenes, segmentation masks, and augmented_scenes.json."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: AugmentationPaths = state["paths"]
    scene_fmt: str = state["scene_fmt"]
    tasks = [(idx, cfg, paths, state["card_assets"], scene_fmt) for idx in range(int(cfg.n_scenes))]
    workers = _resolve_worker_count(int(cfg.scene_generation_workers), len(tasks))
    print(f"Generating {len(tasks):,} augmented scenes with {workers} worker(s)...")
    if workers == 1:
        records = [_scene_generation_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_scene_generation_task, tasks, chunksize=max(1, int(cfg.scene_generation_chunk_size))))
    records.sort(key=lambda row: row["scene"])
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    paths.scenes_labels_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    state["scene_metadata"] = records
    print(f"Wrote {len(records):,} scene labels -> {paths.scenes_labels_path}")
    return state


def run_full_augmentation_pipeline(
    cfg: CreateAugmentedDataConfig | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Recreate augmented cards and scenes from the reference card crops."""
    state = initialize_create_augmented_data_pipeline(cfg, project_root=project_root)
    state = run_card_generation(state)
    state = run_scene_generation(state)
    return state


def plot_scene_preview(state: dict[str, Any]) -> None:
    scene_bgr = state.get("preview_scene_bgr")
    mask = state.get("preview_mask")
    metadata = state.get("preview_metadata")
    if scene_bgr is None or mask is None or metadata is None:
        print("No preview scene available. Run run_scene_preview(state) first.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Preview | active={metadata['active_player']} cards={len(metadata['cards'])}")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Generated foreground mask")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def _training_path(project_root: Path, stored_path: str | Path) -> Path:
    parts = Path(str(stored_path).replace("\\", "/")).parts
    if "training_data" in parts:
        return project_root.joinpath(*parts[parts.index("training_data"):])
    return project_root / Path(*parts)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def summarize_augmented_data(project_root: Path, sample_count: int = 3, seed: int = 42) -> dict[str, Any]:
    """Load generated-data metadata and return compact report statistics."""
    project_root = Path(project_root)
    training_data = project_root / "training_data"
    labels_dir = training_data / "object_labels"
    augmented_data = training_data / "augmented_data"
    reference_rows = _read_csv(labels_dir / "reference_cards.csv")
    augmented_rows = _read_csv(labels_dir / "augmented_cards.csv")
    scene_records = json.loads((labels_dir / "augmented_scenes.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(scene_records), size=min(sample_count, len(scene_records)), replace=False)
    sample_records = [scene_records[int(index)] for index in sorted(sample_indices)]

    player_slots = []
    for record in scene_records:
        counts = record.get("player_card_counts", {})
        player_slots.extend(int(counts.get(player, 0)) for player in ("p1", "p2", "p3", "p4"))

    return {
        "reference_count": len(reference_rows),
        "reference_classes": len({row["card"] for row in reference_rows}),
        "augmented_card_count": len(augmented_rows),
        "augmented_card_classes": len({row["card"] for row in augmented_rows}),
        "scene_count": len(scene_records),
        "scene_image_count": len(list((augmented_data / "scene_images").glob("*.jpg"))),
        "scene_mask_count": len(list((augmented_data / "scene_masks").glob("*.png"))),
        "active_player_counts": Counter(str(record.get("active_player", "unknown")) for record in scene_records),
        "cards_per_scene": [len(record.get("cards", [])) for record in scene_records],
        "cards_per_player_slot": player_slots,
        "class_counts": Counter(row["card"] for row in augmented_rows),
        "sample_records": sample_records,
    }


def print_augmentation_summary(summary: dict[str, Any]) -> None:
    cards_per_scene = summary["cards_per_scene"]
    print(
        "Generated data: "
        f"{summary['reference_count']:,} reference crops, "
        f"{summary['augmented_card_count']:,} augmented crops, "
        f"{summary['scene_count']:,} scenes/{summary['scene_mask_count']:,} masks."
    )
    print(
        "Coverage: "
        f"{summary['reference_classes']} reference classes, "
        f"{summary['augmented_card_classes']} augmented classes, "
        f"cards/scene mean={np.mean(cards_per_scene):.2f} "
        f"range={min(cards_per_scene)}-{max(cards_per_scene)}, "
        f"active={dict(summary['active_player_counts'])}."
    )


def _annotate_bars(axis: plt.Axes) -> None:
    for patch in axis.patches:
        height = float(patch.get_height())
        if height <= 0:
            continue
        axis.annotate(
            f"{height:,.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )


def plot_augmentation_overview(project_root: Path, summary: dict[str, Any], example_count: int = 2) -> None:
    """Single report figure for generated-data balance and visual sanity checks."""
    records = list(summary.get("sample_records", []))[: max(0, example_count)]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        squeeze=False,
    )

    active_counts = summary["active_player_counts"]
    active_labels = ["p1", "p2", "p3", "p4"]
    axes[0, 0].bar(active_labels, [active_counts.get(label, 0) for label in active_labels], color="#4e79a7")
    axes[0, 0].set_title("Active-player balance")
    axes[0, 0].set_ylabel("Scenes")
    _annotate_bars(axes[0, 0])

    cards_per_scene = summary["cards_per_scene"]
    axes[0, 1].hist(cards_per_scene, bins=range(0, max(cards_per_scene) + 2), color="#59a14f", edgecolor="white")
    axes[0, 1].axvline(float(np.mean(cards_per_scene)), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Visible cards per scene")
    axes[0, 1].set_xlabel("Cards")
    axes[0, 1].set_ylabel("Scenes")

    slot_counts = Counter(summary["cards_per_player_slot"])
    slot_x = sorted(slot_counts)
    axes[0, 2].bar(slot_x, [slot_counts[value] for value in slot_x], color="#f28e2b")
    axes[0, 2].set_title("Cards per player slot")
    axes[0, 2].set_xlabel("Cards in hand")
    axes[0, 2].set_ylabel("Player slots")
    _annotate_bars(axes[0, 2])

    class_counts = np.array(list(summary["class_counts"].values()), dtype=float)
    axes[0, 3].hist(class_counts, bins=min(18, max(5, len(class_counts) // 3)), color="#af7aa1", edgecolor="white")
    axes[0, 3].axvline(float(class_counts.mean()), color="black", linestyle="--", linewidth=1)
    axes[0, 3].set_title("Augmented crops per class")
    axes[0, 3].set_xlabel("Crops")
    axes[0, 3].set_ylabel("Classes")

    project_root = Path(project_root)
    for axis in axes[1]:
        axis.axis("off")

    for record_index, record in enumerate(records[:2]):
        image_axis = axes[1, record_index * 2]
        mask_axis = axes[1, record_index * 2 + 1]
        image_path = _training_path(project_root, record["image_path"])
        mask_path = _training_path(project_root, record["mask_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            continue

        image_axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for card in record.get("cards", []):
            x0, y0, x1, y1 = map(int, card.get("bbox", (0, 0, 0, 0)))
            image_axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.1))
        image_axis.set_title(
            f"{record['scene']} | active={record.get('active_player')} | cards={len(record.get('cards', []))}",
            fontsize=9,
        )
        image_axis.axis("off")

        mask_axis.imshow(mask, cmap="gray", vmin=0, vmax=255)
        mask_axis.set_title("Generated foreground mask", fontsize=9)
        mask_axis.axis("off")

    fig.suptitle("Generated training data audit", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_augmentation_summary(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    active_counts = summary["active_player_counts"]
    active_labels = ["p1", "p2", "p3", "p4"]
    axes[0].bar(active_labels, [active_counts.get(label, 0) for label in active_labels], color="#4e79a7")
    axes[0].set_title("Active-player token balance")
    axes[0].set_ylabel("Scenes")

    axes[1].hist(summary["cards_per_scene"], bins=range(0, max(summary["cards_per_scene"]) + 2), color="#59a14f", edgecolor="black")
    axes[1].set_title("Cards per generated scene")
    axes[1].set_xlabel("Visible cards")

    slot_counts = Counter(summary["cards_per_player_slot"])
    slot_x = sorted(slot_counts)
    axes[2].bar(slot_x, [slot_counts[value] for value in slot_x], color="#f28e2b")
    axes[2].set_title("Cards per player slot")
    axes[2].set_xlabel("Cards in hand")

    plt.tight_layout()
    plt.show()


def plot_augmented_scene_examples(project_root: Path, summary: dict[str, Any]) -> None:
    records = summary.get("sample_records", [])
    if not records:
        print("No scene examples available.")
        return

    fig, axes = plt.subplots(len(records), 2, figsize=(12, 4 * len(records)), squeeze=False)
    for row_index, record in enumerate(records):
        image_path = _training_path(Path(project_root), record["image_path"])
        mask_path = _training_path(Path(project_root), record["mask_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            axes[row_index, 0].set_visible(False)
            axes[row_index, 1].set_visible(False)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axes[row_index, 0].imshow(image_rgb)
        for card in record.get("cards", []):
            x0, y0, x1, y1 = map(int, card.get("bbox", (0, 0, 0, 0)))
            axes[row_index, 0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.2))
        axes[row_index, 0].set_title(f"{record['scene']} | active={record.get('active_player')} | cards={len(record.get('cards', []))}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title("Generated card mask")
        axes[row_index, 1].axis("off")

    plt.tight_layout()
    plt.show()
