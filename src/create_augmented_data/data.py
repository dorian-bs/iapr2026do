"""Path discovery, asset dataclasses, and background/token loaders.

Functions accept a `CreateAugmentedDataConfig` plus a `Paths` bundle so the
notebook does not have to thread globals around.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.shared.card_pipeline import find_workspace_root


PLAYERS = ("p1", "p2", "p3", "p4")

TOKEN_STYLE_DIRS = {
    "white_rect": "white_rectangular_black",
    "funky_round": "funky_yellow_round",
}

PLAYER_LAYOUTS = {
    "p1": {"center": (0.50, 0.84), "axis": (1.0, 0.0), "inward": (0.0, -1.0), "angle": 0.0},
    "p2": {"center": (0.84, 0.50), "axis": (0.0, 1.0), "inward": (-1.0, 0.0), "angle": -90.0},
    "p3": {"center": (0.50, 0.16), "axis": (1.0, 0.0), "inward": (0.0, 1.0), "angle": 180.0},
    "p4": {"center": (0.16, 0.50), "axis": (0.0, 1.0), "inward": (1.0, 0.0), "angle": 90.0},
}


@dataclass(frozen=True)
class CardAsset:
    """Reference-card crop and its aligned binary mask."""

    image_id: str
    label: str
    image_bgr: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class TokenAsset:
    """Token crop with a foreground mask."""

    image_bgr: np.ndarray
    mask: np.ndarray
    source_path: Path | None


@dataclass
class Paths:
    """Resolved filesystem paths for input assets and outputs."""

    project_root: Path
    project_dir: Path
    training_data: Path
    reference_csv: Path
    ref_cards_dir: Path
    funky_bg_path: Path
    white_bg_path: Path
    aug_cards_dir: Path
    aug_masks_dir: Path
    aug_labels_dir: Path
    aug_csv_path: Path
    scenes_img_dir: Path
    scenes_mask_dir: Path
    scenes_labels_dir: Path
    scenes_labels_path: Path
    token_asset_dir: Path


def resolve_paths() -> Paths:
    """Locate the workspace root and resolve every asset/output path."""
    project_root = find_workspace_root()
    project_dir = project_root / "project"
    training_data = project_dir / "training_data"
    reference_csv = training_data / "object_labels" / "reference_cards" / "reference_do.csv"
    ref_cards_dir = training_data / "training_images" / "reference_cards"
    funky_bg_path = training_data / "funky_bg.jpg"
    white_bg_path = training_data / "white_bg.jpg"

    aug_cards_dir = training_data / "training_images" / "augmented_cards"
    aug_masks_dir = training_data / "training_masks" / "augmented_cards"
    aug_labels_dir = training_data / "object_labels" / "augmented_cards"
    aug_csv_path = aug_labels_dir / "aug.csv"

    scenes_img_dir = training_data / "training_images" / "augmented_scenes"
    scenes_mask_dir = training_data / "training_masks" / "augmented_scenes"
    scenes_labels_dir = training_data / "object_labels" / "augmented_scenes"
    scenes_labels_path = scenes_labels_dir / "labels.json"

    token_asset_dir = training_data / "training_images" / "token_crops"

    return Paths(
        project_root=project_root,
        project_dir=project_dir,
        training_data=training_data,
        reference_csv=reference_csv,
        ref_cards_dir=ref_cards_dir,
        funky_bg_path=funky_bg_path,
        white_bg_path=white_bg_path,
        aug_cards_dir=aug_cards_dir,
        aug_masks_dir=aug_masks_dir,
        aug_labels_dir=aug_labels_dir,
        aug_csv_path=aug_csv_path,
        scenes_img_dir=scenes_img_dir,
        scenes_mask_dir=scenes_mask_dir,
        scenes_labels_dir=scenes_labels_dir,
        scenes_labels_path=scenes_labels_path,
        token_asset_dir=token_asset_dir,
    )


def ensure_output_dirs(paths: Paths) -> None:
    """Create every output directory and the per-player token folders."""
    for output_dir in (
        paths.aug_cards_dir,
        paths.aug_masks_dir,
        paths.aug_labels_dir,
        paths.scenes_img_dir,
        paths.scenes_mask_dir,
        paths.scenes_labels_dir,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

    for style_dir in TOKEN_STYLE_DIRS.values():
        for player_name in PLAYERS:
            (paths.token_asset_dir / style_dir / player_name).mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: Path) -> list[object]:
    """Return a key that sorts crop_2 before crop_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def read_reference_labels(reference_csv: Path) -> dict[str, str]:
    """Load image_id -> card label mappings from reference.csv."""
    labels: dict[str, str] = {}
    with reference_csv.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = row["image_id"].strip()
            card_label = row["card"].strip()
            if image_id:
                labels[image_id] = card_label
    return labels


def load_crop_aligned_mask(tag_dir: Path, crop_index: int, target_shape: tuple[int, int]) -> np.ndarray:
    """Load aligned crop masks with PNG/JPG support and closed-component fallback."""
    target_height, target_width = target_shape
    mask = None
    for suffix in (".png", ".jpg", ".jpeg"):
        mask_path = tag_dir / "masks" / f"mask_{crop_index}{suffix}"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            break

    if mask is None:
        closed_mask = None
        for suffix in (".png", ".jpg", ".jpeg"):
            closed_path = tag_dir / "closed_components" / f"closed_component_{crop_index}{suffix}"
            closed_mask = cv2.imread(str(closed_path), cv2.IMREAD_GRAYSCALE)
            if closed_mask is not None:
                break
        if closed_mask is not None and np.count_nonzero(closed_mask) > 0:
            bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(closed_mask)
            mask = closed_mask[bbox_y:bbox_y + bbox_height, bbox_x:bbox_x + bbox_width]
        else:
            mask = np.full((target_height, target_width), 255, dtype=np.uint8)

    if mask.shape[:2] != (target_height, target_width):
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.uint8) * 255


def _collect_reference_crop_indices(tag_dir: Path) -> list[int]:
    indices: set[int] = set()
    for pattern in (
        "crops_rgba/crop_*.png",
        "crops/crop_*.png",
        "crops/crop_*.jpg",
        "crops/crop_*.jpeg",
    ):
        for crop_path in tag_dir.glob(pattern):
            try:
                indices.add(int(crop_path.stem.split("_")[-1]))
            except ValueError:
                continue
    return sorted(indices)


def _load_reference_crop_and_mask(tag_dir: Path, crop_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    # Prefer transparent PNG references if available.
    rgba_path = tag_dir / "crops_rgba" / f"crop_{crop_index}.png"
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is not None:
        if rgba.ndim == 2:
            image_bgr = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
            mask = np.where(rgba > 127, 255, 0).astype(np.uint8)
            return image_bgr, mask

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            image_bgr = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
            mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
            return image_bgr, mask

        if rgba.ndim == 3 and rgba.shape[2] == 3:
            image_bgr = rgba
            mask = load_crop_aligned_mask(tag_dir, crop_index, image_bgr.shape[:2])
            return image_bgr, mask

    for suffix in (".png", ".jpg", ".jpeg"):
        crop_path = tag_dir / "crops" / f"crop_{crop_index}{suffix}"
        image_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        mask = load_crop_aligned_mask(tag_dir, crop_index, image_bgr.shape[:2])
        return image_bgr, mask

    return None


def load_reference_card_assets(paths: Paths) -> list[CardAsset]:
    """Load every labeled reference crop and its aligned mask."""
    labels = read_reference_labels(paths.reference_csv)
    assets: list[CardAsset] = []

    for tag_dir in sorted([path for path in paths.ref_cards_dir.iterdir() if path.is_dir()]):
        crop_indices = _collect_reference_crop_indices(tag_dir)
        if not crop_indices:
            continue

        for crop_index in crop_indices:
            image_id = f"{tag_dir.name}_crop_{crop_index}"
            label = labels.get(image_id)
            if label is None:
                continue

            loaded = _load_reference_crop_and_mask(tag_dir, crop_index)
            if loaded is None:
                continue
            image_bgr, mask = loaded

            assets.append(CardAsset(image_id=image_id, label=label, image_bgr=image_bgr, mask=mask))

    if not assets:
        raise RuntimeError("No reference card crops were loaded. Run the reference-card extraction notebook first.")

    return assets


# ---------- Background helpers ---------------------------------------------------


def make_fallback_funky_background(scene_height: int, scene_width: int, rng: np.random.Generator) -> np.ndarray:
    """Create a colorful procedural fallback if funky_bg.jpg is missing."""
    row_indices, col_indices = np.indices((scene_height, scene_width))
    base = np.zeros((scene_height, scene_width, 3), dtype=np.float32)
    base[..., 0] = 80 + 70 * np.sin(col_indices / 55.0)
    base[..., 1] = 120 + 80 * np.sin((row_indices + col_indices) / 85.0)
    base[..., 2] = 120 + 70 * np.cos(row_indices / 45.0)
    noise = rng.normal(0, 12, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def sample_cover_crop(
    source_bgr: np.ndarray,
    scene_height: int,
    scene_width: int,
    rng: np.random.Generator,
    scale_jitter_range: tuple[float, float],
    allow_flip: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize and crop a background image so it fully covers the scene canvas."""
    source_height, source_width = source_bgr.shape[:2]
    cover_scale = max(scene_width / source_width, scene_height / source_height)
    cover_scale *= float(rng.uniform(*scale_jitter_range))

    resized_width = max(scene_width, int(round(source_width * cover_scale)))
    resized_height = max(scene_height, int(round(source_height * cover_scale)))
    resized = cv2.resize(source_bgr, (resized_width, resized_height), interpolation=interpolation)

    if allow_flip and rng.random() < 0.5:
        resized = cv2.flip(resized, 1)

    crop_x = int(rng.integers(0, max(1, resized_width - scene_width + 1)))
    crop_y = int(rng.integers(0, max(1, resized_height - scene_height + 1)))
    return resized[crop_y:crop_y + scene_height, crop_x:crop_x + scene_width].copy()


def apply_local_exposure_field(
    scene_height: int,
    scene_width: int,
    rng: np.random.Generator,
    style_key: str,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Create a smooth multiplicative exposure field with corner and blob variation."""
    yy, xx = np.indices((scene_height, scene_width), dtype=np.float32)
    x_unit = (2.0 * xx / max(1.0, float(scene_width - 1))) - 1.0
    y_unit = (2.0 * yy / max(1.0, float(scene_height - 1))) - 1.0

    theta = float(rng.uniform(-math.pi, math.pi))
    gradient = np.cos(theta) * x_unit + np.sin(theta) * y_unit
    gradient /= max(1.0e-6, float(np.max(np.abs(gradient))))
    gradient_strength = float(rng.uniform(-0.08, 0.08) if style_key == "white_rect" else rng.uniform(-0.12, 0.12))

    corner_x = float(rng.choice([-1.0, 1.0]))
    corner_y = float(rng.choice([-1.0, 1.0]))
    corner_distance = np.sqrt((x_unit - corner_x) ** 2 + (y_unit - corner_y) ** 2)
    corner_distance /= max(1.0e-6, float(corner_distance.max()))
    corner_darkening = float(rng.uniform(0.05, 0.14) if style_key == "white_rect" else rng.uniform(0.06, 0.18))
    corner_term = -corner_darkening * np.clip(1.0 - corner_distance, 0.0, 1.0)

    exposure = 1.0 + gradient_strength * gradient + corner_term
    blob_count = int(rng.integers(1, 3))
    for _ in range(blob_count):
        blob_x = float(rng.uniform(-1.0, 1.0))
        blob_y = float(rng.uniform(-1.0, 1.0))
        blob_sigma = float(rng.uniform(0.25, 0.70))
        blob = np.exp(-((x_unit - blob_x) ** 2 + (y_unit - blob_y) ** 2) / (2.0 * blob_sigma * blob_sigma))
        blob_strength = float(rng.uniform(-0.05, 0.05) if style_key == "white_rect" else rng.uniform(-0.08, 0.08))
        exposure += blob_strength * blob

    exposure = np.clip(exposure, 0.75, 1.25).astype(np.float32)

    if blur_sigma > 0.0:
        kernel_size = max(3, int(round(blur_sigma * 6)) | 1)
        exposure = cv2.GaussianBlur(exposure, (kernel_size, kernel_size), sigmaX=blur_sigma)

    return exposure


def apply_lighting_variation(base_bgr: np.ndarray, rng: np.random.Generator, style_key: str) -> np.ndarray:
    """Apply global and local illumination variation to reduce overfitting."""
    image = base_bgr.astype(np.float32)

    if style_key == "white_rect":
        global_gain = float(rng.uniform(0.93, 1.07))
        global_bias = float(rng.uniform(-10.0, 10.0))
        channel_gain = rng.uniform(0.97, 1.03, size=3).astype(np.float32)
    else:
        global_gain = float(rng.uniform(0.90, 1.12))
        global_bias = float(rng.uniform(-14.0, 14.0))
        channel_gain = rng.uniform(0.95, 1.05, size=3).astype(np.float32)

    image = image * global_gain + global_bias
    image *= channel_gain.reshape(1, 1, 3)

    local_exposure = apply_local_exposure_field(
        scene_height=image.shape[0],
        scene_width=image.shape[1],
        rng=rng,
        style_key=style_key,
        blur_sigma=float(rng.uniform(8.0, 16.0)),
    )
    image *= local_exposure[:, :, None]

    return np.clip(image, 0, 255).astype(np.uint8)


# ---------- Background / token caches -------------------------------------------


@dataclass
class AssetCaches:
    """Runtime caches so backgrounds and tokens are loaded once per session."""

    white_bg_image: np.ndarray | None = None
    funky_bg_image: np.ndarray | None = None
    token_paths: dict[tuple[str, str], list[Path]] = field(default_factory=dict)
    token_resized: dict[tuple[str, str, int, int], list[TokenAsset]] = field(default_factory=dict)
    token_placeholders: dict[tuple[str, str, int, int], TokenAsset] = field(default_factory=dict)


def build_asset_caches(paths: Paths) -> AssetCaches:
    """Pre-load background images so each scene call can skip disk IO."""
    return AssetCaches(
        white_bg_image=cv2.imread(str(paths.white_bg_path), cv2.IMREAD_COLOR),
        funky_bg_image=cv2.imread(str(paths.funky_bg_path), cv2.IMREAD_COLOR),
    )


def make_white_background(
    scene_height: int,
    scene_width: int,
    rng: np.random.Generator,
    paths: Paths,
    caches: AssetCaches,
) -> np.ndarray:
    """Load and crop the white background asset to fill the scene."""
    if caches.white_bg_image is None:
        raise FileNotFoundError(f"Missing white background image at {paths.white_bg_path}")
    return sample_cover_crop(
        source_bgr=caches.white_bg_image,
        scene_height=scene_height,
        scene_width=scene_width,
        rng=rng,
        scale_jitter_range=(1.0, 1.08),
        allow_flip=False,
    )


def make_funky_background(
    scene_height: int,
    scene_width: int,
    rng: np.random.Generator,
    paths: Paths,
    caches: AssetCaches,
) -> np.ndarray:
    """Crop and resize the project funky background to fill the scene."""
    if caches.funky_bg_image is None:
        return make_fallback_funky_background(scene_height, scene_width, rng)
    return sample_cover_crop(
        source_bgr=caches.funky_bg_image,
        scene_height=scene_height,
        scene_width=scene_width,
        rng=rng,
        scale_jitter_range=(1.0, 1.18),
        allow_flip=True,
    )


def make_background(
    style_key: str,
    scene_height: int,
    scene_width: int,
    rng: np.random.Generator,
    paths: Paths,
    caches: AssetCaches,
) -> np.ndarray:
    """Create a style-specific background with scene-level illumination variation."""
    if style_key == "white_rect":
        base_bgr = make_white_background(scene_height, scene_width, rng, paths, caches)
    else:
        base_bgr = make_funky_background(scene_height, scene_width, rng, paths, caches)
    return apply_lighting_variation(base_bgr, rng, style_key)


# ---------- Token helpers --------------------------------------------------------


def token_asset_paths(style_key: str, player_name: str, paths: Paths, caches: AssetCaches) -> list[Path]:
    """Return token crop candidates for one style and player slot."""
    cache_key = (style_key, player_name)
    cached = caches.token_paths.get(cache_key)
    if cached is not None:
        return cached

    token_dir = paths.token_asset_dir / TOKEN_STYLE_DIRS[style_key] / player_name
    candidates: list[Path] = []
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(sorted(token_dir.glob(extension), key=natural_sort_key))

    caches.token_paths[cache_key] = candidates
    return candidates


def token_mask_from_image(raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a token image into BGR pixels plus a foreground mask."""
    if raw_image.ndim == 2:
        image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        mask = (raw_image < 245).astype(np.uint8) * 255
        return image_bgr, mask

    if raw_image.shape[2] == 4:
        image_bgr = raw_image[:, :, :3]
        alpha = raw_image[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
        return image_bgr, mask

    image_bgr = raw_image[:, :, :3]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = ((gray < 245) | (hsv[:, :, 1] > 30)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return image_bgr, mask


def crop_to_mask_bounds(image_bgr: np.ndarray, mask: np.ndarray, padding: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Trim empty token borders before resizing."""
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return image_bgr, mask

    bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(nonzero)
    top = max(0, bbox_y - padding)
    left = max(0, bbox_x - padding)
    bottom = min(mask.shape[0], bbox_y + bbox_height + padding)
    right = min(mask.shape[1], bbox_x + bbox_width + padding)
    return image_bgr[top:bottom, left:right], mask[top:bottom, left:right]


def resize_token_to_canvas(token: TokenAsset, target_size: tuple[int, int]) -> TokenAsset:
    """Resize a token crop into a fixed transparent canvas."""
    target_height, target_width = target_size
    cropped_image, cropped_mask = crop_to_mask_bounds(token.image_bgr, token.mask)
    source_height, source_width = cropped_image.shape[:2]
    scale = min(target_width / max(1, source_width), target_height / max(1, source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized_image = cv2.resize(cropped_image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(cropped_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    canvas_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas_image[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_image
    canvas_mask[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_mask
    return TokenAsset(canvas_image, canvas_mask, token.source_path)


def make_placeholder_token(style_key: str, player_name: str, target_size: tuple[int, int]) -> TokenAsset:
    """Draw a synthetic token until real player-position token crops are available."""
    target_height, target_width = target_size
    image_bgr = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    margin = max(2, int(round(min(target_height, target_width) * 0.08)))

    if style_key == "white_rect":
        cv2.rectangle(
            image_bgr,
            (margin, margin),
            (target_width - margin - 1, target_height - margin - 1),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.rectangle(
            mask,
            (margin, margin),
            (target_width - margin - 1, target_height - margin - 1),
            255,
            thickness=-1,
        )
    else:
        radius = max(2, (min(target_height, target_width) // 2) - margin)
        center = (target_width // 2, target_height // 2)
        cv2.circle(image_bgr, center, radius, (0, 220, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)

    return TokenAsset(image_bgr, mask, None)


def token_target_size(style_key: str, player_name: str, scene_height: int) -> tuple[int, int]:
    """Choose a token canvas size that fits each player side."""
    base_size = max(28, int(round(scene_height * 0.055)))
    if style_key == "white_rect":
        if player_name in ("p2", "p4"):
            return base_size * 2, base_size
        return base_size, base_size * 2
    return base_size * 2, base_size * 2


def cached_resized_tokens(
    style_key: str,
    player_name: str,
    scene_height: int,
    paths: Paths,
    caches: AssetCaches,
) -> list[TokenAsset]:
    """Load and preprocess token assets once for each style/player/size triple."""
    target_height, target_width = token_target_size(style_key, player_name, scene_height)
    cache_key = (style_key, player_name, target_height, target_width)
    cached = caches.token_resized.get(cache_key)
    if cached is not None:
        return cached

    prepared: list[TokenAsset] = []
    for candidate_path in token_asset_paths(style_key, player_name, paths, caches):
        raw_image = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            continue
        image_bgr, mask = token_mask_from_image(raw_image)
        prepared.append(
            resize_token_to_canvas(
                TokenAsset(image_bgr, mask, candidate_path),
                (target_height, target_width),
            )
        )

    caches.token_resized[cache_key] = prepared
    return prepared


def load_or_make_token(
    style_key: str,
    player_name: str,
    rng: np.random.Generator,
    scene_height: int,
    paths: Paths,
    caches: AssetCaches,
) -> TokenAsset:
    """Load a real token crop if present, otherwise draw the configured placeholder."""
    prepared = cached_resized_tokens(style_key, player_name, scene_height, paths, caches)
    if prepared:
        return prepared[int(rng.integers(0, len(prepared)))]

    target_size = token_target_size(style_key, player_name, scene_height)
    placeholder_key = (style_key, player_name, target_size[0], target_size[1])
    placeholder = caches.token_placeholders.get(placeholder_key)
    if placeholder is None:
        placeholder = make_placeholder_token(style_key, player_name, target_size)
        caches.token_placeholders[placeholder_key] = placeholder
    return placeholder
