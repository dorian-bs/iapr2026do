"""Card transformation, scene composition, and augmented-card rendering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import CreateAugmentedDataConfig
from .data import (
    AssetCaches,
    CardAsset,
    PLAYERS,
    PLAYER_LAYOUTS,
    Paths,
    load_or_make_token,
    make_background,
    make_funky_background,
)


def transform_card(
    asset: CardAsset,
    target_height: float,
    angle_degrees: float,
    rng: np.random.Generator,
    apply_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize one card crop+mask, then apply photometric jitter and optional rotation."""
    source_height, source_width = asset.image_bgr.shape[:2]
    scale = float(target_height) / max(1, source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized_image = cv2.resize(asset.image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(asset.mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    alpha = float(rng.uniform(0.88, 1.14))
    beta = float(rng.uniform(-16, 16))
    resized_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)
    noise_std = float(rng.uniform(0.0, 5.5))
    if noise_std > 0.1:
        noise = rng.normal(0.0, noise_std, size=resized_image.shape).astype(np.float32)
        resized_image = np.clip(resized_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if not apply_rotation or abs(float(angle_degrees)) < 1e-6:
        return resized_image, (resized_mask > 127).astype(np.uint8) * 255

    center = (resized_width / 2.0, resized_height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    rotated_width = int(round(resized_height * abs_sin + resized_width * abs_cos))
    rotated_height = int(round(resized_height * abs_cos + resized_width * abs_sin))

    rotation_matrix[0, 2] += rotated_width / 2.0 - center[0]
    rotation_matrix[1, 2] += rotated_height / 2.0 - center[1]

    rotated_image = cv2.warpAffine(
        resized_image,
        rotation_matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        resized_mask,
        rotation_matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, (rotated_mask > 127).astype(np.uint8) * 255


def paste_patch(
    scene_bgr: np.ndarray,
    instance_map: np.ndarray,
    patch_bgr: np.ndarray,
    patch_mask: np.ndarray,
    center_xy: tuple[float, float],
    instance_id: int | None = None,
    erase_instances: bool = False,
) -> tuple[int, int, int, int] | None:
    """Paste a masked patch and optionally write or erase instance ids."""
    scene_height, scene_width = scene_bgr.shape[:2]
    patch_height, patch_width = patch_bgr.shape[:2]
    center_x, center_y = center_xy

    patch_left = int(round(center_x - patch_width / 2.0))
    patch_top = int(round(center_y - patch_height / 2.0))
    patch_right = patch_left + patch_width
    patch_bottom = patch_top + patch_height

    dst_left = max(0, patch_left)
    dst_top = max(0, patch_top)
    dst_right = min(scene_width, patch_right)
    dst_bottom = min(scene_height, patch_bottom)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        return None

    src_left = dst_left - patch_left
    src_top = dst_top - patch_top
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)

    mask_patch = patch_mask[src_top:src_bottom, src_left:src_right] > 127
    if not np.any(mask_patch):
        return None

    image_patch = patch_bgr[src_top:src_bottom, src_left:src_right]
    scene_roi = scene_bgr[dst_top:dst_bottom, dst_left:dst_right]
    instance_roi = instance_map[dst_top:dst_bottom, dst_left:dst_right]

    scene_roi[mask_patch] = image_patch[mask_patch]
    if instance_id is not None:
        instance_roi[mask_patch] = instance_id
    if erase_instances:
        instance_roi[mask_patch] = 0

    visible_y, visible_x = np.where(mask_patch)
    bbox_left = int(dst_left + visible_x.min())
    bbox_top = int(dst_top + visible_y.min())
    bbox_right = int(dst_left + visible_x.max() + 1)
    bbox_bottom = int(dst_top + visible_y.max() + 1)
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def separated_binary_mask(
    instance_map: np.ndarray,
    gap_pixels: int,
    min_visible_card_area: int,
) -> np.ndarray:
    """Convert visible card instances to one binary mask with small gaps between instances."""
    binary_mask = np.zeros(instance_map.shape, dtype=np.uint8)
    instance_ids = [int(instance_id) for instance_id in np.unique(instance_map) if instance_id > 0]

    if gap_pixels > 0:
        kernel_size = 2 * int(gap_pixels) + 1
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:
        gap_kernel = None

    for instance_id in instance_ids:
        visible_instance = (instance_map == instance_id).astype(np.uint8) * 255
        if np.count_nonzero(visible_instance) < min_visible_card_area:
            continue
        if gap_kernel is not None:
            visible_instance = cv2.erode(visible_instance, gap_kernel, iterations=1)
        if np.count_nonzero(visible_instance) >= min_visible_card_area:
            binary_mask[visible_instance > 0] = 255

    return binary_mask


def sample_card_asset(card_assets: list[CardAsset], rng: np.random.Generator) -> CardAsset:
    """Choose one reference card crop."""
    return card_assets[int(rng.integers(0, len(card_assets)))]


def clamped_center(center_xy: np.ndarray, scene_width: int, scene_height: int) -> tuple[float, float]:
    """Keep placement centers comfortably inside the image canvas."""
    margin_x = scene_width * 0.07
    margin_y = scene_height * 0.07
    center_x = float(np.clip(center_xy[0], margin_x, scene_width - margin_x))
    center_y = float(np.clip(center_xy[1], margin_y, scene_height - margin_y))
    return center_x, center_y


def player_card_placements(
    player_name: str,
    card_count: int,
    card_assets: list[CardAsset],
    cfg: CreateAugmentedDataConfig,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Sample card placements in the expected table region for one player."""
    if card_count <= 0:
        return []

    scene_width = cfg.scene_width
    scene_height = cfg.scene_height
    layout = PLAYER_LAYOUTS[player_name]
    base_center = np.array([layout["center"][0] * scene_width, layout["center"][1] * scene_height], dtype=np.float32)
    axis = np.array(layout["axis"], dtype=np.float32)
    inward = np.array(layout["inward"], dtype=np.float32)
    base_angle = float(layout["angle"])

    hand_center_jitter_scale = float(rng.uniform(*cfg.player_hand_center_jitter_fraction_range))
    base_center += np.array(
        [
            float(rng.normal(0, scene_width * hand_center_jitter_scale)),
            float(rng.normal(0, scene_height * hand_center_jitter_scale)),
        ],
        dtype=np.float32,
    )

    target_height = float(rng.uniform(*cfg.player_card_height_fraction_range) * scene_height)
    approximate_card_width = target_height * 0.64
    slot_spacing = approximate_card_width * float(rng.uniform(*cfg.player_slot_spacing_factor_range))
    offsets = (np.arange(card_count, dtype=np.float32) - (card_count - 1) / 2.0) * slot_spacing

    slot_offset_jitter_scale = float(rng.uniform(*cfg.player_slot_offset_jitter_fraction_range))
    inward_jitter_scale = float(rng.uniform(*cfg.player_inward_jitter_fraction_range))
    rotation_jitter_std = float(rng.uniform(*cfg.player_rotation_jitter_deg_range))

    placements: list[dict[str, object]] = []
    for slot_index, offset in enumerate(offsets):
        offset_jitter = float(rng.normal(0, slot_spacing * slot_offset_jitter_scale))
        inward_jitter = float(rng.normal(0, scene_height * inward_jitter_scale))
        center = base_center + axis * (float(offset) + offset_jitter) + inward * inward_jitter
        center_xy = clamped_center(center, scene_width, scene_height)
        angle_degrees = base_angle + float(rng.normal(0, rotation_jitter_std))
        card_height = target_height * float(rng.uniform(0.90, 1.10))

        placements.append(
            {
                "owner": player_name,
                "slot_index": slot_index,
                "asset": sample_card_asset(card_assets, rng),
                "center": center_xy,
                "target_height": card_height,
                "angle_degrees": angle_degrees,
            }
        )

    return placements


def center_card_placement(
    card_assets: list[CardAsset],
    cfg: CreateAugmentedDataConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Sample the required center-card placement."""
    scene_width = cfg.scene_width
    scene_height = cfg.scene_height
    center = np.array(
        [
            scene_width * float(rng.uniform(*cfg.center_position_fraction_range_x)),
            scene_height * float(rng.uniform(*cfg.center_position_fraction_range_y)),
        ],
        dtype=np.float32,
    )
    return {
        "owner": "center",
        "slot_index": 0,
        "asset": sample_card_asset(card_assets, rng),
        "center": clamped_center(center, scene_width, scene_height),
        "target_height": float(rng.uniform(*cfg.center_card_height_fraction_range) * scene_height),
        "angle_degrees": float(rng.uniform(*cfg.center_angle_range_deg)),
    }


def token_side_sign(side_mode: str, rng: np.random.Generator) -> float:
    """Map token side mode to the lateral sign around the active player."""
    normalized_mode = side_mode.strip().lower()
    if normalized_mode == "right":
        return 1.0
    if normalized_mode == "left":
        return -1.0
    if normalized_mode == "center":
        return 0.0
    if normalized_mode == "random":
        return float(rng.choice([-1.0, 1.0]))
    raise ValueError(f"Invalid token_side_mode={side_mode!r}. Expected right, left, center, or random.")


def token_center(
    active_player: str,
    cfg: CreateAugmentedDataConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample a token center near the active player using player-relative side placement."""
    scene_width = cfg.scene_width
    scene_height = cfg.scene_height
    layout = PLAYER_LAYOUTS[active_player]
    base_center = np.array([layout["center"][0] * scene_width, layout["center"][1] * scene_height], dtype=np.float32)
    inward = np.array(layout["inward"], dtype=np.float32)

    # Player-right is a +90 deg rotation in image coordinates.
    right = np.array([-inward[1], inward[0]], dtype=np.float32)

    inward_distance = scene_height * float(rng.uniform(*cfg.token_inward_distance_fraction_range))
    lateral_distance = scene_height * float(rng.uniform(*cfg.token_lateral_distance_fraction_range))
    side_sign = token_side_sign(cfg.token_side_mode, rng)

    jitter_scale = float(cfg.token_center_jitter_fraction)
    jitter_x = float(rng.normal(0, scene_width * jitter_scale))
    jitter_y = float(rng.normal(0, scene_height * jitter_scale))

    center = base_center + inward * inward_distance + right * (side_sign * lateral_distance)
    center += np.array([jitter_x, jitter_y], dtype=np.float32)

    margin_x = scene_width * float(cfg.token_clamp_margin_fraction)
    margin_y = scene_height * float(cfg.token_clamp_margin_fraction)
    center_x = float(np.clip(center[0], margin_x, scene_width - margin_x))
    center_y = float(np.clip(center[1], margin_y, scene_height - margin_y))
    return center_x, center_y


def compose_augmented_scene(
    card_assets: list[CardAsset],
    rng: np.random.Generator,
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compose one augmented scene, its separated card mask, and metadata."""
    scene_width = cfg.scene_width
    scene_height = cfg.scene_height
    style_key = "white_rect" if rng.random() < 0.5 else "funky_round"
    scene_bgr = make_background(style_key, scene_height, scene_width, rng, paths, caches)
    instance_map = np.zeros((scene_height, scene_width), dtype=np.uint16)

    player_counts = {
        player_name: int(rng.integers(cfg.min_cards_per_player, cfg.max_cards_per_player + 1))
        for player_name in PLAYERS
    }

    placements: list[dict[str, object]] = []
    for player_name in PLAYERS:
        placements.extend(
            player_card_placements(player_name, player_counts[player_name], card_assets, cfg, rng)
        )
    placements.append(center_card_placement(card_assets, cfg, rng))

    card_metadata: list[dict[str, object]] = []
    next_instance_id = 1
    for placement in placements:
        asset = placement["asset"]
        assert isinstance(asset, CardAsset)
        patch_bgr, patch_mask = transform_card(
            asset,
            target_height=float(placement["target_height"]),
            angle_degrees=float(placement["angle_degrees"]),
            rng=rng,
        )
        bbox = paste_patch(
            scene_bgr,
            instance_map,
            patch_bgr,
            patch_mask,
            center_xy=placement["center"],
            instance_id=next_instance_id,
        )
        if bbox is None:
            continue

        card_metadata.append(
            {
                "instance_id": next_instance_id,
                "owner": placement["owner"],
                "slot_index": int(placement["slot_index"]),
                "image_id": asset.image_id,
                "label": asset.label,
                "bbox": list(map(int, bbox)),
                "angle_degrees": float(placement["angle_degrees"]),
            }
        )
        next_instance_id += 1

    active_player = PLAYERS[int(rng.integers(0, len(PLAYERS)))]
    token = load_or_make_token(style_key, active_player, rng, scene_height, paths, caches)
    token_bbox = paste_patch(
        scene_bgr,
        instance_map,
        token.image_bgr,
        token.mask,
        center_xy=token_center(active_player, cfg, rng),
        erase_instances=True,
    )

    separated_mask = separated_binary_mask(instance_map, cfg.mask_gap_pixels, cfg.min_visible_card_area)
    metadata = {
        "style": style_key,
        "active_player": active_player,
        "player_card_counts": player_counts,
        "center_card_count": 1,
        "token_bbox": list(map(int, token_bbox)) if token_bbox is not None else None,
        "token_source": str(token.source_path) if token.source_path is not None else "placeholder",
        "mask_gap_pixels": cfg.mask_gap_pixels,
        "cards": card_metadata,
    }
    return scene_bgr, separated_mask, metadata


def render_augmented_card(
    asset: CardAsset,
    rng: np.random.Generator,
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one single-card crop plus its aligned binary mask."""
    canvas_h, canvas_w = cfg.aug_card_canvas

    source_h, source_w = asset.image_bgr.shape[:2]
    max_h = int(round(canvas_h * 0.82))
    max_w = int(round(canvas_w * 0.82))
    fit_scale = min(max_h / max(1, source_h), max_w / max(1, source_w), 1.0)
    target_height = float(max(1, int(round(source_h * fit_scale))))

    patch_bgr, patch_mask = transform_card(
        asset,
        target_height=target_height,
        angle_degrees=0.0,
        rng=rng,
        apply_rotation=False,
    )

    if rng.random() < 0.6:
        tone = int(rng.integers(170, 256))
        canvas = np.full((canvas_h, canvas_w, 3), tone, dtype=np.uint8)
        if rng.random() < 0.5:
            noise = rng.normal(0, 8, size=canvas.shape)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        canvas = make_funky_background(canvas_h, canvas_w, rng, paths, caches)

    instance_map = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    shift_x = float(rng.normal(0, canvas_w * cfg.aug_card_shift_fraction))
    shift_y = float(rng.normal(0, canvas_h * cfg.aug_card_shift_fraction))
    center_xy = (canvas_w / 2.0 + shift_x, canvas_h / 2.0 + shift_y)

    paste_patch(
        scene_bgr=canvas,
        instance_map=instance_map,
        patch_bgr=patch_bgr,
        patch_mask=patch_mask,
        center_xy=center_xy,
        instance_id=1,
    )

    foreground = instance_map > 0
    if np.any(foreground):
        ys, xs = np.where(foreground)
        padding = int(rng.integers(3, 10))
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(canvas_w, int(xs.max()) + 1 + padding)
        y1 = min(canvas_h, int(ys.max()) + 1 + padding)
        canvas = canvas[y0:y1, x0:x1]
        instance_map = instance_map[y0:y1, x0:x1]

    card_mask = np.where(instance_map > 0, 255, 0).astype(np.uint8)
    return canvas, card_mask
