"""Configuration dataclass for the augmented-data generation pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CreateAugmentedDataConfig:
    """All tunable knobs for the augmented-card and augmented-scene generators."""

    seed: int = 67

    # Single-card augmentation knobs.
    n_aug_per_reference: int = 500
    aug_card_canvas: tuple[int, int] = (256, 256)
    aug_card_height_range: tuple[int, int] = (140, 220)
    aug_card_angle_range_deg: tuple[float, float] = (-25.0, 25.0)
    aug_card_shift_fraction: float = 0.12
    # RGB card output: JPEG is usually a better speed/size tradeoff for training.
    aug_card_image_format: str = "jpg"  # png | jpg | jpeg
    aug_card_jpeg_quality: int = 85

    # Scene generation knobs.
    n_scenes: int = 8192
    scene_width: int = 1280
    scene_height: int = 720
    min_cards_per_player: int = 0
    max_cards_per_player: int = 6   
    # Wider erosion gap makes individual cards more separated in target masks.
    mask_gap_pixels: int = 3
    min_visible_card_area: int = 90
    clear_output_dirs: bool = True

    player_card_height_fraction_range: tuple[float, float] = (0.2, 0.22)
    center_card_height_fraction_range: tuple[float, float] = (0.2, 0.22)
    # Wide spacing range allows either stronger overlap or clearly separated cards.
    player_slot_spacing_factor_range: tuple[float, float] = (0.09, 1.5)
    player_hand_center_jitter_fraction_range: tuple[float, float] = (0.006, 0.02)
    player_slot_offset_jitter_fraction_range: tuple[float, float] = (0.05, 0.20)
    player_inward_jitter_fraction_range: tuple[float, float] = (0.008, 0.03)
    player_rotation_jitter_deg_range: tuple[float, float] = (4.0, 45.0)
    center_position_fraction_range_x: tuple[float, float] = (0.40, 0.70)
    center_position_fraction_range_y: tuple[float, float] = (0.40, 0.70)
    center_angle_range_deg: tuple[float, float] = (-70.0, 70.0)
    # RGB scene output: JPEG reduces storage and typically speeds up loading.
    scene_image_format: str = "jpg"  # png | jpg | jpeg
    scene_jpeg_quality: int = 80

    # Card-shadow knobs for scene generation. Direction is sampled once per
    # scene and reused for all cards in that scene.
    scene_shadow_enabled: bool = True
    scene_shadow_offset_fraction_range: tuple[float, float] = (0.006, 0.020)
    scene_shadow_blur_fraction_range: tuple[float, float] = (0.003, 0.010)
    scene_shadow_opacity_range: tuple[float, float] = (0.14, 0.30)

    # Token placement hyperparameters.
    token_inward_distance_fraction_range: tuple[float, float] = (0.0, 0.05)
    # Scale both bounds down to keep side offsets closer to players.
    token_lateral_distance_fraction_range: tuple[float, float] = (0.23, 0.33)
    token_side_mode: str = "right"  # right | left | center | random
    token_center_jitter_fraction: float = 0.008
    token_clamp_margin_fraction: float = 0.02

    # Number of saved scenes to display in the post-generation preview cell.
    preview_saved_sample_count: int = 4
    # Number of single-card previews to display after card generation.
    preview_card_sample_count: int = 12
