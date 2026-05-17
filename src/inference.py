"""End-to-end inference: scene image -> structured UNO game state.

Used by both `main.py` (final Kaggle submission) and the report notebook
(visualisations, benchmarks). Keeping a single inference path here prevents
train/test or notebook/CLI drift.

Pipeline overview
-----------------
1. Segmenter (SceneUNetSmall) produces a per-pixel card-foreground probability.
2. `boxes_from_probability` extracts per-card instance masks + bounding boxes.
3. Each crop+mask is fed to the masked card classifier (4-channel input).
4. Boxes are assigned to a region (center / p1..p4) by geometry only (R6).
5. `active_player` is estimated by a separate detector (see active_player.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.active_player import detect_active_player
from src.shared.card_models import (
    SceneUNetSmall,
    assert_param_cap,
    build_card_classifier,
)
from src.shared.card_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    assign_region,
    boxes_from_probability,
    card_input_to_tensor,
    compose_masked_card_image,
    crop_with_margin,
    format_cards,
    letterbox_image_and_mask,
    segment_scene_probability,
)
from src.shared.model_paths import (
    resolve_classifier_bundle,
    resolve_segmenter_checkpoint,
)


@dataclass
class InferenceConfig:
    """Tunable runtime parameters for the inference pipeline.

    Defaults mirror the values used to produce the submitted Kaggle CSV. The
    classifier-side parameters (image size, mask threshold, normalization) are
    overridden by the saved training config when a bundle provides one.
    """

    segmenter_img_size: int = 256
    segmenter_threshold: float = 0.50
    segmenter_min_component_area: int | None = None
    instance_mask_growth_px: int = 5

    card_img_size: int = 160
    card_mask_threshold: float = 0.50


@dataclass
class CardPrediction:
    """Single classified card extracted from a scene image."""

    box: tuple[int, int, int, int]
    instance_mask: np.ndarray
    label: str
    confidence: float
    region: str           # "center" | "p1" | "p2" | "p3" | "p4"
    mask_coverage: float


@dataclass
class GameState:
    """Final structured prediction for one scene image."""

    image_id: str
    center_card: str
    active_player: str
    player_1_cards: str
    player_2_cards: str
    player_3_cards: str
    player_4_cards: str
    cards: list[CardPrediction] = field(default_factory=list)

    def as_submission_row(self) -> dict[str, str]:
        """Dict matching the exact Kaggle CSV schema (column order matters)."""
        return {
            "image_id": self.image_id,
            "center_card": self.center_card,
            "active_player": self.active_player,
            "player_1_cards": self.player_1_cards,
            "player_2_cards": self.player_2_cards,
            "player_3_cards": self.player_3_cards,
            "player_4_cards": self.player_4_cards,
        }


@dataclass
class InferenceEngine:
    """Holds the loaded models + class table so we pay model loading once."""

    config: InferenceConfig
    device: torch.device
    segmenter: nn.Module
    classifier: nn.Module
    class_names: np.ndarray
    norm_mean: np.ndarray
    norm_std: np.ndarray


def load_engine(
    models_dir: Path,
    config: InferenceConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> InferenceEngine:
    """Load segmenter + classifier from a `project/models` directory layout."""
    config = config or InferenceConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_path = resolve_segmenter_checkpoint(models_dir)
    bundle = resolve_classifier_bundle(models_dir)

    # Saved classifier config is the source of truth for runtime parameters.
    classifier_cfg: dict[str, Any] = {}
    if bundle.get("config_path") and Path(bundle["config_path"]).is_file():
        classifier_cfg = json.loads(Path(bundle["config_path"]).read_text(encoding="utf-8"))

    card_img_size = int(classifier_cfg.get("image_size", config.card_img_size))
    card_mask_threshold = float(classifier_cfg.get("mask_threshold", config.card_mask_threshold))
    norm_mean = np.array(classifier_cfg.get("normalization_mean", IMAGENET_MEAN.tolist()), dtype=np.float32)
    norm_std = np.array(classifier_cfg.get("normalization_std", IMAGENET_STD.tolist()), dtype=np.float32)

    config = InferenceConfig(
        segmenter_img_size=config.segmenter_img_size,
        segmenter_threshold=config.segmenter_threshold,
        segmenter_min_component_area=config.segmenter_min_component_area,
        instance_mask_growth_px=config.instance_mask_growth_px,
        card_img_size=card_img_size,
        card_mask_threshold=card_mask_threshold,
    )

    segmenter = SceneUNetSmall().to(device)
    segmenter.load_state_dict(torch.load(str(seg_path), map_location=device))
    segmenter.eval()

    classes = np.load(bundle["classes_path"], allow_pickle=True)
    checkpoint = torch.load(bundle["model_path"], map_location=device)
    architecture = str(classifier_cfg.get("architecture", "resnet18_small"))
    stem_width = int(classifier_cfg.get("classifier_stem_width", 60))
    dropout = float(classifier_cfg.get("classifier_dropout", 0.20))
    if isinstance(checkpoint, dict):
        architecture = str(checkpoint.get("architecture", architecture))
        stem_width = int(checkpoint.get("stem_width", stem_width))
        dropout = float(checkpoint.get("dropout", dropout))

    classifier, architecture = build_card_classifier(
        n_classes=len(classes),
        input_channels=int(classifier_cfg.get("input_channels", 4)),
        dropout=dropout,
        architecture=architecture,
        stem_width=stem_width,
    )
    classifier = classifier.to(device)

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    classifier.load_state_dict(state_dict)
    classifier.eval()

    if verbose:
        assert_param_cap(segmenter, "SceneUNetSmall (segmenter)")
        assert_param_cap(classifier, f"CardClassifier[{architecture}]")
        print(f"[engine] device={device} | classes={len(classes)} | "
              f"card_img_size={card_img_size} | seg_threshold={config.segmenter_threshold}")

    return InferenceEngine(
        config=config,
        device=device,
        segmenter=segmenter,
        classifier=classifier,
        class_names=classes,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


@torch.no_grad()
def _classify_crop(
    engine: InferenceEngine,
    crop_bgr: np.ndarray,
    crop_mask_u8: np.ndarray,
) -> tuple[str, float, float]:
    """Letterbox + masked-classify a single card crop."""
    crop_lb, mask_lb = letterbox_image_and_mask(crop_bgr, crop_mask_u8, size=engine.config.card_img_size)
    masked = compose_masked_card_image(crop_lb, mask_lb)
    x = card_input_to_tensor(masked, mask_lb, norm_mean=engine.norm_mean, norm_std=engine.norm_std)
    x = x.unsqueeze(0).to(engine.device)

    probs = torch.softmax(engine.classifier(x), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return str(engine.class_names[idx]), float(probs[idx].item()), float((mask_lb > 0).mean())


def predict_cards(engine: InferenceEngine, img_bgr: np.ndarray) -> list[CardPrediction]:
    """Run segmentation + per-card classification on a single image."""
    h, w = img_bgr.shape[:2]
    probability = segment_scene_probability(
        img_bgr, engine.segmenter, engine.device, target_size=engine.config.segmenter_img_size,
    )
    boxes, _mask, instance_masks = boxes_from_probability(
        probability,
        threshold=engine.config.segmenter_threshold,
        min_component_area=engine.config.segmenter_min_component_area,
        instance_mask_growth_px=engine.config.instance_mask_growth_px,
        return_instance_masks=True,
    )

    predictions: list[CardPrediction] = []
    for box, instance_mask in zip(boxes, instance_masks):
        crop_bgr = crop_with_margin(img_bgr, box)
        crop_mask = crop_with_margin((instance_mask * 255).astype(np.uint8), box)
        if crop_bgr.size == 0 or crop_mask.size == 0:
            continue
        label, confidence, mask_coverage = _classify_crop(engine, crop_bgr, crop_mask)
        predictions.append(CardPrediction(
            box=tuple(int(v) for v in box),
            instance_mask=instance_mask.astype(np.uint8),
            label=label,
            confidence=confidence,
            region=assign_region(box, w, h),
            mask_coverage=mask_coverage,
        ))
    return predictions


def _pick_center_card(center_preds: list[CardPrediction], image_w: int, image_h: int) -> str:
    """If several boxes land on the center region, keep the one closest to the
    image center. The challenge always shows exactly one center card."""
    if not center_preds:
        return "EMPTY"
    cx, cy = image_w / 2, image_h / 2

    def dist(pred: CardPrediction) -> float:
        x0, y0, x1, y1 = pred.box
        return float(np.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy))

    return min(center_preds, key=dist).label


def predict_game_state(
    engine: InferenceEngine,
    img_bgr: np.ndarray,
    image_id: str,
) -> GameState:
    """Full pipeline: image -> GameState ready for CSV serialization."""
    h, w = img_bgr.shape[:2]
    predictions = predict_cards(engine, img_bgr)

    region_labels: dict[str, list[str]] = {"center": [], "p1": [], "p2": [], "p3": [], "p4": []}
    region_preds: dict[str, list[CardPrediction]] = {k: [] for k in region_labels}
    for pred in predictions:
        region_preds.setdefault(pred.region, []).append(pred)
        if pred.region != "center":
            region_labels[pred.region].append(pred.label)

    active = detect_active_player(img_bgr, predictions)

    return GameState(
        image_id=image_id,
        center_card=_pick_center_card(region_preds["center"], w, h),
        active_player=active,
        player_1_cards=format_cards(region_labels["p1"]),
        player_2_cards=format_cards(region_labels["p2"]),
        player_3_cards=format_cards(region_labels["p3"]),
        player_4_cards=format_cards(region_labels["p4"]),
        cards=predictions,
    )


def predict_from_path(engine: InferenceEngine, image_path: Path) -> GameState:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return predict_game_state(engine, img_bgr, image_id=image_path.stem)
