"""Scene-level card prediction pipeline combining segmentation and classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uno_vision.classification.predict import CardClassifier, load_card_classifier
from uno_vision.segmentation_card.inference import load_segmenter, segment_image


@dataclass(frozen=True)
class CardRegionPrediction:
    """Predicted label and geometry for one detected card region."""

    index: int
    box: tuple[int, int, int, int]
    area: int
    label: str
    color: str
    rank: str


def split_probability_mask(
    global_prob: np.ndarray,
    high_prob_thresh: float = 0.65,
    card_min_dist: int = 60,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]]]:
    """Build conservative connected regions from a foreground probability mask."""

    _ = card_min_dist
    high_mask = (global_prob >= high_prob_thresh).astype(np.uint8)
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high_mask, connectivity=8)

    img_h, img_w = high_mask.shape
    img_area = img_h * img_w
    min_area = max(1500, int(0.0012 * img_area))
    regions: list[tuple[int, int, int, int, int, int]] = []
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        aspect_ratio = w / max(1, h)
        fill_ratio = area / max(1, w * h)
        if not (0.2 <= aspect_ratio <= 3.5) or fill_ratio < 0.2:
            continue
        regions.append((x, y, w, h, area, label))
    regions.sort(key=lambda region: region[4], reverse=True)

    clean_labels = np.zeros_like(labels, dtype=np.int32)
    for index, (_, _, _, _, _, label) in enumerate(regions, start=1):
        clean_labels[labels == label] = index
    return clean_labels, regions


def regions_from_proposal_boxes(
    global_prob: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    prob_thresh: float = 0.35,
    min_mask_fraction: float = 0.05,
) -> list[tuple[int, int, int, int, int, int]]:
    """Use classical proposal boxes as card instances when the segmenter agrees."""

    regions: list[tuple[int, int, int, int, int, int]] = []
    for label, (x, y, w, h) in enumerate(boxes, start=1):
        prob_crop = global_prob[y:y + h, x:x + w]
        mask_area = int((prob_crop >= prob_thresh).sum())
        if mask_area < max(400, int(min_mask_fraction * w * h)):
            continue
        regions.append((x, y, w, h, mask_area, label))
    regions.sort(key=lambda region: region[4], reverse=True)
    return regions


def classify_regions(
    img_bgr: np.ndarray,
    regions: list[tuple[int, int, int, int, int, int]],
    classifier: CardClassifier,
) -> list[CardRegionPrediction]:
    """Crop each detected card region and classify its UNO color and rank."""

    predictions: list[CardRegionPrediction] = []
    for index, (x, y, w, h, area, _) in enumerate(regions):
        margin = int(0.08 * max(w, h))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img_bgr.shape[1], x + w + margin)
        y1 = min(img_bgr.shape[0], y + h + margin)
        crop = img_bgr[y0:y1, x0:x1]
        pred = classifier.predict_bgr(crop)
        predictions.append(CardRegionPrediction(index, (x0, y0, x1, y1), area, pred.label, pred.color, pred.rank))
    return predictions


def predict_cards_in_image(
    image_path: str | Path,
    max_components: int = 24,
    high_prob_thresh: float = 0.65,
    card_min_dist: int = 60,
    use_proposal_boxes: bool = True,
    proposal_prob_thresh: float = 0.35,
) -> tuple[list[CardRegionPrediction], np.ndarray, np.ndarray]:
    """Run segmentation, region splitting, and card classification for one image."""

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    model, device, _ = load_segmenter()
    global_prob, boxes, _, _ = segment_image(img_bgr, model, device, max_components=max_components)
    if use_proposal_boxes:
        regions = regions_from_proposal_boxes(global_prob, boxes, prob_thresh=proposal_prob_thresh)
        ws_labels = np.zeros(global_prob.shape, dtype=np.int32)
        for index, (x, y, w, h, _, _) in enumerate(regions, start=1):
            ws_labels[y:y + h, x:x + w] = index
    else:
        ws_labels, regions = split_probability_mask(global_prob, high_prob_thresh, card_min_dist)
    classifier = load_card_classifier()
    return classify_regions(img_bgr, regions, classifier), global_prob, ws_labels
