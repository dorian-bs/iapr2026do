from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from uno_vision.classification.predict import CardClassifier, load_card_classifier
from uno_vision.segmentation.inference import load_segmenter, segment_image


@dataclass(frozen=True)
class CardRegionPrediction:
    index: int
    box: tuple[int, int, int, int]
    area: int
    label: str
    color: str
    rank: str


def split_probability_mask(
    global_prob: np.ndarray,
    high_prob_thresh: float = 0.75,
    card_min_dist: int = 60,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int]]]:
    high_mask = (global_prob >= high_prob_thresh).astype(np.uint8)
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    distance = ndi.distance_transform_edt(high_mask)
    coords = peak_local_max(distance, min_distance=card_min_dist, labels=high_mask)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    if markers.max() == 0:
        markers = ndi.label(high_mask)[0]
    ws_labels = watershed(-distance, markers, mask=high_mask)

    img_h, img_w = high_mask.shape
    img_area = img_h * img_w
    min_area = max(1200, int(0.0015 * img_area))
    regions: list[tuple[int, int, int, int, int, int]] = []
    for label in range(1, ws_labels.max() + 1):
        comp = (ws_labels == label).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        x = int(xs.min())
        y = int(ys.min())
        w = int(xs.max()) - x + 1
        h = int(ys.max()) - y + 1
        regions.append((x, y, w, h, area, label))
    regions.sort(key=lambda region: region[4], reverse=True)
    return ws_labels, regions


def classify_regions(
    img_bgr: np.ndarray,
    regions: list[tuple[int, int, int, int, int, int]],
    classifier: CardClassifier,
) -> list[CardRegionPrediction]:
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
    high_prob_thresh: float = 0.75,
    card_min_dist: int = 60,
) -> tuple[list[CardRegionPrediction], np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    model, device, _ = load_segmenter()
    global_prob, _, _, _ = segment_image(img_bgr, model, device, max_components=max_components)
    ws_labels, regions = split_probability_mask(global_prob, high_prob_thresh, card_min_dist)
    classifier = load_card_classifier()
    return classify_regions(img_bgr, regions, classifier), global_prob, ws_labels