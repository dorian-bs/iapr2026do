"""Extraction utilities for turning reference images into labeled card crops and masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from uno_vision.paths import REFERENCE_CARDS_DIR, REFERENCE_IMAGES_DIR


@dataclass(frozen=True)
class ReferenceExtractionResult:
    """Paths produced when one reference image is decomposed into card assets."""

    image_name: str
    output_dir: Path
    crops: list[Path]
    masks: list[Path]
    components: list[Path]


def _odd(value: float) -> int:
    """Return value rounded up to the nearest odd integer >= 3."""
    v = max(3, int(value))
    return v if v % 2 == 1 else v + 1


def _reference_binary_mask(
    gray: np.ndarray,
    canny_blur_size: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    blob_close_size: int = 40,
    blur_after: int = 33,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a coarse foreground mask from Canny edges in a reference image."""

    scale = max(gray.shape) / 4000.0
    k_blur = _odd(canny_blur_size * scale)
    blurred = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    k_close = max(3, int(blob_close_size * scale))

    # Closing turns broken card outlines into connected blobs for component labeling.
    binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    binary = cv2.medianBlur(binary, _odd(blur_after * scale))
    return edges, binary


def extract_reference_card_assets(
    image_name: str,
    num_components: int,
    image_dir: Path = REFERENCE_IMAGES_DIR,
    output_root: Path = REFERENCE_CARDS_DIR,
    # mask hyperparameters (forwarded to _reference_binary_mask)
    canny_blur_size: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    blob_close_size: int = 40,
    blur_after: int = 33,
    # extraction hyperparameters
    min_area_abs: int = 2500,
    min_area_frac: float = 0.002,
    close_size: int = 21,
    open_size: int = 7,
    min_ar: float = 0.3,
    max_ar: float = 2.0,
) -> ReferenceExtractionResult:
    """Extract card crops, component masks, and filled masks from one reference image."""

    image_path = image_dir / f"{image_name}.jpg"
    gray = np.array(Image.open(image_path).convert("L"))
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise FileNotFoundError(f"Cannot read reference image: {image_path}")

    out_dir = output_root / image_name
    components_dir = out_dir / "components"
    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    for directory in (components_dir, masks_dir, crops_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for f in directory.glob("*.jpg"):
            f.unlink()

    edges, binary = _reference_binary_mask(
        gray, canny_blur_size, canny_low, canny_high, blob_close_size, blur_after
    )
    cv2.imwrite(str(out_dir / "canny_edges.jpg"), edges)

    scale = max(gray.shape) / 4000.0
    img_area = gray.shape[0] * gray.shape[1]
    min_area = max(min_area_abs * scale ** 2, min_area_frac * img_area)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary, [contour], -1, 0, thickness=cv2.FILLED)

    # Closing fills card interiors; opening removes residual speckles before labeling.
    masks = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masks, connectivity=8)
    if n_labels <= 1:
        return ReferenceExtractionResult(image_name, out_dir, [], [], [])

    valid: list[tuple[int, int]] = []
    for i in range(n_labels - 1):
        label_idx = i + 1
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        ar = w / max(h, 1)
        if min_ar <= ar <= max_ar:
            valid.append((int(stats[label_idx, cv2.CC_STAT_AREA]), label_idx))
    valid.sort(reverse=True)
    selected_idx = [lbl for _, lbl in valid[: min(num_components, len(valid))]]

    crops: list[Path] = []
    closed_masks: list[Path] = []
    components: list[Path] = []
    for plot_i, label_idx in enumerate(selected_idx):
        component_mask = np.zeros_like(masks)
        component_mask[labels == label_idx] = 255
        component_path = components_dir / f"component_{plot_i}.jpg"
        cv2.imwrite(str(component_path), component_mask)
        components.append(component_path)

        contours_comp, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_comp:
            # Convex hulls produce filled masks that align with rectangular card crops.
            hull = cv2.convexHull(np.vstack(contours_comp))
            closed_mask = np.zeros_like(component_mask)
            cv2.fillPoly(closed_mask, [hull], 255)
        else:
            closed_mask = component_mask.copy()
        closed_path = masks_dir / f"closed_component_{plot_i}.jpg"
        cv2.imwrite(str(closed_path), closed_mask)
        closed_masks.append(closed_path)

        # Crops and masks share the same closed-mask bounding box for later alignment.
        x, y, w, h = cv2.boundingRect(closed_mask)
        crop_path = crops_dir / f"crop_{plot_i}.jpg"
        cv2.imwrite(str(crop_path), img_color[y : y + h, x : x + w])
        crops.append(crop_path)

    return ReferenceExtractionResult(image_name, out_dir, crops, closed_masks, components)
