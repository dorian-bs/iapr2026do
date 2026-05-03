from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from uno_vision.paths import REFERENCE_CARDS_DIR, REFERENCE_IMAGES_DIR


@dataclass(frozen=True)
class ReferenceExtractionResult:
    image_name: str
    output_dir: Path
    crops: list[Path]
    masks: list[Path]
    components: list[Path]


def _reference_binary_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=4,
    )
    adaptive = cv2.medianBlur(adaptive, 3)
    binary = cv2.bitwise_not(adaptive)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((17, 17), np.uint8))
    binary = skeletonize(binary > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((17, 17), np.uint8))
    binary = cv2.medianBlur(binary, 33)
    return adaptive, binary


def extract_reference_card_assets(
    image_name: str,
    num_components: int,
    image_dir: Path = REFERENCE_IMAGES_DIR,
    output_root: Path = REFERENCE_CARDS_DIR,
) -> ReferenceExtractionResult:
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

    adaptive, binary = _reference_binary_mask(gray)
    cv2.imwrite(str(out_dir / "adaptive_thresholding.jpg"), adaptive)

    img_area = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < max(2500, int(0.002 * img_area)):
            cv2.drawContours(binary, [contour], -1, 0, thickness=cv2.FILLED)

    masks = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masks, connectivity=8)
    if n_labels <= 1:
        return ReferenceExtractionResult(image_name, out_dir, [], [], [])

    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_idx = np.argsort(areas)[::-1][: min(num_components, len(areas))] + 1
    crops: list[Path] = []
    closed_masks: list[Path] = []
    components: list[Path] = []
    for plot_i, label_idx in enumerate(selected_idx):
        component_mask = np.zeros_like(masks)
        component_mask[labels == label_idx] = 255
        component_path = components_dir / f"component_{plot_i}.jpg"
        cv2.imwrite(str(component_path), component_mask)
        components.append(component_path)

        closed_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, np.ones((133, 133), np.uint8))
        closed_path = masks_dir / f"closed_component_{plot_i}.jpg"
        cv2.imwrite(str(closed_path), closed_mask)
        closed_masks.append(closed_path)

        x, y, w, h = cv2.boundingRect(closed_mask)
        crop_path = crops_dir / f"crop_{plot_i}.jpg"
        cv2.imwrite(str(crop_path), img_color[y:y + h, x:x + w])
        crops.append(crop_path)

    return ReferenceExtractionResult(image_name, out_dir, crops, closed_masks, components)