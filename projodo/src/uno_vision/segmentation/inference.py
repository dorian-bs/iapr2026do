from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import torch
import torchvision.transforms.functional as TF

from uno_vision.image_ops import letterbox_pil_with_meta, unletterbox_mask
from uno_vision.paths import SEGMENTER_MODELS_DIR
from uno_vision.segmentation.model import UNetSmall


SEGMENTER_IMAGE_SIZE = 256


def load_segmenter(model_path: Path | None = None, device: torch.device | None = None) -> tuple[UNetSmall, torch.device, Path]:
    candidates = [
        model_path,
        SEGMENTER_MODELS_DIR / "segmenter_unet_small.pth",
        SEGMENTER_MODELS_DIR / "segment_unet_small.pth",
    ]
    selected = next((path for path in candidates if path is not None and Path(path).is_file()), None)
    if selected is None:
        raise FileNotFoundError("No segmenter model found in artifacts/models/segmenter.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(selected, map_location=device))
    model.eval()
    return model, device, Path(selected)


def detect_card_boxes_reference_style(img_bgr: np.ndarray, max_components: int = 24):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = h * w
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        4,
    )
    adaptive = cv2.medianBlur(adaptive, 3)
    binary = adaptive.copy()
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((17, 17), np.uint8))
    binary = skeletonize(binary > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((17, 17), np.uint8))
    binary = cv2.medianBlur(binary, 33)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(2500, int(0.002 * img_area))
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary, [contour], -1, 0, thickness=cv2.FILLED)

    masks = binary.copy()
    masks = cv2.morphologyEx(masks, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(masks, connectivity=8)
    if n_labels <= 1:
        return [(0, 0, w, h)], binary, masks

    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_idx = np.argsort(areas)[::-1][: min(max_components, len(areas))] + 1
    boxes = []
    for label_idx in selected_idx:
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        aspect_ratio = box_w / max(1, box_h)
        area = box_w * box_h
        if 0.2 <= aspect_ratio <= 2.5 and area >= min_area:
            boxes.append((x, y, box_w, box_h))
    return boxes or [(0, 0, w, h)], binary, masks


def segment_image(
    img_bgr: np.ndarray,
    model: UNetSmall,
    device: torch.device,
    max_components: int = 24,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    img_h, img_w = img_bgr.shape[:2]
    boxes, binary_debug, masks_debug = detect_card_boxes_reference_style(img_bgr, max_components=max_components)
    global_prob = np.zeros((img_h, img_w), dtype=np.float32)

    for x, y, box_w, box_h in boxes:
        margin = int(0.05 * max(box_w, box_h))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img_w, x + box_w + margin)
        y1 = min(img_h, y + box_h + margin)
        crop_bgr = img_bgr[y0:y1, x0:x1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_lb, meta = letterbox_pil_with_meta(
            crop_pil,
            SEGMENTER_IMAGE_SIZE,
            fill=255,
            interpolation=Image.BILINEAR,
        )
        x_t = TF.to_tensor(crop_lb)
        x_t = TF.normalize(x_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x_t = x_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_t)
            prob_lb = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob_crop = unletterbox_mask(prob_lb, meta)
        global_prob[y0:y1, x0:x1] = np.maximum(global_prob[y0:y1, x0:x1], prob_crop)
    return global_prob, boxes, binary_debug, masks_debug


def segment_image_path(image_path: str | Path, model_path: Path | None = None, max_components: int = 24):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    model, device, selected_model_path = load_segmenter(model_path=model_path)
    global_prob, boxes, binary_debug, masks_debug = segment_image(img_bgr, model, device, max_components=max_components)
    return img_bgr, global_prob, boxes, binary_debug, masks_debug, selected_model_path