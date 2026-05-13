"""Shared image-pipeline primitives: normalization, letterboxing, masking,
geometry, and scene-segmenter inference.

These helpers are used identically by training and inference paths. Keeping
them in one module prevents the train/test drift that previously existed
(e.g. PIL+TF.normalize on one side, cv2+manual normalization on the other).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_workspace_root(markers: Iterable[str] = ("project/training_data", "data")) -> Path:
    """Walk parents until all `markers` exist as siblings."""
    candidate = Path.cwd().resolve()
    markers = tuple(markers)
    while not all((candidate / marker).exists() for marker in markers):
        if candidate.parent == candidate:
            raise FileNotFoundError(
                f"Could not locate workspace root containing {markers}."
            )
        candidate = candidate.parent
    return candidate


def crop_with_margin(arr: np.ndarray, bbox: tuple[int, int, int, int], margin_fraction: float = 0.08) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    h, w = arr.shape[:2]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    margin = int(round(margin_fraction * max(bw, bh)))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    return arr[y0:y1, x0:x1]


def is_reasonable_scene_bbox(bbox: tuple[int, int, int, int], min_side: int = 28, max_aspect: float = 4.5) -> bool:
    x0, y0, x1, y1 = map(int, bbox)
    bw = x1 - x0
    bh = y1 - y0
    if bw < min_side or bh < min_side:
        return False
    aspect = max(bw / max(1, bh), bh / max(1, bw))
    return aspect <= max_aspect


def letterbox_image_and_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    size: int,
    fill_image: int = 128,
    fill_mask: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image+mask preserving aspect ratio onto a square canvas."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    img_canvas = np.full((size, size, 3), fill_image, dtype=np.uint8)
    mask_canvas = np.full((size, size), fill_mask, dtype=np.uint8)

    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    img_canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized
    mask_canvas[y0:y0 + new_h, x0:x0 + new_w] = mask_resized
    return img_canvas, mask_canvas


def compose_masked_card_image(img_bgr: np.ndarray, mask_u8: np.ndarray, bg_fill: int = 128) -> np.ndarray:
    out = np.full_like(img_bgr, bg_fill)
    fg = mask_u8 > 0
    out[fg] = img_bgr[fg]
    return out


def card_input_to_tensor(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    norm_mean: np.ndarray = IMAGENET_MEAN,
    norm_std: np.ndarray = IMAGENET_STD,
) -> torch.Tensor:
    """Pack a masked BGR crop and its mask into a 4-channel CHW tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_chw = (img_chw - norm_mean[:, None, None]) / norm_std[:, None, None]
    mask_ch = (mask_u8.astype(np.float32) / 255.0)[None, :, :]
    stacked = np.concatenate([img_chw, mask_ch], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))


def letterbox_for_segmenter(
    img_bgr: np.ndarray,
    target_size: int,
    fill: int = 255,
) -> tuple[Image.Image, dict[str, int]]:
    """Letterbox scene image using the same PIL path as segmenter training."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    w, h = img_pil.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    canvas = TF.pad(resized, (left, top, right, bottom), fill=fill)
    meta = {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "left": left,
        "top": top,
    }
    return canvas, meta


def unletterbox_probability(prob_lb: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    left, top = int(meta["left"]), int(meta["top"])
    new_w, new_h = int(meta["new_w"]), int(meta["new_h"])
    orig_w, orig_h = int(meta["orig_w"]), int(meta["orig_h"])
    core = prob_lb[top:top + new_h, left:left + new_w]
    return cv2.resize(core, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def segment_scene_probability(
    img_bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
    target_size: int = 256,
) -> np.ndarray:
    """Run the scene segmenter on a full-size image and return a HxW prob map."""
    img_lb, meta = letterbox_for_segmenter(img_bgr, target_size=target_size)
    x = TF.to_tensor(img_lb)
    x = TF.normalize(x, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    x = x.unsqueeze(0).to(device)
    prob_lb = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return unletterbox_probability(prob_lb, meta)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return inter / max(1, area_a + area_b - inter)


def nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(box_iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


def assign_region(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    center_cx: tuple[float, float] = (0.36, 0.64),
    center_cy: tuple[float, float] = (0.30, 0.70),
) -> str:
    """Geometry-only region assignment per AGENT.md R6.

    Player layout: p1=bottom, p2=right, p3=top, p4=left. A box whose centroid
    falls inside the central rectangle is assigned to "center"; otherwise the
    nearest player edge wins (tie-breaker biased toward axial distance).
    """
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2) / image_width
    cy = ((y0 + y1) / 2) / image_height

    if center_cx[0] <= cx <= center_cx[1] and center_cy[0] <= cy <= center_cy[1]:
        return "center"

    distances = {
        "p1": abs(1.00 - cy) + 0.25 * abs(cx - 0.50),  # bottom
        "p2": abs(1.00 - cx) + 0.25 * abs(cy - 0.50),  # right
        "p3": abs(0.00 - cy) + 0.25 * abs(cx - 0.50),  # top
        "p4": abs(0.00 - cx) + 0.25 * abs(cy - 0.50),  # left
    }
    return min(distances, key=distances.get)


def format_cards(cards: list[str]) -> str:
    return "EMPTY" if len(cards) == 0 else ";".join(cards)


def split_touching_component(
    component_mask: np.ndarray,
    min_area: int,
    max_instances: int = 8,
) -> list[np.ndarray]:
    """Try erosion-based seeds to split a single connected blob into multiple
    card-shaped masks (handles touching/overlapping cards in scenes)."""
    component = (component_mask > 0).astype(np.uint8)
    component_area = int(np.count_nonzero(component))
    if component_area < 2 * min_area:
        return [component]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_min_area = max(25, min_area // 5)

    for erode_iters in (1, 2, 3, 4):
        eroded = cv2.erode(component, kernel, iterations=erode_iters)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if n_labels <= 2:
            continue

        candidate_parts: list[np.ndarray] = []
        for label_idx in range(1, n_labels):
            seed_area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if seed_area < seed_min_area:
                continue
            seed = (labels == label_idx).astype(np.uint8)
            grown = cv2.dilate(seed, kernel, iterations=erode_iters)
            grown = ((grown > 0) & (component > 0)).astype(np.uint8)
            if int(np.count_nonzero(grown)) >= min_area:
                candidate_parts.append(grown)

        if len(candidate_parts) < 2:
            continue

        candidate_parts.sort(key=lambda part: int(np.count_nonzero(part)), reverse=True)
        accepted_parts: list[np.ndarray] = []
        occupied = np.zeros_like(component, dtype=np.uint8)

        for part in candidate_parts:
            unique = ((part > 0) & (occupied == 0)).astype(np.uint8)
            if int(np.count_nonzero(unique)) >= min_area:
                accepted_parts.append(unique)
                occupied[unique > 0] = 1
            if len(accepted_parts) >= max_instances:
                break

        if len(accepted_parts) >= 2:
            return accepted_parts

    return [component]


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only connected components whose area is at least `min_area` pixels."""
    min_area = int(min_area)
    if min_area <= 1:
        return (mask > 0).astype(np.uint8)

    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label_idx] = 1
    return cleaned


def boxes_from_probability(
    probability: np.ndarray,
    threshold: float = 0.50,
    max_components: int = 40,
    min_aspect: float = 0.12,
    max_aspect: float = 5.0,
    min_component_area: int | None = None,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    h, w = probability.shape[:2]
    mask = (probability > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    auto_min_area = max(400, int(0.00035 * h * w))
    min_area = auto_min_area if min_component_area is None else max(1, int(min_component_area))
    mask = remove_small_components(mask, min_area)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_split_area = max(220, int(0.55 * min_area))
    boxes: list[tuple[int, int, int, int]] = []

    for label_idx in range(1, n_labels):
        component_area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if component_area < min_area:
            continue

        component_mask = (labels == label_idx).astype(np.uint8)
        for part_mask in split_touching_component(component_mask, min_split_area):
            ys, xs = np.where(part_mask > 0)
            if ys.size == 0:
                continue

            x0, x1 = int(xs.min()), int(xs.max() + 1)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            bw = x1 - x0
            bh = y1 - y0
            area = int(np.count_nonzero(part_mask))
            aspect = bw / max(1, bh)

            if area >= min_split_area and min_aspect <= aspect <= max_aspect:
                boxes.append((x0, y0, x1, y1))

    return nms_boxes(boxes)[:max_components], mask
