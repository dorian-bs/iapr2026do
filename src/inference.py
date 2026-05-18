"""Final UNO game-state inference and submission writing.

This is the only module needed by `main.py`: it loads the trained checkpoints,
runs card segmentation/classification, assembles the Kaggle row, and validates
the CSV schema before writing the submission file.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}

CSV_COLUMNS = (
    "image_id",
    "center_card",
    "active_player",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
)
CARD_COLORS = {"r", "g", "b", "y"}
CARD_VALUES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw_2"}
SPECIAL_CARDS = {"wild", "draw_4"}
ACTIVE_PLAYER_VALUES = {"p1", "p2", "p3", "p4", "EMPTY"}
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

LAYOUT_RATIOS = {
    "x_left": 0.245, "x_mid_left": 0.3125, "x_mid_right": 0.675, "x_right": 0.775,
    "y_top_left": 0.182, "y_top": 0.309, "y_bottom": 0.691, "y_bottom_right": 0.800
}


@dataclass
class InferenceConfig:
    segmenter_img_size: int = 256
    segmenter_threshold: float = 0.50
    segmenter_min_component_area: int | None = 4000
    instance_mask_growth_px: int = 5
    card_img_size: int = 160
    card_mask_threshold: float = 0.50


@dataclass
class CardPrediction:
    box: tuple[int, int, int, int]
    instance_mask: np.ndarray
    label: str
    confidence: float
    region: str
    mask_coverage: float


@dataclass
class GameState:
    image_id: str
    center_card: str
    active_player: str
    player_1_cards: str
    player_2_cards: str
    player_3_cards: str
    player_4_cards: str
    cards: list[CardPrediction] = field(default_factory=list)

    def as_submission_row(self) -> dict[str, str]:
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
    config: InferenceConfig
    device: torch.device
    segmenter: nn.Module
    classifier: nn.Module
    class_names: np.ndarray
    norm_mean: np.ndarray
    norm_std: np.ndarray


class SceneDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SceneUNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = SceneDoubleConv(3, 32)
        self.enc2 = SceneDoubleConv(32, 64)
        self.enc3 = SceneDoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = SceneDoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = SceneDoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = SceneDoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = SceneDoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class CardResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CardResNet18SmallClassifier(nn.Module):
    def __init__(self, n_classes: int, input_channels: int = 4, dropout: float = 0.20, stem_width: int = 60):
        super().__init__()
        c1, c2, c3, c4 = stem_width, stem_width * 2, stem_width * 4, stem_width * 8
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(CardResidualBlock(c1, c1), CardResidualBlock(c1, c1))
        self.layer2 = nn.Sequential(CardResidualBlock(c1, c2, stride=2), CardResidualBlock(c2, c2))
        self.layer3 = nn.Sequential(CardResidualBlock(c2, c3, stride=2), CardResidualBlock(c3, c3))
        self.layer4 = nn.Sequential(CardResidualBlock(c3, c4, stride=2), CardResidualBlock(c4, c4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c4, n_classes))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.head(torch.flatten(x, 1))


def assert_param_cap(model: nn.Module, name: str, cap: int = 12_000_000) -> int:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params <= cap, f"{name} has {n_params:,} params, exceeds {cap:,} cap"
    print(f"[compliance] {name}: {n_params:,} trainable params")
    return n_params


def _single_matching_file(directory: Path, label: str, suffixes: set[str], required: bool = True) -> Path | None:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if not files:
        if required:
            raise FileNotFoundError(f"No {label} found in {directory}.")
        return None
    if len(files) > 1:
        raise RuntimeError(f"Expected exactly one {label} in {directory}, found {len(files)}.")
    return files[0]


def resolve_classifier_bundle(models_dir: Path) -> dict[str, Path | None]:
    used_root = Path(models_dir) / "card_classifier_cnn" / "used"
    bundle_dirs = sorted(path for path in used_root.iterdir() if path.is_dir())
    if len(bundle_dirs) != 1:
        raise RuntimeError(f"Expected exactly one active classifier bundle in {used_root}, found {len(bundle_dirs)}.")

    bundle_dir = bundle_dirs[0]
    return {
        "bundle_dir": bundle_dir,
        "model_path": _single_matching_file(bundle_dir, "classifier checkpoint", CHECKPOINT_SUFFIXES),
        "classes_path": _single_matching_file(bundle_dir, "classifier classes file", {".npy"}),
        "config_path": _single_matching_file(bundle_dir, "classifier config file", {".json"}, required=False),
    }


def load_engine(
    models_dir: Path,
    config: InferenceConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> InferenceEngine:
    config = config or InferenceConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir)

    seg_path = _single_matching_file(models_dir / "segmenter" / "used", "segmenter checkpoint", CHECKPOINT_SUFFIXES)
    bundle = resolve_classifier_bundle(models_dir)
    classifier_cfg: dict[str, Any] = {}
    if bundle["config_path"] is not None:
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
    architecture = str(checkpoint.get("architecture", classifier_cfg.get("architecture", "resnet18_small"))) if isinstance(checkpoint, dict) else str(classifier_cfg.get("architecture", "resnet18_small"))
    if architecture != "resnet18_small":
        raise ValueError(f"Unsupported submitted classifier architecture: {architecture!r}")
    stem_width = int(checkpoint.get("stem_width", classifier_cfg.get("classifier_stem_width", 60))) if isinstance(checkpoint, dict) else int(classifier_cfg.get("classifier_stem_width", 60))
    dropout = float(checkpoint.get("dropout", classifier_cfg.get("classifier_dropout", 0.20))) if isinstance(checkpoint, dict) else float(classifier_cfg.get("classifier_dropout", 0.20))

    classifier = CardResNet18SmallClassifier(
        n_classes=len(classes),
        input_channels=int(classifier_cfg.get("input_channels", 4)),
        dropout=dropout,
        stem_width=stem_width,
    ).to(device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    classifier.load_state_dict(state_dict)
    classifier.eval()

    if verbose:
        assert_param_cap(segmenter, "SceneUNetSmall (segmenter)")
        assert_param_cap(classifier, "CardClassifier[resnet18_small]")
        print(f"[engine] device={device} | classes={len(classes)} | card_img_size={card_img_size} | seg_threshold={config.segmenter_threshold}")

    return InferenceEngine(config, device, segmenter, classifier, classes, norm_mean, norm_std)


def crop_with_margin(arr: np.ndarray, bbox: tuple[int, int, int, int], margin_fraction: float = 0.08) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    image_height, image_width = arr.shape[:2]
    margin = int(round(margin_fraction * max(1, x1 - x0, y1 - y0)))
    return arr[max(0, y0 - margin):min(image_height, y1 + margin), max(0, x0 - margin):min(image_width, x1 + margin)]


def is_background_noisy(img_bgr: np.ndarray) -> bool:
    return bool(img_bgr.std() > 45)


def divide_background(width: int, height: int) -> Dict[str, list]:
    x_left, x_mid_left = int(width * LAYOUT_RATIOS["x_left"]), int(width * LAYOUT_RATIOS["x_mid_left"])
    x_mid_right, x_right = int(width * LAYOUT_RATIOS["x_mid_right"]), int(width * LAYOUT_RATIOS["x_right"])
    y_top_left, y_top = int(height * LAYOUT_RATIOS["y_top_left"]), int(height * LAYOUT_RATIOS["y_top"])
    y_bottom, y_bottom_right = int(height * LAYOUT_RATIOS["y_bottom"]), int(height * LAYOUT_RATIOS["y_bottom_right"])

    return {
        "p3": [(0, 0), (x_right, 0), (x_right, y_top), (x_left, y_top), (x_left, y_top_left), (0, y_top_left)],
        "p4": [(0, y_top_left), (x_left, y_top_left), (x_left, y_top), (x_mid_left, y_top), (x_mid_left, y_bottom), (x_left, y_bottom), (x_left, height), (0, height)],
        "p2": [(x_right, 0), (width, 0), (width, y_bottom_right), (x_right, y_bottom_right), (x_right, y_bottom), (x_mid_right, y_bottom), (x_mid_right, y_top), (x_right, y_top)],
        "p1": [(x_left, y_bottom), (x_right, y_bottom), (x_right, y_bottom_right), (width, y_bottom_right), (width, height), (x_left, height)]
    }


def find_yellow_token(
    img_bgr: np.ndarray, 
    hue_range=(22, 28), sat_min=100, val_min=100,
    dp=1.5, min_dist=50, param1=50, param2=30, min_radius=60, max_radius=100,
    yellow_threshold=0.7, saturation_threshold=0.2
) -> List[Tuple[int, int]]:
    """Scans the whole image and returns a list of (x, y) center points for yellow tokens."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow, upper_yellow = np.array([hue_range[0], sat_min, val_min]), np.array([hue_range[1], 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    circles = cv2.HoughCircles(
        cleaned, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
    )
    
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            circle_mask = np.zeros_like(cleaned)
            cv2.circle(circle_mask, center, radius, 255, -1)
            
            yellow_in_circle = cv2.bitwise_and(cleaned, circle_mask)
            yellow_ratio = np.sum(yellow_in_circle > 0) / np.sum(circle_mask > 0) if np.sum(circle_mask > 0) > 0 else 0
            
            avg_saturation = np.mean(hsv[:, :, 1][circle_mask > 0]) / 255.0 if np.any(circle_mask > 0) else 0
            
            if yellow_ratio >= yellow_threshold and avg_saturation >= saturation_threshold:
                centers.append(center)
                
    return centers


def find_black_token(img_bgr: np.ndarray) -> List[Tuple[int, int]]:
    """Scans the whole image and returns a list of (x, y) center points for black tokens."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((11, 11), np.uint8)
    cleaned = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 35000:
            continue
            
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if 4 <= len(approx) <= 8 and 0.55 < aspect_ratio < 1.65:
            # Calculate the center of the rectangle using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                
    return centers


def detect_active_player(img_bgr: np.ndarray) -> str:
    """
    Detects which player's turn it is based on token position.
    Returns: "p1", "p2", "p3", "p4", or "unknown"
    """
    h, w = img_bgr.shape[:2]
    
    # 1. Detect all valid markers in the entire image first
    if is_background_noisy(img_bgr):
        marker_centers = find_yellow_token(img_bgr)
    else:
        marker_centers = find_black_token(img_bgr)
        
    if not marker_centers:
        return "unknown"
        
    # 2. Map the found coordinates to a sector
    polygons = divide_background(w, h)
    
    for cx, cy in marker_centers:
        for sector_name, polygon in polygons.items():
            pts = np.array(polygon, dtype=np.int32)
            
            # pointPolygonTest returns >= 0 if the point is inside or exactly on the polygon edge
            if cv2.pointPolygonTest(pts, (cx, cy), measureDist=False) >= 0:
                return sector_name
                
    return "unknown"


@torch.no_grad()
def segment_scene_probability(img_bgr: np.ndarray, model: nn.Module, device: torch.device, target_size: int = 256) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    image_width, image_height = img_pil.size
    scale = min(target_size / image_width, target_size / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = img_pil.resize((resized_width, resized_height), Image.BILINEAR)
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2
    canvas = TF.pad(
        resized,
        (left, top, target_size - resized_width - left, target_size - resized_height - top),
        fill=255,
    )
    x = TF.normalize(TF.to_tensor(canvas), mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    prob_lb = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    core = prob_lb[top:top + resized_height, left:left + resized_width]
    return cv2.resize(core, (image_width, image_height), interpolation=cv2.INTER_LINEAR)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return inter / max(1, area_a + area_b - inter)


def boxes_from_probability(
    probability: np.ndarray,
    threshold: float = 0.50,
    max_components: int = 40,
    min_aspect: float = 0.12,
    max_aspect: float = 5.0,
    min_component_area: int | None = None,
    instance_mask_growth_px: int = 0,
    return_instance_masks: bool = False,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray] | tuple[list[tuple[int, int, int, int]], np.ndarray, list[np.ndarray]]:
    image_height, image_width = probability.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = (probability > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = max(400, int(0.00035 * image_height * image_width)) if min_component_area is None else max(1, int(min_component_area))
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_index in range(1, component_count):
        if int(stats[label_index, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label_index] = 1

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    candidate_boxes: list[tuple[int, int, int, int]] = []
    candidate_masks_local: list[tuple[np.ndarray, int, int]] = []
    growth_px = max(0, int(instance_mask_growth_px))
    min_split_area = max(220, int(0.55 * min_area))

    for label_index in range(1, component_count):
        component_area = int(stats[label_index, cv2.CC_STAT_AREA])
        if component_area < min_area:
            continue

        component_x = int(stats[label_index, cv2.CC_STAT_LEFT])
        component_y = int(stats[label_index, cv2.CC_STAT_TOP])
        component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        pad = max(4, growth_px + 1)
        x0 = max(0, component_x - pad)
        y0 = max(0, component_y - pad)
        x1 = min(image_width, component_x + component_width + pad)
        y1 = min(image_height, component_y + component_height + pad)

        part_mask = (labels[y0:y1, x0:x1] == label_index).astype(np.uint8)
        if growth_px > 0:
            part_mask = cv2.dilate(part_mask, kernel, iterations=growth_px)
        ys, xs = np.where(part_mask > 0)
        if ys.size == 0:
            continue

        box = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max() + 1), y0 + int(ys.max() + 1))
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        aspect = box_width / max(1, box_height)
        if int(np.count_nonzero(part_mask)) >= min_split_area and min_aspect <= aspect <= max_aspect:
            candidate_boxes.append(box)
            candidate_masks_local.append((part_mask.astype(np.uint8), x0, y0))

    kept_indices: list[int] = []
    for index in sorted(
        range(len(candidate_boxes)),
        key=lambda i: (candidate_boxes[i][2] - candidate_boxes[i][0]) * (candidate_boxes[i][3] - candidate_boxes[i][1]),
        reverse=True,
    ):
        if all(box_iou(candidate_boxes[index], candidate_boxes[kept]) < 0.35 for kept in kept_indices):
            kept_indices.append(index)
    kept_indices = kept_indices[:max_components]
    boxes = [candidate_boxes[index] for index in kept_indices]

    if return_instance_masks:
        instance_masks: list[np.ndarray] = []
        for index in kept_indices:
            local_mask, offset_x, offset_y = candidate_masks_local[index]
            full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            local_height, local_width = local_mask.shape[:2]
            full_mask[offset_y:offset_y + local_height, offset_x:offset_x + local_width] = local_mask
            instance_masks.append(full_mask)
        return boxes, cleaned, instance_masks
    return boxes, cleaned


def assign_region(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    center_cx: tuple[float, float] = (0.36, 0.64),
    center_cy: tuple[float, float] = (0.30, 0.70),
) -> str:
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2) / image_width
    cy = ((y0 + y1) / 2) / image_height
    if center_cx[0] <= cx <= center_cx[1] and center_cy[0] <= cy <= center_cy[1]:
        return "center"
    distances = {
        "p1": abs(1.00 - cy) + 0.25 * abs(cx - 0.50),
        "p2": abs(1.00 - cx) + 0.25 * abs(cy - 0.50),
        "p3": abs(0.00 - cy) + 0.25 * abs(cx - 0.50),
        "p4": abs(0.00 - cx) + 0.25 * abs(cy - 0.50),
    }
    return min(distances, key=distances.get)


@torch.no_grad()
def _classify_crop(engine: InferenceEngine, crop_bgr: np.ndarray, crop_mask_u8: np.ndarray) -> tuple[str, float, float]:
    crop_height, crop_width = crop_bgr.shape[:2]
    scale = engine.config.card_img_size / max(crop_height, crop_width)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    img_resized = cv2.resize(crop_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(crop_mask_u8, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    img_canvas = np.full((engine.config.card_img_size, engine.config.card_img_size, 3), 128, dtype=np.uint8)
    mask_canvas = np.zeros((engine.config.card_img_size, engine.config.card_img_size), dtype=np.uint8)
    y0 = (engine.config.card_img_size - resized_height) // 2
    x0 = (engine.config.card_img_size - resized_width) // 2
    img_canvas[y0:y0 + resized_height, x0:x0 + resized_width] = img_resized
    mask_canvas[y0:y0 + resized_height, x0:x0 + resized_width] = mask_resized
    img_canvas[mask_canvas <= 0] = 128

    img_rgb = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_chw = (img_chw - engine.norm_mean[:, None, None]) / engine.norm_std[:, None, None]
    mask_ch = (mask_canvas.astype(np.float32) / 255.0)[None, :, :]
    x = torch.from_numpy(np.concatenate([img_chw, mask_ch], axis=0).astype(np.float32)).unsqueeze(0).to(engine.device)

    probs = torch.softmax(engine.classifier(x), dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return str(engine.class_names[pred_idx]), float(probs[pred_idx].item()), float((mask_canvas > 0).mean())


def predict_cards(engine: InferenceEngine, img_bgr: np.ndarray) -> list[CardPrediction]:
    image_height, image_width = img_bgr.shape[:2]
    probability = segment_scene_probability(img_bgr, engine.segmenter, engine.device, engine.config.segmenter_img_size)
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
            box=tuple(int(value) for value in box),
            instance_mask=instance_mask.astype(np.uint8),
            label=label,
            confidence=confidence,
            region=assign_region(box, image_width, image_height),
            mask_coverage=mask_coverage,
        ))
    return predictions


def predict_game_state(engine: InferenceEngine, img_bgr: np.ndarray, image_id: str) -> GameState:
    image_height, image_width = img_bgr.shape[:2]
    predictions = predict_cards(engine, img_bgr)
    active_player = detect_active_player(img_bgr)
    region_labels: dict[str, list[str]] = {"p1": [], "p2": [], "p3": [], "p4": []}
    center_predictions: list[CardPrediction] = []
    for pred in predictions:
        if pred.region == "center":
            center_predictions.append(pred)
        elif pred.region in region_labels:
            region_labels[pred.region].append(pred.label)

    center_card = "EMPTY"
    if center_predictions:
        image_center_x, image_center_y = image_width / 2, image_height / 2
        center_card = min(
            center_predictions,
            key=lambda pred: np.hypot((pred.box[0] + pred.box[2]) / 2 - image_center_x, (pred.box[1] + pred.box[3]) / 2 - image_center_y),
        ).label

    return GameState(
        image_id=image_id,
        center_card=center_card,
        active_player="EMPTY" if active_player == "unknown" else active_player,
        player_1_cards="EMPTY" if not region_labels["p1"] else ";".join(region_labels["p1"]),
        player_2_cards="EMPTY" if not region_labels["p2"] else ";".join(region_labels["p2"]),
        player_3_cards="EMPTY" if not region_labels["p3"] else ";".join(region_labels["p3"]),
        player_4_cards="EMPTY" if not region_labels["p4"] else ";".join(region_labels["p4"]),
        cards=predictions,
    )


def predict_from_path(engine: InferenceEngine, image_path: Path) -> GameState:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return predict_game_state(engine, img_bgr, image_id=Path(image_path).stem)


def _validate_card_token(token: str) -> None:
    if token in SPECIAL_CARDS:
        return
    color, separator, value = token.partition("_")
    if not separator:
        raise ValueError(f"Card token '{token}' is not of the form <color>_<value>.")
    if color not in CARD_COLORS:
        raise ValueError(f"Card token '{token}': color '{color}' not in {CARD_COLORS}.")
    if value not in CARD_VALUES:
        raise ValueError(f"Card token '{token}': value '{value}' not in {CARD_VALUES}.")


def _validate_card_field(field_name: str, value: str) -> None:
    if value == "EMPTY":
        return
    if not value:
        raise ValueError(f"Field {field_name} is blank; use 'EMPTY' instead.")
    for token in value.split(";"):
        token = token.strip()
        if not token:
            raise ValueError(f"Field {field_name} has an empty card token.")
        _validate_card_token(token)


def validate_row(row: dict[str, str]) -> None:
    missing = [column for column in CSV_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Row is missing columns: {missing}")
    if not IMAGE_ID_RE.match(str(row["image_id"])):
        raise ValueError(f"Bad image_id: {row['image_id']!r}")
    if row["active_player"] not in ACTIVE_PLAYER_VALUES:
        raise ValueError(f"active_player must be in {ACTIVE_PLAYER_VALUES}, got {row['active_player']!r}")
    if row["center_card"] != "EMPTY":
        _validate_card_token(row["center_card"])
    for player_field in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
        _validate_card_field(player_field, row[player_field])


def write_submission(rows: Iterable[GameState | dict[str, str]], output_path: Path, validate: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for entry in rows:
            row = entry.as_submission_row() if isinstance(entry, GameState) else dict(entry)
            if validate:
                validate_row(row)
            writer.writerow(row)
    return output_path
