"""Training orchestrator for the scene segmenter (SceneUNetSmall).

Public surface mirrors the train_classifier_CNN pattern:
    SegmenterPipelineConfig
    initialize_segmenter_pipeline(config) -> state
    run_segmenter_training(state) -> state
    plot_training_curves(state)
    plot_segmentation_predictions(state)

The shared `SceneUNetSmall` is reused (R1: <=12M params asserted; R2: weights=None).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

from src.shared.card_models import SceneUNetSmall, assert_param_cap
from src.shared.card_pipeline import IMAGENET_MEAN, IMAGENET_STD, assign_region, find_workspace_root


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class SegmenterPipelineConfig:
    seed: int = 42
    image_size: int = 256
    val_split: float = 0.35
    max_scene_pairs: int | None = None
    epoch_max_train_samples: int | None = None
    cache_in_ram: bool = True
    num_workers: int = 4

    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_loss_bce_weight: float = 0.5
    train_loss_dice_weight: float = 0.5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    use_amp: bool = True
    use_torch_compile: bool = False
    early_stopping_patience: int | None = 8
    log_every_batches: int = 25

    warm_start: bool = False
    warm_start_filename: str = "segmenter_unet_small.pth"
    checkpoint_filename: str = "scene_segmenter_unet_small.pth"

    preview_count: int = 6
    eval_mask_threshold: float = 0.5
    eval_min_component_area: int | None = None

    # Checkpoint selection can optimize for overlap-only metrics or a
    # game-state-aware composite score that includes count quality.
    best_epoch_selection_metric: str = "composite"
    selection_weight_iou: float = 0.24
    selection_weight_dice: float = 0.24
    selection_weight_precision: float = 0.08
    selection_weight_recall: float = 0.08
    selection_weight_count_exact_rate: float = 0.10
    selection_weight_player_exact_rate: float = 0.10
    selection_weight_scene_player_exact_rate: float = 0.10
    selection_weight_count_mae: float = 0.03
    selection_weight_player_count_mae: float = 0.03


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def _letterbox_pil(img, target_size: int, fill: int, interpolation):
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), interpolation)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(img, padding, fill=fill)


def preprocess_scene_pair(img_path: Path, mask_path: Path, image_size: int = 256):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = _letterbox_pil(img, target_size=image_size, fill=255, interpolation=Image.BILINEAR)
    mask = _letterbox_pil(mask, target_size=image_size, fill=0, interpolation=Image.NEAREST)

    img_t = TF.to_tensor(img)
    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    mask_arr = (mask_arr > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

    return img_t, mask_t


def denormalize_image(img_t: torch.Tensor) -> np.ndarray:
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(img_np, 0, 1)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class SceneSegDataset(Dataset):
    def __init__(self, pairs, image_size: int = 256, augment: bool = False, cache_in_ram: bool = True):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache: list = []

        if self.cache_in_ram:
            print(f"  Pre-loading {len(self.pairs)} samples into RAM...", flush=True)
            for i, (img_path, mask_path) in enumerate(self.pairs, start=1):
                self.cache.append(preprocess_scene_pair(img_path, mask_path, image_size=self.image_size))
                if i % 200 == 0 or i == len(self.pairs):
                    print(f"  {i}/{len(self.pairs)} loaded", flush=True)
            print("  Done.", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache_in_ram:
            img_t, mask_t = self.cache[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            img_t, mask_t = preprocess_scene_pair(img_path, mask_path, image_size=self.image_size)

        if self.augment:
            if random.random() < 0.5:
                img_t = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t)
            if random.random() < 0.5:
                img_t = TF.vflip(img_t)
                mask_t = TF.vflip(mask_t)

        return img_t, mask_t


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + eps) / (probs.sum() + targets.sum() + eps)
    return 1.0 - dice


def overlap_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    union = tp + fp + fn

    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    return {
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "dice": float(dice.mean().item()),
    }


def iou_score_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> float:
    return overlap_metrics_from_logits(logits, targets, threshold=threshold, eps=eps)["iou"]


_PLAYER_KEYS = ("p1", "p2", "p3", "p4")


def _resolve_min_component_area(mask_height: int, mask_width: int, configured_value: int | None) -> int:
    if configured_value is not None:
        return max(1, int(configured_value))
    return max(10, int(round(0.00012 * float(mask_height * mask_width))))


def _connected_component_boxes(mask_2d: np.ndarray, min_component_area: int) -> list[tuple[int, int, int, int]]:
    binary = (mask_2d > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    return boxes


def _count_cards_by_player(
    mask_2d: np.ndarray,
    min_component_area: int | None = None,
) -> tuple[dict[str, int], int, int]:
    mask_height, mask_width = mask_2d.shape
    min_area = _resolve_min_component_area(mask_height, mask_width, min_component_area)
    boxes = _connected_component_boxes(mask_2d, min_area)

    player_counts = {player: 0 for player in _PLAYER_KEYS}
    center_count = 0
    for box in boxes:
        region = assign_region(box, image_width=mask_width, image_height=mask_height)
        if region in player_counts:
            player_counts[region] += 1
        elif region == "center":
            center_count += 1

    return player_counts, center_count, len(boxes)


def _count_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    min_component_area: int | None = None,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).to(torch.uint8).cpu().numpy()
    gts = (targets > 0.5).to(torch.uint8).cpu().numpy()

    batch_size = int(preds.shape[0])
    if batch_size == 0:
        return {
            "count_mae": 0.0,
            "count_exact_rate": 0.0,
            "player_count_mae": 0.0,
            "player_exact_rate": 0.0,
            "scene_player_exact_rate": 0.0,
            "pred_count_mean": 0.0,
            "gt_count_mean": 0.0,
            "player_count_mae_p1": 0.0,
            "player_count_mae_p2": 0.0,
            "player_count_mae_p3": 0.0,
            "player_count_mae_p4": 0.0,
        }

    abs_total_error_sum = 0.0
    exact_total_matches = 0
    abs_player_error_sum = 0.0
    exact_player_slots = 0
    exact_player_scenes = 0
    pred_total_sum = 0.0
    gt_total_sum = 0.0
    abs_player_error_by_player = {player: 0.0 for player in _PLAYER_KEYS}

    for sample_idx in range(batch_size):
        pred_counts, _, pred_total = _count_cards_by_player(
            preds[sample_idx, 0], min_component_area=min_component_area,
        )
        gt_counts, _, gt_total = _count_cards_by_player(
            gts[sample_idx, 0], min_component_area=min_component_area,
        )

        pred_total_sum += float(pred_total)
        gt_total_sum += float(gt_total)

        total_error = abs(pred_total - gt_total)
        abs_total_error_sum += float(total_error)
        if total_error == 0:
            exact_total_matches += 1

        sample_all_players_exact = True
        for player in _PLAYER_KEYS:
            player_error = abs(pred_counts[player] - gt_counts[player])
            abs_player_error_sum += float(player_error)
            abs_player_error_by_player[player] += float(player_error)
            if player_error == 0:
                exact_player_slots += 1
            else:
                sample_all_players_exact = False

        if sample_all_players_exact:
            exact_player_scenes += 1

    player_slots = batch_size * len(_PLAYER_KEYS)
    return {
        "count_mae": abs_total_error_sum / batch_size,
        "count_exact_rate": exact_total_matches / batch_size,
        "player_count_mae": abs_player_error_sum / max(player_slots, 1),
        "player_exact_rate": exact_player_slots / max(player_slots, 1),
        "scene_player_exact_rate": exact_player_scenes / batch_size,
        "pred_count_mean": pred_total_sum / batch_size,
        "gt_count_mean": gt_total_sum / batch_size,
        "player_count_mae_p1": abs_player_error_by_player["p1"] / batch_size,
        "player_count_mae_p2": abs_player_error_by_player["p2"] / batch_size,
        "player_count_mae_p3": abs_player_error_by_player["p3"] / batch_size,
        "player_count_mae_p4": abs_player_error_by_player["p4"] / batch_size,
    }


def _inverse_error_score(value: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(value)))


def _selection_metric_score(val_metrics: dict[str, float], cfg: SegmenterPipelineConfig) -> float:
    metric = cfg.best_epoch_selection_metric.strip().lower()

    direct_metrics: dict[str, str] = {
        "val_iou": "iou",
        "iou": "iou",
        "val_dice": "dice",
        "dice": "dice",
        "val_precision": "precision",
        "precision": "precision",
        "val_recall": "recall",
        "recall": "recall",
        "val_count_exact_rate": "count_exact_rate",
        "count_exact_rate": "count_exact_rate",
        "val_player_exact_rate": "player_exact_rate",
        "player_exact_rate": "player_exact_rate",
        "val_scene_player_exact_rate": "scene_player_exact_rate",
        "scene_player_exact_rate": "scene_player_exact_rate",
    }

    inverse_error_metrics: dict[str, str] = {
        "neg_val_count_mae": "count_mae",
        "inv_val_count_mae": "count_mae",
        "neg_val_player_count_mae": "player_count_mae",
        "inv_val_player_count_mae": "player_count_mae",
    }

    if metric in direct_metrics:
        return float(val_metrics[direct_metrics[metric]])
    if metric in inverse_error_metrics:
        return _inverse_error_score(float(val_metrics[inverse_error_metrics[metric]]))
    if metric != "composite":
        raise ValueError(
            "Unsupported best_epoch_selection_metric. "
            "Use one of: composite, val_iou, val_dice, val_precision, val_recall, "
            "val_count_exact_rate, val_player_exact_rate, val_scene_player_exact_rate, "
            "neg_val_count_mae, neg_val_player_count_mae."
        )

    weighted_terms = [
        (float(cfg.selection_weight_iou), float(val_metrics["iou"])),
        (float(cfg.selection_weight_dice), float(val_metrics["dice"])),
        (float(cfg.selection_weight_precision), float(val_metrics["precision"])),
        (float(cfg.selection_weight_recall), float(val_metrics["recall"])),
        (float(cfg.selection_weight_count_exact_rate), float(val_metrics["count_exact_rate"])),
        (float(cfg.selection_weight_player_exact_rate), float(val_metrics["player_exact_rate"])),
        (float(cfg.selection_weight_scene_player_exact_rate), float(val_metrics["scene_player_exact_rate"])),
        (float(cfg.selection_weight_count_mae), _inverse_error_score(float(val_metrics["count_mae"]))),
        (
            float(cfg.selection_weight_player_count_mae),
            _inverse_error_score(float(val_metrics["player_count_mae"])),
        ),
    ]

    total_weight = sum(max(0.0, weight) for weight, _ in weighted_terms)
    if total_weight <= 0.0:
        raise ValueError("Composite selection weights must contain at least one positive value.")

    weighted_sum = sum(max(0.0, weight) * value for weight, value in weighted_terms)
    return float(weighted_sum / total_weight)


def _format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    return f"{minutes:02d}:{rem:02d}"


# --------------------------------------------------------------------------- #
# Training/eval loops
# --------------------------------------------------------------------------- #


def _train_one_epoch_verbose(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    scaler,
    amp_enabled: bool,
    device: torch.device,
    log_every: int = 25,
    channels_last: bool = False,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    seen = 0
    num_batches = len(loader)
    loop_start = time.perf_counter()

    for batch_idx, (imgs, masks) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(imgs)
            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        overlap = overlap_metrics_from_logits(logits.detach(), masks)
        running_loss += loss.item() * bs
        running_iou += overlap["iou"] * bs
        seen += bs

        if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches):
            elapsed = time.perf_counter() - loop_start
            avg_batch_seconds = elapsed / batch_idx
            eta_seconds = avg_batch_seconds * (num_batches - batch_idx)
            avg_loss = running_loss / max(seen, 1)
            avg_iou = running_iou / max(seen, 1)
            print(
                f"    batch {batch_idx:03d}/{num_batches:03d} | "
                f"avg_loss={avg_loss:.4f} avg_iou={avg_iou:.4f} | "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )

    return running_loss / max(seen, 1), running_iou / max(seen, 1), seen


@torch.no_grad()
def _evaluate(
    model,
    loader,
    bce_loss,
    amp_enabled,
    device,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eval_threshold: float = 0.5,
    eval_min_component_area: int | None = None,
):
    model.eval()
    running_loss = 0.0
    metric_names = [
        "iou",
        "precision",
        "recall",
        "dice",
        "count_mae",
        "count_exact_rate",
        "player_count_mae",
        "player_exact_rate",
        "scene_player_exact_rate",
        "pred_count_mean",
        "gt_count_mean",
        "player_count_mae_p1",
        "player_count_mae_p2",
        "player_count_mae_p3",
        "player_count_mae_p4",
    ]
    running_metrics = {name: 0.0 for name in metric_names}
    seen = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(imgs)
            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice

        bs = imgs.size(0)
        overlap = overlap_metrics_from_logits(logits, masks, threshold=eval_threshold)
        count_metrics = _count_metrics_from_logits(
            logits,
            masks,
            threshold=eval_threshold,
            min_component_area=eval_min_component_area,
        )

        running_loss += loss.item() * bs
        for name in ("iou", "precision", "recall", "dice"):
            running_metrics[name] += overlap[name] * bs
        for name in count_metrics:
            running_metrics[name] += count_metrics[name] * bs
        seen += bs

    denom = max(seen, 1)
    results = {name: value / denom for name, value in running_metrics.items()}
    results["loss"] = running_loss / denom
    results["seen"] = float(seen)
    return results


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_pairs(scene_images_dir: Path, scene_masks_dir: Path):
    valid_img_ext = {".jpg", ".jpeg", ".png"}
    valid_mask_ext = {".png", ".jpg", ".jpeg"}

    mask_by_stem: dict[str, Path] = {}
    for mask_path in sorted(scene_masks_dir.iterdir()):
        if mask_path.suffix.lower() in valid_mask_ext:
            mask_by_stem[mask_path.stem] = mask_path

    scene_pairs: list[tuple[Path, Path]] = []
    missing_masks: list[str] = []
    for img_path in sorted(scene_images_dir.iterdir()):
        if img_path.suffix.lower() not in valid_img_ext:
            continue
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is not None and mask_path.is_file():
            scene_pairs.append((img_path, mask_path))
        else:
            missing_masks.append(img_path.name)

    return scene_pairs, missing_masks


def initialize_segmenter_pipeline(config: SegmenterPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or SegmenterPipelineConfig()
    _seed_everything(cfg.seed)

    project_root = find_workspace_root()
    training_root = project_root / "project" / "training_data"
    scene_images_dir = training_root / "training_images" / "augmented_scenes"
    scene_masks_dir = training_root / "training_masks" / "augmented_scenes"
    models_dir = project_root / "project" / "models" / "segmenter" / "used"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not scene_images_dir.exists():
        raise FileNotFoundError(f"Scene images directory not found: {scene_images_dir}")
    if not scene_masks_dir.exists():
        raise FileNotFoundError(f"Scene masks directory not found: {scene_masks_dir}")

    # R3: training-only data — fail loudly if anyone points us at a "test" dir.
    for p in (scene_images_dir, scene_masks_dir):
        assert "test" not in str(p).lower(), f"Refusing to train on path containing 'test': {p}"

    device = _select_device()
    use_amp = cfg.use_amp and device.type == "cuda"

    print(f"Project root: {project_root}")
    print(f"Scene images: {scene_images_dir}")
    print(f"Scene masks: {scene_masks_dir}")
    print(f"Model output dir: {models_dir}")
    print(f"torch: {torch.__version__}")
    print(f"Device: {device}")

    # ---------------- pairs & split ----------------
    scene_pairs, missing_masks = _build_pairs(scene_images_dir, scene_masks_dir)
    print(f"Total scene pairs discovered: {len(scene_pairs)}")
    if missing_masks:
        print(f"Images without matching mask: {len(missing_masks)}")
        print("First missing examples:", missing_masks[:5])

    if len(scene_pairs) == 0:
        raise RuntimeError(
            "No augmented scene pairs found. Check training_images/augmented_scenes and "
            "training_masks/augmented_scenes."
        )

    if cfg.max_scene_pairs is not None:
        if cfg.max_scene_pairs <= 0:
            raise ValueError(f"max_scene_pairs must be > 0 or None, got {cfg.max_scene_pairs}")
        if len(scene_pairs) > cfg.max_scene_pairs:
            subset_rng = random.Random(cfg.seed)
            scene_pairs = subset_rng.sample(scene_pairs, k=cfg.max_scene_pairs)
            print(
                f"Using subset of augmented scenes: {len(scene_pairs)} pairs "
                f"(max_scene_pairs={cfg.max_scene_pairs})"
            )
        else:
            print(
                f"max_scene_pairs={cfg.max_scene_pairs} (no-op: dataset has fewer pairs)"
            )

    train_pairs, val_pairs = train_test_split(
        scene_pairs, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True,
    )
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ---------------- datasets / loaders ----------------
    train_ds = SceneSegDataset(train_pairs, image_size=cfg.image_size, augment=True, cache_in_ram=cfg.cache_in_ram)
    val_ds = SceneSegDataset(val_pairs, image_size=cfg.image_size, augment=False, cache_in_ram=cfg.cache_in_ram)

    is_windows = os.name == "nt"
    effective_workers = cfg.num_workers
    if cfg.cache_in_ram and effective_workers > 0:
        print("[info] cache_in_ram=True with num_workers>0 can stall on Windows; forcing num_workers=0")
        effective_workers = 0

    train_sampler = None
    epoch_train_samples = len(train_ds)
    if cfg.epoch_max_train_samples is not None:
        if cfg.epoch_max_train_samples <= 0:
            raise ValueError(
                f"epoch_max_train_samples must be > 0 or None, got {cfg.epoch_max_train_samples}"
            )
        epoch_train_samples = min(cfg.epoch_max_train_samples, len(train_ds))
        if epoch_train_samples < len(train_ds):
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(cfg.seed)
            train_sampler = RandomSampler(
                train_ds,
                replacement=False,
                num_samples=epoch_train_samples,
                generator=sampler_generator,
            )
            print(
                f"Per-epoch train cap enabled: {epoch_train_samples}/{len(train_ds)} samples "
                f"(epoch_max_train_samples={cfg.epoch_max_train_samples})"
            )
        else:
            print(
                f"epoch_max_train_samples={cfg.epoch_max_train_samples} "
                f"(no-op: train set is smaller)"
            )

    loader_kwargs: dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        num_workers=effective_workers,
        pin_memory=device.type == "cuda",
    )
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if train_sampler is None:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=False, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(
        f"Train samples: {len(train_ds)} (per-epoch: {epoch_train_samples}) | "
        f"Val samples: {len(val_ds)}"
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Loader workers in use: {effective_workers} (windows={is_windows})")

    # ---------------- model / optim ----------------
    model = SceneUNetSmall().to(device)
    channels_last = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    else:
        print("[warning] CUDA not available. Training can be significantly slower on CPU.")

    warm_start_checkpoint = models_dir / cfg.warm_start_filename
    if cfg.warm_start and warm_start_checkpoint.is_file():
        state_dict = torch.load(warm_start_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded warm-start checkpoint: {warm_start_checkpoint}")
    elif cfg.warm_start:
        print("[info] Warm-start checkpoint not found. Training from scratch.")

    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_params = assert_param_cap(model, "SceneUNetSmall")

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.scheduler_patience < 0:
        raise ValueError(f"scheduler_patience must be >= 0, got {cfg.scheduler_patience}")
    if not (0.0 < cfg.scheduler_factor <= 1.0):
        raise ValueError(f"scheduler_factor must be in (0, 1], got {cfg.scheduler_factor}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    scene_checkpoint = models_dir / cfg.checkpoint_filename
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Scene checkpoint: {scene_checkpoint}")

    return {
        "config": cfg,
        "project_root": project_root,
        "training_root": training_root,
        "scene_images_dir": scene_images_dir,
        "scene_masks_dir": scene_masks_dir,
        "models_dir": models_dir,
        "device": device,
        "use_amp": use_amp,
        "channels_last": channels_last,
        "scene_pairs": scene_pairs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "epoch_train_samples": epoch_train_samples,
        "train_loader": train_loader,
        "train_loader_kwargs": loader_kwargs,
        "val_loader": val_loader,
        "model": model,
        "model_params": n_params,
        "bce_loss": bce_loss,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "scene_checkpoint": scene_checkpoint,
        "history": {
            "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
            "val_selection_score": [],
            "val_precision": [], "val_recall": [], "val_dice": [],
            "val_count_mae": [], "val_count_exact_rate": [],
            "val_player_count_mae": [], "val_player_exact_rate": [],
            "val_scene_player_exact_rate": [],
            "val_pred_count_mean": [], "val_gt_count_mean": [],
            "val_player_count_mae_p1": [], "val_player_count_mae_p2": [],
            "val_player_count_mae_p3": [], "val_player_count_mae_p4": [],
            "lr": [], "epoch_seconds": [], "train_seconds": [], "val_seconds": [],
            "images_per_sec": [], "gpu_mem_gb": [],
        },
        "best_val_iou": None,
        "best_epoch": None,
        "training_seconds": None,
    }


# --------------------------------------------------------------------------- #
# Run training
# --------------------------------------------------------------------------- #


def run_segmenter_training(state: dict[str, Any]) -> dict[str, Any]:
    cfg: SegmenterPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    bce_loss: nn.Module = state["bce_loss"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    channels_last: bool = state["channels_last"]
    train_loader: DataLoader = state["train_loader"]
    train_ds: SceneSegDataset = state["train_ds"]
    train_loader_kwargs: dict[str, Any] = state["train_loader_kwargs"]
    val_loader: DataLoader = state["val_loader"]
    epoch_train_samples: int = state.get("epoch_train_samples", len(train_loader.dataset))
    scene_checkpoint: Path = state["scene_checkpoint"]
    history: dict[str, list] = state["history"]

    loss_weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    if cfg.train_loss_bce_weight < 0 or cfg.train_loss_dice_weight < 0 or loss_weight_sum <= 0:
        raise ValueError(
            "train_loss_bce_weight and train_loss_dice_weight must be >= 0 "
            "and at least one must be > 0"
        )
    bce_weight = cfg.train_loss_bce_weight / loss_weight_sum
    dice_weight = cfg.train_loss_dice_weight / loss_weight_sum

    best_val_iou = -1.0
    best_selection_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_start = time.perf_counter()

    # If epochs are capped, cycle through a shuffled index list so all training
    # samples are consumed before any repeat.
    coverage_indices: list[int] | None = None
    coverage_cursor = 0
    coverage_rng: random.Random | None = None
    if epoch_train_samples < len(train_ds):
        coverage_indices = list(range(len(train_ds)))
        coverage_rng = random.Random(cfg.seed)
        coverage_rng.shuffle(coverage_indices)

    print(
        f"Starting training on {device} | train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} | epoch_train_samples={epoch_train_samples} "
        f"| loss_mix(bce/dice)=({bce_weight:.2f}/{dice_weight:.2f})",
        flush=True,
    )
    if epoch_train_samples < len(train_ds):
        print(
            "Per-epoch train cap is active: epochs cycle through a shuffled full "
            "training index pool before repeating samples.",
            flush=True,
        )
    if device.type != "cuda":
        print("[warning] Running on CPU. It is normal if each epoch takes several minutes.", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[Epoch {epoch:02d}/{cfg.epochs}] starting...", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if epoch_train_samples < len(train_ds):
            if coverage_indices is None or coverage_rng is None:
                raise RuntimeError("Coverage sampler state was not initialized.")

            selected_indices: list[int] = []
            while len(selected_indices) < epoch_train_samples:
                remaining_in_cycle = len(coverage_indices) - coverage_cursor
                to_take = min(epoch_train_samples - len(selected_indices), remaining_in_cycle)
                selected_indices.extend(
                    coverage_indices[coverage_cursor: coverage_cursor + to_take]
                )
                coverage_cursor += to_take

                if coverage_cursor >= len(coverage_indices):
                    coverage_rng.shuffle(coverage_indices)
                    coverage_cursor = 0

            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(cfg.seed + epoch)
            epoch_sampler = SubsetRandomSampler(selected_indices, generator=epoch_generator)
            train_loader = DataLoader(
                train_ds,
                shuffle=False,
                sampler=epoch_sampler,
                **train_loader_kwargs,
            )

        train_start = time.perf_counter()
        train_loss, train_iou, n_train = _train_one_epoch_verbose(
            model, train_loader, optimizer, bce_loss, scaler, use_amp, device,
            log_every=cfg.log_every_batches, channels_last=channels_last,
            bce_weight=bce_weight, dice_weight=dice_weight,
        )
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_metrics = _evaluate(
            model, val_loader, bce_loss, use_amp, device,
            bce_weight=bce_weight, dice_weight=dice_weight,
            eval_threshold=cfg.eval_mask_threshold,
            eval_min_component_area=cfg.eval_min_component_area,
        )
        val_loss = float(val_metrics["loss"])
        val_iou = float(val_metrics["iou"])
        val_precision = float(val_metrics["precision"])
        val_recall = float(val_metrics["recall"])
        val_dice = float(val_metrics["dice"])
        val_count_mae = float(val_metrics["count_mae"])
        val_count_exact_rate = float(val_metrics["count_exact_rate"])
        val_player_count_mae = float(val_metrics["player_count_mae"])
        val_player_exact_rate = float(val_metrics["player_exact_rate"])
        val_scene_player_exact_rate = float(val_metrics["scene_player_exact_rate"])
        val_pred_count_mean = float(val_metrics["pred_count_mean"])
        val_gt_count_mean = float(val_metrics["gt_count_mean"])
        val_player_count_mae_p1 = float(val_metrics["player_count_mae_p1"])
        val_player_count_mae_p2 = float(val_metrics["player_count_mae_p2"])
        val_player_count_mae_p3 = float(val_metrics["player_count_mae_p3"])
        val_player_count_mae_p4 = float(val_metrics["player_count_mae_p4"])
        selection_score = _selection_metric_score(val_metrics, cfg)
        val_seconds = time.perf_counter() - val_start

        scheduler.step(val_iou)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_seconds = train_seconds + val_seconds
        images_per_sec = n_train / max(train_seconds, 1e-6)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["val_selection_score"].append(selection_score)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_dice"].append(val_dice)
        history["val_count_mae"].append(val_count_mae)
        history["val_count_exact_rate"].append(val_count_exact_rate)
        history["val_player_count_mae"].append(val_player_count_mae)
        history["val_player_exact_rate"].append(val_player_exact_rate)
        history["val_scene_player_exact_rate"].append(val_scene_player_exact_rate)
        history["val_pred_count_mean"].append(val_pred_count_mean)
        history["val_gt_count_mean"].append(val_gt_count_mean)
        history["val_player_count_mae_p1"].append(val_player_count_mae_p1)
        history["val_player_count_mae_p2"].append(val_player_count_mae_p2)
        history["val_player_count_mae_p3"].append(val_player_count_mae_p3)
        history["val_player_count_mae_p4"].append(val_player_count_mae_p4)
        history["lr"].append(lr_now)
        history["epoch_seconds"].append(epoch_seconds)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["images_per_sec"].append(images_per_sec)
        history["gpu_mem_gb"].append(gpu_mem_gb)

        improved = selection_score > best_selection_score
        if improved:
            best_selection_score = selection_score
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), scene_checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        recent_times = history["epoch_seconds"][-5:]
        eta_seconds = float(np.mean(recent_times) * max(cfg.epochs - epoch, 0))

        if cfg.early_stopping_patience is None:
            status = "best checkpoint saved" if improved else "no improvement"
        else:
            status = (
                "best checkpoint saved"
                if improved
                else f"no improvement ({epochs_without_improvement}/{cfg.early_stopping_patience})"
            )

        print(
            f"  summary | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} | "
            f"val_precision={val_precision:.4f} val_recall={val_recall:.4f} | "
            f"selection({cfg.best_epoch_selection_metric})={selection_score:.4f} | lr={lr_now:.2e}",
            flush=True,
        )
        print(
            f"  counts  | cards(pred/gt)={val_pred_count_mean:.2f}/{val_gt_count_mean:.2f} "
            f"count_mae={val_count_mae:.3f} exact={val_count_exact_rate:.3f} | "
            f"player_mae={val_player_count_mae:.3f} player_exact={val_player_exact_rate:.3f} "
            f"scene_player_exact={val_scene_player_exact_rate:.3f}",
            flush=True,
        )
        print(
            "           "
            f"per-player MAE: p1={val_player_count_mae_p1:.3f} "
            f"p2={val_player_count_mae_p2:.3f} "
            f"p3={val_player_count_mae_p3:.3f} "
            f"p4={val_player_count_mae_p4:.3f}",
            flush=True,
        )
        print(
            f"  times   | train={_format_seconds(train_seconds)} val={_format_seconds(val_seconds)} "
            f"epoch={_format_seconds(epoch_seconds)} | "
            f"speed={images_per_sec:.1f} img/s | gpu_mem={gpu_mem_gb:.2f} GB | "
            f"eta={_format_seconds(eta_seconds)} | {status}",
            flush=True,
        )

        if (
            cfg.early_stopping_patience is not None
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    total_seconds = time.perf_counter() - global_start
    print(f"\nTraining finished in {_format_seconds(total_seconds)}", flush=True)
    print(
        f"Best epoch ({cfg.best_epoch_selection_metric}): {best_epoch} "
        f"with selection score={best_selection_score:.4f}",
        flush=True,
    )
    print(f"Best-epoch val IoU: {best_val_iou:.4f}", flush=True)
    if best_epoch > 0:
        best_idx = best_epoch - 1
        print(
            f"Best val Dice: {history['val_dice'][best_idx]:.4f} | "
            f"Best val count MAE: {history['val_count_mae'][best_idx]:.3f} | "
            f"Best scene player exact: {history['val_scene_player_exact_rate'][best_idx]:.3f}",
            flush=True,
        )
    print(f"Saved best model: {scene_checkpoint}", flush=True)

    state["best_val_iou"] = float(best_val_iou)
    state["best_selection_score"] = float(best_selection_score)
    state["best_selection_metric"] = str(cfg.best_epoch_selection_metric)
    state["best_epoch"] = int(best_epoch)
    state["training_seconds"] = float(total_seconds)
    if best_epoch > 0:
        best_idx = best_epoch - 1
        state["best_val_metrics"] = {
            "val_iou": float(history["val_iou"][best_idx]),
            "val_selection_score": float(history["val_selection_score"][best_idx]),
            "val_dice": float(history["val_dice"][best_idx]),
            "val_precision": float(history["val_precision"][best_idx]),
            "val_recall": float(history["val_recall"][best_idx]),
            "val_count_mae": float(history["val_count_mae"][best_idx]),
            "val_count_exact_rate": float(history["val_count_exact_rate"][best_idx]),
            "val_player_count_mae": float(history["val_player_count_mae"][best_idx]),
            "val_player_exact_rate": float(history["val_player_exact_rate"][best_idx]),
            "val_scene_player_exact_rate": float(history["val_scene_player_exact_rate"][best_idx]),
            "val_pred_count_mean": float(history["val_pred_count_mean"][best_idx]),
            "val_gt_count_mean": float(history["val_gt_count_mean"][best_idx]),
            "val_player_count_mae_p1": float(history["val_player_count_mae_p1"][best_idx]),
            "val_player_count_mae_p2": float(history["val_player_count_mae_p2"][best_idx]),
            "val_player_count_mae_p3": float(history["val_player_count_mae_p3"][best_idx]),
            "val_player_count_mae_p4": float(history["val_player_count_mae_p4"][best_idx]),
        }
    return state


@torch.no_grad()
def evaluate_segmenter_validation(state: dict[str, Any], load_best_checkpoint: bool = True) -> dict[str, float]:
    model: nn.Module = state["model"]
    val_loader: DataLoader = state["val_loader"]
    bce_loss: nn.Module = state["bce_loss"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    scene_checkpoint: Path = state["scene_checkpoint"]
    cfg: SegmenterPipelineConfig = state["config"]

    if load_best_checkpoint and scene_checkpoint.is_file():
        model.load_state_dict(torch.load(scene_checkpoint, map_location=device))

    loss_weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    if loss_weight_sum <= 0:
        raise ValueError("train_loss_bce_weight + train_loss_dice_weight must be > 0.")
    bce_weight = cfg.train_loss_bce_weight / loss_weight_sum
    dice_weight = cfg.train_loss_dice_weight / loss_weight_sum

    metrics = _evaluate(
        model,
        val_loader,
        bce_loss,
        use_amp,
        device,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        eval_threshold=cfg.eval_mask_threshold,
        eval_min_component_area=cfg.eval_min_component_area,
    )
    selection_score = _selection_metric_score(metrics, cfg)

    print(
        f"Validation metrics | loss={metrics['loss']:.4f} iou={metrics['iou']:.4f} "
        f"dice={metrics['dice']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f}",
        flush=True,
    )
    print(
        f"Card counts        | cards(pred/gt)={metrics['pred_count_mean']:.2f}/{metrics['gt_count_mean']:.2f} "
        f"count_mae={metrics['count_mae']:.3f} count_exact={metrics['count_exact_rate']:.3f}",
        flush=True,
    )
    print(
        f"Per-player counts  | player_mae={metrics['player_count_mae']:.3f} "
        f"player_exact={metrics['player_exact_rate']:.3f} "
        f"scene_player_exact={metrics['scene_player_exact_rate']:.3f}",
        flush=True,
    )
    print(
        f"Selection score    | metric={cfg.best_epoch_selection_metric} "
        f"score={selection_score:.4f}",
        flush=True,
    )
    print(
        "Per-player MAE     "
        f"| p1={metrics['player_count_mae_p1']:.3f} "
        f"p2={metrics['player_count_mae_p2']:.3f} "
        f"p3={metrics['player_count_mae_p3']:.3f} "
        f"p4={metrics['player_count_mae_p4']:.3f}",
        flush=True,
    )

    metrics["selection_score"] = float(selection_score)
    state["last_validation_metrics"] = metrics
    return metrics


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def plot_training_curves(state: dict[str, Any]) -> None:
    cfg: SegmenterPipelineConfig = state["config"]
    history = state["history"]
    if len(history["train_loss"]) == 0:
        raise RuntimeError("History is empty. Run the training cell first.")

    epochs_x = np.arange(1, len(history["train_loss"]) + 1)
    if "val_selection_score" in history and len(history["val_selection_score"]) == len(epochs_x):
        best_idx = int(np.argmax(history["val_selection_score"]))
    else:
        best_idx = int(np.argmax(history["val_iou"]))
    best_epoch_plot = int(epochs_x[best_idx])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs_x, history["train_loss"], marker="o", label="train")
    axes[0, 0].plot(epochs_x, history["val_loss"], marker="o", label="val")
    axes[0, 0].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_x, history["train_iou"], marker="o", label="train IoU")
    axes[0, 1].plot(epochs_x, history["val_iou"], marker="o", label="val IoU")
    if "val_dice" in history and len(history["val_dice"]) == len(epochs_x):
        axes[0, 1].plot(epochs_x, history["val_dice"], marker="o", label="val Dice")
    if "val_precision" in history and len(history["val_precision"]) == len(epochs_x):
        axes[0, 1].plot(epochs_x, history["val_precision"], marker="o", label="val Precision")
    if "val_recall" in history and len(history["val_recall"]) == len(epochs_x):
        axes[0, 1].plot(epochs_x, history["val_recall"], marker="o", label="val Recall")
    if "val_selection_score" in history and len(history["val_selection_score"]) == len(epochs_x):
        axes[0, 1].plot(
            epochs_x,
            history["val_selection_score"],
            marker="o",
            linestyle="--",
            color="k",
            label=f"selection ({cfg.best_epoch_selection_metric})",
        )
    axes[0, 1].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Overlap Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_x, history["lr"], marker="o", color="tab:green")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR (log scale)")

    if "train_seconds" in history and len(history["train_seconds"]) == len(epochs_x):
        train_seconds = np.array(history["train_seconds"])
        val_seconds = np.array(history["val_seconds"])
        axes[1, 1].bar(epochs_x, train_seconds, alpha=0.7, label="train seconds")
        axes[1, 1].bar(epochs_x, val_seconds, bottom=train_seconds, alpha=0.7, label="val seconds")
    else:
        axes[1, 1].bar(epochs_x, history["epoch_seconds"], alpha=0.6, label="epoch seconds")

    ax_speed = axes[1, 1].twinx()
    ax_speed.plot(epochs_x, history["images_per_sec"], color="tab:red", marker="o", label="images/sec")
    axes[1, 1].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Epoch Time And Throughput")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Seconds")
    ax_speed.set_ylabel("Images/sec")

    handles_1, labels_1 = axes[1, 1].get_legend_handles_labels()
    handles_2, labels_2 = ax_speed.get_legend_handles_labels()
    axes[1, 1].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    plt.suptitle("Scene Segmenter Training Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.show()

    if "val_count_mae" in history and len(history["val_count_mae"]) == len(epochs_x):
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.5))

        axes2[0].plot(epochs_x, history["val_count_mae"], marker="o", label="total count MAE")
        axes2[0].plot(epochs_x, history["val_player_count_mae"], marker="o", label="player count MAE")
        ax_count_means = axes2[0].twinx()
        ax_count_means.plot(
            epochs_x,
            history["val_pred_count_mean"],
            marker="o",
            linestyle="--",
            color="tab:orange",
            label="pred cards/scene",
        )
        ax_count_means.plot(
            epochs_x,
            history["val_gt_count_mean"],
            marker="o",
            linestyle="--",
            color="tab:green",
            label="gt cards/scene",
        )
        axes2[0].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
        axes2[0].set_title("Count Error")
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("Absolute Error")
        ax_count_means.set_ylabel("Cards Per Scene")
        handles_1, labels_1 = axes2[0].get_legend_handles_labels()
        handles_2, labels_2 = ax_count_means.get_legend_handles_labels()
        axes2[0].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

        axes2[1].plot(
            epochs_x,
            history["val_count_exact_rate"],
            marker="o",
            label="exact total card count",
        )
        axes2[1].plot(
            epochs_x,
            history["val_player_exact_rate"],
            marker="o",
            label="exact player slot count",
        )
        axes2[1].plot(
            epochs_x,
            history["val_scene_player_exact_rate"],
            marker="o",
            label="exact all-player counts",
        )
        axes2[1].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
        axes2[1].set_title("Count Exact-Match Rates")
        axes2[1].set_xlabel("Epoch")
        axes2[1].set_ylabel("Rate")
        axes2[1].set_ylim(0.0, 1.02)
        axes2[1].legend(loc="lower right")

        for ax in axes2.ravel():
            ax.grid(alpha=0.25)

        plt.suptitle("Scene Segmenter Count Diagnostics", fontsize=13)
        plt.tight_layout()
        plt.show()

    print(f"Best epoch: {best_epoch_plot}")
    if "val_selection_score" in history and len(history["val_selection_score"]) == len(epochs_x):
        print(
            f"Best selection score ({cfg.best_epoch_selection_metric}): "
            f"{history['val_selection_score'][best_idx]:.4f}"
        )
    print(f"Best val IoU: {history['val_iou'][best_idx]:.4f}")
    if "val_dice" in history and len(history["val_dice"]) == len(epochs_x):
        print(f"Best val Dice: {history['val_dice'][best_idx]:.4f}")
    if "val_count_mae" in history and len(history["val_count_mae"]) == len(epochs_x):
        print(
            f"Best val count MAE: {history['val_count_mae'][best_idx]:.3f} | "
            f"Best exact all-player rate: {history['val_scene_player_exact_rate'][best_idx]:.3f}"
        )
    print(f"Epoch time at best epoch: {history['epoch_seconds'][best_idx]:.2f}s")
    print(f"Mean throughput: {np.mean(history['images_per_sec']):.2f} img/s")


def plot_segmentation_predictions(state: dict[str, Any], num_show: int | None = None) -> None:
    model: nn.Module = state["model"]
    val_ds: SceneSegDataset = state["val_ds"]
    device: torch.device = state["device"]
    scene_checkpoint: Path = state["scene_checkpoint"]
    cfg: SegmenterPipelineConfig = state["config"]

    if scene_checkpoint.is_file():
        model.load_state_dict(torch.load(scene_checkpoint, map_location=device))
    model.eval()

    n_show = int(num_show or cfg.preview_count)
    n_show = min(n_show, len(val_ds))
    if n_show == 0:
        raise RuntimeError("Validation set is empty.")

    indices = np.linspace(0, len(val_ds) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 3, figsize=(11, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_t, mask_t = val_ds[int(idx)]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_bin = (pred > cfg.eval_mask_threshold).astype(np.float32)

            gt = mask_t[0].cpu().numpy()
            inter = np.logical_and(pred_bin > 0.5, gt > 0.5).sum()
            union = np.logical_or(pred_bin > 0.5, gt > 0.5).sum()
            sample_iou = (inter + 1e-6) / (union + 1e-6)
            pred_counts, _, pred_total = _count_cards_by_player(
                pred_bin,
                min_component_area=cfg.eval_min_component_area,
            )
            gt_counts, _, gt_total = _count_cards_by_player(
                gt,
                min_component_area=cfg.eval_min_component_area,
            )
            sample_player_mae = float(
                np.mean([abs(pred_counts[player] - gt_counts[player]) for player in _PLAYER_KEYS])
            )

            axes[row, 0].imshow(denormalize_image(img_t))
            axes[row, 0].set_title("Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt, cmap="gray")
            axes[row, 1].set_title("GT Mask")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred_bin, cmap="gray")
            axes[row, 2].set_title(
                f"Pred (IoU={sample_iou:.3f}, cards={pred_total}/{gt_total}, "
                f"player-MAE={sample_player_mae:.2f})"
            )
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()
