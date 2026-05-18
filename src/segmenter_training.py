"""Report helpers for the scene-segmenter training stage."""
from __future__ import annotations

import json
import gc
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
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SubsetRandomSampler

from src.inference import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    InferenceEngine,
    SceneUNetSmall,
    assert_param_cap,
    assign_region,
    boxes_from_probability,
    segment_scene_probability,
)


@dataclass
class SegmenterPipelineConfig:
    """Configuration used by the segmenter training notebook."""

    seed: int = 42
    image_size: int = 256
    val_split: float = 0.35
    max_scene_pairs: int | None = None
    epoch_max_train_samples: int | None = 8192
    cache_in_ram: bool = True
    num_workers: int = 4
    persistent_workers: bool = True

    epochs: int = 35
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    train_loss_bce_weight: float = 0.5
    train_loss_dice_weight: float = 0.5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    use_amp: bool = True
    use_torch_compile: bool = False
    early_stopping_patience: int | None = 8
    log_every_batches: int = 0

    warm_start: bool = False
    warm_start_filename: str = "segmenter_unet_small.pth"
    checkpoint_filename: str = "scene_segmenter_unet_small.pth"

    preview_count: int = 6
    eval_mask_threshold: float = 0.5
    eval_min_component_area: int | None = 4000

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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_seconds(seconds: float) -> str:
    seconds = int(round(float(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _shutdown_loader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if iterator is not None:
        try:
            setattr(loader, "_iterator", None)
        except Exception:
            pass


def _letterbox_pil(image: Image.Image, target_size: int, fill: int, interpolation: int) -> Image.Image:
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), interpolation)
    pad_width = target_size - new_width
    pad_height = target_size - new_height
    padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
    return TF.pad(image, padding, fill=fill)


def preprocess_scene_pair(image_path: Path, mask_path: Path, image_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image = _letterbox_pil(image, target_size=image_size, fill=255, interpolation=Image.BILINEAR)
    mask = _letterbox_pil(mask, target_size=image_size, fill=0, interpolation=Image.NEAREST)
    image_t = TF.to_tensor(image)
    image_t = TF.normalize(image_t, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    mask_t = torch.from_numpy((np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)).unsqueeze(0)
    return image_t, mask_t


class SceneSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int = 256, augment: bool = False, cache_in_ram: bool = True):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.cache_in_ram:
            print(f"  Pre-loading {len(self.pairs)} scene pairs into RAM...")
            for index, (image_path, mask_path) in enumerate(self.pairs, start=1):
                self.cache.append(preprocess_scene_pair(image_path, mask_path, image_size=self.image_size))
                if index % 1000 == 0 or index == len(self.pairs):
                    print(f"  {index}/{len(self.pairs)} loaded")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_ram:
            image_t, mask_t = self.cache[index]
        else:
            image_path, mask_path = self.pairs[index]
            image_t, mask_t = preprocess_scene_pair(image_path, mask_path, image_size=self.image_size)
        if self.augment:
            if random.random() < 0.5:
                image_t = TF.hflip(image_t)
                mask_t = TF.hflip(mask_t)
            if random.random() < 0.5:
                image_t = TF.vflip(image_t)
                mask_t = TF.vflip(mask_t)
        return image_t, mask_t


class RotatingCoverageSampler(Sampler[int]):
    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: int, shuffle_epoch: bool = True):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = int(min(max(1, samples_per_epoch), dataset_size))
        self.shuffle_epoch = bool(shuffle_epoch)
        self._rng = random.Random(seed)
        self._indices = list(range(self.dataset_size))
        self._rng.shuffle(self._indices)
        self._cursor = 0

    def __iter__(self):
        selected: list[int] = []
        while len(selected) < self.samples_per_epoch:
            remaining = self.dataset_size - self._cursor
            take = min(self.samples_per_epoch - len(selected), remaining)
            selected.extend(self._indices[self._cursor:self._cursor + take])
            self._cursor += take
            if self._cursor >= self.dataset_size:
                self._rng.shuffle(self._indices)
                self._cursor = 0
        if self.shuffle_epoch:
            self._rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return self.samples_per_epoch
    

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).reshape(-1)
    targets = targets.reshape(-1)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + eps) / (probs.sum() + targets.sum() + eps)
    return 1.0 - dice


def overlap_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = (targets > 0.5).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    union = tp + fp + fn
    return {
        "iou": float(((tp + eps) / (union + eps)).mean().item()),
        "dice": float(((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()),
        "precision": float(((tp + eps) / (tp + fp + eps)).mean().item()),
        "recall": float(((tp + eps) / (tp + fn + eps)).mean().item()),
    }


def _component_count(mask_2d: np.ndarray, min_component_area: int | None = None) -> tuple[dict[str, int], int]:
    height, width = mask_2d.shape
    min_area = max(10, int(round(0.00012 * height * width))) if min_component_area is None else max(1, int(min_component_area))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask_2d > 0).astype(np.uint8), connectivity=8)
    counts = {player: 0 for player in PLAYERS}
    total = 0
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        region = assign_region((x, y, x + w, y + h), width, height)
        if region in counts:
            counts[region] += 1
        total += 1
    return counts, total


PLAYERS = ("p1", "p2", "p3", "p4")


def _count_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float, min_component_area: int | None) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).to(torch.uint8).cpu().numpy()
    truths = (targets > 0.5).to(torch.uint8).cpu().numpy()
    total_abs_error = 0.0
    total_exact = 0
    player_abs_error = 0.0
    player_exact = 0
    scene_player_exact = 0
    pred_total_sum = 0.0
    true_total_sum = 0.0
    per_player_abs = {player: 0.0 for player in PLAYERS}
    for index in range(preds.shape[0]):
        pred_counts, pred_total = _component_count(preds[index, 0], min_component_area)
        true_counts, true_total = _component_count(truths[index, 0], min_component_area)
        pred_total_sum += pred_total
        true_total_sum += true_total
        total_error = abs(pred_total - true_total)
        total_abs_error += total_error
        total_exact += int(total_error == 0)
        all_players_exact = True
        for player in PLAYERS:
            error = abs(pred_counts[player] - true_counts[player])
            player_abs_error += error
            per_player_abs[player] += error
            if error == 0:
                player_exact += 1
            else:
                all_players_exact = False
        scene_player_exact += int(all_players_exact)
    batch_size = max(1, int(preds.shape[0]))
    player_slots = batch_size * len(PLAYERS)
    return {
        "count_mae": total_abs_error / batch_size,
        "count_exact_rate": total_exact / batch_size,
        "player_count_mae": player_abs_error / player_slots,
        "player_exact_rate": player_exact / player_slots,
        "scene_player_exact_rate": scene_player_exact / batch_size,
        "pred_count_mean": pred_total_sum / batch_size,
        "gt_count_mean": true_total_sum / batch_size,
        "player_count_mae_p1": per_player_abs["p1"] / batch_size,
        "player_count_mae_p2": per_player_abs["p2"] / batch_size,
        "player_count_mae_p3": per_player_abs["p3"] / batch_size,
        "player_count_mae_p4": per_player_abs["p4"] / batch_size,
    }


def _selection_score(metrics: dict[str, float], cfg: SegmenterPipelineConfig) -> float:
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
        return float(metrics[direct_metrics[metric]])
    if metric in inverse_error_metrics:
        return 1.0 / (1.0 + max(0.0, float(metrics[inverse_error_metrics[metric]])))
    if metric != "composite":
        raise ValueError(
            "Unsupported best_epoch_selection_metric. "
            "Use one of: composite, val_iou, val_dice, val_precision, val_recall, "
            "val_count_exact_rate, val_player_exact_rate, val_scene_player_exact_rate, "
            "neg_val_count_mae, neg_val_player_count_mae."
        )

    weighted_terms = [
        (float(cfg.selection_weight_iou), float(metrics["iou"])),
        (float(cfg.selection_weight_dice), float(metrics["dice"])),
        (float(cfg.selection_weight_precision), float(metrics["precision"])),
        (float(cfg.selection_weight_recall), float(metrics["recall"])),
        (float(cfg.selection_weight_count_exact_rate), float(metrics["count_exact_rate"])),
        (float(cfg.selection_weight_player_exact_rate), float(metrics["player_exact_rate"])),
        (float(cfg.selection_weight_scene_player_exact_rate), float(metrics["scene_player_exact_rate"])),
        (float(cfg.selection_weight_count_mae), 1.0 / (1.0 + max(0.0, float(metrics["count_mae"])))),
        (
            float(cfg.selection_weight_player_count_mae),
            1.0 / (1.0 + max(0.0, float(metrics["player_count_mae"]))),
        ),
    ]
    total_weight = sum(max(0.0, weight) for weight, _ in weighted_terms)
    if total_weight <= 0.0:
        raise ValueError("Composite selection weights must contain at least one positive value.")

    weighted_sum = sum(max(0.0, weight) * value for weight, value in weighted_terms)
    return float(weighted_sum / total_weight)


def _list_scene_images(scene_images_dir: Path) -> list[Path]:
    return sorted(
        [path for path in scene_images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda path: path.name,
    )


def _build_pairs(scene_images_dir: Path, scene_masks_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    masks_by_stem = {path.stem: path for path in scene_masks_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for image_path in _list_scene_images(scene_images_dir):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing.append(image_path)
        else:
            pairs.append((image_path, mask_path))
    return pairs, missing


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    log_every: int,
    channels_last: bool,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_samples = 0
    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = bce_weight * bce_loss(logits, masks) + dice_weight * dice_loss_from_logits(logits, masks)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_iou += overlap_metrics_from_logits(logits.detach(), masks, threshold=0.5)["iou"] * batch_size
        n_samples += batch_size
        if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
            print(f"    batch {batch_index:04d}/{len(loader):04d} loss={loss.item():.4f}", flush=True)
    return total_loss / max(n_samples, 1), total_iou / max(n_samples, 1), n_samples


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    use_amp: bool,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    eval_threshold: float,
    eval_min_component_area: int | None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    count_totals = {
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
    n_samples = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = bce_weight * bce_loss(logits, masks) + dice_weight * dice_loss_from_logits(logits, masks)
        batch_size = int(images.shape[0])
        metrics = overlap_metrics_from_logits(logits, masks, threshold=eval_threshold)
        counts = _count_metrics_from_logits(logits, masks, threshold=eval_threshold, min_component_area=eval_min_component_area)
        totals["loss"] += float(loss.item()) * batch_size
        for key in ("iou", "dice", "precision", "recall"):
            totals[key] += metrics[key] * batch_size
        for key, value in counts.items():
            count_totals[key] += float(value) * batch_size
        n_samples += batch_size
    out = {key: value / max(n_samples, 1) for key, value in totals.items()}
    out.update({key: value / max(n_samples, 1) for key, value in count_totals.items()})
    out["seen"] = float(n_samples)
    return out


def initialize_segmenter_pipeline(
    config: SegmenterPipelineConfig | None = None,
    project_root: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Build datasets, loaders, model, optimizer, and checkpoint paths."""
    cfg = config or SegmenterPipelineConfig()
    _seed_everything(cfg.seed)
    project_root = Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    training_root = project_root / "training_data"
    scene_images_dir = training_root / "augmented_data" / "scene_images"
    scene_masks_dir = training_root / "augmented_data" / "scene_masks"
    model_output_dir = (Path(models_dir).resolve() if models_dir is not None else project_root / "models") / "segmenter" / "used"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    for path in (scene_images_dir, scene_masks_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing segmenter training input: {path}")
        assert "test" not in str(path).lower(), f"Refusing to train on test path: {path}"

    scene_pairs, missing_masks = _build_pairs(scene_images_dir, scene_masks_dir)
    if cfg.max_scene_pairs is not None:
        if cfg.max_scene_pairs <= 0:
            raise ValueError(f"max_scene_pairs must be > 0 or None, got {cfg.max_scene_pairs}")
        if len(scene_pairs) > cfg.max_scene_pairs:
            rng = random.Random(cfg.seed)
            scene_pairs = rng.sample(scene_pairs, k=int(cfg.max_scene_pairs))
    if len(scene_pairs) < 2:
        raise RuntimeError("Need at least two augmented scene/mask pairs to train the segmenter.")
    train_pairs, val_pairs = train_test_split(scene_pairs, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True)

    device = _select_device()
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    train_ds = SceneSegDataset(train_pairs, image_size=cfg.image_size, augment=True, cache_in_ram=cfg.cache_in_ram)
    val_ds = SceneSegDataset(val_pairs, image_size=cfg.image_size, augment=False, cache_in_ram=cfg.cache_in_ram)
    if cfg.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}")
    effective_workers = int(cfg.num_workers)
    if cfg.cache_in_ram and effective_workers > 0:
        print("[info] cache_in_ram=True; using num_workers=0 to avoid duplicating cached tensors.")
        effective_workers = 0
    epoch_train_samples = len(train_ds)
    train_sampler = None
    if cfg.epoch_max_train_samples is not None:
        if cfg.epoch_max_train_samples <= 0:
            raise ValueError(
                f"epoch_max_train_samples must be > 0 or None, got {cfg.epoch_max_train_samples}"
            )
        epoch_train_samples = min(int(cfg.epoch_max_train_samples), len(train_ds))
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
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "num_workers": effective_workers,
        "pin_memory": device.type == "cuda",
    }
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model: nn.Module = SceneUNetSmall().to(device)
    channels_last = device.type == "cuda"
    if channels_last:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    warm_start_checkpoint = model_output_dir / cfg.warm_start_filename
    if cfg.warm_start and warm_start_checkpoint.is_file():
        model.load_state_dict(torch.load(warm_start_checkpoint, map_location=device), strict=False)
        print(f"Loaded warm-start checkpoint: {warm_start_checkpoint}")
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model_params = assert_param_cap(model, "SceneUNetSmall")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    scene_checkpoint = model_output_dir / cfg.checkpoint_filename
    print(
        "Segmenter data: "
        f"train={len(train_pairs):,}, val={len(val_pairs):,}, "
        f"epoch_samples={epoch_train_samples:,}, missing_masks={len(missing_masks):,}."
    )
    print(
        "Segmenter loader: "
        f"batch_size={cfg.batch_size}, train_batches={len(train_loader):,}, val_batches={len(val_loader):,}, "
        f"workers={effective_workers}, persistent={bool(effective_workers > 0)}, "
        f"pin_memory={device.type == 'cuda'}, cache_in_ram={cfg.cache_in_ram}."
    )
    print(f"Segmenter device: {device} | checkpoint: {scene_checkpoint}")
    return {
        "config": cfg,
        "project_root": project_root,
        "training_root": training_root,
        "scene_images_dir": scene_images_dir,
        "scene_masks_dir": scene_masks_dir,
        "models_dir": model_output_dir,
        "device": device,
        "use_amp": use_amp,
        "channels_last": channels_last,
        "scene_pairs": scene_pairs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "epoch_train_samples": epoch_train_samples,
        "num_workers": effective_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(effective_workers > 0),
        "train_loader": train_loader,
        "train_loader_kwargs": loader_kwargs,
        "val_loader": val_loader,
        "model": model,
        "model_params": model_params,
        "bce_loss": nn.BCEWithLogitsLoss(),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "scene_checkpoint": scene_checkpoint,
        "history": {key: [] for key in (
            "train_loss", "val_loss", "train_iou", "val_iou", "val_selection_score",
            "val_precision", "val_recall", "val_dice", "val_count_mae", "val_count_exact_rate",
            "val_player_count_mae", "val_player_exact_rate", "val_scene_player_exact_rate",
            "val_pred_count_mean", "val_gt_count_mean", "val_player_count_mae_p1",
            "val_player_count_mae_p2", "val_player_count_mae_p3", "val_player_count_mae_p4",
            "lr", "epoch_seconds", "train_seconds", "val_seconds", "images_per_sec", "gpu_mem_gb",
        )},
        "best_val_iou": None,
        "best_epoch": None,
        "training_seconds": None,
    }


def run_segmenter_training(state: dict[str, Any]) -> dict[str, Any]:
    """Train SceneUNetSmall with the original BCE+Dice curriculum."""
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
    history: dict[str, list[Any]] = state["history"]
    scene_checkpoint: Path = state["scene_checkpoint"]
    epoch_train_samples = int(state.get("epoch_train_samples", len(train_loader.dataset)))

    weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    if cfg.train_loss_bce_weight < 0 or cfg.train_loss_dice_weight < 0 or weight_sum <= 0:
        raise ValueError("BCE/Dice loss weights must be non-negative and not both zero.")
    bce_weight = cfg.train_loss_bce_weight / weight_sum
    dice_weight = cfg.train_loss_dice_weight / weight_sum
    best_selection = -float("inf")
    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_start = time.perf_counter()

    coverage_indices: list[int] | None = None
    coverage_cursor = 0
    coverage_rng: random.Random | None = None
    if epoch_train_samples < len(train_ds):
        coverage_indices = list(range(len(train_ds)))
        coverage_rng = random.Random(cfg.seed)
        coverage_rng.shuffle(coverage_indices)

    print(
        f"\n[segmenter] epochs={cfg.epochs}, lr={cfg.learning_rate:.2e}, "
        f"dataset_samples={len(train_ds):,}, samples_per_epoch={epoch_train_samples:,}",
        flush=True,
    )
    print(
        f"  loss_bce/dice={bce_weight:.2f}/{dice_weight:.2f} | "
        f"selection={cfg.best_epoch_selection_metric} | amp={use_amp} | workers={state.get('num_workers', 0)}",
        flush=True,
    )
    if epoch_train_samples < len(train_ds):
        print(
            "  per-epoch train cap is active; the epoch sampler cycles through "
            "the shuffled full train pool before repeating samples.",
            flush=True,
        )

    last_train_loader = train_loader
    for epoch in range(1, cfg.epochs + 1):
        current_train_loader = train_loader
        if epoch_train_samples < len(train_ds):
            if coverage_indices is None or coverage_rng is None:
                raise RuntimeError("Coverage sampler state was not initialized.")

            selected_indices: list[int] = []
            while len(selected_indices) < epoch_train_samples:
                remaining_in_cycle = len(coverage_indices) - coverage_cursor
                to_take = min(epoch_train_samples - len(selected_indices), remaining_in_cycle)
                selected_indices.extend(coverage_indices[coverage_cursor:coverage_cursor + to_take])
                coverage_cursor += to_take
                if coverage_cursor >= len(coverage_indices):
                    coverage_rng.shuffle(coverage_indices)
                    coverage_cursor = 0

            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(cfg.seed + epoch)
            epoch_sampler = SubsetRandomSampler(selected_indices, generator=epoch_generator)
            current_train_loader = DataLoader(
                train_ds,
                shuffle=False,
                sampler=epoch_sampler,
                **train_loader_kwargs,
            )
            last_train_loader = current_train_loader

        train_start = time.perf_counter()
        train_loss, train_iou, n_train = _train_one_epoch(
            model,
            current_train_loader,
            optimizer,
            bce_loss,
            scaler,
            use_amp,
            device,
            bce_weight,
            dice_weight,
            cfg.log_every_batches,
            channels_last,
        )
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = _evaluate(
            model,
            val_loader,
            bce_loss,
            use_amp,
            device,
            bce_weight,
            dice_weight,
            cfg.eval_mask_threshold,
            cfg.eval_min_component_area,
        )
        val_seconds = time.perf_counter() - val_start
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
        selection = _selection_score(val_metrics, cfg)

        scheduler.step(val_iou)
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = train_seconds + val_seconds
        images_per_sec = n_train / max(train_seconds, 1e-9)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0

        improved = selection > best_selection
        if improved:
            best_selection = selection
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), scene_checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["val_selection_score"].append(float(selection))
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

        recent_times = history["epoch_seconds"][-5:]
        eta_seconds = float(np.mean(recent_times) * max(cfg.epochs - epoch, 0))
        status = "best checkpoint saved" if improved else "no improvement"
        if cfg.early_stopping_patience is not None and not improved:
            status = f"{status} ({epochs_without_improvement}/{cfg.early_stopping_patience})"

        print(
            f"  epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} | "
            f"precision={val_precision:.4f} recall={val_recall:.4f} | "
            f"selection({cfg.best_epoch_selection_metric})={selection:.4f} | lr={lr_now:.2e}",
            flush=True,
        )
        print(
            f"    counts | cards(pred/gt)={val_pred_count_mean:.2f}/{val_gt_count_mean:.2f} "
            f"count_mae={val_count_mae:.3f} exact={val_count_exact_rate:.3f} | "
            f"player_mae={val_player_count_mae:.3f} player_exact={val_player_exact_rate:.3f} "
            f"scene_player_exact={val_scene_player_exact_rate:.3f}",
            flush=True,
        )
        print(
            f"    times  | train={_format_seconds(train_seconds)} val={_format_seconds(val_seconds)} "
            f"epoch={_format_seconds(epoch_seconds)} | speed={images_per_sec:.1f} img/s | "
            f"gpu_mem={gpu_mem_gb:.2f} GB | eta={_format_seconds(eta_seconds)} | {status}",
            flush=True,
        )
        if cfg.early_stopping_patience is not None and epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    state["best_val_iou"] = best_val_iou
    state["best_selection_score"] = best_selection
    state["best_selection_metric"] = str(cfg.best_epoch_selection_metric)
    state["best_epoch"] = best_epoch
    state["training_seconds"] = time.perf_counter() - global_start
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
    _shutdown_loader_workers(train_loader)
    if last_train_loader is not train_loader:
        _shutdown_loader_workers(last_train_loader)
    _shutdown_loader_workers(val_loader)
    print(f"Saved best segmenter checkpoint: {scene_checkpoint}")
    print(f"Best epoch {best_epoch}: val_iou={best_val_iou:.4f}, selection={best_selection:.4f}")
    return state


def release_segmenter_training_resources(state: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy training objects after plots have consumed the history."""
    for key in ("train_loader", "val_loader"):
        _shutdown_loader_workers(state.get(key))
    for key in (
        "train_loader",
        "val_loader",
        "train_ds",
        "val_ds",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "bce_loss",
    ):
        state.pop(key, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state["resources_released"] = True
    print("Released segmenter training loaders, cached datasets, and model state from RAM.")
    return state


@torch.no_grad()
def evaluate_segmenter_validation(state: dict[str, Any], load_best_checkpoint: bool = True) -> dict[str, float]:
    cfg: SegmenterPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    if load_best_checkpoint and Path(state["scene_checkpoint"]).is_file():
        model.load_state_dict(torch.load(state["scene_checkpoint"], map_location=state["device"]), strict=False)
    weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    metrics = _evaluate(
        model,
        state["val_loader"],
        state["bce_loss"],
        state["use_amp"],
        state["device"],
        cfg.train_loss_bce_weight / weight_sum,
        cfg.train_loss_dice_weight / weight_sum,
        cfg.eval_mask_threshold,
        cfg.eval_min_component_area,
    )
    metrics["selection_score"] = _selection_score(metrics, cfg)
    print(
        f"Validation | loss={metrics['loss']:.4f} iou={metrics['iou']:.4f} dice={metrics['dice']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} selection={metrics['selection_score']:.4f}"
    )
    return metrics


def run_full_segmenter_training(
    config: SegmenterPipelineConfig | None = None,
    project_root: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    state = initialize_segmenter_pipeline(config, project_root=project_root, models_dir=models_dir)
    return run_segmenter_training(state)


def plot_training_curves(state: dict[str, Any]) -> None:
    history = state.get("history", {})
    if not history or not history.get("train_loss"):
        print("No segmenter training history to plot.")
        return
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    eps = 1e-6

    train_loss = np.clip(np.asarray(history["train_loss"], dtype=np.float64), eps, None)
    val_loss = np.clip(np.asarray(history["val_loss"], dtype=np.float64), eps, None)
    train_iou_gap = np.clip(1.0 - np.asarray(history["train_iou"], dtype=np.float64), eps, 1.0)
    val_iou_gap = np.clip(1.0 - np.asarray(history["val_iou"], dtype=np.float64), eps, 1.0)
    val_dice_gap = np.clip(1.0 - np.asarray(history["val_dice"], dtype=np.float64), eps, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss (log scale)")
    axes[0].legend()
    axes[1].plot(epochs, train_iou_gap, label="1 - train IoU")
    axes[1].plot(epochs, val_iou_gap, label="1 - val IoU")
    axes[1].plot(epochs, val_dice_gap, label="1 - val Dice")
    axes[1].set_yscale("log")
    axes[1].set_title("Overlap gap to 1.0 (log scale)")
    axes[1].set_ylabel("1 - metric")
    axes[2].plot(epochs, history["val_count_mae"], label="count MAE")
    axes[2].plot(epochs, history["val_player_count_mae"], label="player MAE")
    axes[2].set_title("Count errors")
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def _training_path(project_root: Path, stored_path: str | Path) -> Path:
    parts = Path(str(stored_path).replace("\\", "/")).parts
    if "training_data" in parts:
        return project_root.joinpath(*parts[parts.index("training_data"):])
    return project_root / Path(*parts)


def _mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    union = tp + fp + fn
    return {
        "iou": 1.0 if union == 0 else tp / union,
        "dice": 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn),
        "precision": 1.0 if (tp + fp) == 0 else tp / (tp + fp),
        "recall": 1.0 if (tp + fn) == 0 else tp / (tp + fn),
    }


def evaluate_segmenter_checkpoint(
    engine: InferenceEngine,
    project_root: Path,
    sample_count: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the trained segmenter on a small deterministic augmented-scene sample."""
    project_root = Path(project_root)
    records = json.loads((project_root / "training_data" / "object_labels" / "augmented_scenes.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=min(sample_count, len(records)), replace=False)

    rows: list[dict[str, float | str | int]] = []
    examples: list[dict[str, Any]] = []
    for index in sorted(int(value) for value in indices):
        record = records[index]
        image_path = _training_path(project_root, record["image_path"])
        mask_path = _training_path(project_root, record["mask_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask_u8 is None:
            continue

        probability = segment_scene_probability(
            image_bgr,
            engine.segmenter,
            engine.device,
            target_size=engine.config.segmenter_img_size,
        )
        pred_mask = probability > engine.config.segmenter_threshold
        target_mask = mask_u8 > 0
        metrics = _mask_metrics(pred_mask, target_mask)
        boxes, _ = boxes_from_probability(
            probability,
            threshold=engine.config.segmenter_threshold,
            min_component_area=engine.config.segmenter_min_component_area,
            instance_mask_growth_px=engine.config.instance_mask_growth_px,
        )
        rows.append({
            "scene": record["scene"],
            **metrics,
            "true_cards": len(record.get("cards", [])),
            "pred_cards": len(boxes),
            "foreground_ratio_true": float(target_mask.mean()),
            "foreground_ratio_pred": float(pred_mask.mean()),
        })
        if len(examples) < 3:
            examples.append({
                "scene": record["scene"],
                "image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                "target_mask": target_mask.astype(np.uint8),
                "probability": probability,
                "pred_mask": pred_mask.astype(np.uint8),
                "boxes": boxes,
                "true_cards": len(record.get("cards", [])),
                "pred_cards": len(boxes),
            })

    return {
        "rows": rows,
        "examples": examples,
        "threshold": engine.config.segmenter_threshold,
        "sample_count": len(rows),
    }


def print_segmenter_summary(result: dict[str, Any]) -> None:
    rows = result["rows"]
    metrics = {metric: np.mean([float(row[metric]) for row in rows]) for metric in ("iou", "dice", "precision", "recall")}
    count_error = [abs(int(row["pred_cards"]) - int(row["true_cards"])) for row in rows]
    print(
        f"Segmenter audit ({len(rows)} scenes): "
        f"IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}, "
        f"precision/recall={metrics['precision']:.3f}/{metrics['recall']:.3f}, "
        f"count MAE={np.mean(count_error):.2f} cards."
    )


def _annotate_bars(axis: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in axis.patches:
        height = float(patch.get_height())
        if height <= 0:
            continue
        axis.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )


def plot_segmenter_audit(result: dict[str, Any], max_examples: int = 2) -> None:
    """One compact figure for mask quality, component counts, and examples."""
    rows = result["rows"]
    if not rows:
        print("No segmenter rows to plot.")
        return

    examples = list(result.get("examples", []))[: max(0, max_examples)]
    fig, axes = plt.subplots(
        1 + len(examples),
        4,
        figsize=(17, 4 + 3.2 * len(examples)),
        gridspec_kw={"height_ratios": [1.0] + [1.35] * len(examples)},
        squeeze=False,
    )

    metric_names = ["iou", "dice", "precision", "recall"]
    metric_values = [np.mean([float(row[name]) for row in rows]) for name in metric_names]
    axes[0, 0].bar([name.title() for name in metric_names], metric_values, color="#4e79a7")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Mean mask quality")
    _annotate_bars(axes[0, 0])

    true_counts = [int(row["true_cards"]) for row in rows]
    pred_counts = [int(row["pred_cards"]) for row in rows]
    axes[0, 1].scatter(true_counts, pred_counts, color="#f28e2b", alpha=0.8)
    limit = max(true_counts + pred_counts + [1])
    axes[0, 1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("True cards")
    axes[0, 1].set_ylabel("Predicted components")
    axes[0, 1].set_title("Component count")

    count_errors = np.array(pred_counts) - np.array(true_counts)
    bins = np.arange(count_errors.min() - 0.5, count_errors.max() + 1.5, 1)
    axes[0, 2].hist(count_errors, bins=bins, color="#e15759", edgecolor="white")
    axes[0, 2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].set_title("Signed count error")
    axes[0, 2].set_xlabel("Predicted - true cards")
    axes[0, 2].set_ylabel("Scenes")

    true_fg = [float(row["foreground_ratio_true"]) for row in rows]
    pred_fg = [float(row["foreground_ratio_pred"]) for row in rows]
    axes[0, 3].scatter(true_fg, pred_fg, color="#76b7b2", alpha=0.8)
    fg_limit = max(true_fg + pred_fg + [0.01])
    axes[0, 3].plot([0, fg_limit], [0, fg_limit], color="black", linestyle="--", linewidth=1)
    axes[0, 3].set_xlabel("True foreground ratio")
    axes[0, 3].set_ylabel("Predicted foreground ratio")
    axes[0, 3].set_title("Foreground area")

    for row_index, example in enumerate(examples, start=1):
        axes[row_index, 0].imshow(example["image_rgb"])
        for x0, y0, x1, y1 in example["boxes"]:
            axes[row_index, 0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.4))
        axes[row_index, 0].set_title(
            f"{example['scene']} | true={example['true_cards']} pred={example['pred_cards']}",
            fontsize=9,
        )
        axes[row_index, 1].imshow(example["target_mask"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Target mask", fontsize=9)
        axes[row_index, 2].imshow(example["probability"], cmap="viridis", vmin=0, vmax=1)
        axes[row_index, 2].set_title(f"Probability (threshold={result['threshold']:.2f})", fontsize=9)
        axes[row_index, 3].imshow(example["pred_mask"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].set_title("Thresholded mask", fontsize=9)
        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle("Segmenter checkpoint audit", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_segmenter_summary(result: dict[str, Any]) -> None:
    rows = result["rows"]
    if not rows:
        print("No segmenter rows to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metric_names = ["iou", "dice", "precision", "recall"]
    axes[0].bar(metric_names, [np.mean([float(row[name]) for row in rows]) for name in metric_names], color="#4e79a7")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean mask metrics")

    true_counts = [int(row["true_cards"]) for row in rows]
    pred_counts = [int(row["pred_cards"]) for row in rows]
    axes[1].scatter(true_counts, pred_counts, color="#f28e2b", alpha=0.8)
    limit = max(true_counts + pred_counts + [1])
    axes[1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("True cards")
    axes[1].set_ylabel("Predicted components")
    axes[1].set_title("Component-count sanity check")

    plt.tight_layout()
    plt.show()


def plot_segmenter_examples(result: dict[str, Any]) -> None:
    examples = result.get("examples", [])
    if not examples:
        print("No segmenter examples to plot.")
        return

    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)), squeeze=False)
    for row_index, example in enumerate(examples):
        axes[row_index, 0].imshow(example["image_rgb"])
        axes[row_index, 0].set_title(example["scene"])
        axes[row_index, 1].imshow(example["target_mask"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Target mask")
        axes[row_index, 2].imshow(example["probability"], cmap="viridis", vmin=0, vmax=1)
        axes[row_index, 2].set_title("Predicted probability")
        axes[row_index, 3].imshow(example["pred_mask"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].set_title("Thresholded prediction")
        for axis in axes[row_index]:
            axis.axis("off")

    plt.tight_layout()
    plt.show()
