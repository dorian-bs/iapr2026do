"""Training loop for the small card segmentation model."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uno_vision.paths import SEGMENTER_MODELS_DIR
from uno_vision.segmentation.data import SegDataset, collect_segmentation_pairs
from uno_vision.segmentation.model import UNetSmall, dice_loss_from_logits


def _iou_sum_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Accumulate per-sample IoU values for epoch-level reporting."""

    probs = torch.sigmoid(logits.float())
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).sum()


def _resolve_training_logger(logger: logging.Logger | None) -> logging.Logger:
    """Use a provided logger or create a minimal console logger for scripts."""

    if logger is not None:
        return logger
    resolved = logging.getLogger(__name__)
    resolved.setLevel(logging.INFO)
    if resolved.handlers or logging.getLogger().handlers:
        return resolved
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    resolved.addHandler(handler)
    resolved.propagate = False
    return resolved


@dataclass
class SegmentationTrainingHistory:
    """Loss, IoU, timing, and artifact path returned after segmenter training."""

    train_losses: list[float]
    val_losses: list[float]
    train_ious: list[float]
    val_ious: list[float]
    epoch_times: list[float]
    model_path: Path


def train_segmenter(
    pairs: list[tuple[str, str]] | None = None,
    epochs: int = 25,
    batch_size: int = 8,
    num_workers: int | None = None,
    mixed_precision: bool = True,
    logger: logging.Logger | None = None,
    random_state: int = 42,
    output_path: Path = SEGMENTER_MODELS_DIR / "segmenter_unet_small.pth",
) -> SegmentationTrainingHistory:
    """Train the segmenter on allowed reference and augmentation mask pairs."""

    training_logger = _resolve_training_logger(logger)
    pairs = pairs or collect_segmentation_pairs()
    if not pairs:
        raise RuntimeError("No valid image-mask pairs found.")
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=random_state, shuffle=True)
    train_ds = SegDataset(train_pairs, augment=True)
    val_ds = SegDataset(val_pairs, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Keep CPU runs simple while allowing parallel prefetching on CUDA machines.
    worker_count = num_workers
    if worker_count is None:
        worker_count = min(4, max(1, (os.cpu_count() or 1) // 2)) if use_cuda else 0
    if worker_count < 0:
        raise ValueError("num_workers must be non-negative.")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": worker_count,
        "pin_memory": use_cuda,
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = UNetSmall().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Automatic mixed precision is only useful and supported here for CUDA training.
    amp_enabled = mixed_precision and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_ious: list[float] = []
    val_ious: list[float] = []
    epoch_times: list[float] = []
    training_logger.info(
        "Segmenter training start | device=%s | train_samples=%d | val_samples=%d | batch_size=%d | num_workers=%d | amp=%s",
        device,
        len(train_ds),
        len(val_ds),
        batch_size,
        worker_count,
        amp_enabled,
    )
    training_start = perf_counter()
    for epoch_idx in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_iou_sum = torch.zeros((), device=device)
        train_start = perf_counter()
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=use_cuda)
            masks = masks.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(imgs)
                # BCE stabilizes pixel decisions while Dice rewards overlap of full card masks.
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.detach() * imgs.size(0)
            train_iou_sum += _iou_sum_from_logits(logits.detach(), masks)
        train_loss = float((train_loss_sum / len(train_ds)).item())
        train_iou = float((train_iou_sum / len(train_ds)).item())
        train_time = perf_counter() - train_start

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_iou_sum = torch.zeros((), device=device)
        val_start = perf_counter()
        with torch.inference_mode():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=use_cuda)
                masks = masks.to(device, non_blocking=use_cuda)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(imgs)
                    loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
                val_loss_sum += loss * imgs.size(0)
                val_iou_sum += _iou_sum_from_logits(logits, masks)
        val_loss = float((val_loss_sum / len(val_ds)).item())
        val_iou = float((val_iou_sum / len(val_ds)).item())
        val_time = perf_counter() - val_start
        epoch_time = perf_counter() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        epoch_times.append(epoch_time)
        training_logger.info(
            "Segmenter epoch %d/%d | train=%.2fs | val=%.2fs | total=%.2fs | train_loss=%.4f | val_loss=%.4f | train_iou=%.4f | val_iou=%.4f",
            epoch_idx,
            epochs,
            train_time,
            val_time,
            epoch_time,
            train_loss,
            val_loss,
            train_iou,
            val_iou,
        )
        scheduler.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    total_time = perf_counter() - training_start
    training_logger.info(
        "Segmenter training complete | total=%.2fs | avg_epoch=%.2fs | model=%s",
        total_time,
        sum(epoch_times) / len(epoch_times),
        output_path,
    )
    return SegmentationTrainingHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        train_ious=train_ious,
        val_ious=val_ious,
        epoch_times=epoch_times,
        model_path=output_path,
    )
