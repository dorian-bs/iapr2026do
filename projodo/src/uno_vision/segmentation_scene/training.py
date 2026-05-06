"""Training loop for the scene-level foreground segmenter."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uno_vision.paths import SEGMENTER_MODELS_DIR
from uno_vision.segmentation_scene.data import IMG_SIZE, SceneSegDataset, collect_scene_pairs
from uno_vision.segmentation_scene.model import (
    UNetSmall,
    assert_parameter_budget,
    dice_loss_from_logits,
    iou_sum_from_logits,
)


def set_training_seed(seed: int) -> None:
    """Set python, numpy, and torch RNG seeds for reproducible training splits."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
class SceneSegmentationTrainingHistory:
    """Epoch metrics and artifact locations returned after scene segmenter training."""

    train_losses: list[float]
    val_losses: list[float]
    train_ious: list[float]
    val_ious: list[float]
    learning_rates: list[float]
    epoch_times: list[float]
    model_path: Path
    best_epoch: int
    best_val_iou: float
    train_pairs: list[tuple[str, str]]
    val_pairs: list[tuple[str, str]]
    device: str


def train_scene_segmenter(
    pairs: list[tuple[str, str]] | None = None,
    epochs: int = 15,
    batch_size: int = 8,
    val_size: float = 0.15,
    num_workers: int | None = None,
    mixed_precision: bool = True,
    preload_to_ram: bool = True,
    train_augment: bool = True,
    image_size: int = IMG_SIZE,
    random_state: int = 42,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    scheduler_step_size: int = 5,
    scheduler_gamma: float = 0.5,
    warm_start_path: Path | None = None,
    output_path: Path = SEGMENTER_MODELS_DIR / "scene_segmenter_unet_small.pth",
    save_last_checkpoint: bool = False,
    logger: logging.Logger | None = None,
) -> SceneSegmentationTrainingHistory:
    """Train a scene segmenter on synthetic game snapshots and save the best checkpoint."""

    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in the open interval (0, 1).")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")

    training_logger = _resolve_training_logger(logger)
    set_training_seed(random_state)

    pairs = pairs or collect_scene_pairs()
    if not pairs:
        raise RuntimeError(
            "No scene image-mask pairs found in game snapshots. "
            "Generate scene snapshots first."
        )

    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_ds = SceneSegDataset(
        train_pairs,
        augment=train_augment,
        image_size=image_size,
        preload_to_ram=preload_to_ram,
        verbose=True,
    )
    val_ds = SceneSegDataset(
        val_pairs,
        augment=False,
        image_size=image_size,
        preload_to_ram=preload_to_ram,
        verbose=True,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Scene dataset is empty after preprocessing. "
            "Verify scene image-mask files and formats."
        )

    train_pairs = list(train_ds.pairs)
    val_pairs = list(val_ds.pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

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
    params = assert_parameter_budget(model)
    training_logger.info("[compliance] UNetSmall: %s trainable params", f"{params:,}")

    if warm_start_path is not None:
        warm_start_path = Path(warm_start_path)
        if not warm_start_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start_path}")
        model.load_state_dict(torch.load(warm_start_path, map_location=device))
        training_logger.info("Loaded warm-start checkpoint: %s", warm_start_path)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    amp_enabled = mixed_precision and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_ious: list[float] = []
    val_ious: list[float] = []
    learning_rates: list[float] = []
    epoch_times: list[float] = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_logger.info(
        "Scene segmenter training start | device=%s | train_samples=%d | val_samples=%d | "
        "batch_size=%d | num_workers=%d | amp=%s",
        device,
        len(train_ds),
        len(val_ds),
        batch_size,
        worker_count,
        amp_enabled,
    )

    best_epoch = 0
    best_val_iou = float("-inf")
    total_start = perf_counter()

    for epoch_idx in range(1, epochs + 1):
        epoch_start = perf_counter()

        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_iou_sum = torch.zeros((), device=device)
        for images, masks in train_loader:
            images = images.to(device, non_blocking=use_cuda)
            masks = masks.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                # BCE stabilizes per-pixel classification while Dice encourages mask overlap.
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.detach() * images.size(0)
            train_iou_sum += iou_sum_from_logits(logits.detach(), masks)

        train_loss = float((train_loss_sum / len(train_ds)).item())
        train_iou = float((train_iou_sum / len(train_ds)).item())

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_iou_sum = torch.zeros((), device=device)
        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=use_cuda)
                masks = masks.to(device, non_blocking=use_cuda)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
                val_loss_sum += loss * images.size(0)
                val_iou_sum += iou_sum_from_logits(logits, masks)

        val_loss = float((val_loss_sum / len(val_ds)).item())
        val_iou = float((val_iou_sum / len(val_ds)).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

        epoch_time = perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch_idx
            torch.save(model.state_dict(), output_path)

        training_logger.info(
            "Scene epoch %d/%d | lr=%.2e | train_loss=%.4f | val_loss=%.4f | "
            "train_iou=%.4f | val_iou=%.4f | best_val_iou=%.4f (epoch %d) | %.2fs",
            epoch_idx,
            epochs,
            learning_rates[-1],
            train_loss,
            val_loss,
            train_iou,
            val_iou,
            best_val_iou,
            best_epoch,
            epoch_time,
        )

        scheduler.step()

    if save_last_checkpoint:
        last_path = output_path.with_name(f"{output_path.stem}_last{output_path.suffix}")
        torch.save(model.state_dict(), last_path)
        training_logger.info("Saved final-epoch checkpoint: %s", last_path)

    total_time = perf_counter() - total_start
    training_logger.info(
        "Scene segmenter training complete | total=%.2fs | avg_epoch=%.2fs | best_model=%s",
        total_time,
        sum(epoch_times) / len(epoch_times),
        output_path,
    )

    return SceneSegmentationTrainingHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        train_ious=train_ious,
        val_ious=val_ious,
        learning_rates=learning_rates,
        epoch_times=epoch_times,
        model_path=output_path,
        best_epoch=best_epoch,
        best_val_iou=best_val_iou,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        device=str(device),
    )
