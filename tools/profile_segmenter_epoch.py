"""Profile one scene-segmenter training epoch without saving checkpoints."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.segmenter_training import (  # noqa: E402
    SegmenterPipelineConfig,
    _count_metrics_from_logits,
    _evaluate,
    _selection_score,
    dice_loss_from_logits,
    initialize_segmenter_pipeline,
    overlap_metrics_from_logits,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _profile_train_epoch(state: dict[str, Any], max_batches: int | None) -> dict[str, float]:
    cfg: SegmenterPipelineConfig = state["config"]
    device: torch.device = state["device"]
    model: nn.Module = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]
    bce_loss: nn.Module = state["bce_loss"]
    scaler = state["scaler"]
    use_amp: bool = state["use_amp"]
    channels_last: bool = state["channels_last"]
    loader = state["train_loader"]

    weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    bce_weight = cfg.train_loss_bce_weight / weight_sum
    dice_weight = cfg.train_loss_dice_weight / weight_sum

    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_samples = 0
    batches = 0

    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_create_seconds = time.perf_counter() - iterator_start

    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    forward_loss_seconds = 0.0
    backward_step_seconds = 0.0
    metric_seconds = 0.0
    first_batch_wait_seconds = 0.0
    total_start = time.perf_counter()

    while max_batches is None or batches < max_batches:
        data_start = time.perf_counter()
        try:
            images, masks = next(iterator)
        except StopIteration:
            break
        data_elapsed = time.perf_counter() - data_start
        data_wait_seconds += data_elapsed
        if batches == 0:
            first_batch_wait_seconds = data_elapsed

        transfer_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        _sync(device)
        transfer_seconds += time.perf_counter() - transfer_start

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = bce_weight * bce_loss(logits, masks) + dice_weight * dice_loss_from_logits(logits, masks)
        _sync(device)
        forward_loss_seconds += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _sync(device)
        backward_step_seconds += time.perf_counter() - backward_start

        metric_start = time.perf_counter()
        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_iou += overlap_metrics_from_logits(logits.detach(), masks, threshold=0.5)["iou"] * batch_size
        _sync(device)
        metric_seconds += time.perf_counter() - metric_start
        n_samples += batch_size
        batches += 1

    total_seconds = time.perf_counter() - total_start
    return {
        "samples": float(n_samples),
        "batches": float(batches),
        "iterator_create_seconds": iterator_create_seconds,
        "first_batch_wait_seconds": first_batch_wait_seconds,
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "forward_loss_seconds": forward_loss_seconds,
        "backward_step_seconds": backward_step_seconds,
        "metric_seconds": metric_seconds,
        "total_loop_seconds": total_seconds,
        "train_loss": total_loss / max(n_samples, 1),
        "train_iou": total_iou / max(n_samples, 1),
    }


@torch.no_grad()
def _profile_validation(state: dict[str, Any], max_batches: int | None) -> dict[str, float]:
    cfg: SegmenterPipelineConfig = state["config"]
    device: torch.device = state["device"]
    model: nn.Module = state["model"]
    bce_loss: nn.Module = state["bce_loss"]
    use_amp: bool = state["use_amp"]
    loader = state["val_loader"]

    weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    bce_weight = cfg.train_loss_bce_weight / weight_sum
    dice_weight = cfg.train_loss_dice_weight / weight_sum

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
    batches = 0

    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_create_seconds = time.perf_counter() - iterator_start

    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    forward_loss_seconds = 0.0
    overlap_metric_seconds = 0.0
    count_metric_seconds = 0.0
    first_batch_wait_seconds = 0.0
    total_start = time.perf_counter()

    while max_batches is None or batches < max_batches:
        data_start = time.perf_counter()
        try:
            images, masks = next(iterator)
        except StopIteration:
            break
        data_elapsed = time.perf_counter() - data_start
        data_wait_seconds += data_elapsed
        if batches == 0:
            first_batch_wait_seconds = data_elapsed

        transfer_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        _sync(device)
        transfer_seconds += time.perf_counter() - transfer_start

        forward_start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = bce_weight * bce_loss(logits, masks) + dice_weight * dice_loss_from_logits(logits, masks)
        _sync(device)
        forward_loss_seconds += time.perf_counter() - forward_start

        batch_size = int(images.shape[0])
        overlap_start = time.perf_counter()
        metrics = overlap_metrics_from_logits(logits, masks, threshold=cfg.eval_mask_threshold)
        _sync(device)
        overlap_metric_seconds += time.perf_counter() - overlap_start

        count_start = time.perf_counter()
        counts = _count_metrics_from_logits(logits, masks, threshold=cfg.eval_mask_threshold, min_component_area=cfg.eval_min_component_area)
        count_metric_seconds += time.perf_counter() - count_start

        totals["loss"] += float(loss.item()) * batch_size
        for key in ("iou", "dice", "precision", "recall"):
            totals[key] += metrics[key] * batch_size
        for key, value in counts.items():
            count_totals[key] += float(value) * batch_size
        n_samples += batch_size
        batches += 1

    total_seconds = time.perf_counter() - total_start
    metrics = {key: value / max(n_samples, 1) for key, value in totals.items()}
    metrics.update({key: value / max(n_samples, 1) for key, value in count_totals.items()})
    return {
        "samples": float(n_samples),
        "batches": float(batches),
        "iterator_create_seconds": iterator_create_seconds,
        "first_batch_wait_seconds": first_batch_wait_seconds,
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "forward_loss_seconds": forward_loss_seconds,
        "overlap_metric_seconds": overlap_metric_seconds,
        "count_metric_seconds": count_metric_seconds,
        "total_loop_seconds": total_seconds,
        "val_loss": float(metrics["loss"]),
        "val_iou": float(metrics["iou"]),
        "selection_score": float(_selection_score(metrics, cfg)),
    }


def _print_timing(title: str, timings: dict[str, float]) -> None:
    print(f"\n{title}")
    total = timings.get("total_loop_seconds", 0.0)
    for key, value in timings.items():
        if key in {"train_loss", "train_iou", "val_loss", "val_iou", "selection_score"}:
            continue
        if key in {"samples", "batches"}:
            print(f"  {key:28s} {int(value):8d}")
        else:
            pct = (value / total * 100.0) if total > 0 and key.endswith("seconds") else 0.0
            print(f"  {key:28s} {value:8.3f}s {pct:6.1f}%")
    metric_keys = [key for key in ("train_loss", "train_iou", "val_loss", "val_iou", "selection_score") if key in timings]
    if metric_keys:
        print("  metrics " + ", ".join(f"{key}={timings[key]:.4f}" for key in metric_keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-scene-pairs", type=int, default=None)
    parser.add_argument("--train-batches", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-in-ram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--standard-evaluate", action="store_true")
    parser.add_argument("--repeat-train-epochs", type=int, default=1)
    parser.add_argument("--repeat-validation", type=int, default=1)
    args = parser.parse_args()

    cfg = SegmenterPipelineConfig(
        epoch_max_train_samples=args.samples,
        max_scene_pairs=args.max_scene_pairs,
        batch_size=args.batch_size,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers,
        epochs=1,
    )
    init_start = time.perf_counter()
    state = initialize_segmenter_pipeline(cfg, project_root=REPO_ROOT, models_dir=REPO_ROOT / "models")
    init_seconds = time.perf_counter() - init_start
    print(
        f"\nProfile setup: requested_samples={args.samples}, train_samples={state['epoch_train_samples']}, "
        f"train_pairs={len(state['train_pairs'])}, val_pairs={len(state['val_pairs'])}, "
        f"workers={state['num_workers']}, batch_size={cfg.batch_size}, cache_in_ram={cfg.cache_in_ram}, device={state['device']}"
    )
    print(f"Initialization time: {init_seconds:.3f}s")

    for train_index in range(1, max(1, args.repeat_train_epochs) + 1):
        train_timings = _profile_train_epoch(state, args.train_batches)
        _print_timing(f"Train epoch timing (pass {train_index})", train_timings)

    if not args.skip_validation:
        if args.standard_evaluate:
            start = time.perf_counter()
            metrics = _evaluate(
                state["model"],
                state["val_loader"],
                state["bce_loss"],
                state["use_amp"],
                state["device"],
                cfg.train_loss_bce_weight / (cfg.train_loss_bce_weight + cfg.train_loss_dice_weight),
                cfg.train_loss_dice_weight / (cfg.train_loss_bce_weight + cfg.train_loss_dice_weight),
                cfg.eval_mask_threshold,
                cfg.eval_min_component_area,
            )
            print(f"\nStandard _evaluate total: {time.perf_counter() - start:.3f}s | val_iou={metrics['iou']:.4f}")
        for validation_index in range(1, max(1, args.repeat_validation) + 1):
            val_timings = _profile_validation(state, args.val_batches)
            _print_timing(f"Validation timing (pass {validation_index})", val_timings)


if __name__ == "__main__":
    main()