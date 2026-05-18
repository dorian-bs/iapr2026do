"""Profile one classifier-training epoch without saving checkpoints."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier_training import (  # noqa: E402
    ClassifierPipelineConfig,
    _classification_metrics,
    _shutdown_loader_workers,
    _stage_samples_per_epoch,
    _selection_score_from_metrics,
    initialize_training_pipeline,
    make_loader,
    make_rotating_loader,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _profile_train_epoch(
    state: dict[str, Any],
    samples: list[Any],
    stage_name: str,
    samples_per_epoch: int,
    workers: int,
) -> dict[str, float]:
    cfg: ClassifierPipelineConfig = state["config"]
    device: torch.device = state["device"]
    model: nn.Module = state["model"]
    ce_loss: nn.Module = state["ce_loss"]
    use_amp: bool = state["use_amp"]
    scaler = state["scaler"]

    rng = random.Random(cfg.seed + 99_001)
    selected_samples = list(samples)
    if samples_per_epoch < len(selected_samples):
        selected_samples = rng.sample(selected_samples, k=samples_per_epoch)

    loader_start = time.perf_counter()
    loader, _ = make_loader(
        selected_samples,
        state["label_to_index"],
        state["batch_size"],
        cfg.img_size,
        cfg.bbox_margin,
        cfg.mask_threshold,
        shuffle=True,
        augment=True,
        pin_memory=device.type == "cuda",
        num_workers=workers,
        seed=cfg.seed + 99_123,
        balanced=cfg.balanced_sampling,
        samples_per_epoch=None,
        persistent_workers=cfg.persistent_workers,
    )
    loader_create_seconds = time.perf_counter() - loader_start

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.stage_2_lr, weight_decay=cfg.weight_decay)
    model.train()

    loss_sum = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    top3_hits = 0
    confidences: list[float] = []

    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_create_seconds = time.perf_counter() - iterator_start

    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    metric_seconds = 0.0
    first_batch_wait_seconds = 0.0
    batches = 0

    total_start = time.perf_counter()
    try:
        while True:
            data_start = time.perf_counter()
            try:
                images, targets = next(iterator)
            except StopIteration:
                break
            data_elapsed = time.perf_counter() - data_start
            data_wait_seconds += data_elapsed
            if batches == 0:
                first_batch_wait_seconds = data_elapsed

            transfer_start = time.perf_counter()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            _sync(device)
            transfer_seconds += time.perf_counter() - transfer_start

            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = ce_loss(logits, targets)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            _sync(device)
            compute_seconds += time.perf_counter() - compute_start

            metric_start = time.perf_counter()
            probs = torch.softmax(logits.detach(), dim=1)
            pred = probs.argmax(dim=1)
            topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
            batch_size = int(targets.shape[0])
            loss_sum += float(loss.item()) * batch_size
            true_labels.extend(targets.cpu().tolist())
            pred_labels.extend(pred.cpu().tolist())
            top3_hits += int((topk == targets[:, None]).any(dim=1).sum().item())
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
            _sync(device)
            metric_seconds += time.perf_counter() - metric_start
            batches += 1
    finally:
        _shutdown_loader_workers(loader)

    total_seconds = time.perf_counter() - total_start
    metrics = _classification_metrics(loss_sum, len(true_labels), true_labels, pred_labels, top3_hits, confidences)
    return {
        "stage": stage_name,
        "samples": float(len(true_labels)),
        "batches": float(batches),
        "loader_create_seconds": loader_create_seconds,
        "iterator_create_seconds": iterator_create_seconds,
        "first_batch_wait_seconds": first_batch_wait_seconds,
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "compute_seconds": compute_seconds,
        "metric_seconds": metric_seconds,
        "total_loop_seconds": total_seconds,
        "train_loss": float(metrics["loss"]),
        "train_acc": float(metrics["cls_acc"]),
    }


def _profile_train_existing_loader(
    state: dict[str, Any],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    stage_name: str,
) -> dict[str, float]:
    cfg: ClassifierPipelineConfig = state["config"]
    device: torch.device = state["device"]
    model: nn.Module = state["model"]
    ce_loss: nn.Module = state["ce_loss"]
    use_amp: bool = state["use_amp"]
    scaler = state["scaler"]

    model.train()
    loss_sum = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    top3_hits = 0
    confidences: list[float] = []

    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_create_seconds = time.perf_counter() - iterator_start

    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    metric_seconds = 0.0
    first_batch_wait_seconds = 0.0
    batches = 0

    total_start = time.perf_counter()
    while True:
        data_start = time.perf_counter()
        try:
            images, targets = next(iterator)
        except StopIteration:
            break
        data_elapsed = time.perf_counter() - data_start
        data_wait_seconds += data_elapsed
        if batches == 0:
            first_batch_wait_seconds = data_elapsed

        transfer_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _sync(device)
        transfer_seconds += time.perf_counter() - transfer_start

        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, targets)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        _sync(device)
        compute_seconds += time.perf_counter() - compute_start

        metric_start = time.perf_counter()
        probs = torch.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
        batch_size = int(targets.shape[0])
        loss_sum += float(loss.item()) * batch_size
        true_labels.extend(targets.cpu().tolist())
        pred_labels.extend(pred.cpu().tolist())
        top3_hits += int((topk == targets[:, None]).any(dim=1).sum().item())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
        _sync(device)
        metric_seconds += time.perf_counter() - metric_start
        batches += 1

    total_seconds = time.perf_counter() - total_start
    metrics = _classification_metrics(loss_sum, len(true_labels), true_labels, pred_labels, top3_hits, confidences)
    return {
        "stage": stage_name,
        "samples": float(len(true_labels)),
        "batches": float(batches),
        "iterator_create_seconds": iterator_create_seconds,
        "first_batch_wait_seconds": first_batch_wait_seconds,
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "compute_seconds": compute_seconds,
        "metric_seconds": metric_seconds,
        "total_loop_seconds": total_seconds,
        "train_loss": float(metrics["loss"]),
        "train_acc": float(metrics["cls_acc"]),
    }


@torch.no_grad()
def _profile_validation(state: dict[str, Any]) -> dict[str, float]:
    cfg: ClassifierPipelineConfig = state["config"]
    device: torch.device = state["device"]
    model: nn.Module = state["model"]
    ce_loss: nn.Module = state["ce_loss"]
    use_amp: bool = state["use_amp"]
    loader = state["val_loader"]

    model.eval()
    loss_sum = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    top3_hits = 0
    confidences: list[float] = []
    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    metric_seconds = 0.0
    first_batch_wait_seconds = 0.0
    batches = 0
    total_start = time.perf_counter()

    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_create_seconds = time.perf_counter() - iterator_start
    while True:
        data_start = time.perf_counter()
        try:
            images, targets = next(iterator)
        except StopIteration:
            break
        data_elapsed = time.perf_counter() - data_start
        data_wait_seconds += data_elapsed
        if batches == 0:
            first_batch_wait_seconds = data_elapsed

        transfer_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _sync(device)
        transfer_seconds += time.perf_counter() - transfer_start

        compute_start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, targets)
        _sync(device)
        compute_seconds += time.perf_counter() - compute_start

        metric_start = time.perf_counter()
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
        batch_size = int(targets.shape[0])
        loss_sum += float(loss.item()) * batch_size
        true_labels.extend(targets.cpu().tolist())
        pred_labels.extend(pred.cpu().tolist())
        top3_hits += int((topk == targets[:, None]).any(dim=1).sum().item())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
        _sync(device)
        metric_seconds += time.perf_counter() - metric_start
        batches += 1

    total_seconds = time.perf_counter() - total_start
    metrics = _classification_metrics(loss_sum, len(true_labels), true_labels, pred_labels, top3_hits, confidences)
    score = _selection_score_from_metrics(metrics, cfg)
    return {
        "samples": float(len(true_labels)),
        "batches": float(batches),
        "iterator_create_seconds": iterator_create_seconds,
        "first_batch_wait_seconds": first_batch_wait_seconds,
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "compute_seconds": compute_seconds,
        "metric_seconds": metric_seconds,
        "total_loop_seconds": total_seconds,
        "val_loss": float(metrics["loss"]),
        "val_acc": float(metrics["cls_acc"]),
        "selection_score": float(score),
    }


def _print_timing(title: str, timings: dict[str, float]) -> None:
    print(f"\n{title}")
    total = timings.get("total_loop_seconds", 0.0)
    for key, value in timings.items():
        if key in {"stage", "train_loss", "train_acc", "val_loss", "val_acc", "selection_score"}:
            continue
        if key in {"samples", "batches"}:
            print(f"  {key:28s} {int(value):8d}")
        else:
            pct = (value / total * 100.0) if total > 0 and key.endswith("seconds") else 0.0
            print(f"  {key:28s} {value:8.3f}s {pct:6.1f}%")
    metric_keys = [key for key in ("train_loss", "train_acc", "val_loss", "val_acc", "selection_score") if key in timings]
    if metric_keys:
        print("  metrics " + ", ".join(f"{key}={timings[key]:.4f}" for key in metric_keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["augmented", "scene"], default="scene")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--repeat-train-epochs", type=int, default=1)
    parser.add_argument("--reuse-rotating-loader", action="store_true")
    parser.add_argument("--repeat-validation", type=int, default=1)
    args = parser.parse_args()

    cfg_kwargs: dict[str, Any] = {
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "epoch_max_train_samples": args.samples,
    }
    if args.batch_size is not None:
        cfg_kwargs.update(batch_size_cpu=args.batch_size, batch_size_cuda=args.batch_size, batch_size_mps=args.batch_size)
    cfg = ClassifierPipelineConfig(**cfg_kwargs)

    init_start = time.perf_counter()
    state = initialize_training_pipeline(cfg, project_root=REPO_ROOT, models_dir=REPO_ROOT / "models")
    init_seconds = time.perf_counter() - init_start
    device: torch.device = state["device"]
    samples = state["augmented_card_samples"] if args.stage == "augmented" else state["scene_train_samples"]
    stage_name = "augmented_card" if args.stage == "augmented" else "scene_manual"
    samples_per_epoch = min(args.samples, _stage_samples_per_epoch(len(samples), stage_name, state["batch_size"], device, cfg))

    print(f"\nProfile setup: stage={stage_name}, requested_samples={args.samples}, used_samples={samples_per_epoch}, workers={args.workers}, batch_size={state['batch_size']}, device={device}")
    print(f"Initialization time: {init_seconds:.3f}s")
    if args.reuse_rotating_loader:
        loader, _ = make_rotating_loader(
            samples,
            state["label_to_index"],
            state["batch_size"],
            cfg.img_size,
            cfg.bbox_margin,
            cfg.mask_threshold,
            augment=True,
            pin_memory=device.type == "cuda",
            num_workers=args.workers,
            seed=cfg.seed + 99_123,
            samples_per_epoch=samples_per_epoch,
            persistent_workers=cfg.persistent_workers,
            cache_scene_assets=cfg.cache_scene_assets,
            cache_max_scene_assets=cfg.cache_max_scene_assets,
        )
        optimizer = torch.optim.AdamW(state["model"].parameters(), lr=cfg.stage_2_lr, weight_decay=cfg.weight_decay)
        try:
            for epoch_index in range(1, max(1, args.repeat_train_epochs) + 1):
                train_timings = _profile_train_existing_loader(state, loader, optimizer, stage_name)
                _print_timing(f"Train epoch timing (persistent loader pass {epoch_index})", train_timings)
        finally:
            _shutdown_loader_workers(loader)
    else:
        train_timings = _profile_train_epoch(state, samples, stage_name, samples_per_epoch, args.workers)
        _print_timing("Train epoch timing", train_timings)
    if not args.skip_validation:
        for validation_index in range(1, max(1, args.repeat_validation) + 1):
            val_timings = _profile_validation(state)
            _print_timing(f"Validation timing (pass {validation_index})", val_timings)
    _shutdown_loader_workers(state.get("val_loader"))


if __name__ == "__main__":
    main()