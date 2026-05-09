"""Training orchestrator for the masked card classifier.

Public surface preserved for the existing notebook:
    TrainPipelineConfig
    initialize_training_pipeline(config) -> state
    run_training(state) -> state
    save_training_artifacts(state) -> state
    run_validation_diagnostics(state) -> state
    plot_stage_preview(state)
    plot_wrong_predictions(state)

Removed (dead defensive scaffolding): `repair_validation_diagnostics` and
`debug_validation_arrays`. Validation alignment is guaranteed by the
shuffle=False, sampler=None val loader and a deterministic dataset, so these
helpers were patching a phantom bug.
"""
from __future__ import annotations

import json
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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.shared.card_models import assert_param_cap, build_card_classifier
from src.shared.card_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    find_workspace_root,
    letterbox_image_and_mask,
    compose_masked_card_image,
    card_input_to_tensor,
)
from src.shared.card_data import (
    CardSample,
    assert_no_test_inputs,
    augment_card_image_and_mask,
    build_scene_probability_cache,
    derive_scene_predicted_samples,
    load_augmented_card_samples,
    load_reference_samples,
    load_scene_manual_samples,
    make_loader,
    sample_to_crop_and_mask,
)
from src.shared.model_paths import (
    get_classifier_output_bundle_dir,
    resolve_segmenter_checkpoint,
)


@dataclass
class TrainPipelineConfig:
    seed: int = 42
    img_size: int = 160
    segmenter_img_size: int = 256
    bbox_margin: float = 0.08
    mask_threshold: float = 0.50
    val_split: float = 0.20

    stage_1_epochs: int = 4
    stage_2_epochs: int = 0
    stage_3_epochs: int = 15
    stage_4_epochs: int = 15

    stage_1_lr: float = 1e-3
    stage_2_lr: float = 1e-3
    stage_3_lr: float = 3e-4
    stage_4_lr: float = 2e-4

    weight_decay: float = 1e-4
    label_smoothing: float = 0.05

    balanced_sampling: bool = True
    early_stop_patience: int = 3
    min_epochs_per_stage: int = 2
    epoch_max_train_samples: int | None = None

    batch_size_cuda: int = 32
    batch_size_mps: int = 16
    batch_size_cpu: int = 8
    num_workers: int = 0

    reference_target_gpu_mps: int = 1536
    reference_target_cpu: int = 512
    augmented_target_gpu_mps: int = 4096
    augmented_target_cpu: int = 1536
    scene_target_gpu_mps: int = 4096
    scene_target_cpu: int = 1536

    scene_finetune_freeze_bn_stats: bool = True
    grad_clip_norm: float = 1.0

    preview_per_stage: int = 3

    classifier_architecture: str = "resnet18_small"
    classifier_stem_width: int = 60
    classifier_dropout: float = 0.20


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_batch_size(device: torch.device, cfg: TrainPipelineConfig) -> int:
    if device.type == "cuda":
        return cfg.batch_size_cuda
    if device.type == "mps":
        return cfg.batch_size_mps
    return cfg.batch_size_cpu


def _stage_samples_per_epoch(
    n_samples: int, stage: str, batch_size: int, device: torch.device, cfg: TrainPipelineConfig
) -> int:
    on_accel = device.type in {"cuda", "mps"}
    if stage == "reference":
        target = cfg.reference_target_gpu_mps if on_accel else cfg.reference_target_cpu
    elif stage == "augmented_card":
        target = cfg.augmented_target_gpu_mps if on_accel else cfg.augmented_target_cpu
    elif stage in {"scene_manual", "scene_predicted"}:
        target = cfg.scene_target_gpu_mps if on_accel else cfg.scene_target_cpu
    else:
        target = n_samples

    planned = min(int(n_samples), int(target))
    planned = (planned // batch_size) * batch_size
    if planned <= 0:
        planned = min(int(n_samples), int(batch_size))
    return planned


def initialize_training_pipeline(config: TrainPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or TrainPipelineConfig()
    _seed_everything(cfg.seed)

    project_root = find_workspace_root()
    project_dir = project_root / "project"
    training_data = project_dir / "training_data"
    models_dir = project_dir / "models"
    classifier_bundle_dir = get_classifier_output_bundle_dir(models_dir, bundle_name="latest")

    reference_csv = training_data / "object_labels" / "reference_cards" / "reference_do.csv"
    scene_labels_path = training_data / "object_labels" / "augmented_scenes" / "labels.json"
    ref_cards_dir = training_data / "training_images" / "reference_cards"
    scene_images_dir = training_data / "training_images" / "augmented_scenes"
    scene_masks_dir = training_data / "training_masks" / "augmented_scenes"
    aug_csv = training_data / "object_labels" / "augmented_cards" / "aug.csv"
    aug_cards_dir = training_data / "training_images" / "augmented_cards"
    aug_masks_dir = training_data / "training_masks" / "augmented_cards"

    scene_segmenter_path = resolve_segmenter_checkpoint(models_dir)
    card_model_path = classifier_bundle_dir / "card_classifier.pth"
    card_classes_path = classifier_bundle_dir / "classes.npy"
    card_config_path = classifier_bundle_dir / "config.json"

    required_inputs = [
        reference_csv, scene_labels_path, ref_cards_dir, scene_images_dir,
        scene_masks_dir, aug_csv, aug_cards_dir, aug_masks_dir,
    ]
    for path in required_inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    # R3: hard fail if any training input is somehow under a "test" directory.
    assert_no_test_inputs(required_inputs)

    device = _select_device()
    batch_size = _select_batch_size(device, cfg)
    use_amp = device.type == "cuda"

    if cfg.epoch_max_train_samples is not None and cfg.epoch_max_train_samples <= 0:
        raise ValueError(
            f"epoch_max_train_samples must be > 0 or None, got {cfg.epoch_max_train_samples}"
        )

    print(f"Project root: {project_root}")
    print(f"Training data: {training_data}")
    print(f"Model output dir: {models_dir}")
    print(f"Classifier output bundle: {classifier_bundle_dir}")
    print(f"Segmenter checkpoint: {scene_segmenter_path}")
    print(f"Device: {device}")
    print(f"Augmented-card masks: {aug_masks_dir}")

    # ---------------- sample loading ----------------
    reference_samples, missing_ref = load_reference_samples(reference_csv, ref_cards_dir)
    augmented_card_samples, missing_aug, skipped_aug_labels, skipped_aug_files, skipped_aug_masks = load_augmented_card_samples(
        aug_csv, aug_cards_dir, aug_masks_dir=aug_masks_dir, project_root=project_root
    )
    scene_manual_samples, missing_scene, skipped_scene_labels, skipped_scene_boxes, skipped_scene_masks = (
        load_scene_manual_samples(scene_labels_path, scene_images_dir, scene_masks_dir, project_root)
    )
    missing_files = missing_ref + missing_aug + missing_scene

    if not reference_samples:
        raise RuntimeError("No reference samples found. Run create_reference_cards_do.ipynb first.")
    if not augmented_card_samples:
        raise RuntimeError("No augmented-card samples found. Run create_augmented_data_do.ipynb first.")
    if not scene_manual_samples:
        raise RuntimeError("No scene-manual samples found. Run augmented-scene generation first.")

    all_labels = sorted({s.label for s in reference_samples + augmented_card_samples + scene_manual_samples})
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    label_to_index = {label: idx for idx, label in enumerate(label_encoder.classes_)}

    # ---------------- scene split (by scene image, not by card sample) ----------------
    # Splitting at card-sample level lets the same scene appear in both train and val,
    # which inflates val accuracy. Instead, split the unique scene paths so every card
    # from a given scene goes entirely to one side.
    unique_scene_paths = sorted({s.image_path for s in scene_manual_samples})
    scene_train_paths_list, scene_val_paths_list = train_test_split(
        unique_scene_paths, test_size=cfg.val_split, random_state=cfg.seed
    )
    scene_val_path_set = set(scene_val_paths_list)

    scene_train_samples = [s for s in scene_manual_samples if s.image_path not in scene_val_path_set]
    scene_val_samples = [s for s in scene_manual_samples if s.image_path in scene_val_path_set]

    # Verify: no image should appear on both sides.
    train_scene_paths = {s.image_path for s in scene_train_samples}
    val_scene_paths = {s.image_path for s in scene_val_samples}
    overlap = train_scene_paths & val_scene_paths
    assert not overlap, f"Scene-level split produced {len(overlap)} overlapping image(s) — this is a bug."

    scene_predicted_train_samples = derive_scene_predicted_samples(scene_train_samples)
    scene_predicted_val_samples = derive_scene_predicted_samples(scene_val_samples)

    # ---------------- predicted-mask cache ----------------
    scene_paths_for_pred = sorted({
        s.image_path.resolve() for s in scene_predicted_train_samples + scene_predicted_val_samples
    })
    predicted_scene_probs = build_scene_probability_cache(
        scene_paths_for_pred, scene_segmenter_path, device, target_size=cfg.segmenter_img_size,
    )

    # ---------------- per-stage sample budgets ----------------
    ref_per_epoch = _stage_samples_per_epoch(len(reference_samples), "reference", batch_size, device, cfg)
    aug_per_epoch = _stage_samples_per_epoch(len(augmented_card_samples), "augmented_card", batch_size, device, cfg)
    scene_manual_per_epoch = _stage_samples_per_epoch(len(scene_train_samples), "scene_manual", batch_size, device, cfg)
    scene_pred_per_epoch = _stage_samples_per_epoch(
        len(scene_predicted_train_samples), "scene_predicted", batch_size, device, cfg
    )

    if cfg.epoch_max_train_samples is not None:
        cap = int(cfg.epoch_max_train_samples)
        ref_per_epoch = min(ref_per_epoch, cap)
        aug_per_epoch = min(aug_per_epoch, cap)
        scene_manual_per_epoch = min(scene_manual_per_epoch, cap)
        scene_pred_per_epoch = min(scene_pred_per_epoch, cap)

    # ---------------- loaders ----------------
    pin_memory = device.type == "cuda"
    common = dict(
        label_to_index=label_to_index,
        batch_size=batch_size,
        image_size=cfg.img_size,
        bbox_margin=cfg.bbox_margin,
        mask_threshold=cfg.mask_threshold,
        pin_memory=pin_memory,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    train_loader_ref, _ = make_loader(
        samples=reference_samples, shuffle=True, augment=True,
        balanced=cfg.balanced_sampling, samples_per_epoch=ref_per_epoch, **common,
    )
    train_loader_augmented, _ = make_loader(
        samples=augmented_card_samples, shuffle=True, augment=True,
        balanced=cfg.balanced_sampling, samples_per_epoch=aug_per_epoch, **common,
    )
    train_loader_manual, _ = make_loader(
        samples=scene_train_samples, shuffle=True, augment=True,
        balanced=cfg.balanced_sampling, samples_per_epoch=scene_manual_per_epoch, **common,
    )
    train_loader_predicted, _ = make_loader(
        samples=scene_predicted_train_samples, shuffle=True, augment=True,
        predicted_scene_probs=predicted_scene_probs,
        balanced=cfg.balanced_sampling, samples_per_epoch=scene_pred_per_epoch, **common,
    )

    # Validation: deterministic order, no augmentation. We expose two views:
    #   - val_loader (manual masks): historical metric, comparable across stages.
    #   - val_loader_predicted (segmenter masks): mirrors test-time conditions.
    val_loader, _ = make_loader(
        samples=scene_val_samples, shuffle=False, augment=False, **common,
    )
    val_loader_predicted, _ = make_loader(
        samples=scene_predicted_val_samples, shuffle=False, augment=False,
        predicted_scene_probs=predicted_scene_probs, **common,
    )

    # ---------------- model ----------------
    model, classifier_arch = build_card_classifier(
        n_classes=len(label_encoder.classes_),
        input_channels=4,
        dropout=cfg.classifier_dropout,
        architecture=cfg.classifier_architecture,
        stem_width=cfg.classifier_stem_width,
    )
    model = model.to(device)
    model_params = assert_param_cap(model, f"CardClassifier[{classifier_arch}]")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    print(f"Reference samples:           {len(reference_samples)} (per_epoch={ref_per_epoch})")
    print(f"Augmented-card samples:      {len(augmented_card_samples)} (per_epoch={aug_per_epoch})")
    print(f"Scene-manual train/val:      {len(scene_train_samples)}/{len(scene_val_samples)} (per_epoch={scene_manual_per_epoch})")
    print(f"Scene-predicted train:       {len(scene_predicted_train_samples)} (per_epoch={scene_pred_per_epoch})")
    if cfg.epoch_max_train_samples is not None:
        print(f"Global epoch_max_train_samples cap: {cfg.epoch_max_train_samples}")
    print(f"Num classes:                 {len(label_encoder.classes_)}")
    print(f"Skipped aug labels/files/masks: {skipped_aug_labels}/{skipped_aug_files}/{skipped_aug_masks}")
    print(f"Skipped scene labels/boxes/masks: {skipped_scene_labels}/{skipped_scene_boxes}/{skipped_scene_masks}")
    if missing_files:
        print(f"Missing files skipped:       {len(missing_files)} (first: {missing_files[0]})")

    return {
        "config": cfg,
        "project_root": project_root,
        "project_dir": project_dir,
        "training_data": training_data,
        "models_dir": models_dir,
        "classifier_bundle_dir": classifier_bundle_dir,
        "device": device,
        "use_amp": use_amp,
        "batch_size": batch_size,
        "norm_mean": IMAGENET_MEAN,
        "norm_std": IMAGENET_STD,
        "card_model_path": card_model_path,
        "card_classes_path": card_classes_path,
        "card_config_path": card_config_path,
        "scene_segmenter_path": scene_segmenter_path,
        "reference_samples": reference_samples,
        "augmented_card_samples": augmented_card_samples,
        "scene_manual_samples": scene_manual_samples,
        "scene_train_samples": scene_train_samples,
        "scene_val_samples": scene_val_samples,
        "scene_predicted_train_samples": scene_predicted_train_samples,
        "scene_predicted_val_samples": scene_predicted_val_samples,
        "predicted_scene_probs": predicted_scene_probs,
        "label_encoder": label_encoder,
        "label_to_index": label_to_index,
        "reference_samples_per_epoch": ref_per_epoch,
        "augmented_samples_per_epoch": aug_per_epoch,
        "scene_manual_samples_per_epoch": scene_manual_per_epoch,
        "scene_predicted_samples_per_epoch": scene_pred_per_epoch,
        "train_loader_ref": train_loader_ref,
        "train_loader_augmented": train_loader_augmented,
        "train_loader_manual": train_loader_manual,
        "train_loader_predicted": train_loader_predicted,
        "val_loader": val_loader,
        "val_loader_predicted": val_loader_predicted,
        "model": model,
        "model_params": model_params,
        "ce_loss": ce_loss,
        "scaler": scaler,
        "history": [],
        "best_val_acc": None,
        "best_stage": None,
        "best_epoch": None,
        "training_seconds": None,
        "classifier_architecture": classifier_arch,
        "classifier_stem_width": int(cfg.classifier_stem_width),
        "classifier_dropout": float(cfg.classifier_dropout),
    }


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #


def _autocast_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _freeze_batchnorm_running_stats(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def _safe_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf only. We deliberately do NOT clamp to a small range:
    aggressive clamping silently kills gradients and hides upstream bugs.
    Stage scheduling, label smoothing, AdamW, and grad-clip already address
    the actual sources of instability."""
    if not torch.isfinite(logits).all():
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    return logits


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    device: torch.device,
    cfg: TrainPipelineConfig,
    stage_name: str,
) -> dict[str, float]:
    model.train()
    if cfg.scene_finetune_freeze_bn_stats and stage_name in {"scene_manual", "scene_predicted"}:
        model.apply(_freeze_batchnorm_running_stats)

    running_loss = 0.0
    running_acc = 0.0
    seen = 0
    skipped = 0

    for x, targets_cpu in loader:
        x = x.to(device, non_blocking=True)
        targets = targets_cpu.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=_autocast_device_type(device), dtype=torch.float16, enabled=amp_enabled):
            logits = model(x)
            logits = _safe_logits(logits)
            loss = ce_loss(logits.float(), targets)

        if not torch.isfinite(loss):
            skipped += 1
            continue

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            optimizer.step()

        bs = x.size(0)
        pred = torch.argmax(logits, dim=1)
        running_loss += loss.item() * bs
        running_acc += (pred == targets).float().mean().item() * bs
        seen += bs

    return {
        "loss": running_loss / max(seen, 1),
        "cls_acc": running_acc / max(seen, 1),
        "skipped_non_finite_batches": float(skipped),
    }


@torch.no_grad()
def _evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    amp_enabled: bool,
    device: torch.device,
    return_predictions: bool = False,
):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    seen = 0
    skipped = 0

    true_idx: list[int] = []
    pred_idx: list[int] = []

    for x, targets_cpu in loader:
        x = x.to(device, non_blocking=True)
        targets = targets_cpu.to(device, non_blocking=True)

        with torch.autocast(device_type=_autocast_device_type(device), dtype=torch.float16, enabled=amp_enabled):
            logits = model(x)
        logits = _safe_logits(logits)
        loss = ce_loss(logits.float(), targets)
        if not torch.isfinite(loss):
            skipped += 1
            continue

        pred = torch.argmax(logits, dim=1)
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += (pred == targets).float().mean().item() * bs
        seen += bs

        if return_predictions:
            if device.type == "mps":
                torch.mps.synchronize()
            true_idx.extend(targets_cpu.to(dtype=torch.int64).tolist())
            pred_idx.extend(pred.detach().to("cpu", dtype=torch.int64).tolist())

    metrics = {
        "loss": running_loss / max(seen, 1),
        "cls_acc": running_acc / max(seen, 1),
        "skipped_non_finite_batches": float(skipped),
    }
    if return_predictions:
        return metrics, np.asarray(true_idx, dtype=np.int64), np.asarray(pred_idx, dtype=np.int64)
    return metrics


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #


def run_training(state: dict[str, Any]) -> dict[str, Any]:
    cfg: TrainPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    ce_loss: nn.Module = state["ce_loss"]
    scaler = state["scaler"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    val_loader: DataLoader = state["val_loader"]
    val_loader_predicted: DataLoader = state["val_loader_predicted"]

    stage_plan = [
        (
            "stage1_reference",
            cfg.stage_1_epochs,
            state["reference_samples"],
            state["reference_samples_per_epoch"],
            cfg.stage_1_lr,
            "reference",
            None,
        ),
        (
            "stage2_augmented_cards",
            cfg.stage_2_epochs,
            state["augmented_card_samples"],
            state["augmented_samples_per_epoch"],
            cfg.stage_2_lr,
            "augmented_card",
            None,
        ),
        (
            "stage3_scene_manual_masks",
            cfg.stage_3_epochs,
            state["scene_train_samples"],
            state["scene_manual_samples_per_epoch"],
            cfg.stage_3_lr,
            "scene_manual",
            None,
        ),
        (
            "stage4_scene_predicted_masks",
            cfg.stage_4_epochs,
            state["scene_predicted_train_samples"],
            state["scene_predicted_samples_per_epoch"],
            cfg.stage_4_lr,
            "scene_predicted",
            state["predicted_scene_probs"],
        ),
    ]

    # Selection policy: best epoch within the LAST executed stage only.
    # Earlier stages train on simplified data (clean references, augmented cards,
    # manual masks); their val accuracy is inflated relative to test-time, where
    # only the segmenter's predicted masks are available. A higher stage-3 val acc
    # does not imply better real-world performance, so we deliberately keep the
    # final-stage checkpoint even if its number is lower.
    last_stage_best_val_acc = -1.0
    last_stage_best_state_dict: dict[str, torch.Tensor] | None = None
    last_stage_name: str | None = None
    last_stage_best_epoch: int = -1

    history: list[dict[str, float | int | str]] = []
    global_start = time.perf_counter()

    for stage_index, (
        stage_name,
        stage_epochs,
        stage_samples,
        stage_samples_per_epoch,
        stage_lr,
        stage_kind,
        stage_predicted_probs,
    ) in enumerate(stage_plan, start=1):
        if stage_epochs <= 0 or len(stage_samples) == 0:
            print(f"[skip] {stage_name}: no epochs or no samples")
            continue

        samples_per_epoch = int(min(stage_samples_per_epoch, len(stage_samples)))

        optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        stage_best_val_acc = -1.0
        epochs_without_improvement = 0

        print(
            f"\n[{stage_name}] epochs={stage_epochs}, lr={stage_lr:.2e}, "
            f"dataset_samples={len(stage_samples)}, samples_per_epoch={samples_per_epoch}"
        )

        coverage_indices: list[int] | None = None
        coverage_cursor = 0
        coverage_rng: random.Random | None = None
        if samples_per_epoch < len(stage_samples):
            coverage_indices = list(range(len(stage_samples)))
            coverage_rng = random.Random(cfg.seed + stage_index * 10_000)
            coverage_rng.shuffle(coverage_indices)
            print(
                "  per-epoch cap active: epochs cycle through the full stage "
                "sample pool before repeating"
            )

        for epoch in range(1, stage_epochs + 1):
            epoch_start = time.perf_counter()

            epoch_seed = cfg.seed + stage_index * 10_000 + epoch
            epoch_samples = stage_samples
            epoch_samples_per_epoch = samples_per_epoch
            if samples_per_epoch < len(stage_samples):
                if coverage_indices is None or coverage_rng is None:
                    raise RuntimeError("Coverage sampler state was not initialized.")

                selected_indices: list[int] = []
                while len(selected_indices) < samples_per_epoch:
                    remaining_in_cycle = len(coverage_indices) - coverage_cursor
                    to_take = min(samples_per_epoch - len(selected_indices), remaining_in_cycle)
                    selected_indices.extend(
                        coverage_indices[coverage_cursor: coverage_cursor + to_take]
                    )
                    coverage_cursor += to_take

                    if coverage_cursor >= len(coverage_indices):
                        coverage_rng.shuffle(coverage_indices)
                        coverage_cursor = 0

                epoch_samples = [stage_samples[i] for i in selected_indices]
                epoch_samples_per_epoch = None

            stage_loader, _ = make_loader(
                samples=epoch_samples,
                label_to_index=state["label_to_index"],
                batch_size=state["batch_size"],
                image_size=cfg.img_size,
                bbox_margin=cfg.bbox_margin,
                mask_threshold=cfg.mask_threshold,
                shuffle=True,
                augment=True,
                pin_memory=(device.type == "cuda"),
                num_workers=cfg.num_workers,
                seed=cfg.seed,
                predicted_scene_probs=stage_predicted_probs,
                balanced=cfg.balanced_sampling,
                samples_per_epoch=epoch_samples_per_epoch,
                sampler_seed=epoch_seed,
            )

            train_metrics = _train_one_epoch(
                model, stage_loader, optimizer, ce_loss, scaler, use_amp, device, cfg, stage_kind
            )
            val_metrics = _evaluate_one_epoch(model, val_loader, ce_loss, use_amp, device)
            # Stage-4 mirrors test-time: also report predicted-mask val accuracy
            # so we can see if the model actually generalizes to segmenter outputs.
            val_pred_metrics = _evaluate_one_epoch(model, val_loader_predicted, ce_loss, use_amp, device)

            scheduler.step(val_metrics["cls_acc"])
            lr_now = float(optimizer.param_groups[0]["lr"])
            epoch_seconds = float(time.perf_counter() - epoch_start)

            row = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_cls_acc": float(train_metrics["cls_acc"]),
                "val_loss": float(val_metrics["loss"]),
                "val_cls_acc": float(val_metrics["cls_acc"]),
                "val_pred_loss": float(val_pred_metrics["loss"]),
                "val_pred_cls_acc": float(val_pred_metrics["cls_acc"]),
                "lr": lr_now,
                "seconds": epoch_seconds,
                "samples_per_epoch": int(samples_per_epoch),
                "train_skipped_non_finite_batches": int(train_metrics["skipped_non_finite_batches"]),
                "val_skipped_non_finite_batches": int(val_metrics["skipped_non_finite_batches"]),
            }
            history.append(row)

            # Selection metric: predicted-mask val accuracy on stage 4 (test-realistic),
            # manual-mask val accuracy elsewhere. Tracker resets per stage so only
            # the last executed stage's best epoch is retained.
            if last_stage_name != stage_name:
                last_stage_name = stage_name
                last_stage_best_val_acc = -1.0
            selection_acc = val_pred_metrics["cls_acc"] if stage_kind == "scene_predicted" else val_metrics["cls_acc"]
            is_stage_best = selection_acc > last_stage_best_val_acc
            if is_stage_best:
                last_stage_best_val_acc = float(selection_acc)
                last_stage_best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                last_stage_best_epoch = epoch

            stage_improved = val_metrics["cls_acc"] > stage_best_val_acc + 1e-6
            if stage_improved:
                stage_best_val_acc = float(val_metrics["cls_acc"])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"  epoch {epoch:02d}/{stage_epochs} | "
                f"train_acc={train_metrics['cls_acc']*100:5.1f}% loss={train_metrics['loss']:.3f} | "
                f"val(manual)={val_metrics['cls_acc']*100:5.1f}% "
                f"val(pred)={val_pred_metrics['cls_acc']*100:5.1f}% | "
                f"lr={lr_now:.2e} | {epoch_seconds:.1f}s"
                + (" | stage-best" if is_stage_best else "")
            )

            if (
                stage_kind in {"scene_manual", "scene_predicted"}
                and epoch >= cfg.min_epochs_per_stage
                and epochs_without_improvement >= cfg.early_stop_patience
            ):
                print(f"  [early-stop] {stage_name}: validation plateaued for {cfg.early_stop_patience} epochs")
                break

    if last_stage_best_state_dict is None:
        raise RuntimeError("Training produced no checkpoint (all stages had zero epochs).")

    model.load_state_dict(last_stage_best_state_dict)

    state["history"] = history
    state["best_val_acc"] = float(last_stage_best_val_acc)
    state["best_stage"] = last_stage_name
    state["best_epoch"] = int(last_stage_best_epoch)
    state["training_seconds"] = time.perf_counter() - global_start

    print("\nTraining complete.")
    print(f"Last-stage best validation accuracy: {last_stage_best_val_acc * 100:.2f}%")
    print(f"Selected checkpoint: {last_stage_name} epoch {last_stage_best_epoch}")
    print(f"Total training time: {state['training_seconds'] / 60:.2f} min")
    return state


# --------------------------------------------------------------------------- #
# Artifact saving
# --------------------------------------------------------------------------- #


def _stage_plan_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    cfg: TrainPipelineConfig = state["config"]
    entries = [
        (
            "stage1_reference",
            cfg.stage_1_epochs,
            cfg.stage_1_lr,
            state["reference_samples"],
            state["reference_samples_per_epoch"],
        ),
        (
            "stage2_augmented_cards",
            cfg.stage_2_epochs,
            cfg.stage_2_lr,
            state["augmented_card_samples"],
            state["augmented_samples_per_epoch"],
        ),
        (
            "stage3_scene_manual_masks",
            cfg.stage_3_epochs,
            cfg.stage_3_lr,
            state["scene_train_samples"],
            state["scene_manual_samples_per_epoch"],
        ),
        (
            "stage4_scene_predicted_masks",
            cfg.stage_4_epochs,
            cfg.stage_4_lr,
            state["scene_predicted_train_samples"],
            state["scene_predicted_samples_per_epoch"],
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, epochs, lr, samples, samples_per_epoch in entries:
        out.append({
            "stage": name,
            "epochs": epochs,
            "learning_rate": lr,
            "n_samples": len(samples),
            "samples_per_epoch": int(min(samples_per_epoch, len(samples))),
        })
    return out


def save_training_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    cfg: TrainPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    label_encoder: LabelEncoder = state["label_encoder"]

    card_model_path: Path = state["card_model_path"]
    card_classes_path: Path = state["card_classes_path"]
    card_config_path: Path = state["card_config_path"]
    classifier_architecture = str(state.get("classifier_architecture", "torchvision_resnet18"))
    classifier_stem_width = int(state.get("classifier_stem_width", 60))
    classifier_dropout = float(state.get("classifier_dropout", 0.20))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model.__class__.__name__,
            "n_classes": int(len(label_encoder.classes_)),
            "img_size": cfg.img_size,
            "input_channels": 4,
            "architecture": classifier_architecture,
            "stem_width": classifier_stem_width,
            "dropout": classifier_dropout,
        },
        card_model_path,
    )
    np.save(card_classes_path, label_encoder.classes_)

    config = {
        "model_file": card_model_path.name,
        "classes_file": card_classes_path.name,
        "image_size": cfg.img_size,
        "normalization_mean": IMAGENET_MEAN.tolist(),
        "normalization_std": IMAGENET_STD.tolist(),
        "bbox_margin": cfg.bbox_margin,
        "mask_threshold": cfg.mask_threshold,
        "input_mode": "rgb_plus_mask_channel",
        "masked_background_fill": 128,
        "augmented_cards_use_saved_masks": True,
        "architecture": classifier_architecture,
        "classifier_stem_width": classifier_stem_width,
        "classifier_dropout": classifier_dropout,
        "pretrained": False,
        "optimizer": "AdamW",
        "weight_decay": cfg.weight_decay,
        "batch_size": state["batch_size"],
        "device": str(state["device"]),
        "balanced_sampling": cfg.balanced_sampling,
        "early_stop_patience": cfg.early_stop_patience,
        "min_epochs_per_stage": cfg.min_epochs_per_stage,
        "best_selection_policy": "global_best_across_stages_manual_or_predicted_val",
        "scene_finetune_freeze_bn_stats": bool(cfg.scene_finetune_freeze_bn_stats),
        "grad_clip_norm": float(cfg.grad_clip_norm),
        "stage_plan": _stage_plan_summary(state),
        "n_classes": int(len(label_encoder.classes_)),
        "class_names": [str(c) for c in label_encoder.classes_],
        "validation_samples": int(len(state["scene_val_samples"])),
        "best_val_accuracy": float(state.get("best_val_acc") or 0.0),
        "best_stage": state.get("best_stage"),
        "best_epoch": int(state.get("best_epoch") or -1),
        "trainable_params": int(state.get("model_params") or 0),
        "seed": cfg.seed,
    }

    with card_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model checkpoint: {card_model_path}")
    print(f"Saved class names:      {card_classes_path}")
    print(f"Saved config:           {card_config_path}")

    state["saved_config"] = config
    return state


# --------------------------------------------------------------------------- #
# Diagnostics & visualization
# --------------------------------------------------------------------------- #


def run_validation_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    model: nn.Module = state["model"]
    val_loader: DataLoader = state["val_loader"]
    ce_loss: nn.Module = state["ce_loss"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    label_encoder: LabelEncoder = state["label_encoder"]

    val_metrics, val_true_idx, val_pred_idx = _evaluate_one_epoch(
        model, val_loader, ce_loss, use_amp, device, return_predictions=True,
    )

    n_classes = len(label_encoder.classes_)
    valid_mask = (val_true_idx >= 0) & (val_true_idx < n_classes) & (val_pred_idx >= 0) & (val_pred_idx < n_classes)
    invalid = int(np.count_nonzero(~valid_mask))
    if invalid > 0:
        print(f"[warning] Dropping {invalid} invalid validation rows.")
        val_true_idx = val_true_idx[valid_mask]
        val_pred_idx = val_pred_idx[valid_mask]

    if val_true_idx.size == 0:
        raise RuntimeError("No valid validation predictions available.")

    true_labels = label_encoder.inverse_transform(val_true_idx)
    pred_labels = label_encoder.inverse_transform(val_pred_idx)

    print(f"Validation accuracy (manual masks): {val_metrics['cls_acc'] * 100:.2f}%")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    fig_size = max(8, 0.30 * len(label_encoder.classes_))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ConfusionMatrixDisplay.from_predictions(
        true_labels, pred_labels, labels=label_encoder.classes_,
        xticks_rotation=90, colorbar=False, ax=ax,
    )
    ax.set_title(f"Masked classifier validation accuracy: {val_metrics['cls_acc'] * 100:.1f}%")
    plt.tight_layout()
    plt.show()

    history = state.get("history", [])
    if history:
        epochs_x = np.arange(1, len(history) + 1)
        train_acc = [float(r["train_cls_acc"]) for r in history]
        val_acc = [float(r["val_cls_acc"]) for r in history]
        val_pred_acc = [float(r.get("val_pred_cls_acc", float("nan"))) for r in history]
        train_loss = [float(r["train_loss"]) for r in history]
        val_loss = [float(r["val_loss"]) for r in history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs_x, train_acc, marker="o", label="train")
        axes[0].plot(epochs_x, val_acc, marker="o", label="val (manual)")
        axes[0].plot(epochs_x, val_pred_acc, marker="o", label="val (predicted)")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Classification accuracy")
        axes[0].set_xlabel("Global epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs_x, train_loss, marker="o", label="train")
        axes[1].plot(epochs_x, val_loss, marker="o", label="val")
        axes[1].set_title("Cross-entropy loss")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        for idx, r in enumerate(history, start=1):
            axes[0].annotate(str(r["stage"]).replace("stage", "S"), (idx, float(r["val_cls_acc"])),
                             fontsize=7, alpha=0.65)

        plt.tight_layout()
        plt.show()

    state["val_metrics"] = val_metrics
    state["val_true_idx"] = val_true_idx
    state["val_pred_idx"] = val_pred_idx
    state["val_sample_indices"] = np.arange(val_true_idx.size, dtype=np.int64)
    state["true_labels"] = true_labels
    state["pred_labels"] = pred_labels
    return state


def plot_stage_preview(state: dict[str, Any], per_stage: int | None = None) -> None:
    cfg: TrainPipelineConfig = state["config"]
    rng = np.random.default_rng(cfg.seed)
    n = int(per_stage or cfg.preview_per_stage)

    pools = [
        ("reference", state["reference_samples"]),
        ("augmented_card", state["augmented_card_samples"]),
        ("scene_manual", state["scene_train_samples"]),
        ("scene_predicted", state["scene_predicted_train_samples"]),
    ]
    preview: list[CardSample] = []
    for _, pool in pools:
        if pool:
            preview.extend(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())

    if not preview:
        print("No samples available for preview.")
        return

    cols = 4
    rows = int(np.ceil(len(preview) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, sample in zip(axes, preview):
        crop_bgr, mask_u8 = sample_to_crop_and_mask(
            sample,
            bbox_margin=cfg.bbox_margin,
            mask_threshold=cfg.mask_threshold,
            predicted_scene_probs=state["predicted_scene_probs"],
        )
        crop_lb, mask_lb = letterbox_image_and_mask(crop_bgr, mask_u8, size=cfg.img_size)
        overlay = cv2.cvtColor(crop_lb, cv2.COLOR_BGR2RGB).copy()
        overlay[mask_lb > 0] = (0.65 * overlay[mask_lb > 0] + 0.35 * np.array([0, 255, 255])).astype(np.uint8)

        ax.imshow(overlay)
        ax.set_title(f"{sample.stage}\n{sample.label}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(preview):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_wrong_predictions(state: dict[str, Any], show_count: int = 12) -> list[int]:
    if "true_labels" not in state or "pred_labels" not in state:
        raise RuntimeError("Run run_validation_diagnostics first.")

    true_labels = state["true_labels"]
    pred_labels = state["pred_labels"]
    wrong = [i for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t != p]
    print(f"Wrong validation predictions: {len(wrong)} / {len(true_labels)}")

    if not wrong:
        return wrong

    cfg: TrainPipelineConfig = state["config"]
    scene_val_samples = state["scene_val_samples"]
    val_sample_indices = state.get("val_sample_indices")
    model: nn.Module = state["model"]
    device: torch.device = state["device"]

    n_show = min(int(show_count), len(wrong))
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, wrong[:n_show]):
        sample_idx = int(val_sample_indices[int(idx)]) if val_sample_indices is not None else int(idx)
        sample = scene_val_samples[sample_idx]

        crop_bgr, mask_u8 = sample_to_crop_and_mask(
            sample, bbox_margin=cfg.bbox_margin, mask_threshold=cfg.mask_threshold,
            predicted_scene_probs=None,
        )
        crop_lb, mask_lb = letterbox_image_and_mask(crop_bgr, mask_u8, size=cfg.img_size)
        masked_img = compose_masked_card_image(crop_lb, mask_lb)
        x = card_input_to_tensor(masked_img, mask_lb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_index = int(torch.argmax(probs).item())
            pred_conf = float(probs[pred_index].item())

        vis = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        ax.imshow(vis)
        ax.set_title(
            f"true={true_labels[int(idx)]}\npred={pred_labels[int(idx)]} ({pred_conf:.2f})",
            fontsize=8, color="red",
        )
        ax.axis("off")

    for ax in axes[n_show:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return wrong
