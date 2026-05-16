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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
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
    num_workers: int = 4
    persistent_workers: bool = True

    reference_target_gpu_mps: int = 1536
    reference_target_cpu: int = 512
    augmented_target_gpu_mps: int = 4096
    augmented_target_cpu: int = 1536
    scene_target_gpu_mps: int = 4096
    scene_target_cpu: int = 1536

    scene_finetune_freeze_bn_stats: bool = True
    grad_clip_norm: float = 1.0
    predicted_cache_binary_masks: bool = True

    # Optional memory/runtime caps.
    # Limits are applied on unique scene images (not card instances).
    max_loaded_scene_images: int | None = None
    max_predicted_cache_images: int | None = None

    preview_per_stage: int = 3

    classifier_architecture: str = "resnet18_small"
    classifier_stem_width: int = 60
    classifier_dropout: float = 0.20

    # Best-checkpoint selection can optimize either a single metric or a
    # composite score that captures both raw accuracy and class-balance quality.
    best_epoch_selection_metric: str = "composite"
    selection_weight_val_acc: float = 0.35
    selection_weight_macro_f1: float = 0.30
    selection_weight_balanced_acc: float = 0.20
    selection_weight_top3_acc: float = 0.10
    selection_weight_val_loss: float = 0.05


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


def _deterministic_path_subset(paths: list[Path], max_count: int, seed: int) -> list[Path]:
    if max_count <= 0:
        raise ValueError(f"max_count must be > 0, got {max_count}")
    if len(paths) <= max_count:
        return list(paths)
    rng = random.Random(int(seed))
    return sorted(rng.sample(list(paths), k=max_count))


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
    if cfg.max_loaded_scene_images is not None:
        capped_scene_paths = _deterministic_path_subset(
            unique_scene_paths,
            max_count=int(cfg.max_loaded_scene_images),
            seed=cfg.seed + 13_101,
        )
        capped_scene_path_set = {p.resolve() for p in capped_scene_paths}
        if len(capped_scene_paths) < len(unique_scene_paths):
            before_samples = len(scene_manual_samples)
            scene_manual_samples = [
                s for s in scene_manual_samples if s.image_path.resolve() in capped_scene_path_set
            ]
            unique_scene_paths = capped_scene_paths
            print(
                "[cap] scene images limited to "
                f"{len(unique_scene_paths)} unique files "
                f"(max_loaded_scene_images={cfg.max_loaded_scene_images}); "
                f"scene-manual samples: {before_samples} -> {len(scene_manual_samples)}"
            )
        else:
            print(
                "[cap] max_loaded_scene_images="
                f"{cfg.max_loaded_scene_images} (no-op: dataset has fewer unique scene images)"
            )

    if len(unique_scene_paths) < 2:
        raise RuntimeError(
            "Need at least 2 unique scene images after caps to split train/val. "
            "Increase max_loaded_scene_images or disable it."
        )

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

    stage4_enabled = int(cfg.stage_4_epochs) > 0
    if stage4_enabled:
        scene_predicted_train_samples = derive_scene_predicted_samples(scene_train_samples)
        scene_predicted_val_samples = derive_scene_predicted_samples(scene_val_samples)

        # ---------------- predicted-mask cache ----------------
        scene_paths_for_pred = sorted({
            s.image_path.resolve() for s in scene_predicted_train_samples + scene_predicted_val_samples
        })

        if cfg.max_predicted_cache_images is not None:
            capped_pred_paths = _deterministic_path_subset(
                scene_paths_for_pred,
                max_count=int(cfg.max_predicted_cache_images),
                seed=cfg.seed + 17_171,
            )
            capped_pred_path_set = {p.resolve() for p in capped_pred_paths}
            if len(capped_pred_paths) < len(scene_paths_for_pred):
                before_pred_train = len(scene_predicted_train_samples)
                before_pred_val = len(scene_predicted_val_samples)
                scene_predicted_train_samples = [
                    s for s in scene_predicted_train_samples if s.image_path.resolve() in capped_pred_path_set
                ]
                scene_predicted_val_samples = [
                    s for s in scene_predicted_val_samples if s.image_path.resolve() in capped_pred_path_set
                ]
                scene_paths_for_pred = sorted(capped_pred_paths)
                print(
                    "[cap] predicted-mask cache limited to "
                    f"{len(scene_paths_for_pred)} scenes "
                    f"(max_predicted_cache_images={cfg.max_predicted_cache_images}); "
                    f"scene-pred train: {before_pred_train} -> {len(scene_predicted_train_samples)}, "
                    f"scene-pred val: {before_pred_val} -> {len(scene_predicted_val_samples)}"
                )
            else:
                print(
                    "[cap] max_predicted_cache_images="
                    f"{cfg.max_predicted_cache_images} (no-op: dataset has fewer predicted scenes)"
                )

        cache_threshold = cfg.mask_threshold if cfg.predicted_cache_binary_masks else None
        predicted_scene_probs = build_scene_probability_cache(
            scene_paths_for_pred,
            scene_segmenter_path,
            device,
            target_size=cfg.segmenter_img_size,
            mask_threshold=cache_threshold,
        )
        predicted_cache_bytes = int(sum(arr.nbytes for arr in predicted_scene_probs.values()))
        predicted_cache_mode = "uint8_binary_masks" if cfg.predicted_cache_binary_masks else "float32_probabilities"
    else:
        scene_predicted_train_samples = []
        scene_predicted_val_samples = []
        predicted_scene_probs = {}
        predicted_cache_bytes = 0
        predicted_cache_mode = "disabled_stage4_epochs_zero"
        print("[skip] Stage 4 disabled (stage_4_epochs <= 0): predicted-mask cache not built.")

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
        persistent_workers=cfg.persistent_workers,
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
        samples=scene_predicted_train_samples,
        shuffle=bool(stage4_enabled and len(scene_predicted_train_samples) > 0),
        augment=True,
        predicted_scene_probs=(predicted_scene_probs if stage4_enabled else None),
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
        predicted_scene_probs=(predicted_scene_probs if stage4_enabled else None), **common,
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
    print(
        "Predicted scene cache:       "
        f"{predicted_cache_mode} ({predicted_cache_bytes / (1024 ** 2):.1f} MiB)"
    )
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
        "predicted_cache_mode": predicted_cache_mode,
        "predicted_cache_bytes": predicted_cache_bytes,
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
        "best_selection_score": None,
        "best_selection_metric": str(cfg.best_epoch_selection_metric),
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


def _inverse_error_score(value: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(value)))


def _selection_score_from_metrics(metrics: dict[str, float], cfg: TrainPipelineConfig) -> float:
    metric = str(cfg.best_epoch_selection_metric).strip().lower()

    direct_metrics = {
        "val_cls_acc": "cls_acc",
        "cls_acc": "cls_acc",
        "val_macro_f1": "macro_f1",
        "macro_f1": "macro_f1",
        "val_balanced_acc": "balanced_acc",
        "balanced_acc": "balanced_acc",
        "val_top3_acc": "top3_acc",
        "top3_acc": "top3_acc",
        "val_weighted_f1": "weighted_f1",
        "weighted_f1": "weighted_f1",
    }
    if metric in direct_metrics:
        return float(metrics[direct_metrics[metric]])

    if metric in {"neg_val_loss", "inv_val_loss", "loss_inverse"}:
        return _inverse_error_score(float(metrics["loss"]))

    if metric != "composite":
        raise ValueError(
            "Unsupported best_epoch_selection_metric. Use one of: "
            "composite, val_cls_acc, val_macro_f1, val_balanced_acc, "
            "val_top3_acc, val_weighted_f1, neg_val_loss."
        )

    weighted_terms = [
        (float(cfg.selection_weight_val_acc), float(metrics["cls_acc"])),
        (float(cfg.selection_weight_macro_f1), float(metrics["macro_f1"])),
        (float(cfg.selection_weight_balanced_acc), float(metrics["balanced_acc"])),
        (float(cfg.selection_weight_top3_acc), float(metrics["top3_acc"])),
        (float(cfg.selection_weight_val_loss), _inverse_error_score(float(metrics["loss"]))),
    ]

    total_weight = sum(max(0.0, weight) for weight, _ in weighted_terms)
    if total_weight <= 0:
        raise ValueError("Composite selection weights must contain at least one positive value.")

    weighted_sum = sum(max(0.0, weight) * value for weight, value in weighted_terms)
    return float(weighted_sum / total_weight)


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
    running_top3 = 0.0
    running_confidence = 0.0
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

        probs = torch.softmax(logits.float(), dim=1)
        pred = torch.argmax(probs, dim=1)
        top_k = min(3, int(probs.shape[1]))
        topk_idx = torch.topk(probs, k=top_k, dim=1).indices

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += (pred == targets).float().mean().item() * bs
        running_top3 += float((topk_idx == targets.unsqueeze(1)).any(dim=1).float().sum().item())
        running_confidence += float(probs.max(dim=1).values.sum().item())
        seen += bs

        if device.type == "mps":
            torch.mps.synchronize()
        true_idx.extend(targets_cpu.to(dtype=torch.int64).tolist())
        pred_idx.extend(pred.detach().to("cpu", dtype=torch.int64).tolist())

    true_idx_np = np.asarray(true_idx, dtype=np.int64)
    pred_idx_np = np.asarray(pred_idx, dtype=np.int64)

    if true_idx_np.size > 0:
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_idx_np, pred_idx_np, average="macro", zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_idx_np, pred_idx_np, average="weighted", zero_division=0,
        )
        balanced_acc = float(balanced_accuracy_score(true_idx_np, pred_idx_np))
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
        balanced_acc = 0.0

    metrics = {
        "loss": running_loss / max(seen, 1),
        "cls_acc": running_acc / max(seen, 1),
        "top3_acc": running_top3 / max(seen, 1),
        "mean_confidence": running_confidence / max(seen, 1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "balanced_acc": float(balanced_acc),
        "skipped_non_finite_batches": float(skipped),
    }
    if return_predictions:
        return metrics, true_idx_np, pred_idx_np
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
    # manual masks); their scores are often inflated relative to test-time. We
    # therefore keep only the best epoch from the latest executed stage.
    last_stage_best_val_acc = -1.0
    last_stage_best_selection_score = -float("inf")
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

        stage_best_selection_score = -float("inf")
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

        stage_loader_reusable: DataLoader | None = None
        if samples_per_epoch >= len(stage_samples):
            stage_loader_reusable, _ = make_loader(
                samples=stage_samples,
                label_to_index=state["label_to_index"],
                batch_size=state["batch_size"],
                image_size=cfg.img_size,
                bbox_margin=cfg.bbox_margin,
                mask_threshold=cfg.mask_threshold,
                shuffle=True,
                augment=True,
                pin_memory=(device.type == "cuda"),
                num_workers=cfg.num_workers,
                persistent_workers=cfg.persistent_workers,
                seed=cfg.seed,
                predicted_scene_probs=stage_predicted_probs,
                balanced=cfg.balanced_sampling,
                samples_per_epoch=samples_per_epoch,
                sampler_seed=cfg.seed + stage_index * 10_000,
            )

        for epoch in range(1, stage_epochs + 1):
            epoch_start = time.perf_counter()

            epoch_seed = cfg.seed + stage_index * 10_000 + epoch
            if stage_loader_reusable is not None:
                stage_loader = stage_loader_reusable
            else:
                epoch_samples = stage_samples
                epoch_samples_per_epoch = samples_per_epoch
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
                    persistent_workers=cfg.persistent_workers,
                    seed=cfg.seed,
                    predicted_scene_probs=stage_predicted_probs,
                    balanced=cfg.balanced_sampling,
                    samples_per_epoch=epoch_samples_per_epoch,
                    sampler_seed=epoch_seed,
                )

            train_metrics = _train_one_epoch(
                model, stage_loader, optimizer, ce_loss, scaler, use_amp, device, cfg, stage_kind
            )

            # Runtime fix: run exactly one full validation pass per epoch.
            # Stage-4 uses predicted-mask validation (test-time realistic);
            # earlier stages use manual-mask validation.
            val_metrics: dict[str, float]
            val_pred_metrics: dict[str, float]
            stage_eval_metrics: dict[str, float]
            val_metric_name: str
            if stage_kind == "scene_predicted":
                val_metrics = {
                    "loss": float("nan"),
                    "cls_acc": float("nan"),
                    "top3_acc": float("nan"),
                    "mean_confidence": float("nan"),
                    "macro_precision": float("nan"),
                    "macro_recall": float("nan"),
                    "macro_f1": float("nan"),
                    "weighted_precision": float("nan"),
                    "weighted_recall": float("nan"),
                    "weighted_f1": float("nan"),
                    "balanced_acc": float("nan"),
                    "skipped_non_finite_batches": 0.0,
                }
                val_pred_metrics = _evaluate_one_epoch(model, val_loader_predicted, ce_loss, use_amp, device)
                stage_eval_metrics = val_pred_metrics
                stage_val_acc = float(stage_eval_metrics["cls_acc"])
                val_metric_name = "pred"
            else:
                val_metrics = _evaluate_one_epoch(model, val_loader, ce_loss, use_amp, device)
                val_pred_metrics = {
                    "loss": float("nan"),
                    "cls_acc": float("nan"),
                    "top3_acc": float("nan"),
                    "mean_confidence": float("nan"),
                    "macro_precision": float("nan"),
                    "macro_recall": float("nan"),
                    "macro_f1": float("nan"),
                    "weighted_precision": float("nan"),
                    "weighted_recall": float("nan"),
                    "weighted_f1": float("nan"),
                    "balanced_acc": float("nan"),
                    "skipped_non_finite_batches": 0.0,
                }
                stage_eval_metrics = val_metrics
                stage_val_acc = float(stage_eval_metrics["cls_acc"])
                val_metric_name = "manual"

            selection_score = _selection_score_from_metrics(stage_eval_metrics, cfg)

            scheduler.step(selection_score)
            lr_now = float(optimizer.param_groups[0]["lr"])
            epoch_seconds = float(time.perf_counter() - epoch_start)

            row = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_cls_acc": float(train_metrics["cls_acc"]),
                "val_loss": float(val_metrics["loss"]),
                "val_cls_acc": float(val_metrics["cls_acc"]),
                "val_top3_acc": float(val_metrics["top3_acc"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_balanced_acc": float(val_metrics["balanced_acc"]),
                "val_pred_loss": float(val_pred_metrics["loss"]),
                "val_pred_cls_acc": float(val_pred_metrics["cls_acc"]),
                "val_pred_top3_acc": float(val_pred_metrics["top3_acc"]),
                "val_pred_macro_f1": float(val_pred_metrics["macro_f1"]),
                "val_pred_balanced_acc": float(val_pred_metrics["balanced_acc"]),
                "selection_score": float(selection_score),
                "lr": lr_now,
                "seconds": epoch_seconds,
                "samples_per_epoch": int(samples_per_epoch),
                "train_skipped_non_finite_batches": int(train_metrics["skipped_non_finite_batches"]),
                "val_skipped_non_finite_batches": int(
                    val_pred_metrics["skipped_non_finite_batches"]
                    if stage_kind == "scene_predicted"
                    else val_metrics["skipped_non_finite_batches"]
                ),
            }
            history.append(row)

            # Selection metric: predicted-mask val accuracy on stage 4 (test-realistic),
            # manual-mask validation elsewhere. The selection score is configurable
            # and can combine accuracy, macro-F1, balanced accuracy, top-k, and loss.
            # Tracker resets per stage so only the last executed stage's best epoch
            # is retained.
            if last_stage_name != stage_name:
                last_stage_name = stage_name
                last_stage_best_selection_score = -float("inf")
                last_stage_best_val_acc = -1.0
            is_stage_best = selection_score > last_stage_best_selection_score
            if is_stage_best:
                last_stage_best_selection_score = float(selection_score)
                last_stage_best_val_acc = float(stage_val_acc)
                last_stage_best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                last_stage_best_epoch = epoch

            stage_improved = selection_score > stage_best_selection_score + 1e-8
            if stage_improved:
                stage_best_selection_score = float(selection_score)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"  epoch {epoch:02d}/{stage_epochs} | "
                f"train_acc={train_metrics['cls_acc']*100:5.1f}% loss={train_metrics['loss']:.3f} | "
                f"val({val_metric_name})={stage_val_acc*100:5.1f}% "
                f"macro_f1={stage_eval_metrics['macro_f1']:.3f} "
                f"bal_acc={stage_eval_metrics['balanced_acc']:.3f} | "
                f"selection({cfg.best_epoch_selection_metric})={selection_score:.4f} | "
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
    state["best_selection_score"] = float(last_stage_best_selection_score)
    state["best_selection_metric"] = str(cfg.best_epoch_selection_metric)
    state["best_stage"] = last_stage_name
    state["best_epoch"] = int(last_stage_best_epoch)
    state["training_seconds"] = time.perf_counter() - global_start

    print("\nTraining complete.")
    print(
        f"Last-stage best selection score ({cfg.best_epoch_selection_metric}): "
        f"{last_stage_best_selection_score:.4f}"
    )
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
        "max_loaded_scene_images": cfg.max_loaded_scene_images,
        "max_predicted_cache_images": cfg.max_predicted_cache_images,
        "best_selection_policy": "last_stage_best_by_selection_metric",
        "best_selection_metric": str(state.get("best_selection_metric") or cfg.best_epoch_selection_metric),
        "best_selection_score": float(state.get("best_selection_score") or 0.0),
        "selection_weight_val_acc": float(cfg.selection_weight_val_acc),
        "selection_weight_macro_f1": float(cfg.selection_weight_macro_f1),
        "selection_weight_balanced_acc": float(cfg.selection_weight_balanced_acc),
        "selection_weight_top3_acc": float(cfg.selection_weight_top3_acc),
        "selection_weight_val_loss": float(cfg.selection_weight_val_loss),
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
    val_loader_predicted: DataLoader = state["val_loader_predicted"]
    ce_loss: nn.Module = state["ce_loss"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    label_encoder: LabelEncoder = state["label_encoder"]
    cfg: TrainPipelineConfig = state["config"]
    has_predicted_validation = len(state.get("scene_predicted_val_samples", [])) > 0

    val_metrics, val_true_idx, val_pred_idx = _evaluate_one_epoch(
        model, val_loader, ce_loss, use_amp, device, return_predictions=True,
    )
    if has_predicted_validation:
        val_pred_metrics, val_pred_true_idx, val_pred_pred_idx = _evaluate_one_epoch(
            model, val_loader_predicted, ce_loss, use_amp, device, return_predictions=True,
        )
    else:
        val_pred_metrics = {
            "loss": float("nan"),
            "cls_acc": float("nan"),
            "top3_acc": float("nan"),
            "mean_confidence": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_precision": float("nan"),
            "weighted_recall": float("nan"),
            "weighted_f1": float("nan"),
            "balanced_acc": float("nan"),
            "skipped_non_finite_batches": 0.0,
        }
        val_pred_true_idx = np.asarray([], dtype=np.int64)
        val_pred_pred_idx = np.asarray([], dtype=np.int64)

    selection_manual = _selection_score_from_metrics(val_metrics, cfg)
    selection_predicted = (
        _selection_score_from_metrics(val_pred_metrics, cfg)
        if has_predicted_validation
        else float("nan")
    )

    n_classes = len(label_encoder.classes_)
    valid_mask = (val_true_idx >= 0) & (val_true_idx < n_classes) & (val_pred_idx >= 0) & (val_pred_idx < n_classes)
    invalid = int(np.count_nonzero(~valid_mask))
    if invalid > 0:
        print(f"[warning] Dropping {invalid} invalid validation rows.")
        val_true_idx = val_true_idx[valid_mask]
        val_pred_idx = val_pred_idx[valid_mask]

    if has_predicted_validation:
        valid_pred_mask = (
            (val_pred_true_idx >= 0)
            & (val_pred_true_idx < n_classes)
            & (val_pred_pred_idx >= 0)
            & (val_pred_pred_idx < n_classes)
        )
        invalid_pred = int(np.count_nonzero(~valid_pred_mask))
        if invalid_pred > 0:
            print(f"[warning] Dropping {invalid_pred} invalid predicted-mask validation rows.")
            val_pred_true_idx = val_pred_true_idx[valid_pred_mask]
            val_pred_pred_idx = val_pred_pred_idx[valid_pred_mask]

    if val_true_idx.size == 0:
        raise RuntimeError("No valid validation predictions available.")
    if has_predicted_validation and val_pred_true_idx.size == 0:
        raise RuntimeError("No valid predicted-mask validation predictions available.")

    true_labels = label_encoder.inverse_transform(val_true_idx)
    pred_labels = label_encoder.inverse_transform(val_pred_idx)
    if has_predicted_validation:
        true_labels_pred = label_encoder.inverse_transform(val_pred_true_idx)
        pred_labels_pred = label_encoder.inverse_transform(val_pred_pred_idx)
    else:
        true_labels_pred = np.asarray([], dtype=object)
        pred_labels_pred = np.asarray([], dtype=object)

    print(
        "Validation (manual masks) | "
        f"acc={val_metrics['cls_acc'] * 100:.2f}% "
        f"macro_f1={val_metrics['macro_f1']:.3f} "
        f"balanced_acc={val_metrics['balanced_acc']:.3f} "
        f"top3={val_metrics['top3_acc']:.3f} "
        f"selection={selection_manual:.4f}"
    )
    print(classification_report(true_labels, pred_labels, zero_division=0))

    if has_predicted_validation:
        print(
            "Validation (pred masks)   | "
            f"acc={val_pred_metrics['cls_acc'] * 100:.2f}% "
            f"macro_f1={val_pred_metrics['macro_f1']:.3f} "
            f"balanced_acc={val_pred_metrics['balanced_acc']:.3f} "
            f"top3={val_pred_metrics['top3_acc']:.3f} "
            f"selection={selection_predicted:.4f}"
        )
        print(classification_report(true_labels_pred, pred_labels_pred, zero_division=0))
    else:
        print("Validation (pred masks)   | skipped (stage 4 disabled or no predicted samples)")

    fig_size = max(8, 0.30 * len(label_encoder.classes_))
    if has_predicted_validation:
        fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
        ConfusionMatrixDisplay.from_predictions(
            true_labels, pred_labels, labels=label_encoder.classes_,
            xticks_rotation=90, colorbar=False, ax=axes[0],
        )
        axes[0].set_title(f"Manual-mask val acc: {val_metrics['cls_acc'] * 100:.1f}%")
        ConfusionMatrixDisplay.from_predictions(
            true_labels_pred, pred_labels_pred, labels=label_encoder.classes_,
            xticks_rotation=90, colorbar=False, ax=axes[1],
        )
        axes[1].set_title(f"Pred-mask val acc: {val_pred_metrics['cls_acc'] * 100:.1f}%")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        ConfusionMatrixDisplay.from_predictions(
            true_labels, pred_labels, labels=label_encoder.classes_,
            xticks_rotation=90, colorbar=False, ax=ax,
        )
        ax.set_title(f"Manual-mask val acc: {val_metrics['cls_acc'] * 100:.1f}%")
    plt.tight_layout()
    plt.show()

    history = state.get("history", [])
    if history:
        epochs_x = np.arange(1, len(history) + 1)
        train_acc = [float(r["train_cls_acc"]) for r in history]
        val_acc = [float(r["val_cls_acc"]) for r in history]
        val_pred_acc = [float(r.get("val_pred_cls_acc", float("nan"))) for r in history]
        val_macro_f1 = [float(r.get("val_macro_f1", float("nan"))) for r in history]
        val_pred_macro_f1 = [float(r.get("val_pred_macro_f1", float("nan"))) for r in history]
        val_bal_acc = [float(r.get("val_balanced_acc", float("nan"))) for r in history]
        val_pred_bal_acc = [float(r.get("val_pred_balanced_acc", float("nan"))) for r in history]
        val_top3 = [float(r.get("val_top3_acc", float("nan"))) for r in history]
        val_pred_top3 = [float(r.get("val_pred_top3_acc", float("nan"))) for r in history]
        selection_scores = [float(r.get("selection_score", float("nan"))) for r in history]
        train_loss = [float(r["train_loss"]) for r in history]
        val_loss = [float(r["val_loss"]) for r in history]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = np.asarray(axes).reshape(-1)
        axes[0].plot(epochs_x, train_acc, marker="o", label="train")
        axes[0].plot(epochs_x, val_acc, marker="o", label="val (manual)")
        axes[0].plot(epochs_x, val_pred_acc, marker="o", label="val (predicted)")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Classification accuracy")
        axes[0].set_xlabel("Global epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs_x, val_macro_f1, marker="o", label="macro F1 (manual)")
        axes[1].plot(epochs_x, val_pred_macro_f1, marker="o", label="macro F1 (pred)")
        axes[1].plot(epochs_x, val_bal_acc, marker="o", linestyle="--", label="balanced acc (manual)")
        axes[1].plot(epochs_x, val_pred_bal_acc, marker="o", linestyle="--", label="balanced acc (pred)")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Class-Balance Metrics")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("Score")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs_x, train_loss, marker="o", label="train")
        axes[2].plot(epochs_x, val_loss, marker="o", label="val")
        axes[2].set_title("Cross-entropy loss")
        axes[2].set_xlabel("Global epoch")
        axes[2].set_ylabel("Loss")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        axes[3].plot(epochs_x, val_top3, marker="o", label="top3 (manual)")
        axes[3].plot(epochs_x, val_pred_top3, marker="o", label="top3 (pred)")
        axes[3].plot(
            epochs_x,
            selection_scores,
            marker="o",
            linestyle="--",
            color="k",
            label=f"selection ({cfg.best_epoch_selection_metric})",
        )
        axes[3].set_ylim(0.0, 1.0)
        axes[3].set_title("Top-k And Selection")
        axes[3].set_xlabel("Global epoch")
        axes[3].set_ylabel("Score")
        axes[3].grid(alpha=0.3)
        axes[3].legend()

        for idx, r in enumerate(history, start=1):
            axes[0].annotate(
                str(r["stage"]).replace("stage", "S"),
                (idx, float(r["val_cls_acc"])),
                fontsize=7,
                alpha=0.65,
            )

        plt.tight_layout()
        plt.show()

    state["val_metrics"] = val_metrics
    state["val_pred_metrics"] = val_pred_metrics
    state["val_true_idx"] = val_true_idx
    state["val_pred_idx"] = val_pred_idx
    state["val_pred_true_idx"] = val_pred_true_idx
    state["val_pred_pred_idx"] = val_pred_pred_idx
    state["val_sample_indices"] = np.arange(val_true_idx.size, dtype=np.int64)
    state["true_labels"] = true_labels
    state["pred_labels"] = pred_labels
    state["true_labels_pred"] = true_labels_pred
    state["pred_labels_pred"] = pred_labels_pred
    state["val_selection_score_manual"] = float(selection_manual)
    state["val_selection_score_predicted"] = float(selection_predicted)
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
