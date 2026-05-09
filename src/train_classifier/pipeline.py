"""Training orchestrator for the feature-based (non-CNN) DO card classifier.

Public surface used by the notebook:
    TrainPipelineConfig
    initialize_training_pipeline(config) -> state
    run_feature_extraction(state) -> state
    run_training(state) -> state
    save_training_artifacts(state) -> state
    run_validation_diagnostics(state) -> state
    plot_sample_preview(state)
    plot_wrong_predictions(state)
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog as skimage_hog
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.shared.card_data import assert_no_test_inputs, reference_crop_path
from src.shared.card_pipeline import (
    compose_masked_card_image,
    crop_with_margin,
    find_workspace_root,
    is_reasonable_scene_bbox,
)


# --------------------------------------------------------------------------- #
# Config & sample dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CardSample:
    source: str
    label: str
    image_path: Path
    bbox: tuple[int, int, int, int] | None = None
    scene_mask_path: Path | None = None
    mask_path: Path | None = None


def _default_source_weights() -> dict[str, float]:
    return {"reference": 1.15, "augmented_card": 1.0, "augmented_scene": 0.75}


def _default_candidate_settings() -> list[dict[str, Any]]:
    return [
        {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 700, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 900, "max_features": "sqrt", "min_samples_leaf": 2},
        {"n_estimators": 700, "max_features": "log2", "min_samples_leaf": 1},
    ]


@dataclass
class TrainPipelineConfig:
    seed: int = 42
    img_size: int = 128
    fourier_coeffs: int = 32
    center_crop_fraction: float = 0.62
    min_scene_box_side: int = 28
    max_scene_box_aspect: float = 4.5
    val_split: float = 0.20
    bbox_margin: float = 0.08
    source_weights: dict[str, float] = field(default_factory=_default_source_weights)
    candidate_settings: list[dict[str, Any]] = field(default_factory=_default_candidate_settings)
    preview_count: int = 12
    wrong_show_count: int = 16


# --------------------------------------------------------------------------- #
# Helpers (pure functions, no global state)
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _clean_label(label: str) -> str:
    return str(label).strip()


def _resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _read_sample_bgr_and_mask(sample: CardSample, bbox_margin: float) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {sample.image_path}")

    if sample.bbox is not None:
        img_bgr = crop_with_margin(img_bgr, sample.bbox, margin_fraction=bbox_margin)

    if sample.source == "augmented_scene":
        if sample.scene_mask_path is None or sample.bbox is None:
            raise ValueError("augmented_scene sample requires scene_mask_path and bbox.")
        scene_mask = cv2.imread(str(sample.scene_mask_path), cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(f"Cannot read scene mask: {sample.scene_mask_path}")
        mask_u8 = crop_with_margin(scene_mask, sample.bbox, margin_fraction=bbox_margin)
        mask_u8 = np.where(mask_u8 > 127, 255, 0).astype(np.uint8)

    elif sample.source == "augmented_card" and sample.mask_path is not None:
        mask_u8 = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise FileNotFoundError(f"Cannot read augmented-card mask: {sample.mask_path}")
        if mask_u8.shape[:2] != img_bgr.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_u8 = np.where(mask_u8 > 127, 255, 0).astype(np.uint8)

    else:
        mask_u8 = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    if mask_u8.shape[:2] != img_bgr.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    if int(np.count_nonzero(mask_u8)) < 20:
        mask_u8 = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    return img_bgr, mask_u8


def _letterbox_cv2(img_bgr: np.ndarray, size: int, fill: int = 128) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _center_crop(img_bgr: np.ndarray, fraction: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    fh = max(1, int(round(h * fraction)))
    fw = max(1, int(round(w * fraction)))
    y0 = max(0, (h - fh) // 2)
    x0 = max(0, (w - fw) // 2)
    return img_bgr[y0:y0 + fh, x0:x0 + fw]


def _extract_color_feat(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturated = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    if int(saturated.sum()) == 0:
        saturated = None
    h_hist = cv2.calcHist([hsv], [0], saturated, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], saturated, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], saturated, [8], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return feat / (feat.sum() + 1e-6)


def _extract_hog_feat(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return skimage_hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).astype(np.float32)


def _extract_fourier_feat(img_bgr: np.ndarray, n_coeffs: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = 8
    bw[:border, :] = 0
    bw[-border:, :] = 0
    bw[:, :border] = 0
    bw[:, -border:] = 0

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_coeffs, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float32)
    if len(pts) < max(10, n_coeffs + 1):
        return np.zeros(n_coeffs, dtype=np.float32)

    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - np.mean(z)
    magnitude = np.abs(np.fft.fft(z))[1:n_coeffs + 1]
    if len(magnitude) < n_coeffs:
        magnitude = np.pad(magnitude, (0, n_coeffs - len(magnitude)))
    scale = magnitude[0] if magnitude[0] > 1e-6 else (np.mean(magnitude) + 1e-6)
    magnitude = magnitude / (scale + 1e-6)
    return magnitude.astype(np.float32)


def _extract_features(img_bgr: np.ndarray, cfg: TrainPipelineConfig) -> np.ndarray:
    img_lb = _letterbox_cv2(img_bgr, cfg.img_size)
    img_center = _center_crop(img_lb, cfg.center_crop_fraction)
    return np.concatenate([
        _extract_color_feat(img_lb),
        _extract_hog_feat(img_lb),
        _extract_fourier_feat(img_lb, cfg.fourier_coeffs),
        _extract_hog_feat(img_center),
        _extract_fourier_feat(img_center, cfg.fourier_coeffs),
    ]).astype(np.float32)


# R4 compliance: this classifier is a tree ensemble (no trainable params).
# Param-cap rule applies to neural-net models; we still print a sanity note.
def _assert_param_cap_note() -> None:
    print("[R4] ExtraTreesClassifier has no trainable params; param cap not applicable.")


# --------------------------------------------------------------------------- #
# Sample loading
# --------------------------------------------------------------------------- #


def _load_reference_samples(reference_csv: Path, ref_cards_dir: Path) -> tuple[list[CardSample], list[str]]:
    samples: list[CardSample] = []
    missing: list[str] = []
    reference_labels: dict[str, str] = {}
    with reference_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row["image_id"]).strip()
            label = _clean_label(row["card"])
            if image_id and label:
                reference_labels[image_id] = label

    for image_id, label in reference_labels.items():
        image_path = reference_crop_path(image_id, ref_cards_dir)
        if image_path.is_file():
            samples.append(CardSample("reference", label, image_path))
        else:
            missing.append(str(image_path))
    return samples, missing


def _load_augmented_card_samples(
    aug_csv: Path,
    aug_cards_dir: Path,
    aug_masks_dir: Path,
    project_root: Path,
) -> tuple[list[CardSample], list[str], int]:
    samples: list[CardSample] = []
    missing: list[str] = []
    missing_masks = 0
    if not aug_csv.is_file():
        print(f"[warning] Augmented card CSV not found: {aug_csv}")
        return samples, missing, missing_masks
    with aug_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row["image_id"]).strip()
            label = _clean_label(row["card"])
            if not image_id or not label or label == "token":
                continue
            image_path: Path | None = None
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = aug_cards_dir / f"{image_id}{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break

            if image_path is not None:
                mask_path: Path | None = None

                mask_path_raw = str(row.get("mask_path", "")).strip()
                if mask_path_raw:
                    candidate = Path(mask_path_raw)
                    resolved = candidate if candidate.is_absolute() else (project_root / candidate)
                    if resolved.is_file():
                        mask_path = resolved

                if mask_path is None:
                    for suffix in (".png", ".jpg", ".jpeg"):
                        candidate = aug_masks_dir / f"{image_id}{suffix}"
                        if candidate.is_file():
                            mask_path = candidate
                            break

                if mask_path is None:
                    missing_masks += 1

                samples.append(CardSample("augmented_card", label, image_path, mask_path=mask_path))
            else:
                missing.append(str(aug_cards_dir / f"{image_id}.png"))
    return samples, missing, missing_masks


def _load_scene_samples(
    scene_labels_path: Path,
    scene_images_dir: Path,
    scene_masks_dir: Path,
    project_root: Path,
    cfg: TrainPipelineConfig,
) -> tuple[list[CardSample], list[str], int, int, int]:
    samples: list[CardSample] = []
    missing: list[str] = []
    skipped_labels = 0
    skipped_boxes = 0
    skipped_masks = 0
    if not scene_labels_path.is_file():
        print(f"[warning] Augmented scene labels not found: {scene_labels_path}")
        return samples, missing, skipped_labels, skipped_boxes, skipped_masks

    valid_mask_ext = {".png", ".jpg", ".jpeg"}
    valid_img_ext = (".png", ".jpg", ".jpeg")
    scene_mask_by_stem = {
        mask_path.stem: mask_path
        for mask_path in sorted(scene_masks_dir.iterdir())
        if mask_path.suffix.lower() in valid_mask_ext
    }

    with scene_labels_path.open("r", encoding="utf-8") as labels_file:
        scene_metadata = json.load(labels_file)

    for scene_entry in scene_metadata:
        scene_name = str(scene_entry.get("scene", "")).strip()
        default_scene_path = scene_images_dir / f"{scene_name}.png"
        for suffix in valid_img_ext:
            candidate = scene_images_dir / f"{scene_name}{suffix}"
            if candidate.is_file():
                default_scene_path = candidate
                break

        scene_path = _resolve_project_path(
            scene_entry.get("image_path", default_scene_path),
            project_root,
        )
        if not scene_path.is_file():
            missing.append(str(scene_path))
            continue

        scene_mask_path = scene_mask_by_stem.get(scene_path.stem) or scene_mask_by_stem.get(scene_name)

        for card in scene_entry.get("cards", []):
            label = _clean_label(card.get("label", ""))
            bbox_raw = card.get("bbox", [])
            if not label or label == "token":
                skipped_labels += 1
                continue
            if len(bbox_raw) != 4:
                skipped_boxes += 1
                continue
            bbox = tuple(map(int, bbox_raw))
            if not is_reasonable_scene_bbox(
                bbox, min_side=cfg.min_scene_box_side, max_aspect=cfg.max_scene_box_aspect
            ):
                skipped_boxes += 1
                continue
            if scene_mask_path is None or not scene_mask_path.is_file():
                skipped_masks += 1
                continue
            samples.append(
                CardSample(
                    "augmented_scene",
                    label,
                    scene_path,
                    bbox,
                    scene_mask_path=scene_mask_path,
                )
            )
    return samples, missing, skipped_labels, skipped_boxes, skipped_masks


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


def initialize_training_pipeline(config: TrainPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or TrainPipelineConfig()
    _seed_everything(cfg.seed)

    project_root = find_workspace_root()
    project_dir = project_root / "project"
    training_data = project_dir / "training_data"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    reference_csv = training_data / "object_labels" / "reference_cards" / "reference_do.csv"
    aug_csv = training_data / "object_labels" / "augmented_cards" / "aug.csv"
    scene_labels_path = training_data / "object_labels" / "augmented_scenes" / "labels.json"

    ref_cards_dir = training_data / "training_images" / "reference_cards"
    aug_cards_dir = training_data / "training_images" / "augmented_cards"
    aug_masks_dir = training_data / "training_masks" / "augmented_cards"
    scene_images_dir = training_data / "training_images" / "augmented_scenes"
    scene_masks_dir = training_data / "training_masks" / "augmented_scenes"

    card_clf_path = models_dir / "card_clf_do.pkl"
    card_classes_path = models_dir / "card_classes_do.npy"
    card_config_path = models_dir / "card_classifier_do_config.json"

    for required_path in (reference_csv, ref_cards_dir, aug_masks_dir, scene_masks_dir):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required classifier input: {required_path}")

    # R3: hard fail if any training input lives under a "test" path.
    assert_no_test_inputs(
        [
            reference_csv,
            aug_csv,
            scene_labels_path,
            ref_cards_dir,
            aug_cards_dir,
            aug_masks_dir,
            scene_images_dir,
            scene_masks_dir,
        ]
    )

    print(f"Project root: {project_root}")
    print(f"Reference labels: {reference_csv}")
    print(f"Augmented labels: {aug_csv}")
    print(f"Scene labels:     {scene_labels_path}")
    print(f"Augmented masks:  {aug_masks_dir}")
    print(f"Scene masks:      {scene_masks_dir}")
    print(f"Model output:     {card_clf_path}")

    samples: list[CardSample] = []
    missing: list[str] = []

    ref_samples, ref_missing = _load_reference_samples(reference_csv, ref_cards_dir)
    samples.extend(ref_samples)
    missing.extend(ref_missing)

    aug_samples, aug_missing, missing_aug_masks = _load_augmented_card_samples(
        aug_csv, aug_cards_dir, aug_masks_dir, project_root
    )
    samples.extend(aug_samples)
    missing.extend(aug_missing)

    scene_samples, scene_missing, skipped_scene_labels, skipped_scene_boxes, skipped_scene_masks = _load_scene_samples(
        scene_labels_path, scene_images_dir, scene_masks_dir, project_root, cfg
    )
    samples.extend(scene_samples)
    missing.extend(scene_missing)

    source_counts = Counter(s.source for s in samples)
    label_counts = Counter(s.label for s in samples)

    print(f"Total samples: {len(samples)}")
    print(f"By source: {dict(source_counts)}")
    print(f"Classes: {len(label_counts)}")
    print(f"Smallest class counts: {label_counts.most_common()[-8:]}")
    print(f"Augmented cards missing masks: {missing_aug_masks}")
    print(f"Skipped scene labels: {skipped_scene_labels}")
    print(f"Skipped scene boxes:  {skipped_scene_boxes}")
    print(f"Skipped scene masks:  {skipped_scene_masks}")
    if missing:
        print(f"Missing files skipped: {len(missing)}")
        for item in missing[:5]:
            print(f"  {item}")

    if not samples:
        raise RuntimeError("No classifier samples were found. Run the reference and augmentation notebooks first.")

    return {
        "config": cfg,
        "project_root": project_root,
        "models_dir": models_dir,
        "card_clf_path": card_clf_path,
        "card_classes_path": card_classes_path,
        "card_config_path": card_config_path,
        "samples": samples,
        "missing_files": missing,
        "source_counts": source_counts,
        "label_counts": label_counts,
        "missing_aug_masks": missing_aug_masks,
        "skipped_scene_labels": skipped_scene_labels,
        "skipped_scene_boxes": skipped_scene_boxes,
        "skipped_scene_masks": skipped_scene_masks,
    }


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def run_feature_extraction(state: dict[str, Any]) -> dict[str, Any]:
    cfg: TrainPipelineConfig = state["config"]
    samples: list[CardSample] = state["samples"]

    features: list[np.ndarray] = []
    labels: list[str] = []
    valid_samples: list[CardSample] = []
    sample_sources: list[str] = []

    print(f"Extracting features from {len(samples)} samples...")
    for index, sample in enumerate(samples, start=1):
        img_bgr, mask_u8 = _read_sample_bgr_and_mask(sample, cfg.bbox_margin)
        img_bgr = compose_masked_card_image(img_bgr, mask_u8)
        if img_bgr.size == 0:
            continue
        features.append(_extract_features(img_bgr, cfg))
        labels.append(sample.label)
        valid_samples.append(sample)
        sample_sources.append(sample.source)
        if index % 500 == 0 or index == len(samples):
            print(f"  {index}/{len(samples)}")

    X = np.vstack(features)
    labels_arr = np.array(labels)
    sample_sources_arr = np.array(sample_sources)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_arr)

    print(f"Feature matrix: {X.shape}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    state.update({
        "X": X,
        "y": y,
        "labels_arr": labels_arr,
        "sample_sources_arr": sample_sources_arr,
        "valid_samples": valid_samples,
        "label_encoder": label_encoder,
    })
    return state


# --------------------------------------------------------------------------- #
# Training (candidate sweep)
# --------------------------------------------------------------------------- #


def run_training(state: dict[str, Any]) -> dict[str, Any]:
    cfg: TrainPipelineConfig = state["config"]
    X: np.ndarray = state["X"]
    y: np.ndarray = state["y"]
    sample_sources_arr: np.ndarray = state["sample_sources_arr"]

    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    stratify_y = y if min_class_count >= 2 else None
    if stratify_y is None:
        print("[warning] Some classes have only one sample; validation split will not be stratified.")

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=cfg.val_split,
        random_state=cfg.seed,
        stratify=stratify_y,
    )

    per_class_weight = {
        int(class_id): len(y) / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    source_weight_values = np.array(
        [cfg.source_weights.get(s, 1.0) for s in sample_sources_arr], dtype=np.float32
    )
    class_weight_values = np.array(
        [per_class_weight[int(class_id)] for class_id in y], dtype=np.float32
    )
    sample_weights = source_weight_values * class_weight_values

    print(
        f"Sample weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f} "
        f"(mean={sample_weights.mean():.3f})"
    )

    _assert_param_cap_note()

    best_val_acc = -1.0
    best_bundle: tuple[Any, dict[str, Any], np.ndarray] | None = None

    for settings in cfg.candidate_settings:
        candidate = ExtraTreesClassifier(
            random_state=cfg.seed,
            n_jobs=-1,
            class_weight=None,
            **settings,
        )
        # R2: ExtraTrees has no pretrained weights; nothing to disable.
        candidate.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
        candidate_val_pred = candidate.predict(X[val_idx])
        candidate_val_acc = accuracy_score(y[val_idx], candidate_val_pred)
        print(f"Candidate {settings} -> val acc {candidate_val_acc * 100:.2f}%")

        if candidate_val_acc > best_val_acc:
            best_val_acc = candidate_val_acc
            best_bundle = (candidate, settings, candidate_val_pred)

    if best_bundle is None:
        raise RuntimeError("No candidate model was trained.")

    card_clf, best_params, val_pred = best_bundle
    train_pred = card_clf.predict(X[train_idx])
    train_acc = accuracy_score(y[train_idx], train_pred)
    val_acc = accuracy_score(y[val_idx], val_pred)

    print(f"Selected settings: {best_params}")
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Train accuracy: {train_acc * 100:.1f}%")
    print(f"Val accuracy:   {val_acc * 100:.1f}%")

    state.update({
        "card_clf": card_clf,
        "best_params": best_params,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "val_pred": val_pred,
        "train_pred": train_pred,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "sample_weights": sample_weights,
    })
    return state


# --------------------------------------------------------------------------- #
# Artifact saving
# --------------------------------------------------------------------------- #


def save_training_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    cfg: TrainPipelineConfig = state["config"]
    card_clf = state["card_clf"]
    label_encoder: LabelEncoder = state["label_encoder"]
    card_clf_path: Path = state["card_clf_path"]
    card_classes_path: Path = state["card_classes_path"]
    card_config_path: Path = state["card_config_path"]

    joblib.dump(card_clf, card_clf_path)
    np.save(card_classes_path, label_encoder.classes_)

    config = {
        "model_file": card_clf_path.name,
        "classes_file": card_classes_path.name,
        "image_size": cfg.img_size,
        "fourier_coeffs": cfg.fourier_coeffs,
        "center_crop_fraction": cfg.center_crop_fraction,
        "feature_order": ["hsv_histogram", "hog_global", "fourier_global", "hog_center", "fourier_center"],
        "classifier": "ExtraTreesClassifier",
        "mask_guided_features": True,
        "selected_params": state["best_params"],
        "source_weights": cfg.source_weights,
        "n_samples": int(len(state["valid_samples"])),
        "source_counts": dict(state["source_counts"]),
        "validation_accuracy": float(state["val_acc"]),
        "seed": cfg.seed,
    }
    with card_config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    print(f"Saved classifier: {card_clf_path}")
    print(f"Saved classes:    {card_classes_path}")
    print(f"Saved config:     {card_config_path}")

    state["saved_config"] = config
    return state


# --------------------------------------------------------------------------- #
# Diagnostics & visualization
# --------------------------------------------------------------------------- #


def run_validation_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    label_encoder: LabelEncoder = state["label_encoder"]
    y: np.ndarray = state["y"]
    val_idx: np.ndarray = state["val_idx"]
    val_pred: np.ndarray = state["val_pred"]
    val_acc: float = state["val_acc"]

    true_labels = label_encoder.inverse_transform(y[val_idx])
    pred_labels = label_encoder.inverse_transform(val_pred)

    print(classification_report(true_labels, pred_labels, zero_division=0))

    fig_size = max(8, 0.32 * len(label_encoder.classes_))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        pred_labels,
        labels=label_encoder.classes_,
        xticks_rotation=90,
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"DO card classifier validation accuracy: {val_acc * 100:.1f}%")
    plt.tight_layout()
    plt.show()

    state["true_labels"] = true_labels
    state["pred_labels"] = pred_labels
    return state


def plot_sample_preview(state: dict[str, Any]) -> None:
    cfg: TrainPipelineConfig = state["config"]
    samples: list[CardSample] = state["samples"]
    if not samples:
        print("No samples to preview.")
        return

    preview_count = min(int(cfg.preview_count), len(samples))
    rng = np.random.default_rng(cfg.seed)
    preview_indices = rng.choice(len(samples), size=preview_count, replace=False)

    cols = 4
    rows = int(np.ceil(preview_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, sample_index in zip(axes, preview_indices):
        sample = samples[int(sample_index)]
        img_bgr, mask_u8 = _read_sample_bgr_and_mask(sample, cfg.bbox_margin)
        img_bgr = compose_masked_card_image(img_bgr, mask_u8)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{sample.label} ({sample.source})", fontsize=9)
        ax.axis("off")

    for ax in axes[preview_count:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_wrong_predictions(state: dict[str, Any]) -> None:
    cfg: TrainPipelineConfig = state["config"]
    val_idx: np.ndarray = state["val_idx"]
    true_labels = state["true_labels"]
    pred_labels = state["pred_labels"]
    valid_samples: list[CardSample] = state["valid_samples"]

    wrong = [
        (sample_index, true_label, pred_label)
        for sample_index, true_label, pred_label in zip(val_idx, true_labels, pred_labels)
        if true_label != pred_label
    ]
    print(f"Wrong validation predictions: {len(wrong)} / {len(val_idx)}")
    if not wrong:
        return

    show_count = min(int(cfg.wrong_show_count), len(wrong))
    cols = 4
    rows = int(np.ceil(show_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (sample_index, true_label, pred_label) in zip(axes, wrong[:show_count]):
        sample = valid_samples[int(sample_index)]
        img_bgr, mask_u8 = _read_sample_bgr_and_mask(sample, cfg.bbox_margin)
        img_bgr = compose_masked_card_image(img_bgr, mask_u8)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"true: {true_label}\npred: {pred_label}", color="red", fontsize=9)
        ax.axis("off")

    for ax in axes[show_count:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
