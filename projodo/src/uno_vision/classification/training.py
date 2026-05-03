from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample as sk_resample

from uno_vision.classification.features import (
    IMG_SIZE,
    extract_color_features,
    extract_rank_features,
    letterbox_cv2,
    split_card_label,
)
from uno_vision.paths import AUGMENTATIONS_DIR, CLASSIFIER_CLASSES_DIR, CLASSIFIER_MODELS_DIR, REFERENCE_CARDS_DIR


@dataclass
class ClassifierTrainingResult:
    color_clf: ExtraTreesClassifier
    rank_clf: ExtraTreesClassifier
    label_encoder: LabelEncoder
    color_encoder: LabelEncoder
    rank_encoder: LabelEncoder
    pair_to_full_label: dict[tuple[str, str], str]
    metrics: dict[str, float]


def collect_classifier_samples(
    reference_cards_dir: Path = REFERENCE_CARDS_DIR,
    augmentations_dir: Path = AUGMENTATIONS_DIR,
) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    reference_csv = reference_cards_dir / "labels.csv"
    augmentation_csv = augmentations_dir / "labels.csv"
    augmentation_images_dir = augmentations_dir / "images"

    with reference_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"].strip()
            card = row["card"].strip()
            if not image_id or not card:
                continue
            tag, _, idx = image_id.rpartition("_crop_")
            img_path = reference_cards_dir / tag / "crops" / f"crop_{idx}.jpg"
            if img_path.is_file():
                samples.append((str(img_path), card))

    with augmentation_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"].strip()
            card = row["card"].strip()
            if not image_id or not card:
                continue
            img_path = augmentation_images_dir / f"{image_id}.jpg"
            if img_path.is_file():
                samples.append((str(img_path), card))
    return samples


def _build_label_encoders(labels: list[str]):
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    color_labels, rank_labels = zip(*(split_card_label(label) for label in labels))
    color_encoder = LabelEncoder()
    rank_encoder = LabelEncoder()
    y_color = color_encoder.fit_transform(color_labels)
    y_rank = rank_encoder.fit_transform(rank_labels)
    pair_to_full_label = {(color, rank): full for color, rank, full in zip(color_labels, rank_labels, labels)}
    return label_encoder, encoded, color_encoder, rank_encoder, y_color, y_rank, pair_to_full_label


def _extract_feature_matrices(paths: list[str]):
    color_feats = []
    rank_feats = []
    letterboxed_images = []
    for path in paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img_lb = letterbox_cv2(img_bgr, size=IMG_SIZE)
        letterboxed_images.append(img_lb)
        color_feats.append(extract_color_features(img_lb))
        rank_feats.append(extract_rank_features(img_lb))
    return np.array(color_feats), np.array(rank_feats), letterboxed_images


def _oversample_features(features: np.ndarray, labels: np.ndarray, random_state: int):
    counts = np.bincount(labels)
    max_count = int(counts.max())
    feature_parts = [features]
    label_parts = [labels]
    for cls in np.unique(labels):
        deficit = max_count - int(counts[cls])
        if deficit <= 0:
            continue
        mask = labels == cls
        over_features, over_labels = sk_resample(
            features[mask],
            labels[mask],
            replace=True,
            n_samples=deficit,
            random_state=random_state,
        )
        feature_parts.append(over_features)
        label_parts.append(over_labels)
    return np.vstack(feature_parts), np.concatenate(label_parts)


def decode_predictions(pred_c, pred_r, color_encoder, rank_encoder, pair_to_full_label):
    rank_to_full_labels: dict[str, set[str]] = {}
    for (_, rank), full in pair_to_full_label.items():
        rank_to_full_labels.setdefault(rank, set()).add(full)
    rank_unique = {rank: next(iter(fulls)) for rank, fulls in rank_to_full_labels.items() if len(fulls) == 1}

    results = []
    for color, rank in zip(color_encoder.inverse_transform(pred_c), rank_encoder.inverse_transform(pred_r)):
        key = (color, rank)
        if key in pair_to_full_label:
            results.append(pair_to_full_label[key])
        elif rank in rank_unique:
            results.append(rank_unique[rank])
        else:
            candidates = [full for (_, candidate_rank), full in pair_to_full_label.items() if candidate_rank == rank]
            results.append(candidates[0] if candidates else f"{color}_{rank}")
    return results


def train_classifiers(
    samples: list[tuple[str, str]] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ClassifierTrainingResult:
    samples = samples or collect_classifier_samples()
    if not samples:
        raise RuntimeError("No classifier samples found.")
    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    label_encoder, encoded, color_encoder, rank_encoder, y_color, y_rank, pair_to_full_label = _build_label_encoders(labels)
    x_color, x_rank, letterboxed_images = _extract_feature_matrices(paths)

    idx_all = np.arange(len(paths))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded,
    )
    x_col_tr, x_col_va = x_color[train_idx], x_color[val_idx]
    x_rank_tr, x_rank_va = x_rank[train_idx], x_rank[val_idx]
    y_col_tr, y_col_va = y_color[train_idx], y_color[val_idx]
    y_rnk_tr, y_rnk_va = y_rank[train_idx], y_rank[val_idx]
    y_full_va = encoded[val_idx]

    rot_flags = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    aug_col = []
    aug_y_col = []
    for idx in train_idx:
        img = letterboxed_images[idx]
        for flag in rot_flags:
            rotated = cv2.rotate(img, flag)
            aug_col.append(extract_color_features(rotated))
            aug_y_col.append(y_color[idx])
    x_col_tr = np.vstack([x_col_tr, np.array(aug_col)])
    y_col_tr = np.concatenate([y_col_tr, np.array(aug_y_col)])

    rng = np.random.default_rng(random_state)
    aug_rank = []
    aug_y_rank = []
    for idx in train_idx:
        img = letterboxed_images[idx]
        angle = float(rng.uniform(-8.0, 8.0))
        scale = float(rng.uniform(0.95, 1.05))
        tx = int(rng.integers(-6, 7))
        ty = int(rng.integers(-6, 7))
        matrix = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, scale)
        matrix[:, 2] += [tx, ty]
        warped = cv2.warpAffine(
            img,
            matrix,
            (IMG_SIZE, IMG_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
        )
        alpha = float(rng.uniform(0.9, 1.1))
        beta = int(rng.integers(-12, 13))
        photometric = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)
        photometric = cv2.GaussianBlur(photometric, (3, 3), 0)
        aug_rank.append(extract_rank_features(photometric))
        aug_y_rank.append(y_rank[idx])
    x_rank_tr = np.vstack([x_rank_tr, np.array(aug_rank)])
    y_rnk_tr = np.concatenate([y_rnk_tr, np.array(aug_y_rank)])

    x_col_tr, y_col_tr = _oversample_features(x_col_tr, y_col_tr, random_state)
    x_rank_tr, y_rnk_tr = _oversample_features(x_rank_tr, y_rnk_tr, random_state)

    color_clf = ExtraTreesClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    color_clf.fit(x_col_tr, y_col_tr)

    rank_clf = ExtraTreesClassifier(
        n_estimators=900,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    rank_clf.fit(x_rank_tr, y_rnk_tr)

    pred_col = color_clf.predict(x_col_va)
    pred_rnk = rank_clf.predict(x_rank_va)
    pred_full = decode_predictions(pred_col, pred_rnk, color_encoder, rank_encoder, pair_to_full_label)
    true_full = label_encoder.inverse_transform(y_full_va)
    metrics = {
        "color_train_accuracy": float(color_clf.score(x_col_tr, y_col_tr)),
        "color_val_accuracy": float(color_clf.score(x_col_va, y_col_va)),
        "rank_train_accuracy": float(rank_clf.score(x_rank_tr, y_rnk_tr)),
        "rank_val_accuracy": float(rank_clf.score(x_rank_va, y_rnk_va)),
        "full_val_accuracy": float(accuracy_score(true_full, pred_full)),
    }
    return ClassifierTrainingResult(
        color_clf=color_clf,
        rank_clf=rank_clf,
        label_encoder=label_encoder,
        color_encoder=color_encoder,
        rank_encoder=rank_encoder,
        pair_to_full_label=pair_to_full_label,
        metrics=metrics,
    )


def save_classifier_artifacts(
    result: ClassifierTrainingResult,
    classifier_dir: Path = CLASSIFIER_MODELS_DIR,
    classes_dir: Path = CLASSIFIER_CLASSES_DIR,
) -> None:
    classifier_dir.mkdir(parents=True, exist_ok=True)
    classes_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.color_clf, classifier_dir / "color_clf.pkl")
    joblib.dump(result.rank_clf, classifier_dir / "rank_clf.pkl")
    np.save(classes_dir / "label_classes.npy", result.label_encoder.classes_)
    np.save(classes_dir / "color_classes.npy", result.color_encoder.classes_)
    np.save(classes_dir / "rank_classes.npy", result.rank_encoder.classes_)