"""Inference / benchmarking orchestrator for the classical (non-CNN) DO card
classifier (HSV histogram + HOG + Fourier descriptors).

Public surface (used by test_classifier_do.ipynb):
    TestPipelineConfig
    initialize_test_pipeline(config) -> state
    run_single_image_diagnostics(state, threshold=None) -> state
    run_labeled_benchmark(state, benchmark_config=None) -> benchmark dict
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from skimage.feature import hog as skimage_hog

from src.shared.card_models import SceneUNetSmall, assert_param_cap
from src.shared.card_pipeline import (
    assign_region,
    boxes_from_probability,
    box_iou,
    crop_with_margin,
    find_workspace_root,
    format_cards,
    segment_scene_probability,
)


@dataclass
class TestPipelineConfig:
    image_source: str = "augmented_scene"
    image_index: int = 0
    image_id: str | None = None

    img_size_classifier: int = 128
    img_size_segmenter: int = 256
    fourier_coeffs: int = 32
    center_crop_fraction: float = 0.62

    eval_threshold: float = 0.50


REGION_COLORS = {
    "center": "gold",
    "p1": "tab:blue",
    "p2": "tab:green",
    "p3": "tab:red",
    "p4": "tab:purple",
}


# ---------------------------------------------------------------------------
# Classical feature extractors
# ---------------------------------------------------------------------------

def letterbox_cv2(img_bgr: np.ndarray, size: int, fill: int = 128) -> np.ndarray:
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


def center_crop(img_bgr: np.ndarray, fraction: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    fh = max(1, int(round(h * fraction)))
    fw = max(1, int(round(w * fraction)))
    y0 = max(0, (h - fh) // 2)
    x0 = max(0, (w - fw) // 2)
    return img_bgr[y0:y0 + fh, x0:x0 + fw]


def extract_color_feat(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturated = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    if int(saturated.sum()) == 0:
        saturated = None
    h_hist = cv2.calcHist([hsv], [0], saturated, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], saturated, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], saturated, [8], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return feat / (feat.sum() + 1e-6)


def extract_hog_feat(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return skimage_hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).astype(np.float32)


def extract_fourier_feat(img_bgr: np.ndarray, n_coeffs: int) -> np.ndarray:
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


def extract_features(
    img_bgr: np.ndarray,
    img_size: int,
    fourier_coeffs: int,
    center_crop_fraction: float,
    expected_dim: int | None,
) -> np.ndarray:
    img_lb = letterbox_cv2(img_bgr, size=img_size)
    global_feat = np.concatenate([
        extract_color_feat(img_lb),
        extract_hog_feat(img_lb),
        extract_fourier_feat(img_lb, n_coeffs=fourier_coeffs),
    ]).astype(np.float32)

    if expected_dim is None or expected_dim == int(global_feat.size):
        return global_feat

    img_center = center_crop(img_lb, fraction=center_crop_fraction)
    combined_feat = np.concatenate([
        global_feat,
        extract_hog_feat(img_center),
        extract_fourier_feat(img_center, n_coeffs=fourier_coeffs),
    ]).astype(np.float32)

    if expected_dim == int(combined_feat.size):
        return combined_feat

    raise ValueError(
        f"Unsupported feature size for classifier: expected {expected_dim}, "
        f"supported {int(global_feat.size)} or {int(combined_feat.size)}."
    )


def decode_card_prediction(pred_value: object, class_names: np.ndarray) -> str:
    if isinstance(pred_value, (int, np.integer)):
        pred_index = int(pred_value)
        if 0 <= pred_index < len(class_names):
            return str(class_names[pred_index])
    return str(pred_value)


def predict_card_label_and_confidence(
    model: object,
    class_names: np.ndarray,
    feature_2d: np.ndarray,
) -> tuple[str, float | None]:
    pred_value = model.predict(feature_2d)[0]
    label = decode_card_prediction(pred_value, class_names)
    confidence: float | None = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(feature_2d)))
    return label, confidence


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_test_pipeline(config: TestPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or TestPipelineConfig()

    project_root = find_workspace_root()
    project_dir = project_root / "project"
    training_data = project_dir / "training_data"
    models_dir = project_dir / "models"
    challenge_dir = project_root / "data" / "iapr-26-uno-vision-challenge"

    scene_labels_path = training_data / "object_labels" / "augmented_scenes" / "labels.json"
    train_images_dir = challenge_dir / "train_images"
    test_images_dir = challenge_dir / "test_images"

    seg_model_path = models_dir / "scene_segmenter_unet_small.pth"
    card_clf_path = models_dir / "card_clf_do.pkl"
    card_classes_path = models_dir / "card_classes_do.npy"
    card_config_path = models_dir / "card_classifier_do_config.json"

    print(f"Project root: {project_root}")
    print(f"Segmenter:    {seg_model_path}")
    print(f"Classifier:   {card_clf_path}")
    print(f"Classes:      {card_classes_path}")
    print(f"Config:       {card_config_path}")

    for path in (seg_model_path, card_clf_path, card_classes_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}. Run the DO training notebooks first.")

    # Resolve runtime classifier hyperparameters from the saved config (training-time
    # truth wins over local defaults to prevent drift).
    img_size_classifier = int(cfg.img_size_classifier)
    fourier_coeffs = int(cfg.fourier_coeffs)
    center_crop_fraction = float(cfg.center_crop_fraction)
    feature_order: list[str] | None = None

    classifier_config: dict[str, Any] = {}
    if card_config_path.is_file():
        with card_config_path.open("r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if isinstance(loaded_config, dict):
            classifier_config = loaded_config
    else:
        print(f"[warning] Classifier config not found: {card_config_path}")

    if classifier_config:
        if "image_size" in classifier_config:
            img_size_classifier = int(classifier_config["image_size"])
        if "fourier_coeffs" in classifier_config:
            fourier_coeffs = int(classifier_config["fourier_coeffs"])
        if "center_crop_fraction" in classifier_config:
            center_crop_fraction = float(classifier_config["center_crop_fraction"])
    raw_feature_order = classifier_config.get("feature_order")
    if isinstance(raw_feature_order, list):
        feature_order = [str(name) for name in raw_feature_order]
        print(f"Classifier feature order: {feature_order}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Segmenter (compliance: R1/R2 architecture cap enforced by assert_param_cap).
    segmenter = SceneUNetSmall().to(device)
    n_params = assert_param_cap(segmenter, "SceneUNetSmall (segmenter)")
    print(f"[compliance] SceneUNetSmall: {n_params:,} trainable params")
    segmenter.load_state_dict(torch.load(str(seg_model_path), map_location=device))
    segmenter.eval()

    # Classical classifier + class names.
    card_clf = joblib.load(card_clf_path)
    card_classes = np.load(card_classes_path, allow_pickle=True)

    expected_features_attr = int(getattr(card_clf, "n_features_in_", -1))
    classifier_expected_features: int | None = (
        expected_features_attr if expected_features_attr > 0 else None
    )
    if classifier_expected_features is None:
        print("[warning] Could not read n_features_in_ from classifier. Using default feature layout.")
    else:
        print(f"Classifier expects {classifier_expected_features} features per crop.")
        # Probe to fail fast on an unsupported feature layout.
        probe = np.zeros((96, 64, 3), dtype=np.uint8)
        _ = extract_features(
            probe,
            img_size=img_size_classifier,
            fourier_coeffs=fourier_coeffs,
            center_crop_fraction=center_crop_fraction,
            expected_dim=classifier_expected_features,
        )

    print(f"Loaded card classifier with {len(card_classes)} classes.")
    print(f"Classifier image size: {img_size_classifier}")
    print(f"Classifier Fourier coeffs: {fourier_coeffs}")
    print(f"Classifier center-crop fraction: {center_crop_fraction}")
    print(f"Device: {device}")

    # Pick image to diagnose.
    scene_truth = None
    if cfg.image_source == "augmented_scene":
        if not scene_labels_path.is_file():
            raise FileNotFoundError(f"No augmented-scene metadata: {scene_labels_path}")
        with scene_labels_path.open("r", encoding="utf-8") as f:
            all_scene_labels = json.load(f)
        scene_truth = all_scene_labels[cfg.image_index]
        raw_path = scene_truth["image_path"]
        path = Path(raw_path) if isinstance(raw_path, str) else raw_path
        image_path = path if path.is_absolute() else project_root / path
        image_id = scene_truth["scene"]
    elif cfg.image_source in {"train_image", "test_image"}:
        source_dir = train_images_dir if cfg.image_source == "train_image" else test_images_dir
        candidates = sorted(source_dir.glob("*.jpg"))
        if cfg.image_id is not None:
            candidates = [p for p in candidates if p.stem == cfg.image_id]
        if not candidates:
            raise FileNotFoundError(f"No images matching id={cfg.image_id} in {source_dir}")
        image_path = candidates[cfg.image_index]
        image_id = image_path.stem
    else:
        raise ValueError(f"Unknown image_source: {cfg.image_source}")

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    print(f"Image source: {cfg.image_source}")
    print(f"Image id:     {image_id}")
    print(f"Image path:   {image_path}")
    print(f"Image shape:  {w} x {h}")
    if scene_truth is not None:
        print(f"Truth cards:  {len(scene_truth.get('cards', []))}")
        print(f"Truth active player: {scene_truth.get('active_player')}")

    return {
        "config": cfg,
        "project_root": project_root,
        "training_data": training_data,
        "models_dir": models_dir,
        "challenge_dir": challenge_dir,
        "train_images_dir": train_images_dir,
        "test_images_dir": test_images_dir,
        "train_csv_path": challenge_dir / "train.csv",
        "device": device,
        "segmenter": segmenter,
        "segmenter_img_size": int(cfg.img_size_segmenter),
        "card_clf": card_clf,
        "card_classes": card_classes,
        "img_size_classifier": img_size_classifier,
        "fourier_coeffs": fourier_coeffs,
        "center_crop_fraction": center_crop_fraction,
        "classifier_expected_features": classifier_expected_features,
        "classifier_feature_order": feature_order,
        "region_colors": REGION_COLORS,
        "image_id": image_id,
        "image_path": image_path,
        "img_bgr": img_bgr,
        "image_h": h,
        "image_w": w,
        "scene_truth": scene_truth,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_game_state(
    image_bgr: np.ndarray,
    image_id: str,
    threshold: float,
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    h, w = image_bgr.shape[:2]
    probability = segment_scene_probability(
        image_bgr, state["segmenter"], state["device"], target_size=state["segmenter_img_size"],
    )
    boxes, mask = boxes_from_probability(probability, threshold=threshold)

    pred_rows: list[dict[str, Any]] = []
    for box_index, box in enumerate(boxes):
        crop_bgr = crop_with_margin(image_bgr, box)
        if crop_bgr.size == 0:
            continue
        feature = extract_features(
            crop_bgr,
            img_size=state["img_size_classifier"],
            fourier_coeffs=state["fourier_coeffs"],
            center_crop_fraction=state["center_crop_fraction"],
            expected_dim=state["classifier_expected_features"],
        ).reshape(1, -1)
        label, confidence = predict_card_label_and_confidence(
            state["card_clf"], state["card_classes"], feature,
        )
        pred_rows.append({
            "box_index": box_index,
            "box": box,
            "label": label,
            "confidence": confidence,
            "region": assign_region(box, w, h),
        })

    region_cards = {"center": [], "p1": [], "p2": [], "p3": [], "p4": []}
    for row in pred_rows:
        region_cards[str(row["region"])].append(row)

    def center_distance(row: dict[str, Any]) -> float:
        x0, y0, x1, y1 = row["box"]
        return float(np.hypot((x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2))

    center_rows = sorted(region_cards["center"], key=center_distance)
    summary = {
        "image_id": image_id,
        "center_card": str(center_rows[0]["label"]) if center_rows else "EMPTY",
        "active_player": "unknown",
        "player_1_cards": format_cards([str(r["label"]) for r in region_cards["p1"]]),
        "player_2_cards": format_cards([str(r["label"]) for r in region_cards["p2"]]),
        "player_3_cards": format_cards([str(r["label"]) for r in region_cards["p3"]]),
        "player_4_cards": format_cards([str(r["label"]) for r in region_cards["p4"]]),
    }
    return summary, pred_rows, probability, mask, boxes


def run_single_image_diagnostics(state: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    img_bgr = state["img_bgr"]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_id = state["image_id"]
    region_colors = state["region_colors"]
    scene_truth = state["scene_truth"]
    threshold_now = float(threshold if threshold is not None else state["config"].eval_threshold)

    summary, pred_rows, global_prob, mask_bin, boxes = _predict_game_state(
        img_bgr, image_id, threshold_now, state
    )
    if scene_truth is not None:
        summary["active_player"] = scene_truth.get("active_player", "unknown")

    print(f"Detected boxes: {len(boxes)}")
    print(f"Foreground pixels: {int(mask_bin.sum())}")

    fig, axes = plt.subplots(4, 1, figsize=(22, 20))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(global_prob, cmap="viridis")
    axes[1].set_title("Segmenter probability")
    axes[1].axis("off")
    axes[2].imshow(mask_bin, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Binary mask")
    axes[2].axis("off")
    axes[3].imshow(img_rgb)
    for x0, y0, x1, y1 in boxes:
        axes[3].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2))
    axes[3].set_title(f"Boxes ({len(boxes)})")
    axes[3].axis("off")
    plt.tight_layout()
    plt.show()

    print("Predictions:")
    for row in pred_rows:
        conf = "" if row["confidence"] is None else f" conf={row['confidence']:.2f}"
        print(f"  box {row['box_index']:02d} {row['box']} -> {row['label']} [{row['region']}]{conf}")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb)
    for row in pred_rows:
        x0, y0, x1, y1 = row["box"]
        color = region_colors.get(str(row["region"]), "white")
        text = f"{row['region']}: {row['label']}"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.5))
        ax.text(
            x0, max(0, y0 - 8), text, color="white", fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )
    ax.set_title(f"{image_id}: detected and classified cards")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    if scene_truth is None:
        print("No labels.json truth for this image source. Use image_source='augmented_scene' for IoU check.")
    else:
        true_cards = [
            {"box": tuple(map(int, c["bbox"])), "label": str(c["label"]).strip(), "owner": str(c["owner"])}
            for c in scene_truth.get("cards", [])
        ]
        matched = []
        for pred in pred_rows:
            if true_cards:
                best = max(true_cards, key=lambda t: box_iou(pred["box"], t["box"]))
                iou = box_iou(pred["box"], best["box"])
            else:
                best = {"label": "none", "owner": "none", "box": (0, 0, 0, 0)}
                iou = 0.0
            matched.append({
                "pred_label": pred["label"], "pred_region": pred["region"],
                "true_label": best["label"], "true_owner": best["owner"], "iou": iou,
            })

        good = [m for m in matched if m["iou"] >= 0.30]
        label_acc = float(np.mean([m["pred_label"] == m["true_label"] for m in good])) if good else 0.0
        owner_acc = float(np.mean([m["pred_region"] == m["true_owner"] for m in good])) if good else 0.0
        print(f"\nMatched IoU >= 0.30: {len(good)} / {len(pred_rows)}")
        print(f"Label accuracy on matched boxes:        {label_acc * 100:.1f}%")
        print(f"Owner/region accuracy on matched boxes: {owner_acc * 100:.1f}%")
        print("\nFirst matched rows:")
        for m in matched[:20]:
            print(
                f"  pred={m['pred_region']:<6} {m['pred_label']:<12} | "
                f"truth={m['true_owner']:<6} {m['true_label']:<12} | IoU={m['iou']:.2f}"
            )

    for k, v in summary.items():
        print(f"{k}: {v}")

    state["global_prob"] = global_prob
    state["boxes"] = boxes
    state["mask_bin"] = mask_bin
    state["pred_rows"] = pred_rows
    state["summary"] = summary
    return state


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _parse_cards_field(value: str) -> list[str]:
    text = str(value).strip()
    if not text or text.upper() == "EMPTY":
        return []
    return [t.strip() for t in text.split(";") if t.strip() and t.strip().upper() != "EMPTY"]


def _bag_f1(pred_cards: list[str], true_cards: list[str]) -> float:
    pred_counter = Counter(pred_cards)
    true_counter = Counter(true_cards)
    tp = sum((pred_counter & true_counter).values())
    fp = sum((pred_counter - true_counter).values())
    fn = sum((true_counter - pred_counter).values())
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else (2 * tp) / denom


def _segmenter_card_count_metrics(
    pred_rows: list[dict[str, Any]],
    true_summary: dict[str, str],
) -> tuple[int, int, int, int, int, float]:
    true_counts = {
        "center": 0 if str(true_summary["center_card"]).strip().upper() == "EMPTY" else 1,
        "p1": len(_parse_cards_field(true_summary["player_1_cards"])),
        "p2": len(_parse_cards_field(true_summary["player_2_cards"])),
        "p3": len(_parse_cards_field(true_summary["player_3_cards"])),
        "p4": len(_parse_cards_field(true_summary["player_4_cards"])),
    }
    pred_counts = {"center": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0}
    for row in pred_rows:
        region = str(row.get("region", "")).strip()
        if region in pred_counts:
            pred_counts[region] += 1

    true_total = int(sum(true_counts.values()))
    pred_total = int(sum(pred_counts.values()))
    count_diff = pred_total - true_total
    abs_count_diff = abs(count_diff)
    # Region-wise absolute variation avoids cancellation between over/under counts across players.
    region_abs_diff = int(sum(abs(pred_counts[k] - true_counts[k]) for k in true_counts))
    count_quality = 1.0 / (1.0 + float(abs_count_diff))
    return true_total, pred_total, count_diff, abs_count_diff, region_abs_diff, count_quality


def run_labeled_benchmark(state: dict[str, Any], benchmark_config: dict[str, Any] | None = None) -> dict[str, Any]:
    bcfg = benchmark_config or {}
    eval_max_images = bcfg.get("eval_max_images", None)
    eval_random_subset = bool(bcfg.get("eval_random_subset", True))
    eval_seed = int(bcfg.get("eval_seed", 42))
    eval_threshold = float(bcfg.get("eval_threshold", state["config"].eval_threshold))
    worst_k = int(bcfg.get("worst_k", 25))
    top_k = int(bcfg.get("top_k", 25))

    train_csv_path: Path = state["train_csv_path"]
    train_images_dir: Path = state["train_images_dir"]

    if not train_csv_path.is_file():
        raise FileNotFoundError(f"Missing labeled CSV for evaluation: {train_csv_path}")

    with train_csv_path.open("r", newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    gt_rows = [r for r in gt_rows if (train_images_dir / f"{str(r['image_id']).strip()}.jpg").is_file()]
    if not gt_rows:
        raise RuntimeError("No labeled train images found for benchmark.")

    if eval_max_images is not None and len(gt_rows) > eval_max_images:
        if eval_random_subset:
            rng = np.random.default_rng(eval_seed)
            chosen = rng.choice(len(gt_rows), size=eval_max_images, replace=False)
            gt_rows = [gt_rows[int(i)] for i in sorted(chosen)]
        else:
            gt_rows = gt_rows[:eval_max_images]

    print(f"Evaluating on {len(gt_rows)} labeled original images from {train_csv_path}...")
    results: list[dict[str, Any]] = []

    for row_index, row in enumerate(gt_rows, start=1):
        image_id_local = str(row["image_id"]).strip()
        image_path_local = train_images_dir / f"{image_id_local}.jpg"
        image_bgr = cv2.imread(str(image_path_local), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        pred_summary, pred_rows, *_ = _predict_game_state(image_bgr, image_id_local, eval_threshold, state)
        true_summary = {
            "image_id": image_id_local,
            "center_card": str(row["center_card"]).strip(),
            "active_player": str(row["active_player"]).strip(),
            "player_1_cards": str(row["player_1_cards"]).strip(),
            "player_2_cards": str(row["player_2_cards"]).strip(),
            "player_3_cards": str(row["player_3_cards"]).strip(),
            "player_4_cards": str(row["player_4_cards"]).strip(),
        }

        f1s = [
            _bag_f1(_parse_cards_field(pred_summary[f"player_{i}_cards"]),
                    _parse_cards_field(true_summary[f"player_{i}_cards"]))
            for i in (1, 2, 3, 4)
        ]
        center_acc = float(pred_summary["center_card"] == true_summary["center_card"])
        image_score = float(np.mean([center_acc] + f1s))
        true_total_cards, pred_total_cards, count_diff, abs_count_diff, region_abs_diff, count_quality = (
            _segmenter_card_count_metrics(pred_rows, true_summary)
        )
        image_score_strict = float(np.mean([center_acc] + f1s + [count_quality]))

        results.append({
            "image_id": image_id_local,
            "image_path": image_path_local,
            "pred_summary": pred_summary,
            "true_summary": true_summary,
            "pred_rows": pred_rows,
            "center_acc": center_acc,
            "p1_f1": f1s[0], "p2_f1": f1s[1], "p3_f1": f1s[2], "p4_f1": f1s[3],
            "image_score": image_score,
            "image_score_strict": image_score_strict,
            "true_card_count": true_total_cards,
            "pred_card_count": pred_total_cards,
            "card_count_diff": count_diff,
            "abs_card_count_diff": abs_count_diff,
            "region_abs_card_count_diff": region_abs_diff,
            "segmenter_count_quality": count_quality,
        })

        if row_index % 25 == 0 or row_index == len(gt_rows):
            print(f"  processed {row_index}/{len(gt_rows)}")

    if not results:
        raise RuntimeError("Benchmark could not process any image.")

    center_acc = float(np.mean([r["center_acc"] for r in results]))
    p_means = [float(np.mean([r[f"p{i}_f1"] for r in results])) for i in (1, 2, 3, 4)]
    macro_f1 = float(np.mean(p_means))
    overall = float(np.mean([r["image_score"] for r in results]))
    overall_strict = float(np.mean([r["image_score_strict"] for r in results]))
    total_true_cards = int(sum(r["true_card_count"] for r in results))
    total_pred_cards = int(sum(r["pred_card_count"] for r in results))
    avg_signed_card_count_diff = float(np.mean([r["card_count_diff"] for r in results]))
    avg_abs_card_count_diff = float(np.mean([r["abs_card_count_diff"] for r in results]))
    avg_region_abs_card_count_diff = float(np.mean([r["region_abs_card_count_diff"] for r in results]))
    segmenter_count_quality = float(np.mean([r["segmenter_count_quality"] for r in results]))

    print("\nBenchmark summary (original labeled data):")
    print(f"  Center-card accuracy: {center_acc * 100:.2f}%")
    print(f"  Player card F1: p1={p_means[0]:.3f}, p2={p_means[1]:.3f}, p3={p_means[2]:.3f}, p4={p_means[3]:.3f}")
    print(f"  Macro player-card F1: {macro_f1:.3f}")
    print(f"  Overall image score:  {overall:.3f}")
    print(f"  Overall strict score: {overall_strict:.3f} (includes segmenter count-variation quality)")
    print(f"  Segmenter avg |pred-true| cards/image: {avg_abs_card_count_diff:.3f}")
    print(f"  Segmenter avg (pred-true) cards/image: {avg_signed_card_count_diff:.3f}")
    print(f"  Segmenter avg region variation/image:  {avg_region_abs_card_count_diff:.3f}")
    print(f"  Segmenter total cards: pred={total_pred_cards}, true={total_true_cards}")
    print(f"  Segmenter count-quality score: {segmenter_count_quality:.3f}")
    print("  Note: active_player is not scored yet because this notebook does not predict turns.")

    metric_names = [
        "center_acc", "p1_f1", "p2_f1", "p3_f1", "p4_f1", "macro_f1",
        "seg_count_quality", "overall", "overall_strict",
    ]
    metric_values = [center_acc] + p_means + [macro_f1, segmenter_count_quality, overall, overall_strict]
    image_scores = [r["image_score"] for r in results]
    bin_count = min(24, max(8, int(np.sqrt(len(image_scores)))))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(metric_names, metric_values,
                color=["#4e79a7", "#59a14f", "#59a14f", "#59a14f", "#59a14f", "#f28e2b", "#e15759"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Benchmark metrics on original labeled data")
    axes[0].set_ylabel("score")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(image_scores, bins=bin_count, color="#76b7b2", edgecolor="black")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Distribution of per-image scores")
    axes[1].set_xlabel("image score")
    axes[1].set_ylabel("count")
    plt.tight_layout()
    plt.show()

    worst_results = sorted(results, key=lambda x: x["image_score"])[: min(worst_k, len(results))]
    top_results = sorted(results, key=lambda x: x["image_score"], reverse=True)[: min(top_k, len(results))]

    print("\nWorst-performing labeled images:")
    for item in worst_results:
        print(f"  {item['image_id']}: score={item['image_score']:.3f}, "
              f"center(pred/true)={item['pred_summary']['center_card']}/{item['true_summary']['center_card']}")

    print("\nTop-performing labeled images:")
    for item in top_results:
        print(f"  {item['image_id']}: score={item['image_score']:.3f}, "
              f"center(pred/true)={item['pred_summary']['center_card']}/{item['true_summary']['center_card']}")

    _plot_qualitative(worst_results, f"Worst {len(worst_results)} performers: annotated image + predicted mask",
                      state, eval_threshold)
    _plot_qualitative(top_results, f"Top {len(top_results)} performers: annotated image + predicted mask",
                      state, eval_threshold)

    benchmark = {
        "results": results,
        "center_acc": center_acc,
        "p1_f1": p_means[0], "p2_f1": p_means[1], "p3_f1": p_means[2], "p4_f1": p_means[3],
        "macro_f1": macro_f1,
        "overall": overall,
        "overall_strict": overall_strict,
        "cards_total_true": total_true_cards,
        "cards_total_pred": total_pred_cards,
        "avg_signed_card_count_diff": avg_signed_card_count_diff,
        "avg_abs_card_count_diff": avg_abs_card_count_diff,
        "avg_region_abs_card_count_diff": avg_region_abs_card_count_diff,
        "segmenter_count_quality": segmenter_count_quality,
        "worst_results": worst_results,
        "top_results": top_results,
    }
    state["benchmark"] = benchmark
    return benchmark


def _plot_qualitative(examples: list[dict[str, Any]], title: str, state: dict[str, Any], threshold: float) -> None:
    if not examples:
        print(f"No examples to plot for: {title}")
        return

    region_colors = state["region_colors"]
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 4.8 * len(examples)))
    axes = np.array(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_idx, item in enumerate(examples):
        image_bgr = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        probability = segment_scene_probability(
            image_bgr, state["segmenter"], state["device"], target_size=state["segmenter_img_size"],
        )
        _, mask = boxes_from_probability(probability, threshold=threshold)

        ax_img = axes[row_idx, 0]
        ax_img.imshow(image_rgb)
        for pred in item["pred_rows"]:
            x0, y0, x1, y1 = pred["box"]
            color = region_colors.get(str(pred["region"]), "white")
            text = f"{pred['region']}: {pred['label']}"
            confidence = pred.get("confidence")
            if confidence is not None:
                text += f" ({float(confidence):.2f})"
            ax_img.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.5))
            ax_img.text(x0, max(0, y0 - 8), text, color="white", fontsize=8,
                        bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"})

        ax_img.set_title(
            f"{item['image_id']} score={item['image_score']:.3f}\n"
            f"center pred={item['pred_summary']['center_card']} | true={item['true_summary']['center_card']}",
            fontsize=9,
        )
        ax_img.axis("off")

        ax_mask = axes[row_idx, 1]
        ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax_mask.set_title("Predicted segmentation mask", fontsize=9)
        ax_mask.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
