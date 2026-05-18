"""Plotting helpers used by the report notebook.

These functions all take the inference outputs already produced by
`src/inference.py` so the notebook stays short: build a `GameState`, hand it
here, get a figure. Keeping plotting code out of the notebook also keeps the
notebook executable end-to-end on the TA's machine without 200 lines of
matplotlib in every cell.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Polygon, Rectangle
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.inference import predict_from_path

from src.inference import (
    assign_region,
    CardPrediction,
    detect_active_player,
    divide_background,
    GameState,
    InferenceEngine,
    boxes_from_probability,
    find_black_token,
    find_yellow_token,
    is_background_noisy,
    predict_cards,
    predict_game_state,
    segment_scene_probability,
)


REGION_COLORS = {
    "center": "gold",
    "p1": "tab:blue",
    "p2": "tab:green",
    "p3": "tab:red",
    "p4": "tab:purple",
}


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


def plot_pipeline_stages(
    engine: InferenceEngine,
    img_bgr: np.ndarray,
    predictions: list[CardPrediction] | None = None,
    title: str | None = None,
) -> None:
    """Four-panel view: input | segmenter prob | grown instances | classified.

    Designed for the "How the pipeline works" section of the report — it shows
    a single image flowing through every stage so the reader can visually map
    each component to a concrete output.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    probability = segment_scene_probability(
        img_bgr, engine.segmenter, engine.device, target_size=engine.config.segmenter_img_size,
    )
    _, _, instance_masks = boxes_from_probability(
        probability,
        threshold=engine.config.segmenter_threshold,
        min_component_area=engine.config.segmenter_min_component_area,
        instance_mask_growth_px=engine.config.instance_mask_growth_px,
        return_instance_masks=True,
    )
    if predictions is None:
        predictions = predict_cards(engine, img_bgr)

    # Color-coded instance map: each card gets a distinct hue so dilation
    # behaviour and split-touching-component decisions are visible at a glance.
    instance_map = np.zeros((h, w, 3), dtype=np.float32)
    for idx, mask in enumerate(instance_masks):
        rng = np.random.default_rng(idx + 1)
        color = rng.uniform(0.35, 1.0, size=3)
        instance_map[mask > 0] = color

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("(1) Input image")

    axes[1].imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"(2) Segmenter probability\nthreshold={engine.config.segmenter_threshold:.2f}")

    axes[2].imshow(instance_map)
    axes[2].set_title(f"(3) Per-card instance masks\ngrowth=+{engine.config.instance_mask_growth_px}px")

    axes[3].imshow(img_rgb)
    for pred in predictions:
        color = REGION_COLORS.get(pred.region, "white")
        x0, y0, x1, y1 = pred.box
        axes[3].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.2))
        axes[3].text(
            x0, max(0, y0 - 8),
            f"{pred.region}: {pred.label} ({pred.confidence:.2f})",
            color="white", fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )
    axes[3].set_title("(4) Classified cards + region assignment")

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_region_layout(image_w: int = 3000, image_h: int = 2000, map_resolution: int = 480) -> None:
    """Schematic of the fixed player geometry (R6).

    Useful in the report to motivate the `assign_region` heuristic before
    discussing failure cases.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    ax.set_aspect("equal")

    region_order = ["center", "p1", "p2", "p3", "p4"]
    region_to_index = {region: index for index, region in enumerate(region_order)}
    map_w = max(2, int(map_resolution))
    map_h = max(2, int(round(map_w * image_h / image_w)))
    region_map = np.empty((map_h, map_w), dtype=np.uint8)
    for sample_y in range(map_h):
        for sample_x in range(map_w):
            region = assign_region((sample_x, sample_y, sample_x + 1, sample_y + 1), map_w, map_h)
            region_map[sample_y, sample_x] = region_to_index[region]

    region_rgba = np.zeros((map_h, map_w, 4), dtype=np.float32)
    for region, index in region_to_index.items():
        region_rgba[region_map == index] = to_rgba(REGION_COLORS[region], alpha=0.30)
    ax.imshow(region_rgba, extent=(0, image_w, image_h, 0), interpolation="nearest")

    # Central rectangle = "center" region. Outside it, assign_region chooses the
    # nearest player edge with a small penalty for lateral displacement.
    cx_lo, cx_hi = 0.36, 0.64
    cy_lo, cy_hi = 0.30, 0.70
    ax.add_patch(Rectangle(
        (cx_lo * image_w, cy_lo * image_h),
        (cx_hi - cx_lo) * image_w,
        (cy_hi - cy_lo) * image_h,
        fill=False, edgecolor="black", linewidth=1.6,
    ))

    annotations = [
        (0.50, 0.95, "p1 (bottom)", "tab:blue"),
        (0.95, 0.50, "p2 (right)", "tab:green"),
        (0.50, 0.05, "p3 (top)", "tab:red"),
        (0.05, 0.50, "p4 (left)", "tab:purple"),
        (0.50, 0.50, "center", "darkgoldenrod"),
    ]
    for px, py, label, color in annotations:
        ax.text(px * image_w, py * image_h, label, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                bbox={"facecolor": color, "alpha": 0.85, "pad": 3, "edgecolor": "none"})

    ax.set_title("Card-center attribution regions (R6)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _active_player_token_centers(img_bgr: np.ndarray) -> tuple[str, list[tuple[int, int]]]:
    if is_background_noisy(img_bgr):
        return "yellow disk", find_yellow_token(img_bgr)
    return "dark rectangle", find_black_token(img_bgr)


def plot_active_player_detection(
    img_bgr: np.ndarray,
    expected_player: str | None = None,
    title: str | None = None,
) -> None:
    """Overlay the detected active-player token and fixed token sectors."""
    image_height, image_width = img_bgr.shape[:2]
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    token_mode, centers = _active_player_token_centers(img_bgr)
    predicted_player = detect_active_player(img_bgr)
    polygons = divide_background(image_width, image_height)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image_rgb)
    for player, polygon in polygons.items():
        ax.add_patch(Polygon(
            polygon,
            closed=True,
            fill=True,
            facecolor=REGION_COLORS[player],
            alpha=0.15,
            edgecolor=REGION_COLORS[player],
            linewidth=2.0,
        ))
        center_x = float(np.mean([point[0] for point in polygon]))
        center_y = float(np.mean([point[1] for point in polygon]))
        ax.text(
            center_x,
            center_y,
            player,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": REGION_COLORS[player], "alpha": 0.85, "pad": 3, "edgecolor": "none"},
        )

    if centers:
        xs, ys = zip(*centers)
        ax.scatter(xs, ys, s=160, c="none", edgecolors="black", linewidths=3, label="candidate token")
        ax.scatter(xs, ys, s=70, c="yellow" if token_mode == "yellow disk" else "white", edgecolors="black", linewidths=1)
    expected_text = "" if expected_player is None else f" | true={expected_player}"
    ax.set_title(title or f"Active-player token: {token_mode} | pred={predicted_player}{expected_text}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def summarize_active_player_detection(train_csv_path: Path, train_images_dir: Path) -> dict[str, Any]:
    """Evaluate the deterministic token detector against labelled train rows."""
    train_csv_path = Path(train_csv_path)
    train_images_dir = Path(train_images_dir)
    with train_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    results: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row["image_id"]).strip()
        image_path = train_images_dir / f"{image_id}.jpg"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        token_mode, centers = _active_player_token_centers(image_bgr)
        predicted_player = detect_active_player(image_bgr)
        true_player = str(row["active_player"]).strip()
        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "token_mode": token_mode,
            "predicted_player": predicted_player,
            "true_player": true_player,
            "is_correct": predicted_player == true_player,
            "n_candidates": len(centers),
        })

    if not results:
        raise RuntimeError("No labelled images available for active-player evaluation.")

    modes = sorted({result["token_mode"] for result in results})
    mode_accuracy = {
        mode: float(np.mean([result["is_correct"] for result in results if result["token_mode"] == mode]))
        for mode in modes
    }
    return {
        "results": results,
        "accuracy": float(np.mean([result["is_correct"] for result in results])),
        "mode_accuracy": mode_accuracy,
        "mode_counts": Counter(result["token_mode"] for result in results),
        "confusion": Counter((result["true_player"], result["predicted_player"]) for result in results),
        "unknown_count": sum(result["predicted_player"] == "unknown" for result in results),
    }


def print_active_player_summary(summary: dict[str, Any]) -> None:
    mode_text = ", ".join(
        f"{mode}={summary['mode_accuracy'][mode]:.3f} ({summary['mode_counts'][mode]})"
        for mode in sorted(summary["mode_accuracy"])
    )
    print(
        f"Active-player detector: accuracy={summary['accuracy']:.3f}, "
        f"unknown={summary['unknown_count']}, by token/background: {mode_text}."
    )


def plot_active_player_summary(summary: dict[str, Any]) -> None:
    """Two-panel graph: accuracy by token mode and active-player confusion."""
    players = ["p1", "p2", "p3", "p4"]
    predicted_labels = players + ["unknown"]
    confusion = np.zeros((len(players), len(predicted_labels)), dtype=int)
    for row_index, true_player in enumerate(players):
        for col_index, predicted_player in enumerate(predicted_labels):
            confusion[row_index, col_index] = int(summary["confusion"].get((true_player, predicted_player), 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    modes = sorted(summary["mode_accuracy"])
    axes[0].bar(modes, [summary["mode_accuracy"][mode] for mode in modes], color=["#4e79a7", "#f28e2b"][: len(modes)])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Active-player accuracy by token/background")
    _annotate_bars(axes[0])

    image = axes[1].imshow(confusion, cmap="Blues")
    axes[1].set_xticks(range(len(predicted_labels)))
    axes[1].set_xticklabels(predicted_labels)
    axes[1].set_yticks(range(len(players)))
    axes[1].set_yticklabels(players)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Active-player confusion")
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = confusion[row_index, col_index]
            if value:
                axes[1].text(col_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_confidence_histogram(predictions: Iterable[CardPrediction], bins: int = 20) -> None:
    """Histogram of per-card softmax confidences.

    A heavily right-skewed distribution is a good sign (the classifier commits
    to a class). A bump near 1/N_classes flags boxes that the classifier
    cannot resolve and likely need a better mask or a higher seg threshold.
    """
    confidences = [p.confidence for p in predictions]
    if not confidences:
        print("No predictions to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidences, bins=bins, color="#4e79a7", edgecolor="black")
    ax.axvline(float(np.mean(confidences)), color="red", linestyle="--", label=f"mean={np.mean(confidences):.2f}")
    ax.set_xlabel("Softmax confidence")
    ax.set_ylabel("Number of predicted cards")
    ax.set_title("Classifier confidence distribution")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_class_frequency(predictions: Iterable[CardPrediction]) -> None:
    """Bar chart of predicted-class counts.

    Useful for spotting classifier bias: heavily over-represented classes on a
    balanced test set are a red flag for shortcut features (e.g. classifying
    by background color when the mask is too loose).
    """
    counter = Counter(p.label for p in predictions)
    if not counter:
        print("No predictions to plot.")
        return

    labels, counts = zip(*sorted(counter.items()))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.25), 4))
    ax.bar(labels, counts, color="#59a14f", edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Predicted card class frequency (test set)")
    plt.tight_layout()
    plt.show()


def plot_card_count_per_region(game_states: Iterable[GameState]) -> None:
    """Boxplot of #cards predicted per region across the dataset.

    The center should always be 0 or 1; large variance there signals the
    center-vs-player geometry rectangle needs tightening. Player counts give a
    quick sanity check on the segmenter recall.
    """
    region_counts = {"center": [], "p1": [], "p2": [], "p3": [], "p4": []}
    for state in game_states:
        per_region = {k: 0 for k in region_counts}
        for pred in state.cards:
            per_region[pred.region] = per_region.get(pred.region, 0) + 1
        for k, v in per_region.items():
            region_counts[k].append(v)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [region_counts[k] for k in ("center", "p1", "p2", "p3", "p4")],
        labels=["center", "p1", "p2", "p3", "p4"],
    )
    ax.set_ylabel("Cards detected")
    ax.set_title("Per-region card count distribution")
    plt.tight_layout()
    plt.show()


def summarize_prediction_set(game_states: Iterable[GameState]) -> dict[str, Any]:
    """Aggregate inference-only predictions for a compact test-set audit."""
    states = list(game_states)
    predictions = [prediction for state in states for prediction in state.cards]
    region_order = ["center", "p1", "p2", "p3", "p4"]
    region_counts = {region: [] for region in region_order}
    cards_per_image: list[int] = []

    for state in states:
        per_region = {region: 0 for region in region_order}
        for prediction in state.cards:
            if prediction.region in per_region:
                per_region[prediction.region] += 1
        for region in region_order:
            region_counts[region].append(per_region[region])
        cards_per_image.append(len(state.cards))

    return {
        "states": states,
        "predictions": predictions,
        "n_images": len(states),
        "n_cards": len(predictions),
        "confidences": [prediction.confidence for prediction in predictions],
        "class_counts": Counter(prediction.label for prediction in predictions),
        "region_counts": region_counts,
        "cards_per_image": cards_per_image,
    }


def print_prediction_set_summary(summary: dict[str, Any]) -> None:
    confidences = summary["confidences"]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    cards_per_image = summary["cards_per_image"]
    mean_cards = float(np.mean(cards_per_image)) if cards_per_image else 0.0
    print(
        f"Prediction audit: {summary['n_images']} images, {summary['n_cards']} cards, "
        f"mean cards/image={mean_cards:.2f}, mean confidence={mean_conf:.3f}."
    )


def plot_prediction_set_diagnostics(summary: dict[str, Any], top_n: int = 14) -> None:
    """One dashboard for inference-only prediction sanity checks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    confidences = summary["confidences"]
    if confidences:
        axes[0, 0].hist(confidences, bins=20, color="#4e79a7", edgecolor="white")
        axes[0, 0].axvline(float(np.mean(confidences)), color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_title("Classifier confidence")
    axes[0, 0].set_xlabel("Softmax confidence")
    axes[0, 0].set_ylabel("Predicted cards")

    cards_per_image = summary["cards_per_image"]
    if cards_per_image:
        axes[0, 1].hist(cards_per_image, bins=range(0, max(cards_per_image) + 2), color="#59a14f", edgecolor="white")
        axes[0, 1].axvline(float(np.mean(cards_per_image)), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Detected cards per image")
    axes[0, 1].set_xlabel("Cards")
    axes[0, 1].set_ylabel("Images")

    region_order = ["center", "p1", "p2", "p3", "p4"]
    region_counts = summary["region_counts"]
    axes[1, 0].boxplot([region_counts[region] for region in region_order], labels=region_order)
    axes[1, 0].set_title("Per-region card counts")
    axes[1, 0].set_ylabel("Cards")

    top_classes = summary["class_counts"].most_common(top_n)
    if top_classes:
        labels, counts = zip(*top_classes)
        y_positions = np.arange(len(labels))
        axes[1, 1].barh(y_positions, counts, color="#f28e2b")
        axes[1, 1].set_yticks(y_positions)
        axes[1, 1].set_yticklabels(labels, fontsize=8)
        axes[1, 1].invert_yaxis()
    axes[1, 1].set_title(f"Top {min(top_n, len(top_classes))} predicted classes")
    axes[1, 1].set_xlabel("Count")

    fig.suptitle("Inference-only test prediction audit", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_qualitative_grid(
    examples: list[tuple[Path, GameState]],
    cols: int = 2,
    max_rows: int = 4,
) -> None:
    """Compact qualitative grid of (image, predictions) for the report.

    Designed for the failure-mode and success-case sections. Each cell shows
    the raw image with predicted boxes/labels overlaid.
    """
    examples = examples[: cols * max_rows]
    if not examples:
        return
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (image_path, state) in enumerate(examples):
        ax = axes[idx // cols, idx % cols]
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            ax.set_visible(False)
            continue
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        for pred in state.cards:
            color = REGION_COLORS.get(pred.region, "white")
            x0, y0, x1, y1 = pred.box
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
            ax.text(x0, max(0, y0 - 8), f"{pred.label}", color="white", fontsize=7,
                    bbox={"facecolor": color, "alpha": 0.85, "pad": 1, "edgecolor": "none"})
        ax.set_title(state.image_id, fontsize=10)
        ax.axis("off")

    for j in range(len(examples), rows * cols):
        axes[j // cols, j % cols].set_visible(False)
    plt.tight_layout()
    plt.show()


@dataclass
class BenchmarkConfig:
    eval_max_images: int | None = None
    eval_random_subset: bool = True
    eval_seed: int = 42
    eval_threshold: float | None = None
    worst_k: int = 25
    top_k: int = 25
    show_plots: bool = True
    verbose: bool = True
    progress_every: int | None = None


def parse_cards_field(value: str) -> list[str]:
    text = str(value).strip()
    if not text or text.upper() == "EMPTY":
        return []
    return [token.strip() for token in text.split(";") if token.strip() and token.strip().upper() != "EMPTY"]


def bag_f1(pred_cards: list[str], true_cards: list[str]) -> float:
    pred_counter = Counter(pred_cards)
    true_counter = Counter(true_cards)
    true_positive = sum((pred_counter & true_counter).values())
    false_positive = sum((pred_counter - true_counter).values())
    false_negative = sum((true_counter - pred_counter).values())
    denom = 2 * true_positive + false_positive + false_negative
    return 1.0 if denom == 0 else (2 * true_positive) / denom


def run_labeled_benchmark(
    engine: InferenceEngine,
    train_csv_path: Path,
    train_images_dir: Path,
    benchmark_config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the full pipeline against official labelled train.csv rows."""
    config = benchmark_config or BenchmarkConfig()
    eval_threshold = engine.config.segmenter_threshold if config.eval_threshold is None else float(config.eval_threshold)
    eval_engine = replace(engine, config=replace(engine.config, segmenter_threshold=eval_threshold))

    train_csv_path = Path(train_csv_path)
    train_images_dir = Path(train_images_dir)
    if not train_csv_path.is_file():
        raise FileNotFoundError(f"Missing labelled CSV for evaluation: {train_csv_path}")

    with train_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        truth_rows = [
            row for row in csv.DictReader(csv_file)
            if (train_images_dir / f"{str(row['image_id']).strip()}.jpg").is_file()
        ]
    if not truth_rows:
        raise RuntimeError("No labelled train images found for benchmark.")

    if config.eval_max_images is not None and len(truth_rows) > config.eval_max_images:
        if config.eval_random_subset:
            rng = np.random.default_rng(config.eval_seed)
            chosen = rng.choice(len(truth_rows), size=config.eval_max_images, replace=False)
            truth_rows = [truth_rows[int(index)] for index in sorted(chosen)]
        else:
            truth_rows = truth_rows[: config.eval_max_images]

    if config.verbose:
        print(f"Benchmark: {len(truth_rows)} labelled images, segmenter threshold={eval_threshold:.2f}.")
    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(truth_rows, start=1):
        image_id = str(row["image_id"]).strip()
        image_path = train_images_dir / f"{image_id}.jpg"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        pred_state = predict_game_state(eval_engine, image_bgr, image_id)
        pred_summary = pred_state.as_submission_row()
        true_summary = {
            "image_id": image_id,
            "center_card": str(row["center_card"]).strip(),
            "active_player": str(row["active_player"]).strip(),
            "player_1_cards": str(row["player_1_cards"]).strip(),
            "player_2_cards": str(row["player_2_cards"]).strip(),
            "player_3_cards": str(row["player_3_cards"]).strip(),
            "player_4_cards": str(row["player_4_cards"]).strip(),
        }

        player_f1s = [
            bag_f1(
                parse_cards_field(pred_summary[f"player_{player_index}_cards"]),
                parse_cards_field(true_summary[f"player_{player_index}_cards"]),
            )
            for player_index in (1, 2, 3, 4)
        ]
        center_acc = float(pred_summary["center_card"] == true_summary["center_card"])
        active_acc = float(pred_summary["active_player"] == true_summary["active_player"])
        image_score = float(np.mean([center_acc, active_acc] + player_f1s))

        true_counts = {
            "center": 0 if true_summary["center_card"].upper() == "EMPTY" else 1,
            "p1": len(parse_cards_field(true_summary["player_1_cards"])),
            "p2": len(parse_cards_field(true_summary["player_2_cards"])),
            "p3": len(parse_cards_field(true_summary["player_3_cards"])),
            "p4": len(parse_cards_field(true_summary["player_4_cards"])),
        }
        pred_counts = {"center": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0}
        for prediction in pred_state.cards:
            if prediction.region in pred_counts:
                pred_counts[prediction.region] += 1

        true_card_count = int(sum(true_counts.values()))
        pred_card_count = int(sum(pred_counts.values()))
        card_count_diff = pred_card_count - true_card_count
        abs_card_count_diff = abs(card_count_diff)
        region_abs_diff = int(sum(abs(pred_counts[region] - true_counts[region]) for region in true_counts))
        count_quality = 1.0 / (1.0 + float(abs_card_count_diff))

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "pred_state": pred_state,
            "pred_summary": pred_summary,
            "true_summary": true_summary,
            "true_counts": true_counts,
            "pred_counts": pred_counts,
            "center_acc": center_acc,
            "active_acc": active_acc,
            "p1_f1": player_f1s[0],
            "p2_f1": player_f1s[1],
            "p3_f1": player_f1s[2],
            "p4_f1": player_f1s[3],
            "image_score": image_score,
            "image_score_strict": float(np.mean([center_acc, active_acc] + player_f1s + [count_quality])),
            "true_card_count": true_card_count,
            "pred_card_count": pred_card_count,
            "card_count_diff": card_count_diff,
            "abs_card_count_diff": abs_card_count_diff,
            "region_abs_card_count_diff": region_abs_diff,
            "segmenter_count_quality": count_quality,
        })

        if config.progress_every and (row_index % config.progress_every == 0 or row_index == len(truth_rows)):
            print(f"  processed {row_index}/{len(truth_rows)}")

    if not results:
        raise RuntimeError("Benchmark could not process any image.")

    center_acc = float(np.mean([result["center_acc"] for result in results]))
    active_acc = float(np.mean([result["active_acc"] for result in results]))
    player_means = [float(np.mean([result[f"p{player_index}_f1"] for result in results])) for player_index in (1, 2, 3, 4)]
    macro_f1 = float(np.mean(player_means))
    overall = float(np.mean([result["image_score"] for result in results]))
    overall_strict = float(np.mean([result["image_score_strict"] for result in results]))
    segmenter_count_quality = float(np.mean([result["segmenter_count_quality"] for result in results]))

    worst_results = sorted(results, key=lambda result: result["image_score"])[: min(config.worst_k, len(results))]
    top_results = sorted(results, key=lambda result: result["image_score"], reverse=True)[: min(config.top_k, len(results))]
    if config.verbose:
        weakest = ", ".join(f"{item['image_id']}={item['image_score']:.2f}" for item in worst_results) or "none"
        strongest = ", ".join(f"{item['image_id']}={item['image_score']:.2f}" for item in top_results) or "none"
        print(
            "Benchmark summary: "
            f"center={center_acc:.3f}, active={active_acc:.3f}, player_macro={macro_f1:.3f}, "
            f"overall={overall:.3f}, strict={overall_strict:.3f}, "
            f"count_MAE={np.mean([result['abs_card_count_diff'] for result in results]):.2f}, "
            f"cards pred/true={sum(result['pred_card_count'] for result in results)}/{sum(result['true_card_count'] for result in results)}."
        )
        print(f"Weakest labelled images: {weakest}.")
        print(f"Top labelled images: {strongest}.")

    benchmark = {
        "results": results,
        "center_acc": center_acc,
        "active_acc": active_acc,
        "p1_f1": player_means[0],
        "p2_f1": player_means[1],
        "p3_f1": player_means[2],
        "p4_f1": player_means[3],
        "macro_f1": macro_f1,
        "overall": overall,
        "overall_strict": overall_strict,
        "cards_total_true": int(sum(result["true_card_count"] for result in results)),
        "cards_total_pred": int(sum(result["pred_card_count"] for result in results)),
        "avg_signed_card_count_diff": float(np.mean([result["card_count_diff"] for result in results])),
        "avg_abs_card_count_diff": float(np.mean([result["abs_card_count_diff"] for result in results])),
        "avg_region_abs_card_count_diff": float(np.mean([result["region_abs_card_count_diff"] for result in results])),
        "segmenter_count_quality": segmenter_count_quality,
        "worst_results": worst_results,
        "top_results": top_results,
        "eval_threshold": eval_threshold,
    }

    if config.show_plots:
        plot_benchmark_summary(benchmark)
        plot_benchmark_examples(worst_results, "Lowest-scoring labelled examples", eval_engine)
        plot_benchmark_examples(top_results, "Top-scoring labelled examples", eval_engine)
    return benchmark


def plot_benchmark_summary(benchmark: dict[str, object]) -> None:
    """Four-panel benchmark dashboard for the final report."""
    metric_names = ["center", "active", "p1", "p2", "p3", "p4", "macro", "overall"]
    metric_values = [
        float(benchmark["center_acc"]),
        float(benchmark["active_acc"]),
        float(benchmark["p1_f1"]),
        float(benchmark["p2_f1"]),
        float(benchmark["p3_f1"]),
        float(benchmark["p4_f1"]),
        float(benchmark["macro_f1"]),
        float(benchmark["overall"]),
    ]
    results = list(benchmark["results"])
    image_scores = [float(result["image_score"]) for result in results]
    bin_count = min(24, max(8, int(np.sqrt(len(image_scores)))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(metric_names, metric_values, color=["#4e79a7", "#edc948"] + ["#59a14f"] * 4 + ["#f28e2b", "#76b7b2"])
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_title("Structured-state scores")
    axes[0, 0].tick_params(axis="x", rotation=25)
    _annotate_bars(axes[0, 0])

    axes[0, 1].hist(image_scores, bins=bin_count, color="#76b7b2", edgecolor="white")
    axes[0, 1].axvline(float(benchmark["overall"]), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].set_title("Per-image score distribution")
    axes[0, 1].set_xlabel("Image score")
    axes[0, 1].set_ylabel("Images")

    true_counts = [int(result["true_card_count"]) for result in results]
    pred_counts = [int(result["pred_card_count"]) for result in results]
    scatter = axes[1, 0].scatter(true_counts, pred_counts, c=image_scores, cmap="viridis", vmin=0, vmax=1, alpha=0.85)
    limit = max(true_counts + pred_counts + [1])
    axes[1, 0].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("True cards")
    axes[1, 0].set_ylabel("Predicted cards")
    axes[1, 0].set_title("Detection count vs label")
    colorbar = fig.colorbar(scatter, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("image score")

    region_order = ["center", "p1", "p2", "p3", "p4"]
    region_mae = [
        np.mean([abs(int(result["pred_counts"][region]) - int(result["true_counts"][region])) for result in results])
        for region in region_order
    ]
    axes[1, 1].bar(region_order, region_mae, color="#e15759")
    axes[1, 1].set_title("Mean absolute count error by region")
    axes[1, 1].set_ylabel("Cards")
    _annotate_bars(axes[1, 1])

    fig.suptitle("Labelled benchmark audit", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_benchmark_examples(
    examples: list[dict[str, object]],
    title: str,
    engine: InferenceEngine | None = None,
    cols: int | None = None,
) -> None:
    """Qualitative benchmark rows: annotated prediction beside predicted mask.

    `cols` is kept for older notebook calls; the report layout is fixed at two columns.
    """
    if not examples:
        print(f"No examples to plot for: {title}")
        return

    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(14.5, 5.0 * rows), squeeze=False)

    for index, item in enumerate(examples):
        image_axis = axes[index, 0]
        mask_axis = axes[index, 1]
        image_path = Path(item["image_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            image_axis.axis("off")
            mask_axis.axis("off")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred_state = item["pred_state"]
        predictions = pred_state.cards if isinstance(pred_state, GameState) else []

        image_axis.imshow(image_rgb)
        for pred in predictions:
            color = REGION_COLORS.get(pred.region, "white")
            x0, y0, x1, y1 = pred.box
            image_axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.2))
            image_axis.text(
                x0,
                max(0, y0 - 8),
                f"{pred.region}: {pred.label} ({pred.confidence:.2f})",
                color="white",
                fontsize=7,
                bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
            )

        pred_summary = item["pred_summary"]
        true_summary = item["true_summary"]
        image_axis.set_title(
            f"{item['image_id']} score={float(item['image_score']):.3f}\n"
            f"center {pred_summary['center_card']} -> {true_summary['center_card']} | "
            f"active {pred_summary['active_player']} -> {true_summary['active_player']} | "
            f"cards {item['pred_card_count']} -> {item['true_card_count']}",
            fontsize=9,
        )
        image_axis.axis("off")

        mask_title = "Predicted mask"
        if engine is not None:
            probability = segment_scene_probability(
                image_bgr,
                engine.segmenter,
                engine.device,
                target_size=engine.config.segmenter_img_size,
            )
            _, predicted_mask = boxes_from_probability(
                probability,
                threshold=engine.config.segmenter_threshold,
                min_component_area=engine.config.segmenter_min_component_area,
                instance_mask_growth_px=engine.config.instance_mask_growth_px,
            )
            mask_title = f"Predicted mask (threshold={engine.config.segmenter_threshold:.2f})"
        else:
            predicted_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            for pred in predictions:
                predicted_mask[pred.instance_mask > 0] = 1

        mask_axis.imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
        mask_axis.set_title(mask_title, fontsize=9)
        mask_axis.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

########## Interpretability visualizations ##########

def get_classifier_model(engine):
    """Dynamically extracts the classifier nn.Module from the inference engine."""
    classifier_model = None
    for attr_name in dir(engine):
        attr = getattr(engine, attr_name, None)
        if isinstance(attr, torch.nn.Module) and "classifier" in attr_name.lower():
            classifier_model = attr
            break
            
    if classifier_model is None:
        for attr_name in dir(engine):
            attr = getattr(engine, attr_name, None)
            if isinstance(attr, torch.nn.Module) and "segment" not in attr_name.lower():
                classifier_model = attr
                break
                
    if classifier_model is None:
        raise ValueError("Could not find the classifier network within the 'engine' object.")
    return classifier_model


def run_saliency_pipeline(engine, image_path):
    """
    Runs inference on an image path while attaching a forward hook 
    to capture 4-channel classifier inputs, then computes saliency maps.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Could not find the specified image at: {img_path.resolve()}")
        
    classifier_model = get_classifier_model(engine)
    
    captured_inputs = []
    captured_predictions = []

    def hook_fn(module, field_input, field_output):
        if len(field_input) > 0:
            captured_inputs.append(field_input[0].detach().cpu())
        captured_predictions.append(field_output.detach().cpu())

    # Register hook, run complete inference pipeline, and guarantee removal
    hook_handle = classifier_model.register_forward_hook(hook_fn)
    print(f"Processing image: {img_path.name}")
    try:
        _ = predict_from_path(engine, img_path)
    finally:
        hook_handle.remove()

    print(f"Successfully processed scene! Detected {len(captured_inputs)} card components.")
    
    classifier_model.eval()
    device = next(classifier_model.parameters()).device
    saliency_results = []

    # Compute feature importances via backpropagation gradients
    # Look for class names inside the engine structure
    # Usually stored in engine.class_names or engine.config / engine.classifier
    class_names = getattr(engine, 'class_names', None)
    if class_names is None and hasattr(engine, 'classifier'):
        class_names = getattr(engine.classifier, 'class_names', None)

    # Compute feature importances via backpropagation gradients
    for idx, (inp, out) in enumerate(zip(captured_inputs, captured_predictions)):
        inp_var = inp.clone().requires_grad_(True)
        
        with torch.set_grad_enabled(True):
            logits = classifier_model(inp_var.to(device))
            pred_class = logits.argmax(dim=1).item()
            score = logits[0, pred_class]
            score.backward()
            
        saliency = inp_var.grad.cpu().numpy()[0]   # (4, H, W)
        input_np = inp.numpy()[0]                  # (4, H, W)
        
        rgb_saliency = np.max(np.abs(saliency[:3]), axis=0)
        mask_saliency = np.abs(saliency[3])
        
        rgb_img = input_np[:3].transpose(1, 2, 0)
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        mask_img = input_np[3]
        
        # Determine the string name if class_names array exists, otherwise fallback to the index
        if class_names is not None and pred_class < len(class_names):
            class_label = class_names[pred_class]
        else:
            class_label = f"Class {pred_class}" # Fallback if names array isn't exposed there
        
        saliency_results.append({
            'card_idx': idx,
            'pred_class': class_label,  # <--- Changed this to hold the string label
            'rgb_img': rgb_img,
            'mask_img': mask_img,
            'rgb_saliency': rgb_saliency,
            'mask_saliency': mask_saliency
        })
        return saliency_results


def plot_saliency_dashboard(saliency_results):
    """Plots a 4-column feature-importance visualization dashboard for every card."""
    for res in saliency_results:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(res['rgb_img'])
        axes[0].set_title(f"Card #{res['card_idx']} - Isolated Crop\nPredicted Class: {res['pred_class']}")
        axes[0].axis('off')
        
        axes[1].imshow(res['mask_img'], cmap='gray')
        axes[1].set_title("Fed Mask Channel")
        axes[1].axis('off')
        
        axes[2].imshow(res['rgb_saliency'], cmap='hot')
        axes[2].set_title("RGB Feature Importance\n(Numbers/Symbols/Colors)")
        axes[2].axis('off')
        
        axes[3].imshow(res['mask_saliency'], cmap='hot')
        axes[3].set_title("Mask Feature Importance\n(Overlap boundaries)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()