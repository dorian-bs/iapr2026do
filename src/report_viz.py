"""Plotting helpers used by the report notebook.

These functions all take the inference outputs already produced by
`src/inference.py` so the notebook stays short: build a `GameState`, hand it
here, get a figure. Keeping plotting code out of the notebook also keeps the
notebook executable end-to-end on the TA's machine without 200 lines of
matplotlib in every cell.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.inference import CardPrediction, GameState, InferenceEngine, predict_cards
from src.shared.card_pipeline import (
    boxes_from_probability,
    segment_scene_probability,
)


REGION_COLORS = {
    "center": "gold",
    "p1": "tab:blue",
    "p2": "tab:green",
    "p3": "tab:red",
    "p4": "tab:purple",
}


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


def plot_region_layout(image_w: int = 1000, image_h: int = 1000) -> None:
    """Schematic of the fixed player geometry (R6).

    Useful in the report to motivate the `assign_region` heuristic before
    discussing failure cases.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    ax.set_aspect("equal")

    # Central rectangle = "center" region. Outside the rectangle, the closest
    # player edge wins (see `src.shared.card_pipeline.assign_region`).
    cx_lo, cx_hi = 0.36, 0.64
    cy_lo, cy_hi = 0.30, 0.70
    ax.add_patch(Rectangle(
        (cx_lo * image_w, cy_lo * image_h),
        (cx_hi - cx_lo) * image_w,
        (cy_hi - cy_lo) * image_h,
        fill=True, facecolor="gold", alpha=0.4, edgecolor="black",
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
                color=color, fontsize=11, fontweight="bold")

    ax.set_title("Fixed player geometry (R6)")
    ax.axis("off")
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
        tick_labels=["center", "p1", "p2", "p3", "p4"],
    )
    ax.set_ylabel("Cards detected")
    ax.set_title("Per-region card count distribution")
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
