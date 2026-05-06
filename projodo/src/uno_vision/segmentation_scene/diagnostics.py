"""Diagnostics and visualization helpers for scene-segmentation training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

from uno_vision.segmentation_scene.data import IMAGE_MEAN, IMAGE_STD, SceneSegDataset

if TYPE_CHECKING:
    from uno_vision.segmentation_scene.training import SceneSegmentationTrainingHistory


@dataclass(frozen=True)
class SceneValidationMetric:
    """Per-sample overlap and foreground-fraction metrics on the validation split."""

    idx: int
    image: str
    mask: str
    iou: float
    dice: float
    gt_fg_frac: float
    pred_fg_frac: float


def tensor_to_rgb(image_t: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to RGB for plotting."""

    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std = np.array(IMAGE_STD, dtype=np.float32)
    image = image_t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * std + mean, 0.0, 1.0)


def iou_and_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float]:
    """Return IoU and Dice scores for boolean binary masks."""

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = float(intersection / (union + 1e-8))
    dice = float((2.0 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-8))
    return iou, dice


def evaluate_validation_pairs(
    model: torch.nn.Module,
    device: torch.device,
    dataset: SceneSegDataset,
    pairs: list[tuple[str, str]],
    threshold: float = 0.5,
) -> list[SceneValidationMetric]:
    """Compute overlap metrics for each validation pair."""

    metrics: list[SceneValidationMetric] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_t, mask_t = dataset[idx]
            logits = model(image_t.unsqueeze(0).to(device))
            pred_prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            gt = mask_t[0].numpy() > 0.5
            pred = pred_prob > threshold
            iou, dice = iou_and_dice(pred, gt)
            image_path, mask_path = pairs[idx]

            metrics.append(
                SceneValidationMetric(
                    idx=idx,
                    image=Path(image_path).name,
                    mask=Path(mask_path).name,
                    iou=iou,
                    dice=dice,
                    gt_fg_frac=float(gt.mean()),
                    pred_fg_frac=float(pred.mean()),
                )
            )
    return metrics


def summarize_values(values: list[float]) -> dict[str, float]:
    """Return mean, median, min, and max for a list of scalar values."""

    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def metric_summary_table(metrics: list[SceneValidationMetric]) -> dict[str, dict[str, float]]:
    """Build summary statistics for key validation metrics."""

    if not metrics:
        raise ValueError("No metrics available to summarize.")

    return {
        "iou": summarize_values([row.iou for row in metrics]),
        "dice": summarize_values([row.dice for row in metrics]),
        "gt_fg_frac": summarize_values([row.gt_fg_frac for row in metrics]),
        "pred_fg_frac": summarize_values([row.pred_fg_frac for row in metrics]),
    }


def print_metric_summary(metrics: list[SceneValidationMetric]) -> None:
    """Print compact summary lines for validation metrics."""

    summary = metric_summary_table(metrics)
    for name in ("iou", "dice", "gt_fg_frac", "pred_fg_frac"):
        stats = summary[name]
        print(
            f"{name:12s} mean={stats['mean']:.3f} median={stats['median']:.3f} "
            f"min={stats['min']:.3f} max={stats['max']:.3f}"
        )


def metrics_to_dicts(metrics: list[SceneValidationMetric]) -> list[dict[str, float | int | str]]:
    """Convert metric dataclasses to JSON-serializable dictionaries."""

    return [asdict(row) for row in metrics]


def plot_training_curves(history: SceneSegmentationTrainingHistory) -> plt.Figure:
    """Plot training/validation loss and IoU curves for one run."""

    epochs = np.arange(1, len(history.train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history.train_losses, label="train")
    axes[0].plot(epochs, history.val_losses, label="val")
    axes[0].axvline(history.best_epoch, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.train_ious, label="train")
    axes[1].plot(epochs, history.val_ious, label="val")
    axes[1].axvline(history.best_epoch, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, np.array(history.train_ious) - np.array(history.val_ious), label="train - val")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Generalization gap")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    return fig


def plot_metric_distributions(metrics: list[SceneValidationMetric]) -> plt.Figure:
    """Plot IoU and Dice histograms plus foreground-fraction scatter."""

    if not metrics:
        raise ValueError("No metrics available for plotting.")

    ious = np.array([row.iou for row in metrics], dtype=np.float32)
    dices = np.array([row.dice for row in metrics], dtype=np.float32)
    gt_fracs = np.array([row.gt_fg_frac for row in metrics], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(ious, bins=20, range=(0, 1))
    axes[0].set_title("Validation IoU")
    axes[0].set_xlabel("IoU")
    axes[0].set_ylabel("Samples")

    axes[1].hist(dices, bins=20, range=(0, 1))
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Dice")

    axes[2].scatter(gt_fracs, ious, alpha=0.7)
    axes[2].set_title("Mask size vs IoU")
    axes[2].set_xlabel("GT foreground fraction")
    axes[2].set_ylabel("IoU")
    axes[2].set_ylim(0.0, 1.02)

    fig.tight_layout()
    return fig


def plot_validation_preview(
    dataset: SceneSegDataset,
    pairs: list[tuple[str, str]],
    max_samples: int = 6,
) -> plt.Figure:
    """Visualize image, mask, and overlay previews from a validation dataset."""

    preview_count = min(max_samples, len(dataset))
    if preview_count == 0:
        raise ValueError("Dataset is empty; cannot build a preview figure.")

    fig, axes = plt.subplots(preview_count, 3, figsize=(12, 3.2 * preview_count), squeeze=False)
    for row_idx in range(preview_count):
        image_t, mask_t = dataset[row_idx]
        image_np = tensor_to_rgb(image_t)
        mask_np = mask_t[0].numpy() > 0.5
        overlay = image_np.copy()
        overlay[mask_np] = 0.55 * overlay[mask_np] + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)

        image_path, _ = pairs[row_idx]
        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title(Path(image_path).stem, fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Target mask", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Mask overlay", fontsize=9)
        axes[row_idx, 2].axis("off")

    fig.tight_layout()
    return fig


def plot_worst_validation_samples(
    model: torch.nn.Module,
    device: torch.device,
    dataset: SceneSegDataset,
    metrics: list[SceneValidationMetric],
    threshold: float = 0.5,
    max_samples: int = 8,
) -> plt.Figure:
    """Visualize the lowest-IoU validation samples and their error maps."""

    if not metrics:
        raise ValueError("No metrics available for worst-sample visualization.")

    worst = sorted(metrics, key=lambda row: row.iou)[: min(max_samples, len(metrics))]
    fig, axes = plt.subplots(len(worst), 5, figsize=(16, 3.0 * len(worst)), squeeze=False)

    model.eval()
    with torch.no_grad():
        for row_idx, metric_row in enumerate(worst):
            image_t, mask_t = dataset[int(metric_row.idx)]
            logits = model(image_t.unsqueeze(0).to(device))
            pred_prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            image_np = tensor_to_rgb(image_t)
            gt = mask_t[0].numpy() > 0.5
            pred = pred_prob > threshold

            fp = np.logical_and(pred, np.logical_not(gt))
            fn = np.logical_and(np.logical_not(pred), gt)
            diff = np.zeros((*gt.shape, 3), dtype=np.float32)
            diff[fp] = [1.0, 0.35, 0.0]
            diff[fn] = [0.1, 0.75, 1.0]

            axes[row_idx, 0].imshow(image_np)
            axes[row_idx, 0].set_title(metric_row.image, fontsize=8)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 1].set_title("GT", fontsize=8)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
            axes[row_idx, 2].set_title("Prob", fontsize=8)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 3].set_title(f"Pred IoU={metric_row.iou:.3f}", fontsize=8)
            axes[row_idx, 3].axis("off")

            axes[row_idx, 4].imshow(image_np)
            axes[row_idx, 4].imshow(diff, alpha=0.65)
            axes[row_idx, 4].set_title("Error overlay", fontsize=8)
            axes[row_idx, 4].axis("off")

    fig.tight_layout()
    return fig
