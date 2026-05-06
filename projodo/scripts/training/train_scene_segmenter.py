from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.paths import REPORTS_DIR, SEGMENTER_MODELS_DIR
from uno_vision.segmentation_scene.data import SceneSegDataset, collect_scene_pairs
from uno_vision.segmentation_scene.diagnostics import (
    evaluate_validation_pairs,
    metric_summary_table,
    metrics_to_dicts,
    plot_metric_distributions,
    plot_training_curves,
    plot_validation_preview,
    plot_worst_validation_samples,
    print_metric_summary,
)
from uno_vision.segmentation_scene.inference import load_scene_segmenter
from uno_vision.segmentation_scene.training import train_scene_segmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the UNO scene foreground segmenter and emit debugging reports."
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SEGMENTER_MODELS_DIR / "scene_segmenter_unet_small.pth",
    )
    parser.add_argument("--save-last-checkpoint", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--preview-samples", type=int, default=6)
    parser.add_argument("--worst-samples", type=int, default=8)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR / "scene_segmenter")
    return parser


def save_json(path: Path, payload: object) -> None:
    """Write one JSON artifact under the diagnostics report directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()

    pairs = collect_scene_pairs()
    print(f"Loaded {len(pairs)} scene image-mask pairs.")

    history = train_scene_segmenter(
        pairs=pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        mixed_precision=not args.no_mixed_precision,
        preload_to_ram=not args.no_preload,
        train_augment=not args.no_train_augment,
        image_size=args.image_size,
        random_state=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step_size=args.step_size,
        scheduler_gamma=args.gamma,
        warm_start_path=args.warm_start,
        output_path=args.output_path,
        save_last_checkpoint=args.save_last_checkpoint,
    )

    print(f"Saved best scene segmenter to {history.model_path}")
    print(f"Best val IoU: {history.best_val_iou:.4f} at epoch {history.best_epoch}")

    if args.skip_diagnostics:
        return 0

    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_ds = SceneSegDataset(
        history.val_pairs,
        augment=False,
        image_size=args.image_size,
        preload_to_ram=not args.no_preload,
        verbose=False,
    )

    model, device, selected_model_path = load_scene_segmenter(history.model_path)
    metrics = evaluate_validation_pairs(
        model=model,
        device=device,
        dataset=val_ds,
        pairs=history.val_pairs,
        threshold=args.threshold,
    )

    print("Validation summary:")
    print_metric_summary(metrics)

    history_payload = {
        "model_path": str(history.model_path),
        "best_epoch": history.best_epoch,
        "best_val_iou": history.best_val_iou,
        "device": history.device,
        "train_losses": history.train_losses,
        "val_losses": history.val_losses,
        "train_ious": history.train_ious,
        "val_ious": history.val_ious,
        "learning_rates": history.learning_rates,
        "epoch_times": history.epoch_times,
        "train_samples": len(history.train_pairs),
        "val_samples": len(history.val_pairs),
        "selected_model_path": str(selected_model_path),
    }

    save_json(reports_dir / "history.json", history_payload)
    save_json(reports_dir / "validation_metrics.json", metrics_to_dicts(metrics))
    save_json(reports_dir / "validation_summary.json", metric_summary_table(metrics))

    if not args.no_plots:
        training_fig = plot_training_curves(history)
        training_fig.savefig(reports_dir / "training_curves.png", dpi=160, bbox_inches="tight")
        plt.close(training_fig)

        metric_fig = plot_metric_distributions(metrics)
        metric_fig.savefig(reports_dir / "validation_distributions.png", dpi=160, bbox_inches="tight")
        plt.close(metric_fig)

        if args.preview_samples > 0:
            preview_fig = plot_validation_preview(
                dataset=val_ds,
                pairs=history.val_pairs,
                max_samples=args.preview_samples,
            )
            preview_fig.savefig(reports_dir / "validation_preview.png", dpi=160, bbox_inches="tight")
            plt.close(preview_fig)

        if args.worst_samples > 0:
            worst_fig = plot_worst_validation_samples(
                model=model,
                device=device,
                dataset=val_ds,
                metrics=metrics,
                threshold=args.threshold,
                max_samples=args.worst_samples,
            )
            worst_fig.savefig(reports_dir / "validation_worst_samples.png", dpi=160, bbox_inches="tight")
            plt.close(worst_fig)

    print(f"Diagnostics saved to {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
