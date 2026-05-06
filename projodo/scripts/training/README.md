# Training Scripts

Place segmentation and classifier training scripts here.

Train the segmenter with GPU-friendly defaults:

```powershell
python scripts/training/train_segmenter.py --epochs 25 --batch-size 16
```

Use `--num-workers` to tune data-loading parallelism and `--no-mixed-precision` to disable CUDA AMP if needed.

Train the scene segmenter and emit debug artifacts:

```powershell
python scripts/training/train_scene_segmenter.py --epochs 15 --batch-size 8
```

Scene training writes JSON summaries and optional plots into `artifacts/reports/scene_segmenter`.
Useful flags:

- `--warm-start artifacts/models/segmenter/segmenter_unet_small.pth` to fine-tune from an existing checkpoint.
- `--scenes-dir data/processed/game_snapshots` to override the default snapshot pair location.
- `--no-plots` to skip figure generation when running remotely.
- `--skip-diagnostics` to train only and avoid the post-training validation report.
