# Training Scripts

Place segmentation and classifier training scripts here.

Train the segmenter with GPU-friendly defaults:

```powershell
python scripts/training/train_segmenter.py --epochs 25 --batch-size 16
```

Use `--num-workers` to tune data-loading parallelism and `--no-mixed-precision` to disable CUDA AMP if needed.
