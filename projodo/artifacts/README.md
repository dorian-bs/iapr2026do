# Artifacts Organization

Generated project outputs that are not source data live here. Keep the repository root for entry points, documentation, and configuration.

```text
artifacts/
  models/
    segmenter/              # trained segmentation weights (.pth, .pt, checkpoints)
    classifiers/            # trained classifier objects (.pkl, .joblib)
      classes/              # label/class lookup arrays (.npy)
  submissions/              # generated Kaggle submission CSVs
  reports/                  # generated metrics, plots, and experiment summaries
  logs/                     # local training/inference logs; ignored by git except this README
  runs/                     # local experiment run outputs; ignored by git
  cache/                    # artifact-level caches; ignored by git
```

Model files in `artifacts/models/` are project-generated assets used by training, testing, and the final inference pipeline.
