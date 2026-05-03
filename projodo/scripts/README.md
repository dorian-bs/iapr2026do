# Scripts Organization

Runnable scripts live here. Shared implementation should go in `src/uno_vision/`; scripts should stay thin and call that package.

```text
scripts/
  data/                     # data preparation and augmentation entry points
  training/                 # model and classifier training entry points
  evaluation/               # validation, diagnostics, and metrics entry points
  inference/                # prediction and submission-generation helpers
```

Keep `main.py` at the repository root as the final challenge entry point, and let it import reusable code from `src/uno_vision/`.
