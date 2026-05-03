# Data Organization

This project predicts UNO game state from scene images: center card, active player, and each player's cards.

Use this layout to keep official inputs separate from script-generated artifacts:

```text
data/
  raw/                    # official challenge files: train/test/reference images and CSVs
  processed/
    reference_cards/       # crops, masks, components, and labels derived from reference cards
    augmentations/         # augmented crop images, masks, and labels derived from reference cards
  interim/                 # temporary experiment outputs; ignored by git
  cache/                   # reusable local caches; ignored by git
  external/                # external data, if ever allowed; ignored by git
```

The challenge slides forbid external data for the graded solution, so `data/external/` should remain unused unless the rules change.

Trained models, classifier objects, generated submissions, logs, and reports belong in `artifacts/`, not in `data/` or the repository root.
