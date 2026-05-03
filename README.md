# UNO Vision Project

This repository is organized around a reusable Python package plus thin scripts for data preparation, training, evaluation, and inference.

```text
data/                    official and generated data
artifacts/               trained models, submissions, reports, logs, and runs
src/uno_vision/           reusable project package
scripts/                  command-line entry points built on src/uno_vision
notebooks/                exploratory notebooks and experiment history
main.py                   final project entry point
```

Key package areas:

- `uno_vision.data`: reference-card extraction and augmentation generation.
- `uno_vision.segmentation`: U-Net model, datasets, training, and inference helpers.
- `uno_vision.classification`: feature extraction, classifier training, and card prediction.
- `uno_vision.pipeline`: end-to-end card-region prediction utilities.

Run scripts from the repository root, for example:

```powershell
python scripts/evaluation/check_artifacts.py
python scripts/training/train_classifiers.py
python scripts/inference/predict_cards.py data/raw/test_images/L1000793.jpg
```

The current pipeline predicts card regions and labels. The final game-state submission logic still needs active-player and player-hand assembly on top of these building blocks.