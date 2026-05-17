# IAPR 2026 — Final Project: UNO Vision Challenge

EPFL EE-451 final project. Recover the structured game state (center card,
active player, four hands) from snapshots of a multiplayer UNO game.

Group: `<group_id>` — `<kaggle_group_name>`. See `report.ipynb` for the full
write-up.

## Repository layout

```
.
├── main.py                 # produces the Kaggle submission CSV
├── report.ipynb            # the project report notebook
├── config/requirements.txt
├── data/
│   └── iapr-26-uno-vision-challenge/   # official Kaggle assets
├── models/                 # trained model checkpoints used at inference
│   ├── segmenter/used/scene_segmenter_unet_small.pth
│   └── card_classifier_cnn/used/latest/{card_classifier.pth, classes.npy, config.json}
├── training_data/          # generated reference/augmented data (regenerable)
│   ├── object_labels/{reference_cards.csv, augmented_cards.csv, augmented_scenes.json}
│   ├── reference_data/{card_crops, token_crops, backgrounds}
│   └── augmented_data/{card_images, card_masks, scene_images, scene_masks}
└── src/
   ├── inference.py             # models, inference, schema validation, CSV writing
   ├── data_augmentation.py     # regenerate augmented cards/scenes + report plots
   ├── segmenter_training.py    # train the U-Net segmenter + checkpoint diagnostics
   ├── classifier_training.py   # train/save the 4-channel classifier + config diagnostics
   └── report_viz.py            # final pipeline plots and labelled train.csv benchmark
```

## How to reproduce the Kaggle submission

```bash
pip install -r config/requirements.txt
python main.py
```

This loads the segmenter + classifier from `models/`, runs inference
on every image in `data/iapr-26-uno-vision-challenge/test_images/`, validates
the schema, and writes `submission.csv` at the repo root.

To run the full report (with figures, benchmark, and embedded submission):

```bash
jupyter notebook report.ipynb
```

## Regenerating training data and retraining models

The report notebook now contains executable cells for the three training
stages. They are guarded by explicit flags so opening the report does not
rewrite data or launch long GPU jobs accidentally:

- `RUN_FULL_AUGMENTATION = True` rebuilds augmented card crops, synthetic
   scenes, segmentation masks, and `augmented_scenes.json` from the reference card crops.
- `RUN_SEGMENTER_TRAINING = True` trains `SceneUNetSmall` on generated scenes
   and rewrites `models/segmenter/used/scene_segmenter_unet_small.pth`.
- `RUN_CLASSIFIER_TRAINING = True` runs the two-stage classifier curriculum
   (augmented cards, then generated scene crops) and rewrites
   `models/card_classifier_cnn/used/latest/`.

The same APIs can be called from Python:

```python
from src.data_augmentation import CreateAugmentedDataConfig, run_full_augmentation_pipeline
from src.segmenter_training import SegmenterPipelineConfig, run_full_segmenter_training
from src.classifier_training import TrainPipelineConfig, run_full_classifier_training

run_full_augmentation_pipeline(CreateAugmentedDataConfig())
run_full_segmenter_training(SegmenterPipelineConfig())
run_full_classifier_training(TrainPipelineConfig())
```

These calls only use `training_data/` and challenge-provided assets; the test
set remains inference-only.

## Compliance summary

The submission is bound by the IAPR 2026 challenge rules. The most important
constraints are summarised below; the project README and code comments make
the corresponding checks explicit at every stage.

| ID | Rule | Where enforced |
|----|------|----------------|
| R1 | No external datasets | Augmented data is composed only from challenge-provided cards/backgrounds. |
| R2 | No pretrained weights | Submitted models are custom architectures in `src/inference.py`, loaded from our checkpoints only. |
| R3 | Test set = inference only | Training notebooks operate on `training_data/` and challenge training assets; `main.py` only reads `test_images/`. |
| R4 | ≤12M params per model | `assert_param_cap` runs on every model load (`src/inference.py`, training pipelines). |
| R5 | Strict submission schema | `validate_row` in `src/inference.py` runs on every CSV row before writing. |
| R6 | Fixed player geometry | `assign_region` and `detect_active_player` in `src/inference.py` (p1=bottom, p2=right, p3=top, p4=left). |

## Pipeline overview

1. **Segmenter** (`SceneUNetSmall`) — small U-Net producing a per-pixel card
   foreground probability on a 256-px letterboxed input.
2. **Instance extraction** — threshold + connected components + bounded mask
   growth, yielding one mask per detected card.
3. **Classifier** (`CardResNet18SmallClassifier`) — compact ResNet-18 trained
   from random init, takes a 4-channel input (RGB + binary mask) so the
   network is forced to ignore background variation.
4. **Region assignment** — geometric assignment of each box to `center` or one
   of `p1..p4` per R6.
5. **Active player** — OpenCV token detection chooses the dark rectangular
   marker on plain backgrounds or the yellow round marker on textured
   backgrounds, then maps the marker center to the fixed player sectors.

## Final submission packaging

The submission archive must be named
`final_group_<group_id>_<kaggle_group_name>.zip` and contain at least:

- `report.ipynb`
- `main.py`
- `src/`
- the model checkpoints required by `main.py`
   (`models/segmenter/used/...` and
   `models/card_classifier_cnn/used/latest/...`)
- `config/requirements.txt`
