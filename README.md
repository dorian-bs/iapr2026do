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
├── requirements.txt
├── data/
│   └── iapr-26-uno-vision-challenge/   # official Kaggle assets
├── project/
│   ├── models/             # trained model checkpoints used at inference
│   │   ├── segmenter/used/scene_segmenter_unet_small.pth
│   │   └── card_classifier_cnn/used/latest/{card_classifier.pth, classes.npy, config.json}
│   ├── notebooks/          # training & data-generation notebooks (one per stage)
│   └── training_data/      # generated reference/augmented data (regenerable)
└── src/
    ├── inference.py        # scene image -> GameState (end-to-end inference)
    ├── submission.py       # submission CSV writer + schema validator (R5)
    ├── active_player.py    # token-detection stub (active player) — WIP
    ├── report_viz.py       # plotting helpers used by report.ipynb
    ├── shared/             # cross-stage primitives (models, masking, geometry)
    ├── create_reference_cards/   # extract reference cards from labelled images
    ├── create_augmented_data/    # synthesize training compositions
    ├── train_segmenter/          # train the U-Net segmenter
    ├── train_classifier_CNN/     # train the masked card classifier
    └── test_classifier_CNN/      # labelled-data benchmark used by the report
```

## How to reproduce the Kaggle submission

```bash
pip install -r requirements.txt
python main.py
```

This loads the segmenter + classifier from `project/models/`, runs inference
on every image in `data/iapr-26-uno-vision-challenge/test_images/`, validates
the schema, and writes `submission.csv` at the repo root.

To run the full report (with figures, benchmark, and embedded submission):

```bash
jupyter notebook report.ipynb
```

## Compliance summary

The submission is bound by the IAPR 2026 challenge rules. The most important
constraints are summarised below; the project README and code comments make
the corresponding checks explicit at every stage.

| ID | Rule | Where enforced |
|----|------|----------------|
| R1 | No external datasets | Augmented data is composed only from challenge-provided cards/backgrounds. |
| R2 | No pretrained weights | All models built with `weights=None` (`src/shared/card_models.py`). |
| R3 | Test set = inference only | Training notebooks operate on `train_images/`; `main.py` only reads `test_images/`. |
| R4 | ≤12M params per model | `assert_param_cap` runs on every model load (`src/inference.py`, training pipelines). |
| R5 | Strict submission schema | `src/submission.validate_row` runs on every CSV row before writing. |
| R6 | Fixed player geometry | `assign_region` in `src/shared/card_pipeline.py` (p1=bottom, p2=right, p3=top, p4=left). |

## Pipeline overview

1. **Segmenter** (`SceneUNetSmall`) — small U-Net producing a per-pixel card
   foreground probability on a 256-px letterboxed input.
2. **Instance extraction** — threshold + connected components + erosion-based
   touching-component split + bounded mask growth, yielding one mask per card.
3. **Classifier** (`CardResNet18SmallClassifier`) — compact ResNet-18 trained
   from random init, takes a 4-channel input (RGB + binary mask) so the
   network is forced to ignore background variation.
4. **Region assignment** — geometric assignment of each box to `center` or one
   of `p1..p4` per R6.
5. **Active player** — token detection (see `src/active_player.py`, **work in
   progress**).

## Final submission packaging

The submission archive must be named
`final_group_<group_id>_<kaggle_group_name>.zip` and contain at least:

- `report.ipynb`
- `main.py`
- `src/`
- the model checkpoints required by `main.py`
  (`project/models/segmenter/used/...` and
  `project/models/card_classifier_cnn/used/latest/...`)
- `requirements.txt`
