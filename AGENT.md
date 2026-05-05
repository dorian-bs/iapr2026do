# AGENT.md - UNO Vision Project

Project guidance for AI agents working in this repository. This file replaces the old
`CLAUDE.md` that came from a single-prompt prototype attempt.

The current repository contains useful baseline code from that attempt, but the final
project should be built incrementally by the student. Agents should preserve compliance
constraints, help with focused tasks, and avoid taking over broad implementation work
unless explicitly asked.

---

## 1. Collaboration Mode

- The student wants to code as much as possible themselves.
- Prefer explanation, review, debugging help, small targeted edits, and clear next steps.
- Do not rewrite large parts of the pipeline or generate an entire final solution unless
  the student explicitly asks for that scope.
- Treat the generated baseline as a reference and starting point, not as the final design.
- Keep changes small, readable, and easy for the student to understand and continue.
- When making code changes, explain what changed and why in plain engineering terms.
- If a task risks violating the competition rules below, stop and ask before proceeding.

---

## 2. Hard Rules - Competition Compliance

These rules are constraints from the EPFL IAPR 2026 UNO Vision challenge. Violating one
can cost major grading penalties. Treat them as inviolable.

| ID | Rule | One-line summary |
|----|------|------------------|
| R1 | No external datasets | Use only `iapr-26-uno-vision-challenge` Kaggle assets. |
| R2 | No pretrained weights | Architectures are allowed; pretrained weights are not. |
| R3 | Test set = inference only | Never train, augment, pseudo-label, or tune on test images. |
| R4 | <= 12M trainable params per model | Print and assert parameter counts for every model. |
| R5 | Strict submission CSV format | Validate the exact Kaggle schema before submission. |
| R6 | Fixed player geometry | p1=bottom, p2=right, p3=top, p4=left. |

### R1 - No External Datasets

Forbidden:
- ImageNet, COCO, OpenImages, HuggingFace datasets, other Kaggle datasets, scraped images,
  GitHub UNO datasets, personal photos, or any other non-competition image source.

Allowed:
- Official Kaggle challenge files only.
- Synthetic compositing from provided cards/backgrounds.
- Augmentations derived only from provided training/reference data.
- Bootstrap-labeled crops only if they come from allowed training images, never test images.

### R2 - No Pretrained Models or Weights

Forbidden:
- `pretrained=True`, `weights="DEFAULT"`, Hugging Face `from_pretrained`, ONNX model zoos,
  downloaded `.pt`/`.pth`/`.bin`/`.safetensors` weights, CLIP, DINO, SAM, YOLO, MediaPipe,
  OpenCV DNN model loading, Detectron2/Ultralytics model zoos.

Allowed:
- Custom models initialized randomly.
- Library architectures only when explicitly instantiated with `weights=None` or
  `pretrained=False`, with a short compliance comment.
- Project-generated weights trained only from allowed challenge data.

No transfer learning and no distillation from pretrained teachers.

### R3 - Test Set Is Inference Only

Forbidden:
- Training, fine-tuning, data augmentation, pseudo-labeling, batch-norm/statistics updates,
  or hyperparameter selection using test images or leaderboard feedback.

Allowed:
- Running inference on test images to produce the final submission CSV.

Training pipelines should keep train/test path variables separate and enforce checks such as:

```python
assert "test" not in str(path).lower()
```

### R4 - Parameter Cap

Every model instantiation used for training or inference must print and assert the trainable
parameter count:

```python
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n <= 12_000_000, f"Model has {n:,} params, exceeds 12M cap"
print(f"[compliance] {model.__class__.__name__}: {n:,} trainable params")
```

For multi-model pipelines, report each model and the total.

### R5 - Submission CSV Format

Columns, in exact order:

```text
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
```

Rules:
- `image_id` must match test IDs exactly, case-sensitive, with no extension.
- Card encoding is `<color>_<value>` where color is one of `r,g,b,y` and value is one of
  `0..9,skip,reverse,draw_2`.
- Special cards are `wild` and `draw_4`, with no color prefix.
- Multi-card hands are semicolon-separated with no spaces. Duplicates are repeated. Order is
  irrelevant unless the official metric later says otherwise.
- Empty hands must be the literal string `EMPTY`, never blank or NaN.
- `active_player` must be one of `p1,p2,p3,p4`.
- Before any final submission, add or run a schema validator against this format.

### R6 - Fixed Player Geometry

- `p1` = bottom
- `p2` = right
- `p3` = top
- `p4` = left
- Center card = middle
- Missing players or empty hands should be encoded as `EMPTY`.

---

## 3. Library Policy

Allowed libraries:
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `opencv-python` for image processing, but no `cv2.dnn` model loading
- `scikit-image`, `scikit-learn`
- `torch`, `torchvision` architectures only, no pretrained weights
- `Pillow`, `imageio`, `tqdm`
- `kagglehub`, `jupyter`, `ipykernel`
- Existing declared project dependencies in `projodo/requirements.txt`

Forbidden or high-risk libraries:
- `transformers` weights, `timm` with pretrained weights, `ultralytics`, `mmdetection` model
  zoos, `mediapipe`, `face_recognition`, `easyocr`, `paddleocr`, `clip`, `open_clip`,
  `segment_anything`, or any package that auto-downloads weights or data at import/runtime.

Ask before adding any new dependency that is not already in `projodo/requirements.txt`.

---

## 4. Current Repository Setup

The active project code is in `projodo/`.

```text
iapr2026do/
├── AGENT.md                         # this guidance file
├── data/                            # official Kaggle challenge assets at workspace root
│   └── iapr-26-uno-vision-challenge/
├── project/                         # legacy scratch/generated extraction outputs
└── projodo/                         # active Python project from the prototype attempt
    ├── main.py                      # current CLI entry point
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── raw/                     # challenge files copied into project layout
    │   └── processed/               # generated reference cards and augmentations
    ├── artifacts/
    │   ├── models/                  # generated project models/classes
    │   ├── submissions/             # generated Kaggle CSVs
    │   ├── reports/
    │   ├── logs/
    │   ├── runs/
    │   └── cache/
    ├── notebooks/                   # exploratory notebooks and experiment history
    ├── scripts/                     # thin command-line entry points
    └── src/uno_vision/              # reusable package code
```

Important current state:
- The current pipeline predicts card regions and card labels.
- Final game-state assembly is not complete yet: active player detection, hand assignment,
  center-card selection, and final submission generation still need work.
- `projodo/main.py` currently runs single-image card-region prediction, not full Kaggle
  submission reproduction.
- `project/` appears to be legacy scratch output from the earlier attempt; prefer code and
  data paths under `projodo/` unless the student asks otherwise.

---

## 5. Useful Commands

Run commands from `projodo/` unless noted otherwise.

Install dependencies in the chosen Python environment:

```powershell
pip install -r requirements.txt
```

Check expected generated artifacts:

```powershell
python scripts/evaluation/check_artifacts.py
```

Run current single-image inference:
w
```powershell
python scripts/inference/predict_cards.py data/raw/test_images/L1000793.jpg
```

or:

```powershell
python main.py --image data/raw/test_images/L1000793.jpg --json
```

Train scripts exist for the current baseline, but before running or modifying training code,
verify R1-R4 compliance and test-data isolation.

---

## 6. Development Guidelines

- Put reusable logic in `projodo/src/uno_vision/`.
- Keep scripts in `projodo/scripts/` thin and focused on command-line orchestration.
- Keep notebooks exploratory; important logic should move into `src/uno_vision/` or scripts.
- Do not store new generated outputs in the repository root. Use `projodo/artifacts/` or
  `projodo/data/processed/` as appropriate.
- Do not modify official raw data except to copy it into the project layout when needed.
- Keep code style consistent with the existing package.
- Prefer simple, inspectable classical/image-processing baselines before adding heavier models.
- Add comments and docstrings only when they clarify intent, data flow, compliance-relevant
  constraints, or non-obvious image-processing/modeling choices.
- Every reusable Python source file in `projodo/src/uno_vision/` should start with a short
  module docstring that explains the file's role in the project.
- Public functions, classes, and dataclasses should have concise docstrings that describe
  what they return or coordinate, especially for training, inference, data generation, and
  feature extraction code.
- Prefer short comments before complex blocks over inline narration. Good comments explain
  why a threshold, morphology step, augmentation, split, or artifact path exists; avoid
  comments that merely restate the next line of code.
- Keep comments current when changing behavior. Remove or revise stale comments in the same
  edit that changes the code they describe.
- Do not commit, reset, or discard changes unless the student explicitly asks.

---

## 7. Pre-Submission Checklist

- [ ] No external data sources are downloaded or referenced.
- [ ] All learned weights are trained only from allowed challenge data.
- [ ] All model constructors explicitly avoid pretrained weights.
- [ ] Trainable parameter counts are printed and asserted <= 12M for every model.
- [ ] No test image is used in any training, augmentation, pseudo-labeling, or tuning step.
- [ ] Final CSV columns exactly match the required order.
- [ ] Empty hands use `EMPTY`.
- [ ] Player geometry follows p1=bottom, p2=right, p3=top, p4=left.
- [ ] A submission validator passes before upload.
- [ ] `projodo/main.py` can reproduce the submitted CSV when the final pipeline is complete.
