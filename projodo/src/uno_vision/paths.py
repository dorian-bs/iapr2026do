"""Centralized filesystem paths for the active projodo package layout."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest project directory that contains the expected data and CLI files."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data").exists() and (candidate / "main.py").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()

# Official challenge data copied into the project layout.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_IMAGES_DIR = RAW_DATA_DIR / "reference_images"
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
TEST_IMAGES_DIR = RAW_DATA_DIR / "test_images"
TRAIN_CSV_PATH = RAW_DATA_DIR / "train.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

# Generated data derived only from the allowed challenge assets.
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_CARDS_DIR = PROCESSED_DATA_DIR / "reference_cards"
AUGMENTATIONS_DIR = PROCESSED_DATA_DIR / "augmentations"
GAME_SNAPSHOTS_DIR = PROCESSED_DATA_DIR / "game_snapshots"

# Runtime artifacts created by training, inference, and evaluation scripts.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SEGMENTER_MODELS_DIR = MODELS_DIR / "segmenter"
CLASSIFIER_MODELS_DIR = MODELS_DIR / "classifiers"
CLASSIFIER_CLASSES_DIR = CLASSIFIER_MODELS_DIR / "classes"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
