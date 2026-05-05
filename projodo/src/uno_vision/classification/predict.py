"""Load classifier artifacts and predict UNO card labels from BGR crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from uno_vision.classification.features import (
    compose_card_label,
    extract_color_features,
    extract_rank_features,
    letterbox_cv2,
)
from uno_vision.paths import CLASSIFIER_CLASSES_DIR, CLASSIFIER_MODELS_DIR


@dataclass(frozen=True)
class CardPrediction:
    """Classifier output for one cropped card image."""

    label: str
    color: str
    rank: str


@dataclass
class CardClassifier:
    """Pair of color/rank classifiers plus their saved class vocabularies."""

    color_clf: object
    rank_clf: object
    color_classes: np.ndarray
    rank_classes: np.ndarray

    def predict_bgr(self, img_bgr: np.ndarray) -> CardPrediction:
        """Predict the full UNO label for a single BGR card crop."""

        img_lb = letterbox_cv2(img_bgr)
        color_vec = extract_color_features(img_lb).reshape(1, -1)
        rank_vec = extract_rank_features(img_lb).reshape(1, -1)
        color_idx = int(self.color_clf.predict(color_vec)[0])
        rank_idx = int(self.rank_clf.predict(rank_vec)[0])
        color = str(self.color_classes[color_idx])
        rank = str(self.rank_classes[rank_idx])
        return CardPrediction(compose_card_label(color, rank), color, rank)


def load_card_classifier(
    classifier_dir: Path = CLASSIFIER_MODELS_DIR,
    classes_dir: Path = CLASSIFIER_CLASSES_DIR,
) -> CardClassifier:
    """Load trained classifier models and class arrays from artifact directories."""

    color_model_path = classifier_dir / "color_clf.pkl"
    rank_model_path = classifier_dir / "rank_clf.pkl"
    color_classes_path = classes_dir / "color_classes.npy"
    rank_classes_path = classes_dir / "rank_classes.npy"
    missing = [
        path
        for path in (color_model_path, rank_model_path, color_classes_path, rank_classes_path)
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing classifier artifact(s): {joined}")
    return CardClassifier(
        color_clf=joblib.load(color_model_path),
        rank_clf=joblib.load(rank_model_path),
        color_classes=np.load(color_classes_path, allow_pickle=True),
        rank_classes=np.load(rank_classes_path, allow_pickle=True),
    )
