from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.paths import CLASSIFIER_CLASSES_DIR, CLASSIFIER_MODELS_DIR, SEGMENTER_MODELS_DIR


EXPECTED = [
    SEGMENTER_MODELS_DIR / "segmenter_unet_small.pth",
    CLASSIFIER_MODELS_DIR / "color_clf.pkl",
    CLASSIFIER_MODELS_DIR / "rank_clf.pkl",
    CLASSIFIER_CLASSES_DIR / "label_classes.npy",
    CLASSIFIER_CLASSES_DIR / "color_classes.npy",
    CLASSIFIER_CLASSES_DIR / "rank_classes.npy",
]


def main() -> int:
    missing = []
    for path in EXPECTED:
        status = "ok" if path.exists() else "missing"
        print(f"{status:7} {path}")
        if not path.exists():
            missing.append(path)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())