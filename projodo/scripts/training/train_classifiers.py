from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.classification.training import collect_classifier_samples, save_classifier_artifacts, train_classifiers


def main() -> int:
    parser = argparse.ArgumentParser(description="Train UNO color and rank classifiers.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = collect_classifier_samples()
    print(f"Loaded {len(samples)} classifier samples.")
    result = train_classifiers(samples=samples, test_size=args.test_size, random_state=args.seed)
    save_classifier_artifacts(result)
    for key, value in result.metrics.items():
        print(f"{key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())