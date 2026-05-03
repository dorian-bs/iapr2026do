from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.pipeline.cards import predict_cards_in_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict card labels in one image with the current pipeline.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--max-components", type=int, default=24)
    args = parser.parse_args()
    predictions, _, _ = predict_cards_in_image(args.image, max_components=args.max_components)
    print(json.dumps([prediction.__dict__ for prediction in predictions], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())