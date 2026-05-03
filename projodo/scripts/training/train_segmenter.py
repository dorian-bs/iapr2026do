from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.segmentation.data import collect_segmentation_pairs
from uno_vision.segmentation.training import train_segmenter


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the UNO card segmentation model.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = collect_segmentation_pairs()
    print(f"Loaded {len(pairs)} segmentation pairs.")
    history = train_segmenter(pairs=pairs, epochs=args.epochs, batch_size=args.batch_size, random_state=args.seed)
    print(f"Saved segmenter to {history.model_path}")
    print(f"Final val IoU: {history.val_ious[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())