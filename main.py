"""Reproduce the Kaggle submission CSV from scratch.

Usage
-----
    python main.py
    python main.py --test-dir data/iapr-26-uno-vision-challenge/test_images \\
                   --models-dir models \\
                   --output submission.csv

This is the script referenced in the IAPR 2026 submission instructions: running
it from the repository root must regenerate the file uploaded to Kaggle, byte
for byte (assuming the same model checkpoints under `models/`).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.inference import InferenceConfig, load_engine, predict_from_path, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the IAPR 2026 UNO submission CSV.")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/iapr-26-uno-vision-challenge/test_images"),
        help="Folder containing the Kaggle test images (.jpg).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Folder with the segmenter + classifier bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path of the submission CSV to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_images = sorted(args.test_dir.glob("*.jpg"))
    if args.limit is not None:
        test_images = test_images[: args.limit]
    if not test_images:
        raise FileNotFoundError(f"No test images found under {args.test_dir}.")

    engine = load_engine(args.models_dir, config=InferenceConfig(), verbose=True)

    start_time = time.time()
    game_states = []
    for image_path in tqdm(test_images, desc="Predicting"):
        game_states.append(predict_from_path(engine, image_path))

    output_path = write_submission(game_states, args.output, validate=True)
    elapsed = time.time() - start_time
    print(f"[done] Wrote {len(game_states)} rows to {output_path} in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
