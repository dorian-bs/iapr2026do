from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.data.augmentations import generate_augmentation_images, generate_augmentation_masks


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate augmented card crop images and masks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-masks", action="store_true")
    args = parser.parse_args()
    if not args.skip_images:
        image_count = generate_augmentation_images(seed=args.seed)
        print(f"Generated {image_count} augmented images.")
    if not args.skip_masks:
        mask_count = generate_augmentation_masks(seed=args.seed)
        print(f"Generated {mask_count} augmented masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())