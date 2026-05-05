from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.data.reference_cards_base import extract_reference_card_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract reference-card crops, masks, and components.")
    parser.add_argument("image_name", help="Reference image id, for example L1000768.")
    parser.add_argument("--components", type=int, required=True, help="Number of largest components to keep.")
    args = parser.parse_args()
    result = extract_reference_card_assets(args.image_name, args.components)
    print(f"Wrote {len(result.crops)} crops, {len(result.masks)} masks, and {len(result.components)} components to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())