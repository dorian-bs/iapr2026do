from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from uno_vision.pipeline.cards import predict_cards_in_image


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="UNO vision project entry point.")
	parser.add_argument(
		"--image",
		type=Path,
		help="Run the current card-region prediction pipeline on one image.",
	)
	parser.add_argument(
		"--max-components",
		type=int,
		default=24,
		help="Maximum candidate components used by the segmenter pre-detection step.",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Print predictions as JSON instead of readable text.",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.image is None:
		parser.print_help()
		return 0

	predictions, _, _ = predict_cards_in_image(args.image, max_components=args.max_components)
	rows = [
		{
			"index": prediction.index,
			"box": prediction.box,
			"area": prediction.area,
			"label": prediction.label,
			"color": prediction.color,
			"rank": prediction.rank,
		}
		for prediction in predictions
	]
	if args.json:
		print(json.dumps(rows, indent=2))
	else:
		for row in rows:
			print(f"{row['index']:02d} {row['label']} box={row['box']} area={row['area']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
