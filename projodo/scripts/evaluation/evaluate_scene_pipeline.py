from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uno_vision.classification.predict import load_card_classifier
from uno_vision.paths import REPORTS_DIR, TRAIN_CSV_PATH, TRAIN_IMAGES_DIR
from uno_vision.pipeline.cards import classify_regions, regions_from_proposal_boxes, split_probability_mask
from uno_vision.segmentation.inference import load_segmenter, segment_image


COLORS = {"r", "g", "b", "y"}
RANKS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw_2"}
SPECIAL_CARDS = {"wild", "draw_4"}


def split_cards(value: str) -> list[str]:
    if value == "EMPTY":
        return []
    return value.split(";")


def expected_cards(row: dict[str, str]) -> list[str]:
    cards = []
    if row["center_card"] != "EMPTY":
        cards.append(row["center_card"])
    for key in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
        cards.extend(split_cards(row[key]))
    return cards


def is_valid_card_label(label: str) -> bool:
    if label in SPECIAL_CARDS:
        return True
    if "_" not in label:
        return False
    color, rank = label.split("_", 1)
    return color in COLORS and rank in RANKS


def short_list(items: list[str], max_items: int = 5) -> str:
    if not items:
        return "-"
    shown = items[:max_items]
    suffix = "" if len(items) <= max_items else f" +{len(items) - max_items}"
    return ";".join(shown) + suffix


def save_overlay(image_id: str, img_bgr, predictions, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = img_bgr.copy()
    for pred in predictions:
        x0, y0, x1, y1 = pred.box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(
            overlay,
            pred.label,
            (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / f"{image_id}_overlay.jpg"), overlay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate scene-level card detection on labeled train images.")
    parser.add_argument("--limit", type=int, default=12, help="Number of train images to evaluate. Use 0 for all.")
    parser.add_argument("--max-components", type=int, default=24)
    parser.add_argument("--proposal-prob-thresh", type=float, default=0.35)
    parser.add_argument("--high-prob-thresh", type=float, default=0.65)
    parser.add_argument("--card-min-dist", type=int, default=60)
    parser.add_argument("--use-mask-split", action="store_true", help="Use connected mask regions instead of proposal boxes.")
    parser.add_argument("--save-overlays", action="store_true", help="Save predicted boxes to artifacts/reports/scene_eval.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with TRAIN_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[:args.limit]

    model, device, model_path = load_segmenter()
    classifier = load_card_classifier()
    overlay_dir = REPORTS_DIR / "scene_eval"

    total_expected = 0
    total_predicted = 0
    total_matched = 0
    total_abs_count_error = 0
    total_invalid = 0

    print(f"Loaded segmenter: {model_path}")
    print(f"Evaluating {len(rows)} train image(s)")
    print("image_id  exp pred match invalid missing extra")

    for row in rows:
        image_id = row["image_id"]
        image_path = TRAIN_IMAGES_DIR / f"{image_id}.jpg"
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"{image_id}  missing image")
            continue

        expected = expected_cards(row)
        global_prob, boxes, _, _ = segment_image(img_bgr, model, device, max_components=args.max_components)
        if args.use_mask_split:
            _, regions = split_probability_mask(global_prob, args.high_prob_thresh, args.card_min_dist)
        else:
            regions = regions_from_proposal_boxes(global_prob, boxes, prob_thresh=args.proposal_prob_thresh)
        predictions = classify_regions(img_bgr, regions, classifier)
        predicted = [pred.label for pred in predictions]

        expected_counter = Counter(expected)
        predicted_counter = Counter(predicted)
        matched = sum((expected_counter & predicted_counter).values())
        missing = list((expected_counter - predicted_counter).elements())
        extra = list((predicted_counter - expected_counter).elements())
        invalid = [label for label in predicted if not is_valid_card_label(label)]

        total_expected += len(expected)
        total_predicted += len(predicted)
        total_matched += matched
        total_abs_count_error += abs(len(expected) - len(predicted))
        total_invalid += len(invalid)

        print(
            f"{image_id:8} {len(expected):3d} {len(predicted):4d} {matched:5d} "
            f"{len(invalid):7d} {short_list(missing):22s} {short_list(extra)}"
        )

        if args.save_overlays:
            save_overlay(image_id, img_bgr, predictions, overlay_dir)

    n_images = max(1, len(rows))
    precision = total_matched / max(1, total_predicted)
    recall = total_matched / max(1, total_expected)
    print("\nSummary")
    print(f"Avg count error: {total_abs_count_error / n_images:.2f} cards/image")
    print(f"Label precision: {precision:.3f}")
    print(f"Label recall:    {recall:.3f}")
    print(f"Invalid labels:  {total_invalid}")
    if args.save_overlays:
        print(f"Overlays saved to: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
