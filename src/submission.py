"""Submission CSV writing and schema validation (R5).

The Kaggle challenge enforces a strict format. We validate before writing so a
typo cannot silently waste a submission slot.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from src.inference import GameState


CSV_COLUMNS = (
    "image_id",
    "center_card",
    "active_player",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
)

_COLORS = {"r", "g", "b", "y"}
_VALUES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw_2"}
_SPECIAL = {"wild", "draw_4"}
_ACTIVE_PLAYER_VALUES = {"p1", "p2", "p3", "p4", "EMPTY"}
_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_card_token(token: str) -> None:
    if token in _SPECIAL:
        return
    if "_" not in token:
        raise ValueError(f"Card token '{token}' is not of the form <color>_<value>.")
    color, _, value = token.partition("_")
    if color not in _COLORS:
        raise ValueError(f"Card token '{token}': color '{color}' not in {_COLORS}.")
    if value not in _VALUES:
        raise ValueError(f"Card token '{token}': value '{value}' not in {_VALUES}.")


def _validate_card_field(field_name: str, value: str) -> None:
    if value == "EMPTY":
        return
    if not value:
        raise ValueError(f"Field {field_name} is blank; use 'EMPTY' instead.")
    for token in value.split(";"):
        token = token.strip()
        if not token:
            raise ValueError(f"Field {field_name} has an empty card token.")
        _validate_card_token(token)


def validate_row(row: dict[str, str]) -> None:
    """Raise ValueError if `row` does not conform to the Kaggle schema."""
    missing = [c for c in CSV_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Row is missing columns: {missing}")
    if not _IMAGE_ID_RE.match(str(row["image_id"])):
        raise ValueError(f"Bad image_id: {row['image_id']!r}")
    if row["active_player"] not in _ACTIVE_PLAYER_VALUES:
        raise ValueError(f"active_player must be in {_ACTIVE_PLAYER_VALUES}, got {row['active_player']!r}")
    if row["center_card"] != "EMPTY":
        _validate_card_token(row["center_card"])
    for player_field in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
        _validate_card_field(player_field, row[player_field])


def write_submission(
    rows: Iterable[GameState | dict[str, str]],
    output_path: Path,
    *,
    validate: bool = True,
) -> Path:
    """Write a submission CSV. Validates every row by default."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for entry in rows:
            row = entry.as_submission_row() if isinstance(entry, GameState) else dict(entry)
            if validate:
                validate_row(row)
            writer.writerow(row)
    return output_path
