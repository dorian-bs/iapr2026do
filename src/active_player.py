"""Active player detection from the "active" token next to a player.

STATUS: STUB — owned by another teammate, not yet implemented.

Per the challenge spec, the active player is the one with a token (a small
disk/marker) placed near them. The expected detection pipeline is classical
image processing:

    1. Color-segment the token in HSV space (the token is a saturated color
       distinct from cards and table).
    2. Filter blob candidates by area and circularity.
    3. Take the centroid of the strongest candidate and map it to the closest
       player region using the same geometry rules as `assign_region`
       (p1=bottom, p2=right, p3=top, p4=left).

Until the teammate's implementation lands, this stub returns "EMPTY", which is
the valid sentinel per R5 (the submission validator will accept it).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.inference import CardPrediction


# Placeholder return value while the detector is under development.
_UNKNOWN_ACTIVE_PLAYER = "EMPTY"


def detect_active_player(
    img_bgr: np.ndarray,
    card_predictions: "list[CardPrediction]",
) -> str:
    """Return one of {"p1", "p2", "p3", "p4", "EMPTY"}.

    Parameters
    ----------
    img_bgr : np.ndarray
        Original BGR scene image. The token detector operates here.
    card_predictions : list[CardPrediction]
        Already-detected cards. Useful to mask out card pixels before searching
        for the token, so card-color blobs are not confused with the token.

    Notes
    -----
    Replace this stub with the teammate's implementation. The function
    signature is part of the inference contract and should be preserved.
    """
    # TODO(teammate): plug the token-detection pipeline here.
    _ = img_bgr, card_predictions
    return _UNKNOWN_ACTIVE_PLAYER
