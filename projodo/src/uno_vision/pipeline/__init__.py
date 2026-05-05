"""High-level prediction pipeline exports for scene-to-card inference."""

from uno_vision.pipeline.cards import CardRegionPrediction, predict_cards_in_image

__all__ = ["CardRegionPrediction", "predict_cards_in_image"]
