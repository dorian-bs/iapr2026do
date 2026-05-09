"""Pipeline modules for the reference card extraction notebook."""

from .pipeline import (
    ReferenceCardsPipelineConfig,
    ReferenceExtractionResult,
    initialize_reference_cards_pipeline,
    run_extraction,
    run_preview,
)
from .manual_pipeline import (
    ManualReferenceConfig,
    ManualReferenceImageResult,
    initialize_manual_reference_pipeline,
    run_manual_reference_split,
    summarize_manual_reference_split,
)
from .viz import plot_crop_previews, plot_pipeline_steps

__all__ = [
    "ReferenceCardsPipelineConfig",
    "ReferenceExtractionResult",
    "ManualReferenceConfig",
    "ManualReferenceImageResult",
    "initialize_manual_reference_pipeline",
    "initialize_reference_cards_pipeline",
    "plot_crop_previews",
    "plot_pipeline_steps",
    "run_manual_reference_split",
    "run_extraction",
    "run_preview",
    "summarize_manual_reference_split",
]
