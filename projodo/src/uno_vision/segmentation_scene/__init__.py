"""Scene-segmentation package for training, inference, and diagnostics."""

from uno_vision.segmentation_scene.data import IMG_SIZE, SceneSegDataset, collect_scene_pairs
from uno_vision.segmentation_scene.inference import load_scene_segmenter, segment_scene_image, segment_scene_image_path
from uno_vision.segmentation_scene.model import UNetSmall
from uno_vision.segmentation_scene.training import SceneSegmentationTrainingHistory, train_scene_segmenter

__all__ = [
    "IMG_SIZE",
    "SceneSegDataset",
    "SceneSegmentationTrainingHistory",
    "UNetSmall",
    "collect_scene_pairs",
    "load_scene_segmenter",
    "segment_scene_image",
    "segment_scene_image_path",
    "train_scene_segmenter",
]
