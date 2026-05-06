"""Inference helpers for loading a scene segmenter and predicting scene masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from uno_vision.image_ops import letterbox_pil_with_meta, unletterbox_mask
from uno_vision.paths import SEGMENTER_MODELS_DIR
from uno_vision.segmentation_scene.data import IMAGE_MEAN, IMAGE_STD, IMG_SIZE
from uno_vision.segmentation_scene.model import UNetSmall, assert_parameter_budget


DEFAULT_SCENE_SEGMENTER_NAME = "scene_segmenter_unet_small.pth"


def load_scene_segmenter(
    model_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[UNetSmall, torch.device, Path]:
    """Load a trained scene-segmentation model onto the requested device."""

    candidates = [
        model_path,
        SEGMENTER_MODELS_DIR / DEFAULT_SCENE_SEGMENTER_NAME,
    ]
    selected = next((Path(path) for path in candidates if path is not None and Path(path).is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            "No scene segmenter checkpoint found. "
            f"Expected: {SEGMENTER_MODELS_DIR / DEFAULT_SCENE_SEGMENTER_NAME}"
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    assert_parameter_budget(model)
    model.load_state_dict(torch.load(selected, map_location=device))
    model.eval()
    return model, device, selected


def segment_scene_image(
    image_bgr: np.ndarray,
    model: UNetSmall,
    device: torch.device,
    image_size: int = IMG_SIZE,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a foreground probability map and thresholded mask for one scene image."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_lb, meta = letterbox_pil_with_meta(
        image_pil,
        target_size=image_size,
        fill=255,
        interpolation=Image.BILINEAR,
    )
    image_t = TF.to_tensor(image_lb)
    image_t = TF.normalize(image_t, mean=IMAGE_MEAN, std=IMAGE_STD)
    image_t = image_t.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_t)
        prob_lb = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = unletterbox_mask(prob_lb, meta)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def segment_scene_image_path(
    image_path: str | Path,
    model_path: Path | None = None,
    threshold: float = 0.5,
    image_size: int = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Read a scene image, run segmentation, and return outputs with model path."""

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    model, device, selected_model_path = load_scene_segmenter(model_path=model_path)
    prob, mask = segment_scene_image(
        image_bgr=image_bgr,
        model=model,
        device=device,
        image_size=image_size,
        threshold=threshold,
    )
    return image_bgr, prob, mask, selected_model_path
