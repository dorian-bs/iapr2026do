"""Plot helpers for the reference card extraction notebook."""
from __future__ import annotations

from math import ceil
from typing import Any

import cv2
import numpy as np

from .pipeline import (
    ReferenceExtractionResult,
    _rounded_rotated_rect_mask,
    load_reference_image,
)


def _fit_image_for_axis(img: np.ndarray, ax, renderer) -> np.ndarray:
    """Downsample to plot size (never upsample)."""
    bbox = ax.get_window_extent(renderer=renderer)
    h, w = img.shape[:2]
    scale = min(max(1, int(bbox.width)) / w, max(1, int(bbox.height)) / h, 1.0)
    if scale < 1.0:
        return cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def plot_pipeline_steps(state: dict[str, Any]) -> None:
    """Original image + all mask pipeline steps."""
    import matplotlib.pyplot as plt

    preview = state.get("preview")
    if preview is None:
        raise RuntimeError("run_preview(state) must be called before plot_pipeline_steps.")

    color_rgb = preview["color_rgb"]
    pipeline_steps = preview["pipeline_steps"]
    title = preview["image_name"]

    panels = [("0 - Original", color_rgb), *pipeline_steps.items()]
    n_cols = 3
    n_rows = max(1, ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (lbl, img) in zip(axes, panels):
        display = _fit_image_for_axis(img, ax, renderer)
        ax.imshow(display, cmap="gray" if display.ndim == 2 else None)
        ax.set_title(lbl, fontsize=9)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_single_crop_preview(
    result: ReferenceExtractionResult,
    image_dir,
    rounded_corner_ratio: float,
    fixed_card_width: int | None,
    fixed_card_height: int | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    _, color_rgb = load_reference_image(result.image_name, image_dir)
    h, w = color_rgb.shape[:2]
    n_cards = max(len(result.components), 1)

    final_components_colored = np.zeros((h, w, 3), dtype=np.uint8)
    fill_layer = np.zeros_like(color_rgb)
    closed_union = np.zeros((h, w), dtype=bool)
    rounded_entries: list[tuple[np.ndarray, np.ndarray]] = []

    for i, comp_path in enumerate(result.components):
        comp = cv2.imread(str(comp_path), cv2.IMREAD_GRAYSCALE)
        if comp is None:
            continue

        rgb = (np.array(hsv_to_rgb([i / n_cards, 0.85, 0.95])) * 255).astype(np.uint8)
        final_components_colored[comp > 0] = rgb

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        closed_mask = _rounded_rotated_rect_mask(
            comp.shape,
            cv2.minAreaRect(np.vstack(contours)),
            rounded_corner_ratio,
            fixed_card_width=fixed_card_width,
            fixed_card_height=fixed_card_height,
        )
        closed_region = closed_mask > 0
        fill_layer[closed_region] = rgb
        closed_union |= closed_region

        closed_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if closed_contours:
            rounded_entries.append((max(closed_contours, key=cv2.contourArea), rgb))

    overlay = color_rgb.copy()
    alpha = 0.1
    overlay[closed_union] = (
        (1.0 - alpha) * color_rgb[closed_union] + alpha * fill_layer[closed_union]
    ).astype(np.uint8)
    for contour, rgb in rounded_entries:
        cv2.drawContours(overlay, [contour], -1, rgb.tolist(), 8, lineType=cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes[0].imshow(_fit_image_for_axis(final_components_colored, axes[0], renderer))
    axes[0].set_title("Final selected component mask (colored, unclosed)")
    axes[0].axis("off")

    axes[1].imshow(_fit_image_for_axis(overlay, axes[1], renderer))
    axes[1].set_title("Reference image + fitted rounded-rectangle overlay")
    axes[1].axis("off")

    plt.suptitle(result.image_name, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_crop_previews(state: dict[str, Any]) -> None:
    """Colored components + rounded overlay preview for every extracted image."""
    cfg = state["config"]
    image_dir = state["reference_images_dir"]
    results = state.get("results") or {}
    for result in results.values():
        _plot_single_crop_preview(
            result,
            image_dir=image_dir,
            rounded_corner_ratio=cfg.rounded_corner_ratio,
            fixed_card_width=cfg.fixed_card_width,
            fixed_card_height=cfg.fixed_card_height,
        )
