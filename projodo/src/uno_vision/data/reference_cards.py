"""Extraction and visualization utilities for reference card images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from uno_vision.paths import REFERENCE_CARDS_DIR, REFERENCE_IMAGES_DIR


@dataclass(frozen=True)
class ReferenceExtractionResult:
    """Paths produced when one reference image is decomposed into card assets."""

    image_name: str
    output_dir: Path
    crops: list[Path]
    masks: list[Path]
    components: list[Path]


def load_reference_image(
    image_name: str,
    image_dir: Path = REFERENCE_IMAGES_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray, color_rgb) arrays for a reference image."""
    image_path = image_dir / f"{image_name}.jpg"
    gray = np.array(Image.open(image_path).convert("L"))
    color_bgr = cv2.imread(str(image_path))
    if color_bgr is None:
        raise FileNotFoundError(f"Cannot read reference image: {image_path}")
    return gray, cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def _odd(value: float) -> int:
    """Return value rounded up to the nearest odd integer >= 3."""
    v = max(3, int(value))
    return v if v % 2 == 1 else v + 1


def _reference_binary_mask(
    gray: np.ndarray,
    canny_blur_size: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    blob_close_size: int = 40,
    blur_after: int = 33,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a coarse foreground mask from Canny edges in a reference image."""

    scale = max(gray.shape) / 4000.0
    k_blur = _odd(canny_blur_size * scale)
    blurred = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    k_close = max(3, int(blob_close_size * scale))

    # Closing turns broken card outlines into connected blobs for component labeling.
    binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    binary = cv2.medianBlur(binary, _odd(blur_after * scale))
    return edges, binary


# def _mask_pipeline_steps(
#     gray: np.ndarray,
#     canny_blur_size: int = 5,
#     canny_low: int = 30,
#     canny_high: int = 100,
#     blob_close_size: int = 40,
#     blur_after: int = 33,
#     min_area_abs: int = 2500,
#     min_area_frac: float = 0.002,
#     close_size: int = 21,
#     open_size: int = 7,
#     min_ar: float = 0.3,
#     max_ar: float = 2.0,
# ) -> dict[str, np.ndarray]:
#     """Return each intermediate mask image keyed by step label, for debugging."""
#     scale = max(gray.shape) / 4000.0
#     img_area = gray.shape[0] * gray.shape[1]

#     k_blur = _odd(canny_blur_size * scale)
#     blurred = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
#     edges = cv2.Canny(blurred, canny_low, canny_high)
#     k_close = max(3, int(blob_close_size * scale))
#     closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
#     binary = cv2.medianBlur(closed_edges, _odd(blur_after * scale))

#     no_small = binary.copy()
#     min_area = max(min_area_abs * scale ** 2, min_area_frac * img_area)
#     contours, _ = cv2.findContours(no_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         if cv2.contourArea(c) < min_area:
#             cv2.drawContours(no_small, [c], -1, 0, cv2.FILLED)

#     closed2 = cv2.morphologyEx(no_small, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
#     opened = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))

#     n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
#     rng = np.random.default_rng(42)
#     colors = np.vstack([[0, 0, 0], rng.integers(80, 220, (max(n_labels - 1, 0), 3))])
#     labeled = colors[labels].astype(np.uint8)
#     for lbl in range(1, n_labels):
#         w, h = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
#         if not (min_ar <= w / max(h, 1) <= max_ar):
#             labeled[labels == lbl] = 40  # dim rejected components

#     return {
#         "1 · GaussianBlur [canny_blur_size]": blurred,
#         "2 · Canny [canny_low / canny_high]": edges,
#         "3 · Morph CLOSE [blob_close_size]": closed_edges,
#         "4 · Median blur [blur_after]": binary,
#         "5 · Small contours removed [min_area_*]": no_small,
#         "6 · Morph CLOSE [close_size]": closed2,
#         "7 · Morph OPEN [open_size]": opened,
#         "8 · AR filter [min_ar / max_ar]": labeled,
#     }


def extract_reference_card_assets(
    image_name: str,
    num_components: int,
    image_dir: Path = REFERENCE_IMAGES_DIR,
    output_root: Path = REFERENCE_CARDS_DIR,
    # mask hyperparameters (forwarded to _reference_binary_mask)
    canny_blur_size: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    blob_close_size: int = 40,
    blur_after: int = 33,
    # extraction hyperparameters
    min_area_abs: int = 2500,
    min_area_frac: float = 0.002,
    close_size: int = 21,
    open_size: int = 7,
    min_ar: float = 0.3,
    max_ar: float = 2.0,
) -> ReferenceExtractionResult:
    """Extract card crops, component masks, and filled masks from one reference image."""

    image_path = image_dir / f"{image_name}.jpg"
    gray = np.array(Image.open(image_path).convert("L"))
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise FileNotFoundError(f"Cannot read reference image: {image_path}")

    out_dir = output_root / image_name
    components_dir = out_dir / "components"
    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    for directory in (components_dir, masks_dir, crops_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for f in directory.glob("*.jpg"):
            f.unlink()

    edges, binary = _reference_binary_mask(
        gray, canny_blur_size, canny_low, canny_high, blob_close_size, blur_after
    )

    scale = max(gray.shape) / 4000.0
    img_area = gray.shape[0] * gray.shape[1]
    min_area = max(min_area_abs * scale ** 2, min_area_frac * img_area)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary, [contour], -1, 0, thickness=cv2.FILLED)

    # Closing fills card interiors; opening removes residual speckles before labeling.
    masks = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masks, connectivity=8)
    if n_labels <= 1:
        return ReferenceExtractionResult(image_name, out_dir, [], [], [])

    valid: list[tuple[int, int]] = []
    for i in range(n_labels - 1):
        label_idx = i + 1
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        ar = w / max(h, 1)
        if min_ar <= ar <= max_ar:
            valid.append((int(stats[label_idx, cv2.CC_STAT_AREA]), label_idx))
    valid.sort(reverse=True)
    selected_idx = [lbl for _, lbl in valid[: min(num_components, len(valid))]]

    crops: list[Path] = []
    closed_masks: list[Path] = []
    components: list[Path] = []
    for plot_i, label_idx in enumerate(selected_idx):
        component_mask = np.zeros_like(masks)
        component_mask[labels == label_idx] = 255
        component_path = components_dir / f"component_{plot_i}.jpg"
        cv2.imwrite(str(component_path), component_mask)
        components.append(component_path)

        contours_comp, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_comp:
            # Convex hulls produce filled masks; minAreaRect gives the card's true orientation.
            hull = cv2.convexHull(np.vstack(contours_comp))
            closed_mask = np.zeros_like(component_mask)
            cv2.fillPoly(closed_mask, [hull], 255)

            (cx, cy), (rw, rh), angle = cv2.minAreaRect(hull)
            if rw < rh:
                rw, rh = rh, rw
                angle -= 90
            rw_i, rh_i = int(np.ceil(rw)), int(np.ceil(rh))
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            warped_img = cv2.warpAffine(
                img_color, M, (img_color.shape[1], img_color.shape[0]),
                borderValue=(255, 255, 255),
            )
            warped_mask = cv2.warpAffine(
                closed_mask, M, (closed_mask.shape[1], closed_mask.shape[0]),
                flags=cv2.INTER_NEAREST, borderValue=0,
            )
            x0 = max(0, int(round(cx - rw / 2)))
            y0 = max(0, int(round(cy - rh / 2)))
            x1 = min(warped_img.shape[1], x0 + rw_i)
            y1 = min(warped_img.shape[0], y0 + rh_i)
            crop_color = warped_img[y0:y1, x0:x1]
            crop_mask_tight = warped_mask[y0:y1, x0:x1]
        else:
            closed_mask = component_mask.copy()
            x, y, w, h = cv2.boundingRect(closed_mask)
            crop_color = img_color[y:y + h, x:x + w]
            crop_mask_tight = closed_mask[y:y + h, x:x + w]

        # Rotate landscape crops 90° so all saved cards are portrait (height >= width).
        if crop_color.shape[1] > crop_color.shape[0]:
            crop_color = cv2.rotate(crop_color, cv2.ROTATE_90_CLOCKWISE)
            crop_mask_tight = cv2.rotate(crop_mask_tight, cv2.ROTATE_90_CLOCKWISE)

        # Mask and crop are both tight rotated crops — the same shape — for later alignment.
        closed_path = masks_dir / f"closed_component_{plot_i}.jpg"
        cv2.imwrite(str(closed_path), crop_mask_tight)
        closed_masks.append(closed_path)

        crop_path = crops_dir / f"crop_{plot_i}.jpg"
        cv2.imwrite(str(crop_path), crop_color)
        crops.append(crop_path)

    return ReferenceExtractionResult(image_name, out_dir, crops, closed_masks, components)


# ── Visualization helpers ──────────────────────────────────────────────────────

def _fit_image_for_axis(img: np.ndarray, ax, renderer) -> np.ndarray:
    """Downsample img to the display pixel area of ax (never upscales)."""
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


# def plot_pipeline_steps(
#     color_rgb: np.ndarray,
#     pipeline_steps: dict[str, np.ndarray],
#     title: str = "",
# ) -> None:
#     """Display the 3×3 mask-pipeline debug grid."""
#     import matplotlib.pyplot as plt

#     panels = [("0 · Original", color_rgb), *pipeline_steps.items()]
#     fig, axes = plt.subplots(3, 3, figsize=(15, 11))
#     for ax in axes.flat:
#         ax.axis("off")
#     fig.canvas.draw()
#     renderer = fig.canvas.get_renderer()
#     for ax, (lbl, img) in zip(axes.flat, panels):
#         display = _fit_image_for_axis(img, ax, renderer)
#         ax.imshow(display, cmap="gray" if display.ndim == 2 else None)
#         ax.set_title(lbl, fontsize=9)
#     plt.suptitle(title, fontsize=12)
#     plt.tight_layout()
#     plt.show()


def plot_crop_preview(
    result: ReferenceExtractionResult,
    title: str = "",
    n: int = 4,
) -> None:
    """Display component / mask / mask-outlined-crop grid for the first n cards."""
    import matplotlib.pyplot as plt

    rows = []
    for component_path, mask_path, crop_path in zip(
        result.components[:n], result.masks[:n], result.crops[:n]
    ):
        component = cv2.imread(str(component_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if component is None or mask is None or crop_bgr is None:
            raise FileNotFoundError(
                f"Could not read one of: {component_path.name}, {mask_path.name}, {crop_path.name}"
            )
        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mask_rs = (
            cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            if mask.shape != crop.shape[:2]
            else mask
        )
        mask_bin = np.where(mask_rs > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_outlined = crop.copy()
        if contours:
            cv2.drawContours(crop_outlined, contours, -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        rows.append((component, mask, crop_outlined, component_path.name, mask_path.name, crop_path.name))

    if not rows:
        raise RuntimeError("No reference-card assets were generated.")

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, (comp, msk, crop_out, comp_name, msk_name, crop_name) in enumerate(rows):
        axes[i, 0].imshow(_fit_image_for_axis(comp, axes[i, 0], renderer), cmap="gray")
        axes[i, 0].set_title(comp_name)
        axes[i, 1].imshow(_fit_image_for_axis(msk, axes[i, 1], renderer), cmap="gray")
        axes[i, 1].set_title(msk_name)
        axes[i, 2].imshow(_fit_image_for_axis(crop_out, axes[i, 2], renderer))
        axes[i, 2].set_title(f"{crop_name} + mask outline")
    plt.suptitle(f"{title} · mask outline overlay", y=1.01)
    plt.tight_layout()
    plt.show()
