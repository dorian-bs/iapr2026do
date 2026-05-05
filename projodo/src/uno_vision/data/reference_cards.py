"""Extraction and visualization utilities for reference card images."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

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


def _rounded_rect_mask(height: int, width: int, corner_radius: int) -> np.ndarray:
    """Return a filled axis-aligned rounded rectangle mask of size (height, width)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask

    # Keep radius valid for tiny crops; if too small, fall back to full rectangle.
    max_radius = (min(height, width) - 1) // 2
    radius = min(max(corner_radius, 0), max_radius)
    if radius <= 0:
        mask[:, :] = 255
        return mask

    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, thickness=-1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, thickness=-1)

    cv2.circle(mask, (radius, radius), radius, 255, thickness=-1)
    cv2.circle(mask, (width - radius - 1, radius), radius, 255, thickness=-1)
    cv2.circle(mask, (radius, height - radius - 1), radius, 255, thickness=-1)
    cv2.circle(mask, (width - radius - 1, height - radius - 1), radius, 255, thickness=-1)
    return mask


def _rounded_rotated_rect_mask(
    shape: tuple[int, int],
    rect: tuple[tuple[float, float], tuple[float, float], float],
    rounded_corner_ratio: float,
    fixed_card_width: int | None = None,
    fixed_card_height: int | None = None,
) -> np.ndarray:
    """Return a full-canvas mask of a rounded rotated rectangle."""
    h, w = shape
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle -= 90

    if fixed_card_width is not None or fixed_card_height is not None:
        if fixed_card_width is None or fixed_card_height is None:
            raise ValueError(
                "Both fixed_card_width and fixed_card_height must be set together."
            )
        fw = max(1, int(round(fixed_card_width)))
        fh = max(1, int(round(fixed_card_height)))
        rw, rh = float(max(fw, fh)), float(min(fw, fh))

    rw_i, rh_i = int(np.ceil(rw)), int(np.ceil(rh))
    corner_radius = int(round(min(rh_i, rw_i) * rounded_corner_ratio))
    local_mask = _rounded_rect_mask(rh_i, rw_i, corner_radius)

    # Compose local mask in the rotated frame, then map back to original coordinates.
    rotated_canvas = np.zeros((h, w), dtype=np.uint8)
    x0 = int(round(cx - rw / 2))
    y0 = int(round(cy - rh / 2))
    x1 = x0 + rw_i
    y1 = y0 + rh_i

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(w, x1)
    dst_y1 = min(h, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return rotated_canvas

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    rotated_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = local_mask[src_y0:src_y1, src_x0:src_x1]

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M_inv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(
        rotated_canvas,
        M_inv,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def _reference_binary_mask(
    gray: np.ndarray,
    adaptive_block_size: int = 11,
    adaptive_c: int = 4,
    pre_blur_size: int = 3,
    dilate_size_1: int = 17,
    dilate_size_2: int = 17,
    post_blur_size: int = 33,
    min_area_abs: int = 100000,
) -> np.ndarray:
    """Build a foreground mask using adaptive threshold + skeletonize pipeline."""
    adaptive_image = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=_odd(adaptive_block_size),
        C=adaptive_c,
    )
    median_filtered = cv2.medianBlur(adaptive_image, _odd(pre_blur_size))
    binary = median_filtered.copy()
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)
    kernel1 = np.ones((dilate_size_1, dilate_size_1), np.uint8)
    dilated1 = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1)
    skeleton = skeletonize(dilated1 > 0).astype(np.uint8) * 255
    kernel2 = np.ones((dilate_size_2, dilate_size_2), np.uint8)
    dilated2 = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel2)
    smoothed = cv2.medianBlur(dilated2, _odd(post_blur_size))
    no_small = smoothed.copy()
    contours, _ = cv2.findContours(no_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area_abs:
            cv2.drawContours(no_small, [c], -1, 0, cv2.FILLED)
    return no_small


def _mask_pipeline_steps(
    gray: np.ndarray,
    adaptive_block_size: int = 11,
    adaptive_c: int = 4,
    pre_blur_size: int = 3,
    dilate_size_1: int = 17,
    dilate_size_2: int = 17,
    post_blur_size: int = 33,
    min_area_abs: int = 100000,
    min_ar: float = 0.3,
    max_ar: float = 2.0,
) -> dict[str, np.ndarray]:
    """Return each intermediate mask image keyed by step label, for debugging."""
    # Step 1: Adaptive threshold
    adaptive_image = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=_odd(adaptive_block_size),
        C=adaptive_c,
    )

    # Step 2: Median filter right after adaptive threshold to remove noise
    median_filtered = cv2.medianBlur(adaptive_image, _odd(pre_blur_size))

    # Step 3: Invert if background is white, then dilate
    binary = median_filtered.copy()
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)
    kernel1 = np.ones((dilate_size_1, dilate_size_1), np.uint8)
    dilated1 = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1)

    # Step 4: Skeletonize
    skeleton = skeletonize(dilated1 > 0).astype(np.uint8) * 255

    # Step 5: Dilate skeleton to thicken it
    kernel2 = np.ones((dilate_size_2, dilate_size_2), np.uint8)
    dilated2 = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel2)

    # Step 6: Smooth
    smoothed = cv2.medianBlur(dilated2, _odd(post_blur_size))

    # Step 7: Remove small contours
    no_small = smoothed.copy()
    contours, _ = cv2.findContours(no_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area_abs:
            cv2.drawContours(no_small, [c], -1, 0, cv2.FILLED)

    # Step 8: AR filter visualization
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_small, connectivity=8)
    rng = np.random.default_rng(42)
    colors = np.vstack([[0, 0, 0], rng.integers(80, 220, (max(n_labels - 1, 0), 3))])
    labeled = colors[labels].astype(np.uint8)
    for lbl in range(1, n_labels):
        w, h = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
        if not (min_ar <= w / max(h, 1) <= max_ar):
            labeled[labels == lbl] = 40

    return {
        "1 · Adaptive threshold [block/C]": adaptive_image,
        "2 · Median filter [pre_blur_size]": median_filtered,
        "3 · Invert + Dilate [dilate_size_1]": dilated1,
        "4 · Skeletonize": skeleton,
        "5 · Dilate skeleton [dilate_size_2]": dilated2,
        "6 · Smooth [post_blur_size]": smoothed,
        "7 · Small contours removed [min_area_abs]": no_small,
        "8 · AR filter [min_ar / max_ar]": labeled,
    }


def extract_reference_card_assets(
    image_name: str,
    num_components: int,
    image_dir: Path = REFERENCE_IMAGES_DIR,
    output_root: Path = REFERENCE_CARDS_DIR,
    # mask hyperparameters (forwarded to _reference_binary_mask)
    adaptive_block_size: int = 11,
    adaptive_c: int = 4,
    pre_blur_size: int = 3,
    dilate_size_1: int = 17,
    dilate_size_2: int = 17,
    post_blur_size: int = 33,
    min_area_abs: int = 100000,
    # extraction hyperparameters
    min_ar: float = 0.3,
    max_ar: float = 2.0,
    fixed_card_width: int | None = None,
    fixed_card_height: int | None = None,
    rounded_corner_ratio: float = 0.08,
) -> ReferenceExtractionResult:
    """Extract card crops, component masks, and fitted rounded-rectangle masks."""

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

    masks = _reference_binary_mask(
        gray, adaptive_block_size, adaptive_c, pre_blur_size,
        dilate_size_1, dilate_size_2, post_blur_size, min_area_abs,
    )

    use_fixed_size = fixed_card_width is not None or fixed_card_height is not None
    if use_fixed_size and (fixed_card_width is None or fixed_card_height is None):
        raise ValueError(
            "Both fixed_card_width and fixed_card_height must be set together."
        )
    fixed_long: float | None = None
    fixed_short: float | None = None
    if use_fixed_size:
        fw = max(1, int(round(fixed_card_width)))
        fh = max(1, int(round(fixed_card_height)))
        fixed_long = float(max(fw, fh))
        fixed_short = float(min(fw, fh))

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
            # Fit an oriented rectangle, then build a rounded rectangle in rectified crop space.
            contour_points = np.vstack(contours_comp)
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour_points)
            if rw < rh:
                rw, rh = rh, rw
                angle -= 90
            if fixed_long is not None and fixed_short is not None:
                rw, rh = fixed_long, fixed_short
            rw_i, rh_i = int(np.ceil(rw)), int(np.ceil(rh))
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            warped_img = cv2.warpAffine(
                img_color, M, (img_color.shape[1], img_color.shape[0]),
                borderValue=(255, 255, 255),
            )
            x0 = max(0, int(round(cx - rw / 2)))
            y0 = max(0, int(round(cy - rh / 2)))
            x1 = min(warped_img.shape[1], x0 + rw_i)
            y1 = min(warped_img.shape[0], y0 + rh_i)
            crop_color = warped_img[y0:y1, x0:x1]

            crop_h, crop_w = crop_color.shape[:2]
            corner_radius = int(round(min(crop_h, crop_w) * rounded_corner_ratio))
            crop_mask_tight = _rounded_rect_mask(crop_h, crop_w, corner_radius)
        else:
            x, y, bw, bh = cv2.boundingRect(component_mask)
            if fixed_long is not None and fixed_short is not None:
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                rw_i = int(fixed_long)
                rh_i = int(fixed_short)
                x0 = max(0, int(round(cx - rw_i / 2)))
                y0 = max(0, int(round(cy - rh_i / 2)))
                x1 = min(img_color.shape[1], x0 + rw_i)
                y1 = min(img_color.shape[0], y0 + rh_i)
                crop_color = img_color[y0:y1, x0:x1]
            else:
                crop_color = img_color[y:y + bh, x:x + bw]

            crop_h, crop_w = crop_color.shape[:2]
            corner_radius = int(round(min(crop_h, crop_w) * rounded_corner_ratio))
            crop_mask_tight = _rounded_rect_mask(crop_h, crop_w, corner_radius)

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


def plot_pipeline_steps(
    color_rgb: np.ndarray,
    pipeline_steps: dict[str, np.ndarray],
    title: str = "",
) -> None:
    """Display mask-pipeline debug panels in an adaptive 3-column grid."""
    import matplotlib.pyplot as plt

    panels = [("0 · Original", color_rgb), *pipeline_steps.items()]
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


def plot_crop_preview(
    result: ReferenceExtractionResult,
    title: str = "",
    n: int = 4,
    rounded_corner_ratio: float = 0.08,
    fixed_card_width: int | None = None,
    fixed_card_height: int | None = None,
) -> None:
    """Display two composite views then a per-card grid for the first n cards."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    if not result.components:
        raise RuntimeError("No reference-card assets were generated.")

    _, color_rgb = load_reference_image(result.image_name)
    h, w = color_rgb.shape[:2]
    n_cards = len(result.components)

    # Build panel 1: unclosed selected component masks, color-coded per card.
    final_components_colored = np.zeros((h, w, 3), dtype=np.uint8)
    # Build panel 2: reference image with semi-transparent fitted rounded-rectangle fills
    fill_layer = np.zeros_like(color_rgb)
    closed_union = np.zeros((h, w), dtype=bool)
    rounded_entries: list[tuple[np.ndarray, np.ndarray]] = []
    for i, comp_path in enumerate(result.components):
        comp = cv2.imread(str(comp_path), cv2.IMREAD_GRAYSCALE)
        if comp is None:
            continue
        rgb = (np.array(hsv_to_rgb([i / max(n_cards, 1), 0.85, 0.95])) * 255).astype(np.uint8)
        final_components_colored[comp > 0] = rgb
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect = cv2.minAreaRect(np.vstack(contours))

        closed_mask = _rounded_rotated_rect_mask(
            comp.shape,
            rect,
            rounded_corner_ratio,
            fixed_card_width=fixed_card_width,
            fixed_card_height=fixed_card_height,
        )
        closed_region = closed_mask > 0
        fill_layer[closed_region] = rgb
        closed_union |= closed_region
        closed_contours, _ = cv2.findContours(
            closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

    # # Per-card grid for the first n cards
    # rows = []
    # for component_path, mask_path, crop_path in zip(
    #     result.components[:n], result.masks[:n], result.crops[:n]
    # ):
    #     component = cv2.imread(str(component_path), cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    #     crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    #     if component is None or mask is None or crop_bgr is None:
    #         raise FileNotFoundError(
    #             f"Could not read one of: {component_path.name}, {mask_path.name}, {crop_path.name}"
    #         )
    #     crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    #     mask_rs = (
    #         cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    #         if mask.shape != crop.shape[:2]
    #         else mask
    #     )
    #     mask_bin = np.where(mask_rs > 0, 255, 0).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     crop_outlined = crop.copy()
    #     if contours:
    #         cv2.drawContours(crop_outlined, contours, -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)
    #     rows.append((component, mask, crop_outlined, component_path.name, mask_path.name, crop_path.name))

    # fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3 * len(rows)))
    # axes = np.atleast_2d(axes)
    # for ax in axes.flat:
    #     ax.axis("off")
    # fig.canvas.draw()
    # renderer = fig.canvas.get_renderer()
    # for i, (comp, msk, crop_out, comp_name, msk_name, crop_name) in enumerate(rows):
    #     axes[i, 0].imshow(_fit_image_for_axis(comp, axes[i, 0], renderer), cmap="gray")
    #     axes[i, 0].set_title(comp_name)
    #     axes[i, 1].imshow(_fit_image_for_axis(msk, axes[i, 1], renderer), cmap="gray")
    #     axes[i, 1].set_title(msk_name)
    #     axes[i, 2].imshow(_fit_image_for_axis(crop_out, axes[i, 2], renderer))
    #     axes[i, 2].set_title(f"{crop_name} + mask outline")
    # plt.suptitle(f"{title} · per-card preview (first {n})", y=1.01)
    # plt.tight_layout()
    # plt.show()
