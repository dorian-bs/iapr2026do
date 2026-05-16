"""Plot helpers for the augmented-data generation pipeline."""
from __future__ import annotations

import json
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_card_preview(state: dict[str, Any], one_per_card: bool = True) -> None:
    """Show augmented single-card crops.

    By default, displays one sample for each unique card label so label
    assignment can be verified quickly.
    """
    aug_rows = state["aug_rows"]
    if not aug_rows:
        print("No augmented cards were generated.")
        return

    paths = state["paths"]
    cfg = state["cfg"]
    rng = state["card_preview_rng"]
    if one_per_card:
        label_to_indices: dict[str, list[int]] = {}
        for idx, row in enumerate(aug_rows):
            label = str(row.get("card", "")).strip()
            if not label:
                continue
            label_to_indices.setdefault(label, []).append(idx)

        labels = sorted(label_to_indices.keys())
        if not labels:
            print("No labelled augmented cards were found in aug_rows.")
            return

        preview_indices = [
            int(rng.choice(label_to_indices[label]))
            for label in labels
        ]
        print(f"Previewing one sample per label: {len(labels)} labels")
    else:
        preview_count = min(int(cfg.preview_card_sample_count), len(aug_rows))
        if preview_count == 0:
            print("Preview disabled (preview_card_sample_count == 0).")
            return
        preview_indices = rng.choice(len(aug_rows), size=preview_count, replace=False).tolist()
        print(f"Previewing random subset: {len(preview_indices)} samples")

    cols = 6
    rows_per_page = 6
    page_size = cols * rows_per_page

    for page_start in range(0, len(preview_indices), page_size):
        page_indices = preview_indices[page_start: page_start + page_size]
        rows = int(np.ceil(len(page_indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, index in zip(axes, page_indices):
            row = aug_rows[int(index)]
            image_path = None
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = paths.aug_cards_dir / f"{row['image_id']}{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break

            if image_path is None:
                ax.text(0.5, 0.5, "missing image", ha="center", va="center")
                ax.set_title(str(row.get("card", "?")), fontsize=8)
                ax.axis("off")
                continue

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                ax.text(0.5, 0.5, "read error", ha="center", va="center")
                ax.set_title(str(row.get("card", "?")), fontsize=8)
                ax.axis("off")
                continue

            ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(str(row.get("card", "?")), fontsize=8)
            ax.axis("off")

        for ax in axes[len(page_indices):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def plot_scene_preview(state: dict[str, Any]) -> None:
    """Render and display one freshly composed scene plus its mask and overlay."""
    preview_scene_bgr = state["preview_scene_bgr"]
    preview_mask = state["preview_mask"]
    preview_metadata = state["preview_metadata"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].imshow(cv2.cvtColor(preview_scene_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Synthetic scene")
    axes[0].axis("off")

    axes[1].imshow(preview_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Separated card mask")
    axes[1].axis("off")

    overlay = cv2.cvtColor(preview_scene_bgr, cv2.COLOR_BGR2RGB).copy()
    overlay[preview_mask > 0] = (0.65 * overlay[preview_mask > 0] + 0.35 * np.array([255, 40, 40])).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Mask overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(json.dumps({
        "style": preview_metadata["style"],
        "active_player": preview_metadata["active_player"],
        "player_card_counts": preview_metadata["player_card_counts"],
        "visible_cards": len(preview_metadata["cards"]),
        "token_source": preview_metadata["token_source"],
    }, indent=2))


def plot_saved_scenes(state: dict[str, Any]) -> None:
    """Quick visual check of saved scenes/masks after the generation pass."""
    paths = state["paths"]
    cfg = state["cfg"]
    if not paths.scenes_labels_path.exists():
        print("Run scene generation first to create saved scenes and masks.")
        return

    with paths.scenes_labels_path.open("r", encoding="utf-8") as labels_file:
        saved_metadata = json.load(labels_file)

    sample_count = min(int(cfg.preview_saved_sample_count), len(saved_metadata))
    if sample_count == 0:
        print("Preview disabled (preview_saved_sample_count == 0).")
        return

    fig, axes = plt.subplots(sample_count, 3, figsize=(18, 4.5 * sample_count))
    axes = np.atleast_2d(axes)

    for row_index, metadata in enumerate(saved_metadata[:sample_count]):
        image_path = paths.project_root / metadata["image_path"]
        mask_path = paths.project_root / metadata["mask_path"]
        scene_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        scene_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        overlay_rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB).copy()
        overlay_rgb[scene_mask > 0] = (0.65 * overlay_rgb[scene_mask > 0] + 0.35 * np.array([255, 40, 40])).astype(np.uint8)

        axes[row_index, 0].imshow(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title(f"{metadata['scene']} scene")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(scene_mask, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title("separated mask")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(overlay_rgb)
        axes[row_index, 2].set_title(f"{metadata['style']} / active {metadata['active_player']}")
        axes[row_index, 2].axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Previewed {sample_count} saved samples from {paths.scenes_labels_path}")
