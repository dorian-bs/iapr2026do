"""Manual-reference splitter for card crops and labels.

This pipeline consumes user-drawn reference masks (black background / white cards)
and exports:
  * Standard crops + binary masks for compatibility with existing loaders.
  * Transparent PNG crops (RGBA) for segmenter dataset building blocks.
  * A synchronized `reference_manual.csv` where existing labels are preserved and
    newly discovered crops are appended with empty labels.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class ManualReferenceConfig:
    """Config for splitting manually-masked reference images."""

    # Process all manual masks when None.
    image_names: list[str] | None = None
    preview_image_name: str | None = "L1000765"

    # Manual mask postprocessing.
    mask_threshold: int = 127
    min_component_area_abs: int = 1500
    row_group_tolerance_fraction: float = 0.60
    row_group_min_tolerance_px: int = 10
    enforce_portrait_orientation: bool = True

    # Tag naming and overwrite behavior.
    output_tag_suffix: str = "_manual"
    clean_previous_outputs: bool = True

    # Exports.
    write_jpg_crops: bool = True
    write_binary_masks: bool = True
    write_transparent_png_crops: bool = True
    write_indexed_previews: bool = True
    preview_subdir_name: str = "previews"

    # Label synchronization.
    manual_reference_csv_name: str = "reference_manual.csv"
    seed_from_reference_csv: bool = True
    seed_reference_csv_name: str = "reference_do.csv"
    write_components_csv: bool = True

    # Paths (relative to workspace root).
    manual_masks_subpath: tuple[str, ...] = (
        "data",
        "iapr-26-uno-vision-challenge",
        "reference_manual_mask",
    )
    reference_images_subpath: tuple[str, ...] = (
        "data",
        "iapr-26-uno-vision-challenge",
        "reference_images",
    )
    reference_cards_subpath: tuple[str, ...] = (
        "project",
        "training_data",
        "training_images",
        "reference_cards",
    )
    reference_labels_subpath: tuple[str, ...] = (
        "project",
        "training_data",
        "object_labels",
        "reference_cards",
    )


@dataclass(frozen=True)
class ManualReferenceImageResult:
    """Export summary for one reference image split."""

    image_name: str
    output_tag: str
    output_dir: Path
    component_count: int
    crop_paths: list[Path]
    mask_paths: list[Path]
    rgba_crop_paths: list[Path]
    preview_path: Path | None


def _resolve_project_root(start: Path | None = None) -> Path:
    project_root = (start or Path.cwd()).resolve()
    while not (project_root / "data").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    return project_root


def _list_manual_masks(manual_masks_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in manual_masks_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def _find_reference_image(image_name: str, reference_images_dir: Path) -> Path:
    for suffix in VALID_IMAGE_EXTENSIONS:
        candidate = reference_images_dir / f"{image_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing source reference image for {image_name} in {reference_images_dir}."
    )


def _clear_outputs(directory: Path) -> None:
    if not directory.exists():
        return
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        for p in directory.glob(suffix):
            p.unlink()


def _group_rows(components: list[dict[str, Any]], cfg: ManualReferenceConfig) -> list[dict[str, Any]]:
    if not components:
        return []

    median_h = float(np.median([float(c["h"]) for c in components]))
    row_tol = max(float(cfg.row_group_min_tolerance_px), cfg.row_group_tolerance_fraction * median_h)

    components_by_y = sorted(components, key=lambda c: float(c["cy"]))
    rows: list[dict[str, Any]] = []
    for comp in components_by_y:
        placed = False
        for row in rows:
            if abs(float(comp["cy"]) - float(row["mean_cy"])) <= row_tol:
                row["items"].append(comp)
                row["mean_cy"] = float(np.mean([float(item["cy"]) for item in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"mean_cy": float(comp["cy"]), "items": [comp]})

    rows.sort(key=lambda r: float(r["mean_cy"]))
    ordered: list[dict[str, Any]] = []
    for row in rows:
        ordered.extend(sorted(row["items"], key=lambda c: float(c["cx"])))
    return ordered


def _load_label_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.is_file():
        return {}

    out: dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = str(row.get("image_id", "")).strip()
            card = str(row.get("card", "")).strip()
            if image_id:
                out[image_id] = card
    return out


def _make_rgba_crop(crop_bgr: np.ndarray, crop_mask_u8: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(crop_mask_u8 > 0, 255, 0).astype(np.uint8)
    return rgba


def _split_one_manual_reference(
    image_name: str,
    mask_path: Path,
    cfg: ManualReferenceConfig,
    reference_images_dir: Path,
    reference_cards_dir: Path,
    labels_dir: Path,
    preview_dir: Path,
) -> tuple[ManualReferenceImageResult, list[dict[str, Any]]]:
    source_image_path = _find_reference_image(image_name, reference_images_dir)
    image_bgr = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read source image: {source_image_path}")

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read manual mask: {mask_path}")
    if mask_raw.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"Mask shape mismatch for {image_name}: mask={mask_raw.shape[:2]}, image={image_bgr.shape[:2]}"
        )

    binary_mask = np.where(mask_raw > int(cfg.mask_threshold), 255, 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    components: list[dict[str, Any]] = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < int(cfg.min_component_area_abs):
            continue

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, centroids[lbl])
        component_mask = np.where(labels == lbl, 255, 0).astype(np.uint8)

        components.append(
            {
                "label_id": int(lbl),
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "mask": component_mask,
            }
        )

    ordered_components = _group_rows(components, cfg)
    output_tag = f"{image_name}{cfg.output_tag_suffix}"
    out_dir = reference_cards_dir / output_tag

    components_dir = out_dir / "components"
    closed_components_dir = out_dir / "closed_components"
    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    rgba_dir = out_dir / "crops_rgba"

    for directory in (components_dir, closed_components_dir, masks_dir, crops_dir, rgba_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if cfg.clean_previous_outputs:
            _clear_outputs(directory)

    crop_paths: list[Path] = []
    mask_paths: list[Path] = []
    rgba_paths: list[Path] = []
    label_rows: list[dict[str, Any]] = []

    preview = image_bgr.copy()

    for idx, comp in enumerate(ordered_components):
        x = int(comp["x"])
        y = int(comp["y"])
        w = int(comp["w"])
        h = int(comp["h"])

        full_component_mask = np.asarray(comp["mask"], dtype=np.uint8)
        component_path = components_dir / f"component_{idx}.jpg"
        cv2.imwrite(str(component_path), full_component_mask)

        closed_component_path = closed_components_dir / f"closed_component_{idx}.jpg"
        cv2.imwrite(str(closed_component_path), full_component_mask)

        crop_bgr = image_bgr[y:y + h, x:x + w]
        crop_mask = full_component_mask[y:y + h, x:x + w]
        crop_mask_u8 = np.where(crop_mask > 0, 255, 0).astype(np.uint8)

        if crop_bgr.size == 0:
            continue

        if cfg.enforce_portrait_orientation and crop_bgr.shape[1] > crop_bgr.shape[0]:
            crop_bgr = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)
            crop_mask_u8 = cv2.rotate(crop_mask_u8, cv2.ROTATE_90_CLOCKWISE)

        crop_path = crops_dir / f"crop_{idx}.jpg"
        if cfg.write_jpg_crops:
            cv2.imwrite(str(crop_path), crop_bgr)
            crop_paths.append(crop_path)

        mask_path_out = masks_dir / f"mask_{idx}.jpg"
        if cfg.write_binary_masks:
            cv2.imwrite(str(mask_path_out), crop_mask_u8)
            mask_paths.append(mask_path_out)

        rgba_path = rgba_dir / f"crop_{idx}.png"
        if cfg.write_transparent_png_crops:
            rgba_crop = _make_rgba_crop(crop_bgr, crop_mask_u8)
            cv2.imwrite(str(rgba_path), rgba_crop)
            rgba_paths.append(rgba_path)

        image_id = f"{output_tag}_crop_{idx}"
        source_image_id = f"{image_name}_crop_{idx}"
        label_rows.append(
            {
                "image_name": image_name,
                "output_tag": output_tag,
                "image_id": image_id,
                "source_image_id": source_image_id,
                "component_id": int(comp["label_id"]),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": int(comp["area"]),
            }
        )

        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
        cv2.putText(
            preview,
            str(idx),
            (x + 5, max(22, y + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    preview_path: Path | None = None
    if cfg.write_indexed_previews and ordered_components:
        preview_path = preview_dir / f"reference_manual_{image_name}_preview.jpg"
        cv2.imwrite(str(preview_path), preview)

    result = ManualReferenceImageResult(
        image_name=image_name,
        output_tag=output_tag,
        output_dir=out_dir,
        component_count=len(ordered_components),
        crop_paths=crop_paths,
        mask_paths=mask_paths,
        rgba_crop_paths=rgba_paths,
        preview_path=preview_path,
    )
    return result, label_rows


def _write_reference_manual_csv(
    label_rows: list[dict[str, Any]],
    manual_csv_path: Path,
    existing_labels: dict[str, str],
    seed_labels: dict[str, str],
) -> tuple[int, int]:
    rows_to_write: list[dict[str, str]] = []
    newly_unlabeled = 0
    prefilled_from_seed = 0

    for row in label_rows:
        image_id = str(row["image_id"])
        source_image_id = str(row["source_image_id"])

        card = existing_labels.get(image_id, "")
        if not card:
            seed_card = seed_labels.get(image_id, "")
            if not seed_card:
                seed_card = seed_labels.get(source_image_id, "")
            if seed_card:
                card = seed_card
                prefilled_from_seed += 1
            else:
                newly_unlabeled += 1

        rows_to_write.append({"image_id": image_id, "card": card})

    with manual_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "card"])
        writer.writeheader()
        writer.writerows(rows_to_write)

    return newly_unlabeled, prefilled_from_seed


def _write_components_csv(
    label_rows: list[dict[str, Any]],
    components_csv_path: Path,
    labels_by_image_id: dict[str, str],
) -> None:
    with components_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "output_tag",
                "image_id",
                "source_image_id",
                "component_id",
                "x",
                "y",
                "w",
                "h",
                "area",
                "card",
            ],
        )
        writer.writeheader()
        for row in label_rows:
            image_id = str(row["image_id"])
            writer.writerow(
                {
                    **row,
                    "card": labels_by_image_id.get(image_id, ""),
                }
            )


def initialize_manual_reference_pipeline(
    config: ManualReferenceConfig | None = None,
) -> dict[str, Any]:
    """Resolve paths and discover which manual masks to process."""
    cfg = config or ManualReferenceConfig()
    project_root = _resolve_project_root()

    manual_masks_dir = project_root.joinpath(*cfg.manual_masks_subpath)
    reference_images_dir = project_root.joinpath(*cfg.reference_images_subpath)
    reference_cards_dir = project_root.joinpath(*cfg.reference_cards_subpath)
    labels_dir = project_root.joinpath(*cfg.reference_labels_subpath)

    if not manual_masks_dir.exists():
        raise FileNotFoundError(f"Manual mask directory not found: {manual_masks_dir}")
    if not reference_images_dir.exists():
        raise FileNotFoundError(f"Reference image directory not found: {reference_images_dir}")

    reference_cards_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = labels_dir / cfg.preview_subdir_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    manual_masks = _list_manual_masks(manual_masks_dir)
    if not manual_masks:
        raise RuntimeError(f"No mask images found in: {manual_masks_dir}")

    available_names = [p.stem for p in manual_masks]
    if cfg.image_names is None:
        selected_names = available_names
    else:
        selected_names = [name for name in cfg.image_names if name in set(available_names)]
        missing = [name for name in cfg.image_names if name not in set(available_names)]
        if missing:
            print(f"[warning] Missing manual masks for: {missing}")

    print(f"Project root: {project_root}")
    print(f"Manual masks: {manual_masks_dir}")
    print(f"Reference images: {reference_images_dir}")
    print(f"Reference output root: {reference_cards_dir}")
    print(f"Label output dir: {labels_dir}")
    print(f"Selected images: {selected_names}")

    return {
        "config": cfg,
        "project_root": project_root,
        "manual_masks_dir": manual_masks_dir,
        "reference_images_dir": reference_images_dir,
        "reference_cards_dir": reference_cards_dir,
        "labels_dir": labels_dir,
        "preview_dir": preview_dir,
        "available_image_names": available_names,
        "selected_image_names": selected_names,
        "results": {},
        "label_rows": [],
        "reference_manual_csv": labels_dir / cfg.manual_reference_csv_name,
        "reference_manual_components_csv": labels_dir / "reference_manual_components.csv",
    }


def run_manual_reference_split(state: dict[str, Any]) -> dict[str, Any]:
    """Split manual masks and synchronize `reference_manual.csv`."""
    cfg: ManualReferenceConfig = state["config"]
    manual_masks_dir: Path = state["manual_masks_dir"]
    reference_images_dir: Path = state["reference_images_dir"]
    reference_cards_dir: Path = state["reference_cards_dir"]
    labels_dir: Path = state["labels_dir"]
    preview_dir: Path = state["preview_dir"]

    manual_csv_path: Path = state["reference_manual_csv"]
    components_csv_path: Path = state["reference_manual_components_csv"]
    seed_csv_path = labels_dir / cfg.seed_reference_csv_name

    existing_labels = _load_label_map(manual_csv_path)
    seed_labels = _load_label_map(seed_csv_path) if cfg.seed_from_reference_csv else {}

    results: dict[str, ManualReferenceImageResult] = {}
    all_label_rows: list[dict[str, Any]] = []

    for image_name in state["selected_image_names"]:
        mask_path: Path | None = None
        for suffix in VALID_IMAGE_EXTENSIONS:
            candidate = manual_masks_dir / f"{image_name}{suffix}"
            if candidate.is_file():
                mask_path = candidate
                break
        if mask_path is None:
            print(f"[warning] Mask not found for {image_name}, skipping.")
            continue

        result, label_rows = _split_one_manual_reference(
            image_name=image_name,
            mask_path=mask_path,
            cfg=cfg,
            reference_images_dir=reference_images_dir,
            reference_cards_dir=reference_cards_dir,
            labels_dir=labels_dir,
            preview_dir=preview_dir,
        )
        results[image_name] = result
        all_label_rows.extend(label_rows)

        print(
            f"{image_name}: {result.component_count} components | "
            f"jpg={len(result.crop_paths)} masks={len(result.mask_paths)} rgba={len(result.rgba_crop_paths)}"
        )

    def _row_key(row: dict[str, Any]) -> tuple[str, int]:
        image_id = str(row["image_id"])
        tag, _, idx = image_id.rpartition("_crop_")
        try:
            n_idx = int(idx)
        except ValueError:
            n_idx = 10**9
        return tag, n_idx

    all_label_rows = sorted(all_label_rows, key=_row_key)

    newly_unlabeled, prefilled_from_seed = _write_reference_manual_csv(
        label_rows=all_label_rows,
        manual_csv_path=manual_csv_path,
        existing_labels=existing_labels,
        seed_labels=seed_labels,
    )

    labels_by_image_id = _load_label_map(manual_csv_path)
    if cfg.write_components_csv:
        _write_components_csv(
            label_rows=all_label_rows,
            components_csv_path=components_csv_path,
            labels_by_image_id=labels_by_image_id,
        )

    state["results"] = results
    state["label_rows"] = all_label_rows
    state["sync_summary"] = {
        "rows_written": len(all_label_rows),
        "newly_unlabeled": newly_unlabeled,
        "prefilled_from_seed": prefilled_from_seed,
        "manual_csv_path": manual_csv_path,
        "components_csv_path": components_csv_path,
    }

    print(
        "reference_manual.csv sync: "
        f"rows={len(all_label_rows)} | prefilled={prefilled_from_seed} | empty={newly_unlabeled}"
    )
    print(f"Reference manual CSV: {manual_csv_path}")
    if cfg.write_components_csv:
        print(f"Reference components CSV: {components_csv_path}")

    return state


def summarize_manual_reference_split(state: dict[str, Any]) -> dict[str, Any]:
    """Return compact split metrics for notebook display."""
    results: dict[str, ManualReferenceImageResult] = state.get("results", {})
    sync_summary = dict(state.get("sync_summary", {}))

    per_image = {
        name: {
            "components": r.component_count,
            "jpg_crops": len(r.crop_paths),
            "jpg_masks": len(r.mask_paths),
            "rgba_png_crops": len(r.rgba_crop_paths),
        }
        for name, r in results.items()
    }

    return {
        "images": per_image,
        "total_images": len(results),
        "total_components": int(sum(r.component_count for r in results.values())),
        **sync_summary,
    }
