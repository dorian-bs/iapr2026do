"""Top-level orchestration for the augmented-data generation notebook.

Public surface mirrors the train_classifier_CNN pipeline pattern:
    CreateAugmentedDataConfig
    initialize_create_augmented_data_pipeline(cfg) -> state
    run_card_generation(state) -> state
    run_scene_preview(state) -> state
    run_scene_generation(state) -> state
    plot_card_preview(state)
    plot_scene_preview(state)
    plot_saved_scenes(state)
"""
from __future__ import annotations

import os
import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from .augmentation import compose_augmented_scene, render_augmented_card
from .config import CreateAugmentedDataConfig
from .data import (
    AssetCaches,
    PLAYERS,
    Paths,
    build_asset_caches,
    ensure_output_dirs,
    load_or_make_token,
    load_reference_card_assets,
    resolve_paths,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(values):
        return values


_SCENE_WORKER_CARD_ASSETS: list[Any] | None = None
_SCENE_WORKER_CFG: CreateAugmentedDataConfig | None = None
_SCENE_WORKER_PATHS: Paths | None = None
_SCENE_WORKER_CACHES: AssetCaches | None = None
_SCENE_WORKER_FMT: str | None = None

_CARD_WORKER_CARD_ASSETS: list[Any] | None = None
_CARD_WORKER_CFG: CreateAugmentedDataConfig | None = None
_CARD_WORKER_PATHS: Paths | None = None
_CARD_WORKER_CACHES: AssetCaches | None = None
_CARD_WORKER_FMT: str | None = None


def _resolve_card_worker_count(cfg: CreateAugmentedDataConfig, total_augmented: int) -> int:
    requested = int(cfg.aug_card_generation_workers)
    if requested < 0:
        raise ValueError("aug_card_generation_workers must be >= 0.")

    if requested == 0:
        auto_workers = min(8, max(1, os.cpu_count() or 1))
        return min(auto_workers, max(1, total_augmented))

    return min(requested, max(1, total_augmented))


def _resolve_card_backend(cfg: CreateAugmentedDataConfig) -> str:
    backend = str(cfg.aug_card_generation_backend).strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError("aug_card_generation_backend must be either 'process' or 'thread'.")
    return backend


def _resolve_card_chunk_size(cfg: CreateAugmentedDataConfig) -> int:
    return max(1, int(cfg.aug_card_generation_chunk_size))


def _resolve_scene_worker_count(cfg: CreateAugmentedDataConfig) -> int:
    requested = int(cfg.scene_generation_workers)
    if requested < 0:
        raise ValueError("scene_generation_workers must be >= 0.")

    if requested == 0:
        auto_workers = min(8, max(1, os.cpu_count() or 1))
        return min(auto_workers, max(1, int(cfg.n_scenes)))

    return min(requested, max(1, int(cfg.n_scenes)))


def _resolve_scene_backend(cfg: CreateAugmentedDataConfig) -> str:
    backend = str(cfg.scene_generation_backend).strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError("scene_generation_backend must be either 'process' or 'thread'.")
    return backend


def _resolve_scene_chunk_size(cfg: CreateAugmentedDataConfig) -> int:
    return max(1, int(cfg.scene_generation_chunk_size))


def _prime_scene_runtime_caches(
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
) -> None:
    """Warm token caches so workers avoid repeated setup in hot loops."""
    warmup_rng = np.random.default_rng(cfg.seed + 31_415)
    for style_key in ("white_rect", "funky_round"):
        for player_name in PLAYERS:
            load_or_make_token(style_key, player_name, warmup_rng, cfg.scene_height, paths, caches)


def _compose_and_save_scene(
    scene_index: int,
    scene_seed: int,
    card_assets: list[Any],
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
    scene_fmt: str,
) -> dict[str, Any]:
    scene_rng = np.random.default_rng(scene_seed)
    scene_bgr, scene_mask, scene_metadata = compose_augmented_scene(
        card_assets, scene_rng, cfg, paths, caches
    )
    scene_name = f"aug_scene_{scene_index:05d}"
    image_path = paths.scenes_img_dir / f"{scene_name}.{scene_fmt}"
    mask_path = paths.scenes_mask_dir / f"{scene_name}.png"

    _write_rgb_image(image_path, scene_bgr, fmt=scene_fmt, jpeg_quality=cfg.scene_jpeg_quality)
    mask_ok = cv2.imwrite(str(mask_path), scene_mask)
    if not mask_ok:
        raise OSError(f"Failed to write image: {mask_path}")

    return {
        "scene": scene_name,
        "image_path": str(image_path.relative_to(paths.project_root)),
        "mask_path": str(mask_path.relative_to(paths.project_root)),
        **scene_metadata,
    }


def _initialize_scene_worker(
    card_assets: list[Any],
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
    scene_fmt: str,
) -> None:
    global _SCENE_WORKER_CARD_ASSETS
    global _SCENE_WORKER_CFG
    global _SCENE_WORKER_PATHS
    global _SCENE_WORKER_CACHES
    global _SCENE_WORKER_FMT

    _SCENE_WORKER_CARD_ASSETS = card_assets
    _SCENE_WORKER_CFG = cfg
    _SCENE_WORKER_PATHS = paths
    _SCENE_WORKER_CACHES = caches
    _SCENE_WORKER_FMT = scene_fmt


def _run_scene_worker(task: tuple[int, int]) -> dict[str, Any]:
    scene_index, scene_seed = task
    if (
        _SCENE_WORKER_CARD_ASSETS is None
        or _SCENE_WORKER_CFG is None
        or _SCENE_WORKER_PATHS is None
        or _SCENE_WORKER_CACHES is None
        or _SCENE_WORKER_FMT is None
    ):
        raise RuntimeError("Scene worker was not initialized.")

    return _compose_and_save_scene(
        scene_index=scene_index,
        scene_seed=scene_seed,
        card_assets=_SCENE_WORKER_CARD_ASSETS,
        cfg=_SCENE_WORKER_CFG,
        paths=_SCENE_WORKER_PATHS,
        caches=_SCENE_WORKER_CACHES,
        scene_fmt=_SCENE_WORKER_FMT,
    )


def _normalize_image_format(value: str) -> str:
    fmt = str(value).strip().lower().lstrip(".")
    if fmt == "jpeg":
        return "jpg"
    if fmt in {"jpg", "png"}:
        return fmt
    raise ValueError(f"Unsupported image format: {value}. Use one of: png, jpg, jpeg.")


def _clamp_jpeg_quality(value: int) -> int:
    return max(1, min(100, int(value)))


def _write_rgb_image(path, img_bgr: np.ndarray, fmt: str, jpeg_quality: int) -> None:
    params: list[int] = []
    if fmt == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), _clamp_jpeg_quality(jpeg_quality)]
    ok = cv2.imwrite(str(path), img_bgr, params) if params else cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def initialize_create_augmented_data_pipeline(cfg: CreateAugmentedDataConfig) -> dict[str, Any]:
    """Resolve paths, prepare output dirs, load card assets, and prime caches."""
    paths = resolve_paths()
    ensure_output_dirs(paths)
    caches = build_asset_caches(paths)
    card_assets = load_reference_card_assets(paths)
    labels_loaded = sorted({asset.label for asset in card_assets})
    card_fmt = _normalize_image_format(cfg.aug_card_image_format)
    scene_fmt = _normalize_image_format(cfg.scene_image_format)

    print(f"Project root: {paths.project_root}")
    print(f"Reference cards: {paths.ref_cards_dir}")
    print(f"Reference labels: {paths.reference_csv}")
    print(f"Funky background: {paths.funky_bg_path}")
    print(f"White background: {paths.white_bg_path}")
    print(f"Augmented cards: {paths.aug_cards_dir}")
    print(f"Augmented card masks: {paths.aug_masks_dir}")
    print(f"Augmented labels: {paths.aug_csv_path}")
    print(f"Scene images: {paths.scenes_img_dir}")
    print(f"Scene masks: {paths.scenes_mask_dir}")
    print(f"Card image format: {card_fmt} (jpeg_quality={_clamp_jpeg_quality(cfg.aug_card_jpeg_quality)})")
    print(f"Scene image format: {scene_fmt} (jpeg_quality={_clamp_jpeg_quality(cfg.scene_jpeg_quality)})")
    print(f"Token crop folders: {paths.token_asset_dir}")
    print(f"Loaded {len(card_assets)} reference crops across {len(labels_loaded)} labels.")

    return {
        "cfg": cfg,
        "paths": paths,
        "caches": caches,
        "card_assets": card_assets,
        "labels": labels_loaded,
        "aug_rows": [],
        "scene_metadata": [],
        "preview_scene_bgr": None,
        "preview_mask": None,
        "preview_metadata": None,
        "card_preview_rng": None,
    }


def _clear_augmented_card_outputs(paths: Paths) -> None:
    """Remove only files produced by this augmented-card generator."""
    paths.aug_cards_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("aug_*.png", "aug_*.jpg", "aug_*.jpeg"):
        for output_file in paths.aug_cards_dir.glob(pattern):
            if output_file.is_file():
                output_file.unlink()

    paths.aug_masks_dir.mkdir(parents=True, exist_ok=True)
    for output_file in paths.aug_masks_dir.glob("aug_*.png"):
        if output_file.is_file():
            output_file.unlink()

    paths.aug_labels_dir.mkdir(parents=True, exist_ok=True)
    if paths.aug_csv_path.exists():
        paths.aug_csv_path.unlink()


def _clear_scene_outputs(paths: Paths) -> None:
    """Remove only files produced by this scene generator."""
    for output_dir in (paths.scenes_img_dir, paths.scenes_mask_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for output_file in output_dir.glob("aug_scene_*"):
            if output_file.is_file():
                output_file.unlink()

    paths.scenes_labels_dir.mkdir(parents=True, exist_ok=True)
    if paths.scenes_labels_path.exists():
        paths.scenes_labels_path.unlink()


def _compose_and_save_augmented_card(
    aug_index: int,
    aug_seed: int,
    asset_index: int,
    card_assets: list[Any],
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
    card_fmt: str,
) -> dict[str, str]:
    asset = card_assets[asset_index]
    rng_card = np.random.default_rng(aug_seed)
    aug_img, aug_mask = render_augmented_card(asset, rng_card, cfg, paths, caches)
    image_id = f"aug_{aug_index:05d}"
    image_path = paths.aug_cards_dir / f"{image_id}.{card_fmt}"
    mask_path = paths.aug_masks_dir / f"{image_id}.png"

    _write_rgb_image(image_path, aug_img, fmt=card_fmt, jpeg_quality=cfg.aug_card_jpeg_quality)
    mask_ok = cv2.imwrite(str(mask_path), aug_mask)
    if not mask_ok:
        raise OSError(f"Failed to write image: {mask_path}")

    return {
        "image_id": image_id,
        "card": asset.label,
        "mask_path": str(mask_path.relative_to(paths.project_root)),
    }


def _initialize_card_worker(
    card_assets: list[Any],
    cfg: CreateAugmentedDataConfig,
    paths: Paths,
    caches: AssetCaches,
    card_fmt: str,
) -> None:
    global _CARD_WORKER_CARD_ASSETS
    global _CARD_WORKER_CFG
    global _CARD_WORKER_PATHS
    global _CARD_WORKER_CACHES
    global _CARD_WORKER_FMT

    _CARD_WORKER_CARD_ASSETS = card_assets
    _CARD_WORKER_CFG = cfg
    _CARD_WORKER_PATHS = paths
    _CARD_WORKER_CACHES = caches
    _CARD_WORKER_FMT = card_fmt


def _run_card_worker(task: tuple[int, int, int]) -> dict[str, str]:
    aug_index, aug_seed, asset_index = task
    if (
        _CARD_WORKER_CARD_ASSETS is None
        or _CARD_WORKER_CFG is None
        or _CARD_WORKER_PATHS is None
        or _CARD_WORKER_CACHES is None
        or _CARD_WORKER_FMT is None
    ):
        raise RuntimeError("Card worker was not initialized.")

    return _compose_and_save_augmented_card(
        aug_index=aug_index,
        aug_seed=aug_seed,
        asset_index=asset_index,
        card_assets=_CARD_WORKER_CARD_ASSETS,
        cfg=_CARD_WORKER_CFG,
        paths=_CARD_WORKER_PATHS,
        caches=_CARD_WORKER_CACHES,
        card_fmt=_CARD_WORKER_FMT,
    )


def run_card_generation(state: dict[str, Any]) -> dict[str, Any]:
    """Generate single-card augmented crops and save the labels CSV."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: Paths = state["paths"]
    caches: AssetCaches = state["caches"]
    card_assets = state["card_assets"]

    if cfg.clear_output_dirs:
        _clear_augmented_card_outputs(paths)

    total_augmented = len(card_assets) * cfg.n_aug_per_reference
    card_fmt = _normalize_image_format(cfg.aug_card_image_format)
    rng_cards = np.random.default_rng(cfg.seed + 1)

    card_tasks = [
        (aug_index, int(aug_seed), aug_index % len(card_assets))
        for aug_index, aug_seed in enumerate(
            rng_cards.integers(
                0,
                np.iinfo(np.uint32).max,
                size=total_augmented,
                dtype=np.uint32,
            )
        )
    ]

    aug_rows: list[dict[str, str]] = []
    worker_count = _resolve_card_worker_count(cfg, total_augmented)
    backend = _resolve_card_backend(cfg)

    if worker_count <= 1:
        for task in tqdm(card_tasks):
            aug_rows.append(
                _compose_and_save_augmented_card(
                    aug_index=task[0],
                    aug_seed=task[1],
                    asset_index=task[2],
                    card_assets=card_assets,
                    cfg=cfg,
                    paths=paths,
                    caches=caches,
                    card_fmt=card_fmt,
                )
            )
    else:
        print(f"Generating augmented cards with {worker_count} {backend} workers...")
        executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        with executor_cls(
            max_workers=worker_count,
            initializer=_initialize_card_worker,
            initargs=(card_assets, cfg, paths, caches, card_fmt),
        ) as executor:
            if backend == "process":
                mapped = executor.map(_run_card_worker, card_tasks, chunksize=_resolve_card_chunk_size(cfg))
            else:
                mapped = executor.map(_run_card_worker, card_tasks)

            for row in tqdm(mapped, total=len(card_tasks)):
                aug_rows.append(row)

    with paths.aug_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "card", "mask_path"])
        writer.writeheader()
        writer.writerows(aug_rows)

    print(f"Saved {len(aug_rows)} augmented cards to: {paths.aug_cards_dir}")
    print(f"Saved {len(aug_rows)} augmented card masks to: {paths.aug_masks_dir}")
    print(f"Saved augmented labels to: {paths.aug_csv_path}")

    state["aug_rows"] = aug_rows
    state["card_preview_rng"] = rng_cards
    return state


def run_scene_preview(state: dict[str, Any]) -> dict[str, Any]:
    """Compose one preview scene without saving it to disk."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: Paths = state["paths"]
    caches: AssetCaches = state["caches"]
    card_assets = state["card_assets"]

    preview_rng = np.random.default_rng(cfg.seed)
    preview_scene_bgr, preview_mask, preview_metadata = compose_augmented_scene(
        card_assets, preview_rng, cfg, paths, caches
    )

    state["preview_scene_bgr"] = preview_scene_bgr
    state["preview_mask"] = preview_mask
    state["preview_metadata"] = preview_metadata
    return state


def run_scene_generation(state: dict[str, Any]) -> dict[str, Any]:
    """Generate the full augmented scene dataset and write metadata JSON."""
    cfg: CreateAugmentedDataConfig = state["cfg"]
    paths: Paths = state["paths"]
    caches: AssetCaches = state["caches"]
    card_assets = state["card_assets"]

    if cfg.clear_output_dirs:
        _clear_scene_outputs(paths)

    scene_fmt = _normalize_image_format(cfg.scene_image_format)
    _prime_scene_runtime_caches(cfg, paths, caches)

    scene_tasks = [
        (scene_index, int(scene_seed))
        for scene_index, scene_seed in enumerate(
            np.random.default_rng(cfg.seed).integers(
                0,
                np.iinfo(np.uint32).max,
                size=int(cfg.n_scenes),
                dtype=np.uint32,
            )
        )
    ]

    all_scene_metadata: list[dict[str, Any]] = []
    worker_count = _resolve_scene_worker_count(cfg)
    backend = _resolve_scene_backend(cfg)

    if worker_count <= 1:
        for task in tqdm(scene_tasks):
            scene_metadata = _compose_and_save_scene(
                scene_index=task[0],
                scene_seed=task[1],
                card_assets=card_assets,
                cfg=cfg,
                paths=paths,
                caches=caches,
                scene_fmt=scene_fmt,
            )
            all_scene_metadata.append(scene_metadata)
    else:
        print(f"Generating scenes with {worker_count} {backend} workers...")
        executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        with executor_cls(
            max_workers=worker_count,
            initializer=_initialize_scene_worker,
            initargs=(card_assets, cfg, paths, caches, scene_fmt),
        ) as executor:
            if backend == "process":
                mapped = executor.map(_run_scene_worker, scene_tasks, chunksize=_resolve_scene_chunk_size(cfg))
            else:
                mapped = executor.map(_run_scene_worker, scene_tasks)

            for scene_metadata in tqdm(mapped, total=len(scene_tasks)):
                all_scene_metadata.append(scene_metadata)

    with paths.scenes_labels_path.open("w", encoding="utf-8") as labels_file:
        json.dump(all_scene_metadata, labels_file, indent=2)

    print(f"Saved {len(all_scene_metadata)} scenes to: {paths.scenes_img_dir}")
    print(f"Saved {len(all_scene_metadata)} masks to:  {paths.scenes_mask_dir}")
    print(f"Saved scene metadata to:   {paths.scenes_labels_path}")

    state["scene_metadata"] = all_scene_metadata
    return state
