"""Helpers to resolve active model artifacts from folder-based layouts.

Layout conventions:
  models/segmenter/used/        -> exactly one active checkpoint file
  models/segmenter/backup/      -> storage only
  models/card_classifier_cnn/used/<bundle_dir>/
                                        -> exactly one active bundle directory
  models/card_classifier_cnn/backup/<bundle_dir>/
                                        -> storage only
"""
from __future__ import annotations

from pathlib import Path


_CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}


def ensure_model_layout_dirs(models_dir: Path) -> dict[str, Path]:
    segmenter_used = models_dir / "segmenter" / "used"
    segmenter_backup = models_dir / "segmenter" / "backup"
    classifier_root = models_dir / "card_classifier_cnn"
    classifier_used = classifier_root / "used"
    classifier_backup = classifier_root / "backup"

    segmenter_used.mkdir(parents=True, exist_ok=True)
    segmenter_backup.mkdir(parents=True, exist_ok=True)
    classifier_used.mkdir(parents=True, exist_ok=True)
    classifier_backup.mkdir(parents=True, exist_ok=True)

    return {
        "segmenter_used": segmenter_used,
        "segmenter_backup": segmenter_backup,
        "classifier_root": classifier_root,
        "classifier_used": classifier_used,
        "classifier_backup": classifier_backup,
    }


def _single_file(
    directory: Path,
    *,
    label: str,
    suffixes: set[str] | None = None,
    required: bool,
) -> Path | None:
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if suffixes is not None:
        narrowed = [p for p in files if p.suffix.lower() in suffixes]
        if narrowed:
            files = narrowed

    if not files:
        if required:
            raise FileNotFoundError(
                f"No {label} found in {directory}."
            )
        return None
    if len(files) > 1:
        raise RuntimeError(
            f"Expected exactly one {label} in {directory}, found {len(files)}. "
            "Keep only one active artifact there."
        )
    return files[0]


def resolve_segmenter_checkpoint(models_dir: Path) -> Path:
    layout = ensure_model_layout_dirs(models_dir)
    used_dir = layout["segmenter_used"]

    try:
        return _single_file(
            used_dir,
            label="segmenter checkpoint",
            suffixes=_CHECKPOINT_SUFFIXES,
            required=True,
        )
    except FileNotFoundError:
        # Backward compatibility with the previous flat layout.
        legacy = models_dir / "scene_segmenter_unet_small.pth"
        if legacy.is_file():
            return legacy
        raise


def resolve_classifier_bundle(models_dir: Path) -> dict[str, Path | None]:
    layout = ensure_model_layout_dirs(models_dir)
    used_root = layout["classifier_used"]
    bundle_dirs = sorted(p for p in used_root.iterdir() if p.is_dir())

    if len(bundle_dirs) > 1:
        raise RuntimeError(
            f"Expected exactly one active classifier bundle in {used_root}, "
            f"found {len(bundle_dirs)}. Move extras to backup."
        )

    if len(bundle_dirs) == 1:
        bundle_dir = bundle_dirs[0]
        model_path = _single_file(
            bundle_dir,
            label="classifier checkpoint",
            suffixes=_CHECKPOINT_SUFFIXES,
            required=True,
        )
        classes_path = _single_file(
            bundle_dir,
            label="classifier classes file (.npy)",
            suffixes={".npy"},
            required=True,
        )
        config_path = _single_file(
            bundle_dir,
            label="classifier config file (.json)",
            suffixes={".json"},
            required=False,
        )
        return {
            "bundle_dir": bundle_dir,
            "model_path": model_path,
            "classes_path": classes_path,
            "config_path": config_path,
        }

    raise FileNotFoundError(
        "No active classifier bundle found. Expected either one folder under "
        f"{used_root}."
    )