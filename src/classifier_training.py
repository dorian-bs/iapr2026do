"""Report helpers for the masked card-classifier training stage."""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, WeightedRandomSampler

from src.inference import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CardResNet18SmallClassifier,
    assert_param_cap,
    resolve_classifier_bundle,
)


@dataclass
class ClassifierPipelineConfig:
    """Configuration used by the final classifier training notebook."""

    seed: int = 42
    img_size: int = 160
    bbox_margin: float = 0.08
    mask_threshold: float = 0.50
    val_split: float = 0.20

    stage_1_epochs: int = 4
    stage_2_epochs: int = 90

    stage_1_lr: float = 1e-3
    stage_2_lr: float = 3e-4

    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    balanced_sampling: bool = True
    early_stop_patience: int = 6
    min_epochs_per_stage: int = 2
    epoch_max_train_samples: int | None = 8192

    batch_size_cuda: int = 32
    batch_size_mps: int = 16
    batch_size_cpu: int = 8
    num_workers: int = 4
    persistent_workers: bool = True
    cache_scene_assets: bool = True
    cache_max_scene_assets: int = 128

    augmented_target_gpu_mps: int = 4096
    augmented_target_cpu: int = 1536
    scene_target_gpu_mps: int = 4096
    scene_target_cpu: int = 1536

    grad_clip_norm: float = 1.0
    max_loaded_scene_images: int | None = None
    preview_per_stage: int = 3

    classifier_architecture: str = "resnet18_small"
    classifier_stem_width: int = 60
    classifier_dropout: float = 0.20

    best_epoch_selection_metric: str = "composite"
    selection_weight_val_acc: float = 0.35
    selection_weight_macro_f1: float = 0.30
    selection_weight_balanced_acc: float = 0.20
    selection_weight_top3_acc: float = 0.10
    selection_weight_val_loss: float = 0.05


@dataclass(frozen=True)
class CardSample:
    stage: str
    label: str
    image_path: Path
    bbox: tuple[int, int, int, int] | None = None
    scene_mask_path: Path | None = None
    mask_path: Path | None = None


SCENE_STAGES = {"scene_manual"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_seconds(seconds: float) -> str:
    seconds = int(round(float(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _shutdown_loader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if iterator is not None:
        try:
            setattr(loader, "_iterator", None)
        except Exception:
            pass


def _select_batch_size(device: torch.device, cfg: ClassifierPipelineConfig) -> int:
    if device.type == "cuda":
        return cfg.batch_size_cuda
    if device.type == "mps":
        return cfg.batch_size_mps
    return cfg.batch_size_cpu


def _stage_samples_per_epoch(n_samples: int, stage: str, batch_size: int, device: torch.device, cfg: ClassifierPipelineConfig) -> int:
    on_accel = device.type in {"cuda", "mps"}
    if stage == "augmented_card":
        target = cfg.augmented_target_gpu_mps if on_accel else cfg.augmented_target_cpu
    else:
        target = cfg.scene_target_gpu_mps if on_accel else cfg.scene_target_cpu
    planned = min(int(n_samples), int(target))
    planned = (planned // batch_size) * batch_size
    if planned <= 0:
        planned = min(int(n_samples), int(batch_size))
    if cfg.epoch_max_train_samples is not None:
        planned = min(planned, int(cfg.epoch_max_train_samples))
    return max(1, planned) if n_samples > 0 else 0


def _assert_no_test_inputs(paths: list[Path]) -> None:
    for path in paths:
        assert "test" not in str(path).lower(), f"Refusing to train on test path: {path}"


def _find_image_by_stem(directory: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_augmented_card_samples(
    aug_csv: Path,
    aug_cards_dir: Path,
    aug_masks_dir: Path,
    project_root: Path,
) -> tuple[list[CardSample], list[str], int, int, int]:
    samples: list[CardSample] = []
    missing: list[str] = []
    skipped_labels = 0
    skipped_files = 0
    skipped_masks = 0
    with aug_csv.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row.get("image_id", "")).strip()
            label = str(row.get("card", "")).strip()
            if not image_id or not label or label == "token":
                skipped_labels += 1
                continue
            image_path: Path | None = None
            raw_image = str(row.get("image_path", "")).strip()
            if raw_image:
                candidate = Path(raw_image)
                if not candidate.is_absolute():
                    candidate = project_root / _strip_project_prefix(candidate)
                if candidate.is_file():
                    image_path = candidate
            if image_path is None:
                image_path = _find_image_by_stem(aug_cards_dir, image_id)
            if image_path is None:
                skipped_files += 1
                missing.append(str(aug_cards_dir / f"{image_id}.jpg"))
                continue
            mask_path: Path | None = None
            raw_mask = str(row.get("mask_path", "")).strip()
            if raw_mask:
                candidate = Path(raw_mask)
                if not candidate.is_absolute():
                    candidate = project_root / _strip_project_prefix(candidate)
                if candidate.is_file():
                    mask_path = candidate
            if mask_path is None:
                mask_path = _find_image_by_stem(aug_masks_dir, image_id)
            if mask_path is None:
                skipped_masks += 1
            samples.append(CardSample("augmented_card", label, image_path, mask_path=mask_path))
    return samples, missing, skipped_labels, skipped_files, skipped_masks


def _strip_project_prefix(path: Path) -> Path:
    parts = path.parts
    if parts and parts[0].lower() == "project":
        return Path(*parts[1:])
    return path


def _is_reasonable_scene_bbox(bbox: tuple[int, int, int, int], min_side: int = 28, max_aspect: float = 4.5) -> bool:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width < min_side or height < min_side:
        return False
    return max(width / max(1, height), height / max(1, width)) <= max_aspect


def load_scene_manual_samples(
    scene_labels_path: Path,
    scene_images_dir: Path,
    scene_masks_dir: Path,
    project_root: Path,
) -> tuple[list[CardSample], list[str], int, int, int]:
    records = json.loads(scene_labels_path.read_text(encoding="utf-8"))
    masks_by_stem = {path.stem: path for path in scene_masks_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    samples: list[CardSample] = []
    missing: list[str] = []
    skipped_labels = 0
    skipped_boxes = 0
    skipped_masks = 0
    for record in records:
        scene_name = str(record.get("scene", "")).strip()
        raw_path = record.get("image_path")
        scene_path = None
        if raw_path:
            candidate = Path(str(raw_path))
            scene_path = candidate if candidate.is_absolute() else project_root / _strip_project_prefix(candidate)
        if scene_path is None or not scene_path.is_file():
            scene_path = _find_image_by_stem(scene_images_dir, scene_name) or scene_path
        if scene_path is None or not scene_path.is_file():
            missing.append(scene_name)
            continue
        scene_mask_path = masks_by_stem.get(scene_path.stem) or masks_by_stem.get(scene_name)
        for card in record.get("cards", []):
            label = str(card.get("label", "")).strip()
            bbox_raw = card.get("bbox", [])
            if not label or label == "token":
                skipped_labels += 1
                continue
            if len(bbox_raw) != 4:
                skipped_boxes += 1
                continue
            bbox = tuple(map(int, bbox_raw))
            if not _is_reasonable_scene_bbox(bbox):
                skipped_boxes += 1
                continue
            if scene_mask_path is None or not scene_mask_path.is_file():
                skipped_masks += 1
                continue
            samples.append(CardSample("scene_manual", label, scene_path, bbox=bbox, scene_mask_path=scene_mask_path))
    return samples, missing, skipped_labels, skipped_boxes, skipped_masks


def crop_with_margin(array: np.ndarray, bbox: tuple[int, int, int, int], margin_fraction: float = 0.08) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    height, width = array.shape[:2]
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    margin = int(round(max(box_w, box_h) * margin_fraction))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return array[y0:y1, x0:x1]


def letterbox_image_and_mask(image_bgr: np.ndarray, mask_u8: np.ndarray, size: int, fill_image: int = 128, fill_mask: int = 0) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_bgr.shape[:2]
    scale = size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image_resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_u8, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    image_canvas = np.full((size, size, 3), fill_image, dtype=np.uint8)
    mask_canvas = np.full((size, size), fill_mask, dtype=np.uint8)
    y0 = (size - new_height) // 2
    x0 = (size - new_width) // 2
    image_canvas[y0:y0 + new_height, x0:x0 + new_width] = image_resized
    mask_canvas[y0:y0 + new_height, x0:x0 + new_width] = mask_resized
    return image_canvas, mask_canvas


def compose_masked_card_image(image_bgr: np.ndarray, mask_u8: np.ndarray, bg_fill: int = 128) -> np.ndarray:
    output = np.full_like(image_bgr, bg_fill)
    output[mask_u8 > 0] = image_bgr[mask_u8 > 0]
    return output


def card_input_to_tensor(image_bgr: np.ndarray, mask_u8: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    image_chw = (image_chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    mask_ch = (mask_u8.astype(np.float32) / 255.0)[None, :, :]
    return torch.from_numpy(np.concatenate([image_chw, mask_ch], axis=0).astype(np.float32))


def sample_to_crop_and_mask(
    sample: CardSample,
    bbox_margin: float,
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
    image_crop = crop_with_margin(image_bgr, sample.bbox, bbox_margin) if sample.bbox is not None else image_bgr
    if sample.stage == "augmented_card":
        if sample.mask_path is not None:
            mask_crop = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_crop is None:
                raise FileNotFoundError(f"Cannot read mask: {sample.mask_path}")
            if mask_crop.shape[:2] != image_crop.shape[:2]:
                mask_crop = cv2.resize(mask_crop, (image_crop.shape[1], image_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_crop = np.where(mask_crop > 127, 255, 0).astype(np.uint8)
        else:
            mask_crop = np.full(image_crop.shape[:2], 255, dtype=np.uint8)
    elif sample.stage == "scene_manual":
        if sample.scene_mask_path is None or sample.bbox is None:
            raise ValueError("scene_manual sample requires bbox and scene_mask_path")
        scene_mask = cv2.imread(str(sample.scene_mask_path), cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(f"Cannot read scene mask: {sample.scene_mask_path}")
        mask_crop = np.where(crop_with_margin(scene_mask, sample.bbox, bbox_margin) > 127, 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown sample stage: {sample.stage}")
    if mask_crop.shape[:2] != image_crop.shape[:2]:
        mask_crop = cv2.resize(mask_crop, (image_crop.shape[1], image_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    if int(np.count_nonzero(mask_crop)) < 20:
        mask_crop = np.full(image_crop.shape[:2], 255, dtype=np.uint8)
    return image_crop, mask_crop


def _warp_affine_pair(image_bgr: np.ndarray, mask_u8: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    image_out = cv2.warpAffine(image_bgr, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    mask_out = cv2.warpAffine(mask_u8, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image_out, mask_out


def augment_card_image_and_mask(image_bgr: np.ndarray, mask_u8: np.ndarray, stage: str) -> tuple[np.ndarray, np.ndarray]:
    original_image = image_bgr.copy()
    original_mask = mask_u8.copy()
    height, width = image_bgr.shape[:2]
    if stage in SCENE_STAGES and random.random() < 0.90:
        angle = random.uniform(-14.0, 14.0)
        if random.random() < 0.35:
            angle += random.choice([90.0, 180.0, 270.0])
        image_bgr, mask_u8 = _warp_affine_pair(
            image_bgr,
            mask_u8,
            angle,
            random.uniform(0.90, 1.10),
            random.uniform(-0.05, 0.05) * width,
            random.uniform(-0.05, 0.05) * height,
        )
    if random.random() < 0.65:
        image_bgr = cv2.convertScaleAbs(image_bgr, alpha=random.uniform(0.82, 1.18), beta=random.uniform(-18.0, 18.0))
    if random.random() < 0.35:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.80, 1.25)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        image_bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.15:
        image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    if random.random() < 0.20:
        image_bgr = np.clip(image_bgr.astype(np.float32) + np.random.normal(0, 4, image_bgr.shape), 0, 255).astype(np.uint8)
    if stage in SCENE_STAGES and random.random() < 0.30:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=1) if random.random() < 0.5 else cv2.dilate(mask_u8, kernel, iterations=1)
    if int(np.count_nonzero(mask_u8)) < 20:
        return original_image, original_mask
    return image_bgr, mask_u8


class CardMaskedDataset(Dataset):
    def __init__(
        self,
        samples: list[CardSample],
        label_to_index: dict[str, int],
        image_size: int,
        bbox_margin: float,
        mask_threshold: float,
        augment: bool,
        cache_scene_assets: bool = True,
        cache_max_scene_assets: int = 128,
    ):
        self.samples = samples
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.bbox_margin = bbox_margin
        self.mask_threshold = mask_threshold
        self.augment = augment
        self.cache_scene_assets = bool(cache_scene_assets)
        self.cache_max_scene_assets = max(0, int(cache_max_scene_assets))
        self._scene_image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._scene_mask_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _read_scene_cached(self, path: Path, flags: int, cache: OrderedDict[Path, np.ndarray]) -> np.ndarray:
        if self.cache_scene_assets and self.cache_max_scene_assets > 0:
            cached = cache.get(path)
            if cached is not None:
                cache.move_to_end(path)
                return cached
        array = cv2.imread(str(path), flags)
        if array is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if self.cache_scene_assets and self.cache_max_scene_assets > 0:
            cache[path] = array
            cache.move_to_end(path)
            while len(cache) > self.cache_max_scene_assets:
                cache.popitem(last=False)
        return array

    def _sample_to_crop_and_mask(self, sample: CardSample) -> tuple[np.ndarray, np.ndarray]:
        if sample.stage != "scene_manual":
            return sample_to_crop_and_mask(sample, self.bbox_margin, self.mask_threshold)
        if sample.scene_mask_path is None or sample.bbox is None:
            raise ValueError("scene_manual sample requires bbox and scene_mask_path")
        image_bgr = self._read_scene_cached(sample.image_path, cv2.IMREAD_COLOR, self._scene_image_cache)
        scene_mask = self._read_scene_cached(sample.scene_mask_path, cv2.IMREAD_GRAYSCALE, self._scene_mask_cache)
        image_crop = crop_with_margin(image_bgr, sample.bbox, self.bbox_margin)
        mask_crop = np.where(crop_with_margin(scene_mask, sample.bbox, self.bbox_margin) > 127, 255, 0).astype(np.uint8)
        if mask_crop.shape[:2] != image_crop.shape[:2]:
            mask_crop = cv2.resize(mask_crop, (image_crop.shape[1], image_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(mask_crop)) < 20:
            mask_crop = np.full(image_crop.shape[:2], 255, dtype=np.uint8)
        return image_crop, mask_crop

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_crop, mask_crop = self._sample_to_crop_and_mask(sample)
        image_lb, mask_lb = letterbox_image_and_mask(image_crop, mask_crop, self.image_size)
        if self.augment:
            image_lb, mask_lb = augment_card_image_and_mask(image_lb, mask_lb, sample.stage)
        masked_image = compose_masked_card_image(image_lb, mask_lb)
        return card_input_to_tensor(masked_image, mask_lb), torch.tensor(self.label_to_index[sample.label], dtype=torch.long)


class RotatingCoverageSampler(Sampler[int]):
    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: int, shuffle_epoch: bool = True):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = int(min(max(1, samples_per_epoch), dataset_size))
        self.shuffle_epoch = bool(shuffle_epoch)
        self._rng = random.Random(seed)
        self._indices = list(range(self.dataset_size))
        self._rng.shuffle(self._indices)
        self._cursor = 0

    def __iter__(self):
        selected: list[int] = []
        while len(selected) < self.samples_per_epoch:
            remaining = self.dataset_size - self._cursor
            take = min(self.samples_per_epoch - len(selected), remaining)
            selected.extend(self._indices[self._cursor:self._cursor + take])
            self._cursor += take
            if self._cursor >= self.dataset_size:
                self._rng.shuffle(self._indices)
                self._cursor = 0
        if self.shuffle_epoch:
            self._rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return self.samples_per_epoch


def _seed_loader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(0)


def _make_balanced_sampler(samples: list[CardSample], seed: int, samples_per_epoch: int | None) -> WeightedRandomSampler:
    counts = Counter(sample.label for sample in samples)
    weights = torch.DoubleTensor([1.0 / counts[sample.label] for sample in samples])
    n_samples = len(samples) if samples_per_epoch is None else int(min(samples_per_epoch, len(samples)))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=False, generator=generator)


def make_loader(
    samples: list[CardSample],
    label_to_index: dict[str, int],
    batch_size: int,
    image_size: int,
    bbox_margin: float,
    mask_threshold: float,
    shuffle: bool,
    augment: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    balanced: bool = False,
    samples_per_epoch: int | None = None,
    persistent_workers: bool = True,
    cache_scene_assets: bool = True,
    cache_max_scene_assets: int = 128,
) -> tuple[DataLoader, CardMaskedDataset]:
    dataset = CardMaskedDataset(samples, label_to_index, image_size, bbox_margin, mask_threshold, augment, cache_scene_assets, cache_max_scene_assets)
    sampler = None
    generator = torch.Generator().manual_seed(seed)
    if samples and balanced:
        sampler = _make_balanced_sampler(samples, seed, samples_per_epoch)
    elif samples_per_epoch is not None and samples:
        n_samples = int(min(max(1, samples_per_epoch), len(samples)))
        if n_samples < len(samples):
            sampler = RandomSampler(dataset, replacement=False, num_samples=n_samples, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        worker_init_fn=_seed_loader_worker if num_workers > 0 else None,
        generator=generator,
    )
    return loader, dataset


def make_rotating_loader(
    samples: list[CardSample],
    label_to_index: dict[str, int],
    batch_size: int,
    image_size: int,
    bbox_margin: float,
    mask_threshold: float,
    augment: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    samples_per_epoch: int,
    persistent_workers: bool = True,
    cache_scene_assets: bool = True,
    cache_max_scene_assets: int = 128,
) -> tuple[DataLoader, CardMaskedDataset]:
    dataset = CardMaskedDataset(samples, label_to_index, image_size, bbox_margin, mask_threshold, augment, cache_scene_assets, cache_max_scene_assets)
    sampler = RotatingCoverageSampler(len(samples), samples_per_epoch, seed=seed, shuffle_epoch=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        worker_init_fn=_seed_loader_worker if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, dataset


def _deterministic_path_subset(paths: list[Path], max_count: int, seed: int) -> list[Path]:
    if len(paths) <= max_count:
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(paths, k=max_count))


def initialize_training_pipeline(
    config: ClassifierPipelineConfig | None = None,
    project_root: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Load datasets and initialize the two-stage classifier training state."""
    cfg = config or ClassifierPipelineConfig()
    _seed_everything(cfg.seed)
    project_root = Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    training_data = project_root / "training_data"
    models_dir = Path(models_dir).resolve() if models_dir is not None else project_root / "models"
    classifier_bundle_dir = models_dir / "card_classifier_cnn" / "used" / "latest"
    classifier_bundle_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = training_data / "object_labels"
    augmented_data = training_data / "augmented_data"
    scene_labels_path = labels_dir / "augmented_scenes.json"
    scene_images_dir = augmented_data / "scene_images"
    scene_masks_dir = augmented_data / "scene_masks"
    aug_csv = labels_dir / "augmented_cards.csv"
    aug_cards_dir = augmented_data / "card_images"
    aug_masks_dir = augmented_data / "card_masks"
    required = [scene_labels_path, scene_images_dir, scene_masks_dir, aug_csv, aug_cards_dir, aug_masks_dir]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing classifier training input: {path}")
    _assert_no_test_inputs(required)

    device = _select_device()
    batch_size = _select_batch_size(device, cfg)
    use_amp = device.type == "cuda"
    augmented_card_samples, missing_aug, skipped_aug_labels, skipped_aug_files, skipped_aug_masks = load_augmented_card_samples(aug_csv, aug_cards_dir, aug_masks_dir, project_root)
    scene_manual_samples, missing_scene, skipped_scene_labels, skipped_scene_boxes, skipped_scene_masks = load_scene_manual_samples(scene_labels_path, scene_images_dir, scene_masks_dir, project_root)
    if not augmented_card_samples:
        raise RuntimeError("No augmented-card samples loaded.")
    if not scene_manual_samples:
        raise RuntimeError("No scene-manual samples loaded.")

    unique_scene_paths = sorted({sample.image_path for sample in scene_manual_samples})
    if cfg.max_loaded_scene_images is not None and len(unique_scene_paths) > cfg.max_loaded_scene_images:
        unique_scene_paths = _deterministic_path_subset(unique_scene_paths, int(cfg.max_loaded_scene_images), cfg.seed + 13_101)
        keep = {path.resolve() for path in unique_scene_paths}
        scene_manual_samples = [sample for sample in scene_manual_samples if sample.image_path.resolve() in keep]
    if len(unique_scene_paths) < 2:
        raise RuntimeError("Need at least two unique scene images for classifier train/val split.")
    scene_train_paths, scene_val_paths = train_test_split(unique_scene_paths, test_size=cfg.val_split, random_state=cfg.seed)
    scene_val_set = set(scene_val_paths)
    scene_train_samples = [sample for sample in scene_manual_samples if sample.image_path not in scene_val_set]
    scene_val_samples = [sample for sample in scene_manual_samples if sample.image_path in scene_val_set]

    all_labels = sorted({sample.label for sample in augmented_card_samples + scene_manual_samples})
    label_to_index = {label: index for index, label in enumerate(all_labels)}
    pin_memory = device.type == "cuda"
    if cfg.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}")
    effective_workers = int(cfg.num_workers)
    common = dict(
        label_to_index=label_to_index,
        batch_size=batch_size,
        image_size=cfg.img_size,
        bbox_margin=cfg.bbox_margin,
        mask_threshold=cfg.mask_threshold,
        pin_memory=pin_memory,
        num_workers=effective_workers,
        persistent_workers=cfg.persistent_workers,
        cache_scene_assets=cfg.cache_scene_assets,
        cache_max_scene_assets=cfg.cache_max_scene_assets,
        seed=cfg.seed,
    )
    aug_per_epoch = _stage_samples_per_epoch(len(augmented_card_samples), "augmented_card", batch_size, device, cfg)
    scene_manual_per_epoch = _stage_samples_per_epoch(len(scene_train_samples), "scene_manual", batch_size, device, cfg)

    val_loader, _ = make_loader(samples=scene_val_samples, shuffle=False, augment=False, **common)
    model = CardResNet18SmallClassifier(
        n_classes=len(all_labels),
        input_channels=4,
        dropout=cfg.classifier_dropout,
        stem_width=cfg.classifier_stem_width,
    ).to(device)
    model_params = assert_param_cap(model, f"CardClassifier[{cfg.classifier_architecture}]")
    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print(
        "Classifier data: "
        f"augmented={len(augmented_card_samples):,} (per_epoch={aug_per_epoch:,}), "
        f"scene_train={len(scene_train_samples):,} (per_epoch={scene_manual_per_epoch:,}), "
        f"scene_val={len(scene_val_samples):,}, classes={len(all_labels):,}."
    )
    print(
        "Classifier loader: "
        f"batch_size={batch_size}, val_batches={len(val_loader):,}, workers={effective_workers}, "
        f"persistent={bool(cfg.persistent_workers and effective_workers > 0)}, pin_memory={pin_memory}."
    )
    print(f"Classifier device: {device} | checkpoint: {classifier_bundle_dir / 'card_classifier.pth'}")
    print(
        "Missing/skipped: "
        f"aug_missing={len(missing_aug)}, scene_missing={len(missing_scene)} | "
        f"aug_skip labels/files/masks={skipped_aug_labels}/{skipped_aug_files}/{skipped_aug_masks} | "
        f"scene_skip labels/boxes/masks={skipped_scene_labels}/{skipped_scene_boxes}/{skipped_scene_masks}"
    )
    return {
        "config": cfg,
        "project_root": project_root,
        "training_data": training_data,
        "models_dir": models_dir,
        "classifier_bundle_dir": classifier_bundle_dir,
        "device": device,
        "use_amp": use_amp,
        "batch_size": batch_size,
        "num_workers": effective_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(cfg.persistent_workers and effective_workers > 0),
        "cache_scene_assets": cfg.cache_scene_assets,
        "cache_max_scene_assets": cfg.cache_max_scene_assets,
        "card_model_path": classifier_bundle_dir / "card_classifier.pth",
        "card_classes_path": classifier_bundle_dir / "classes.npy",
        "card_config_path": classifier_bundle_dir / "config.json",
        "augmented_card_samples": augmented_card_samples,
        "scene_manual_samples": scene_manual_samples,
        "scene_train_samples": scene_train_samples,
        "scene_val_samples": scene_val_samples,
        "class_names": all_labels,
        "label_to_index": label_to_index,
        "augmented_samples_per_epoch": aug_per_epoch,
        "scene_manual_samples_per_epoch": scene_manual_per_epoch,
        "val_loader": val_loader,
        "model": model,
        "model_params": model_params,
        "ce_loss": ce_loss,
        "scaler": scaler,
        "history": [],
    }


def _selection_score_from_metrics(metrics: dict[str, float], cfg: ClassifierPipelineConfig) -> float:
    metric = cfg.best_epoch_selection_metric.strip().lower()
    if metric in {"val_acc", "acc", "accuracy"}:
        return float(metrics["cls_acc"])
    if metric in {"macro_f1", "f1"}:
        return float(metrics["macro_f1"])
    if metric in {"balanced_acc", "balanced_accuracy"}:
        return float(metrics["balanced_acc"])
    if metric in {"top3", "top3_acc"}:
        return float(metrics["top3_acc"])
    loss_score = 1.0 / (1.0 + max(0.0, float(metrics["loss"])))
    return (
        cfg.selection_weight_val_acc * float(metrics["cls_acc"])
        + cfg.selection_weight_macro_f1 * float(metrics["macro_f1"])
        + cfg.selection_weight_balanced_acc * float(metrics["balanced_acc"])
        + cfg.selection_weight_top3_acc * float(metrics["top3_acc"])
        + cfg.selection_weight_val_loss * loss_score
    )


def _classification_metrics(loss_sum: float, n_samples: int, true_labels: list[int], pred_labels: list[int], top3_hits: int, confidences: list[float]) -> dict[str, float]:
    if n_samples == 0:
        return {key: 0.0 for key in ("loss", "cls_acc", "top3_acc", "mean_confidence", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "balanced_acc")}
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
    return {
        "loss": loss_sum / n_samples,
        "cls_acc": float(np.mean(np.array(true_labels) == np.array(pred_labels))),
        "top3_acc": top3_hits / n_samples,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "balanced_acc": float(balanced_accuracy_score(true_labels, pred_labels)) if len(set(true_labels)) > 1 else 0.0,
    }


def _train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, ce_loss: nn.Module, scaler: torch.amp.GradScaler | None, use_amp: bool, device: torch.device, cfg: ClassifierPipelineConfig) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    top3_hits = 0
    confidences: list[float] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, targets)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        probs = torch.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
        batch_size = int(targets.shape[0])
        loss_sum += float(loss.item()) * batch_size
        true_labels.extend(targets.cpu().tolist())
        pred_labels.extend(pred.cpu().tolist())
        top3_hits += int((topk == targets[:, None]).any(dim=1).sum().item())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return _classification_metrics(loss_sum, len(true_labels), true_labels, pred_labels, top3_hits, confidences)


@torch.no_grad()
def _evaluate_one_epoch(model: nn.Module, loader: DataLoader, ce_loss: nn.Module, use_amp: bool, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    top3_hits = 0
    confidences: list[float] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
        batch_size = int(targets.shape[0])
        loss_sum += float(loss.item()) * batch_size
        true_labels.extend(targets.cpu().tolist())
        pred_labels.extend(pred.cpu().tolist())
        top3_hits += int((topk == targets[:, None]).any(dim=1).sum().item())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return _classification_metrics(loss_sum, len(true_labels), true_labels, pred_labels, top3_hits, confidences)


def run_training(state: dict[str, Any]) -> dict[str, Any]:
    """Run the notebook's staged classifier curriculum."""
    cfg: ClassifierPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    ce_loss: nn.Module = state["ce_loss"]
    device: torch.device = state["device"]
    use_amp: bool = state["use_amp"]
    stage_plan = [
        ("stage1_augmented_cards", cfg.stage_1_epochs, state["augmented_card_samples"], state["augmented_samples_per_epoch"], cfg.stage_1_lr),
        ("stage2_scene_manual_masks", cfg.stage_2_epochs, state["scene_train_samples"], state["scene_manual_samples_per_epoch"], cfg.stage_2_lr),
    ]
    history: list[dict[str, float | int | str]] = []
    last_stage_best_score = -float("inf")
    last_stage_best_acc = -1.0
    last_stage_best_state: dict[str, torch.Tensor] | None = None
    last_stage_name: str | None = None
    last_stage_best_epoch = -1
    global_start = time.perf_counter()
    for stage_index, (stage_name, stage_epochs, stage_samples, stage_samples_per_epoch, stage_lr) in enumerate(stage_plan, start=1):
        if stage_epochs <= 0 or len(stage_samples) == 0:
            print(f"[skip] {stage_name}: no epochs or no samples")
            continue
        samples_per_epoch = int(min(stage_samples_per_epoch, len(stage_samples)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        stage_best_score = -float("inf")
        epochs_without_improvement = 0
        print(
            f"\n[{stage_name}] epochs={stage_epochs}, lr={stage_lr:.2e}, "
            f"dataset_samples={len(stage_samples):,}, samples_per_epoch={samples_per_epoch:,}, "
            f"batch_size={state['batch_size']}, workers={state.get('num_workers', cfg.num_workers)}"
        )
        if samples_per_epoch < len(stage_samples):
            print("  per-epoch cap active: one persistent loader cycles through the full stage sample pool")

        reusable_loader: DataLoader | None = None
        if samples_per_epoch < len(stage_samples):
            reusable_loader, _ = make_rotating_loader(
                stage_samples,
                state["label_to_index"],
                state["batch_size"],
                cfg.img_size,
                cfg.bbox_margin,
                cfg.mask_threshold,
                augment=True,
                pin_memory=device.type == "cuda",
                num_workers=state.get("num_workers", cfg.num_workers),
                seed=cfg.seed + stage_index * 10_000,
                samples_per_epoch=samples_per_epoch,
                persistent_workers=cfg.persistent_workers,
                cache_scene_assets=cfg.cache_scene_assets,
                cache_max_scene_assets=cfg.cache_max_scene_assets,
            )
        else:
            reusable_loader, _ = make_loader(
                stage_samples,
                state["label_to_index"],
                state["batch_size"],
                cfg.img_size,
                cfg.bbox_margin,
                cfg.mask_threshold,
                shuffle=True,
                augment=True,
                pin_memory=device.type == "cuda",
                num_workers=state.get("num_workers", cfg.num_workers),
                seed=cfg.seed + stage_index * 10_000,
                balanced=cfg.balanced_sampling,
                samples_per_epoch=samples_per_epoch,
                persistent_workers=cfg.persistent_workers,
                cache_scene_assets=cfg.cache_scene_assets,
                cache_max_scene_assets=cfg.cache_max_scene_assets,
            )
        for epoch in range(1, stage_epochs + 1):
            loader = reusable_loader
            epoch_start = time.perf_counter()
            try:
                train_metrics = _train_one_epoch(model, loader, optimizer, ce_loss, state["scaler"], use_amp, device, cfg)
            finally:
                if loader is not reusable_loader:
                    _shutdown_loader_workers(loader)
            val_metrics = _evaluate_one_epoch(model, state["val_loader"], ce_loss, use_amp, device)
            score = _selection_score_from_metrics(val_metrics, cfg)
            scheduler.step(score)
            epoch_seconds = time.perf_counter() - epoch_start
            if last_stage_name != stage_name:
                last_stage_name = stage_name
                last_stage_best_score = -float("inf")
                last_stage_best_acc = -1.0
            is_best = score > last_stage_best_score
            if is_best:
                last_stage_best_score = float(score)
                last_stage_best_acc = float(val_metrics["cls_acc"])
                last_stage_best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                last_stage_best_epoch = epoch
            if score > stage_best_score + 1e-8:
                stage_best_score = float(score)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_cls_acc": float(train_metrics["cls_acc"]),
                "val_loss": float(val_metrics["loss"]),
                "val_cls_acc": float(val_metrics["cls_acc"]),
                "val_top3_acc": float(val_metrics["top3_acc"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_balanced_acc": float(val_metrics["balanced_acc"]),
                "selection_score": float(score),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(epoch_seconds),
                "samples_per_epoch": int(samples_per_epoch),
            }
            history.append(row)
            print(
                f"  epoch {epoch:03d}/{stage_epochs} | train_acc={train_metrics['cls_acc']*100:5.1f}% loss={train_metrics['loss']:.3f} | "
                f"val_acc={val_metrics['cls_acc']*100:5.1f}% top3={val_metrics['top3_acc']*100:5.1f}% "
                f"macro_f1={val_metrics['macro_f1']:.3f} bal_acc={val_metrics['balanced_acc']:.3f} | "
                f"selection({cfg.best_epoch_selection_metric})={score:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} time={_format_seconds(epoch_seconds)}"
                + (" | stage-best" if is_best else "")
            )
            if stage_index == len(stage_plan) and epoch >= cfg.min_epochs_per_stage and epochs_without_improvement >= cfg.early_stop_patience:
                print(f"  [early-stop] {stage_name}: validation plateaued for {cfg.early_stop_patience} epochs")
                break
        _shutdown_loader_workers(reusable_loader)
    if last_stage_best_state is None:
        raise RuntimeError("Classifier training produced no checkpoint; all stages were disabled or empty.")
    model.load_state_dict(last_stage_best_state)
    state["history"] = history
    state["best_val_acc"] = last_stage_best_acc
    state["best_selection_score"] = last_stage_best_score
    state["best_selection_metric"] = cfg.best_epoch_selection_metric
    state["best_stage"] = last_stage_name
    state["best_epoch"] = last_stage_best_epoch
    state["training_seconds"] = time.perf_counter() - global_start
    _shutdown_loader_workers(state.get("val_loader"))
    print("\nClassifier training complete.")
    print(
        f"Selected {last_stage_name} epoch {last_stage_best_epoch} | "
        f"val_acc={last_stage_best_acc*100:.2f}% | "
        f"selection({cfg.best_epoch_selection_metric})={last_stage_best_score:.4f} | "
        f"total_time={_format_seconds(state['training_seconds'])}"
    )
    return state


def _stage_plan_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    cfg: ClassifierPipelineConfig = state["config"]
    entries = [
        ("stage1_augmented_cards", cfg.stage_1_epochs, cfg.stage_1_lr, state["augmented_card_samples"], state["augmented_samples_per_epoch"]),
        ("stage2_scene_manual_masks", cfg.stage_2_epochs, cfg.stage_2_lr, state["scene_train_samples"], state["scene_manual_samples_per_epoch"]),
    ]
    return [
        {
            "stage": name,
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "n_samples": int(len(samples)),
            "samples_per_epoch": int(min(samples_per_epoch, len(samples))) if samples else 0,
        }
        for name, epochs, lr, samples, samples_per_epoch in entries
    ]


def save_training_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Save the trained classifier bundle consumed by src.inference.load_engine."""
    cfg: ClassifierPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    card_model_path: Path = state["card_model_path"]
    card_classes_path: Path = state["card_classes_path"]
    card_config_path: Path = state["card_config_path"]
    card_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model.__class__.__name__,
            "n_classes": len(state["class_names"]),
            "img_size": cfg.img_size,
            "input_channels": 4,
            "architecture": cfg.classifier_architecture,
            "stem_width": cfg.classifier_stem_width,
            "dropout": cfg.classifier_dropout,
        },
        card_model_path,
    )
    np.save(card_classes_path, np.array(state["class_names"]))
    config = {
        "model_file": card_model_path.name,
        "classes_file": card_classes_path.name,
        "image_size": cfg.img_size,
        "normalization_mean": IMAGENET_MEAN.tolist(),
        "normalization_std": IMAGENET_STD.tolist(),
        "bbox_margin": cfg.bbox_margin,
        "mask_threshold": cfg.mask_threshold,
        "input_mode": "rgb_plus_mask_channel",
        "masked_background_fill": 128,
        "augmented_cards_use_saved_masks": True,
        "architecture": cfg.classifier_architecture,
        "classifier_stem_width": cfg.classifier_stem_width,
        "classifier_dropout": cfg.classifier_dropout,
        "pretrained": False,
        "optimizer": "AdamW",
        "weight_decay": cfg.weight_decay,
        "batch_size": state["batch_size"],
        "device": str(state["device"]),
        "balanced_sampling": cfg.balanced_sampling,
        "early_stop_patience": cfg.early_stop_patience,
        "min_epochs_per_stage": cfg.min_epochs_per_stage,
        "max_loaded_scene_images": cfg.max_loaded_scene_images,
        "best_selection_policy": "last_stage_best_by_selection_metric",
        "best_selection_metric": state.get("best_selection_metric") or cfg.best_epoch_selection_metric,
        "best_selection_score": float(state.get("best_selection_score") or 0.0),
        "selection_weight_val_acc": cfg.selection_weight_val_acc,
        "selection_weight_macro_f1": cfg.selection_weight_macro_f1,
        "selection_weight_balanced_acc": cfg.selection_weight_balanced_acc,
        "selection_weight_top3_acc": cfg.selection_weight_top3_acc,
        "selection_weight_val_loss": cfg.selection_weight_val_loss,
        "grad_clip_norm": cfg.grad_clip_norm,
        "stage_plan": _stage_plan_summary(state),
        "n_classes": len(state["class_names"]),
        "class_names": [str(name) for name in state["class_names"]],
        "validation_samples": len(state["scene_val_samples"]),
        "best_val_accuracy": float(state.get("best_val_acc") or 0.0),
        "best_stage": state.get("best_stage"),
        "best_epoch": int(state.get("best_epoch") or -1),
        "trainable_params": int(state.get("model_params") or 0),
        "seed": cfg.seed,
    }
    card_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    state["saved_config"] = config
    print(f"Saved model checkpoint: {card_model_path}")
    print(f"Saved class names: {card_classes_path}")
    print(f"Saved config: {card_config_path}")
    return state


def run_full_classifier_training(
    config: ClassifierPipelineConfig | None = None,
    project_root: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    state = initialize_training_pipeline(config, project_root=project_root, models_dir=models_dir)
    state = run_training(state)
    return save_training_artifacts(state)


def plot_stage_preview(state: dict[str, Any], samples_per_stage: int = 3, seed: int = 42) -> None:
    rng = random.Random(seed)
    stages = [
        ("augmented", state.get("augmented_card_samples", [])),
        ("scene manual", state.get("scene_train_samples", [])),
    ]
    selected: list[tuple[str, CardSample]] = []
    for label, samples in stages:
        if samples:
            for sample in rng.sample(samples, k=min(samples_per_stage, len(samples))):
                selected.append((label, sample))
    if not selected:
        print("No classifier samples to preview.")
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3 * len(selected)), squeeze=False)
    cfg: ClassifierPipelineConfig = state["config"]
    for row_index, (stage_label, sample) in enumerate(selected):
        image_crop, mask_crop = sample_to_crop_and_mask(sample, cfg.bbox_margin, cfg.mask_threshold)
        image_lb, mask_lb = letterbox_image_and_mask(image_crop, mask_crop, cfg.img_size)
        masked = compose_masked_card_image(image_lb, mask_lb)
        axes[row_index, 0].imshow(cv2.cvtColor(image_lb, cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title(f"{stage_label}: {sample.label}")
        axes[row_index, 1].imshow(mask_lb, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title("mask")
        axes[row_index, 2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[row_index, 2].set_title("4-channel RGB view")
        for axis in axes[row_index]:
            axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_classifier_training_curves(state: dict[str, Any]) -> None:
    history = state.get("history", [])
    if not history:
        print("No classifier training history to plot.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(1, len(history) + 1)
    axes[0].plot(x, [row["train_loss"] for row in history], label="train")
    axes[0].plot(x, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(x, [row["train_cls_acc"] for row in history], label="train")
    axes[1].plot(x, [row["val_cls_acc"] for row in history], label="val")
    axes[1].plot(x, [row["val_top3_acc"] for row in history], label="val top3")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[2].plot(x, [row["selection_score"] for row in history])
    axes[2].set_title("Selection score")
    plt.tight_layout()
    plt.show()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _training_path(project_root: Path, stored_path: str | Path) -> Path:
    parts = Path(str(stored_path).replace("\\", "/")).parts
    if "training_data" in parts:
        return project_root.joinpath(*parts[parts.index("training_data"):])
    return project_root / Path(*parts)


def _card_image_path(project_root: Path, image_id: str, stored_path: str | None = None) -> Path:
    if stored_path:
        candidate = _training_path(project_root, stored_path)
        if candidate.is_file():
            return candidate
    image_dir = project_root / "training_data" / "augmented_data" / "card_images"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No augmented-card image found for {image_id}")


def summarize_classifier_training(models_dir: Path, project_root: Path) -> dict[str, Any]:
    """Read the saved classifier config and generated-label distribution."""
    bundle = resolve_classifier_bundle(Path(models_dir))
    config = json.loads(Path(bundle["config_path"]).read_text(encoding="utf-8"))
    labels_dir = Path(project_root) / "training_data" / "object_labels"
    augmented_rows = _read_csv(labels_dir / "augmented_cards.csv")
    reference_rows = _read_csv(labels_dir / "reference_cards.csv")
    class_names = [str(name) for name in config.get("class_names", [])]

    return {
        "bundle": bundle,
        "config": config,
        "stage_plan": config.get("stage_plan", []),
        "class_names": class_names,
        "augmented_class_counts": Counter(row["card"] for row in augmented_rows),
        "reference_class_counts": Counter(row["card"] for row in reference_rows),
        "augmented_rows": augmented_rows,
    }


def print_classifier_summary(summary: dict[str, Any]) -> None:
    config = summary["config"]
    print(
        "Classifier: "
        f"{config.get('architecture')} ({config.get('input_mode')}), "
        f"{config.get('n_classes')} classes, "
        f"{int(config.get('trainable_params', 0)):,} params."
    )
    print(
        "Checkpoint: "
        f"val_acc={float(config.get('best_val_accuracy', 0.0)):.3f}, "
        f"selection={float(config.get('best_selection_score', 0.0)):.3f}, "
        f"best={config.get('best_stage')}/epoch {config.get('best_epoch')}, "
        f"val_samples={int(config.get('validation_samples', 0)):,}."
    )


def _annotate_bars(axis: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in axis.patches:
        height = float(patch.get_height())
        if height <= 0:
            continue
        axis.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )


def plot_classifier_training_summary(summary: dict[str, Any]) -> None:
    config = summary["config"]
    stage_plan = summary["stage_plan"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    stage_labels = [str(stage["stage"]).replace("stage", "s") for stage in stage_plan]
    axes[0].bar(stage_labels, [int(stage.get("n_samples", 0)) for stage in stage_plan], color="#4e79a7")
    axes[0].set_title("Training samples by stage")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Samples")

    axes[1].bar(stage_labels, [int(stage.get("epochs", 0)) for stage in stage_plan], color="#59a14f")
    axes[1].set_title("Epoch budget by stage")
    axes[1].tick_params(axis="x", rotation=25)

    metric_names = ["val_acc", "selection"]
    metric_values = [
        float(config.get("best_val_accuracy", 0.0)),
        float(config.get("best_selection_score", 0.0)),
    ]
    axes[2].bar(metric_names, metric_values, color="#f28e2b")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Saved checkpoint metrics")

    plt.tight_layout()
    plt.show()


def plot_classifier_input_examples(project_root: Path, summary: dict[str, Any], sample_count: int = 4, seed: int = 42) -> None:
    rows = summary["augmented_rows"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=min(sample_count, len(rows)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3 * len(indices)), squeeze=False)

    for row_index, source_index in enumerate(indices):
        row = rows[int(source_index)]
        image_bgr = cv2.imread(str(_card_image_path(Path(project_root), row["image_id"], row.get("image_path"))), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(_training_path(Path(project_root), row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            for col_index in range(3):
                axes[row_index, col_index].set_visible(False)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masked_rgb = image_rgb.copy()
        masked_rgb[mask <= 0] = 128
        axes[row_index, 0].imshow(image_rgb)
        axes[row_index, 0].set_title(f"{row['card']} crop")
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title("Saved mask")
        axes[row_index, 2].imshow(masked_rgb)
        axes[row_index, 2].set_title("Classifier input RGB")
        for axis in axes[row_index]:
            axis.axis("off")

    plt.tight_layout()
    plt.show()
