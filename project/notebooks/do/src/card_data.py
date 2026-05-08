"""Dataset assembly for the masked card classifier.

Responsibilities:
  * Load reference / augmented-card / scene-manual sample metadata.
  * Compose (image, mask) pairs from disk per stage.
  * Apply the training-time augmentation policy.
  * Build the predicted-mask cache from the frozen scene segmenter.

Compliance: per AGENT.md R3, no test-set paths are touched here. All sources
live under `project/training_data/...`; an explicit assertion enforces this.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from card_models import SceneUNetSmall, assert_param_cap
from card_pipeline import (
    crop_with_margin,
    is_reasonable_scene_bbox,
    letterbox_image_and_mask,
    compose_masked_card_image,
    card_input_to_tensor,
    segment_scene_probability,
)


@dataclass(frozen=True)
class CardSample:
    stage: str
    label: str
    image_path: Path
    bbox: tuple[int, int, int, int] | None = None
    scene_mask_path: Path | None = None


# Stages that feed solid-mask references (cards fill the whole crop).
SOLID_MASK_STAGES = {"reference", "augmented_card"}
SCENE_STAGES = {"scene_manual", "scene_predicted"}


def _assert_no_test_paths(paths: list[Path]) -> None:
    """R3: training data must not include any test-set directory."""
    for p in paths:
        s = str(p).lower()
        assert "test" not in s, f"R3 violation: training input references test path: {p}"


def reference_crop_path(image_id: str, ref_cards_dir: Path) -> Path:
    tag, separator, crop_index = image_id.rpartition("_crop_")
    if separator == "":
        raise ValueError(f"Unexpected reference image_id: {image_id}")
    return ref_cards_dir / tag / "crops" / f"crop_{crop_index}.jpg"


def load_reference_samples(reference_csv: Path, ref_cards_dir: Path) -> tuple[list[CardSample], list[str]]:
    samples: list[CardSample] = []
    missing: list[str] = []

    with reference_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row["image_id"]).strip()
            label = str(row["card"]).strip()
            if not image_id or not label:
                continue
            image_path = reference_crop_path(image_id, ref_cards_dir)
            if image_path.is_file():
                samples.append(CardSample("reference", label, image_path))
            else:
                missing.append(str(image_path))
    return samples, missing


def load_augmented_card_samples(
    aug_csv: Path,
    aug_cards_dir: Path,
    valid_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> tuple[list[CardSample], list[str], int, int]:
    samples: list[CardSample] = []
    missing: list[str] = []
    skipped_labels = 0
    skipped_files = 0

    with aug_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_id = str(row["image_id"]).strip()
            label = str(row["card"]).strip()

            if not image_id or not label or label == "token":
                skipped_labels += 1
                continue

            image_path: Path | None = None
            for suffix in valid_ext:
                candidate = aug_cards_dir / f"{image_id}{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break

            if image_path is None:
                skipped_files += 1
                missing.append(str(aug_cards_dir / f"{image_id}.jpg"))
                continue

            samples.append(CardSample("augmented_card", label, image_path))

    return samples, missing, skipped_labels, skipped_files


def load_scene_manual_samples(
    scene_labels_path: Path,
    scene_images_dir: Path,
    scene_masks_dir: Path,
    project_root: Path,
) -> tuple[list[CardSample], list[str], int, int, int]:
    samples: list[CardSample] = []
    missing: list[str] = []
    skipped_labels = 0
    skipped_boxes = 0
    skipped_masks = 0

    valid_mask_ext = {".png", ".jpg", ".jpeg"}
    scene_mask_by_stem = {
        mask_path.stem: mask_path
        for mask_path in sorted(scene_masks_dir.iterdir())
        if mask_path.suffix.lower() in valid_mask_ext
    }

    with scene_labels_path.open("r", encoding="utf-8") as labels_file:
        scene_metadata = json.load(labels_file)

    for entry in scene_metadata:
        scene_name = str(entry.get("scene", "")).strip()
        default_path = scene_images_dir / f"{scene_name}.jpg"
        raw_path = entry.get("image_path", default_path)
        path = Path(raw_path) if isinstance(raw_path, str) else default_path
        scene_path = path if path.is_absolute() else project_root / path

        if not scene_path.is_file():
            missing.append(str(scene_path))
            continue

        scene_mask_path = scene_mask_by_stem.get(scene_path.stem) or scene_mask_by_stem.get(scene_name)

        for card in entry.get("cards", []):
            label = str(card.get("label", "")).strip()
            bbox_raw = card.get("bbox", [])

            if not label or label == "token":
                skipped_labels += 1
                continue
            if len(bbox_raw) != 4:
                skipped_boxes += 1
                continue

            bbox = tuple(map(int, bbox_raw))
            if not is_reasonable_scene_bbox(bbox):
                skipped_boxes += 1
                continue

            if scene_mask_path is None or not scene_mask_path.is_file():
                skipped_masks += 1
                continue

            samples.append(
                CardSample(
                    stage="scene_manual",
                    label=label,
                    image_path=scene_path,
                    bbox=bbox,
                    scene_mask_path=scene_mask_path,
                )
            )

    return samples, missing, skipped_labels, skipped_boxes, skipped_masks


def derive_scene_predicted_samples(scene_train_samples: list[CardSample]) -> list[CardSample]:
    return [
        CardSample(
            stage="scene_predicted",
            label=s.label,
            image_path=s.image_path,
            bbox=s.bbox,
            scene_mask_path=None,
        )
        for s in scene_train_samples
    ]


def read_crop_image(sample: CardSample, bbox_margin: float) -> np.ndarray:
    img_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
    if sample.bbox is not None:
        img_bgr = crop_with_margin(img_bgr, sample.bbox, margin_fraction=bbox_margin)
    return img_bgr


def sample_to_crop_and_mask(
    sample: CardSample,
    bbox_margin: float,
    mask_threshold: float,
    predicted_scene_probs: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    img_crop = read_crop_image(sample, bbox_margin)

    if sample.stage in SOLID_MASK_STAGES:
        mask_crop = np.full(img_crop.shape[:2], 255, dtype=np.uint8)

    elif sample.stage == "scene_manual":
        if sample.scene_mask_path is None or sample.bbox is None:
            raise ValueError("scene_manual sample requires bbox and scene_mask_path.")
        scene_mask = cv2.imread(str(sample.scene_mask_path), cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(f"Cannot read scene mask: {sample.scene_mask_path}")
        mask_crop = crop_with_margin(scene_mask, sample.bbox, margin_fraction=bbox_margin)
        mask_crop = np.where(mask_crop > 127, 255, 0).astype(np.uint8)

    elif sample.stage == "scene_predicted":
        if predicted_scene_probs is None or sample.bbox is None:
            raise ValueError("scene_predicted sample requires bbox and predicted_scene_probs.")
        prob_map = predicted_scene_probs.get(str(sample.image_path.resolve()))
        if prob_map is None:
            raise KeyError(f"Missing predicted probability map: {sample.image_path}")
        prob_crop = crop_with_margin(prob_map, sample.bbox, margin_fraction=bbox_margin)
        mask_crop = np.where(prob_crop > mask_threshold, 255, 0).astype(np.uint8)

    else:
        raise ValueError(f"Unknown sample stage: {sample.stage}")

    if mask_crop.shape[:2] != img_crop.shape[:2]:
        mask_crop = cv2.resize(mask_crop, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    if int(np.count_nonzero(mask_crop)) < 20:
        # Mask collapsed: fall back to a solid mask so we at least classify the crop.
        mask_crop = np.full(img_crop.shape[:2], 255, dtype=np.uint8)

    return img_crop, mask_crop


def _warp_affine_pair(img_bgr, mask_u8, angle, scale, tx, ty):
    h, w = img_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    img_out = cv2.warpAffine(
        img_bgr, matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128),
    )
    mask_out = cv2.warpAffine(
        mask_u8, matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return img_out, mask_out


def _warp_perspective_pair(img_bgr, mask_u8, max_jitter_fraction=0.045):
    h, w = img_bgr.shape[:2]
    jitter = max_jitter_fraction * min(h, w)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.random.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    img_out = cv2.warpPerspective(
        img_bgr, matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128),
    )
    mask_out = cv2.warpPerspective(
        mask_u8, matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return img_out, mask_out


def augment_card_image_and_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Photometric+geometric augmentation. Falls back to the original pair if
    the augmented mask would collapse below 20 nonzero pixels."""
    original_img = img_bgr.copy()
    original_mask = mask_u8.copy()
    h, w = img_bgr.shape[:2]

    if random.random() < 0.85:
        base_angle = random.uniform(-12.0, 12.0)
        # Augmented and scene crops are arbitrarily oriented — allow 90° flips.
        if stage in {"augmented_card", "scene_manual", "scene_predicted"} and random.random() < 0.30:
            base_angle += random.choice([90.0, 180.0, 270.0])
        scale = random.uniform(0.92, 1.08)
        tx = random.uniform(-0.04, 0.04) * w
        ty = random.uniform(-0.04, 0.04) * h
        img_bgr, mask_u8 = _warp_affine_pair(img_bgr, mask_u8, base_angle, scale, tx, ty)

    if random.random() < 0.25:
        img_bgr, mask_u8 = _warp_perspective_pair(img_bgr, mask_u8)

    if random.random() < 0.65:
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-18.0, 18.0)
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.80, 1.25)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if random.random() < 0.15:
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

    if random.random() < 0.20:
        noise = np.random.normal(0.0, 4.0, size=img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Mask jitter only on noisy scene masks; reference masks are crisp by design.
    if stage in SCENE_STAGES and random.random() < 0.30:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if random.random() < 0.5:
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        else:
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    if int(np.count_nonzero(mask_u8)) < 20:
        return original_img, original_mask
    return img_bgr, mask_u8


class CardMaskedDataset(Dataset):
    def __init__(
        self,
        samples: list[CardSample],
        label_to_index: dict[str, int],
        image_size: int,
        bbox_margin: float,
        mask_threshold: float,
        augment: bool,
        predicted_scene_probs: dict[str, np.ndarray] | None,
    ):
        self.samples = samples
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.bbox_margin = bbox_margin
        self.mask_threshold = mask_threshold
        self.augment = augment
        self.predicted_scene_probs = predicted_scene_probs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_crop, mask_crop = sample_to_crop_and_mask(
            sample,
            bbox_margin=self.bbox_margin,
            mask_threshold=self.mask_threshold,
            predicted_scene_probs=self.predicted_scene_probs,
        )
        img_lb, mask_lb = letterbox_image_and_mask(img_crop, mask_crop, size=self.image_size)

        if self.augment:
            img_lb, mask_lb = augment_card_image_and_mask(img_lb, mask_lb, sample.stage)

        masked_img = compose_masked_card_image(img_lb, mask_lb)
        x = card_input_to_tensor(masked_img, mask_lb)
        target = torch.tensor(self.label_to_index[sample.label], dtype=torch.long)
        return x, target


def make_balanced_sampler(
    samples: list[CardSample],
    seed: int,
    samples_per_epoch: int | None = None,
) -> WeightedRandomSampler:
    label_counts = Counter(s.label for s in samples)
    weights = torch.DoubleTensor([1.0 / label_counts[s.label] for s in samples])
    n = len(samples) if samples_per_epoch is None else int(min(samples_per_epoch, len(samples)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=n, replacement=True, generator=generator)


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
    predicted_scene_probs: dict[str, np.ndarray] | None = None,
    balanced: bool = False,
    samples_per_epoch: int | None = None,
) -> tuple[DataLoader, CardMaskedDataset]:
    dataset = CardMaskedDataset(
        samples=samples,
        label_to_index=label_to_index,
        image_size=image_size,
        bbox_margin=bbox_margin,
        mask_threshold=mask_threshold,
        augment=augment,
        predicted_scene_probs=predicted_scene_probs,
    )
    sampler = (
        make_balanced_sampler(samples, seed=seed, samples_per_epoch=samples_per_epoch)
        if balanced and len(samples) > 0
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset


@torch.no_grad()
def build_scene_probability_cache(
    scene_paths: list[Path],
    segmenter_path: Path,
    device: torch.device,
    target_size: int = 256,
    progress_every: int = 100,
) -> dict[str, np.ndarray]:
    """Run the frozen scene segmenter once per unique scene and cache the
    full-resolution probability map keyed by resolved absolute path."""
    if not segmenter_path.is_file():
        raise FileNotFoundError(f"Missing scene segmenter checkpoint: {segmenter_path}")

    segmenter = SceneUNetSmall().to(device)
    assert_param_cap(segmenter, "SceneUNetSmall")
    segmenter.load_state_dict(torch.load(segmenter_path, map_location=device))
    segmenter.eval()

    cache: dict[str, np.ndarray] = {}
    print(f"Building predicted scene masks for {len(scene_paths)} scenes...")
    for idx, scene_path in enumerate(scene_paths, start=1):
        img_bgr = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read scene image: {scene_path}")
        prob = segment_scene_probability(img_bgr, segmenter, device, target_size=target_size)
        cache[str(scene_path.resolve())] = prob.astype(np.float32)
        if idx % progress_every == 0 or idx == len(scene_paths):
            print(f"  {idx}/{len(scene_paths)}")
    return cache


def assert_no_test_inputs(paths: list[Path]) -> None:
    """Public wrapper for R3 assertion."""
    _assert_no_test_paths(paths)
