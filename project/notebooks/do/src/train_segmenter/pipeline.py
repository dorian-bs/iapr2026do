"""Training orchestrator for the scene segmenter (SceneUNetSmall).

Public surface mirrors the train_classifier_CNN pattern:
    SegmenterPipelineConfig
    initialize_segmenter_pipeline(config) -> state
    run_segmenter_training(state) -> state
    plot_training_curves(state)
    plot_segmentation_predictions(state)

The shared `SceneUNetSmall` is reused (R1: <=12M params asserted; R2: weights=None).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.shared.card_models import SceneUNetSmall, assert_param_cap
from src.shared.card_pipeline import IMAGENET_MEAN, IMAGENET_STD, find_workspace_root


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class SegmenterPipelineConfig:
    seed: int = 42
    image_size: int = 256
    val_split: float = 0.35
    max_scene_pairs: int | None = None
    epoch_max_train_samples: int | None = None
    cache_in_ram: bool = True
    num_workers: int = 4

    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_loss_bce_weight: float = 0.5
    train_loss_dice_weight: float = 0.5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    use_amp: bool = True
    use_torch_compile: bool = False
    early_stopping_patience: int | None = 8
    log_every_batches: int = 25

    warm_start: bool = False
    warm_start_filename: str = "segmenter_unet_small.pth"
    checkpoint_filename: str = "scene_segmenter_unet_small.pth"

    preview_count: int = 6


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def _letterbox_pil(img, target_size: int, fill: int, interpolation):
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), interpolation)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(img, padding, fill=fill)


def preprocess_scene_pair(img_path: Path, mask_path: Path, image_size: int = 256):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = _letterbox_pil(img, target_size=image_size, fill=255, interpolation=Image.BILINEAR)
    mask = _letterbox_pil(mask, target_size=image_size, fill=0, interpolation=Image.NEAREST)

    img_t = TF.to_tensor(img)
    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    mask_arr = (mask_arr > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

    return img_t, mask_t


def denormalize_image(img_t: torch.Tensor) -> np.ndarray:
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(img_np, 0, 1)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class SceneSegDataset(Dataset):
    def __init__(self, pairs, image_size: int = 256, augment: bool = False, cache_in_ram: bool = True):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache: list = []

        if self.cache_in_ram:
            print(f"  Pre-loading {len(self.pairs)} samples into RAM...", flush=True)
            for i, (img_path, mask_path) in enumerate(self.pairs, start=1):
                self.cache.append(preprocess_scene_pair(img_path, mask_path, image_size=self.image_size))
                if i % 200 == 0 or i == len(self.pairs):
                    print(f"  {i}/{len(self.pairs)} loaded", flush=True)
            print("  Done.", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache_in_ram:
            img_t, mask_t = self.cache[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            img_t, mask_t = preprocess_scene_pair(img_path, mask_path, image_size=self.image_size)

        if self.augment:
            if random.random() < 0.5:
                img_t = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t)
            if random.random() < 0.5:
                img_t = TF.vflip(img_t)
                mask_t = TF.vflip(mask_t)

        return img_t, mask_t


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + eps) / (probs.sum() + targets.sum() + eps)
    return 1.0 - dice


def iou_score_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def _format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    return f"{minutes:02d}:{rem:02d}"


# --------------------------------------------------------------------------- #
# Training/eval loops
# --------------------------------------------------------------------------- #


def _train_one_epoch_verbose(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    scaler,
    amp_enabled: bool,
    device: torch.device,
    log_every: int = 25,
    channels_last: bool = False,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    seen = 0
    num_batches = len(loader)
    loop_start = time.perf_counter()

    for batch_idx, (imgs, masks) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(imgs)
            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_iou += iou_score_from_logits(logits.detach(), masks) * bs
        seen += bs

        if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches):
            elapsed = time.perf_counter() - loop_start
            avg_batch_seconds = elapsed / batch_idx
            eta_seconds = avg_batch_seconds * (num_batches - batch_idx)
            avg_loss = running_loss / max(seen, 1)
            avg_iou = running_iou / max(seen, 1)
            print(
                f"    batch {batch_idx:03d}/{num_batches:03d} | "
                f"avg_loss={avg_loss:.4f} avg_iou={avg_iou:.4f} | "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )

    return running_loss / max(seen, 1), running_iou / max(seen, 1), seen


@torch.no_grad()
def _evaluate(model, loader, bce_loss, amp_enabled, device, bce_weight: float = 0.5, dice_weight: float = 0.5):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    seen = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(imgs)
            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = bce_weight * loss_bce + dice_weight * loss_dice

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_iou += iou_score_from_logits(logits, masks) * bs
        seen += bs

    return running_loss / max(seen, 1), running_iou / max(seen, 1), seen


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_pairs(scene_images_dir: Path, scene_masks_dir: Path):
    valid_img_ext = {".jpg", ".jpeg", ".png"}
    valid_mask_ext = {".png", ".jpg", ".jpeg"}

    mask_by_stem: dict[str, Path] = {}
    for mask_path in sorted(scene_masks_dir.iterdir()):
        if mask_path.suffix.lower() in valid_mask_ext:
            mask_by_stem[mask_path.stem] = mask_path

    scene_pairs: list[tuple[Path, Path]] = []
    missing_masks: list[str] = []
    for img_path in sorted(scene_images_dir.iterdir()):
        if img_path.suffix.lower() not in valid_img_ext:
            continue
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is not None and mask_path.is_file():
            scene_pairs.append((img_path, mask_path))
        else:
            missing_masks.append(img_path.name)

    return scene_pairs, missing_masks


def initialize_segmenter_pipeline(config: SegmenterPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or SegmenterPipelineConfig()
    _seed_everything(cfg.seed)

    project_root = find_workspace_root()
    training_root = project_root / "project" / "training_data"
    scene_images_dir = training_root / "training_images" / "augmented_scenes"
    scene_masks_dir = training_root / "training_masks" / "augmented_scenes"
    models_dir = project_root / "project" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not scene_images_dir.exists():
        raise FileNotFoundError(f"Scene images directory not found: {scene_images_dir}")
    if not scene_masks_dir.exists():
        raise FileNotFoundError(f"Scene masks directory not found: {scene_masks_dir}")

    # R3: training-only data — fail loudly if anyone points us at a "test" dir.
    for p in (scene_images_dir, scene_masks_dir):
        assert "test" not in str(p).lower(), f"Refusing to train on path containing 'test': {p}"

    device = _select_device()
    use_amp = cfg.use_amp and device.type == "cuda"

    print(f"Project root: {project_root}")
    print(f"Scene images: {scene_images_dir}")
    print(f"Scene masks: {scene_masks_dir}")
    print(f"Model output dir: {models_dir}")
    print(f"torch: {torch.__version__}")
    print(f"Device: {device}")

    # ---------------- pairs & split ----------------
    scene_pairs, missing_masks = _build_pairs(scene_images_dir, scene_masks_dir)
    print(f"Total scene pairs discovered: {len(scene_pairs)}")
    if missing_masks:
        print(f"Images without matching mask: {len(missing_masks)}")
        print("First missing examples:", missing_masks[:5])

    if len(scene_pairs) == 0:
        raise RuntimeError(
            "No augmented scene pairs found. Check training_images/augmented_scenes and "
            "training_masks/augmented_scenes."
        )

    if cfg.max_scene_pairs is not None:
        if cfg.max_scene_pairs <= 0:
            raise ValueError(f"max_scene_pairs must be > 0 or None, got {cfg.max_scene_pairs}")
        if len(scene_pairs) > cfg.max_scene_pairs:
            subset_rng = random.Random(cfg.seed)
            scene_pairs = subset_rng.sample(scene_pairs, k=cfg.max_scene_pairs)
            print(
                f"Using subset of augmented scenes: {len(scene_pairs)} pairs "
                f"(max_scene_pairs={cfg.max_scene_pairs})"
            )
        else:
            print(
                f"max_scene_pairs={cfg.max_scene_pairs} (no-op: dataset has fewer pairs)"
            )

    train_pairs, val_pairs = train_test_split(
        scene_pairs, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True,
    )
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ---------------- datasets / loaders ----------------
    train_ds = SceneSegDataset(train_pairs, image_size=cfg.image_size, augment=True, cache_in_ram=cfg.cache_in_ram)
    val_ds = SceneSegDataset(val_pairs, image_size=cfg.image_size, augment=False, cache_in_ram=cfg.cache_in_ram)

    is_windows = os.name == "nt"
    effective_workers = cfg.num_workers
    if cfg.cache_in_ram and effective_workers > 0:
        print("[info] cache_in_ram=True with num_workers>0 can stall on Windows; forcing num_workers=0")
        effective_workers = 0

    train_sampler = None
    epoch_train_samples = len(train_ds)
    if cfg.epoch_max_train_samples is not None:
        if cfg.epoch_max_train_samples <= 0:
            raise ValueError(
                f"epoch_max_train_samples must be > 0 or None, got {cfg.epoch_max_train_samples}"
            )
        epoch_train_samples = min(cfg.epoch_max_train_samples, len(train_ds))
        if epoch_train_samples < len(train_ds):
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(cfg.seed)
            train_sampler = RandomSampler(
                train_ds,
                replacement=False,
                num_samples=epoch_train_samples,
                generator=sampler_generator,
            )
            print(
                f"Per-epoch train cap enabled: {epoch_train_samples}/{len(train_ds)} samples "
                f"(epoch_max_train_samples={cfg.epoch_max_train_samples})"
            )
        else:
            print(
                f"epoch_max_train_samples={cfg.epoch_max_train_samples} "
                f"(no-op: train set is smaller)"
            )

    loader_kwargs: dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        num_workers=effective_workers,
        pin_memory=device.type == "cuda",
    )
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if train_sampler is None:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=False, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(
        f"Train samples: {len(train_ds)} (per-epoch: {epoch_train_samples}) | "
        f"Val samples: {len(val_ds)}"
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Loader workers in use: {effective_workers} (windows={is_windows})")

    # ---------------- model / optim ----------------
    model = SceneUNetSmall().to(device)
    channels_last = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    else:
        print("[warning] CUDA not available. Training can be significantly slower on CPU.")

    warm_start_checkpoint = models_dir / cfg.warm_start_filename
    if cfg.warm_start and warm_start_checkpoint.is_file():
        state_dict = torch.load(warm_start_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded warm-start checkpoint: {warm_start_checkpoint}")
    elif cfg.warm_start:
        print("[info] Warm-start checkpoint not found. Training from scratch.")

    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_params = assert_param_cap(model, "SceneUNetSmall")

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.scheduler_patience < 0:
        raise ValueError(f"scheduler_patience must be >= 0, got {cfg.scheduler_patience}")
    if not (0.0 < cfg.scheduler_factor <= 1.0):
        raise ValueError(f"scheduler_factor must be in (0, 1], got {cfg.scheduler_factor}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    scene_checkpoint = models_dir / cfg.checkpoint_filename
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Scene checkpoint: {scene_checkpoint}")

    return {
        "config": cfg,
        "project_root": project_root,
        "training_root": training_root,
        "scene_images_dir": scene_images_dir,
        "scene_masks_dir": scene_masks_dir,
        "models_dir": models_dir,
        "device": device,
        "use_amp": use_amp,
        "channels_last": channels_last,
        "scene_pairs": scene_pairs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "epoch_train_samples": epoch_train_samples,
        "train_loader": train_loader,
        "train_loader_kwargs": loader_kwargs,
        "val_loader": val_loader,
        "model": model,
        "model_params": n_params,
        "bce_loss": bce_loss,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "scene_checkpoint": scene_checkpoint,
        "history": {
            "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
            "lr": [], "epoch_seconds": [], "train_seconds": [], "val_seconds": [],
            "images_per_sec": [], "gpu_mem_gb": [],
        },
        "best_val_iou": None,
        "best_epoch": None,
        "training_seconds": None,
    }


# --------------------------------------------------------------------------- #
# Run training
# --------------------------------------------------------------------------- #


def run_segmenter_training(state: dict[str, Any]) -> dict[str, Any]:
    cfg: SegmenterPipelineConfig = state["config"]
    model: nn.Module = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    bce_loss: nn.Module = state["bce_loss"]
    use_amp: bool = state["use_amp"]
    device: torch.device = state["device"]
    channels_last: bool = state["channels_last"]
    train_loader: DataLoader = state["train_loader"]
    train_ds: SceneSegDataset = state["train_ds"]
    train_loader_kwargs: dict[str, Any] = state["train_loader_kwargs"]
    val_loader: DataLoader = state["val_loader"]
    epoch_train_samples: int = state.get("epoch_train_samples", len(train_loader.dataset))
    scene_checkpoint: Path = state["scene_checkpoint"]
    history: dict[str, list] = state["history"]

    loss_weight_sum = cfg.train_loss_bce_weight + cfg.train_loss_dice_weight
    if cfg.train_loss_bce_weight < 0 or cfg.train_loss_dice_weight < 0 or loss_weight_sum <= 0:
        raise ValueError(
            "train_loss_bce_weight and train_loss_dice_weight must be >= 0 "
            "and at least one must be > 0"
        )
    bce_weight = cfg.train_loss_bce_weight / loss_weight_sum
    dice_weight = cfg.train_loss_dice_weight / loss_weight_sum

    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_start = time.perf_counter()

    print(
        f"Starting training on {device} | train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} | epoch_train_samples={epoch_train_samples} "
        f"| loss_mix(bce/dice)=({bce_weight:.2f}/{dice_weight:.2f})",
        flush=True,
    )
    if epoch_train_samples < len(train_ds):
        print(
            "Per-epoch train cap is active: each epoch draws a fresh random subset "
            "from the full training set.",
            flush=True,
        )
    if device.type != "cuda":
        print("[warning] Running on CPU. It is normal if each epoch takes several minutes.", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[Epoch {epoch:02d}/{cfg.epochs}] starting...", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if epoch_train_samples < len(train_ds):
            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(cfg.seed + epoch)
            epoch_sampler = RandomSampler(
                train_ds,
                replacement=False,
                num_samples=epoch_train_samples,
                generator=epoch_generator,
            )
            train_loader = DataLoader(
                train_ds,
                shuffle=False,
                sampler=epoch_sampler,
                **train_loader_kwargs,
            )

        train_start = time.perf_counter()
        train_loss, train_iou, n_train = _train_one_epoch_verbose(
            model, train_loader, optimizer, bce_loss, scaler, use_amp, device,
            log_every=cfg.log_every_batches, channels_last=channels_last,
            bce_weight=bce_weight, dice_weight=dice_weight,
        )
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss, val_iou, _ = _evaluate(
            model, val_loader, bce_loss, use_amp, device,
            bce_weight=bce_weight, dice_weight=dice_weight,
        )
        val_seconds = time.perf_counter() - val_start

        scheduler.step(val_iou)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_seconds = train_seconds + val_seconds
        images_per_sec = n_train / max(train_seconds, 1e-6)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["lr"].append(lr_now)
        history["epoch_seconds"].append(epoch_seconds)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["images_per_sec"].append(images_per_sec)
        history["gpu_mem_gb"].append(gpu_mem_gb)

        improved = val_iou > best_val_iou
        if improved:
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), scene_checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        recent_times = history["epoch_seconds"][-5:]
        eta_seconds = float(np.mean(recent_times) * max(cfg.epochs - epoch, 0))

        if cfg.early_stopping_patience is None:
            status = "best checkpoint saved" if improved else "no improvement"
        else:
            status = (
                "best checkpoint saved"
                if improved
                else f"no improvement ({epochs_without_improvement}/{cfg.early_stopping_patience})"
            )

        print(
            f"  summary | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f} | lr={lr_now:.2e}",
            flush=True,
        )
        print(
            f"  times   | train={_format_seconds(train_seconds)} val={_format_seconds(val_seconds)} "
            f"epoch={_format_seconds(epoch_seconds)} | "
            f"speed={images_per_sec:.1f} img/s | gpu_mem={gpu_mem_gb:.2f} GB | "
            f"eta={_format_seconds(eta_seconds)} | {status}",
            flush=True,
        )

        if (
            cfg.early_stopping_patience is not None
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    total_seconds = time.perf_counter() - global_start
    print(f"\nTraining finished in {_format_seconds(total_seconds)}", flush=True)
    print(f"Best val IoU: {best_val_iou:.4f} at epoch {best_epoch}", flush=True)
    print(f"Saved best model: {scene_checkpoint}", flush=True)

    state["best_val_iou"] = float(best_val_iou)
    state["best_epoch"] = int(best_epoch)
    state["training_seconds"] = float(total_seconds)
    return state


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def plot_training_curves(state: dict[str, Any]) -> None:
    history = state["history"]
    if len(history["train_loss"]) == 0:
        raise RuntimeError("History is empty. Run the training cell first.")

    epochs_x = np.arange(1, len(history["train_loss"]) + 1)
    best_idx = int(np.argmax(history["val_iou"]))
    best_epoch_plot = int(epochs_x[best_idx])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs_x, history["train_loss"], marker="o", label="train")
    axes[0, 0].plot(epochs_x, history["val_loss"], marker="o", label="val")
    axes[0, 0].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_x, history["train_iou"], marker="o", label="train")
    axes[0, 1].plot(epochs_x, history["val_iou"], marker="o", label="val")
    axes[0, 1].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("IoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_x, history["lr"], marker="o", color="tab:green")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR (log scale)")

    if "train_seconds" in history and len(history["train_seconds"]) == len(epochs_x):
        train_seconds = np.array(history["train_seconds"])
        val_seconds = np.array(history["val_seconds"])
        axes[1, 1].bar(epochs_x, train_seconds, alpha=0.7, label="train seconds")
        axes[1, 1].bar(epochs_x, val_seconds, bottom=train_seconds, alpha=0.7, label="val seconds")
    else:
        axes[1, 1].bar(epochs_x, history["epoch_seconds"], alpha=0.6, label="epoch seconds")

    ax_speed = axes[1, 1].twinx()
    ax_speed.plot(epochs_x, history["images_per_sec"], color="tab:red", marker="o", label="images/sec")
    axes[1, 1].axvline(best_epoch_plot, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Epoch Time And Throughput")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Seconds")
    ax_speed.set_ylabel("Images/sec")

    handles_1, labels_1 = axes[1, 1].get_legend_handles_labels()
    handles_2, labels_2 = ax_speed.get_legend_handles_labels()
    axes[1, 1].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    plt.suptitle("Scene Segmenter Training Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Best epoch: {best_epoch_plot}")
    print(f"Best val IoU: {history['val_iou'][best_idx]:.4f}")
    print(f"Epoch time at best epoch: {history['epoch_seconds'][best_idx]:.2f}s")
    print(f"Mean throughput: {np.mean(history['images_per_sec']):.2f} img/s")


def plot_segmentation_predictions(state: dict[str, Any], num_show: int | None = None) -> None:
    model: nn.Module = state["model"]
    val_ds: SceneSegDataset = state["val_ds"]
    device: torch.device = state["device"]
    scene_checkpoint: Path = state["scene_checkpoint"]
    cfg: SegmenterPipelineConfig = state["config"]

    if scene_checkpoint.is_file():
        model.load_state_dict(torch.load(scene_checkpoint, map_location=device))
    model.eval()

    n_show = int(num_show or cfg.preview_count)
    n_show = min(n_show, len(val_ds))
    if n_show == 0:
        raise RuntimeError("Validation set is empty.")

    indices = np.linspace(0, len(val_ds) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 3, figsize=(11, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_t, mask_t = val_ds[int(idx)]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.float32)

            gt = mask_t[0].cpu().numpy()
            inter = np.logical_and(pred_bin > 0.5, gt > 0.5).sum()
            union = np.logical_or(pred_bin > 0.5, gt > 0.5).sum()
            sample_iou = (inter + 1e-6) / (union + 1e-6)

            axes[row, 0].imshow(denormalize_image(img_t))
            axes[row, 0].set_title("Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt, cmap="gray")
            axes[row, 1].set_title("GT Mask")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred_bin, cmap="gray")
            axes[row, 2].set_title(f"Pred Mask (IoU={sample_iou:.3f})")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()
