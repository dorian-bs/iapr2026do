from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uno_vision.paths import SEGMENTER_MODELS_DIR
from uno_vision.segmentation.data import SegDataset, collect_segmentation_pairs
from uno_vision.segmentation.model import UNetSmall, dice_loss_from_logits, iou_score_from_logits


@dataclass
class SegmentationTrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    train_ious: list[float]
    val_ious: list[float]
    model_path: Path


def train_segmenter(
    pairs: list[tuple[str, str]] | None = None,
    epochs: int = 25,
    batch_size: int = 8,
    random_state: int = 42,
    output_path: Path = SEGMENTER_MODELS_DIR / "segmenter_unet_small.pth",
) -> SegmentationTrainingHistory:
    pairs = pairs or collect_segmentation_pairs()
    if not pairs:
        raise RuntimeError("No valid image-mask pairs found.")
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=random_state, shuffle=True)
    train_ds = SegDataset(train_pairs, augment=True)
    val_ds = SegDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_ious: list[float] = []
    val_ious: list[float] = []
    for _ in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_iou += iou_score_from_logits(logits.detach(), masks) * imgs.size(0)
        train_loss /= len(train_ds)
        train_iou /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_iou += iou_score_from_logits(logits, masks) * imgs.size(0)
        val_loss /= len(val_ds)
        val_iou /= len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        scheduler.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return SegmentationTrainingHistory(train_losses, val_losses, train_ious, val_ious, output_path)