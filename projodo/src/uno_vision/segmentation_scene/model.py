"""Small U-Net scene segmenter and overlap metrics used during training."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout the U-Net encoder and decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    """Compact U-Net for scene foreground segmentation under the 12M parameter cap."""

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder-decoder segmentation with skip connections."""

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters for compliance checks."""

    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def assert_parameter_budget(model: nn.Module, max_params: int = 12_000_000) -> int:
    """Assert and return the model parameter count under the competition limit."""

    params = count_trainable_parameters(model)
    assert params <= max_params, f"Model has {params:,} params, exceeds {max_params:,} limit"
    return params


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss from raw logits."""

    probs = torch.sigmoid(logits.float()).view(-1)
    targets = targets.float().view(-1)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + eps) / (probs.sum() + targets.sum() + eps)


def iou_sum_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Accumulate per-sample IoU values for epoch-level reporting."""

    preds = (torch.sigmoid(logits.float()) > thresh).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).sum()


def iou_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute mean IoU after thresholding sigmoid probabilities."""

    preds = (torch.sigmoid(logits.float()) > thresh).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())
