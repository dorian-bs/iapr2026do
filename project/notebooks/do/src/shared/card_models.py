"""Shared model architectures for the masked card classifier pipeline.

Both the training and inference paths use the same nets, so they live here once
to prevent silent drift between train- and test-time architectures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SceneDoubleConv(nn.Module):
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


class SceneUNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = SceneDoubleConv(3, 32)
        self.enc2 = SceneDoubleConv(32, 64)
        self.enc3 = SceneDoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = SceneDoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = SceneDoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = SceneDoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = SceneDoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class CardResNet18Classifier(nn.Module):
    """ResNet-18 with a custom 4-channel input (RGB + binary mask).

    Compliance: torchvision architecture used with weights=None per R2.
    """

    def __init__(self, n_classes: int, input_channels: int = 4, dropout: float = 0.20):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_classes))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def assert_param_cap(model: nn.Module, name: str, cap: int = 12_000_000) -> int:
    n = count_trainable_params(model)
    assert n <= cap, f"{name} has {n:,} params, exceeds {cap:,} cap"
    print(f"[compliance] {name}: {n:,} trainable params")
    return n
