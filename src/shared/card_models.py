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


class CardResidualBlock(nn.Module):
    """Basic residual block used by the compact card classifier."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CardResNet18SmallClassifier(nn.Module):
    """Compact residual classifier (~9.85M params with stem_width=60)."""

    def __init__(
        self,
        n_classes: int,
        input_channels: int = 4,
        dropout: float = 0.20,
        stem_width: int = 60,
    ):
        super().__init__()
        if stem_width <= 0:
            raise ValueError(f"stem_width must be > 0, got {stem_width}")

        c1, c2, c3, c4 = stem_width, stem_width * 2, stem_width * 4, stem_width * 8

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            CardResidualBlock(c1, c1, stride=1),
            CardResidualBlock(c1, c1, stride=1),
        )
        self.layer2 = nn.Sequential(
            CardResidualBlock(c1, c2, stride=2),
            CardResidualBlock(c2, c2, stride=1),
        )
        self.layer3 = nn.Sequential(
            CardResidualBlock(c2, c3, stride=2),
            CardResidualBlock(c3, c3, stride=1),
        )
        self.layer4 = nn.Sequential(
            CardResidualBlock(c3, c4, stride=2),
            CardResidualBlock(c4, c4, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c4, n_classes))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_card_classifier(
    n_classes: int,
    input_channels: int = 4,
    dropout: float = 0.20,
    architecture: str = "resnet18_small",
    stem_width: int = 60,
) -> tuple[nn.Module, str]:
    """Build a classifier by architecture name and return model + canonical name."""

    arch = str(architecture).strip().lower()
    if arch in {"resnet18_small", "custom_resnet18_small", "card_resnet18_small"}:
        return (
            CardResNet18SmallClassifier(
                n_classes=n_classes,
                input_channels=input_channels,
                dropout=dropout,
                stem_width=stem_width,
            ),
            "resnet18_small",
        )
    if arch in {"resnet18", "torchvision_resnet18", "card_resnet18"}:
        return (
            CardResNet18Classifier(
                n_classes=n_classes,
                input_channels=input_channels,
                dropout=dropout,
            ),
            "torchvision_resnet18",
        )

    raise ValueError(
        f"Unsupported classifier architecture '{architecture}'. "
        "Use one of: torchvision_resnet18, resnet18_small."
    )


def assert_param_cap(model: nn.Module, name: str, cap: int = 12_000_000) -> int:
    n = count_trainable_params(model)
    assert n <= cap, f"{name} has {n:,} params, exceeds {cap:,} cap"
    print(f"[compliance] {name}: {n:,} trainable params")
    return n
