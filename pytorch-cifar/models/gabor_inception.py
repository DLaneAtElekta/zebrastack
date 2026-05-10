"""Canonical Inception V1 with Gabor-initialized 3x3 and 5x5 convs.

Step 1 of the V1 -> V4 (fixed Gabor bank) investigation. The Inception block
restores the original GoogLeNet 'with dim reduction' design --
parallel 1x1 | 1x1->3x3 | 1x1->5x5 | maxpool->1x1 -- so there is a real 5x5
weight tensor to initialize. (The CIFAR-tuned googlenet.py in this folder
factors the 5x5 branch into two stacked 3x3 convs, which leaves no 5x5 to
seed with a Gabor.)

The 3x3 and 5x5 spatial convs are filled with tiled Gabor kernels. With
freeze_gabor=True they match the 'V4 fixed bank' framing; the default
(False) treats Gabor as initialization only and lets SGD refine.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gabor bank construction (NumPy). Convention matches resnet18-vae/filter_utils.py:
# the complex carrier is exp(freq * (x sin theta + y cos theta) j), and the
# Gaussian envelope uses sigma = 1/freq.
# ---------------------------------------------------------------------------

def _meshgrid(sz: int):
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz),
        np.linspace(-(sz // 2), sz // 2, sz),
    )


def _gauss(xs, ys, sigma):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(xs * xs + ys * ys) / (2.0 * sigma * sigma)
    )


def _complex_gabor(xs, ys, freq, angle_rad):
    sigma = 1.0 / freq
    carrier = np.exp(freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j)
    return carrier * _gauss(xs, ys, sigma)


def gabor_bank(kernel_size: int, directions: int, frequencies: Iterable[float]) -> np.ndarray:
    """Build a real-valued Gabor bank.

    For each frequency: one Gaussian (DC) kernel plus (real, imag) pair per
    orientation. Returns array of shape (n_kernels, kernel_size, kernel_size).
    """
    xs, ys = _meshgrid(kernel_size)
    kernels = []
    for freq in frequencies:
        kernels.append(_gauss(xs, ys, 1.0 / freq))
        for d in range(directions):
            angle = d * np.pi / float(directions)
            g = _complex_gabor(xs, ys, freq, angle)
            kernels.append(np.real(g))
            kernels.append(np.imag(g))
    return np.stack(kernels, axis=0).astype(np.float32)


def _default_frequencies(kernel_size: int):
    if kernel_size <= 3:
        return [1.5]
    if kernel_size <= 5:
        return [1.0, 0.5]
    return [1.0, 0.5, 0.25]


def init_conv_with_gabor(
    conv: nn.Conv2d,
    directions: int = 4,
    frequencies: Optional[Iterable[float]] = None,
    freeze: bool = False,
) -> None:
    """Fill conv.weight (out, in, K, K) with tiled Gabor kernels.

    Each output channel gets one Gabor (cycling through the bank), replicated
    across all input channels and scaled by 1/in_channels so the response is
    the Gabor-filtered mean of the inputs.
    """
    out_c, in_c, kH, kW = conv.weight.shape
    if kH != kW:
        raise ValueError(f"expected square kernel, got {kH}x{kW}")
    K = kH

    if frequencies is None:
        frequencies = _default_frequencies(K)

    bank = gabor_bank(K, directions, frequencies)
    n_kernels = bank.shape[0]

    weight = np.zeros((out_c, in_c, K, K), dtype=np.float32)
    for o in range(out_c):
        weight[o, :, :, :] = bank[o % n_kernels] / float(in_c)

    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(weight))
        if conv.bias is not None:
            conv.bias.zero_()

    if freeze:
        conv.weight.requires_grad_(False)
        if conv.bias is not None:
            conv.bias.requires_grad_(False)


# ---------------------------------------------------------------------------
# Canonical Inception V1 block and GoogLeNet (channel widths match
# pytorch-cifar/models/googlenet.py for A/B comparison).
# ---------------------------------------------------------------------------

class Inception(nn.Module):
    def __init__(
        self,
        in_planes: int,
        n1x1: int, n3x3red: int, n3x3: int,
        n5x5red: int, n5x5: int, pool_planes: int,
        gabor_init: bool = True,
        freeze_gabor: bool = False,
    ):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        self.conv3 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            self.conv3,
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        self.conv5 = nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            self.conv5,
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

        if gabor_init:
            init_conv_with_gabor(self.conv3, freeze=freeze_gabor)
            init_conv_with_gabor(self.conv5, freeze=freeze_gabor)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GaborGoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        gabor_init: bool = True,
        freeze_gabor: bool = False,
    ):
        super().__init__()
        opts = dict(gabor_init=gabor_init, freeze_gabor=freeze_gabor)

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, **opts)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, **opts)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, **opts)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, **opts)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, **opts)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, **opts)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, **opts)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, **opts)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, **opts)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out); out = self.b3(out); out = self.maxpool(out)
        out = self.a4(out); out = self.b4(out); out = self.c4(out)
        out = self.d4(out); out = self.e4(out); out = self.maxpool(out)
        out = self.a5(out); out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def test():
    net = GaborGoogLeNet()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"trainable params: {trainable:,}  frozen params: {frozen:,}")

    net_frozen = GaborGoogLeNet(freeze_gabor=True)
    frozen2 = sum(p.numel() for p in net_frozen.parameters() if not p.requires_grad)
    print(f"freeze_gabor=True frozen params: {frozen2:,}")


if __name__ == "__main__":
    test()
