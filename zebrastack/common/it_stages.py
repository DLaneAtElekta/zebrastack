"""V1 -> V2 -> V4 -> PIT -> CIT -> AIT ventral-stream model.

Extends the recursive-filters V4 backbone with progressively coarser-
scale Inception-style stages modelling the inferotemporal hierarchy
(PIT, CIT, AIT). Each IT stage is a small 1x1 Conv2d "reduction"
(analogous to Inception's channel-reducing 1x1) followed by a
per-channel Gabor power bank at that stage's spatial scale, followed
by spatial average-pool downsampling. Final classification reads
from the spatially-pooled AIT representation.

The fixed Gabor + power primitive carries through every level, so the
whole stack is hierarchical bandpass filtering with learnable channel
mixing at each level -- the structural pattern Lane (1995) proposed
for V1 -> V4 carried through to IT, with 1x1 convs supplying the
cross-channel mixing that lets the model build compound features.

Inception parallel:
    Inception 1x1 reduce + 3x3/5x5 conv  ~   ITStage 1x1 reduce + Gabor power
    Inception max/avg pool branch        ~   ITStage spatial avgpool downsample
    Inception concat across branches     ~   per-channel Gabor expands feature dim
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v4_recursive_filters import (
    FilterBankSpec,
    GaborPowerBank,
    LieGroupCells,
    RecursiveFiltersV4,
    standard_orientations,
)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Computes a learned per-channel gating from a global summary of the
    feature map: globally average-pool, run through a small MLP
    (squeeze + excite), apply sigmoid, multiply by the original feature
    map. Biologically analogous to attentional modulation in V4/IT --
    cells whose preferred features are present in the global context
    are amplified, others suppressed.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(n_channels // reduction, 4)
        self.fc1 = nn.Conv2d(n_channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(-1, -2), keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ITStage(nn.Module):
    """One IT-level stage: 1x1 channel reduce + per-channel Gabor power + pool.

    Parameters mirror the V4 recursive-filter pattern: a small set of
    orientations and a couple of spatial frequencies appropriate to this
    level of the hierarchy. Channels-in get reduced via a 1x1 Conv2d so
    the per-channel Gabor expansion stays manageable.

    Output shape: (B, n_reduce * n_orient * n_freq, H/down, W/down).
    """

    def __init__(
        self,
        in_channels: int,
        n_reduce: int,
        gabor_orientations: Sequence[float],
        gabor_frequencies: Sequence[float],
        kernel_size: int,
        sigma_to_period: float = 0.5,
        downsample: int = 2,
        use_relu_on_reduce: bool = True,
        use_bn: bool = True,
        use_attention: bool = False,
        attention_reduction: int = 4,
    ):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, n_reduce, kernel_size=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(n_reduce) if use_bn else nn.Identity()
        self.use_relu_on_reduce = use_relu_on_reduce
        self.n_reduce = n_reduce
        self.se = SEBlock(n_reduce, reduction=attention_reduction) if use_attention else None

        spec = FilterBankSpec(
            orientations=tuple(gabor_orientations),
            frequencies=tuple(gabor_frequencies),
            kernel_size=kernel_size,
            sigma_to_period=sigma_to_period,
        )
        self.spec = spec
        self.bank = GaborPowerBank(spec)
        self.bn_post = nn.BatchNorm2d(n_reduce * self.bank.n_outputs) if use_bn else nn.Identity()
        self.downsample = downsample

    @property
    def n_outputs(self) -> int:
        return self.n_reduce * self.bank.n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.bn(x)
        if self.use_relu_on_reduce:
            x = F.relu(x)
        if self.se is not None:
            x = self.se(x)
        b, c, h, w = x.shape
        x_flat = x.reshape(b * c, 1, h, w)
        out = self.bank(x_flat).view(b, c * self.bank.n_outputs, h, w)
        out = self.bn_post(out)
        if self.downsample > 1:
            out = F.avg_pool2d(out, kernel_size=self.downsample, stride=self.downsample)
        return out


class V4Backbone(nn.Module):
    """Wraps RecursiveFiltersV4 + V2 Gabor and returns a flat feature map.

    Concatenates the V4 power maps (bandpass + DC slot) and V2 Gabor
    features into a single (B, C, H, W) tensor that downstream IT
    stages can consume. This is the V1 -> V2 -> V4 stack already
    described, repackaged as a feature extractor for the ventral stack.
    """

    def __init__(self, backbone: RecursiveFiltersV4):
        super().__init__()
        self.backbone = backbone

    @property
    def n_outputs(self) -> int:
        v4_channels = (
            self.backbone.v1_spec.n_frequencies
            * self.backbone.v1_spec.n_orientations
            * (self.backbone.recursive_octaves
               + (1 if self.backbone.use_v4_dc_channel else 0))
            * len(self.backbone.recursive_orientations)
        )
        v2_extra = (
            self.backbone.v2_gabor.n_outputs
            if self.backbone.v2_gabor is not None else 0
        )
        return v4_channels + v2_extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        flat_blocks = []
        for block in out["v4_power"]:
            b, n_init, n_rf, n_ro, h, w = block.shape
            flat_blocks.append(block.reshape(b, n_init * n_rf * n_ro, h, w))
        v4_flat = torch.cat(flat_blocks, dim=1)
        if "v2_gabor" in out:
            return torch.cat([v4_flat, out["v2_gabor"]], dim=1)
        return v4_flat


class FullVentralStream(nn.Module):
    """V1 -> V2 -> V4 -> PIT -> CIT -> AIT classifier.

    The V1>V2>V4 backbone is the existing recursive-filters model. PIT,
    CIT, and AIT each follow the same template: 1x1 reduce + per-channel
    Gabor power bank + spatial avgpool. The Gabor frequencies decrease
    by an octave per stage, so each stage operates at coarser scale than
    the last, mirroring the increasing receptive-field size along the
    ventral stream. AIT does global average pooling; a final 1x1 Conv2d
    + softmax classifier reads from the pooled AIT representation.
    """

    def __init__(
        self,
        v4_backbone: RecursiveFiltersV4,
        n_classes: int,
        pit_n_reduce: int = 16,
        cit_n_reduce: int = 16,
        ait_n_reduce: int = 16,
        pit_frequencies: Sequence[float] = (0.05, 0.025),
        cit_frequencies: Sequence[float] = (0.05, 0.025),
        ait_frequencies: Sequence[float] = (0.05, 0.025),
        n_orientations: int = 4,
        kernel_size: int = 21,
        downsample: int = 2,
        use_skip: bool = False,
        use_attention: bool = False,
        use_v4_mix: bool = False,
    ):
        super().__init__()
        self.backbone = V4Backbone(v4_backbone)
        self.use_skip = use_skip

        from .v4_recursive_filters import V2Mix as ChannelMix
        if use_v4_mix:
            self.v4_mix = ChannelMix(self.backbone.n_outputs, trainable=True)
        else:
            self.v4_mix = None

        orientations = standard_orientations(n_orientations)
        self.pit = ITStage(
            in_channels=self.backbone.n_outputs,
            n_reduce=pit_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=pit_frequencies,
            kernel_size=kernel_size,
            downsample=downsample,
            use_attention=use_attention,
        )
        self.cit = ITStage(
            in_channels=self.pit.n_outputs,
            n_reduce=cit_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=cit_frequencies,
            kernel_size=kernel_size,
            downsample=downsample,
            use_attention=use_attention,
        )
        self.ait = ITStage(
            in_channels=self.cit.n_outputs,
            n_reduce=ait_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=ait_frequencies,
            kernel_size=kernel_size,
            downsample=1,
            use_attention=use_attention,
        )

        self.input_norm = nn.BatchNorm2d(self.backbone.n_outputs)

        if use_skip:
            # Multi-scale skip: globally average-pool every stage,
            # BN-normalize per stage to balance magnitudes, concatenate,
            # feed to a single linear classifier.
            self.v4_pool_bn = nn.BatchNorm1d(self.backbone.n_outputs)
            self.pit_pool_bn = nn.BatchNorm1d(self.pit.n_outputs)
            self.cit_pool_bn = nn.BatchNorm1d(self.cit.n_outputs)
            self.ait_pool_bn = nn.BatchNorm1d(self.ait.n_outputs)
            total_dim = (
                self.backbone.n_outputs
                + self.pit.n_outputs
                + self.cit.n_outputs
                + self.ait.n_outputs
            )
            self.classifier = nn.Linear(total_dim, n_classes)
        else:
            self.classifier = nn.Conv2d(self.ait.n_outputs, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v4 = self.backbone(x)
        v4_norm = self.input_norm(v4)
        if self.v4_mix is not None:
            v4_norm = self.v4_mix(v4_norm)
        pit = self.pit(v4_norm)
        cit = self.cit(pit)
        ait = self.ait(cit)

        if self.use_skip:
            v4_pool = self.v4_pool_bn(v4.mean(dim=(-1, -2)))
            pit_pool = self.pit_pool_bn(pit.mean(dim=(-1, -2)))
            cit_pool = self.cit_pool_bn(cit.mean(dim=(-1, -2)))
            ait_pool = self.ait_pool_bn(ait.mean(dim=(-1, -2)))
            concat = torch.cat([v4_pool, pit_pool, cit_pool, ait_pool], dim=1)
            logits = self.classifier(concat)
            return {
                "v4": v4, "pit": pit, "cit": cit, "ait": ait,
                "v4_pool": v4_pool, "pit_pool": pit_pool,
                "cit_pool": cit_pool, "ait_pool": ait_pool,
                "logits": logits,
            }

        ait_pooled = ait.mean(dim=(-1, -2), keepdim=True)
        logits = self.classifier(ait_pooled).squeeze(-1).squeeze(-1)
        return {
            "v4": v4,
            "pit": pit,
            "cit": cit,
            "ait": ait,
            "ait_pooled": ait_pooled,
            "logits": logits,
        }

    @torch.no_grad()
    def recalibrate_bn(self, calibration_x: torch.Tensor, batch_size: int = 16):
        """Set BN running stats to cumulative averages over the training set.

        Switches every BatchNorm2d to cumulative-average mode
        (``momentum=None``) and runs one pass over the training data, so
        the running mean/var after the pass are the exact training-set
        statistics rather than a momentum-decayed approximation. Then
        restores the saved momenta and switches the model to eval mode.
        """
        saved = []
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                saved.append((m, m.momentum, m.training))
                m.momentum = None
                m.reset_running_stats()
                m.train()
        try:
            for i in range(0, calibration_x.shape[0], batch_size):
                _ = self(calibration_x[i:i+batch_size])
        finally:
            for m, momentum, was_training in saved:
                m.momentum = momentum
                if not was_training:
                    m.eval()
        self.eval()
