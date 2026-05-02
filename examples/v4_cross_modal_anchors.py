"""V4 features for cross-modality anchor matching.

Same image, different "modality" -- no spatial warp, just intensity
transforms that simulate the kinds of changes that occur between
imaging modalities (e.g. CT vs MRI, bright-field vs fluorescence).
The hypothesis is that the bandpass-Gabor V4 features should detect
the same anchor locations across modalities better than gradient-
based detectors that depend on absolute intensity.

Modality pairs tested:
  * identity (sanity)
  * intensity inversion         (negative image)
  * gamma 0.4 / gamma 2.5       (different brightness curves)
  * gradient-magnitude only     (edges, no interior intensity)
  * high-pass filtered          (low-freq structure removed)
  * additive Gaussian noise     (sensor noise)

Run:
    python examples/v4_cross_modal_anchors.py
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import struct
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.it_stages import V4Backbone
from zebrastack.common.v4_recursive_filters import (
    GaborPowerBank, RecursiveFiltersV4, standard_v1_spec,
)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        data = f.read(n * h * w)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, h, w).float() / 255.0


def harris_on_features(features, sigma=2.0, kappa=0.04):
    f = features
    fx = torch.zeros_like(f); fy = torch.zeros_like(f)
    fx[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
    fy[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
    Ixx = (fx * fx).sum(dim=1, keepdim=True)
    Ixy = (fx * fy).sum(dim=1, keepdim=True)
    Iyy = (fy * fy).sum(dim=1, keepdim=True)
    ksize = int(round(6 * sigma)) | 1
    half = ksize // 2
    grid = torch.arange(ksize, dtype=torch.float32) - half
    g1d = torch.exp(-(grid ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kx = g1d.view(1, 1, 1, ksize); ky = g1d.view(1, 1, ksize, 1)
    def smooth(t):
        t = F.pad(t, (half, half, half, half), mode="reflect")
        t = F.conv2d(t, kx); t = F.conv2d(t, ky); return t
    Ixx = smooth(Ixx); Ixy = smooth(Ixy); Iyy = smooth(Iyy)
    det = Ixx * Iyy - Ixy * Ixy
    tr = Ixx + Iyy
    return (det - kappa * tr * tr).squeeze(1)


def nms_topk(score, k, window=9):
    s = score.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(s, kernel_size=window, stride=1, padding=window // 2)
    is_max = (s == pooled).squeeze(0).squeeze(0)
    s_flat = score * is_max.float()
    flat = s_flat.view(-1)
    vals, idx = flat.topk(min(k, flat.numel()))
    h, w = score.shape
    ys = idx // w; xs = idx % w
    keep = vals > 0
    return torch.stack([ys[keep], xs[keep]], dim=-1)


def overlap_count(pts_a: torch.Tensor, pts_b: torch.Tensor, tol: float) -> int:
    """Count anchors in pts_a that have a match in pts_b within tol pixels."""
    if pts_a.numel() == 0 or pts_b.numel() == 0:
        return 0
    diff = pts_a.float().unsqueeze(1) - pts_b.float().unsqueeze(0)
    d = (diff ** 2).sum(dim=-1).sqrt()
    return int((d.min(dim=-1).values < tol).sum().item())


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    fx = torch.zeros_like(image); fy = torch.zeros_like(image)
    fx[..., :, 1:-1] = (image[..., :, 2:] - image[..., :, :-2]) * 0.5
    fy[..., 1:-1, :] = (image[..., 2:, :] - image[..., :-2, :]) * 0.5
    g = (fx * fx + fy * fy).sqrt()
    g = g / (g.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8))
    return (g - 0.5) * 2.0


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    ksize = int(round(6 * sigma)) | 1
    half = ksize // 2
    grid = torch.arange(ksize, dtype=torch.float32) - half
    g1d = torch.exp(-(grid ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kx = g1d.view(1, 1, 1, ksize); ky = g1d.view(1, 1, ksize, 1)
    x = F.pad(image, (half, half, half, half), mode="reflect")
    return F.conv2d(F.conv2d(x, kx), ky)


def make_modalities(image: torch.Tensor, rng: torch.Generator) -> dict[str, torch.Tensor]:
    """Same scene, different intensity transforms."""
    img01 = (image + 1.0) * 0.5  # back to [0, 1]
    out = {}
    out["identity"]    = image.clone()
    out["inverted"]    = (1.0 - img01) * 2.0 - 1.0
    out["gamma 0.4"]   = (img01.clamp(0, 1) ** 0.4) * 2.0 - 1.0
    out["gamma 2.5"]   = (img01.clamp(0, 1) ** 2.5) * 2.0 - 1.0
    out["edges only"]  = gradient_magnitude(image)
    blurred = gaussian_blur(image, sigma=3.0)
    hp = image - blurred
    out["high-pass"]   = hp / (hp.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8))
    noise = torch.randn(image.shape, generator=rng) * 0.4
    out["noisy"]       = (image + noise).clamp(-1.0, 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp")
    parser.add_argument("--out", default="examples/v4_cross_modal_anchors.png")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--nms-window", type=int, default=9)
    args = parser.parse_args()
    matplotlib.use("Agg")

    print("Loading Fashion-MNIST...")
    imgs = read_idx_images(os.path.join(args.data_dir, "fmnist-train-images-idx3-ubyte.gz"))
    rng = torch.Generator().manual_seed(7)
    idx = torch.randperm(imgs.shape[0], generator=rng)[: args.n_images]
    images = imgs[idx]
    images = F.interpolate(images.unsqueeze(1), size=(args.size, args.size),
                           mode="bilinear", align_corners=False)
    images = (images - 0.5) * 2.0

    print("Building feature extractors...")
    spec = standard_v1_spec(
        n_orientations=4, frequencies=(0.30, 0.18, 0.10), kernel_size=13,
    )
    v1_bank = GaborPowerBank(spec).eval()
    v4_sharp = V4Backbone(RecursiveFiltersV4(
        v1_spec=spec, recursive_octaves=2, recursive_kernel_size=21,
        use_v2_pooling=False,
        use_v4_dc_channel=False,
        use_v2_gabor=False,
    )).eval()
    v4_pooled = V4Backbone(RecursiveFiltersV4(
        v1_spec=spec, recursive_octaves=2, recursive_kernel_size=21,
        use_v2_pooling=True, v2_sigma_to_period=2.0,
        use_v4_dc_channel=False,
        use_v2_gabor=True, v2_gabor_frequencies=(0.06, 0.03),
        v2_gabor_kernel_size=17,
    )).eval()
    feature_extractors = {
        "V1 only":   v1_bank,
        "V4 sharp":  v4_sharp,
        "V4 pooled": v4_pooled,
    }

    detector_names = list(feature_extractors.keys()) + ["grayscale"]

    def detect(img: torch.Tensor, name: str) -> torch.Tensor:
        if name == "grayscale":
            score = harris_on_features(img)[0]
        else:
            with torch.no_grad():
                feats = feature_extractors[name](img)
            score = harris_on_features(feats)[0]
        return nms_topk(score, args.top_k, window=args.nms_window)

    rng_mod = torch.Generator().manual_seed(11)
    sample_modalities = make_modalities(images[0:1], rng_mod)
    modality_names = list(sample_modalities.keys())

    tols = [2.0, 4.0, 8.0]
    matched = {tol: {d: {m: 0 for m in modality_names} for d in detector_names}
               for tol in tols}
    n_anchor = {d: {m: 0 for m in modality_names} for d in detector_names}

    fig_rows = args.n_images
    fig_cols = len(modality_names)
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(2.4 * fig_cols, 2.4 * fig_rows))
    if fig_rows == 1:
        axes = axes[None, :]

    colors = {"V1 only": "tab:blue", "V4 sharp": "tab:green",
              "V4 pooled": "tab:purple", "grayscale": "tab:orange"}
    markers = {"V1 only": "o", "V4 sharp": "s", "V4 pooled": "^", "grayscale": "+"}

    for r in range(args.n_images):
        img = images[r:r+1]
        modalities = make_modalities(img, rng_mod)
        # Reference: detect on identity image
        ref_anchors = {d: detect(modalities["identity"], d) for d in detector_names}
        for c, (m_name, m_img) in enumerate(modalities.items()):
            ax = axes[r, c]
            ax.imshow(m_img.squeeze().numpy(), cmap="gray")
            ax.set_title(f"{m_name}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            anchors_m = {d: detect(m_img, d) for d in detector_names}
            if r == 0:
                for d in detector_names:
                    pts = anchors_m[d]
                    if len(pts) > 0:
                        ax.scatter(pts[:, 1].numpy(), pts[:, 0].numpy(),
                                   s=15, marker=markers[d],
                                   facecolors="none" if markers[d] not in ("+", "x") else None,
                                   edgecolors=colors[d],
                                   c=(colors[d] if markers[d] in ("+", "x") else None),
                                   linewidths=0.8)
            for d in detector_names:
                n = len(anchors_m[d])
                n_anchor[d][m_name] += n
                for tol in tols:
                    matched[tol][d][m_name] += overlap_count(
                        ref_anchors[d], anchors_m[d], tol)

    if args.n_images > 0:
        axes[0, 0].legend(
            handles=[
                plt.Line2D([0], [0], marker=markers[d], color=colors[d],
                           linestyle="", label=d, markerfacecolor="none")
                for d in detector_names
            ],
            fontsize=6, loc="upper right",
        )

    print("\nCross-modality anchor overlap "
          "(matches between identity and modality, summed over images):")
    for tol in tols:
        print(f"\n--- tolerance {tol} px ---")
        header = f"{'modality':16s} | " + " | ".join(f"{d:>11s}" for d in detector_names)
        print(header); print("-" * len(header))
        for m in modality_names:
            row = []
            for d in detector_names:
                m_count = matched[tol][d][m]
                n = n_anchor[d][m]
                row.append(f"{m_count:>3d}/{n:<3d} {m_count/max(n,1):4.0%}")
            print(f"{m:16s} | " + " | ".join(row))

    fig.suptitle("Cross-modality anchor detection: same scene, different intensity transforms",
                 y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
