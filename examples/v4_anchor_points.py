"""V4 features as anchor-point detector for deformable registration.

Computes a Harris-style cornerness response on the rich V4 feature map
(320 oriented + multi-scale + Lie-group channels) instead of the
single-channel grayscale gradient. Locations where the feature
*vector* changes rapidly in two directions are candidate anchors --
analogous to Harris corners but using multi-scale, multi-orientation
content rather than raw pixel gradients.

Tests repeatability: detect anchors in the original image and in a
known-warped version, then count how many anchors line up under the
warp (within a small pixel tolerance). Compares against plain
grayscale Harris as a baseline.

Run:
    python examples/v4_anchor_points.py
"""

from __future__ import annotations

import argparse
import gzip
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
    RecursiveFiltersV4, standard_v1_spec,
)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        data = f.read(n * h * w)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, h, w).float() / 255.0


def harris_on_features(features: torch.Tensor, sigma: float = 2.0,
                       kappa: float = 0.04) -> torch.Tensor:
    """Harris cornerness on a multi-channel feature map.

    Standard Harris corner detector but the structure tensor is summed
    over the input feature channels rather than computed from a single
    grayscale image. Pixels where the feature vector changes rapidly in
    two directions get high scores.
    """
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
    kx = g1d.view(1, 1, 1, ksize)
    ky = g1d.view(1, 1, ksize, 1)
    def smooth(t):
        t = F.pad(t, (half, half, half, half), mode="reflect")
        t = F.conv2d(t, kx)
        t = F.conv2d(t, ky)
        return t
    Ixx = smooth(Ixx); Ixy = smooth(Ixy); Iyy = smooth(Iyy)

    det = Ixx * Iyy - Ixy * Ixy
    tr = Ixx + Iyy
    return (det - kappa * tr * tr).squeeze(1)


def harris_grayscale(image: torch.Tensor, sigma: float = 2.0,
                     kappa: float = 0.04) -> torch.Tensor:
    """Standard Harris on a single-channel image (B, 1, H, W) -> (B, H, W)."""
    return harris_on_features(image, sigma=sigma, kappa=kappa)


def nms_topk(score: torch.Tensor, k: int, window: int = 9,
             min_score: float | None = None) -> torch.Tensor:
    """Non-maximum suppression. Returns (k, 2) tensor of (y, x) coordinates."""
    s = score.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(s, kernel_size=window, stride=1, padding=window // 2)
    is_max = (s == pooled).squeeze(0).squeeze(0)
    s_flat = score * is_max.float()
    if min_score is not None:
        s_flat = s_flat * (s_flat > min_score).float()
    flat = s_flat.view(-1)
    vals, idx = flat.topk(min(k, flat.numel()))
    h, w = score.shape
    ys = idx // w
    xs = idx % w
    keep = vals > 0
    return torch.stack([ys[keep], xs[keep]], dim=-1)


def warp_image(image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply a 2x3 affine theta to an (B, 1, H, W) image via grid_sample."""
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def transform_points(pts: torch.Tensor, theta: torch.Tensor,
                     h: int, w: int) -> torch.Tensor:
    """Transform anchor (y, x) coordinates by the inverse of theta.

    grid_sample warps the image so output(p) = input(theta @ p_norm).
    To find where an anchor in the *output* image came from in the
    *input* image, apply theta directly to the normalized output point.
    Reverses the transformation to give the corresponding input-image
    pixel for each output anchor.
    """
    if pts.numel() == 0:
        return pts
    ys = pts[:, 0].float()
    xs = pts[:, 1].float()
    nx = (xs / (w - 1)) * 2.0 - 1.0
    ny = (ys / (h - 1)) * 2.0 - 1.0
    nh = torch.stack([nx, ny, torch.ones_like(nx)], dim=-1)
    th = theta[0]
    out = nh @ th.t()
    sx = (out[:, 0] + 1.0) * 0.5 * (w - 1)
    sy = (out[:, 1] + 1.0) * 0.5 * (h - 1)
    return torch.stack([sy, sx], dim=-1)


def repeatability(pts_orig: torch.Tensor, pts_warped: torch.Tensor,
                  theta: torch.Tensor, h: int, w: int,
                  tol: float = 3.0) -> tuple[int, int]:
    """Count how many warped-image anchors lie close to a transformed orig anchor."""
    if pts_orig.numel() == 0 or pts_warped.numel() == 0:
        return 0, max(len(pts_warped), 1)
    pts_warped_in_orig = transform_points(pts_warped.float(), theta, h, w)
    diff = pts_orig.float().unsqueeze(0) - pts_warped_in_orig.unsqueeze(1)
    d = (diff ** 2).sum(dim=-1).sqrt()
    matched = (d.min(dim=-1).values < tol).sum().item()
    return int(matched), len(pts_warped)


def plot_anchors(ax, image: torch.Tensor, pts: torch.Tensor, title: str,
                 color: str = "lime"):
    ax.imshow(image.squeeze().numpy(), cmap="gray")
    if len(pts) > 0:
        ax.scatter(pts[:, 1].numpy(), pts[:, 0].numpy(),
                   s=20, facecolors="none", edgecolors=color, linewidths=1.0)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp")
    parser.add_argument("--out", default="examples/v4_anchor_points.png")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--nms-window", type=int, default=9)
    parser.add_argument("--repeatability-tol", type=float, default=4.0)
    args = parser.parse_args()
    matplotlib.use("Agg")

    print("Loading Fashion-MNIST images...")
    imgs = read_idx_images(os.path.join(args.data_dir, "fmnist-train-images-idx3-ubyte.gz"))

    rng = torch.Generator().manual_seed(7)
    idx = torch.randperm(imgs.shape[0], generator=rng)[: args.n_images]
    images = imgs[idx]
    images = F.interpolate(images.unsqueeze(1), size=(args.size, args.size),
                           mode="bilinear", align_corners=False)
    images = (images - 0.5) * 2.0

    print("Building feature extractors (fixed analytic weights)...")
    spec = standard_v1_spec(
        n_orientations=4, frequencies=(0.30, 0.18, 0.10), kernel_size=13,
    )
    from zebrastack.common.v4_recursive_filters import GaborPowerBank
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
    print(f"  V1 only:   {spec.n_frequencies * spec.n_orientations} channels")
    print(f"  V4 sharp:  {v4_sharp.n_outputs} channels (no V2 pool / V2 Gabor)")
    print(f"  V4 pooled: {v4_pooled.n_outputs} channels (with V2 pool + V2 Gabor)")

    warps = {
        "translate(+5,+3)px": torch.tensor([[[1.0, 0.0, -10.0/(args.size-1)],
                                              [0.0, 1.0, -6.0/(args.size-1)]]]),
        "rotate +10 deg":     torch.tensor([[[ 0.9848,  0.1736, 0.0],
                                              [-0.1736,  0.9848, 0.0]]]),
        "scale 1.1":          torch.tensor([[[1.0/1.1, 0.0, 0.0],
                                              [0.0, 1.0/1.1, 0.0]]]),
        "shear x by 0.1":     torch.tensor([[[1.0, -0.1, 0.0],
                                              [0.0,  1.0, 0.0]]]),
    }

    detector_names = list(feature_extractors.keys()) + ["grayscale"]
    repeat = {name: {d: 0 for d in detector_names} for name in warps}
    total  = {name: {d: 0 for d in detector_names} for name in warps}

    fig_rows = args.n_images
    fig_cols = 1 + len(warps)
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(3.4 * fig_cols, 3.4 * fig_rows))
    if fig_rows == 1:
        axes = axes[None, :]

    def detect_anchors(img: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "grayscale":
            score = harris_grayscale(img)[0]
        else:
            with torch.no_grad():
                feats = feature_extractors[name](img)
            score = harris_on_features(feats)[0]
        pts = nms_topk(score, args.top_k, window=args.nms_window)
        return score, pts

    for r in range(args.n_images):
        img = images[r:r+1]
        h_im = w_im = args.size

        # Score and detect anchors in original image for each detector.
        anchors_orig = {name: detect_anchors(img, name)[1] for name in detector_names}

        # Show original image with all anchor sets overlaid.
        ax = axes[r, 0]
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        colors = {"V1 only": "tab:blue", "V4 sharp": "tab:green",
                  "V4 pooled": "tab:purple", "grayscale": "tab:orange"}
        markers = {"V1 only": "o", "V4 sharp": "s", "V4 pooled": "^", "grayscale": "x"}
        for name in detector_names:
            pts = anchors_orig[name]
            if len(pts) > 0:
                ax.scatter(pts[:, 1].numpy(), pts[:, 0].numpy(),
                           s=18, marker=markers[name],
                           facecolors="none" if markers[name] != "x" else None,
                           edgecolors=colors[name], linewidths=0.9, label=name)
        ax.set_title(f"img {r}: anchors"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.legend(fontsize=6, loc="upper right")

        # For each warp, detect anchors in warped image and compute repeatability.
        for w_idx, (name, theta) in enumerate(warps.items()):
            warped = warp_image(img, theta)
            ax = axes[r, 1 + w_idx]
            ax.imshow(warped.squeeze().numpy(), cmap="gray")
            stats = []
            for d in detector_names:
                _, pts_w = detect_anchors(warped, d)
                m, n_w = repeatability(anchors_orig[d], pts_w, theta, h_im, w_im,
                                        tol=args.repeatability_tol)
                repeat[name][d] += m
                total[name][d]  += n_w
                if r == 0 and len(pts_w) > 0:
                    ax.scatter(pts_w[:, 1].numpy(), pts_w[:, 0].numpy(),
                               s=18, marker=markers[d],
                               facecolors="none" if markers[d] != "x" else None,
                               edgecolors=colors[d], linewidths=0.9)
                stats.append(f"{d}: {m}/{n_w}")
            ax.set_title(f"{name}\n" + "\n".join(stats), fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    print("\nRepeatability (matches / warped anchors, summed over images):")
    header = f"{'warp':22s} | " + " | ".join(f"{d:>14s}" for d in detector_names)
    print(header); print("-" * len(header))
    for wn in warps:
        row = []
        for d in detector_names:
            r_val = repeat[wn][d] / max(total[wn][d], 1)
            row.append(f"{repeat[wn][d]:>3d}/{total[wn][d]:<3d} {r_val:4.0%}")
        print(f"{wn:22s} | " + " | ".join(row))

    # Multi-tolerance evaluation: are V4 anchors roughly correct but
    # less precise, or genuinely worse?
    tols = [2.0, 4.0, 8.0, 16.0]
    print(f"\nRepeatability vs match tolerance (averaged over warps and images):")
    header = f"{'tolerance':>10s} | " + " | ".join(f"{d:>10s}" for d in detector_names)
    print(header); print("-" * len(header))
    for tol in tols:
        row = []
        for d in detector_names:
            matched = 0; n_warped = 0
            for r_idx in range(args.n_images):
                img = images[r_idx:r_idx+1]
                _, pts_orig = detect_anchors(img, d)
                for theta in warps.values():
                    warped = warp_image(img, theta)
                    _, pts_w = detect_anchors(warped, d)
                    m, nw = repeatability(pts_orig, pts_w, theta,
                                           args.size, args.size, tol=tol)
                    matched += m; n_warped += nw
            row.append(f"{matched / max(n_warped, 1):10.1%}")
        print(f"{tol:>10.1f} | " + " | ".join(row))

    fig.suptitle("V4 vs grayscale Harris anchors under known warps "
                 f"(top-{args.top_k} per image, tolerance {args.repeatability_tol}px)",
                 y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
