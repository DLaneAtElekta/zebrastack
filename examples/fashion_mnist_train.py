"""Train FullVentralStream on Fashion-MNIST.

Fashion-MNIST is grayscale 28x28 with 10 clothing categories. The images
are bilinearly upsampled to 64x64 so the V4 receptive fields fit, and
fed through the same V1>V2>V4>PIT>CIT>AIT stack used elsewhere.

Run:
    python examples/fashion_mnist_train.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4, standard_v1_spec,
)
from zebrastack.common.it_stages import FullVentralStream


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def read_idx_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = f.read(n * h * w)
    arr = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, h, w)
    return arr.float() / 255.0


def read_idx_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = f.read(n)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def upsample(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(images.unsqueeze(1), size=(size, size), mode="bilinear",
                         align_corners=False)


def confusion(model, loader, device, n_classes):
    model.eval()
    matrix = [[0] * n_classes for _ in range(n_classes)]
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            preds = model(x)["logits"].argmax(dim=-1)
            for p, t in zip(preds.tolist(), y.tolist()):
                matrix[t][p] += 1
                if p == t: correct += 1
                total += 1
    return correct / max(total, 1), matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp")
    parser.add_argument("--out", default="examples/fashion_mnist_train.png")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-skip", action="store_true",
                        help="Use multi-scale skip connections to the classifier.")
    parser.add_argument("--use-attention", action="store_true",
                        help="Use SE channel attention on the 1x1 reduce in each IT stage.")
    parser.add_argument("--use-v2-mix", action="store_true",
                        help="Enable V2 1x1 cross-channel mixing of V1 power maps.")
    parser.add_argument("--use-v4-mix", action="store_true",
                        help="Enable V4 1x1 cross-channel mixing on V4 backbone output.")
    args = parser.parse_args()
    matplotlib.use("Agg")

    device = args.device
    print(f"Device: {device}")

    print("Loading Fashion-MNIST...")
    train_imgs = read_idx_images(os.path.join(args.data_dir, "fmnist-train-images-idx3-ubyte.gz"))
    train_labels = read_idx_labels(os.path.join(args.data_dir, "fmnist-train-labels-idx1-ubyte.gz"))
    test_imgs = read_idx_images(os.path.join(args.data_dir, "fmnist-t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx_labels(os.path.join(args.data_dir, "fmnist-t10k-labels-idx1-ubyte.gz"))

    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(train_imgs.shape[0], generator=rng)
    train_idx = perm[: args.n_train]
    test_perm = torch.randperm(test_imgs.shape[0], generator=rng)
    test_idx = test_perm[: args.n_test]

    train_x = (upsample(train_imgs[train_idx], args.size) - 0.5) * 2.0
    train_y = train_labels[train_idx]
    test_x = (upsample(test_imgs[test_idx], args.size) - 0.5) * 2.0
    test_y = test_labels[test_idx]
    print(f"Train: {len(train_y)}  test: {len(test_y)}  shape: {train_x.shape}")

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y),
                             batch_size=args.batch_size, shuffle=False)

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.30, 0.18, 0.10),
        kernel_size=13,
    )
    v4 = RecursiveFiltersV4(
        v1_spec=spec,
        recursive_octaves=2,
        recursive_kernel_size=21,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=False,
        use_v2_gabor=True,
        v2_gabor_frequencies=(0.06, 0.03),
        v2_gabor_kernel_size=17,
        use_v2_mixing=args.use_v2_mix,
        trainable_v2_mix=True,
    )
    model = FullVentralStream(
        v4_backbone=v4,
        n_classes=10,
        pit_n_reduce=24,
        cit_n_reduce=24,
        ait_n_reduce=24,
        pit_frequencies=(0.05, 0.025),
        cit_frequencies=(0.05, 0.025),
        ait_frequencies=(0.05, 0.025),
        kernel_size=11,
        downsample=2,
        use_skip=args.use_skip,
        use_attention=args.use_attention,
        use_v4_mix=args.use_v4_mix,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0; total_correct = 0; total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)["logits"]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += x.shape[0]
        print(f"  epoch {epoch+1:2d}: loss={total_loss/total:.4f} train_acc={total_correct/total:.3f}")

    print("\n  recalibrating BN running stats...")
    cal_x = train_x[: min(512, len(train_x))].to(device)
    model.recalibrate_bn(cal_x)

    test_acc, cm = confusion(model, test_loader, device, 10)
    print(f"\n  test accuracy: {test_acc:.3f}")
    print("Per-class recall:")
    for i, name in enumerate(CLASS_NAMES):
        n = sum(cm[i])
        r = cm[i][i] / n if n else 0.0
        print(f"  {name:12s}: {r:.2f}  ({cm[i][i]}/{n})")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    cm_t = torch.tensor(cm, dtype=torch.float32)
    cm_norm = cm_t / cm_t.sum(dim=-1, keepdim=True).clamp_min(1)
    im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            v = cm[i][j]
            if v > 0:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    extras = []
    if args.use_skip:     extras.append("skip")
    if args.use_attention: extras.append("attn")
    if args.use_v2_mix:   extras.append("V2mix")
    if args.use_v4_mix:   extras.append("V4mix")
    extras_str = (" + " + " + ".join(extras)) if extras else ""
    ax.set_title(f"FullVentralStream{extras_str} on Fashion-MNIST ({args.size}px)\n"
                 f"test acc = {test_acc:.3f}, params = {n_trainable}")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
