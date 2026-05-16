"""Lightweight perceptual distance for RD pattern images.

Pure-numpy: convolves an image with a multi-scale Gabor bank (via FFT),
takes the power, pools spatially, log-compresses and L2-normalizes the
result. Distance is the Euclidean distance between feature vectors.
"""

import numpy as np
from numpy.fft import fft2, ifft2


def _gabor_kernel(size, freq, theta, sigma):
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(xr ** 2 + yr ** 2) / (2.0 * sigma ** 2))
    carrier = np.exp(1j * 2.0 * np.pi * freq * xr)
    g = envelope * carrier
    g = g - g.mean()
    return g


def gabor_bank(size=15, freqs=(0.08, 0.16, 0.32), n_orient=6,
               sigma_factor=0.8):
    bank = []
    for f in freqs:
        sigma = sigma_factor / f
        for i in range(n_orient):
            theta = i * np.pi / n_orient
            bank.append(_gabor_kernel(size, f, theta, sigma))
    return bank


def _fft_power(image_fft, kernel, shape):
    H, W = shape
    kh, kw = kernel.shape
    pad = np.zeros((H, W), dtype=np.complex128)
    sh = (H - kh) // 2
    sw = (W - kw) // 2
    pad[sh:sh + kh, sw:sw + kw] = kernel
    K = fft2(np.fft.ifftshift(pad))
    resp = ifft2(image_fft * K)
    return np.abs(resp) ** 2


def perceptual_features(image, bank=None, n_bins=4):
    """Compute pooled Gabor-power features. Returns a 1-D feature vector."""
    if bank is None:
        bank = gabor_bank()
    image = image.astype(np.float64)
    image = (image - image.mean()) / (image.std() + 1e-8)
    H, W = image.shape
    bh, bw = H // n_bins, W // n_bins
    fft_im = fft2(image)
    feats = []
    for k in bank:
        power = _fft_power(fft_im, k, image.shape)
        pooled = (power[:bh * n_bins, :bw * n_bins]
                  .reshape(n_bins, bh, n_bins, bw)
                  .mean(axis=(1, 3)))
        feats.append(pooled.ravel())
    f = np.concatenate(feats)
    f = np.log1p(f)
    n = np.linalg.norm(f)
    if n > 0:
        f = f / n
    return f


def perceptual_distance(image_a, image_b, bank=None, n_bins=4):
    fa = perceptual_features(image_a, bank=bank, n_bins=n_bins)
    fb = perceptual_features(image_b, bank=bank, n_bins=n_bins)
    return float(np.linalg.norm(fa - fb))
