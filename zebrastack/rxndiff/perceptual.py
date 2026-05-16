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


def gabor_energy_grid(image, freqs=(0.05, 0.10, 0.20, 0.40), n_orient=6,
                      kernel_size=15, sigma_factor=0.8, standardize=False):
    """Total Gabor power per (scale, orientation), shape ``(n_scales, n_orient)``.

    With ``standardize=False`` the image is only zero-meaned, so total
    energy depends on contrast — useful for distinguishing rich textures
    from near-uniform fields. With ``standardize=True`` the image is
    z-scored first; energy then only reflects spectral shape.
    """
    img = image.astype(np.float64)
    if standardize:
        img = (img - img.mean()) / (img.std() + 1e-8)
    else:
        img = img - img.mean()
    fft_im = fft2(img)
    grid = np.zeros((len(freqs), n_orient), dtype=np.float64)
    for s, f in enumerate(freqs):
        sigma = sigma_factor / f
        for o in range(n_orient):
            theta = o * np.pi / n_orient
            k = _gabor_kernel(kernel_size, f, theta, sigma)
            grid[s, o] = _fft_power(fft_im, k, img.shape).mean()
    return grid


def _entropy(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def autocorrelation_peak(image, min_lag=2, max_lag=24):
    """Largest off-DC peak of the autocorrelation, normalised by the
    zero-lag value. High (~1) for periodic / strongly-coherent patterns,
    low (~0) for noise."""
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()
    if not np.isfinite(img).all():
        return 0.0
    F = fft2(img)
    ac = np.fft.fftshift(np.real(ifft2(np.abs(F) ** 2)))
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    zero = ac[cy, cx]
    if zero <= 1e-12:
        return 0.0
    r = max(1, int(max_lag))
    region = ac[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].copy()
    rcy = min(cy, r)
    rcx = min(cx, r)
    m = max(1, int(min_lag))
    lo_y = max(0, rcy - m)
    hi_y = min(region.shape[0], rcy + m + 1)
    lo_x = max(0, rcx - m)
    hi_x = min(region.shape[1], rcx + m + 1)
    region[lo_y:hi_y, lo_x:hi_x] = -np.inf
    peak = float(region.max())
    if not np.isfinite(peak):
        return 0.0
    return max(0.0, peak / zero)


def structure_score(image, freqs=(0.05, 0.10, 0.20, 0.40), n_orient=6,
                    min_std=0.03, kernel_size=15, sigma_factor=0.8,
                    coherence_min_lag=2, coherence_max_lag=24,
                    coherence_weight=2.0):
    """Multi-scale Gabor 'richness' score.

    Returns ``0.0`` when the image is near-uniform (``std < min_std``) —
    this filters out the trivial degenerate outputs the search is
    otherwise tempted to drift toward. Otherwise:

        E       = total Gabor energy across all (scale, orientation) cells
        H_scale = entropy of the marginal-over-orientation energy
        H_orient= entropy of the marginal-over-scale energy
        C       = normalised autocorrelation peak (0..~1)
        score   = log1p(E) * H_scale * H_orient * (1 + coherence_weight * C)

    The Gabor terms reward energy spread across multiple scales and
    orientations; the autocorrelation factor penalises pure noise
    (which has flat spectrum but no spatial coherence) so the search is
    pushed toward multi-scale *organised* textures rather than monotone
    fields, single gratings, single-scale spots, or noise.
    """
    img = np.asarray(image, dtype=np.float64)
    if not np.isfinite(img).all():
        return 0.0
    if img.std() < min_std:
        return 0.0
    grid = gabor_energy_grid(img, freqs=freqs, n_orient=n_orient,
                             kernel_size=kernel_size,
                             sigma_factor=sigma_factor, standardize=False)
    total = float(grid.sum())
    if total <= 0:
        return 0.0
    H_s = _entropy(grid.sum(axis=1))
    H_o = _entropy(grid.sum(axis=0))
    coh = autocorrelation_peak(img, min_lag=coherence_min_lag,
                               max_lag=coherence_max_lag)
    return float(np.log1p(total) * H_s * H_o
                 * (1.0 + coherence_weight * coh))
