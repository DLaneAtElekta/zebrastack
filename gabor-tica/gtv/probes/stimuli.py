"""Probe generators. All images are float32 tensors in [-1, 1], shape (H, W),
unless noted. Frequencies are in cycles/pixel, angles in radians."""

import math

import torch


def _grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centered coordinates, origin at the image center."""
    c = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y, x = torch.meshgrid(c, c, indexing="ij")
    return x, y


def grating(
    size: int,
    freq: float,
    theta: float,
    phase: float = 0.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    """Sinusoidal grating whose wave vector points along ``theta``."""
    x, y = _grid(size)
    u = x * math.cos(theta) + y * math.sin(theta)
    return contrast * torch.cos(2 * math.pi * freq * u + phase)


def drifting_grating(
    size: int,
    freq: float,
    theta: float,
    n_frames: int,
    speed: float = 1.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    """Grating drifting ``speed`` pixels/frame along ``theta``. Shape (T, H, W)."""
    step = 2 * math.pi * freq * speed
    return torch.stack(
        [grating(size, freq, theta, -t * step, contrast) for t in range(n_frames)]
    )


def oriented_noise(
    size: int,
    theta: float,
    freq: float = 0.15,
    bandwidth: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Band-pass noise concentrated around orientation ``theta`` and ``freq``.

    ``bandwidth`` is the angular std (radians) of the orientation filter.
    """
    noise = torch.randn(size, size, generator=generator)
    fy = torch.fft.fftfreq(size)
    fx = torch.fft.fftfreq(size)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fx**2 + fy**2)
    ang = torch.atan2(fy, fx)
    # orientation distance modulo pi (a wave vector and its negation are the same orientation)
    d = torch.remainder(ang - theta + math.pi / 2, math.pi) - math.pi / 2
    radial = torch.exp(-0.5 * (torch.log((r + 1e-8) / freq) / 0.4) ** 2)
    angular = torch.exp(-0.5 * (d / bandwidth) ** 2)
    out = torch.fft.ifft2(torch.fft.fft2(noise) * radial * angular).real
    return out / (out.abs().max() + 1e-8)


def contrast_modulated_grating(
    size: int,
    envelope_freq: float,
    envelope_theta: float,
    carrier_theta: float = 0.0,
    carrier_freq: float = 0.25,
    depth: float = 1.0,
    generator: torch.Generator | None = None,
    envelope_phase: float = 0.0,
) -> torch.Tensor:
    """Second-order stimulus: a carrier whose *contrast* follows a grating.

    Mean luminance is flat, so a linear filter at the envelope frequency sees
    nothing; a filter-rectify-filter (V2-like) stage should.
    """
    carrier = oriented_noise(size, carrier_theta, carrier_freq, generator=generator)
    envelope = 0.5 * (1 + depth * grating(size, envelope_freq, envelope_theta, envelope_phase))
    return carrier * envelope


def texture_pair(
    size: int,
    theta_a: float,
    theta_b: float,
    boundary_theta: float = math.pi / 2,
    freq: float = 0.15,
    generator: torch.Generator | None = None,
    offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two oriented textures meeting at a straight boundary.

    Returns ``(image, mask)`` where ``mask`` is 1 on texture B's side.
    ``boundary_theta`` is the direction of the boundary's normal and
    ``offset`` its signed distance (pixels) from the image center.
    """
    a = oriented_noise(size, theta_a, freq, generator=generator)
    b = oriented_noise(size, theta_b, freq, generator=generator)
    x, y = _grid(size)
    mask = ((x * math.cos(boundary_theta) + y * math.sin(boundary_theta)) > offset).float()
    return a * (1 - mask) + b * mask, mask


def curvature_fragment(
    size: int,
    curvature: float,
    theta: float = 0.0,
    length: float | None = None,
    width: float = 1.5,
) -> torch.Tensor:
    """A bright contour fragment of signed ``curvature`` (1/pixels) on a dark field.

    The fragment passes through the image center, tangent to direction
    ``theta``; ``curvature == 0`` gives a straight bar. Pasupathy-Connor-style
    shapes for V4 probing are built from these fragments.
    """
    length = size * 0.6 if length is None else length
    x, y = _grid(size)
    # rotate into the fragment frame: tangent along +u, normal along +v
    u = x * math.cos(theta) + y * math.sin(theta)
    v = -x * math.sin(theta) + y * math.cos(theta)
    if abs(curvature) < 1e-6:
        dist = v.abs()
        arc = u
    else:
        r = 1.0 / curvature
        # circle centered at (0, r) in the fragment frame
        rho = torch.sqrt(u**2 + (v - r) ** 2)
        dist = (rho - abs(r)).abs()
        arc = abs(r) * torch.atan2(u, math.copysign(1.0, r) * (r - v))
    on_arc = (arc.abs() <= length / 2).float()
    img = torch.exp(-0.5 * (dist / width) ** 2) * on_arc
    return 2 * img - 1


JUNCTION_ARMS = {
    "line": (0.0, math.pi),
    "L": (0.0, math.pi / 2),
    "T": (0.0, math.pi, math.pi / 2),
    "X": (0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
}


def junction(
    size: int,
    kind: str,
    theta: float = 0.0,
    arm_length: float | None = None,
    width: float = 1.5,
    center: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Bright bars radiating from ``center``: a straight line, L-corner,
    T-junction or X-crossing (keys of ``JUNCTION_ARMS``), rotated by ``theta``.

    Every kind has the same arm length, so kinds differ in configuration
    rather than only in total contour length (line and L have equal ink).
    """
    arm_length = size * 0.3 if arm_length is None else arm_length
    x, y = _grid(size)
    x, y = x - center[0], y - center[1]
    img = torch.zeros(size, size)
    for a in JUNCTION_ARMS[kind]:
        ang = theta + a
        u = x * math.cos(ang) + y * math.sin(ang)
        v = -x * math.sin(ang) + y * math.cos(ang)
        along = u.clamp(0, arm_length)
        dist = torch.sqrt((u - along) ** 2 + v**2)
        img = torch.maximum(img, torch.exp(-0.5 * (dist / width) ** 2))
    return 2 * img - 1
