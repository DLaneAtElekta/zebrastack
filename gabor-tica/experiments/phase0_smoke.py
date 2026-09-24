"""Phase 0 exit check: probes render, and a dummy identity stage passes end to
end through the viz tools.

    python experiments/phase0_smoke.py [--config configs/default.yaml]
"""

import argparse
import math
from pathlib import Path

import torch

from gtv.config import load_config
from gtv.probes import (
    contrast_modulated_grating,
    curvature_fragment,
    drifting_grating,
    grating,
    texture_pair,
)
from gtv.stages import build_stack
from gtv.viz import fantasy_grid, filter_atlas, sheet_map

ROOT = Path(__file__).resolve().parents[1]


def make_probe_batch(cfg: dict, gen: torch.Generator) -> tuple[torch.Tensor, list[str]]:
    size = cfg["image_size"]
    p = cfg["probes"]
    imgs, titles = [], []
    for k in range(p["n_orientations"]):
        theta = k * math.pi / p["n_orientations"]
        imgs.append(grating(size, p["spatial_freqs"][1], theta, contrast=p["contrast"]))
        titles.append(f"grating {math.degrees(theta):.0f}°")
    for f in p["spatial_freqs"]:
        imgs.append(grating(size, f, 0.0))
        titles.append(f"sf {f}")
    imgs.append(drifting_grating(size, 0.1, math.pi / 4, n_frames=4)[-1])
    titles.append("drift t=3")
    imgs.append(contrast_modulated_grating(size, 0.03, 0.0, carrier_theta=math.pi / 2, generator=gen))
    titles.append("2nd-order")
    img, _ = texture_pair(size, 0.0, math.pi / 2, generator=gen)
    imgs.append(img)
    titles.append("texture pair")
    for k in (-0.08, 0.0, 0.08):
        imgs.append(curvature_fragment(size, k))
        titles.append(f"curv {k:+.2f}")
    return torch.stack(imgs).unsqueeze(1), titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    x, titles = make_probe_batch(cfg, gen)
    stack = build_stack(cfg["stages"])
    y = stack(x)
    assert torch.equal(x, y), "identity stack should not change its input"

    fantasy_grid(x, titles=titles).savefig(out / "probes.png", dpi=120)
    fantasy_grid(y, titles=titles).savefig(out / "stack_output.png", dpi=120)
    # stand-ins until Phase 1/2 produce real filters and sheets
    filter_atlas(x[: cfg["probes"]["n_orientations"], :, 16:48, 16:48]).savefig(out / "filter_atlas.png", dpi=120)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    pinwheel = torch.remainder(0.5 * torch.atan2(yy, xx), math.pi)
    sheet_map(pinwheel, strength=(xx**2 + yy**2).sqrt(), title="synthetic pinwheel").savefig(out / "sheet_map.png", dpi=120)

    print(f"Phase 0 smoke passed: {len(titles)} probes through {len(stack)} stage(s); figures in {out}")


if __name__ == "__main__":
    main()
