# gabor-tica — Gabor–TICA Ventral Stream

A largely unsupervised V1 → V2 → V4 → PIT → AIT hierarchy: fixed/tethered Gabor
filtering at every stage, learned topographic ICA (TICA) across channels, a
wake–sleep generative path, temporal-coherence invariance, and a thalamic module
that uses top-down templates for attention and expectation.

The full plan is in [`docs/PLAN.md`](docs/PLAN.md). Work is prototype-driven:
each phase ends with a runnable system and a concrete check.

## Setup

```bash
cd gabor-tica
pip install -e ".[dev]"            # add ",scattering" for Kymatio (Phase 2)
pytest                             # probe + pipeline tests
python experiments/phase0_smoke.py # renders figures into runs/phase0/
python experiments/phase1_v1.py    # V1 exit checks -> runs/phase1/report.json
```

## Layout

```
gabor-tica/
  docs/PLAN.md         implementation plan (source of truth for phases)
  configs/             plain-YAML experiment configs; stages declared by `kind`
  gtv/
    probes/            synthetic stimuli with known ground truth
    stages/            stage blocks (gabor, normalize, tica, block)
    viz/               filter atlas, topographic sheet map, fantasy grid
    data/              natural images, video, category sets        (Phase 1+)
    generative/        top-down decoders, wake–sleep               (Phase 3)
    temporal/          temporal coherence / slowness               (Phase 4)
    thalamus/          gain, expectation, routing                  (Phases 6–8)
  experiments/         one runnable script per phase check
  tests/
```

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Infrastructure and probes | ✅ probes, identity stage, viz, smoke test |
| 1 | Developmental V1 (quadrature Gabor, energy, divisive norm, log) | ✅ fixed bank; exit checks pass (developmental variant deferred) |
| 2 | V2 block, Design A vs. B, TICA | next |
| 3 | Generative path, wake–sleep (or FE-1, Section 7) | — |
| 4 | Temporal coherence | — |
| 5 | V4 → PIT → AIT | — |
| 6 | Thalamic gain (attention field) | — |
| 7 | Expectation channel | — |
| 8 | Context topography and routing (stretch) | — |

## Phase 1 results (`configs/phase1.yaml`)

`V1Stage` (`kind: v1_gabor`): 8 orientations × 3 octave-spaced scales (0.25,
0.125, 0.0625 cycles/px) of frequency-domain quadrature Gabors → energy →
Heeger divisive normalization → log.

| Check | Result |
|---|---|
| Orientation peaks at nominal | all 24 units, within the 5° sampling step; median HWHH ≈ 10° |
| SF peaks at nominal (filter energy) | within 3% for all units |
| Phase invariance (complex cells) | modulation < 0.1% |
| Contrast invariance | tuning-shape corr > 0.999 over 10× contrast; amplitude ratio 100× raw → 5× normalized |
| Natural photos: raw energy | skew ≈ 14, excess kurtosis ≈ 300 |
| Natural photos: log(normalized) | skew ≈ 0.4, excess kurtosis ≈ −0.6 |

Notes for later phases:
- Normalization shifts the *edge* scales' SF peaks (up to +17% at 0.25 cyc/px),
  because the pool has no filters beyond the bank. Adding a scale above/below,
  or `norm_pool_groups`, changes this.
- Tuning is fairly narrow (HWHH ≈ 10° on energy); raise `aspect` toward 1 or
  lower `bandwidth_octaves` for broader tuning.
- `log_eps` sets the log floor; 1e-3 visibly piles ~1.6% of responses at the floor, 1e-4 ~0.2%.
- Deferred: the optional developmental variant (learning Gabors from retinal-wave noise).

## Related code elsewhere in this repo

- `fashionmnist/oriented_powermap_2d.py` — Keras oriented power maps (Gabor energy); reference for Phase 1.
- `zebrastack/rxndiff/` and `examples/hierarchical_rxndiff.ipynb` — Stage-1 top-down routing on Gabor-power channels.
- `activation-atlas/gabor_objectives.py` — Gabor-parameterized objectives for feature visualization.
