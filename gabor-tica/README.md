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
| 1 | Developmental V1 (quadrature Gabor, energy, divisive norm, log) | next |
| 2 | V2 block, Design A vs. B, TICA | — |
| 3 | Generative path, wake–sleep (or FE-1, Section 7) | — |
| 4 | Temporal coherence | — |
| 5 | V4 → PIT → AIT | — |
| 6 | Thalamic gain (attention field) | — |
| 7 | Expectation channel | — |
| 8 | Context topography and routing (stretch) | — |

## Related code elsewhere in this repo

- `fashionmnist/oriented_powermap_2d.py` — Keras oriented power maps (Gabor energy); reference for Phase 1.
- `zebrastack/rxndiff/` and `examples/hierarchical_rxndiff.ipynb` — Stage-1 top-down routing on Gabor-power channels.
- `activation-atlas/gabor_objectives.py` — Gabor-parameterized objectives for feature visualization.
