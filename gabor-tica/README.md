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
python experiments/phase2_v2.py    # V2 A-vs-B + TICA (~2 min); --config configs/phase2_rica.yaml for RICA
python experiments/phase2_envelope_sweep.py  # second-order probe across envelope frequencies (~4 min)
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
| 2 | V2 block, Design A vs. B, TICA | ✅ B ≥ A on all probes; complete TICA and overcomplete RICA |
| 3 | Generative path, wake–sleep (or FE-1, Section 7) | next |
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

## Phase 2 results (`configs/phase2.yaml`, `configs/phase2_rica.yaml`)

`V2Stage` (`kind: v2_tica`) on the Phase 1 V1 decimated 2×, fitted on photo patches.
- **Design B:** second-order quadrature Gabors (0.0625, 0.03125 cyc/px × 4
  orientations) on each V1 channel, frequency-decreasing paths only (160
  channels), plus 24 pooled first-order channels, then a 1×1 TICA.
- **Design A:** TICA learned on 8×8×24 neighborhoods of the V1 maps (a strided conv).

Both reduce to 64 PCA dims, then TICA on an 8×8 torus (complete) or 144 units on
a 12×12 torus (RICA, 2.25× overcomplete). Readout = linear classifier on TICA's
signed coefficients plus sheet-pooled energy, held out, averaged over 3 splits.

| Probe | V1 only | A (complete) | B (complete) | A (RICA) | B (RICA) |
|---|---|---|---|---|---|
| First-order texture (8 orientations) | 1.00 | 1.00 | 0.99 | 1.00 | 0.98 |
| **Second-order texture** (4 envelope orientations) | 0.51 | 0.79 | **1.00** | 0.85 | **1.00** |
| Junctions (line / L / T / X) | 1.00 | 0.99 | 1.00 | 0.98 | 1.00 |
| Texture-boundary AUC | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Energy corr. near / far on sheet | — | 4.5× | 5.2× | 9.6× | 155× |
| Unit sparsity (excess kurtosis) | — | 1.45 | 0.11 | 1.29 | 0.14 |

Findings and caveats:
- The one probe that separates the designs is second-order texture: V1 reads
  0.51 (chance is 0.25; see the envelope sweep below for why it is above
  chance), A (linear over V1 maps) partly recovers it through rectification,
  B's filter-rectify-filter path solves it. The probe's envelope frequency
  (0.0625) equals one of B's fixed scales, which the sweep below checks.
- First-order texture, junctions and boundaries are at ceiling even for V1
  alone, so they don't yet discriminate A from B. Harder variants (lower
  contrast, added noise, finer class spacing) are needed before Phase 5 leans on them.
- Readout matters: rectified-only `|s|` hid first-order information carried by
  the sign (B dropped to 0.91 on texture); signed-only `s` hid A's second-order
  information (0.56). Both TICA outputs are used.
- B's TICA units are nearly Gaussian (kurtosis ≈ 0.1), because the log at the
  end of each stage Gaussianizes its input. TICA still finds the variance
  dependencies (strong topography), but there is little sparse structure left
  for it. Worth trying TICA before the log (on normalized energy) in Phase 3.
- The orientation-preference sheet maps use pooled energy, which averages over
  sheet neighbors, so their smoothness is partly built in. The near/far
  energy-correlation ratio on raw coefficients is the unbiased topography measure.
- TICA implementation notes: complete TICA uses Riemannian gradient descent on
  the orthogonal group (Adam stalls against the re-orthonormalization); sheet
  layout is refined by greedy unit swaps that lower the TICA loss (gradients
  alone leave patchy maps); RICA normalizes rows inside the objective
  (otherwise every unit collapses to zero).

## Phase 2 follow-up: envelope-frequency sweep

`python experiments/phase2_envelope_sweep.py` (~4 min). Second-order texture
decoding (4-way, chance 0.25) as the envelope frequency varies, with two extra
B variants: second-order scales shifted +½ octave, and four scales spanning the sweep.

| Envelope (cyc/px) | V1 | A | B (0.0625, 0.031) | B shifted (0.088, 0.044) | B 4 scales |
|---|---|---|---|---|---|
| 0.0156 | 0.25 | 0.45 | 0.92 | 0.75 | 0.98 |
| 0.022 | 0.31 | 0.52 | 0.99 | 0.93 | 1.00 |
| 0.031 | 0.37 | 0.58 | 1.00 | 1.00 | 1.00 |
| 0.044 | 0.58 | 0.75 | 1.00 | 1.00 | 1.00 |
| 0.0625 | 0.57 | 0.81 | 1.00 | 1.00 | 1.00 |
| 0.088 | 0.71 | 0.94 | 0.79 | 0.99 | 0.95 |
| 0.125 | 0.84 | 0.95 | 0.54 | 0.57 | 0.87 |
| 0.177 | 0.96 | 0.99 | 0.47 | 0.48 | 0.59 |

What it shows:
- **The probe is only purely second-order at low envelope frequencies.**
  Modulating the carrier adds Fourier sidebands that V1 energy can read, so V1
  climbs from chance (0.0156) to 0.96 (0.177). Only below ~0.03 is V1 near chance.
- **In the purely second-order regime B's advantage is architectural.** B
  beats A by 0.4–0.5 there, including half an octave below its lowest scale,
  and the shifted B still beats A by 0.3 at 0.0156.
- **Scale choice matters at the edges.** Each B variant's high plateau moves
  with its scales (shifted B holds 0.99 at 0.088 where B drops to 0.79), and
  the four-scale bank widens the plateau.
- **B loses first-order information that it contains.** Above ~0.09, A and
  even V1 alone beat every B variant. The 64-dim PCA before TICA keeps 11% of
  the 24 first-order channels' variance against 38% for the 160 second-order
  ones, so the first-order channels are crowded out: at 0.177 they alone decode
  0.93, but B after TICA gets 0.59. A fix is to whiten the two channel groups
  with separate budgets (or weight them equally) before TICA.

## Related code elsewhere in this repo

- `fashionmnist/oriented_powermap_2d.py` — Keras oriented power maps (Gabor energy); reference for Phase 1.
- `zebrastack/rxndiff/` and `examples/hierarchical_rxndiff.ipynb` — Stage-1 top-down routing on Gabor-power channels.
- `activation-atlas/gabor_objectives.py` — Gabor-parameterized objectives for feature visualization.
