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
python experiments/phase2_envelope_sweep.py  # second-order probe across envelope frequencies (~5 min)
python experiments/phase3_generative.py      # generative path + wake-sleep (~6 min)
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
    generative/        decoder, latent priors, wake–sleep, fantasy diagnostics (Phase 3)
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
| 3 | Generative path, wake–sleep (or FE-1, Section 7) | ⚠️ 3a ✅; 3b stable but misses the R²-drop check by 0.002; 3c no gain |
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
  channels), plus 24 pooled first-order channels, whitened as two groups
  (24 + 40 dims), then a 1×1 TICA.
- **Design A:** TICA learned on 8×8×24 neighborhoods of the V1 maps (a strided conv).

Both reduce to 64 PCA dims, then TICA on an 8×8 torus (complete) or 144 units on
a 12×12 torus (RICA, 2.25× overcomplete). Readout = linear classifier on TICA's
signed coefficients plus sheet-pooled energy, held out, averaged over 3 splits.

| Probe | V1 only | A (complete) | B (complete) | A (RICA) | B (RICA) |
|---|---|---|---|---|---|
| First-order texture (8 orientations) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Second-order texture** (4 envelope orientations) | 0.51 | 0.79 | **1.00** | 0.85 | **1.00** |
| Junctions (line / L / T / X) | 1.00 | 0.99 | 1.00 | 0.98 | 0.99 |
| Texture-boundary AUC | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Energy corr. near / far on sheet | — | 4.5× | 8.9× | 9.6× | 29× |
| Unit sparsity (excess kurtosis) | — | 1.45 | 0.21 | 1.29 | 0.21 |

B whitens its first-order (24) and second-order (160) channels with separate
PCA budgets of 24 and 40 dims (`whiten_groups`), then re-whitens jointly;
see the envelope sweep below for why.

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
- B's TICA units are nearly Gaussian (kurtosis ≈ 0.2), because the log at the
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

`python experiments/phase2_envelope_sweep.py` (~5 min). Second-order texture
decoding (4-way, chance 0.25) as the envelope frequency varies. B variants:
Phase 2 scales, scales shifted +½ octave, four scales spanning the sweep, and
the original single-PCA B for reference.

| Envelope (cyc/px) | V1 | A | B, shared PCA (old) | B | B shifted (0.088, 0.044) | B 4 scales |
|---|---|---|---|---|---|---|
| 0.0156 | 0.25 | 0.45 | 0.92 | 0.93 | 0.73 | 0.98 |
| 0.022 | 0.31 | 0.52 | 0.99 | 1.00 | 0.91 | 1.00 |
| 0.031 | 0.37 | 0.58 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.044 | 0.58 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.0625 | 0.57 | 0.81 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.088 | 0.71 | 0.94 | 0.79 | 0.88 | 1.00 | 0.93 |
| 0.125 | 0.83 | 0.95 | 0.54 | 0.83 | 0.87 | 0.92 |
| 0.177 | 0.96 | 0.99 | 0.47 | 0.88 | 0.90 | 0.93 |

What it shows:
- **The probe is only purely second-order at low envelope frequencies.**
  Modulating the carrier adds Fourier sidebands that V1 energy can read, so V1
  climbs from chance (0.0156) to 0.96 (0.177). Only below ~0.03 is V1 near chance.
- **In the purely second-order regime B's advantage is architectural.** B
  beats A by 0.4–0.5 there, including half an octave below its lowest scale,
  and the shifted B still beats A by ~0.3 at 0.0156.
- **Scale choice matters at the edges.** Each B variant's plateau moves with
  its scales (shifted B holds 1.00 at 0.088 where B gets 0.88), and the
  four-scale bank widens it.
- **One shared PCA crowded out B's first-order channels.** The 64-dim PCA
  kept 11% of the 24 first-order channels' variance vs 38% for the 160
  second-order ones, so at high envelope frequencies B fell below V1 alone
  (0.47 at 0.177). Whitening the groups separately (24 + 40 dims, then a
  full-rank joint re-whitening so TICA still sees white input) lifts B to
  0.83–0.88 there. Budget splits tried at 0.177: shared 0.59, 24+40 0.93,
  16+48 0.92, 8+56 0.85.
- **What remains of the high-frequency gap is mostly readout overfitting.**
  A and B each give the readout 128 features but it trains on 192 images;
  with 768 training images B reaches 0.94 / 0.97 at 0.125 / 0.177 against A's
  0.99 / 1.00. A keeps a small edge on this first-order leakage, which its
  learned 8×8 spatial filters over V1 maps can pick up directly.

## Phase 3 results (`configs/phase3.yaml`)

Generative path from Design B's 64 V2 latents (16×16 map) down to V1 log-power
maps (24 channels, 32×32); no pixels. Trained on patches from 6 repo photos,
evaluated on 2 held-out photos.

- **Decoder:** convolutional factor analysis, bilinear 2× upsampling then a
  3×3 conv, per-channel Gaussian noise with spatially correlated sampling
  fitted to the real residuals.
- **Recognition:** the fixed Design B front end plus the Phase 2
  whitening + TICA as a learnable affine map, with a Gaussian q.
- **Plausibility measure:** mean effect size of 7 groups of V1-map summary
  statistics (per-channel mean, spread, skew, lag-1/lag-4 spatial
  autocorrelation, orientation and scale correlations) against held-out real
  maps. Real-vs-real across photos sets the floor. (A real-vs-fantasy
  classifier saturated at 1.0 for everything, including near-perfect
  reconstructions, so it could not rank models.)

| | Held-out R² | Statistic gap (0 = matched) |
|---|---|---|
| Real vs real (floor) | — | 0.12 |
| 3a reconstruction + noise | 0.948 | 0.29 |
| 3a fantasy, iid prior (standard FA) | — | 1.01 |
| 3a fantasy, per-unit spatial prior | — | 0.85 |
| 3a fantasy, global + local stationary prior | — | **0.34** |
| 3b after wake–sleep, global + local prior | 0.916 | 0.37 |
| 3c dependent-variance prior | — | 0.39 |
| White noise (ceiling) | — | 3.67 |

Checks: all pass except `stable_reconstruction`. Wake–sleep lowers held-out R²
by 0.032 against a 0.03 limit set before the run.

Findings:
- **The prior matters most, and needs two levels.** Recognized V2 latents are
  smooth (lag-1 corr 0.84, overlapping receptive fields), correlated across
  units at nonzero lags, and about a quarter of their variance is a per-patch
  constant (overall contrast and texture of the patch). The prior that works is
  z = a + r: a Gaussian per-patch offset vector plus a stationary local field
  with the full lagged cross-unit covariance, estimated from the map interior
  with a flat-top taper and sampled on a larger torus, then cropped. The
  standard iid FA prior is 3× worse. Fitting the prior to recognized latents is
  the prior's own wake-phase (maximum-likelihood) update.
- **Wake–sleep is stable only with care, and never beat 3a.** The first
  version diverged: R² fell to 0.24 as sleep rewrote the recognition, and
  prior refits jolted the fantasies. Stabilizers that work: a fixed prior, q's
  noise initialized from its measured error on fantasies, a low sleep rate
  (3e-4), and a tether toward the TICA start (recognition "on rails"). With
  them, recognition stays close to TICA (median latent corr 0.97), and the
  sheet's topography gets sharper (near/far energy ratio 13.8 → 31). But R²
  drops 0.03 and fantasies get slightly worse. The wake phase trains the
  decoder on noisy q samples, and recognition, an affine map on fixed
  nonlinear features, is trained on fantasies whose features are measurably
  off-distribution: wake–sleep's sleep-phase bias. At sleep rate 3e-3,
  recognition drifts to a different but equally decodable code (corr 0.43),
  losing the TICA organization. Next options per the plan: reweighted
  wake–sleep, or the Section 7 free-energy track (FE-1).
- **Dependent variances (3c) add nothing measurable here.** Per-patch gain
  spread is small (τ ≈ 0.16) and fantasies do not improve. This is expected:
  the log at the end of each stage Gaussianizes the latents (Phase 2 open
  issue). A variance model is more likely to matter if TICA moves before the log.
- **Generative fields** (`projective_fields.png`): some V2 units drive all
  orientations together (contrast gain), some trade fine against coarse
  scales, some are orientation-specific. Sheet neighbors generate somewhat
  more similar V1 patterns than distant units (|corr| 0.23 vs 0.16).
- The decoder's stride-2 transposed convolution left period-2 checkerboard
  residuals; bilinear upsampling plus a conv removed them. Fitted stages,
  priors and decoders now round-trip through `state_dict` (fitted buffers
  resize on load).
- Not done: the Gabor tether (V1 filters drifting "on rails"). V1 is fixed,
  since a V1-power-map generative model gives no signal for moving the V1
  filters themselves; that needs a pixel-level term.

## Related code elsewhere in this repo

- `fashionmnist/oriented_powermap_2d.py` — Keras oriented power maps (Gabor energy); reference for Phase 1.
- `zebrastack/rxndiff/` and `examples/hierarchical_rxndiff.ipynb` — Stage-1 top-down routing on Gabor-power channels.
- `activation-atlas/gabor_objectives.py` — Gabor-parameterized objectives for feature visualization.
