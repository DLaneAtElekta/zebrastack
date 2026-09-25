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
python experiments/fe1_free_energy.py        # Section 7 FE-1: one free-energy objective (~8 min)
python experiments/fe2_precision.py          # FE-2: learned per-channel precision, V1 norm on/off (~8 min)
python experiments/fe3_context_precision.py  # FE-3 (V2-level analog): context precision, target in clutter (~15 min)
python experiments/fe3_prior_test.py         # is the per-location prior FE-3's bottleneck? (~10 min)
python experiments/phase4_temporal.py        # Phase 4: temporal coherence on drift sequences (~6 min)
python experiments/phase5_hierarchy.py       # Phase 5: V4 -> PIT -> AIT on Fashion-MNIST (~15 min; downloads data once)
python experiments/phase6_attention.py       # Phase 6: attention from top-down templates (needs the Phase 5 checkpoint; ~20 min)
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
    data/              natural photos, drift sequences, Fashion-MNIST (cached in data_cache/)
    generative/        decoder, latent priors, wake–sleep, fantasy diagnostics (Phase 3)
    temporal/          temporal coherence objectives for TICA       (Phase 4)
    thalamus/          attention fields (Phase 6); expectation, routing planned (7–8)
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
| FE-1 | Free energy replaces wake–sleep (Section 7) | ⚠️ 3 of 4 checks; loses linear second-order readout at the preset precision |
| FE-2 | Learned per-channel precision | ⚠️ 2 of 4 checks; precision learns cleanly with V1 normalization, but doesn't fix the second-order readout |
| FE-3 | Context-conditioned precision (V2-level analog) | ❌ no d′ gain; no headroom exists at V1–V2 (reconstruction always matches the input ceiling) |
| 4 | Temporal coherence | ❌ selectivity kept, but invariance gain +0.017 < 0.05; the fixed Design B front end leaves little for W to change |
| 5 | V4 → PIT → AIT | ✅ recognition stack; AIT decoding 0.79, category clusters p < 0.001 (generative path and log-polar input deferred) |
| 6 | Thalamic gain (attention field) | ❌ templates work (8/10); feature gain is information-neutral here; spatial field +0.19 d′ (below the 0.3 bar) |
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

## Section 7, FE-1 results (`configs/fe1.yaml`)

Wake–sleep replaced by one variational free energy, minimized over encoder
and decoder together on real V1 maps (no fantasies, so no sleep-phase bias):

    F = ½ π ‖x − g(z)‖² − (N/2) log π + Σ_locations Σ_i √(Σ_j h_ij z_j² + ε) − H[q]

q(z|x) is Gaussian: the fixed Design B front end with affine heads for the
mean and log σ (amortized, one reparameterized sample). g is the Phase 3
decoder. π is fixed (FE-1). Training uses KL warm-up (500 steps) and, every
250 steps, re-seats units on the sheet by energy correlation. F is invariant
to permuting units except through the TICA prior, which the re-seating
lowers; as in Phase 2, gradients alone cannot make those moves.

| Run | Held-out R² | Active units | Topography (active) | 2nd-order texture from μ (env 0.016 / 0.0625) | … from reconstructed V1 maps |
|---|---|---|---|---|---|
| Phase 2 TICA + 3a decoder | 0.948 | 64 | 8.1 | 0.91 / 1.00 | — |
| Π = I, TICA start | 0.902 | 6 | — | 0.69 / 0.79 | 0.92 / 1.00 |
| π ≈ 20, TICA start | **0.960** | 38 | 5.6 | 0.70 / 0.79 | 0.90 / 1.00 |
| π ≈ 20, random start | 0.960 | 55 | **2.1** (from 0.9) | 0.67 / 0.82 | 0.91 / 1.00 |
| π ≈ 150, TICA start (added for diagnosis) | **0.966** | 60 | 3.9 | 0.84 / 0.99 | 0.90 / 1.00 |

Checks (declared before running; the π ≈ 150 run was added afterwards and is
not used by them): no divergence ✅, reconstruction matches 3a ✅, topography
emerges from a random start ✅ (2.1 > 2), second-order texture retained ❌.

Findings:
- **FE beats wake–sleep on reconstruction** (0.960 vs 3b's 0.916 and 3a's
  0.948) and trains stably. Training on real data only removes wake–sleep's
  failure mode.
- **Π = I collapses the posterior**, as expected in standardized units: 6 of
  64 units carry information. Even π ≈ 20 (1 / the 3a residual variance)
  leaves 26 units collapsed. The collapsed units form contiguous regions of
  the sheet. KL warm-up barely changes the count (23 vs 25 active at π ≈ 20
  without re-seating), so collapse reflects how few V2 dimensions V1 power
  maps need at that noise level, not an optimization artifact.
- **The second-order failure is about readability, not information.** Every
  FE model's reconstructed V1 maps decode second-order texture as well as real
  maps do, so the envelope is in the latents. At low precision it is not
  linearly readable from the latent means: FE latents behave like a map of the
  envelope, with its phase spread across locations, while TICA units behave
  like invariant (rectified-energy) detectors, which survive the spatially
  averaged readout. Raising precision to π ≈ 150 keeps 60 units active and
  restores the readout to 0.84 / 0.99. So the precision is the key free
  parameter. That is exactly FE-2 (learn π per channel), and the invariance
  gap is what temporal coherence (Phase 4 / FE-5) is meant to close.
- **Topography:** from a TICA start it is largely kept (8.1 → 5.6). From a
  random start a weak map emerges (0.9 → 2.1) only with re-seating; the TICA
  start also reaches lower F (1.15 vs 1.24), so it is the better optimum.
- F is still falling at 1,500 steps (mostly the prior term), so these runs are
  not fully converged.

## Section 7, FE-2 results (`configs/fe2.yaml`)

The precision π is now learned per V1 channel by minimizing F: held at FE-1's
value (log π = 3) for 500 steps, then freed, clamped to log π ∈ [−2, 8]
(plan 7.5). Run for V1 with and without divisive normalization (new
`V1Stage(normalize=False)`: log of raw energy); each variant gets its own V2
TICA fit.

| V1 | Precision | Held-out R² | Active units | Topography | 2nd-order texture (env 0.016 / 0.0625) | Learned log π, fine / mid / coarse |
|---|---|---|---|---|---|---|
| normalized | fixed (FE-1) | 0.960 | 38 | 5.6 | 0.64 / 0.83 | 3.0 / 3.0 / 3.0 |
| normalized | **learned** | 0.960 | 40 | 6.6 | 0.63 / 0.86 | 2.5 / 3.4 / 4.2 |
| unnormalized | fixed | 0.982* | 51 | 2.3 | 0.81 / 1.00 | 3.0 / 3.0 / 3.0 |
| unnormalized | learned | 0.982* | 47 | 2.2 | 0.81 / 1.00 | 2.6 / 3.1 / 3.4 |

\* Not comparable with the normalized rows. Without normalization every
channel shares a large per-patch log-contrast component, which is easy to
predict and inflates R².

Checks (declared before running): no runaway ✅, learned precision matches its
maximum-likelihood value ❌ (normalized ✅ within 0.01; unnormalized lags by
0.84, see below), second-order readout fixed ❌ (0.86 < 0.95), reconstruction
kept ✅.

Findings:
- **What the model learns to trust:** with normalization, precision rises
  steeply with scale (π ≈ 12 fine, 30 mid, 65 coarse). V2 predicts coarse V1
  channels well and fine ones poorly; oblique orientations get slightly more
  precision than cardinal ones. It matches the maximum-likelihood precision
  (1 / residual variance) within 0.05 per scale.
- **Learned precision does not fix FE-1's second-order readout, as
  predicted.** The contrast envelope rides on the fine carrier channels,
  exactly the ones learned precision down-weights. FE-1's diagnostic run
  (uniform π ≈ 150) worked because it up-weighted them. What precision to put
  on fine channels is a modeling choice that ML precision alone gets
  "wrong" for this purpose. Context-dependent precision (FE-3, attention as
  precision) is the plan's mechanism for up-weighting task-relevant channels.
- **Interaction with divisive normalization (the plan's FE-2 question):** the
  hypothesis that normalization equalizes precision across channels is
  rejected. Normalization makes precision *more* uneven (spread across
  channels 0.72 vs 0.33 in log π): by dividing out shared local contrast, it
  removes the easy-to-predict common component and leaves fine channels
  relatively noisier. It also partly divides out slow contrast modulation (the
  V1 pool blur is 4 px), which weakens the second-order signal: without
  normalization the FE latents decode the envelope far better (0.81 / 1.00 vs
  0.63 / 0.86). In the plan's Section 7 terms, divisive normalization already
  applies a locally inferred precision; learning a second, global per-channel
  precision on top of it is partly redundant with it and partly in conflict.
- **Without normalization, precision did not converge in 1,000 free steps.**
  It lagged its maximum-likelihood value by 0.76 and was still rising: as
  precision rises the decoder fits tighter, the residual shrinks, and the
  target rises again (the slow runaway plan 7.5 warns about). It never
  reached the bound. Normalization's bounded residuals let precision settle.

## Section 7, FE-3 results (`configs/fe3.yaml`)

The plan's FE-3 check (cat d′ in clutter; false alarms) needs categories and
Phases 5–7. This is its **V2-level analog**. Natural photo clutter has a
texture patch blended into the center:
- **target:** contrast modulation at 45°;
- **distractor:** modulation at 135° (the "non-cat");
- **absent:** unmodulated.

Added power is matched across kinds, and carrier orientation and envelope
phase are random, so only second-order structure separates target from
absent. Detection d′ (target vs absent) and false alarms (distractors
scored as targets, at a threshold catching 80% of targets) come from a
held-out linear readout of the posterior means.

Context precision needs inference that reads F (the amortized encoder
ignores Π at test time), so FE-3 adds settling: gradient descent on F from
the encoder's guess, with the step scaled by the accuracy term's curvature
(power iteration). Precision profiles, all on the same trained FE-2 model:
- **base:** FE-2's learned precision;
- **FE-3 context:** fitted by minimizing F on target-context scenes (the
  closed-form optimum, 1 / residual variance, iterated with settling);
- **expected-signal:** precision raised where the target is expected to add
  variance (attention as the precision of expected signals), as a comparator;
- **fine ±3:** extreme profiles, for diagnosis.

| Inference | Precision | d′ | False alarms (distractor) | Latent change vs encoder | Natural-image R² |
|---|---|---|---|---|---|
| amortized | — | **1.60** | **0.33** | — | 0.960 |
| 40 gradient steps | base | 1.60 | 0.34 | 11% | 0.966 |
| 40 gradient steps | FE-3 context | 1.59 | 0.33 | 7% | 0.966 |
| 40 gradient steps | expected-signal | 1.59 | 0.34 | 9% | 0.967 |
| converged (L-BFGS)* | base | 0.46 | 0.84 | 64% | 0.954 |
| converged* | FE-3 context | 0.47 | 0.85 | 62% | 0.961 |
| converged* | expected-signal | 0.45 | 0.86 | 61% | 0.965 |
| converged* | fine +3 | 0.51 | 0.81 | 73% | 0.971 |
| converged* | fine −3 | 0.19 | 0.86 | 78% | 0.884 |

\* Added after the first run, for diagnosis; not used by the checks.

Checks (declared before running): FE-3 raises d′ by ≥ 0.3 ❌ (−0.01); no
extra false alarms ✅.

Findings:
- **Context precision had no effect with short settling.** The accuracy term
  is stiff (curvature ≈ 1450), so 40 gradient steps move the latents only
  7–11%, and precision cannot act.
- **With converged settling, precision acts in the expected direction but
  cannot rescue the task.** Up-weighting fine channels helps (0.51) and
  down-weighting hurts (0.19), and the two profiles' latents differ by 73%.
  But converged inference itself wrecks detection (1.60 → 0.46; false alarms
  0.33 → 0.84) while reconstructing natural images as well as or better than
  the encoder. The model's own F optimum is indifferent to the invariant,
  TICA-like structure the task needs: with a linear decoder and a prior that
  treats locations independently, many latent codes explain a V1 map equally
  well, and settling drifts away from the encoder's structured one. This
  answers the plan's FE-4 question (amortized vs hybrid) early, and
  negatively, for this generative model: iterative inference hurts.
- **F-learned context precision is not attention.** Fitted on target
  scenes, it rises most on coarse channels (+1.4 log units), where the
  cluttered scenes are easy to predict, not on the channels that carry the
  target. Precision chosen by F alone tracks predictability, not relevance.
  The expected-signal comparator does pick the target's channels (fine and
  mid, strongest at 45°), but with this model no precision profile helps.
- Implication for the plan (Section 7.3): the attention half of each accuracy
  term (precision) needs its expectation half. The context must predict
  *content* (the top-down template of Phases 6–7), and the generative model
  must be strong enough that its F optimum keeps the structure the task
  depends on: spatially coupled latent priors (Phase 3 showed they matter),
  richer decoders, and invariance from temporal coherence (Phase 4).

## FE-3 follow-up: is the per-location prior the bottleneck? (`configs/fe3_prior_test.yaml`)

Same FE-2 model and FE-3 scenes. Converged (L-BFGS) settling with base
precision under four latent priors:
- the per-location TICA prior;
- a **spatial Gaussian** prior (`SpectralGaussianPrior`): a stationary
  Gaussian over the whole latent map with the full cross-spectral matrix,
  fitted to encoder latents on natural images. Its zero-frequency bin carries
  the per-patch component (Phase 3's global + local structure as an energy).
  It rates real latent maps about 7× more probable than spatially shuffled ones;
- both combined;
- no prior at all (control).

Detection is read two ways: from the latents (as in FE-3), and from the
reconstructed V1 maps through the Phase 2 TICA stage.

| Inference, prior | d′ from latents | d′ from reconstructed V1 maps | False alarms (recon. readout) | Latent change |
|---|---|---|---|---|
| encoder only | 1.60 | 2.09 | 0.24 | — |
| converged, TICA per-location | 0.46 | **2.26** | 0.20 | 64% |
| converged, spatial Gaussian | 0.92 | 2.19 | 0.18 | 37% |
| converged, TICA + spatial | 0.52 | **2.27** | 0.19 | 56% |
| converged, no prior | 1.16 | 2.13 | 0.21 | 108% |

Check (declared before running): the spatial prior recovers d′ ≥ 1.2 from
the latents ❌ (0.92).

Findings:
- **The per-location prior is not the bottleneck.** A spatially coupled prior
  helps the latent readout (0.46 → 0.92) but recovers well under the
  two-thirds criterion, and settling with no prior still loses a third of d′.
- **Converged settling does not lose target information; it moves it.**
  Read through the reconstructed V1 maps, every condition detects the target
  better than the encoder's latents do (2.1–2.3 vs 1.60), and TICA-prior
  settling gives the best detection and the fewest false alarms. FE-3's
  "collapse" was a coordinate effect: settling shifts the target signal into
  latent directions (along directions the decoder barely sees, and under the
  prior's shrinkage) that a spatially averaged linear readout of the latents
  cannot use. This is FE-1's "map vs detector" split again.
- Implication: FE-3's precision conditions should be re-scored with the
  reconstruction readout before concluding that context precision cannot act
  as attention (done below). And the latent code needs to stay detector-like
  under inference (invariance from temporal coherence, or a readout stage that
  re-applies the fixed second-order front end, as the reconstruction route does).

### FE-3 re-scored with the reconstruction readout

`fe3_context_precision.py` now reports both readouts, plus the reconstruction
readout applied directly to the input scene V1 maps (the ceiling that closer
reconstructions approach).

| Condition | d′ via latents | d′ via reconstruction | False alarms (recon.) |
|---|---|---|---|
| **input V1 maps (no model)** | — | **2.20** | 0.18 |
| encoder only | 1.60 | 2.09 | 0.24 |
| 40 steps, base / FE-3 context / expected-signal | 1.59–1.60 | 2.12–2.14 | 0.19–0.20 |
| converged, base | 0.46 | 2.26 | 0.20 |
| converged, FE-3 context | 0.47 | 2.22 | 0.19 |
| converged, expected-signal | 0.45 | 2.22 | 0.21 |
| converged, fine +3 | 0.51 | 2.24 | 0.19 |
| converged, fine −3 | 0.19 | **0.97** | 0.52 |

Re-score check (declared before the re-run, same thresholds): FE-3 context
precision raises reconstruction-readout d′ by ≥ 0.3 over base ❌ (−0.04); no
extra false alarms ✅.

- **The test is saturated, so it cannot show attention.** With base
  precision, converged settling already reconstructs the target structure as
  well as the input carries it (2.26 vs 2.20). No precision profile can add
  information the input lacks, so there is no headroom for a gain.
- **Precision does control what the model keeps.** Cutting fine-channel
  precision halves detection (0.97) and more than doubles false alarms, while
  every other profile stays at ceiling. Attention-as-precision here can
  protect task-relevant information but has nothing to add in this regime.
- To test attention as a *gain*, the task needs headroom between base
  reconstruction and the input ceiling. The plan's Phase 6 gain model
  (attention scaling the drive before normalization) is the other route: it
  changes the representation itself, not only how faithfully the input is
  reconstructed.

### Looking for headroom (`--calibrate-headroom`)

Rule fixed before measuring: find a target contrast where base-precision
reconstruction falls at least 0.3 d′ below the input ceiling (with the ceiling
≥ 1.0), then rerun FE-3 there; if none exists, report that.

| Target contrast | Input ceiling d′ | Converged base reconstruction d′ | Headroom |
|---|---|---|---|
| 0.15 | 0.52 | 0.55 | −0.02 |
| 0.25 | 0.82 | 0.92 | −0.10 |
| 0.35 | 1.18 | 1.26 | −0.08 |
| 0.50 | 1.03 | 1.08 | −0.05 |

**No headroom exists along this axis.** Lowering target contrast is the same
axis as raising clutter, since the scene is clutter plus a scaled target.
At every level the reconstruction matches or slightly beats the input: it acts
as a mild denoiser, keeping the target structure and dropping some clutter.
(The 0.35 vs 0.5 ordering is within split-to-split noise.) Detection is always
limited by the input, never by the model, so context precision has nothing to
recover and the FE-3 attention test cannot show a gain at V1–V2 on this task.

The one regime where this model does lose target information is structural:
envelopes above the latent grid's resolution (the V2 grid is 4 px, so
modulations above ~0.125 cycles/px cannot be represented; compare the Phase 2
envelope sweep). Precision cannot recover a representational limit, so that
would not be a fair attention test either. Attention-as-precision needs a
bottleneck that weighting can move: capacity limits in a deeper hierarchy
(Phase 5), or the gain route of Phase 6.

## Phase 4 results (`configs/phase4.yaml`)

No video in the repo, so training uses the plan's alternative: **simulated
drift** over the photos (`gtv.data.drift_pairs`). Each pair is a crop and the
same crop shifted ≤ 2 px, rotated ≤ 3° and rescaled ≤ 3%. Design B V2 stages
are fitted on 256 such pairs (36,864 location pairs), all from the same
random start:
- **still:** TICA on each frame independently;
- **bubbles** (Hyvärinen, Hurri & Väyrynen): energy pooled over the sheet
  neighborhood and both frames, √(h(s_t² + w·s_{t+1}²)), rewarding energy
  that persists in the same pool;
- **coherence** (Hurri & Hyvärinen): TICA minus λ·E[Σ s_i(t)² s_i(t+1)²],
  temporal coherence of activity levels.

The plan's "slowness penalty" was tried first and dropped. Linear slowness is
provably inert with a complete orthonormal W (Σ E[(Δs)²] = trace(WCWᵀ) is
rotation-invariant). An unnormalized penalty on pooled-energy differences is
minimized by mixing sources, not by persistence. Both are covered by tests.
The pre-declared sweep named slowness weights; activity coherence replaced
it with weights scaled to its term, and the pass rule was unchanged.

Invariance is the median, over units, of the correlation across 128 held-out
patches between a unit's pooled energy for an image and for a transformed
version (same location).

| Model | Shift 2 px | Shift 4 px | Rotate 10° | Scale 1.1 | Mean | Drift coherence | Orientation selectivity | 2nd-order texture (0.016 / 0.0625) |
|---|---|---|---|---|---|---|---|---|
| still | 0.786 | 0.543 | 0.346 | 0.608 | 0.571 | 0.792 | 0.048 | 0.88 / 1.00 |
| bubbles w = 1 | 0.795 | 0.560 | 0.375 | 0.623 | **0.588** | 0.791 | 0.042 | 0.89 / 1.00 |
| bubbles w = 3 | 0.785 | 0.550 | 0.346 | 0.619 | 0.575 | 0.788 | 0.049 | 0.89 / 1.00 |
| coherence 0.3 | 0.791 | 0.540 | 0.351 | 0.625 | 0.577 | 0.788 | 0.045 | 0.90 / 1.00 |
| coherence 1 | 0.782 | 0.535 | 0.342 | 0.625 | 0.571 | 0.784 | 0.054 | 0.90 / 1.00 |
| coherence 3 | 0.779 | 0.522 | 0.327 | 0.613 | 0.560 | 0.778 | 0.060 | 0.90 / 1.00 |

First-order texture and junction decoding stay at 0.99–1.00 for every model.

Check (declared before running): some temporal model improves mean invariance
by ≥ 0.05 while keeping decoding (within 0.02) and orientation selectivity
(≥ 0.8×) ❌. Selectivity is kept by every model; the best invariance gain is
+0.017 (bubbles w = 1).

Findings:
- **Temporal learning barely moves Design B.** Its invariance is mostly set
  by the fixed front end: the pooled first-order channels are already highly
  invariant (median 0.94), the second-order channels less so (0.71). Learning
  only rotates channels within the 64 whitened dimensions.
- **But there is headroom within that space.** The directions slowest under
  the test transforms reach energy invariance 0.87 (median direction 0.54).
  The temporal terms at these weights do not pull W there against TICA's
  sparsity. More weight did not help either: bubbles w = 3 and coherence λ = 3
  score below w = 1. Activity coherence also rewards heavy-tailed units, not
  only persistent ones, so it is partly confounded with sparsity.
- Orientation selectivity of Design B's pooled energies is low for every
  model (≈ 0.05), since its units combine energy-like features. The selectivity
  check therefore rests mainly on the probe decoding, which is at ceiling.
- Where temporal coherence should matter more: stages whose *features* are
  learned (Design A's spatial filters over V1 maps, or the free-energy
  encoder), and longer sequences (pooling over more than two frames, larger
  drifts), where persistence is a stronger signal than one small step.

## Phase 5 results (`configs/phase5.yaml`)

**Categories:** Fashion-MNIST (10 clothing classes), downloaded once from the
dataset's GitHub repository into `data_cache/` (git-ignored). CIFAR-10 and
Hugging Face are blocked by this environment's network policy. Images are
upsampled 2× and centered on the 64 × 64 canvas. The repo's `LittleCarDb1` is
a die-cast car catalogue (96 models × ~7 paint jobs, one viewpoint): a
possible later test of shape identity across colour, but too few images per
class to learn categories from.

**Stack:** V1 and the Design B V2 as validated on natural photos. V4, PIT
and AIT (`HigherStage`) repeat the Design B block on the stage below's TICA
output: fixed Gabors (4 orientations, 0.25 cycles per map cell) on every
input channel, energy, normalization, log, 2× pooling, plus the pooled input
channels, then group-whitened complete TICA. They are fitted greedily and
without labels on 4,000 Fashion-MNIST images.

| Stage | Map | Units (sheet) | TICA radius | Skip in |
|---|---|---|---|---|
| V4 | 8 × 8 | 100 (10 × 10) | 1 | V1 |
| PIT | 4 × 4 | 144 (12 × 12) | 2 | V2 |
| AIT | 2 × 2 | 196 (14 × 14) | 2 | — |

**Results** (linear readout of 2×2-pooled [s, pooled energy]; 3,000 train /
2,000 held-out test images; tolerance = accuracy on transformed test images
with the decoder trained on originals):

| Stage | Accuracy | Shift 6 px | Rotate 15° | Scale 0.85 |
|---|---|---|---|---|
| pixels | 0.814 | 0.36 | 0.50 | 0.66 |
| V1 | 0.784 | 0.65 | 0.70 | 0.71 |
| V2 | **0.863** | **0.78** | **0.74** | **0.79** |
| V4 | 0.857 | 0.76 | 0.69 | 0.78 |
| PIT | 0.843 | 0.72 | 0.61 | 0.76 |
| AIT | 0.791 | 0.65 | 0.52 | 0.70 |

**AIT sheet clustering:** each unit's preferred category comes from its own
unpooled energy (pooled energy averages sheet neighbors, which would build
clustering in). The fraction of sheet neighbors sharing a preferred category is
**0.26** vs 0.11 for shuffled sheets (p < 0.0005, 2,000 shuffles); for
superordinate groups (tops / footwear / other) it is **0.47** vs 0.33
(p < 0.0005). The map shows footwear, trouser, T-shirt and bag regions.

Checks (declared before running): AIT decoding ≥ 0.5 ✅ (0.79; chance 0.1);
AIT category clusters above the 95th percentile of shuffled sheets ✅.

Findings and caveats:
- **Topographic category clusters emerge without labels.** TICA's
  arrangement puts units with correlated energies together, and units driven
  by the same category co-activate.
- **But the stack above V2 adds neither category information nor
  invariance; it loses some.** Accuracy peaks at V2 (0.86) and falls to
  0.79 at AIT, below raw pixels. Rotation tolerance falls from 0.85 (V2) to
  0.66 (AIT). Fixed Gabors on TICA outputs plus unsupervised greedy TICA do
  not build more abstract features here. Each stage's complete TICA is only a
  rotation of its whitened features, and the fixed front ends discard
  information at each 2× pool. Candidates: learned (not fixed) upper-stage
  filters, more units than a complete basis (RICA), temporal coherence at the
  stages whose features are learned, and the plan's generative/top-down path.
- **The curvature probe is saturated** (V1–PIT all ≈ 1.0 for three
  curvature levels at any orientation), so it does not show V4 curvature
  tuning. Harder probes (curvature with matched length and orientation
  statistics, or Pasupathy–Connor shapes) are needed.
- Two robustness fixes came out of this phase: the whitener floors
  near-zero eigenvalues (a skip connection can be spanned by a stage's own
  channels; `n_floored` reports it), and orthonormalization runs in float64
  with a QR fallback (large sheets with 5 × 5 pools made float32 SVD fail
  intermittently).
- Deferred from the plan's Phase 5: log-polar input, and the generative
  decoders with wake–sleep (or free energy) for the full stack.

## Phase 6 results (`configs/phase6.yaml`)

Uses the Phase 5 stack as saved (`runs/phase5/phase5_stages.pt`); no model
weights are refitted.

**Top-down path** (built here; deferred in Phase 5): convolutional FA
decoders AIT → PIT → V4, fitted by regression on the stack's own activity for
2,000 single items (`gtv.generative.topdown`). V4 outputs are mapped to V4's
energy channels by the pseudo-inverse of its whitening + TICA. A category
template: clamp AIT to the category's mean activity and decode down.
**Templates are category-specific:** for 8 of 10 categories the decoded
template correlates best with that category's actual V4 activity on held-out
items (median diagonal correlation 0.71) ✅.

**Attention** at V4, as the normalization model of attention: the gain
multiplies V4's Gabor energies before divisive normalization (`gain` argument
of `HigherStage`). Two fields:
- feature-only gain A_c = exp(β z_c), with z_c the target template's
  deviation from the mean template;
- (added after the first run, check declared before it) spatial
  feature-similarity gain A(x, y) = exp(β m(x, y)), with m the local match
  between V4 features and the template, z-scored per image (Treue &
  Martinez-Trujillo). No location cue is needed.

**Task:** a sneaker among 3 other Fashion-MNIST items (native 28 px) on the
64 px canvas, vs 4 distractors (absent), vs a sandal or ankle-boot lookalike
(false alarms), 240 scenes each. d′ comes from a held-out linear readout of
AIT, retrained per condition.

| Condition | AIT d′ | False alarms (lookalike) | V4 d′ |
|---|---|---|---|
| no attention | 0.37 | 0.73 | 0.60 |
| feature gain β = 0.5 / 1 / 2 | 0.39 / **0.45** / 0.44 | 0.75 / 0.78 / 0.78 | 0.64 / 0.65 / 0.67 |
| feature gain, wrong template (bag) | 0.41 | 0.74 | 0.68 |
| spatial field β = 0.5 / 1 / 2 | 0.41 / 0.38 / **0.56** | 0.71 / 0.75 / 0.78 | 0.60 / 0.60 / 0.63 |
| spatial field, wrong template (bag) | 0.37 | 0.73 | 0.58 |

Checks (declared before the respective runs): templates category-specific ✅;
feature gain raises d′ by ≥ 0.3 ❌ (+0.08); spatial field raises d′ by ≥ 0.3
❌ (+0.19); spatial field target-specific by ≥ 0.2 ❌ (0.19).

Findings:
- **Feature-only gain is information-neutral in this network.** 99.4% of its
  effect on V4 features is a constant offset per channel: the log after
  normalization turns multiplicative gain into an additive shift, which
  downstream standardization removes. Even before the log, normalized
  responses with and without gain correlate at 0.994. The network is
  noiseless, so per-channel rescaling cannot add information, and a
  wrong-category template does as well as the right one. Attention as gain
  needs a bottleneck after the gain: noise, saturation or limited capacity.
  (The same lesson as FE-3's missing headroom.)
- **The spatial field acts on the real bottleneck, but not yet reliably.**
  Clutter hurts through pooling (d′ falls from 0.60 at V4 to 0.37 at AIT as
  receptive fields mix the target with its neighbours). A field that
  suppresses non-matching locations before pooling gave the only sizeable,
  specific gain (β = 2: +0.19, wrong template +0.00), but it is below both
  bars, not monotonic in β, and within split-to-split noise (roughly ±0.1–0.15
  at 240 scenes per kind).
- **Attention raises false alarms** on lookalikes (0.73 → 0.78 at the
  strongest settings), the plan's motivation for Phase 7's expectation channel.
- A small tuning shift toward the target (Çukur et al.): with β = 1 feature
  gain, 59% of V4 units shift their relative preference toward sneakers
  (median log ratio +0.06).
- The base detection is weak (AIT d′ 0.37), and the stack loses information
  above V2 (Phase 5). A stronger upper hierarchy is the precondition for a
  decisive attention test. Larger scene sets would also shrink the d′ noise.

## Related code elsewhere in this repo

- `fashionmnist/oriented_powermap_2d.py` — Keras oriented power maps (Gabor energy); reference for Phase 1.
- `zebrastack/rxndiff/` and `examples/hierarchical_rxndiff.ipynb` — Stage-1 top-down routing on Gabor-power channels.
- `activation-atlas/gabor_objectives.py` — Gabor-parameterized objectives for feature visualization.
