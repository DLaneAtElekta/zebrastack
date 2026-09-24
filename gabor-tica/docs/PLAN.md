# Gabor–TICA Ventral Stream with Thalamocortical Modulation
## Implementation Plan

---

## 1. Goal

Build a biologically aligned, largely unsupervised visual hierarchy (V1 → V2 → V4 → PIT → AIT) in which:

- every stage carries **fixed or tethered spatial Gabor filtering** (developmental prior),
- **learning happens mainly in cross-channel reductions**, via topographic ICA (TICA),
- a **generative (top-down) path** is trained with wake–sleep,
- **invariance** comes from temporal coherence on video, and
- a **thalamic module** uses the generative path to modulate each layer under task context ("cat mode"), keeping attention and expectation as separate channels, and
- optionally, all of the above is trained under **one variational free-energy objective** (Section 7).

The work is prototype-driven. Every phase ends with a runnable system and a concrete check before anything more is added.

---

## 2. Architecture Summary

### 2.1 Stage block (repeated per ventral area)

```
input map (channels C, H×W)
   │
   ├─► [1] Spatial Gabor bank per channel   (fixed / tethered; coarser scale each stage)
   ├─► [2] Energy: |quadrature pair|²
   ├─► [3] Attention field A (thalamic gain)       ← Phase 6
   ├─► [4] Divisive normalization + log
   ├─► [5] Spatial pooling (2× decimation)
   └─► [6] TICA across channels (learned, topographic sheet)
   │
output map (C', H/2 × W/2)  → next stage (+ skip to stage n+2)
```

### 2.2 Hierarchy

| Stage | Input | Expected emergent selectivity | Relative RF size |
|---|---|---|---|
| V1 | pixels (optionally log-polar) | orientation, spatial frequency, complex cells, pinwheels | 1× |
| V2 | V1 energy | texture statistics, junctions, second-order boundaries | ~2–3× |
| V4 | V2 (+ V1 skip) | curvature, contour fragments | ~6× |
| PIT/TEO | V4 (+ V2 skip) | object parts, configurations | ~15× |
| AIT/TE | PIT | view-tolerant objects, category clusters | ~30×+ |

### 2.3 Parallel paths

- **Recognition (bottom-up):** the stage blocks above.
- **Generative (top-down):** a mirror decoder from each stage to the stage below, trained in the wake phase.
- **Thalamic module:** per-stage attention and expectation fields computed from top-down generative templates.

---

## 3. Guiding Principles

1. **Fixed before learned.** Start each component fixed or closed-form, and swap in learning only once the fixed version works.
2. **Two stages before five.** Prove V1 + V2 thoroughly before extending the hierarchy.
3. **Every phase has a probe.** Use synthetic stimuli with known ground truth (gratings, texture pairs, curvature shapes) alongside natural data.
4. **Keep the generative path honest.** Inspect fantasies visually at every stage, since they are the best diagnostic of what the model believes.
5. **Separate knobs.** Gain, expectation, neighborhoods and routing are independent modules that can each be toggled.

---

## 4. Tech Stack

- **Python 3.11+, PyTorch** as the core framework
- **Kymatio**: scattering transforms (a Design B prototype comes almost for free)
- **NumPy/SciPy** for Gabor construction, log-polar resampling and whitening
- **Matplotlib / Plotly** for filter atlases, topographic maps and fantasy grids
- **Weights & Biases or TensorBoard** for experiment tracking
- **Hydra or plain YAML configs** so stages can be swapped declaratively

### Suggested repo layout

```
gtv/
  configs/
  data/            # loaders: natural images, video, category sets, probes
  stages/
    gabor.py       # banks, quadrature pairs, energy, log-polar sampling
    normalize.py   # divisive normalization (Heeger-style)
    tica.py        # TICA / RICA loss, topographic neighborhoods
    block.py       # full stage block
  generative/
    decoders.py    # per-stage top-down models
    wake_sleep.py
  temporal/
    coherence.py   # bubbles / slowness objectives
  thalamus/
    gain.py        # attention field (Reynolds–Heeger)
    expectation.py # predictive subtraction
    routing.py     # context neighborhoods, pulvinar routing (stretch)
  probes/          # gratings, textures, Pasupathy–Connor shapes, clutter
  viz/
  experiments/
```

---

## 5. Data

| Purpose | Source (suggested) |
|---|---|
| Natural image statistics | van Hateren natural images, or a natural-scene subset of ImageNet |
| Temporal coherence | natural video (egocentric or nature footage), or simulated saccade and drift sequences over still images |
| Category emergence and "cat" tests | ImageNet / COCO category subsets (cat vs. distractors), cluttered composites |
| Developmental phase | synthetic retinal-wave-like correlated noise |
| Probes | generated gratings, contrast-modulated textures, curvature shape set, texture-boundary stimuli |

---

## 6. Phases

### Phase 0: Infrastructure and Probes

**Build**
- Data loaders, config system and experiment tracking.
- Probe generators: drifting gratings, second-order (contrast-modulated) gratings, texture pairs, curvature shapes, cluttered target scenes.
- Visualization: filter atlas, topographic sheet viewer (color by preferred feature), fantasy grid.

**Exit criteria:** probes render, and a dummy identity stage passes end to end through the viz tools.

---

### Phase 1: Developmental V1

**Build**
- Quadrature Gabor bank (L orientations × J scales), parameterized by frequency, orientation, envelope width and aspect ratio.
- Energy, divisive normalization, then log.
- *Optional developmental variant:* learn the V1 filters with Hebbian or sparse learning on retinal-wave-like noise, and confirm Gabor-like receptive fields emerge.

**Validate**
- Orientation and SF tuning curves on gratings.
- Contrast invariance of tuning after normalization.
- Heavy-tailed raw energy distributions become approximately Gaussian after the log.

**Exit criteria:** clean tuning curves; normalized log-energy is roughly Gaussian.

---

### Phase 2: V2 Block (Design A vs. Design B)

**Build**
- **Design B (primary):** second-order Gabor filtering of each V1 channel at a coarser scale (Kymatio 2-layer scattering as the first prototype). Keep only frequency-decreasing paths to control channel growth (order-2 paths ≈ L²·J(J−1)/2).
- **Design A (baseline):** learned V2 filters directly on the V1 energy maps.
- In both, apply **TICA across channels** on a 2D topographic sheet.

**TICA objective (sketch)**

```python
# s = W @ x  (x: normalized log-energy patch/channel vector)
# h: fixed topographic neighborhood kernel on a 2D sheet
pooled = conv2d_on_sheet(s**2, h)              # local energy
loss   = torch.sqrt(pooled + eps).sum()          # sparsity of pooled energy
loss  += lam * ((W.T @ W @ x - x)**2).sum()      # RICA reconstruction (allows overcomplete W)
```

Start with complete TICA on whitened input using orthonormal W, then switch to the RICA form for overcompleteness.

**Validate**
- Texture-pair discrimination and texture-boundary detection.
- Junction and corner selectivity on probes.
- Smoothness of the topographic map, and the presence of feature clusters.
- A–B comparison: selectivity, sparsity and linear decodability of texture classes.

**Exit criteria:** B matches or beats A on texture and junction probes. If not, revisit the Gabor scales or the TICA neighborhood size.

---

### Phase 3: Generative Path and Wake–Sleep (V1–V2)

**Build**
- **Step 3a:** the simplest generative model: factor analysis from V2 latents down to V1 log-power maps (generating power maps only, not pixels).
- **Step 3b:** wake–sleep loop:
  - *Wake:* recognition infers V2 latents from real images, and the generative weights are updated.
  - *Sleep:* the generative model fantasizes, and the recognition (TICA/reduction) weights are updated.
- **Step 3c:** swap factor analysis for a TICA-style generative model of dependent variances (Karklin–Lewicki flavor).
- Consider **reweighted wake–sleep** if sleep-phase bias or mode-averaging hurts.
- **Gabor tether:** keep layer-1 filters fixed, or allow drift along the Gabor parameter family (on "rails") with a prior pulling toward the developmental values.

**Validate**
- Fantasies look like plausible V1 power maps.
- Reconstruction error on held-out images.
- Wake–sleep stays stable over training (no collapse of the recognition path).

**Exit criteria:** stable training and recognizable fantasies at the V1 power-map level.

> **Alternative:** Section 7 replaces wake–sleep from this phase onward with a single free-energy objective (step FE-1).

---

### Phase 4: Temporal Coherence

**Build**
- Video or saccade-sequence loader.
- Add a temporal-coherence term to TICA (bubbles-style: correlations of pooled energy across adjacent frames), or add a slowness penalty on V2 outputs.

**Validate**
- Invariance index: response stability under translation, small rotation and scale change, while selectivity is preserved.
- Compare against the Phase 2 model trained on still images only.

**Exit criteria:** invariance improves measurably without losing selectivity.

---

### Phase 5: Extend the Hierarchy (V4 → PIT → AIT)

**Build**
- Repeat the stage block for V4, PIT and AIT.
- Add skip connections (V1 → V4, V2 → PIT).
- Switch the input to log-polar sampling (cortical magnification).
- Increase TICA neighborhood sizes per stage, and let retinotopy degrade naturally.
- Extend the generative decoders and wake–sleep to the full stack. Train greedily, layer by layer, then fine-tune jointly.

**Validate**
- V4: curvature tuning on the shape probe set.
- PIT/AIT: linear decodability of object categories, and the degree of view tolerance.
- AIT sheet: category clustering (look for face-, body- and animal-like patches).
- Fantasies clamped at the top layer produce category-consistent lower-layer patterns.

**Exit criteria:** above-chance category decoding at AIT with emergent topographic category clusters.

---

### Phase 6: Thalamic Gain (Attention Field)

**Build**
- **Template generation:** clamp the AIT latents to "cat", then run the generative path downward to get the expected activity at each stage.
- **Attention field:** convert each stage's template into a per-channel (then spatial × channel) gain map A.
- **Apply using the normalization model of attention:**

```python
drive = A * E                               # E: stage energy
R     = drive / (sigma + pool_kernel(drive))  # divisive normalization
```

- Start with **feature-only gain at one stage (V4)**, then extend to all stages and add spatial fields.

**Validate**
- Cat-detection sensitivity (d′) in cluttered scenes, with and without attention.
- Tuning shifts: do unit preferences move toward cat features, in the direction of the Çukur et al. effect?

**Exit criteria:** d′ improves in clutter with no retraining. The generative weights alone provide the templates.

---

### Phase 7: Expectation Channel

**Build**
- A separate prior channel: subtract (or divisively explain away) the predicted component from each stage's input before passing it up, so units carry prediction error.
- Attention (precision/gain) and expectation (subtraction) become independent controls.

**Validate**
- **False-alarm test:** the attention-only model versus attention + expectation on non-cat images. The goal is fewer hallucinated cats.
- **Expectation suppression:** expected cats evoke lower but more selective responses (a Kok et al.–style sharpening signature).

**Exit criteria:** the false-alarm rate drops while d′ is maintained.

---

### Phase 8 (Stretch): Context-Dependent Topography and Routing

**Build**
- **Context neighborhoods:** make the TICA pooling kernel h(i,j) a function of context (a low-rank modulation driven by the thalamic signal), so cat-relevant units pool together.
- **Pulvinar routing:** shifter-style gating that selects which region and scale feeds the next stage (after Olshausen, Anderson & Van Essen 1993), giving object-centered processing.

**Validate**
- Object-centered invariance under attention.
- Compare gain-only against gain + neighborhoods + routing on detection and localization.

**Exit criteria:** measurable gain over Phase 7 on at least one of detection, localization or invariance.

---

## 7. Unified Variational Free-Energy Training (Alternative Track)

This track replaces the separate wake–sleep, attention and expectation modules with **one objective**, the variational free energy F. It is minimized with respect to beliefs (fast, per input) and weights (slow, across data). Phases 1–2 stay as they are. This track can replace Phase 3 onward, or run alongside it as a comparison.

### 7.1 Generative Model

- Latents z₁…z_L, one per stage; the top latent z_L is conditioned on a context c (e.g., "cat").
- Each level predicts the one below: gₗ₊₁(zₗ₊₁) → zₗ, and g₁(z₁) → x (V1 normalized log-energy).
- Prediction errors:

```
ε₀ = x  − g₁(z₁)
εₗ = zₗ − gₗ₊₁(zₗ₊₁)
```

### 7.2 Objective

```
F = Σₗ [ ½ εₗᵀ Πₗ εₗ − ½ log|Πₗ| ]     accuracy (precision-weighted)
  + Σₗ −log p_TICA(zₗ)                topographic sparsity prior
  − H[q(z|x)]                         complexity / entropy
  + ½ λ ‖θ − θ_Gabor‖²                developmental tether (hyperprior on parameters)
  + Σₜ −log p(zₜ | zₜ₋₁)              temporal-coherence prior (video)
```

with the TICA prior

```
−log p_TICA(z) = Σᵢ √( Σⱼ hᵢⱼ zⱼ² + ε )
```

### 7.3 Mapping Plan Components to Terms

| Plan component | Free-energy term |
|---|---|
| TICA (Phase 2) | topographic sparse prior on each zₗ |
| Divisive normalization (Phase 1) | inference of a local scale latent (Gaussian scale mixture); dividing by it applies a locally inferred precision |
| Gabor tether (Phase 3) | Gaussian hyperprior on θ (variational Bayes over parameters) |
| Temporal coherence (Phase 4) | dynamical prior p(zₜ \| zₜ₋₁) |
| Expectation channel (Phase 7) | the prediction g(z) subtracted inside εₗ; ascending signals carry error |
| Attention / thalamic gain (Phase 6) | the precision Πₗ weighting each error; −log\|Π\| prevents runaway precision |
| Thalamic control ("cat mode") | hyper-latent predicting log Πₗ from context c |

The attention and expectation modules from Phases 6–7 are therefore not separate components here: they are the two halves of every accuracy term.

### 7.4 Optimization Options

1. **Amortized (VAE-style).** Stage blocks act as the encoder q(z|x), trained through reparameterization. Use a ladder or hierarchical VAE layout (Ladder VAE, NVAE) that merges bottom-up and top-down signals at each level, analogous to cortical feedforward/feedback convergence. Fast, but limited ability to use context dynamically.
2. **Iterative predictive coding** (Rao & Ballard 1999; Friston 2005). Settle z by gradient descent on F for each input, then update the weights. Slower, but precision and context reshape inference dynamically.
3. **Hybrid (recommended).** Initialize with the amortized encoder, then run a few settling steps (iterative amortized inference, Marino et al. 2018).

```python
# Hybrid inference + learning (sketch)
z = encoder(x)                          # amortized initial guess
for _ in range(K):                      # K ≈ 5–20 settling steps
    F = free_energy(x, z, theta, Pi(c))
    z = z - eta_z * grad(F, z)
F = free_energy(x, z, theta, Pi(c))
theta, phi = step(F)                    # generative + encoder weights (slow)
```

### 7.5 Practical Cautions

- **No closed-form KL** under sparse/TICA priors, so use Monte Carlo estimates.
- **Posterior collapse** in upper layers: use ladder-style top-down inference, skip connections and KL warm-up.
- **Runaway precision:** bound log Π, or hold it fixed at first, then free it gradually.

### 7.6 Phased Path for This Track

| Step | Build | Check |
|---|---|---|
| FE-1 | V1→V2, Gaussian q, TICA prior on z₂, fixed Π = I, amortized | reconstruction; TICA-like maps emerge as in Phase 2 |
| FE-2 | learnable per-channel Π | how learned precision interacts with divisive normalization |
| FE-3 | Π conditioned on context c | cat d′ in clutter (Phase 6) and false-alarm rate (Phase 7), with no separate attention module |
| FE-4 | add K settling steps | amortized vs. hybrid on detection and calibration |
| FE-5 | temporal prior on video; extend to V4 → AIT | invariance index; category clustering (Phases 4–5) |

---

## 8. Evaluation Matrix

| Phase | Primary metric | Probe / data |
|---|---|---|
| 1 | tuning-curve quality, contrast invariance | gratings |
| 2 | texture/junction selectivity, map smoothness | texture pairs, junction probes |
| 3 | reconstruction error, fantasy plausibility | held-out natural images |
| 4 | invariance index vs. selectivity | video, transformed images |
| 5 | category decoding, topographic clustering | category subsets |
| 6 | d′ in clutter, tuning shift | cluttered cat scenes |
| 7 | false-alarm rate, expectation suppression | non-cat and expected-cat sets |
| 8 | localization, object-centered invariance | cluttered scenes with offsets |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Channel explosion in Design B | frequency-decreasing path pruning; aggressive TICA reduction; cap channels per stage |
| Wake–sleep instability / sleep trained on garbage fantasies | greedy layer-wise training; reweighted wake–sleep; warm-start recognition from Phase 2 |
| Phase loss makes pixel-level generation hard | generate power maps only at first; add complex amplitude/phase latents later if pixels are needed |
| Weak invariance at the top | temporal coherence (Phase 4) before extending the hierarchy |
| Attention causing hallucinations | separate expectation channel (Phase 7); calibrate gain magnitudes |
| Topography not emerging at AIT | tune neighborhood size; add a smoothness penalty on the sheet (TDANN-style) |
| Posterior collapse in upper layers (free-energy track) | ladder-style top-down inference; skip connections; KL warm-up |
| Runaway learned precision | bound log Π; start fixed, free gradually |
| Compute cost of a deep scattering stack | pool aggressively; restrict orientations/scales at higher stages; use GPU-accelerated Kymatio |

---

## 10. Open Design Decisions

1. **Compression scope per stage:** across channels only, or across small spatial neighborhoods too? (Neighborhoods likely matter from V2 upward.)
2. **Gabor adaptation:** fully fixed, drift along the Gabor family (on rails), or free with a tether?
3. **Developmental phase:** hand-specified Gabors, or learned from retinal-wave noise?
4. **Generative model family per stage:** factor analysis → TICA-variance model → something more expressive?
5. **Attention field granularity:** per channel, per location, or both?
6. **Training objective framing:** plain wake–sleep (Phase 3 onward) or the unified free-energy track (Section 7)? If free energy: amortized, iterative, or hybrid inference, and how many settling steps?

---

## 11. Key References

- Felleman & Van Essen (1991): distributed hierarchical processing in primate cortex
- Hyvärinen & Hoyer (2001): topographic ICA and complex cells
- Hyvärinen, Hurri & Väyrynen (2003): "bubbles" framework (sparseness, topography, temporal coherence)
- Le et al. (2011, 2012): RICA; large-scale unsupervised feature learning (stacked TICA-like)
- Bruna & Mallat (2013): invariant scattering convolution networks
- Oyallon et al.: hybrid scattering + CNN
- Chubb & Sperling; Landy & Graham: filter–rectify–filter second-order vision
- Heeger (1992); Schwartz & Simoncelli (2001): divisive normalization
- Karklin & Lewicki (2009): emergence of complex-cell properties by learning to generalize in natural scenes
- Hinton, Dayan, Frey & Neal (1995): wake–sleep
- Rao & Ballard (1999): predictive coding in visual cortex
- Friston (2005): a theory of cortical responses
- Wainwright & Simoncelli (2000): Gaussian scale mixtures and divisive normalization
- Sønderby et al. (2016): Ladder VAE
- Vahdat & Kautz (2020): NVAE
- Marino, Yue & Mandt (2018): iterative amortized inference
- Bornschein & Bengio (2015): reweighted wake–sleep
- Freeman & Simoncelli (2011, 2013): V2 and texture statistics
- Pasupathy & Connor (2001): shape/curvature coding in V4
- Reynolds & Heeger (2009): normalization model of attention
- Çukur et al. (2013): attention warps semantic representation
- Olshausen, Anderson & Van Essen (1993): dynamic routing
- Saalmann & Kastner: pulvinar and cortical synchrony
- Kanai et al. (2015): predictive processing, precision and the pulvinar
- Kok, Jehee & de Lange (2012): expectation sharpens sensory representations
- Lee et al. (TDANN); Blauch, Behrmann & Plaut (2022): topographic deep networks
