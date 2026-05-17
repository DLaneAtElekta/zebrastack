# Stage-1 follow-up: top-down routing for Gabor-power channels

This document is the running plan for the work in PR #4 — what this PR ships,
what was deliberately left out, the test that decides whether we need Stage 2,
and the Stage-2 design itself. Lives next to the code so it's easy to amend as
we learn.

## What this PR ships (Stage 1)

Three new modules and a test file under `resnet18-vae/`:

| File | Purpose |
| --- | --- |
| `topdown_se.py` | `TopDownRoutedSE` — SE block with an optional top-down vector `t` projected to an additive logit bias on the channel gate. `td_proj` is zero-init, so the module is identical to baseline `SqueezeExcitation` at construction. |
| `topdown_oriented_powermap.py` | `TopDownOrientedPowerMap` — `OrientedPowerMap` subclass that swaps the inner SE for `TopDownRoutedSE` (weights copied) and exposes `forward(x, t=None)`. |
| `topdown_encoder.py` | `TopDownEncoder` — two-pass encoder. Pass 1 = pure bottom-up to coarse `mu_0`. Pass 2 = project `mu_0` to `t_v1`, `t_v2` via zero-init heads, bias V1/V2 gates, rerun the upper stack. |
| `test_topdown.py` | Six unit tests: SE no-op invariants (`t=None` and zero-init `td_proj` with `t≠None`), gradient flow + post-step modulation, OPM no-op invariant, encoder smoke test. All passing on CPU. |

### Why the no-op invariants matter

Both the `td_proj` weight inside each SE and both `td_head_v{1,2}` linear layers
are zero-initialized. That means the first training step is bit-for-bit
equivalent to the baseline `Encoder` — we never destabilize a working
configuration to bolt on the feedback pathway. The pathway is learned from a
known-good starting point.

This also matters as a regression guard: if a future refactor breaks the
no-op invariant tests, we'll know the wiring drifted from the baseline.

## What was deliberately left out

| Item | Reason | Pointer |
| --- | --- | --- |
| `vae.py` integration | One-line swap (`Encoder` → `TopDownEncoder` in `VAE.__init__`); deferred until we have a reason to train. | `vae.py` |
| Decoder-side routing | The decoder still consumes `mu` only. No top-down into decoder OPMs yet. | `decoder.py` |
| Iterative refinement (K > 1 BP iterations) | Current encoder is exactly K = 1. Architecture supports looping the two passes at no cost. | `topdown_encoder.py:forward_dict` |
| Plaid/transparency disambiguation stimulus | The Stage-1 failure test — see next section. | TBD notebook |
| Training run | Out of scope per agreed plan; needs the disambiguation test result first to know what we're measuring. | — |

## The Stage-1 failure-mode test (run this before Stage 2)

Stage 1 routes the gate multiplicatively. That handles "prior shapes the
readout of evidence" but leaves the *residual* untouched: two hypotheses that
predict the same Gabor-power channel activation cannot compete, because the
mechanism cannot subtract one prediction from the evidence to see what's left.

The cleanest falsification stimulus is **plaid vs. two transparent gratings**:

- Stimulus A — plaid: sum of two oriented gratings at θ₁, θ₂ (single surface,
  multiplicative interference).
- Stimulus B — transparency: same two gratings perceived as two surfaces
  (additive overlay, classic motion-transparency setup).

At the Gabor-power level these are essentially indistinguishable — the same
energy at the same orientations. Higher-order structure (junction statistics,
phase coherence across space, surface segmentation) is what separates them.

Expected outcomes:

- **Baseline `Encoder`**: cannot disambiguate. Both map to similar `mu`.
- **Stage 1 `TopDownEncoder`**: still cannot disambiguate. The gate can
  reweight V1 channels but the upward message is the same; the top-down hint
  has nothing to "cancel against."
- **Stage 2 (subtractive)**: should disambiguate, because subtracting the
  top-down prediction of one hypothesis exposes a residual that selects
  against it.

If Stage 1 *does* somehow disambiguate (e.g., by exploiting tiny systematic
differences in our synthetic stimulus), the test is broken and we need a
better one before claiming explaining-away is the missing piece. So: validate
the stimulus first (confirm Gabor-power signatures match to within ε), then
run all three configs.

## Stage 2 design (Rao-Ballard subtractive)

Stage 1 message at V1: `up = gate(SE_bias_by_t) ⊙ bottom_up`.

Stage 2 message at V1: `up = bottom_up − predict_v1_from_t(t)`, plus an
optional gain term. The predicted activations come from a small decoder head
(e.g., a 1×1 conv from a learned spatial broadcast of `t`, channels = V1
channel count). Implementation sketch:

```python
class SubtractivePCBlock(nn.Module):
    def __init__(self, channels, td_dim):
        super().__init__()
        self.predict = nn.Conv2d(td_dim, channels, 1)  # spatial broadcast outside
        self.gain    = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, bottom_up, t_spatial):
        # t_spatial: (B, td_dim, H, W) -- top-down hypothesis lifted to V1 grid.
        prediction = self.predict(t_spatial)
        residual = bottom_up - prediction
        return self.gain * residual
```

Two design knobs:

1. **Lifting `t` to V1 spatial grid.** Options: (a) tile/broadcast the latent
   vector (cheap, no spatial structure to the prediction — only channel-level);
   (b) tile a per-spatial-location latent map from a small upsampling decoder
   (matches the generative-model intuition better, costs more). Start with (a)
   — it's the minimal departure from Stage 1 and isolates the additive-vs-
   subtractive variable.
2. **Where to apply the subtraction.** Inside `OrientedPowerMap` after the
   Gabor conv + BN + ReLU + change_res, replacing or augmenting the SE step.
   Replacing is the clean experiment; augmenting (both gate and subtract)
   muddies the comparison. Replace.

Bisimulation framing:

- Stage 1 = **weak bisimulation** of BP — matches the *posterior mode* the
  generative model converges to, fails to match the *evidence flow*
  (explaining-away).
- Stage 2 = **strong bisimulation** — both posterior mode *and* per-channel
  residual messages match the underlying generative inference (up to the
  approximations baked into the architecture).

## Two-pass cost and mitigation

`TopDownEncoder.forward_dict` runs the upper stack (V3, residual blocks 1–3,
FC) twice — once to compute `mu_0`, once with V1/V2 biased. On a 512² CXR8
input at batch 4 this roughly doubles encoder FLOPs and activation memory.

Cheap mitigation if it becomes a bottleneck: cache the pass-1 activations of
everything *above* the routed layers (V3 + residual blocks 1–3) and only
re-run V1, V2 in pass 2. The pass-2 upper stack runs on the new V2 output, so
we can't fully skip it, but we can reuse the pass-1 `x_v4` *for the top-down
head* (use `x_v4_pass1` to predict `t`, not just `mu_0`). That trades off
faithfulness — pass 2 is no longer a clean BP iteration — for speed. Defer
until profiling says it matters.

## Open questions

1. **Should `td_dim` match a meaningful basis or stay a free bottleneck?** Right
   now `td_dim_v1 = td_dim_v2 = 64` is arbitrary. There's an argument for
   tying it to the number of pattern-theory *generators* at that level (i.e.,
   "what is V4 telling V1 about?" → it's saying "these N hypotheses, with
   these activation strengths"). Would need a Pattern Theory–side commitment
   to a generator count.
2. **Where does the top-down signal come from in the final model?** This PR
   uses `mu` (encoder output). Alternatives: decoder intermediate features
   (more faithful to generative-model feedback), explicit prior from a
   separate context net, or an iterative loop over `mu`. The choice matters
   for what the model can express; deferred until Stage 2 is in.
3. **Does the trainable-gabor-filters branch help here?** Branch CLAUDE.md
   notes that branch disabled SE entirely. With top-down-driven SE, the
   information content of the gate is no longer degenerate, so SE may be
   useful again *and* Gabors may be fine to leave trainable. Worth a small
   experiment after Stage 2.
4. **Loss term for the top-down head?** Currently no auxiliary loss; `td_proj`
   and `td_head_v{1,2}` train only through the standard VAE loss flowing back
   through pass 2. A predictive-coding-style loss
   `||top_down_prediction − bottom_up_evidence||²` would give the top-down
   path a direct training signal and likely accelerate learning. Natural fit
   for Stage 2 (the prediction is already an explicit tensor there).

## Running the tests

```bash
cd resnet18-vae
python -m unittest test_topdown -v
```

Six tests, ~1.5 s on CPU.
