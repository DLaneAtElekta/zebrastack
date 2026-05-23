# Bayesian routing: closed-form Gaussian product at V1 / V2

This document is the running plan for the Bayesian routing experiment — a
*parallel* alternative to Stage 2 (Rao-Ballard subtractive predictive coding).
Both replace the multiplicative SE gate from Stage 1; they differ in what they
replace it with.

| | Mechanism | Bisimulation regime |
| --- | --- | --- |
| Stage 1 (shipped) | `up = gate(SE_bias_by_t) ⊙ bottom_up` | Multiplicative reweighting; matches posterior mode in the limit, no residual messages. |
| Stage 2 (planned) | `up = bottom_up − predict(t)` (+ gain) | Subtractive PC; matches per-channel residual messages explicitly. |
| **This PR** | `up = (Λ ⊙ μ(t) + Λ_d ⊙ bottom_up) / (Λ + Λ_d)` | Closed-form Gaussian product; matches the posterior mode of the implied linear-Gaussian generative model exactly. |

The three options are not mutually exclusive design endpoints; they are
different points on the "how explicit is your generative-model commitment"
axis. This PR sits at the explicit end.

## What this PR ships

Three new modules and a test file under `resnet18-vae/`:

| File | Purpose |
| --- | --- |
| `bayesian_routing.py` | `BayesianRoutingLayer` — per-channel Gaussian product. Diagonal Λ. Prior mean and log-precision are linear projections of `t`; data precision is a learned per-channel parameter. At init the prior precision is ~0, so the layer is numerically identity on the evidence. |
| `bayes_oriented_powermap.py` | `BayesianRoutedOrientedPowerMap` — `OrientedPowerMap` subclass that *replaces* the inner SE block with `BayesianRoutingLayer`. Exposes `forward(x, t=None)`. With `t=None` the routing layer is the identity, so the parent's `Sequential` yields "OPM with SE swapped for Identity". |
| `bayes_encoder.py` | `BayesianRoutedEncoder` — two-pass encoder paralleling `TopDownEncoder`. Pass 1 = bottom-up to coarse `mu_0`. Pass 2 = project `mu_0` into `t_v1` / `t_v2` via zero-init heads, route V1/V2 through the Bayesian layer, rerun the upper stack. |
| `test_bayesian_routing.py` | Seven unit tests: identity invariants for `t=None` and at-init `t≠None`, posterior-between-prior-and-evidence sanity check, gradient flow into `to_mu` / `to_log_diag` / `log_lambda_data`, OPM no-op equivalence, encoder smoke. All passing on CPU. |

### Why the identity invariants matter

Three pathways are initialized to be (numerically) identity:

  1. `to_mu` zero-init → prior mean is 0.
  2. `to_log_diag` zero-weight + large-negative-bias → prior precision ~ exp(-10) ≈ 4.5e-5.
  3. `log_lambda_data` zero-init → data precision is 1.

So the posterior at step 0 is `(eps · 0 + 1 · e) / (eps + 1) ≈ e`. The
top-down pathway starts from a known-good baseline (evidence passes through)
and is learned from there. This is the same discipline used in Stage 1 —
never destabilize a working configuration to bolt on the feedback pathway.

There is one deliberate departure from Stage 1's invariant: this PR's
baseline is not "Encoder with SE", it's "Encoder with SE replaced by
Identity" in the routed stages. The Bayesian layer is a *replacement*, not an
augmentation. Stage 2 makes the same choice (see STAGE1_FOLLOWUP §"Stage 2
design"). Comparing this PR's training results against Stage 1 will therefore
involve a confound — the SE-vs-no-SE difference — that needs to be controlled
for (see "Comparison protocol" below).

## What was deliberately left out

| Item | Reason | Pointer |
| --- | --- | --- |
| Low-rank Λ (the `U Uᵀ` term in the design sketch) | Diagonal is the minimal first prototype. SMW solve is unnecessary until we have evidence diagonal is too restrictive. | `bayesian_routing.py` |
| Multi-level coordinate ascent | One BP iteration only (the two-pass encoder), like `TopDownEncoder`. The level structure (V1↔V2 messages, sweep-up / sweep-down) is future work. | `bayes_encoder.py:forward_dict` |
| Spatial structure in the prior | Prior μ and Λ are spatially constant per channel per batch element. The evidence varies spatially. A spatially-resolved prior (broadcast `t` through a small upsampler) is the natural next step if the constant-prior version is too coarse. | `bayesian_routing.py:forward` |
| Decoder-side Bayesian routing | Decoder still consumes `mu` only. | `decoder.py` |
| `vae.py` integration | One-line swap (`Encoder` → `BayesianRoutedEncoder`); deferred until there's a reason to train. | `vae.py` |
| Auxiliary loss on prior / data precision | Currently both train only through the standard VAE loss flowing back through pass 2. A free-energy-style loss is a natural fit; deferred. | — |
| Direct head-to-head training run against Stage 1 | Out of scope; needs the comparison protocol below to be agreed first. | — |

## Comparison protocol (before claiming this is better than Stage 1 or 2)

The three routing styles need to be compared on:

  1. **The plaid-vs-transparency stimulus** from STAGE1_FOLLOWUP §"Stage-1
     failure-mode test". Stage 1 is predicted to fail this. Stage 2 (subtractive)
     is predicted to pass it via explaining-away. This PR's prediction is
     subtler: the Gaussian product *can* push the posterior off the evidence
     mean when the prior disagrees, so on a stimulus where the top-down pathway
     has learned a "this is a plaid" vs "this is two surfaces" hypothesis, the
     posterior at V1 should shift accordingly. Whether that constitutes
     "explaining away" depends on what the prior precision learns. If Λ stays
     small the posterior tracks evidence and Stage 1's failure mode reappears.

  2. **An SE-vs-Identity baseline control.** This PR's routed stages lose the
     baseline SE gating; Stage 1's keep it. To attribute any performance
     difference to Bayesian-vs-multiplicative-routing rather than to losing
     SE, run a fourth config: `Encoder` with `se = nn.Identity()` in V1 and
     V2, no top-down. That isolates the SE contribution and lets the routing
     comparison be apples-to-apples.

  3. **Held-out reconstruction quality** at fixed compute budget. Pass 2 of
     the two-pass encoder roughly doubles encoder FLOPs in all three routing
     modes; the comparison should be at matched wall-clock, not matched
     epochs.

Run order: SE-vs-Identity control first (cheap, settles whether SE matters at
all). If SE matters, all three routing modes need it accounted for. If not,
proceed to the head-to-head.

## Open questions

1. **Does diagonal Λ suffice?** Channel-correlated priors are the obvious
   missing piece. The low-rank `U Uᵀ` term is the cheap upgrade (O(C R²) via
   SMW). Worth measuring how often the posterior-vs-prior alignment is
   bottlenecked by the diagonal assumption before adding the rank term.

2. **What should `log_lambda_data` represent semantically?** Right now it's a
   free per-channel parameter that learns "how much should I trust this
   channel's bottom-up evidence". One could interpret it as the inverse noise
   variance of the Gabor-power readout. Two design choices: (a) keep it
   free, let SGD pick; (b) tie it to a measured noise statistic (e.g., per-
   channel variance over a corpus of natural images). (a) for now.

3. **Spatial prior, take two.** If the spatially-constant prior is too coarse
   to discriminate (e.g., on the plaid stimulus, where the disambiguating
   info is in junction statistics that vary across the image), the next
   smallest step is to predict a (C, H, W) prior-mean field instead of a (C,)
   prior-mean vector. That's a 1×1 conv from a tiled `t`, which is
   structurally identical to the Stage 2 prediction head. At that point
   "Bayesian product" and "Stage 2 subtractive" differ only in the form of
   the update (closed-form Gaussian vs Λ_d-weighted residual). Worth keeping
   them as separate code paths until the comparison is run.

4. **Two-pass cost.** Same as Stage 1: pass 2 reruns V3 + the residual stack.
   Same mitigation applies (cache pass-1 upper-stack activations, only rerun
   V1 / V2). Defer until profiling says it matters.

5. **Coordinate ascent across levels.** The design sketch in the source
   message proposes per-level Gaussian products with iteration to
   convergence. This PR is one Gaussian product per routed level with K = 1
   BP iterations. The encoder architecture supports looping the two passes;
   adding sweep-down (V2 → V1 prior updates) is the more interesting move,
   and is what makes this "honest" Bayes rather than "feedforward
   approximation to Bayes". Deferred until the K = 1 version trains.

## On the amortization half of the source message

The source message proposed two cross-pollinated ideas:

  * **Vision-as-Brimstone:** the Bayesian routing layer above. *In scope here.*
  * **Brimstone-as-vision:** an amortized network that predicts CG-converged
    fluence adjustments given a course state. *Targets a codebase that is not
    in this repo* (radiation-therapy planning system). A toy version inside
    `resnet18-vae/` would be sketch-only with no real plan structure to
    consume, so it's deliberately not included. If a useful toy lives
    anywhere it probably lives next to the radiotherapy code.

## Running the tests

```bash
cd resnet18-vae
python -m unittest test_bayesian_routing -v
```

Seven tests, ~1.3 s on CPU. Existing Stage-1 tests (`test_topdown`) are
unaffected and still pass.
