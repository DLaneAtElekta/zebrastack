# Variational Bayesian Feature Visualization

This document describes the variational Bayesian approach to feature visualization implemented in this project.

## Overview

Traditional feature visualization optimizes a single image to maximize activations at a target layer/channel. The variational approach instead:

1. **Parameterizes the input as a distribution** q(x) rather than a point estimate
2. **Matches a target posterior distribution** over activations using KL divergence
3. **Regularizes with entropy** to encourage diverse, robust visualizations

## Mathematical Formulation

### Problem Statement

Given a neural network f and target layer/channel, find a distribution over inputs q(x) such that the distribution of activations q(a) = q(f(x)) matches a desired posterior p(a).

### Objective Function

```
min_θ  KL(q(a|θ) || p(a)) - λ H[q(a|θ)]

where:
- θ are the parameters of the input distribution
- KL is the Kullback-Leibler divergence
- H is the entropy
- λ is the entropy regularization weight
```

### Reparameterization Trick

The input distribution is parameterized as:

```
q(x) = N(μ, diag(σ²))

Sampling: x = μ + σ ⊙ ε, where ε ~ N(0, I)
```

This allows gradients to flow through the sampling process via the parameters (μ, σ).

### Target Posteriors

We implement three types of priors/posteriors:

#### 1. Laplace Prior (Sparsity)

Encourages sparse activation patterns:

```
p(a) = Laplace(μ, b)

KL ≈ -Σ log(p(a_i))
```

**Use case**: Find features that activate selectively (few strong activations)

#### 2. Sparse Gaussian Prior (High Activation + Sparsity)

Encourages high activations with L1 regularization:

```
p(a) = N(μ_high, σ²) with L1 penalty

Loss = -log p(a) + λ_sparse |a - μ_high|
```

**Use case**: Maximize activations while maintaining sparsity

#### 3. Kurtosis Matching

Matches the kurtosis (tail behavior) of the activation distribution:

```
Kurtosis = E[(a - μ)^4] / (E[(a - μ)^2])²

Loss = (Kurtosis(a) - target_kurtosis)²
```

**Use case**: Control whether activations are:
- Super-Gaussian (kurtosis > 3): Peaky with heavy tails
- Sub-Gaussian (kurtosis < 3): Flat, uniform-like
- Gaussian (kurtosis = 3): Normal distribution

### Entropy Regularization

Entropy is estimated using the Hessian of the loss:

```
H ≈ -0.5 log(det(H_θ))

where H_θ is the Hessian w.r.t. parameters θ
```

We implement two approximations:

1. **Diagonal Hessian**: Fast, assumes independence
2. **Hutchinson's Estimator**: Stochastic trace estimation

**Effect**: Higher entropy = broader basin = more robust/diverse features

## Implementation

### Core Components

#### 1. `VariationalImageParameter`

Parameterizes input as Gaussian distribution with reparameterization trick.

```python
from variational_visualization import VariationalImageParameter

param = VariationalImageParameter(image_size=224, channels=3)

# Sample images
images = param.sample(n_samples=16)

# Get mean (mode)
mean_image = param.get_mean()

# Get variance
variance = param.get_variance()
```

#### 2. `ActivationDistributionPrior`

Defines target distributions with different statistical properties.

```python
from variational_visualization import ActivationDistributionPrior

prior = ActivationDistributionPrior()

# Laplace (sparsity)
loss = prior.laplace_prior(activations, target_mean=0.0, target_scale=1.0)

# Sparse Gaussian (high activation)
loss = prior.sparse_gaussian_prior(activations, target_mean=5.0)

# Kurtosis matching
loss = prior.kurtosis_matching_loss(activations, target_kurtosis=5.0)
```

#### 3. `HessianEntropyEstimator`

Estimates entropy using second-order information.

```python
from variational_visualization import HessianEntropyEstimator

estimator = HessianEntropyEstimator()

# Diagonal approximation (fast)
entropy = estimator.diagonal_hessian_entropy(loss, parameters)

# Hutchinson estimator (more accurate)
entropy = estimator.hutchinson_trace_estimator(loss, parameters, n_samples=10)
```

#### 4. `VariationalFeatureVisualizer`

Main class that orchestrates variational optimization.

```python
from variational_visualization import VariationalFeatureVisualizer
from lucent.modelzoo import inceptionv1

model = inceptionv1(pretrained=True)

visualizer = VariationalFeatureVisualizer(
    model=model,
    layer_name='mixed4a_1x1_pre_relu_conv',
    channel_idx=42,
    prior_type='sparse_gaussian',
    entropy_weight=0.01
)

visualizer.register_hook()

# Optimization loop
optimizer = torch.optim.Adam(visualizer.image_param.parameters(), lr=0.05)

for iteration in range(100):
    images, activations = visualizer.forward(n_samples=16)
    losses = visualizer.compute_loss(activations)

    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

# Get result
final_image = visualizer.get_visualization()
```

## Usage Examples

### Example 1: Basic Visualization with Laplace Prior

```python
from lucent.modelzoo import inceptionv1
from variational_visualization import VariationalFeatureVisualizer

model = inceptionv1(pretrained=True)

viz = VariationalFeatureVisualizer(
    model=model,
    layer_name='mixed4a_3x3_bottleneck_pre_relu_conv',
    channel_idx=10,
    prior_type='laplace',
    entropy_weight=0.0
)

viz.register_hook()
optimizer = torch.optim.Adam(viz.image_param.parameters(), lr=0.1)

for i in range(100):
    images, acts = viz.forward(n_samples=16)
    losses = viz.compute_loss(acts, compute_entropy=False)

    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

result = viz.get_visualization()
```

### Example 2: Comparing Multiple Priors

```python
priors = ['laplace', 'sparse_gaussian', 'kurtosis']
results = {}

for prior_type in priors:
    viz = VariationalFeatureVisualizer(
        model=model,
        layer_name='mixed4a_1x1_pre_relu_conv',
        channel_idx=42,
        prior_type=prior_type
    )

    # ... optimize ...

    results[prior_type] = viz.get_visualization()
```

### Example 3: Entropy Regularization

```python
# Without entropy regularization
viz_no_entropy = VariationalFeatureVisualizer(
    model=model,
    layer_name='mixed4a_1x1_pre_relu_conv',
    channel_idx=42,
    entropy_weight=0.0
)

# With entropy regularization
viz_with_entropy = VariationalFeatureVisualizer(
    model=model,
    layer_name='mixed4a_1x1_pre_relu_conv',
    channel_idx=42,
    entropy_weight=0.05  # Encourage diversity
)
```

## Running the Demo

A comprehensive demo is provided in `demo_variational_visualization.py`:

```bash
python demo_variational_visualization.py
```

This will:
1. Compare different prior types on the same layer/channel
2. Show the effect of entropy regularization
3. Save visualizations, variance maps, and loss curves to `screen_captures/variational/`

## Unit Tests

Comprehensive unit tests are provided in `test_variational_visualization.py`:

```bash
pytest test_variational_visualization.py -v
```

Tests cover:
- ✅ Reparameterization trick correctness
- ✅ Gradient flow through sampling
- ✅ Prior distributions (Laplace, sparse Gaussian, kurtosis)
- ✅ Hessian entropy estimation
- ✅ End-to-end optimization

All 29 tests should pass.

## Comparison with Traditional Approach

| Aspect | Traditional | Variational |
|--------|------------|-------------|
| **Optimization target** | Single image | Distribution over images |
| **Objective** | Maximize activation | Match posterior distribution |
| **Regularization** | Hand-crafted (blur, jitter) | Entropy via Hessian |
| **Output** | One image | Mean + variance |
| **Uncertainty** | None | Variance map shows uncertainty |
| **Robustness** | Can overfit | Entropy encourages broader basins |

## Advantages

1. **Principled framework**: Grounded in Bayesian inference
2. **Uncertainty quantification**: Variance maps show where model is uncertain
3. **Flexible priors**: Control sparsity, kurtosis, or other statistics
4. **Entropy regularization**: Automatic regularization via second-order information
5. **Robustness**: Less prone to adversarial-like artifacts

## Limitations

1. **Computational cost**: Requires multiple forward passes (Monte Carlo sampling)
2. **Hessian computation**: Expensive for large images (use diagonal approximation)
3. **Convergence**: May be slower than traditional approach
4. **Hyperparameters**: More parameters to tune (entropy weight, n_samples, etc.)

## Future Extensions

1. **Better Hessian approximations**: Use KFAC or natural gradient
2. **Hierarchical priors**: Multi-scale priors for better structure
3. **Amortized inference**: Train encoder to predict q(x) from target
4. **Multi-channel objectives**: Joint distribution over multiple channels
5. **Temporal coherence**: For video or sequence models

## References

- **Reparameterization trick**: Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- **Hessian entropy**: MacKay (1992) - Bayesian Methods for Neural Networks
- **Laplace approximation**: Murphy (2012) - Machine Learning: A Probabilistic Perspective
- **Feature visualization**: Olah et al. (2017) - Feature Visualization

## Citation

If you use this code, please cite:

```bibtex
@software{variational_feature_viz,
  title={Variational Bayesian Feature Visualization},
  author={activation-atlas contributors},
  year={2025},
  url={https://github.com/yourusername/activation-atlas}
}
```

## License

Same as the parent activation-atlas project.
