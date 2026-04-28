# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ResNet18-VAE is a PyTorch implementation of a Variational Autoencoder (VAE) that combines ResNet-inspired architecture with a biologically-inspired visual processing model called the "Herring Stack". Despite the name, the implementation uses ResNet-50 as a base (not ResNet-18).

**Key Concept**: The model implements a hierarchical visual processing system inspired by the visual cortex, with oriented Gabor filter banks at the input (V1-like) followed by residual blocks, culminating in a VAE architecture for image reconstruction.

## Git Branches

### master (7e77167)
Main development branch. Latest commit: "consolidate clahe -- move to separate transform"
- Stable baseline implementation with frozen Gabor filters
- CLAHE preprocessing moved to separate transform
- STN code present but can be toggled

### trainable-gabor-filters (903ba59)
**Experimental branch** exploring learnable Gabor filters with enhanced regularization.

**Key changes from master:**
- **Trainable Gabor weights**: `requires_grad=True` for Gabor conv filters (previously frozen)
- **Enhanced regularization**: Added Dropout2d (p=0.1), extra BatchNorm, disabled SqueezeExcitation
- **Expanded filter bank**: Doubled Gabor filters with 1.5x frequency variants for more diversity
- **Architecture simplification**: STN fully disabled, decoder sparsity masking removed
- **Training adjustments**:
  - Learning rate: 1e-3 → 1e-4
  - KL beta: 2.0 → 0.2 (emphasis on reconstruction)
  - Batch size: 20 → 4
  - Latent dim: 16 → 1024 (4×4 → 32×32 spatial)
  - Basis reset: 8x weight increase, MSE → L1 loss
  - Dream training delayed to epoch 99
  - Added horizontal flip augmentation

**Purpose**: Test whether allowing Gabor filters to learn task-specific refinements improves reconstruction quality while maintaining biologically-inspired structure through basis reset regularization.

### warp-lut-investigation (0c1496a)
Investigation branch for warp look-up table functionality. Commit: "warp lut investigation"

## Running the Code

### Training
```bash
# Train the VAE model
python vae.py --train

# Train with specific dataset
python cxr8_dataset.py --train "b10s/b60s/b10/b60"
```

Training checkpoints are saved to `runs/` directory as `YYYYMMDD_epoch_NN.zip` files. The model automatically loads the most recent checkpoint if available.

### Inference
```bash
# Run inference on images
python vae.py --infer <directory_path>
python cxr8_dataset.py --infer <directory_path>
```

### Testing
```bash
# Run filter utility tests
python test_filter_utils.py

# Run module tests with unittest
python -m unittest oriented_powermap.py
```

### Environment Setup
The code expects the `DATA_TEMP` environment variable to point to the data directory:
- CXR8 dataset should be in `$DATA_TEMP/cxr8/`
- Dataset metadata: `Data_Entry_2017_v2020.csv`
- Images: `images/` subdirectory

## High-Level Architecture

### Core Model Components (vae.py)

**VAE Class**: Main model combining encoder and decoder
- Input: Multi-channel chest X-ray images (512x512 by default)
- Latent dimension: 32x32 (1024 dimensions)
- Loss: Combination of reconstruction loss (BCE + MSE) and KL divergence
- Includes experimental Spatial Transformer Network (STN) code (currently disabled)

**Key Functions**:
- `vae_loss()`: Computes total loss including reconstruction and KL divergence, with perceptual losses at multiple scales (x_v1, x_v2, x_v4)
- `reparameterize()`: Implements the reparameterization trick for backpropagation through stochastic sampling
- `train_vae()`: Main training loop with data augmentation
- `train_dream()`: Experimental latent space regularization by training on random latent vectors

### Encoder (encoder.py)

**Multi-scale Visual Processing**:
1. **V1 Layers** (oriented_powermap): Four cascaded OrientedPowerMap modules that apply Gabor filters
   - `oriented_powermap` → `oriented_powermap_2` → `oriented_powermap_3` → `oriented_powermap_4`
   - Each layer reduces spatial resolution by 2x

2. **Residual Processing**: Three residual block stages with increasing channels (64 → 128 → 256)
   - Each stage: conv1x1 → multiple OrientedPowerMap residual passes → downsample
   - Penpenultimate: 3 residual passes, 64 channels
   - Penultimate: 6 residual passes, 128 channels
   - Final: 3 residual passes, 256 channels

3. **Latent Representation**: Flattened features → FC layers → (mu, log_var)

**Important**: The encoder returns intermediate features (x_v1, x_v2, x_v4) for perceptual loss computation.

### Decoder (decoder.py)

**Hierarchical Upsampling**:
1. **FC to Conv**: Latent vector → FC layer → reshape to spatial feature map
2. **Three Upsampling Stages**: Reverse of encoder with bilinear upsampling
   - First: 256 channels, 3 residual passes
   - Second: 128 channels, 6 residual passes
   - Third: 64 channels, 6 residual passes
3. **V1-like Reconstruction**: Four cascaded OrientedPowerMap transpose modules
   - `conv_transpose_1` through `conv_transpose_4`
   - Each upsamples by 2x using OrientedPowerMap with `out_res="*2"`

**Mirror Architecture**: Decoder roughly mirrors encoder structure for symmetric reconstruction.

### OrientedPowerMap (oriented_powermap.py)

**Gabor Filter Bank Module**: Core building block implementing V1-like oriented receptive fields
- Creates Gabor filters at multiple orientations (default 7 directions) and frequencies
- Implements both power map (magnitude) and phase-preserving modes
- Architecture: conv1x1_pre → gabor_conv → BatchNorm → ReLU → resolution_change → conv1x1_post
- Includes residual shortcut connection
- Supports resolution changes: `None`, `"/2"` (AvgPool), `"^2"` (MaxPool), `"*2"` (Upsample)
- `basis_reset()`: Loss term to encourage filters to remain close to original Gabor basis

### Dataset (cxr8_dataset.py)

**Cxr8Dataset**: CXR8 chest X-ray dataset loader with advanced preprocessing
- Multi-scale CLAHE (Contrast Limited Adaptive Histogram Equalization) with tile sizes: 16, 8, 4
- Rotation-invariant CLAHE: applies CLAHE at multiple angles [0°, 30°, 60°] then averages
- Input channels: [clahe_16, clahe_8, clahe_4, original]
- Normalization: Z-score normalization → sigmoid for bounded values
- Returns: image tensor, masks, finding labels, filenames

### Filter Utilities (filter_utils.py)

**Gabor Filter Generation**:
- `make_oriented_map()`: Creates oriented Gabor filter bank (real/imaginary pairs for power maps)
- `make_oriented_map_stack_phases()`: Creates filters preserving phase information
- `gabor()`: Generates individual Gabor kernels with specified frequency, orientation, sigma
- Frequencies: [1.0, 0.5, 0.25] with golden ratio spacing historically considered

## Important Implementation Details

### Training Strategy
- **Two-pass reconstruction**: Forward pass through VAE twice, accumulating outputs (`result_dict_2["x_back"] = result_dict_2["x_back"] + result_dict["x_back"]`)
- **Mixed loss weights**: Configurable BCE vs MSE (`l1_weight` parameter, typically 0.3-0.8)
- **KL beta**: 0.2 (lower than standard 1.0 for β-VAE)
- **Basis reset regularization**: Weighted 8.0x to keep OrientedPowerMap filters near Gabor initialization
- **Random batch skipping**: 50% of batches skipped randomly during training
- **Perceptual losses**: Cross-entropy at V1/V4 layers (when enabled)

### Device Management
- All OrientedPowerMap modules explicitly moved to device with `.to(device)`
- Model uses CUDA if available
- Careful device management required due to complex multi-module architecture

### Model Persistence
- Checkpoints include: epoch number, model state dict, optimizer state dict, loss
- Naming convention: `YYYYMMDD_epoch_NN.zip`
- Automatically resumes from latest checkpoint in `runs/` directory

### Known Experimental Features (Currently Disabled)
- **STN (Spatial Transformer Network)**: Code present but disabled (lines 208-348 in vae.py)
- **V4 perceptual loss**: Toggleable with `use_v4` flag
- **Sparsity masks**: Commented out threshold-based masking in decoder
- **Dream training**: Latent space regularization (activated after epoch 99)

## Code Organization Patterns

- Each major component (encoder, decoder, oriented_powermap) can run standalone with `if __name__ == "__main__"` tests
- Forward methods often have both `forward()` and `forward_dict()` variants - use `forward_dict()` when intermediate outputs needed
- Extensive use of `torchinfo.summary()` for architecture inspection
- Logging to file: `runs/YYYYMMDD_vae_main.log`

## Common Gotchas

1. **Input size must be divisible by 2^N**: Due to cascaded downsampling/upsampling (typically 2^4 = 16)
2. **Memory management**: Explicit `torch.cuda.empty_cache()` calls throughout training loop
3. **Gradient clipping**: Commented out but available (`torch.nn.utils.clip_grad.clip_grad_norm_`)
4. **Batch size**: Default is 4 (small due to large image size and model complexity)
5. **CLAHE caching**: Dataset has caching capability but currently disabled

## Refactoring Notes

See `TODO_REFACTOR.md` for planned architectural improvements:
- Each layer should have forward and reverse (encoder/decoder pairs)
- OrientedPowerMap modules should include bottleneck structure
- Loss functions should be configurable per layer (L1, BCE, cross-entropy)
