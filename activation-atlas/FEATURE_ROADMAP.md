# Feature Roadmap - Activation Atlas

This document outlines proposed features and enhancements for the activation-atlas neural network visualization system.

## Priority 1: Core Analysis & Quality Improvements

### 1.1 Depth Estimation Integration ⭐ (Your Request)

**Module**: `depth_analysis.py`

Integrate monocular depth estimation to analyze the 3D structure and depth perception of generated visualizations.

**Implementation Details**:
- **Model**: Depth Anything V2 (NeurIPS 2024)
  - Multiple model sizes: Small (25M), Base (97M), Large (335M)
  - Fast inference (10x faster than SD-based models)
  - High accuracy on synthetic and real images
- **Integration Points**:
  - Add depth estimation as post-processing step after visualization generation
  - Store depth maps alongside visualizations in CSV database
  - Compute depth statistics: mean depth, depth variance, depth range
  - Visualize depth maps as heatmaps and 3D surfaces
- **New Features**:
  - `estimate_depth(image_path)` - Generate depth map for visualization
  - `compute_depth_metrics(depth_map)` - Extract depth statistics
  - `visualize_depth_3d(image, depth_map)` - Create 3D visualization
  - `compare_depth_across_layers()` - Analyze how depth perception varies by layer
  - `depth_based_scoring()` - Score visualizations based on depth complexity
- **Use Cases**:
  - Understand which layers/channels encode depth information
  - Identify 3D vs 2D texture preference in different layers
  - Evolutionary optimization toward depth-rich visualizations
  - Comparative analysis: early layers vs deep layers depth perception
- **Dependencies**:
  - `depth-anything-v2` or via HuggingFace Transformers
  - `matplotlib` for depth visualization
  - `open3d` (optional) for advanced 3D rendering

**New Database Columns**:
- `depth_mean` - Average depth value
- `depth_std` - Depth variation/complexity
- `depth_range` - Min to max depth span
- `depth_map_path` - Path to saved depth visualization
- `depth_histogram` - JSON-encoded depth distribution

---

### 1.2 Complete InceptionV1 Channel Atlas Generation ⭐⭐⭐ (Your Request #3)

**Module**: `atlas_generator.py`, `batch_scheduler.py`, `atlas_viewer.py`

Generate a comprehensive atlas of all channel visualizations for InceptionV1 layers mixed3a-5b.

**Scope**:
- **Target Layers**: mixed3a, mixed3b, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b
- **Visualization Type**: Channel objectives (all channels in each layer)
- **Estimated Scale**:
  - **mixed3a**: ~256 channels
  - **mixed3b**: ~480 channels
  - **mixed4a**: ~512 channels
  - **mixed4b**: ~512 channels
  - **mixed4c**: ~512 channels
  - **mixed4d**: ~528 channels
  - **mixed4e**: ~832 channels
  - **mixed5a**: ~832 channels
  - **mixed5b**: ~1024 channels
  - **Total**: ~5,500 visualizations

**Implementation Details**:

**Batch Generation System**:
- **Parallel generation**: Distribute across multiple GPUs
- **Job scheduling**: Queue-based system with priority levels
- **Progress tracking**: Real-time progress dashboard
- **Fault tolerance**: Automatic retry on failures, checkpoint recovery
- **Resource management**: Smart GPU allocation and memory management

**Generation Configuration**:
```python
atlas_config = {
    'layers': ['mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c',
               'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b'],
    'objective_type': 'channel',  # All channels per layer
    'image_size': 384,  # High quality images
    'optimization_steps': 512,
    'enable_wrapping': True,
    'output_dir': 'atlas/inception_v1_complete',
    'parallel_workers': 8,  # Number of GPUs
}
```

**Automated Pipeline**:
1. **Discovery Phase**: Query model for channel counts per layer
2. **Planning Phase**: Generate job list (~5,500 tasks)
3. **Execution Phase**: Distributed generation with load balancing
4. **Validation Phase**: Verify all images generated successfully
5. **Organization Phase**: Create directory structure and metadata
6. **Visualization Phase**: Generate atlas HTML viewer

**Directory Structure**:
```
atlas/inception_v1_complete/
├── mixed3a/
│   ├── channel_000.png
│   ├── channel_001.png
│   ├── ...
│   ├── channel_255.png
│   └── metadata.csv
├── mixed3b/
│   ├── channel_000.png
│   ├── ...
│   └── metadata.csv
├── ...
├── mixed5b/
│   ├── channel_000.png
│   ├── ...
│   ├── channel_1023.png
│   └── metadata.csv
├── atlas_index.html        # Interactive viewer
├── atlas_metadata.csv      # Master database
└── generation_log.txt      # Complete generation log
```

**New Features**:
- `AtlasGenerator` class - Main orchestrator for atlas generation
- `generate_complete_atlas()` - Generate all visualizations for specified layers
- `generate_layer_atlas()` - Generate all channels for a single layer
- `AtlasScheduler` - Job scheduling and GPU allocation
- `AtlasValidator` - Verify completeness and quality
- `create_atlas_viewer()` - Generate interactive HTML viewer
- `export_atlas_grid()` - Create grid images (e.g., 16×16 channel grids)
- `compare_layers_atlas()` - Compare channel distributions across layers

**Interactive Atlas Viewer**:
- **Grid view**: Thumbnail grid of all channels per layer
- **Detail view**: Click to see full-resolution image
- **Filtering**: Filter by layer, channel index, quality metrics
- **Search**: Semantic search using CLIP embeddings (if available)
- **Statistics**: Layer-wise statistics and distributions
- **Navigation**: Easy navigation between layers and channels
- **Export**: Download subsets or create custom collections
- **Comparison**: Side-by-side comparison of channels
- **Depth integration**: View depth maps alongside visualizations (if Depth feature enabled)

**Optimization Strategies**:
- **Smart batching**: Group similar channels for better GPU utilization
- **Caching**: Cache model activations and reuse across channels
- **Mixed precision**: FP16 for faster generation
- **Checkpoint recovery**: Resume from interruptions
- **Progressive quality**: Generate low-res previews first, refine later
- **Adaptive scheduling**: Prioritize failing jobs, skip completed ones

**Quality Control**:
- **Automatic validation**: Check for blank/corrupted images
- **Quality metrics**: Compute diversity, sharpness, colorfulness
- **Outlier detection**: Flag unusual or failed visualizations
- **Retry mechanism**: Regenerate failed images with adjusted parameters
- **Human review**: Flag subset for manual quality check

**Time Estimates**:
- **Single GPU (V100)**: ~3-4 days for complete atlas
- **4 GPUs**: ~20-24 hours
- **8 GPUs**: ~12-15 hours
- **16 GPUs**: ~6-8 hours
(Assumes 30-60 seconds per visualization)

**Integration with Visualization System**:
- Uses existing `EnhancedVisualizationGenerator`
- All visualizations logged to master CSV database
- Compatible with mutation and evolution systems
- Depth analysis integration (compute depth for all atlas images)
- Training checkpoint integration (generate atlas at different training stages)

**Use Cases**:
- **Model understanding**: Complete overview of what each layer detects
- **Layer comparison**: See how feature complexity evolves through network
- **Channel analysis**: Identify redundant or unique channels
- **Publication**: High-quality figures for papers and presentations
- **Interactive exploration**: Share with researchers via web viewer
- **Baseline dataset**: Reference set for future experiments
- **Transfer learning**: Identify which channels to fine-tune

**Advanced Features**:
- **Multi-model atlas**: Generate atlases for multiple checkpoints
- **Differential atlas**: Show how channels change during training
- **Semantic clustering**: Automatically group similar channels
- **Channel importance**: Rank channels by activation magnitude on ImageNet
- **Ablation preview**: Predict impact of removing channels

**Command Line Interface**:
```bash
# Generate complete atlas
python atlas_generator.py \
    --layers mixed3a mixed3b mixed4a mixed4b mixed4c mixed4d mixed4e mixed5a mixed5b \
    --output-dir atlas/complete \
    --gpus 8 \
    --image-size 384

# Generate single layer
python atlas_generator.py \
    --layer mixed4c \
    --output-dir atlas/mixed4c_only \
    --gpus 4

# Resume interrupted generation
python atlas_generator.py \
    --resume atlas/complete \
    --gpus 8

# Create interactive viewer
python atlas_viewer.py \
    --atlas-dir atlas/complete \
    --port 8080
```

**Dependencies**:
- Existing visualization system (no new dependencies)
- Optional: `jinja2` for HTML viewer generation
- Optional: `tqdm` for progress bars

**Deliverables**:
1. ~5,500 high-quality channel visualizations
2. Master CSV database with all metadata
3. Interactive HTML viewer for exploration
4. Layer-wise grid visualizations
5. Generation report with statistics
6. Reusable pipeline for future models

---

### 1.3 Enhanced Continuous/Wrapping Visualizations ⭐⭐⭐⭐ (Your Request #4)

**Module**: `continuous_visualizations.py`, enhancements to `wrapping_transforms.py`

**Status**: Basic wrapping transforms already exist! This feature enhances them with production-ready features.

Generate seamlessly tileable/continuous visualizations using advanced wrapping techniques.

**Current Implementation** (Already in codebase):
- ✅ `horizontal_wrap_transform()` - Edge blending for seamless horizontal wrapping
- ✅ `horizontal_tile_transform()` - Horizontal tiling during optimization
- ✅ `create_tiled_parameter()` - Tiled initialization
- ✅ `wrap_transform()` and `tile_transform()` - Ready-to-use transform functions
- ✅ Demo script: `demo_wrapping.py`

**Enhancement Plan**:

**1. Multi-Directional Wrapping**:
```python
# Current: Horizontal only
# New: All directions
- horizontal_wrap_transform()  # ✅ Already exists
- vertical_wrap_transform()     # NEW
- bidirectional_wrap_transform() # NEW - both H & V
- radial_wrap_transform()       # NEW - circular wrapping
- spherical_wrap_transform()    # NEW - 360° spherical mapping
```

**2. Advanced Tiling Patterns**:
- **Grid tiling**: NxM tile grids (not just horizontal)
- **Hexagonal tiling**: Honeycomb patterns
- **Voronoi tiling**: Organic cell-like patterns
- **Escher-style tiling**: Interlocking patterns
- **Kaleidoscope effects**: Symmetric pattern generation

**3. Seamless Boundary Optimization**:
```python
# Current: Edge blending post-optimization
# New: Enforce continuity during optimization
class ContinuousParameterization:
    - Toroidal topology (wrap-around in both directions)
    - Cylindrical topology (wrap horizontally only)
    - Spherical topology (both ends meet at poles)
    - FFT-based parametrization (frequency domain wrapping)
```

**4. Quality Metrics for Wrapping**:
- **Seam visibility score**: Measure how visible the seam is
- **Continuity metrics**: Derivative discontinuity at boundaries
- **Tileability score**: How well the image tiles
- **Symmetry detection**: Detect emergent symmetries
- **Fourier analysis**: Frequency-domain seam detection

**5. Interactive Wrapping Tools**:
- **Wrap strength slider**: Adjust wrap_factor interactively
- **Live preview**: Real-time tiling preview (show 3×3 grid)
- **Seam editor**: Manually adjust problematic seams
- **Export formats**: Tileable textures, seamless backgrounds, patterns

**6. Integration with Atlas Generation**:
```python
# Generate wrapping versions of all atlas visualizations
atlas_config = {
    'enable_wrapping': True,
    'wrapping_types': ['horizontal', 'vertical', 'bidirectional'],
    'generate_tiled_previews': True,  # Show 3×3 tile preview
    'export_formats': ['png', 'svg', 'webp'],
}
```

**7. Use Case-Specific Presets**:
```python
# Texture generation
texture_preset = {
    'wrapping_type': 'bidirectional',
    'wrap_factor': 0.8,
    'enforce_continuity': True,
    'optimization_steps': 1024,  # More steps for better seams
}

# Wallpaper/background generation
wallpaper_preset = {
    'wrapping_type': 'horizontal',
    'tile_count': 3,
    'aspect_ratio': '16:9',
    'resolution': (3840, 2160),  # 4K
}

# Kaleidoscope art
kaleidoscope_preset = {
    'wrapping_type': 'radial',
    'symmetry_order': 8,  # 8-fold symmetry
    'center_weight': 2.0,
}

# VR/360° content
vr_preset = {
    'wrapping_type': 'spherical',
    'equirectangular': True,
    'resolution': (4096, 2048),
}
```

**8. Batch Generation Tools**:
```bash
# Generate wrapping versions of existing visualizations
python continuous_generator.py \
    --input-dir atlas/complete \
    --output-dir atlas/continuous \
    --wrapping-types horizontal vertical bidirectional \
    --quality-threshold 0.8

# Generate texture pack
python continuous_generator.py \
    --preset texture \
    --count 100 \
    --resolution 1024 \
    --export-previews
```

**9. Export Formats**:
- **PNG**: Standard images with transparent option
- **Tileable SVG**: Vector format for infinite scaling
- **3×3 Preview**: Show tiling in context
- **Animated GIF**: Show seamless scrolling
- **WebGL texture**: Ready for game engines
- **Material files**: PBR textures for 3D software

**10. Quality Assurance**:
- **Seam detection**: Automatically detect visible seams
- **Continuity validation**: Check gradient discontinuities
- **Visual inspection**: Generate comparison views
- **A/B testing**: Compare wrapped vs non-wrapped
- **User rating integration**: Track which wrapping methods work best

**Advanced Features**:

**11. Fourier-Based Wrapping**:
```python
# Optimize in frequency domain for perfect continuity
class FourierContinuousParam:
    """
    Parametrize image in Fourier domain where periodic
    boundary conditions are natural.
    """
    - Zero out non-periodic frequencies
    - Enforce conjugate symmetry for real images
    - Smooth falloff at boundaries
```

**12. Neural Texture Synthesis Integration**:
- Use wrapping visualizations as texture exemplars
- Generate infinite variations of continuous patterns
- Style transfer with tiling constraints
- Procedural texture generation guided by visualizations

**13. Comparison Dashboard**:
- Side-by-side: wrapped vs non-wrapped
- Tiling preview: show 3×3 grid automatically
- Quality metrics: seam visibility, continuity scores
- Export best candidates

**Implementation Details**:

**New Functions**:
```python
# Enhanced wrapping transforms
def vertical_wrap_transform(img, wrap_factor=0.3)
def bidirectional_wrap_transform(img, h_factor=0.3, v_factor=0.3)
def radial_wrap_transform(img, wrap_factor=0.3, center=None)
def spherical_wrap_transform(img, wrap_factor=0.3)

# Quality assessment
def compute_seam_visibility(img, direction='horizontal')
def compute_continuity_score(img, direction='horizontal')
def compute_tileability_score(img)

# Visualization helpers
def create_tiled_preview(img, tile_count=(3, 3))
def export_tileable_texture(img, filename, format='png')
def create_scrolling_animation(img, duration=5.0, fps=30)

# Integration with generator
class ContinuousVisualizationGenerator(EnhancedVisualizationGenerator):
    def generate_continuous_visualization(
        self,
        objective_params,
        wrapping_type='bidirectional',
        wrap_factor=0.5,
        quality_threshold=0.8
    )
```

**Configuration Example**:
```python
from continuous_visualizations import ContinuousVisualizationGenerator

generator = ContinuousVisualizationGenerator(
    model_name='inception_v1',
    image_size=512,
    wrapping_config={
        'enabled': True,
        'types': ['horizontal', 'vertical', 'bidirectional'],
        'wrap_factor': 0.5,
        'enforce_continuity': True,
        'generate_previews': True,
        'quality_check': True,
    }
)

# Generate continuous visualization
result = generator.generate_continuous_visualization(
    objective_params={'layer': 'mixed4c', 'channel': 42},
    wrapping_type='bidirectional',
    export_formats=['png', 'preview', 'animated'],
)
```

**Atlas Integration**:
```python
# Generate continuous atlas
python atlas_generator.py \
    --layers mixed3a mixed3b mixed4a mixed4b mixed4c mixed4d mixed4e mixed5a mixed5b \
    --enable-continuous \
    --wrapping-types bidirectional \
    --generate-tiled-previews \
    --output-dir atlas/continuous_complete
```

**Use Cases**:
- **Seamless textures**: For games, 3D modeling, graphic design
- **Wallpapers**: Desktop/mobile backgrounds that tile infinitely
- **Pattern design**: Fashion, textile, surface design
- **VR/AR content**: 360° environments and skyboxes
- **Video backgrounds**: Looping video content
- **Generative art**: Infinite patterns for digital art
- **Research**: Study periodic structures in neural representations
- **Data augmentation**: Tileable training data for CNNs

**Benefits**:
- ✅ **Already partially implemented** - leverage existing code
- 🎨 Production-ready tileable outputs
- 🔄 Perfect continuity at boundaries
- 📊 Quality metrics for wrapping effectiveness
- 🎯 Multiple wrapping strategies for different use cases
- 🖼️ Export in multiple formats for different applications
- 🔧 Easy integration with existing visualization pipeline

**Time Estimates**:
- Basic enhancements (vertical, bidirectional): 1-2 weeks
- Quality metrics: 1 week
- Advanced features (radial, spherical): 2-3 weeks
- Interactive tools: 2 weeks
- Full integration with atlas: 1 week

**Dependencies**:
- Uses existing `wrapping_transforms.py` (already in codebase)
- `scipy` for FFT-based methods
- `opencv` for seam detection (optional)
- `pillow` for export formats

**Priority**: This feature has a working foundation already in the codebase (`wrapping_transforms.py`, `demo_wrapping.py`), making it quick to enhance and productionize.

---

### 1.4 Perceptual Quality Metrics

**Module**: `quality_metrics.py`

Add objective quality assessment beyond user ratings.

**Features**:
- **CLIP-based scoring**: Measure semantic coherence using CLIP embeddings
- **Frechet Distance**: Compare generated images to natural image distributions
- **Structural similarity**: SSIM, MS-SSIM for texture quality
- **Edge/frequency analysis**: Measure information content in different frequency bands
- **Colorfulness metrics**: Compute color diversity and saturation
- **Integration**: Automatically compute during generation, store in CSV

**Benefits**:
- Objective quality benchmarking
- Better evolutionary fitness functions
- Automatic outlier detection (failed generations)

---

### 1.5 Activation Statistics Analysis

**Module**: `activation_statistics.py`

Analyze the actual activation patterns during visualization optimization.

**Features**:
- Record activation magnitude over optimization steps
- Track gradient flow and dead neurons
- Analyze activation sparsity and distribution
- Visualize optimization trajectory in activation space
- Compare activation patterns across objective types

**Use Cases**:
- Understand what makes certain objectives harder to optimize
- Detect optimization failures early
- Tune learning rates and optimization parameters per layer

---

### 1.6 Information-Theoretic Segmentation Objective ⭐⭐⭐ (Lyn Hibbard's Algorithm)

**Module**: `segmentation_objective.py`, enhancements to `objective_generators.py`

Integrate Lyn Hibbard's information-theoretic segmentation algorithm to guide visualization optimization toward well-structured images with clear figure-ground separation.

**Conceptual Foundation**:

The core principle is to maximize the **relative entropy (KL divergence) between "figure" and "ground"** regions:
- **Figure**: Central region containing the primary neural feature being visualized
- **Ground**: Surrounding background region with significantly different statistical properties
- **Information-Theoretic Measure**: Use relative entropy to quantify the distinctness between regions

This encourages the optimization to generate interpretable visualizations where the activated feature is clearly separated from the background, improving both visual quality and interpretability.

**Implementation Details**:

**1. Segmentation Module** (`segmentation_objective.py`):
```python
class SegmentationObjective:
    """
    Information-theoretic segmentation objective using Lyn Hibbard's algorithm.
    """
    - compute_figure_ground_separation()
    - estimate_figure_region()  # e.g., center region, thresholded region
    - compute_relative_entropy()  # KL divergence between figure/ground
    - compute_segmentation_loss()  # Differentiable loss for optimization
    - visualize_figure_ground()  # Show segmentation overlay
```

**2. Region Detection Strategies**:
- **Spatial center**: Fixed NxN center region (simple baseline)
- **Thresholded**: Regions above intensity threshold (adaptive)
- **Attention-based**: Regions with highest gradient magnitude
- **Watershed**: Connected component analysis for natural boundaries
- **Multi-scale**: Hierarchical detection at multiple scales

**3. Statistical Measures for Entropy Computation**:

For each region (figure/ground), compute distributions:
```python
# Histogram-based entropy
- Intensity distribution (grayscale)
- RGB color distribution
- Gabor response distribution
- Texture statistics (entropy, contrast, correlation)

# Statistical moments
- Mean, variance, skewness, kurtosis
- Spatial autocorrelation
- Edge density and orientation

# Information-theoretic measures
- Shannon entropy H(region)
- Relative entropy D_KL(figure || ground)
- Mutual information I(figure; ground)
- Jensen-Shannon divergence (symmetric alternative)
```

**4. Integration with Optimization Loop**:

During visualization generation, combine objectives:
```python
total_loss = (
    lambda_primary * primary_objective_loss  # Channel/neuron/Gabor objective
    + lambda_segmentation * segmentation_loss  # Information-theoretic separation
    + lambda_regularization * regularization_loss  # Standard regularization
)
```

Configurable weights (defaults):
- `lambda_primary`: 1.0 (main feature visualization)
- `lambda_segmentation`: 0.3-0.5 (encourage segmentation, secondary)
- `lambda_regularization`: 0.01 (prevent artifacts)

**5. New Database Columns**:
```python
- figure_entropy           # Shannon entropy of figure region
- ground_entropy          # Shannon entropy of ground region
- relative_entropy        # KL divergence D_KL(figure || ground)
- segmentation_quality    # Overall figure-ground separation score
- figure_region_path      # Visualization of segmented regions
- figure_area_fraction    # What % of image is "figure"
- figure_center_offset    # How centered is the figure
```

**6. Configuration in Enhanced Generator**:

```python
segmentation_config = {
    'enabled': True,
    'region_detection': 'thresholded',  # or 'center', 'attention', 'watershed'
    'entropy_measure': 'kl_divergence',  # or 'jensen_shannon', 'mutual_information'
    'figure_weight': 0.4,  # Strength of segmentation objective
    'visualize_regions': True,  # Save figure/ground overlay
}

# Use in generation
generator = EnhancedVisualizationGenerator(
    model_name='inception_v1',
    segmentation_config=segmentation_config
)

result = generator.generate_single_visualization(
    objective_params={
        'layer': 'mixed4c',
        'channel': 42,
    },
    enable_segmentation=True,
)
```

**7. Segmentation-Aware Evolution**:

Mutation system can target segmentation quality:
```python
# Reward mutations that improve segmentation
fitness_score = (
    alpha * user_rating
    + beta * relative_entropy  # Information-theoretic separation
    + gamma * novelty_score
)

# Select parents with good segmentation
best_segmented = visualizations.nlargest(k, 'relative_entropy')
```

**8. Analysis Tools**:
```python
def compare_segmentation_quality(visualizations):
    """Compare segmentation effectiveness across layers/channels"""
    # Plot: relative_entropy vs layer
    # Plot: segmentation_quality vs user_rating
    # Identify: which layers benefit most from segmentation

def analyze_figure_statistics(image, figure_region):
    """Extract and analyze figure region statistics"""
    # Texture features, color histograms, edge density

def create_segmentation_visualization(image, figure_mask):
    """Create publication-ready figure-ground visualization"""
    # Overlay showing figure/ground boundary
    # Entropy heatmaps for each region
```

**9. Use Cases**:
- **Improved interpretability**: Visualizations with clear semantic structure
- **Evolutionary guidance**: Fitness function incorporating information theory
- **Layer analysis**: Identify which layers naturally produce segmentable features
- **Quality control**: Automatic flagging of "messy" vs "well-segmented" visualizations
- **Publication-ready**: Figures with clear feature separation for papers
- **Research questions**:
  - Do deeper layers produce more segmentable features?
  - Can we predict segmentation quality from layer type?
  - How does segmentation quality correlate with neural network performance?

**10. Advanced Features**:

**Multi-Scale Segmentation**:
```python
# Compute segmentation at multiple scales
scales = [32, 64, 128, 256]
segmentation_objectives = [
    create_segmentation_objective(scale, entropy_measure='kl')
    for scale in scales
]
# Combine objectives: maximize segmentation at all scales
```

**Adaptive Weighting**:
```python
# Adjust lambda_segmentation based on optimization progress
def adaptive_segmentation_weight(step, total_steps):
    # Start low, increase as optimization converges
    # Helps primary objective explore, then refine structure
    return 0.1 + 0.4 * (step / total_steps)
```

**Information-Theoretic Metrics Dashboard**:
- Plot relative entropy over optimization steps
- Compare figure vs ground entropy distributions
- Visualize segmentation quality across layers/channels
- Identify "hard to segment" vs "naturally segmentable" features

**11. Research Integration**:

**Comparison with Hibbard's Original Work**:
- Implement core relative entropy calculation
- Compare to original segmentation algorithm
- Validate that KL divergence effectively separates regions
- Document differences/improvements for neural features

**Theoretical Justification**:
- High relative entropy = figure and ground have very different statistics
- This enforces structure without hand-crafted spatial priors
- Information-theoretic approach is principled and interpretable
- Connection to visual perception research on figure-ground separation

**12. Dependencies**:
- `numpy`, `scipy` (for entropy computation)
- `scikit-image` (optional, for watershed/morphology)
- `opencv-python` (optional, for advanced region detection)
- No new external dependencies required for basic version

**13. Performance**:
- Minimal overhead (~10-20% slower generation) due to entropy computation
- Can be computed in parallel with primary objective
- Entropy computation is differentiable (can backprop through it)

**Time Estimates**:
- Basic implementation (center region + KL divergence): 3-4 days
- Region detection methods: 2-3 days
- Integration with EnhancedVisualizationGenerator: 2 days
- Mutation system enhancement: 2 days
- Analysis tools and dashboard: 3-4 days
- Total: ~2 weeks for complete implementation

**Integration with Roadmap**:
- Works with existing objective types (channel, neuron, Gabor)
- Enhances atlas generation (1.2) - produces more interpretable visualizations
- Complements perceptual quality metrics (1.4) - adds structural quality
- Improves mutation system - segmentation quality as fitness component
- Compatible with depth analysis (1.1) - both provide structural understanding

**Benefits Over Pure Generative Optimization**:
- ✅ Emergence of semantic structure from information theory (not imposed)
- ✅ Highly interpretable - figure-ground separation is cognitively natural
- ✅ Reproducible - information-theoretic measures are objective and mathematical
- ✅ Scalable - works with any visualization objective
- ✅ Research-grade - principled approach suitable for publications
- ✅ Tunable - adjustable weights allow balancing primary vs segmentation objectives

---

## Priority 2: Advanced Objective Types

### 2.1 Multi-Layer Objectives

**Module**: Extensions to `objective_generators.py`

Create objectives that combine multiple layers.

**Features**:
- **Style transfer objectives**: Match style from one layer, content from another
- **Hierarchical objectives**: Combine early layer (texture) + late layer (semantics)
- **Cross-layer correlations**: Maximize similarity between two layer activations
- **Layer interpolation**: Smooth transitions between layer visualizations

**Implementation**:
- `generate_multilayer_objective_params()`
- `create_style_content_objective()`
- `create_hierarchical_objective()`

---

### 2.2 Temporal/Video Objectives

**Module**: `temporal_objectives.py`

Generate video sequences with coherent visualizations.

**Features**:
- **Smooth interpolation**: Create videos morphing between objectives
- **Temporal consistency**: Enforce frame-to-frame coherence
- **Animation objectives**: Create "breathing" effects (see TODO in codebase)
- **Rotation/3D transforms**: Visualizations from multiple viewpoints
- **Integration with Video Depth Anything**: Consistent depth across video frames

**Implementation**:
- Parameter interpolation between keyframe objectives
- Optical flow constraints for smoothness
- Frame-by-frame generation with warm-starting

---

### 2.3 Adversarial Objectives

**Module**: `adversarial_objectives.py`

Create objectives that maximize interesting adversarial properties.

**Features**:
- **Fooling objectives**: Images that activate one channel but visually resemble another class
- **Robust features**: Objectives that remain stable under perturbations
- **Attribution-based objectives**: Target specific attribution patterns
- **Counterfactual objectives**: Minimal changes to flip predictions

---

## Priority 3: Visualization & Interface Enhancements

### 3.1 Advanced Gradio Features

**Module**: Extensions to `gradio_visualization_explorer.py`

**Features**:
- **Drag-and-drop mutation**: Drag two visualizations to create a crossover child
- **Side-by-side comparison**: Compare multiple visualizations with synchronized zoom
- **Filter by depth metrics**: Filter visualizations by depth complexity (after depth integration)
- **Batch operations**: Rate/tag multiple visualizations at once
- **Export collections**: Export filtered sets as image galleries or videos
- **Real-time generation**: Stream visualization progress with intermediate results
- **Parameter editors**: Visual sliders for Gabor parameters with live preview

---

### 3.2 3D Visualization Interface

**Module**: `visualization_3d.py`

Interactive 3D representations of visualizations and depth maps.

**Features**:
- **3D surface plots**: Render images as 3D surfaces using depth
- **Point cloud visualization**: Convert depth maps to 3D point clouds
- **VR/AR export**: Export visualizations for immersive viewing
- **360° rotation**: Interactive rotation of depth-enhanced visualizations
- **Stereo pair generation**: Create stereoscopic 3D images

**Tools**: `open3d`, `trimesh`, `PyVista` for 3D rendering

---

### 3.3 Activation Atlas/Map Visualization

**Module**: `atlas_builder.py`

Build 2D maps of neural network activation space (like original Activation Atlas paper).

**Features**:
- **UMAP/t-SNE projection**: Reduce high-dimensional activations to 2D
- **Grid layout**: Arrange visualizations in semantic grid
- **Interactive exploration**: Click on atlas to generate similar visualizations
- **Cluster analysis**: Identify semantic clusters in activation space
- **Path tracing**: Show trajectories through activation space during optimization

---

## Priority 4: Performance & Scalability

### 4.1 Distributed Generation

**Module**: `distributed_generator.py`

Scale up generation using multiple GPUs or distributed computing.

**Features**:
- **Multi-GPU support**: Parallelize batch generation across GPUs
- **Ray/Dask integration**: Distribute generation across compute cluster
- **Cloud integration**: AWS/GCP batch job support
- **Progress tracking**: Centralized monitoring of distributed jobs
- **Fault tolerance**: Resume interrupted generation jobs

---

### 4.2 Caching & Optimization

**Module**: Extensions to `visualization_logger.py`

**Features**:
- **Model activation caching**: Cache intermediate activations for faster iteration
- **Smart checkpointing**: Save optimization state for resuming
- **Parameter deduplication**: Detect and avoid generating duplicate visualizations
- **Incremental CSV**: Stream results to CSV without loading entire database
- **SQLite backend**: Optional faster database backend for large-scale experiments

---

### 4.3 Mixed Precision & JIT Compilation

**Module**: `optimization_utils.py`

Speed up generation with modern PyTorch features.

**Features**:
- **Automatic Mixed Precision (AMP)**: FP16 training for 2x speedup
- **TorchScript compilation**: JIT compile model for faster inference
- **ONNX export**: Export objectives for cross-platform optimization
- **Custom CUDA kernels**: Optimize bottleneck operations

---

## Priority 5: Analysis & Research Tools

### 5.1 Comparative Analysis Dashboard

**Module**: `analysis_dashboard.py` (Streamlit or Dash)

Comprehensive analysis interface for research.

**Features**:
- **Layer comparison**: Side-by-side analysis of different layers
- **Objective type comparison**: Compare channel vs neuron vs Gabor objectives
- **Evolution tracking**: Visualize mutation lineages and fitness over generations
- **Statistical summaries**: Distribution plots, correlation matrices
- **Export reports**: Generate PDF/HTML reports with figures
- **Depth analysis** (after Priority 1.1): Compare depth characteristics across layers

---

### 5.2 Automated Hypothesis Testing

**Module**: `hypothesis_testing.py`

Automate scientific experiments on visualizations.

**Features**:
- **A/B testing framework**: Compare two objective types statistically
- **Correlation analysis**: Find relationships between parameters and quality
- **Regression models**: Predict quality from parameters
- **Causal inference**: Identify what parameter changes cause quality improvements
- **Automated report generation**: LaTeX/markdown reports with statistical tests

---

### 5.3 Feature Attribution Integration

**Module**: `attribution_analysis.py`

Integrate interpretability methods to analyze visualizations.

**Features**:
- **GradCAM integration**: Show which image regions drive activations
- **Integrated Gradients**: Attribution maps for understanding optimization
- **SHAP for images**: Feature importance for visualization quality
- **Attention visualization**: If using vision transformers
- **Cross-reference with depth**: Correlate depth with attribution patterns

---

## Priority 6: Dataset & Training Enhancements

### 6.1 InceptionV1 Training from Scratch ⭐⭐ (Your Request #2)

**Module**: `train_inception.py`, `imagenet_loader.py`, `training_config.py`

Train InceptionV1 (GoogLeNet) from scratch on ImageNet dataset with modern best practices.

**Implementation Details**:

**Dataset Setup**:
- **ImageNet ILSVRC2012**: 1.3M training images, 50K validation images, 1000 classes
- **Download**: Via https://www.image-net.org/download.php (requires approval, ~5 days wait)
- **Files Required**:
  - `ILSVRC2012_img_train.tar` (138 GB)
  - `ILSVRC2012_img_val.tar` (6.3 GB)
  - `ILSVRC2012_devkit_t12.tar.gz` (metadata)
- **PyTorch Integration**: Use `torchvision.datasets.ImageNet` class with automatic extraction

**Data Preprocessing**:
- Resize images to 256×256 (scale shorter side, center crop)
- Subtract per-channel mean pixel values (ImageNet statistics)
- Extract random 224×224 patches during training
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Data Augmentation Pipeline**:
```python
# Geometric augmentations
- Random horizontal flips (p=0.5)
- Random 224×224 crops from 256×256 images
- Random rotation (±15 degrees)
- Random scale (0.8 to 1.2)

# Color augmentations
- PCA color jittering (ImageNet-style)
- Random brightness/contrast/saturation
- Random hue shifts
- ColorJitter with probability 0.8

# Advanced augmentations
- RandAugment or AutoAugment
- Mixup (alpha=0.2) or CutMix
- Random erasing (p=0.25)
```

**Training Configuration**:
- **Model**: InceptionV1 (GoogLeNet) - 6.8M parameters, 147 layers
- **Batch size**: 256 (distributed across GPUs)
- **Epochs**: 90-120 epochs for full convergence
- **Optimizer**: SGD with momentum=0.9 or Adam (adaptive learning)
- **Initial learning rate**: 0.1 (SGD) or 0.001 (Adam)
- **Learning rate schedule**:
  - Step decay: multiply by 0.1 at epochs [30, 60, 90]
  - Cosine annealing with warm restarts
  - Warmup: Linear increase for first 5 epochs
- **Weight decay**: 1e-4 (L2 regularization)
- **Gradient clipping**: Max norm = 1.0

**Regularization**:
- **Dropout**: 0.4 in fully connected layers (original paper)
- **Batch normalization**: Add to all convolutional layers (modern practice)
- **Label smoothing**: ε=0.1 to prevent overconfidence
- **Stochastic depth**: Drop layers with probability during training

**Auxiliary Classifiers** (InceptionV1 specific):
- Two auxiliary classifiers at intermediate layers
- Weight auxiliary losses: 0.3 × (aux1_loss + aux2_loss) + 1.0 × main_loss
- Helps gradient flow in deep network
- Disabled during inference

**Multi-GPU Training**:
- **DataParallel** or **DistributedDataParallel** (DDP preferred)
- Synchronized batch normalization across GPUs
- Gradient accumulation if memory limited
- Mixed precision training (FP16) with `torch.cuda.amp`

**Monitoring & Logging**:
- **Metrics**: Top-1 accuracy, Top-5 accuracy, loss curves
- **Validation**: Every epoch on 50K validation images
- **Checkpointing**: Save best model, last model, periodic checkpoints
- **Tensorboard**: Real-time training visualization
- **Weights & Biases** (optional): Advanced experiment tracking
- **Early stopping**: Stop if no improvement for 10 epochs

**Expected Performance**:
- **Target**: Top-5 error ~6.7% (original paper)
- **Training time**:
  - Single GPU (V100): ~7-10 days
  - 4× V100 GPUs: ~2-3 days
  - 8× A100 GPUs: ~1-2 days
- **Convergence**: Typically converges after 90 epochs

**Integration with Visualization System**:
- **Checkpoint compatibility**: Save in format compatible with Lucent
- **Layer tracking**: Log layer statistics during training
- **Periodic visualization**: Generate feature visualizations every N epochs
- **Compare trained vs pretrained**: Visualize differences in learned features
- **Incremental visualization**: Visualize model at different training stages
- **Custom objectives**: Train with custom objectives from visualization system

**New Features**:
- `ImageNetDataModule` - DataLoader with all augmentations
- `InceptionV1Trainer` - Training loop with all best practices
- `TrainingConfig` - Comprehensive configuration dataclass
- `train_from_scratch()` - Main training entry point
- `resume_from_checkpoint()` - Resume interrupted training
- `evaluate_model()` - Comprehensive evaluation on validation set
- `visualize_training_progress()` - Generate visualizations during training
- `compare_checkpoints()` - Compare models at different training stages

**Configuration Files**:
```yaml
# config/inception_training.yaml
model:
  name: inception_v1
  num_classes: 1000
  auxiliary_classifiers: true
  dropout: 0.4

dataset:
  root: /path/to/imagenet
  train_split: train
  val_split: val
  num_workers: 8

training:
  batch_size: 256
  epochs: 90
  optimizer: sgd
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 1e-4
  scheduler: cosine_annealing

augmentation:
  random_crop: 224
  random_flip: true
  color_jitter: true
  mixup_alpha: 0.2
```

**Dependencies**:
- `torch >= 2.0`
- `torchvision >= 0.15`
- `pytorch-lightning` (optional, for cleaner training code)
- `timm` (optional, for advanced augmentations)
- `tensorboard` or `wandb` for logging

**Use Cases**:
- **Research**: Study how features evolve during training
- **Comparison**: Compare randomly initialized vs pretrained features
- **Custom objectives**: Train models optimized for visualization quality
- **Transfer learning**: Train on ImageNet, fine-tune on custom datasets
- **Ablation studies**: Test different training configurations
- **Reproducibility**: Verify published results

**Validation**:
- Reproduce original InceptionV1 results (~93.3% Top-5 accuracy)
- Compare with PyTorch pretrained models
- Generate visualizations at checkpoints: epoch 0, 10, 30, 60, 90
- Analyze how feature visualizations change during training

---

### 6.2 Custom Model Support

**Module**: Extensions to `lucent_layer_utils.py`

Support arbitrary PyTorch models beyond current set.

**Features**:
- **Auto-discovery**: Automatically find visualizable layers in any model
- **Vision Transformers**: Support ViT, DINO, MAE, etc.
- **Diffusion models**: Visualize U-Net layers in Stable Diffusion
- **Custom architectures**: Easy integration of user-provided models
- **Model zoo expansion**: Pre-configured support for 50+ popular models

---

### 6.3 Dataset-Conditioned Objectives

**Module**: `dataset_objectives.py`

Use real images to guide visualizations.

**Features**:
- **Style matching**: Generate visualizations matching real image styles
- **Class-conditional**: Generate visualizations for specific ImageNet classes
- **Exemplar-based**: Start from real images and optimize
- **Data augmentation**: Use visualizations for training data augmentation
- **Transfer learning**: Visualizations as pre-training for downstream tasks

---

## Priority 7: Experimental Features

### 7.1 NeRF-Style 3D Visualizations ⭐⭐⭐⭐

**Module**: `nerf_visualizations.py`, `nerf_renderer.py`, `volumetric_optimization.py`

Generate 3D neural radiance fields (NeRFs) representing network activations with full volumetric representation and novel view synthesis.

**Conceptual Foundation**:

NeRF-based visualization creates a **continuous 3D volumetric representation** of what a neural network channel or neuron "sees" across all spatial dimensions. Unlike 2D feature visualizations, NeRF-based approach:
- Generates 3D-consistent features that look coherent from any viewing angle
- Captures volumetric structure (not just surface appearance)
- Enables interactive 3D exploration and 360° rotation
- Integrates naturally with depth estimation for geometry guidance
- Produces publication-ready 3D visualizations and videos

**Implementation Details**:

**1. Core NeRF Architecture**:

```python
class ActivationNeRF:
    """
    Neural Radiance Field for network activation visualization.

    Represents activation patterns as a continuous 3D function:
    F_θ(x, y, z, θ, φ) → (r, g, b, density)

    Where:
    - (x, y, z): 3D spatial coordinates
    - (θ, φ): View direction (azimuth, elevation)
    - (r, g, b): RGB color at that point
    - density: Volume opacity at that point
    """

    # Positional encoding for coordinates
    - encode_position(x, y, z)          # Fourier features for position
    - encode_direction(theta, phi)      # Fourier features for viewpoint

    # Core network components
    - mlp_coarse(pos_encoded, dir_encoded) → (rgb, density)
    - mlp_fine(pos_encoded, dir_encoded)   → (rgb, density)  # Hierarchical sampling

    # Rendering operations
    - render_rays(rays, samples_per_ray)
    - volumetric_rendering(rgb, density, z_vals)
    - compute_loss(rendered, target)
```

**2. Input Representation Strategies**:

**Option A: Channel NeRF** (most practical):
```python
# Optimize a NeRF that maximizes activation of a specific channel
objective_params = {
    'layer': 'mixed4c',
    'channel': 42,
    'nerf_type': 'channel',      # Maximize channel activation
    'use_depth_guidance': True,   # Use depth estimation to guide geometry
}

# Result: 3D volumetric visualization of what channel 42 encodes
```

**Option B: Neuron NeRF** (spatial):
```python
# Optimize a NeRF that maximizes a specific spatial location (neuron)
objective_params = {
    'layer': 'mixed4c',
    'spatial_pos': (10, 15),      # (y, x) position in feature map
    'nerf_type': 'neuron',
}
```

**Option C: Multi-Channel NeRF** (semantic):
```python
# Optimize a NeRF combining multiple channels
objective_params = {
    'layer': 'mixed4c',
    'channels': [10, 20, 30],     # RGB from three channels
    'channel_assignment': 'rgb',  # R←ch10, G←ch20, B←ch30
    'nerf_type': 'multi_channel',
}
```

**3. NeRF Optimization Loop**:

```python
# Combine standard objectives with volumetric rendering
total_loss = (
    lambda_activation * activation_loss         # Primary: maximize channel activation
    + lambda_rendering * rendering_loss         # Perceptual quality of rendered view
    + lambda_depth * depth_guidance_loss        # Encourage 3D geometric consistency
    + lambda_segmentation * segmentation_loss   # (Optional) Figure-ground separation
    + lambda_smooth * smoothness_loss           # Spatial smoothness
)

# Optimization process:
for step in range(num_steps):
    # 1. Sample random camera rays
    rays = sample_camera_rays(cam_params)

    # 2. Coarse pass: hierarchical sampling
    z_samples_coarse = stratified_sample(ray_length)
    rgb_coarse, density_coarse = mlp_coarse(rays, z_samples_coarse)

    # 3. Fine pass: importance sampling
    z_samples_fine = importance_sample(density_coarse, z_samples_coarse)
    rgb_fine, density_fine = mlp_fine(rays, z_samples_fine)

    # 4. Volume rendering
    image_fine = volumetric_render(rgb_fine, density_fine, z_samples_fine)

    # 5. Compute activation loss (target activations)
    activation_loss = compute_activation_loss(z_samples_fine, density_fine)

    # 6. Compute rendering loss (optional: match target 2D image)
    rendering_loss = mse(image_fine, target_image)

    # 7. Depth guidance (optional: integrate depth estimation)
    depth_from_nerf = compute_expected_depth(density_fine, z_vals)
    depth_from_estimator = depth_anything(render_2d(image_fine))
    depth_loss = depth_consistency(depth_from_nerf, depth_from_estimator)

    # 8. Backward pass
    total_loss = ... (as above)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

**4. Positional Encoding (Critical for NeRF)**:

```python
def positional_encoding(x, L=10):
    """
    Map coordinates to high-dimensional space for better function approximation.
    Essential for NeRF to capture fine details.

    γ(x) = [sin(2^0 π x), cos(2^0 π x), ..., sin(2^(L-1) π x), cos(2^(L-1) π x)]
    """
    encoded = []
    for i in range(L):
        encoded.append(torch.sin((2**i) * np.pi * x))
        encoded.append(torch.cos((2**i) * np.pi * x))
    return torch.cat(encoded)

# For 3D position: encode x, y, z separately → 6L dimensions
# For direction: encode theta, phi separately → 4L dimensions
```

**5. Rendering Pipeline**:

```python
def render(rays, model, num_coarse_samples, num_fine_samples):
    """
    Ray marching through volumetric NeRF representation.
    """
    # 1. Coarse sampling: uniformly sample along ray
    z_coarse = torch.linspace(z_near, z_far, num_coarse_samples)
    pts_coarse = rays.origin[:, None] + rays.direction[:, None] * z_coarse[None, :]

    # 2. Evaluate coarse NeRF
    rgb_coarse, density_coarse = model.mlp_coarse(
        encode_position(pts_coarse),
        encode_direction(rays.direction)
    )

    # 3. Volumetric rendering (coarse)
    weights = compute_weights(density_coarse, z_coarse)
    image_coarse = weighted_sum(rgb_coarse * weights)

    # 4. Fine sampling: importance sampling based on coarse weights
    z_fine = importance_sample(weights, z_coarse)
    pts_fine = rays.origin[:, None] + rays.direction[:, None] * z_fine[None, :]

    # 5. Evaluate fine NeRF
    rgb_fine, density_fine = model.mlp_fine(
        encode_position(pts_fine),
        encode_direction(rays.direction)
    )

    # 6. Final volumetric rendering (fine)
    weights_fine = compute_weights(density_fine, z_fine)
    image_fine = weighted_sum(rgb_fine * weights_fine)

    return image_coarse, image_fine, weights_fine, z_fine
```

**6. Camera Control & Viewpoint Sampling**:

```python
# Define camera trajectory for exploration
camera_poses = [
    {'azimuth': 0, 'elevation': 0, 'distance': 1.5},      # Front
    {'azimuth': 90, 'elevation': 0, 'distance': 1.5},     # Right
    {'azimuth': 180, 'elevation': 0, 'distance': 1.5},    # Back
    {'azimuth': 270, 'elevation': 0, 'distance': 1.5},    # Left
    {'azimuth': 0, 'elevation': 45, 'distance': 1.5},     # Top
]

# Render multiple viewpoints
for pose in camera_poses:
    rays = pose_to_rays(pose)
    image = render(rays, nerf_model, ...)
    save_image(f"view_{pose['azimuth']}_{pose['elevation']}.png", image)
```

**7. Depth Guidance Integration**:

```python
# Use Depth Anything V2 to guide volumetric geometry
def depth_guided_loss(image_rendered, nerf_depth, learnable_depth_scale):
    """
    Align NeRF-derived depth with depth estimation model.

    This creates a "reality anchor" - the NeRF learns 3D structure
    that's consistent with depth prediction models trained on real images.
    """

    # Get depth from depth model
    depth_model = load_depth_anything_v2('large')
    depth_predicted = depth_model(image_rendered)

    # Extract depth from NeRF volumetric representation
    depth_from_nerf = compute_expected_depth(
        density_volumetric=density_samples,
        z_values=z_samples,
    )

    # Align scales and compute loss
    depth_scaled = depth_from_nerf * learnable_depth_scale
    loss = huber_loss(depth_scaled, depth_predicted)

    return loss
```

**8. Video Generation from NeRF**:

```python
# Generate smooth video by interpolating camera poses
camera_trajectory = interpolate_poses(keyframe_poses, num_frames=300)

video_frames = []
for pose in camera_trajectory:
    rays = pose_to_rays(pose)
    frame = render(rays, nerf_model)
    video_frames.append(frame)

# Encode to video file
write_video('nerf_visualization_360.mp4', video_frames, fps=30)
```

**9. Output Formats**:
- **Single image**: High-quality render from canonical viewpoint
- **Multi-view grid**: 2×3 or 3×3 grid showing different angles
- **360° video**: Full rotation animation (mp4, webm)
- **Interactive viewer**: Web-based 3D visualization (Three.js)
- **3D mesh export**: Extract iso-surface as .obj for 3D software
- **Point cloud**: Export density field as point cloud

**10. New Database Columns**:
```python
- nerf_model_path         # Path to saved NeRF weights
- nerf_canonical_image    # Front-facing view
- nerf_multiview_grid     # 6-view grid for thumbnail
- nerf_video_path         # Path to 360° video
- nerf_depth_guidance     # Was depth guidance used?
- nerf_training_steps     # Optimization steps for NeRF
- nerf_loss_history       # JSON-encoded training curve
- nerf_rendering_time     # Time to render reference images
```

**11. Integration with Existing System**:

```python
# Seamless integration with EnhancedVisualizationGenerator
generator = EnhancedVisualizationGenerator(
    model_name='inception_v1',
    visualization_type='nerf',  # NEW
    nerf_config={
        'enable_depth_guidance': True,
        'enable_segmentation': True,  # Can combine with seg objective
        'camera_trajectory': 'full_rotation',
        'render_resolution': 512,
        'num_training_steps': 2000,
    }
)

result = generator.generate_single_visualization(
    objective_params={'layer': 'mixed4c', 'channel': 42},
    visualization_type='nerf',
)

# Output includes:
# - result['canonical_image']: Front view
# - result['multiview_grid']: 6-view comparison
# - result['video_path']: 360° rotation video
# - result['nerf_weights']: Saved model for interactive exploration
```

**12. Advanced Features**:

**Animated NeRFs** (temporal 4D):
```python
# Extend NeRF to include time dimension: F_θ(x, y, z, t, θ, φ)
# Generate morphing NeRF between two channels

nerf_source = NeRF(channel=10)   # Initial state
nerf_target = NeRF(channel=42)   # Target state

# Interpolate between them
for alpha in linspace(0, 1, 60):
    nerf_interpolated = blend_nerf_weights(nerf_source, nerf_target, alpha)
    render_frame = render(rays, nerf_interpolated)
    # Frame shows smooth transition
```

**Semantic NeRFs** (multi-channel blending):
```python
# Composite multiple channels into single NeRF with semantic meaning
# E.g., Red = edge detector, Green = blob detector, Blue = texture

nerf_semantic = MultiChannelNeRF(
    channels={'red': 10, 'green': 20, 'blue': 30},
    blend_type='additive',  # or 'multiplicative', 'weighted'
)
```

**13. Configuration Example**:
```yaml
# config/nerf_config.yaml
nerf:
  model:
    coarse_network: [256, 256, 256, 256]  # MLP architecture
    fine_network: [256, 256, 256, 256, 256, 256, 256]
    pos_encoding_freq: 10  # Positional encoding frequency
    dir_encoding_freq: 4   # Direction encoding frequency

  rendering:
    num_coarse_samples: 64
    num_fine_samples: 128
    z_near: 0.1
    z_far: 8.0

  optimization:
    num_steps: 2000
    learning_rate: 1e-3
    optimizer: adam
    scheduler: exponential_decay

  guidance:
    depth_enabled: true
    depth_weight: 0.1
    segmentation_enabled: true
    segmentation_weight: 0.3

  rendering:
    output_resolution: 512
    num_views: 6
    generate_video: true
    video_fps: 30
```

**14. Time Estimates**:
- Basic NeRF implementation: 4-5 days
- Positional encoding & rendering: 2-3 days
- Depth guidance integration: 2-3 days
- Segmentation combination: 1-2 days
- Video generation: 1 day
- Interactive viewer: 2-3 days
- Analysis tools & dashboard: 2-3 days
- Total: ~3-4 weeks for complete implementation

**15. Research Opportunities**:
- **Novel view synthesis**: Can the NeRF generalize to unseen viewpoints?
- **3D semantics**: Do different layers encode different 3D structures?
- **Depth consistency**: How well do NeRF depths match real depth models?
- **Temporal coherence**: Can we create animated NeRFs showing feature evolution?
- **Cross-layer relationships**: How do 3D structures evolve across layers?

**Benefits**:
- ✅ Full 3D understanding of activation patterns
- ✅ Novel view synthesis for new perspectives
- ✅ Integration with depth estimation for geometric grounding
- ✅ Publication-ready visualizations with 360° videos
- ✅ Interactive exploration of neural representations
- ✅ Combination with segmentation for interpretable structure
- ✅ Unique research contribution for interpretability papers

---

### 7.2 Music/Audio-Driven Visualizations

**Module**: `audio_visualizations.py`

Create visualizations synchronized with audio.

**Features**:
- Beat-synchronized parameter modulation
- Frequency-responsive objective weights
- Music video generation from visualizations
- Audio feature extraction (tempo, key, mood) → visualization parameters

---

### 7.3 Interactive Optimization

**Module**: `interactive_optimization.py`

Human-in-the-loop optimization with real-time feedback.

**Features**:
- Live parameter tuning during optimization
- Interrupt and redirect optimization mid-run
- Collaborative optimization (multiple users)
- Gamification: Turn optimization into interactive exploration

---

## Implementation Priority Summary

**High Priority (Next 3-6 months)**:
1. ⭐ Depth Estimation Integration (1.1) - YOUR REQUEST #1
2. ⭐⭐ InceptionV1 Training from Scratch (6.1) - YOUR REQUEST #2
3. ⭐⭐⭐ Complete InceptionV1 Channel Atlas (1.2) - YOUR REQUEST #3
4. ⭐⭐⭐⭐ Enhanced Continuous/Wrapping Visualizations (1.3) - YOUR REQUEST #4 (Already 50% complete!)
5. ⭐⭐⭐⭐ Information-Theoretic Segmentation Objective (1.6) - Lyn Hibbard's Algorithm
6. ⭐⭐⭐⭐ NeRF-Style 3D Visualizations (7.1) - Neural Radiance Fields for Activation Visualization
7. Perceptual Quality Metrics (1.4)

**Medium Priority (6-12 months)**:
8. Activation Statistics Analysis (1.5)
9. Temporal/Video Objectives (2.2)
10. 3D Visualization Interface (3.2)
11. Distributed Generation (4.1)
12. Comparative Analysis Dashboard (5.1)

**Lower Priority (12+ months)**:
13. All remaining features based on research needs and user feedback

---

## Technical Debt & Code Quality

### Immediate Improvements

1. **Add comprehensive tests**: Unit tests for all objective generators
2. **Type hints everywhere**: Complete type annotation coverage
3. **Documentation**: Auto-generated API docs with Sphinx
4. **CLI interface**: Unified command-line tool for all operations
5. **Configuration system**: YAML/JSON configs instead of hardcoded parameters
6. **Logging improvements**: Structured logging with log levels
7. **Error handling**: Better error messages and recovery
8. **Code formatting**: Enforce black + isort + flake8
9. **Pre-commit hooks**: Automated code quality checks
10. **CI/CD**: GitHub Actions for testing and deployment

---

## Research Directions

### Open Questions to Explore

1. **Depth and Layer Depth**: Do deeper layers encode more 3D structure?
2. **Optimal mutation strategies**: What mutation rates/strengths work best?
3. **Cross-model transfer**: Do good parameters transfer across models?
4. **Semantic clustering**: Can we automatically discover semantic categories?
5. **Predictive models**: Can we predict quality before generation?
6. **Compression**: Can we compress visualizations to essential parameters?
7. **Artistic style**: Can we evolve visualizations matching artistic styles?
8. **Biological correspondence**: Do visualizations match V1 simple cells?

---

## Community Features

1. **Public gallery**: Web gallery of best visualizations
2. **Parameter sharing**: Share objective parameters as JSON
3. **Challenges/competitions**: Community challenges for best visualizations
4. **Model marketplace**: Upload and visualize custom models
5. **Tutorial notebooks**: Jupyter notebooks for education
6. **Paper integration**: Easy generation of figures for papers

---

## Notes

- This roadmap is a living document and will evolve based on:
  - User feedback and requests
  - Research findings
  - Technical feasibility
  - Community contributions

- All features should maintain backward compatibility with existing CSV databases

- Priority should be given to features that:
  - Enable new research insights
  - Significantly improve quality
  - Make the system more accessible
  - Scale to larger experiments

**Last Updated**: 2025-10-14
**Contributors**: Claude Code AI, User Requirements
