# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a **neural network feature visualization system** built on top of Lucent (a PyTorch library for neural network interpretability). It provides tools to:
- Generate visualizations of what neural network layers and channels "see"
- Create spatial and Gabor-filtered objectives for targeted feature visualization
- Mutate and evolve visualizations using genetic algorithms
- Score and rate visualizations using various metrics
- Browse and explore visualizations through interactive web interfaces (Gradio & Streamlit)

The system uses **parametric feature visualization** to understand what specific neurons, channels, or spatial regions in neural networks respond to.

## Python Environment

- **Python version**: 3.11 (specified in `.python-version`)
- **Virtual environment**: `.venv/` directory
- **Activation**:
  ```bash
  # Windows
  .venv\Scripts\activate

  # Linux/Mac
  source .venv/bin/activate
  ```

## Core Architecture

### 1. Objective Generation System (`objective_generators.py`)

The central system for creating visualization objectives with four types:
- **Channel objectives**: Target entire channels across spatial dimensions
- **Neuron objectives**: Target specific spatial positions (x,y) in a channel
- **Center objectives**: Target NxN regions with configurable offsets
- **Gabor objectives**: Target features using Gabor filter weights (orientation, frequency, phase)

Key functions:
- `generate_random_objective_params()` - Generate random objective parameters
- `create_objective_from_params()` - Convert parameters to Lucent objectives
- `mutate_objective_params()` - Create mutations with configurable rate/strength

### 2. Spatial Objectives (`spatial_objectives.py`)

Provides spatial targeting within feature maps:
- `create_center_nxn_objective()` - Create NxN center region objectives with offsets
- `create_corner_objectives()` - Target all four corners
- `create_edge_objectives()` - Target edges (top/bottom/left/right)
- `create_grid_objectives()` - Create 3x3 grid of objectives
- `create_random_objective()` - High-level wrapper combining multiple objective types

### 3. Gabor Objectives (`gabor_objectives.py`)

Advanced spatial filtering using Gabor functions:
- `create_gabor_weighted_objective()` - Create Gabor-filtered objectives with full parameter control
- `create_gabor_preset_objectives()` - Presets: edge_detector, vertical_edges, texture_fine, blob_detector, etc.
- `create_multi_orientation_gabor_objective()` - Combine multiple orientations
- Parameters include: sigma, lambda_freq, theta, psi, gamma (standard Gabor parameters)

**Important**: Gabor objectives in `spatial_objectives.py` use **logarithmic scaling** for sigma, lambda, and gamma parameters to explore parameter space more effectively.

### 4. Visualization Logging (`visualization_logger.py`)

Comprehensive pandas-based logging system:
- `VisualizationLogger` class - Central logging with CSV persistence
- Schema includes: objective params, spatial params, Gabor params, generation settings, quality metrics, parent/mutation tracking
- `log_visualization()` - Add new visualization to database
- `get_by_id()`, `get_by_layer()`, `get_best_rated()` - Query functions
- `export_for_gradio()` - Export subset for interactive exploration

### 5. Enhanced Generator (`enhanced_visualization_generator.py`)

Main generation engine with logging integration:
- `EnhancedVisualizationGenerator` class - Orchestrates generation with logging
- `generate_single_visualization()` - Generate one visualization with full parameter tracking
- `generate_batch()` - Batch generation with optional mutation
- `export_for_gradio()` - Export data for web interface

### 6. Mutation System (`visualization_mutator.py`)

Evolutionary algorithm for improving visualizations:
- `VisualizationMutator` class - Genetic algorithm implementation
- `MutationConfig` dataclass - Configure mutation parameters
- `VisualizationScore` dataclass - Multi-component scoring (user rating, time, novelty, complexity, fitness)
- `create_single_mutation()` - Create child from parent
- `evolutionary_generation()` - Multi-generation evolution with selection
- `analyze_mutation_success()` - Statistics on mutation effectiveness

### 7. Wrapping Transforms (`wrapping_transforms.py`)

Custom transforms for creating tileable/wrapping visualizations:
- `horizontal_wrap_transform()` - Blend left/right edges for seamless tiling
- `create_tiled_parameter()` - Initialize with tiled patterns
- `wrap_transform()` - Main wrapping transform for render pipeline

### 8. Layer Utilities (`lucent_layer_utils.py`)

Model introspection utilities:
- `get_visualizable_layers()` - Extract all layers compatible with Lucent
- `get_channels_from_lucent_name()` - Get channel count for a layer
- `get_layer_dimensions()` - Get spatial dimensions (H, W, C) for a layer
- Supports InceptionV1, ResNet, VGG, DenseNet naming conventions

### 9. Interactive Interfaces

**Gradio Interface** (`simple_gradio_interface.py`, `gradio_visualization_explorer.py`):
- Web-based visualization browser
- Generate new visualizations interactively
- Rate and score existing visualizations
- Create mutations with parameter control
- Launch with `launch_simple_explorer()`

**Streamlit Interface** (`streamlit_scoring.py`):
- Random image selection from folders
- YAML file generation for scoring workflows
- Recursive folder scanning

## Common Development Workflows

### Running Demos

The comprehensive demo shows all features:
```bash
python comprehensive_demo.py
```

This demonstrates:
1. Objective parameter generation
2. Enhanced visualization generation with logging
3. Mutation and evolution
4. Gradio interface launch

### Generate Visualizations

```bash
# Generate random visualizations
python generate_random_objective_visualizatio.py
```

### Launch Web Interfaces

```bash
# Gradio interface
python simple_gradio_interface.py

# Streamlit interface
streamlit run streamlit_scoring.py
```

### Debugging

VS Code launch configurations are provided in `.vscode/launch.json`:
- "Debug Python Module" - Debug any Python file
- "Debug Gradio Interface" - Debug Gradio app
- "Debug Enhanced Visualization Generator" - Debug generator
- "Debug Streamlit App" - Debug Streamlit with various port configurations

## Directory Structure

```
activation-atlas/
├── *.py                      # Main modules (described above)
├── .venv/                    # Virtual environment
├── screen_captures/          # Generated visualization images
├── logs/                     # Text logs with timestamps
├── onnx/                     # ONNX model exports (ignored)
├── demos_examples/           # Example scripts
│   ├── demo_gabor_objectives.py
│   └── example_usage.py
├── .vscode/                  # VS Code configuration
│   ├── launch.json           # Debug configurations
│   └── tasks.json            # Build tasks
└── IISStaticSite/           # Web deployment assets
```

## Key Dependencies

- **lucent** - Core visualization library (PyTorch-based)
- **torch** - PyTorch for neural networks
- **pandas** - DataFrame-based logging
- **gradio** - Web interface for interactive exploration
- **streamlit** - Alternative web interface
- **numpy** - Numerical operations
- **PIL/Pillow** - Image processing

## Important Implementation Details

### Lucent Naming Convention
Lucent uses underscore-separated names for layers (e.g., `mixed4a_1x1_pre_relu_conv`) while PyTorch uses dots (e.g., `mixed4a.1x1.pre_relu.conv`). The `lucent_layer_utils.py` module handles this conversion.

### CSV Database Schema
All visualizations are logged to CSV with a comprehensive schema that includes:
- Objective parameters (type, layer, channel, spatial, Gabor)
- Generation settings (image size, steps, learning rate)
- Quality metrics (generation time, loss, user rating)
- Parent/mutation tracking (parent_id, generation, mutation_type)
- Full JSON-serialized parameter dictionaries for exact reproduction

### Visualization ID Format
Visualizations use hash-based IDs generated from parameters to ensure uniqueness and enable parameter-based lookups.

### Gabor Parameter Ranges
When generating random Gabor objectives:
- **sigma**: 0.5 to 8.0 (log scale in spatial_objectives.py)
- **lambda_freq**: 0.5 to 6.0 (wavelength)
- **theta**: 0 to π (orientation)
- **psi**: 0 to 2π (phase)
- **gamma**: 0.5 to 2.0 (aspect ratio)

### Mutation System
The mutation system uses a multi-component fitness score:
- User rating (subjective quality)
- Generation time (efficiency)
- Novelty score (parameter uniqueness)
- Complexity score (objective complexity)
Combined with configurable weights into overall fitness.

## Model Support

Currently focused on **InceptionV1** but the architecture supports:
- InceptionV1, InceptionV3
- ResNet family (ResNet18, 34, 50, etc.)
- VGG family
- DenseNet family

All models accessed through `lucent.modelzoo`.

## Output Files

- **Visualizations**: `screen_captures/*.png` - Generated images
- **Database**: `visualizations_*.csv` or `demo_visualizations.csv` - Parameter database
- **Logs**: `logs/visualization_log_*.txt` - Detailed generation logs
- **Gradio exports**: `demo_gradio_data/` or similar - Subsets for web interface

## Tips for Development

1. **Always use the objective generators** - Don't manually create objectives; use `generate_random_objective_params()` for proper logging
2. **Enable CSV logging** - Pass `csv_filename` to ensure all parameters are tracked
3. **Use the mutation system for exploration** - Better than pure random generation
4. **Check layer compatibility** - Use `get_visualizable_layers()` before generating
5. **Gradio interface for exploration** - Best way to browse and rate visualizations interactively
6. **Wrapping transforms are optional** - Set `enable_wrapping=False` if not needed
7. **Generation is slow** - Each visualization takes 10-60 seconds depending on parameters
