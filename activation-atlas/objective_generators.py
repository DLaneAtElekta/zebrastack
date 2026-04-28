"""
Objective parameter generators for creating random visualization parameters.

This module provides generator functions for all objective types:
- Channel objectives: Random channel selection
- Neuron objectives: Random channel + spatial coordinates
- Center objectives: Random channel + position + size
- Gabor objectives: Random channel + position + Gabor parameters

Each generator returns a dictionary of parameters that can be used to create
objectives and stored for later analysis/mutation.
"""

import random
import numpy as np
from typing import Dict, Any, List, Optional
from lucent_layer_utils import get_layer_dimensions, get_channels_from_lucent_name


def generate_channel_objective_params(
    model, 
    layers: List[str], 
    layer_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate random parameters for a channel objective.
    
    Args:
        model: PyTorch model
        layers: List of available layer names
        layer_name: Specific layer to use (optional, random if None)
        
    Returns:
        Dictionary with channel objective parameters
    """
    if layer_name is None:
        layer_name = random.choice(layers)
    
    total_channels = get_channels_from_lucent_name(model, layer_name)
    if total_channels is None or total_channels <= 0:
        total_channels = 64  # fallback
    
    channel_idx = random.randint(0, total_channels - 1)
    
    return {
        "objective_type": "channel",
        "layer_name": layer_name,
        "channel_idx": channel_idx,
        "total_channels": total_channels,
        "spatial_params": None,
        "gabor_params": None,
    }


def generate_neuron_objective_params(
    model, 
    layers: List[str], 
    layer_name: Optional[str] = None,
    input_size: int = 384
) -> Dict[str, Any]:
    """
    Generate random parameters for a neuron objective.
    
    Args:
        model: PyTorch model
        layers: List of available layer names
        layer_name: Specific layer to use (optional, random if None)
        input_size: Input image size for dimension calculation
        
    Returns:
        Dictionary with neuron objective parameters
    """
    if layer_name is None:
        layer_name = random.choice(layers)
    
    total_channels = get_channels_from_lucent_name(model, layer_name)
    if total_channels is None or total_channels <= 0:
        total_channels = 64  # fallback
    
    channel_idx = random.randint(0, total_channels - 1)
    
    # Get spatial dimensions
    dims = get_layer_dimensions(model, layer_name, input_size)
    if dims:
        height, width, _ = dims
    else:
        height, width = 7, 7  # fallback for small layers
    
    # Random spatial coordinates within layer
    x = random.randint(0, width - 1) if width > 1 else 0
    y = random.randint(0, height - 1) if height > 1 else 0
    
    return {
        "objective_type": "neuron",
        "layer_name": layer_name,
        "channel_idx": channel_idx,
        "total_channels": total_channels,
        "spatial_params": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "gabor_params": None,
    }


def generate_center_objective_params(
    model, 
    layers: List[str], 
    layer_name: Optional[str] = None,
    input_size: int = 384
) -> Dict[str, Any]:
    """
    Generate random parameters for a center objective.
    
    Args:
        model: PyTorch model
        layers: List of available layer names
        layer_name: Specific layer to use (optional, random if None)
        input_size: Input image size for dimension calculation
        
    Returns:
        Dictionary with center objective parameters
    """
    if layer_name is None:
        layer_name = random.choice(layers)
    
    total_channels = get_channels_from_lucent_name(model, layer_name)
    if total_channels is None or total_channels <= 0:
        total_channels = 64  # fallback
    
    channel_idx = random.randint(0, total_channels - 1)
    
    # Get spatial dimensions
    dims = get_layer_dimensions(model, layer_name, input_size)
    if dims:
        height, width, _ = dims
    else:
        height, width = 7, 7  # fallback
    
    # Random center size (3x3, 5x5, 7x7, etc.)
    size_options = [3, 5, 7, 9, 11]
    # Filter sizes that fit in the layer
    valid_sizes = [s for s in size_options if s <= min(height, width)]
    if not valid_sizes:
        valid_sizes = [3]  # minimum fallback
    
    center_size = random.choice(valid_sizes)
    
    # Random offset within reasonable bounds
    max_offset_x = max(0, (width - center_size) // 2)
    max_offset_y = max(0, (height - center_size) // 2)
    
    offset_x = random.randint(-max_offset_x, max_offset_x) if max_offset_x > 0 else 0
    offset_y = random.randint(-max_offset_y, max_offset_y) if max_offset_y > 0 else 0
    
    return {
        "objective_type": f"center_{center_size}x{center_size}",
        "layer_name": layer_name,
        "channel_idx": channel_idx,
        "total_channels": total_channels,
        "spatial_params": {
            "center_size": center_size,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "width": width,
            "height": height,
        },
        "gabor_params": None,
    }


def generate_gabor_objective_params(
    model, 
    layers: List[str], 
    layer_name: Optional[str] = None,
    input_size: int = 384
) -> Dict[str, Any]:
    """
    Generate random parameters for a Gabor objective.
    
    Args:
        model: PyTorch model
        layers: List of available layer names
        layer_name: Specific layer to use (optional, random if None)
        input_size: Input image size for dimension calculation
        
    Returns:
        Dictionary with Gabor objective parameters
    """
    if layer_name is None:
        layer_name = random.choice(layers)
    
    total_channels = get_channels_from_lucent_name(model, layer_name)
    if total_channels is None or total_channels <= 0:
        total_channels = 64  # fallback
    
    # Gabor can use float channel indices for interpolation
    channel_idx = random.uniform(0, total_channels - 1)
    
    # Get spatial dimensions
    dims = get_layer_dimensions(model, layer_name, input_size)
    if dims:
        height, width, _ = dims
    else:
        height, width = 7, 7  # fallback
    
    # Random spatial coordinates
    x = random.uniform(0, width - 1) if width > 1 else 0
    y = random.uniform(0, height - 1) if height > 1 else 0
    
    # Random Gabor parameters
    sigma = random.uniform(0.5, 3.0)  # Standard deviation
    lambda_freq = random.uniform(1.0, 8.0)  # Wavelength
    theta = random.uniform(0, np.pi)  # Orientation
    psi = random.uniform(0, 2 * np.pi)  # Phase
    gamma = random.uniform(0.3, 1.5)  # Aspect ratio
    
    # Random size for Gabor filter
    size_options = [3, 5, 7, 9]
    valid_sizes = [s for s in size_options if s <= min(height, width)]
    if not valid_sizes:
        valid_sizes = [3]
    size = random.choice(valid_sizes)
    
    return {
        "objective_type": "gabor",
        "layer_name": layer_name,
        "channel_idx": channel_idx,
        "total_channels": total_channels,
        "spatial_params": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "gabor_params": {
            "sigma": sigma,
            "lambda_freq": lambda_freq,
            "theta": theta,
            "psi": psi,
            "gamma": gamma,
            "size": size,
        },
    }


def generate_random_objective_params(
    model, 
    layers: List[str],
    objective_types: Optional[List[str]] = None,
    num_objectives: int = 1,
    input_size: int = 384
) -> List[Dict[str, Any]]:
    """
    Generate random parameters for multiple objectives.
    
    Args:
        model: PyTorch model
        layers: List of available layer names
        objective_types: List of objective types to choose from
        num_objectives: Number of objectives to generate
        input_size: Input image size for dimension calculation
        
    Returns:
        List of objective parameter dictionaries
    """
    if objective_types is None:
        objective_types = ["channel", "neuron", "center", "gabor"]
    
    generators = {
        "channel": generate_channel_objective_params,
        "neuron": generate_neuron_objective_params,
        "center": generate_center_objective_params,
        "gabor": generate_gabor_objective_params,
    }
    
    objectives = []
    for _ in range(num_objectives):
        obj_type = random.choice(objective_types)
        
        # Handle center variants
        if obj_type.startswith("center"):
            generator = generators["center"]
        else:
            generator = generators.get(obj_type, generators["channel"])
        
        try:
            params = generator(model, layers, input_size=input_size)
            objectives.append(params)
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            print(f"Warning: Failed to generate {obj_type} objective: {e}")
            # Fallback to channel objective
            fallback_params = generate_channel_objective_params(model, layers)
            objectives.append(fallback_params)
    
    return objectives


def create_objective_from_params(params: Dict[str, Any]):
    """
    Create a Lucent objective from parameter dictionary.
    
    Args:
        params: Parameter dictionary from generator functions
        
    Returns:
        Lucent objective function
    """
    from lucent.optvis import objectives
    from spatial_objectives import create_center_nxn_objective
    from gabor_objectives import create_gabor_weighted_objective
    
    obj_type = params["objective_type"]
    layer_name = params["layer_name"]
    channel_idx = params["channel_idx"]
    
    if obj_type == "channel":
        return objectives.channel(layer_name, int(channel_idx))
    
    elif obj_type == "neuron":
        spatial = params["spatial_params"]
        return objectives.neuron(layer_name, int(channel_idx), (spatial["x"], spatial["y"]))
    
    elif obj_type.startswith("center"):
        spatial = params["spatial_params"]
        return create_center_nxn_objective(
            layer_name, 
            int(channel_idx), 
            size=spatial["center_size"],
            with_offset=(spatial["offset_x"], spatial["offset_y"])
        )
    
    elif obj_type == "gabor":
        spatial = params["spatial_params"]
        gabor = params["gabor_params"]
        return create_gabor_weighted_objective(
            layer_name,
            channel_idx,  # Keep as float for interpolation
            size=gabor["size"],
            sigma=gabor["sigma"],
            lambda_freq=gabor["lambda_freq"],
            theta=gabor["theta"],
            psi=gabor["psi"],
            gamma=gabor["gamma"]
        )
    
    else:
        # Fallback to channel objective
        return objectives.channel(layer_name, int(channel_idx))


def mutate_objective_params(
    params: Dict[str, Any], 
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.2
) -> Dict[str, Any]:
    """
    Create a mutated version of objective parameters.
    
    Args:
        params: Original parameter dictionary
        mutation_rate: Probability of mutating each parameter
        mutation_strength: How much to vary parameters (0.0-1.0)
        
    Returns:
        Mutated parameter dictionary
    """
    mutated = params.copy()
    
    obj_type = params["objective_type"]
    
    # Mutate channel index
    if random.random() < mutation_rate:
        total_channels = params["total_channels"]
        current_channel = params["channel_idx"]
        
        if obj_type == "gabor":
            # For Gabor, keep as float and vary within range
            variation = mutation_strength * total_channels * 0.1
            new_channel = current_channel + random.uniform(-variation, variation)
            mutated["channel_idx"] = max(0, min(total_channels - 1, new_channel))
        else:
            # For others, vary integer channel
            variation = max(1, int(mutation_strength * total_channels * 0.1))
            delta = random.randint(-variation, variation)
            new_channel = current_channel + delta
            mutated["channel_idx"] = max(0, min(total_channels - 1, new_channel))
    
    # Mutate spatial parameters
    if params["spatial_params"] and random.random() < mutation_rate:
        spatial = mutated["spatial_params"].copy()
        
        if "x" in spatial and "y" in spatial:
            # Mutate position
            width, height = spatial["width"], spatial["height"]
            x_variation = mutation_strength * width * 0.2
            y_variation = mutation_strength * height * 0.2
            
            spatial["x"] = max(0, min(width - 1, 
                spatial["x"] + random.uniform(-x_variation, x_variation)))
            spatial["y"] = max(0, min(height - 1, 
                spatial["y"] + random.uniform(-y_variation, y_variation)))
        
        if "offset_x" in spatial and "offset_y" in spatial:
            # Mutate center offsets
            max_offset = min(spatial["width"], spatial["height"]) // 4
            offset_variation = mutation_strength * max_offset
            
            spatial["offset_x"] += random.uniform(-offset_variation, offset_variation)
            spatial["offset_y"] += random.uniform(-offset_variation, offset_variation)
            spatial["offset_x"] = max(-max_offset, min(max_offset, spatial["offset_x"]))
            spatial["offset_y"] = max(-max_offset, min(max_offset, spatial["offset_y"]))
        
        mutated["spatial_params"] = spatial
    
    # Mutate Gabor parameters
    if params["gabor_params"] and random.random() < mutation_rate:
        gabor = mutated["gabor_params"].copy()
        
        # Mutate each Gabor parameter with appropriate ranges
        if random.random() < mutation_rate:
            gabor["sigma"] = max(0.1, min(5.0, 
                gabor["sigma"] + random.uniform(-0.5, 0.5) * mutation_strength))
        
        if random.random() < mutation_rate:
            gabor["lambda_freq"] = max(0.5, min(10.0, 
                gabor["lambda_freq"] + random.uniform(-2.0, 2.0) * mutation_strength))
        
        if random.random() < mutation_rate:
            gabor["theta"] = (gabor["theta"] + 
                random.uniform(-np.pi/4, np.pi/4) * mutation_strength) % np.pi
        
        if random.random() < mutation_rate:
            gabor["psi"] = (gabor["psi"] + 
                random.uniform(-np.pi, np.pi) * mutation_strength) % (2 * np.pi)
        
        if random.random() < mutation_rate:
            gabor["gamma"] = max(0.1, min(2.0, 
                gabor["gamma"] + random.uniform(-0.3, 0.3) * mutation_strength))
        
        mutated["gabor_params"] = gabor
    
    return mutated


if __name__ == "__main__":
    print("🎯 Objective Parameter Generators")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- generate_channel_objective_params()")
    print("- generate_neuron_objective_params()")
    print("- generate_center_objective_params()")
    print("- generate_gabor_objective_params()")
    print("- generate_random_objective_params()")
    print("- create_objective_from_params()")
    print("- mutate_objective_params()")
    print()
    print("Example usage:")
    print("params = generate_random_objective_params(model, layers, num_objectives=3)")
    print("obj = create_objective_from_params(params[0])")
    print("mutated = mutate_objective_params(params[0], mutation_rate=0.5)")