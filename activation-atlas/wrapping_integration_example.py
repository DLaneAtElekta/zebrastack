"""
Example integration of wrapping transforms into your existing visualization workflow.

This shows how to add wrapping effects to your generate_random_objective_visualizatio.py
"""

import random
from wrapping_transforms import wrap_transform, tile_transform, create_tiled_parameter


def enhanced_visualize_to_file(
    model,
    use_objective,
    sampled_channels,
    layer_name,
    obj,
    with_transforms,
    transforms_label,
    transform_details,
    enable_wrapping=True,
    wrap_factor=None,
    tile_count=None,
):
    """Enhanced version of your visualize_to_file function with wrapping support."""
    from generate_random_objective_visualizatio import (
        generate_filename, 
        log_visualization_params,
        GENERATED_IMAGE_SIZE
    )
    from lucent.optvis import render, param
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Generate filename (potentially with wrapping info)
    wrap_suffix = ""
    if enable_wrapping and wrap_factor:
        wrap_suffix = f"_wrap{int(wrap_factor*100)}"
    if enable_wrapping and tile_count:
        wrap_suffix += f"_tile{tile_count}"
    
    filename = generate_filename(
        use_objective + wrap_suffix, 
        sampled_channels, 
        transforms_label, 
        layer_name
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info("Generating %s visualization: %s", transforms_label, filename)
    
    # Prepare transforms with optional wrapping
    final_transforms = with_transforms.copy() if with_transforms else []
    
    def param_func():
        return param.image(GENERATED_IMAGE_SIZE)
    
    if enable_wrapping:
        if wrap_factor and wrap_factor > 0:
            final_transforms.append(wrap_transform(wrap_factor))
            
        if tile_count and tile_count > 1:
            final_transforms.append(tile_transform(tile_count))
            # Optionally start with tiled parameter
            param_func = create_tiled_parameter(GENERATED_IMAGE_SIZE, tile_count)
    
    # Render with potential wrapping
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=param_func,
        transforms=final_transforms if final_transforms else None,
        show_image=False,
        save_image=True,
        image_name=filename,
    )
    
    # Update transform details to include wrapping info
    enhanced_details = transform_details
    if enable_wrapping and wrap_factor:
        enhanced_details += f", wrap({wrap_factor})"
    if enable_wrapping and tile_count:
        enhanced_details += f", tiles({tile_count})"
    
    log_visualization_params(
        use_objective,
        sampled_channels,
        enhanced_details,
        filename,
        transforms_label,
    )


def enhanced_generate_for_model(model, layers):
    """Enhanced version with random wrapping effects."""
    from lucent.optvis import transform
    
    # Your existing code...
    use_objective = random.choice(
        ["channel", "neuron", "center_3x3", "center_5x5", "center_7x7"]
    )
    sampled_channels = random.randint(1, 8)
    
    # Create objective (your existing code)
    from spatial_objectives import create_random_objective
    obj = create_random_objective(
        model, layers, objective_types=[use_objective], sampled_channels=sampled_channels
    )
    
    # Randomly decide on wrapping effects
    enable_wrapping = random.choice([True, False])
    wrap_factor = random.uniform(0.2, 0.6) if enable_wrapping else None
    tile_count = random.randint(2, 4) if enable_wrapping and random.choice([True, False]) else None
    
    # Your existing transforms
    all_transforms = [
        transform.pad(16),
        transform.jitter(8),
        transform.random_scale([n / 100.0 for n in range(80, 120)]),
        transform.random_rotate(
            list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))
        ),
        transform.jitter(2),
    ]
    
    # Generate with potential wrapping
    enhanced_visualize_to_file(
        model,
        use_objective,
        sampled_channels,
        random.choice(layers),
        obj,
        all_transforms,
        "full_transforms_with_wrapping",
        "full transforms + wrapping",
        enable_wrapping=enable_wrapping,
        wrap_factor=wrap_factor,
        tile_count=tile_count,
    )


if __name__ == "__main__":
    print("Example of integrating wrapping into your workflow")
    print("Copy the enhanced functions to your main script!")