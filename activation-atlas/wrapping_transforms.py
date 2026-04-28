"""
Custom transforms for creating horizontally wrapping visualizations with Lucent.

This module provides transforms that can create tiling/wrapping effects
for use with render.render_vis().
"""

import torch
import torch.nn.functional as F
from lucent.optvis.param import image


def horizontal_wrap_transform(img, wrap_factor=0.3):
    """
    Transform that creates horizontal wrapping by blending left and right edges.
    
    Args:
        img: Input image tensor [batch, channels, height, width]
        wrap_factor: How much to blend edges (0.0 = no wrap, 1.0 = full wrap)
    
    Returns:
        Modified image with horizontal wrapping effect
    """
    if not isinstance(img, torch.Tensor):
        return img
    
    # Ensure we have the right dimensions
    if len(img.shape) != 4:
        return img
    
    batch, channels, height, width = img.shape
    
    # Create a copy to modify
    wrapped_img = img.clone()
    
    # Define edge width for blending with random variation (1% - 25% of width)
    import random
    overlap_percentage = random.uniform(0.01, 0.25)  # Random between 1% and 25%
    edge_width = max(1, int(width * overlap_percentage))
    
    # Blend left edge with right edge content
    left_edge = wrapped_img[:, :, :, :edge_width]
    right_edge = wrapped_img[:, :, :, -edge_width:]
    
    # Create blending weights
    alpha = torch.linspace(wrap_factor, 0, edge_width).view(1, 1, 1, -1).to(img.device)
    
    # Apply wrapping blend
    wrapped_img[:, :, :, :edge_width] = (1 - alpha) * left_edge + alpha * right_edge
    wrapped_img[:, :, :, -edge_width:] = alpha.flip(-1) * left_edge + (1 - alpha.flip(-1)) * right_edge
    
    return wrapped_img


def create_tiled_parameter(size=224, tile_count=3):
    """
    Create a parameter function that generates tiled initial images.
    
    Args:
        size: Image size
        tile_count: Number of horizontal tiles
    
    Returns:
        Parameter function for use with render_vis
    """
    def tiled_param():
        # Create base image
        base_img = image(size)
        
        # Get tile width
        tile_width = size // tile_count
        
        # Create tiled version by repeating sections
        tiled = torch.zeros_like(base_img)
        for i in range(tile_count):
            start_x = i * tile_width
            end_x = min((i + 1) * tile_width, size)
            
            # Use the first tile pattern for all tiles
            if i == 0:
                source_tile = base_img[:, :, :, start_x:end_x]
            
            tiled[:, :, :, start_x:end_x] = source_tile[:, :, :, :end_x-start_x]
        
        return tiled
    
    return tiled_param


def horizontal_tile_transform(img, tile_factor=3):
    """
    Transform that creates horizontal tiling effect during optimization.
    
    Args:
        img: Input image tensor
        tile_factor: Number of horizontal tiles
    
    Returns:
        Tiled image
    """
    if not isinstance(img, torch.Tensor) or len(img.shape) != 4:
        return img
    
    batch, channels, height, width = img.shape
    tile_width = width // tile_factor
    
    if tile_width < 1:
        return img
    
    # Create tiled version
    tiled_img = img.clone()
    
    # Take the leftmost tile and repeat it
    source_tile = img[:, :, :, :tile_width]
    
    for i in range(1, tile_factor):
        start_x = i * tile_width
        end_x = min((i + 1) * tile_width, width)
        if end_x > start_x:
            tiled_img[:, :, :, start_x:end_x] = source_tile[:, :, :, :end_x-start_x]
    
    return tiled_img


# Transform functions for use in render_vis transforms list
def wrap_transform(wrap_factor=0.3):
    """Create a transform function for horizontal wrapping."""
    return lambda img: horizontal_wrap_transform(img, wrap_factor)


def tile_transform(tile_factor=3):
    """Create a transform function for horizontal tiling."""
    return lambda img: horizontal_tile_transform(img, tile_factor)


# Example usage functions
def create_wrapping_visualization(model, objective, image_size=224, wrap_factor=0.3):
    """
    Create a visualization with horizontal wrapping effect.
    
    Args:
        model: PyTorch model
        objective: Lucent objective
        image_size: Size of generated image
        wrap_factor: Strength of wrapping effect (0.0-1.0)
    
    Returns:
        Generated image with wrapping effect
    """
    from lucent.optvis import render, param
    
    # Create transforms with wrapping
    transforms = [
        wrap_transform(wrap_factor),
    ]
    
    return render.render_vis(
        model,
        objective_f=objective,
        param_f=lambda: param.image(image_size),
        transforms=transforms,
        show_image=False,
    )


def create_tiled_visualization(model, objective, image_size=224, tile_count=3):
    """
    Create a visualization with horizontal tiling effect.
    
    Args:
        model: PyTorch model
        objective: Lucent objective
        image_size: Size of generated image
        tile_count: Number of horizontal tiles
    
    Returns:
        Generated tiled image
    """
    from lucent.optvis import render
    
    # Use tiled parameter function
    param_func = create_tiled_parameter(image_size, tile_count)
    
    # Optional: add tiling transform during optimization
    transforms = [
        tile_transform(tile_count),
    ]
    
    return render.render_vis(
        model,
        objective_f=objective,
        param_f=param_func,
        transforms=transforms,
        show_image=False,
    )


if __name__ == "__main__":
    # Demo usage
    print("Wrapping Transforms for Lucent")
    print("=============================")
    print()
    print("Example usage:")
    print("from lucent.optvis import objectives")
    print("from lucent.modelzoo import inceptionv1")
    print("from wrapping_transforms import create_wrapping_visualization, create_tiled_visualization")
    print()
    print("model = inceptionv1(pretrained=True)")
    print("obj = objectives.channel('mixed4a', 42)")
    print()
    print("# Create wrapping visualization")
    print("img1 = create_wrapping_visualization(model, obj, wrap_factor=0.5)")
    print()
    print("# Create tiled visualization")
    print("img2 = create_tiled_visualization(model, obj, tile_count=4)")