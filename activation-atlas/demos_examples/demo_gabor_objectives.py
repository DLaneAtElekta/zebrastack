# Example script demonstrating Gabor-weighted objectives
import random
from gabor_objectives import (
    create_gabor_weighted_objective,
    create_gabor_preset_objectives, 
    create_multi_orientation_gabor_objective,
    visualize_gabor_weights
)
from lucent_layer_utils import get_visualizable_layers
from lucent.modelzoo import inceptionv1
import numpy as np

def demo_gabor_objectives():
    """Demonstrate different types of Gabor objectives."""
    
    print("=== Gabor Objectives Demo ===")
    
    # Load model
    model = inceptionv1(pretrained=True)
    layers = get_visualizable_layers(model)
    
    demo_layer = random.choice(layers)
    demo_channel = random.randint(0, 50)
    
    print(f"Using layer: {demo_layer}, channel: {demo_channel}")
    
    # 1. Basic Gabor edge detector
    print("\n1. Creating horizontal edge detector...")
    edge_obj = create_gabor_weighted_objective(
        demo_layer, demo_channel,
        size=7,
        with_offset=(0, 0),
        sigma=(1.5, 1.5),
        theta=0,  # horizontal edges
        lambda_freq=3.0,
        spatial_weight=1.0
    )
    
    # 2. Preset objectives
    print("\n2. Creating preset objectives...")
    presets = {
        'edge': create_gabor_preset_objectives(demo_layer, demo_channel, 'edge_detector'),
        'vertical': create_gabor_preset_objectives(demo_layer, demo_channel, 'vertical_edges'), 
        'diagonal': create_gabor_preset_objectives(demo_layer, demo_channel, 'diagonal_edges'),
        'texture': create_gabor_preset_objectives(demo_layer, demo_channel, 'texture_fine'),
        'blob': create_gabor_preset_objectives(demo_layer, demo_channel, 'blob_detector')
    }
    
    for name, obj in presets.items():
        print(f"  {name}: {'✅' if obj else '❌'}")
    
    # 3. Multi-orientation objective
    print("\n3. Creating multi-orientation objective...")
    multi_obj = create_multi_orientation_gabor_objective(
        demo_layer, demo_channel,
        orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        weights=[1.0, 0.8, 1.0, 0.8],  # Slightly emphasize cardinal directions
        size=7,
        sigma_x=1.5, sigma_y=1.5,
        lambda_freq=3.0
    )
    
    print(f"Multi-orientation objective: {'✅' if multi_obj else '❌'}")
    
    # 4. Custom Gabor with offsets
    print("\n4. Creating offset Gabor objectives...")
    offset_objectives = []
    
    for offset_name, (x_off, y_off) in [('center', (0, 0)), ('right', (5, 0)), ('top', (0, -5))]:
        obj = create_gabor_weighted_objective(
            demo_layer, demo_channel,
            size=7,
            with_offset=(x_off, y_off),
            theta=np.pi/4,  # diagonal edges
            lambda_freq=2.5,
            sigma=(1.2, 1.2)
        )
        offset_objectives.append((offset_name, obj))
        print(f"  {offset_name} offset ({x_off}, {y_off}): {'✅' if obj else '❌'}")
    
    # 5. Visualize some weight patterns
    print("\n5. Weight pattern analysis...")
    patterns = {
        'horizontal_edges': {'theta': 0, 'lambda_freq': 3.0},
        'vertical_edges': {'theta': np.pi/2, 'lambda_freq': 3.0},
        'fine_texture': {'theta': 0, 'lambda_freq': 1.5, 'sigma_x': 0.8, 'sigma_y': 0.8},
        'coarse_texture': {'theta': 0, 'lambda_freq': 4.0, 'sigma_x': 2.0, 'sigma_y': 2.0}
    }
    
    for pattern_name, params in patterns.items():
        weights = visualize_gabor_weights(size=7, **params)
        print(f"  {pattern_name}: range [{weights.min():.3f}, {weights.max():.3f}], std {weights.std():.3f}")
    
    print("\n=== Demo Complete ===")
    
    return {
        'basic_edge': edge_obj,
        'presets': presets,
        'multi_orientation': multi_obj,
        'offset_objectives': offset_objectives
    }

def compare_gabor_vs_regular():
    """Compare Gabor objectives with regular center objectives."""
    
    print("\n=== Gabor vs Regular Objectives Comparison ===")
    
    from spatial_objectives import create_center_nxn_objective
    
    model = inceptionv1(pretrained=True)
    layers = get_visualizable_layers(model)
    
    test_layer = random.choice(layers)
    test_channel = 42
    
    print(f"Comparing objectives for {test_layer}:{test_channel}")
    
    # Regular 7x7 center objective
    regular_obj = create_center_nxn_objective(test_layer, test_channel, size=7)
    
    # Gabor edge detector (7x7)
    gabor_edge = create_gabor_preset_objectives(test_layer, test_channel, 'edge_detector')
    
    # Gabor blob detector (7x7)  
    gabor_blob = create_gabor_preset_objectives(test_layer, test_channel, 'blob_detector')
    
    print(f"Regular 7x7 center: {'✅' if regular_obj else '❌'}")
    print(f"Gabor edge detector: {'✅' if gabor_edge else '❌'}")
    print(f"Gabor blob detector: {'✅' if gabor_blob else '❌'}")
    
    return {
        'regular': regular_obj,
        'gabor_edge': gabor_edge,
        'gabor_blob': gabor_blob
    }

def generate_gabor_visualization_batch(layers, num_examples=3):
    """Generate a batch of visualizations using different Gabor objectives."""
    
    print(f"\n=== Generating {num_examples} Gabor Visualizations ===")
    
    objectives = []
    
    for i in range(num_examples):
        layer = random.choice(layers)
        channel = random.randint(0, 100)
        
        # Random Gabor parameters
        theta = random.uniform(0, np.pi)
        lambda_freq = random.uniform(1.5, 4.0)
        sigma = random.uniform(1.0, 2.0)
        x_offset = random.randint(-3, 3)
        y_offset = random.randint(-3, 3)
        
        obj = create_gabor_weighted_objective(
            layer, channel,
            size=7,
            with_offset=(x_offset, y_offset),
            sigma=(sigma, sigma),
            theta=theta,
            lambda_freq=lambda_freq
        )
        
        if obj:
            objectives.append({
                'objective': obj,
                'params': {
                    'layer': layer,
                    'channel': channel,
                    'theta': theta,
                    'lambda_freq': lambda_freq,
                    'sigma': sigma,
                    'offset': (x_offset, y_offset)
                }
            })
            
            print(f"  {i+1}. {layer}:{channel} θ={theta:.2f} λ={lambda_freq:.1f} σ={sigma:.1f} offset={x_offset},{y_offset}")
    
    print(f"Generated {len(objectives)} Gabor objectives")
    return objectives

if __name__ == "__main__":
    # Run demos
    demo_results = demo_gabor_objectives()
    comparison_results = compare_gabor_vs_regular()
    
    # Load model for batch generation
    model = inceptionv1(pretrained=True)
    layers = get_visualizable_layers(model)
    gabor_batch = generate_gabor_visualization_batch(layers, 5)
    
    print("\n" + "="*60)
    print("🎉 All demos completed successfully!")
    print("\nTo use these objectives with render.render_vis():")
    print("obj = create_gabor_preset_objectives('mixed4a', 42, 'edge_detector')")
    print("render.render_vis(model, obj, param_f=lambda: param.image(256), show_inline=True)")
    print("\nOr with custom parameters:")
    print("obj = create_gabor_weighted_objective(")
    print("    'mixed4a', 42, size=7, theta=np.pi/4, lambda_freq=2.5)")
    print("render.render_vis(model, obj, show_inline=True)")