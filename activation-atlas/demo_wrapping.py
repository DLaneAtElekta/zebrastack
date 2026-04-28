"""
Demo script showing how to create horizontally wrapping visualizations with Lucent.

This demonstrates different approaches to create wrapping/tiling effects:
1. Edge blending for seamless wrapping
2. Tile repetition for pattern effects
3. Custom parameter initialization for tiled patterns
"""

from lucent.optvis import render, param, objectives
from lucent.modelzoo import inceptionv1
from wrapping_transforms import (
    create_wrapping_visualization,
    create_tiled_visualization,
    wrap_transform,
    tile_transform,
    create_tiled_parameter
)


def demo_wrapping_effects():
    """Demonstrate different wrapping visualization techniques."""
    print("🎨 Horizontal Wrapping Demo for Lucent")
    print("=" * 50)
    
    # Load model
    print("Loading Inception V1 model...")
    model = inceptionv1(pretrained=True)
    model.eval()
    
    # Get a good layer for visualization
    test_layer = 'mixed4a_1x1_pre_relu_conv'  # Known good layer
    test_channel = 42
    
    print(f"Using layer: {test_layer}, channel: {test_channel}")
    
    # Create objective
    obj = objectives.channel(test_layer, test_channel)
    
    print("\n1. Standard Visualization (no wrapping)")
    print("-" * 30)
    
    # Standard visualization for comparison
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(256),
        show_image=False,
        save_image=True,
        image_name="demo_standard.png"
    )
    print("✅ Saved: demo_standard.png")
    
    print("\n2. Edge Blending Wrapping")
    print("-" * 30)
    
    # Wrapping with edge blending
    _ = create_wrapping_visualization(
        model, obj, image_size=256, wrap_factor=0.4
    )
    print("✅ Created wrapping visualization")
    
    print("\n3. Horizontal Tiling")
    print("-" * 30)
    
    # Tiled visualization
    _ = create_tiled_visualization(
        model, obj, image_size=256, tile_count=4
    )
    print("✅ Created tiled visualization")
    
    print("\n4. Manual Transform Setup")
    print("-" * 30)
    
    # Manual setup with custom transforms
    manual_transforms = [
        wrap_transform(wrap_factor=0.6),
        tile_transform(tile_factor=3),
    ]
    
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(256),
        transforms=manual_transforms,
        show_image=False,
        save_image=True,
        image_name="demo_manual_wrapping.png"
    )
    print("✅ Saved: demo_manual_wrapping.png")
    
    print("\n5. Tiled Parameter Initialization")
    print("-" * 30)
    
    # Start with tiled pattern
    tiled_param_func = create_tiled_parameter(size=256, tile_count=3)
    
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=tiled_param_func,
        show_image=False,
        save_image=True,
        image_name="demo_tiled_param.png"
    )
    print("✅ Saved: demo_tiled_param.png")
    
    print("\n6. Combined Effects")
    print("-" * 30)
    
    # Combine multiple wrapping techniques
    combined_transforms = [
        param.jitter(4),
        wrap_transform(wrap_factor=0.3),
        param.jitter(2),
    ]
    
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=create_tiled_parameter(size=256, tile_count=2),
        transforms=combined_transforms,
        show_image=False,
        save_image=True,
        image_name="demo_combined_wrapping.png"
    )
    print("✅ Saved: demo_combined_wrapping.png")
    
    print("\n🎉 Demo complete!")
    print("\nGenerated files:")
    print("- demo_standard.png (baseline)")
    print("- demo_manual_wrapping.png (edge blending + tiling)")
    print("- demo_tiled_param.png (tiled initialization)")
    print("- demo_combined_wrapping.png (combined effects)")
    
    print("\nTips for wrapping visualizations:")
    print("• wrap_factor controls edge blending strength (0.0-1.0)")
    print("• tile_count determines number of horizontal repetitions")
    print("• Combine with jitter and other transforms for best results")
    print("• Lower wrap_factor (0.2-0.4) often looks more natural")
    print("• Higher tile_count (3-5) creates more obvious patterns")


def test_different_objectives():
    """Test wrapping with different objective types."""
    print("\n🔬 Testing Wrapping with Different Objectives")
    print("=" * 50)
    
    model = inceptionv1(pretrained=True)
    model.eval()
    
    test_layer = 'mixed4a_1x1_pre_relu_conv'
    
    # Test different objective types
    objectives_to_test = [
        ("channel", objectives.channel(test_layer, 42)),
        ("neuron", objectives.neuron(test_layer, 42, (5, 5))),
    ]
    
    for obj_name, obj in objectives_to_test:
        print(f"\nTesting {obj_name} objective...")
        
        try:
            _ = create_wrapping_visualization(
                model, obj, image_size=128, wrap_factor=0.3
            )
            print(f"✅ {obj_name} wrapping successful")
            
            # Save with descriptive name
            render.render_vis(
                model,
                objective_f=obj,
                param_f=lambda: param.image(128),
                transforms=[wrap_transform(0.3)],
                show_image=False,
                save_image=True,
                image_name=f"demo_wrap_{obj_name}.png"
            )
            print(f"✅ Saved: demo_wrap_{obj_name}.png")
            
        except Exception as e:
            print(f"❌ {obj_name} failed: {e}")


if __name__ == "__main__":
    # Run the main demo
    demo_wrapping_effects()
    
    # Test with different objectives
    test_different_objectives()
    
    print("\n" + "="*50)
    print("🌟 All wrapping demos complete!")
    print("Check the generated PNG files to see the effects.")