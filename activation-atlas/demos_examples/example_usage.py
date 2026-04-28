"""
Example usage of the Lucent layer utilities.

This script demonstrates how to use the layer analysis functions
to find and visualize features in different PyTorch models.
"""

from lucent_layer_utils import (
    get_visualizable_layers,
    categorize_visualizable_layers,
    get_recommended_layers_for_visualization,
    print_model_summary,
)
from spatial_objectives import create_random_objective
from lucent.modelzoo import inceptionv1, resnet152
from lucent.optvis import render, param


def demo_inception_v1():
    """Demo with Inception V1"""
    print("=== Inception V1 Demo ===")
    model = inceptionv1(pretrained=True)

    # Get all visualizable layers
    layers = get_visualizable_layers(model)
    print(f"Found {len(layers)} visualizable layers")

    # Get recommendations
    recommended = get_recommended_layers_for_visualization(model)
    print("Recommended layers:")
    for category, layer_list in recommended.items():
        print(f"  {category}: {layer_list[:3]}")

    # Create and visualize a random objective
    print("\nCreating random visualization...")
    obj = create_random_objective(model, layers)
    if obj:
        try:
            # Visualize (this will take some time)
            render.render_vis(
                model, obj, param_f=lambda: param.image(224), show_inline=True
            )
        except Exception as e:
            print(f"Visualization failed: {e}")

    print()


def demo_resnet152():
    """Demo with ResNet152"""
    print("=== ResNet152 Demo ===")
    model = resnet152(pretrained=True)

    # Print comprehensive model summary
    print_model_summary(model)

    # Get categorized layers
    categories = categorize_visualizable_layers(model)
    print("Layer distribution:")
    for category, layers in categories.items():
        print(f"  {category}: {len(layers)} layers")

    print()


def interactive_layer_explorer():
    """Interactive layer exploration"""
    print("=== Interactive Layer Explorer ===")

    # Let user choose model
    print("Choose a model:")
    print("1. Inception V1")
    print("2. ResNet152")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        model = inceptionv1(pretrained=True)
        model_name = "Inception V1"
    elif choice == "2":
        model = resnet152(pretrained=True)
        model_name = "ResNet152"
    else:
        print("Invalid choice")
        return

    print(f"\nLoaded {model_name}")

    # Get layers
    layers = get_visualizable_layers(model)
    print(f"Found {len(layers)} visualizable layers")

    # Show first 20 layers
    print("\nFirst 20 visualizable layers:")
    for i, layer in enumerate(layers[:20], 1):
        print(f"  {i:2}. {layer}")

    # Let user pick a layer to visualize
    layer_choice = input(
        f"\nEnter layer number to visualize (1-{min(20, len(layers))}): "
    ).strip()

    try:
        layer_idx = int(layer_choice) - 1
        if 0 <= layer_idx < min(20, len(layers)):
            selected_layer = layers[layer_idx]
            print(f"Selected: {selected_layer}")

            # Create objective for this layer
            from lucent_layer_utils import get_channels_from_lucent_name
            from lucent.optvis import objectives

            num_channels = get_channels_from_lucent_name(model, selected_layer)
            if num_channels:
                channel = min(10, num_channels // 2)  # Pick middle channel
                obj = objectives.channel(selected_layer, channel)

                print(
                    f"Creating visualization for {selected_layer}, channel {channel}..."
                )
                render.render_vis(
                    model, obj, param_f=lambda: param.image(224), show_inline=True
                )
            else:
                print("Could not determine channels for this layer")
        else:
            print("Invalid layer number")
    except ValueError:
        print("Please enter a valid number")


if __name__ == "__main__":
    print("🎨 Lucent Layer Utilities Examples")
    print("=" * 40)

    # Run demos
    demo_inception_v1()
    demo_resnet152()

    # Uncomment the following line for interactive mode
    # interactive_layer_explorer()

    print("✅ Demo complete!")
