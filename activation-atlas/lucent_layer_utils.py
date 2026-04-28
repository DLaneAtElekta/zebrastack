"""
Utility functions to get all visualizable layers in PyTorch models for Lucent.

This module provides comprehensive functions to:
1. Get all layers that can be visualized by Lucent
2. Categorize layers by feature complexity
3. Get recommended layers for visualization
4. Test individual layers for visualization compatibility

Works with various model architectures including:
- ResNet, ResNeXt
- Inception V1, V3
- VGG
- DenseNet
- And others supported by Lucent
"""

import random
import torch
from lucent.optvis import objectives
from lucent.modelzoo.util import get_model_layers


def get_layer_dimensions(model, layer_name, input_size=224):
    """
    Get the spatial dimensions (height, width) of a layer's output.

    Args:
        model: PyTorch model
        layer_name: Name of the layer (Lucent format with underscores)
        input_size: Input image size (default 224 for most models)

    Returns:
        tuple: (height, width, channels) or None if failed
    """
    model.eval()
    device = next(model.parameters()).device

    # Create a dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    try:
        with torch.no_grad():
            # Forward pass and capture activations
            activations = {}

            def hook_fn(name):
                def hook(_, __, output):
                    activations[name] = output

                return hook

            # Register hooks for all modules
            hooks = []
            for name, module in model.named_modules():
                # Convert PyTorch naming to Lucent naming
                lucent_name = name.replace(".", "_")
                if lucent_name == layer_name:
                    hook = module.register_forward_hook(hook_fn(lucent_name))
                    hooks.append(hook)

            # Forward pass
            _ = model(dummy_input)

            # Clean up hooks
            for hook in hooks:
                hook.remove()

            # Get dimensions
            if layer_name in activations:
                output = activations[layer_name]
                if len(output.shape) == 4:  # [batch, channels, height, width]
                    _, channels, height, width = output.shape
                    return (height, width, channels)
                else:
                    print(f"Layer {layer_name} has shape {output.shape} (not 4D)")
                    return None
            else:
                print(f"Layer {layer_name} not found in activations")
                return None

    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"Error getting dimensions for {layer_name}: {e}")
        return None


def get_channels_from_lucent_name(model, lucent_layer_name):
    """
    Get number of output channels from a Lucent-style layer name.

    This function handles different model architectures and naming conventions:
    - ResNet: layer1_0_conv1 -> layer1.0.conv1
    - Inception: mixed3a_1x1_pre_relu_conv -> mixed3a.1x1 branch
    - VGG: features_0 -> features.0
    """
    try:
        # Method 1: Handle Inception V1 style names (e.g., "mixed3a_1x1_pre_relu_conv")
        if "mixed" in lucent_layer_name:
            parts = lucent_layer_name.split("_")
            if len(parts) >= 2:
                mixed_name = parts[0]  # e.g., "mixed3a"
                if hasattr(model, mixed_name):
                    mixed_module = getattr(model, mixed_name)

                    # Look for the specific branch
                    branch_name = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                    if hasattr(mixed_module, branch_name):
                        branch_module = getattr(mixed_module, branch_name)
                        if hasattr(branch_module, "conv") and hasattr(
                            branch_module.conv, "out_channels"
                        ):
                            return branch_module.conv.out_channels
                        elif hasattr(branch_module, "out_channels"):
                            return branch_module.out_channels

        # Method 2: Handle conv2d style names (e.g., "conv2d0_pre_relu_conv")
        elif "conv2d" in lucent_layer_name:
            parts = lucent_layer_name.split("_")
            conv_name = parts[0]  # e.g., "conv2d0"
            if hasattr(model, conv_name):
                conv_module = getattr(model, conv_name)
                if hasattr(conv_module, "conv") and hasattr(
                    conv_module.conv, "out_channels"
                ):
                    return conv_module.conv.out_channels
                elif hasattr(conv_module, "out_channels"):
                    return conv_module.out_channels

        # Method 3: Handle ResNet style names (e.g., "layer1_0_conv1")
        else:
            # Convert underscore notation to dot notation for ResNet-style models
            pytorch_name = lucent_layer_name.replace("_", ".")

            module = model
            for part in pytorch_name.split("."):
                if part.isdigit():
                    # Handle numeric indices (like layer1.0)
                    module = module[int(part)]
                else:
                    # Handle named attributes
                    module = getattr(module, part)

            # Check if it's a conv layer and get output channels
            if hasattr(module, "out_channels"):
                return module.out_channels
            elif hasattr(module, "num_features"):  # BatchNorm layers
                return module.num_features

        # Method 4: Brute force search through all modules
        # If direct methods fail, search through all named modules
        for name, module in model.named_modules():
            if lucent_layer_name in name or name.endswith(
                lucent_layer_name.split("_")[-1]
            ):
                if hasattr(module, "out_channels"):
                    return module.out_channels
                elif hasattr(module, "num_features"):
                    return module.num_features

        return None

    except (AttributeError, IndexError, TypeError):
        return None


def get_visualizable_layers(model, include_detailed_info=False):
    """
    Get all layers in a PyTorch model that can be visualized by Lucent.

    Args:
        model: PyTorch model
        include_detailed_info: If True, returns detailed info about each layer

    Returns:
        If include_detailed_info=False: List of layer names that can be visualized
        If include_detailed_info=True: Dict with layer info including channels, type, etc.

    Example:
        model = resnet50(pretrained=True)
        layers = get_visualizable_layers(model)
        print(f"Found {len(layers)} visualizable layers")
    """
    # Get all layers using Lucent's utility
    all_layers = get_model_layers(model)

    visualizable_layers = []
    layer_details = {}

    # Patterns to skip - these typically don't work well for visualization
    skip_patterns = [
        "pool",
        "dropout",
        "fc",
        "avgpool",
        "adaptiveavgpool",
        "flatten",
        "relu",
        "leakyrelu",
        "elu",
        "gelu",
        "softmax",
        "sigmoid",
        "tanh",
        "batchnorm",
        "layernorm",
        "instancenorm",
        "groupnorm",
        "aux",
        "logits",
        "classifier",
        "features",
        "bn",
        "norm",
    ]

    for layer_name in all_layers:
        # Skip layers that typically don't work for visualization
        layer_lower = layer_name.lower()
        if any(pattern in layer_lower for pattern in skip_patterns):
            continue

        try:
            # Get channel count using our function
            num_channels = get_channels_from_lucent_name(model, layer_name)

            if num_channels is not None and num_channels > 0:
                visualizable_layers.append(layer_name)

                if include_detailed_info:
                    layer_info = {
                        "name": layer_name,
                        "channels": num_channels,
                        "lucent_compatible": True,
                    }

                    # Try to get the actual PyTorch module for more details
                    try:
                        # This might not work for all layer names, but try anyway
                        pytorch_name = layer_name.replace("_", ".")
                        module = model
                        for part in pytorch_name.split("."):
                            if part.isdigit():
                                module = module[int(part)]
                            else:
                                module = getattr(module, part)

                        layer_info["type"] = type(module).__name__
                        if hasattr(module, "kernel_size"):
                            layer_info["kernel_size"] = getattr(
                                module, "kernel_size", None
                            )
                        if hasattr(module, "stride"):
                            layer_info["stride"] = getattr(module, "stride", None)

                    except (AttributeError, KeyError, TypeError):
                        layer_info["type"] = "Unknown"

                    layer_details[layer_name] = layer_info

        except (AttributeError, KeyError, TypeError):
            continue

    if include_detailed_info:
        return layer_details
    else:
        return visualizable_layers


def categorize_visualizable_layers(model):
    """
    Categorize visualizable layers by their likely feature complexity.

    Args:
        model: PyTorch model

    Returns:
        Dict with categories: 'early', 'middle', 'deep' containing layer names

    Example:
        categories = categorize_visualizable_layers(model)
        print(f"Early layers: {len(categories['early'])}")
        print(f"Middle layers: {len(categories['middle'])}")
        print(f"Deep layers: {len(categories['deep'])}")
    """
    layers = get_visualizable_layers(model, include_detailed_info=True)

    categorized = {"early": [], "middle": [], "deep": []}

    # Simple heuristic based on layer names and channel counts
    for layer_name, info in layers.items():
        channels = info["channels"]

        # Early layers: typically have fewer channels
        if channels <= 128:
            categorized["early"].append(layer_name)
        # Deep layers: typically have many channels
        elif channels >= 512:
            categorized["deep"].append(layer_name)
        # Middle layers: intermediate channel counts
        else:
            categorized["middle"].append(layer_name)

    return categorized


def get_recommended_layers_for_visualization(model, num_per_category=5):
    """
    Get a curated list of recommended layers for visualization.

    Args:
        model: PyTorch model
        num_per_category: Number of layers to recommend per category

    Returns:
        Dict with recommended layers by category

    Example:
        recommended = get_recommended_layers_for_visualization(model, 3)
        for category, layers in recommended.items():
            print(f"{category}: {layers}")
    """
    categorized = categorize_visualizable_layers(model)

    recommendations = {}
    layer_details = get_visualizable_layers(model, include_detailed_info=True)

    for category, layers in categorized.items():
        # Sort by channel count for consistent selection
        if layers:
            sorted_layers = sorted(layers, key=lambda x: layer_details[x]["channels"])

            # Select evenly spaced layers from each category
            step = max(1, len(sorted_layers) // num_per_category)
            recommendations[category] = sorted_layers[::step][:num_per_category]
        else:
            recommendations[category] = []

    return recommendations


def test_layer_for_visualization(model, layer_name, test_channel=0):
    """
    Test if a specific layer can be used for visualization.

    Args:
        model: PyTorch model
        layer_name: Name of the layer to test
        test_channel: Channel index to test (default: 0)

    Returns:
        Dict with test results including success status and error info

    Example:
        result = test_layer_for_visualization(model, 'mixed4a_1x1_pre_relu_conv')
        if result['can_visualize']:
            print(f"✅ Layer {layer_name} works for visualization")
        else:
            print(f"❌ Layer {layer_name} failed: {result['error']}")
    """
    result = {
        "layer_name": layer_name,
        "can_visualize": False,
        "channels": None,
        "error": None,
    }

    try:
        # Check if we can get channel count
        num_channels = get_channels_from_lucent_name(model, layer_name)
        result["channels"] = num_channels

        if num_channels is None or num_channels <= 0:
            result["error"] = "No valid channels found"
            return result

        # Test if we can create an objective
        test_channel = min(test_channel, num_channels - 1)
        obj = objectives.channel(layer_name, test_channel)

        if obj is not None:
            result["can_visualize"] = True
        else:
            result["error"] = "Failed to create objective"

    except (AttributeError, ValueError, TypeError) as e:
        result["error"] = str(e)

    return result


def create_random_objective(model, layers_list=None, max_attempts=50):
    """
    Create a random objective with valid channel index for visualization.

    Args:
        model: PyTorch model
        layers_list: List of layer names to choose from (if None, uses all visualizable layers)
        max_attempts: Maximum number of attempts to create a valid objective

    Returns:
        Lucent objective or None if failed

    Example:
        obj = create_random_objective(model)
        if obj:
            render.render_vis(model, obj, show_inline=True)
    """
    if layers_list is None:
        layers_list = get_visualizable_layers(model)

    if not layers_list:
        print("No layers available for visualization")
        return None

    attempts = 0

    while attempts < max_attempts:
        layer_name = random.choice(layers_list)
        attempts += 1

        # Get the number of channels for this layer
        num_channels = get_channels_from_lucent_name(model, layer_name)

        if num_channels is not None and num_channels > 0:
            # Pick a random valid channel index
            channel_idx = random.randint(0, num_channels - 1)

            try:
                # Test if this layer name works with Lucent
                obj = objectives.channel(layer_name, channel_idx)

                print(f"✅ Selected layer: {layer_name}")
                print(f"   Available channels: 0-{num_channels-1}")
                print(f"   Selected channel: {channel_idx}")

                return obj
            except (AttributeError, ValueError, TypeError) as e:
                print(f"   ✗ Layer {layer_name} failed: {e}")
                continue
        else:
            continue

    print(f"❌ Failed to create objective after {max_attempts} attempts")
    return None


def print_model_summary(model):
    """
    Print a comprehensive summary of the model's visualizable layers.

    Args:
        model: PyTorch model

    Example:
        print_model_summary(resnet50(pretrained=True))
    """
    print(f"=== Model Summary: {type(model).__name__} ===")

    # Get all layers
    all_layers = get_model_layers(model)
    visualizable = get_visualizable_layers(model)
    categorized = categorize_visualizable_layers(model)

    print(f"Total layers in model: {len(all_layers)}")
    print(f"Visualizable layers: {len(visualizable)}")
    print(f"Percentage visualizable: {100 * len(visualizable) / len(all_layers):.1f}%")
    print()

    # Show categorization
    for category, layers in categorized.items():
        print(f"{category.capitalize()} layers ({len(layers)}):")
        if layers:
            # Show first few examples with channel counts
            layer_details = get_visualizable_layers(model, include_detailed_info=True)
            for layer in layers[:5]:
                channels = layer_details[layer]["channels"]
                print(f"  {layer} ({channels} channels)")
            if len(layers) > 5:
                print(f"  ... and {len(layers) - 5} more")
        print()

    # Recommendations
    recommended = get_recommended_layers_for_visualization(model, 3)
    print("Recommended layers for visualization:")
    for category, layers in recommended.items():
        if layers:
            print(f"  {category.capitalize()}: {', '.join(layers[:3])}")
    print()


# Main demo function
def main():
    """Demonstrate the layer analysis functions"""
    from lucent.modelzoo import inceptionv1

    print("=== Lucent Layer Utilities Demo ===")

    # Load a model
    print("Loading Inception V1 model...")
    model = inceptionv1(pretrained=True)

    # Print comprehensive summary
    print_model_summary(model)

    # Test creating a random objective
    print("Creating random objective...")
    obj = create_random_objective(model)

    if obj:
        print("🎉 Successfully created random objective!")
        print("You can now use this with render.render_vis()")
    else:
        print("❌ Failed to create random objective")


if __name__ == "__main__":
    main()
