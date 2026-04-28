# Gabor-weighted spatial objectives for Lucent
import torch
import numpy as np
from lucent.optvis import objectives


def gabor_function(
    x, y, sigma_x=1.0, sigma_y=1.0, theta=0, lambda_freq=1.0, psi=0, gamma=1.0
):
    """
    Compute Gabor function value at position (x, y).

    Args:
        x, y: Spatial coordinates
        sigma_x: Standard deviation in x direction
        sigma_y: Standard deviation in y direction
        theta: Orientation angle (radians)
        lambda_freq: Wavelength of the sinusoidal component
        psi: Phase offset
        gamma: Aspect ratio (ellipticity)

    Returns:
        Gabor function value at (x, y)
    """
    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope
    gaussian = np.exp(
        -0.5 * ((x_theta**2 / sigma_x**2) + (gamma**2 * y_theta**2 / sigma_y**2))
    )

    # Sinusoidal component
    sinusoid = np.cos(2 * np.pi * x_theta / lambda_freq + psi)

    return gaussian * sinusoid


def create_gabor_weighted_objective(
    layer_name,
    channel_idx,
    size=7,
    with_offset=(0, 0),
    sigma=(1.0, 1.0),
    theta=0.0,
    lambda_freq=2.0,
    psi=0.0,
    gamma=1.0,
    normalize_weights=True,
    spatial_weight=1.0,
):
    """
    Create an objective that targets an NxN region with Gabor function weights.

    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        size: Size of the region (should be odd, e.g., 7 for 7x7)
        x_offset: Horizontal offset from center (positive = right, negative = left)
        y_offset: Vertical offset from center (positive = down, negative = up)
        sigma_x: Gabor sigma in x direction
        sigma_y: Gabor sigma in y direction
        theta: Gabor orientation angle in radians (0 = horizontal)
        lambda_freq: Gabor wavelength
        psi: Gabor phase offset
        gamma: Gabor aspect ratio
        normalize_weights: Whether to normalize Gabor weights to sum to 1
        spatial_weight: Overall scaling factor

    Returns:
        Lucent objective with Gabor-weighted spatial targeting
    """

    # Ensure size is odd for symmetric centering
    if size % 2 == 0:
        # print(
        #     f"Warning: Even size {size} provided, using {size+1} for symmetric centering"
        # )
        size += 1

    # Calculate the radius (how many pixels from center)
    radius = size // 2

    # Pre-compute Gabor weights for the NxN grid
    gabor_weights = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # Convert array indices to spatial coordinates relative to center
            x = j - radius  # horizontal position relative to center
            y = i - radius  # vertical position relative to center
            gabor_weights[i, j] = gabor_function(
                x, y, sigma[0], sigma[1], theta, lambda_freq, psi, gamma
            )

    # Normalize weights if requested
    if normalize_weights:
        weight_sum = np.abs(gabor_weights).sum()
        if weight_sum > 0:
            gabor_weights = gabor_weights / weight_sum

    # print(f"Gabor weights shape: {gabor_weights.shape}")
    # print(f"Gabor weight range: [{gabor_weights.min():.3f}, {gabor_weights.max():.3f}]")

    # Create a custom objective that targets the Gabor-weighted region
    def gabor_weighted_obj(net):
        # Get the activations for the specified layer and channel
        layer_acts = net(layer_name)

        if len(layer_acts.shape) != 4:  # Should be [batch, channels, height, width]
            print(
                f"Warning: Layer {layer_name} has unexpected shape {layer_acts.shape}"
            )
            return -layer_acts[:, channel_idx].mean()

        # Get spatial dimensions
        _, _, h, w = layer_acts.shape

        # Calculate center coordinates
        center_h, center_w = h // 2, w // 2

        # Apply offsets
        target_h = center_h + with_offset[1]
        target_w = center_w + with_offset[0]

        # Define NxN region around target position (with bounds checking)
        h_start = max(0, target_h - radius)
        h_end = min(h, target_h + radius + 1)
        w_start = max(0, target_w - radius)
        w_end = min(w, target_w + radius + 1)

        # Check if the offset region is valid (not entirely out of bounds)
        if h_start >= h or w_start >= w or h_end <= 0 or w_end <= 0:
            print(
                f"Warning: Offset region ({with_offset[0]}, {with_offset[1]}) "
                f"is out of bounds for {h}x{w} feature map"
            )
            # Fallback to center region
            h_start = max(0, center_h - radius)
            h_end = min(h, center_h + radius + 1)
            w_start = max(0, center_w - radius)
            w_end = min(w, center_w + radius + 1)

        # Extract NxN offset region
        offset_region = layer_acts[:, channel_idx, h_start:h_end, w_start:w_end]

        # Handle case where extracted region is smaller than expected (near boundaries)
        actual_h, actual_w = offset_region.shape[-2:]

        if actual_h != size or actual_w != size:
            # Crop or pad the Gabor weights to match the actual extracted region
            weight_h_start = max(0, (size - actual_h) // 2)
            weight_h_end = weight_h_start + actual_h
            weight_w_start = max(0, (size - actual_w) // 2)
            weight_w_end = weight_w_start + actual_w

            cropped_weights = gabor_weights[
                weight_h_start:weight_h_end, weight_w_start:weight_w_end
            ]
        else:
            cropped_weights = gabor_weights

        # Convert weights to tensor on same device as activations
        weights_tensor = torch.tensor(
            cropped_weights, dtype=offset_region.dtype, device=offset_region.device
        )

        # Apply Gabor weights element-wise and sum
        weighted_activations = (offset_region * weights_tensor).sum()

        # Return negative (Lucent maximizes by minimizing negative)
        return -weighted_activations * spatial_weight

    # Create the objective using Lucent's Objective class
    return objectives.Objective(gabor_weighted_obj)


def create_gabor_preset_objectives(
    layer_name, channel_idx, preset="edge_detector", **kwargs
):
    """
    Create Gabor objectives with common preset configurations.

    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        preset: Preset configuration:
            - "edge_detector": Horizontal edge detection
            - "vertical_edges": Vertical edge detection
            - "diagonal_edges": Diagonal edge detection
            - "texture_fine": Fine texture detection
            - "texture_coarse": Coarse texture detection
            - "blob_detector": Blob/spot detection
        **kwargs: Additional parameters to override preset values

    Returns:
        Gabor-weighted objective
    """
    presets = {
        "edge_detector": {
            "size": 7,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "theta": 0,  # horizontal edges
            "lambda_freq": 3.0,
            "psi": 0,
            "gamma": 1.0,
        },
        "vertical_edges": {
            "size": 7,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "theta": np.pi / 2,  # vertical edges
            "lambda_freq": 3.0,
            "psi": 0,
            "gamma": 1.0,
        },
        "diagonal_edges": {
            "size": 7,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "theta": np.pi / 4,  # 45 degree edges
            "lambda_freq": 3.0,
            "psi": 0,
            "gamma": 1.0,
        },
        "texture_fine": {
            "size": 9,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "theta": 0,
            "lambda_freq": 1.5,  # higher frequency
            "psi": 0,
            "gamma": 1.0,
        },
        "texture_coarse": {
            "size": 11,
            "sigma_x": 2.0,
            "sigma_y": 2.0,
            "theta": 0,
            "lambda_freq": 4.0,  # lower frequency
            "psi": 0,
            "gamma": 1.0,
        },
        "blob_detector": {
            "size": 7,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "theta": 0,
            "lambda_freq": 10.0,  # very low frequency (almost pure Gaussian)
            "psi": 0,
            "gamma": 1.0,
        },
    }

    if preset not in presets:
        print(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return None

    # Merge preset with any user overrides
    params = presets[preset].copy()
    params.update(kwargs)

    print(f"Creating Gabor objective with preset: {preset}")
    print(f"Parameters: {params}")

    return create_gabor_weighted_objective(layer_name, channel_idx, **params)


def visualize_gabor_weights(
    size=7, sigma_x=1.0, sigma_y=1.0, theta=0, lambda_freq=2.0, psi=0, gamma=1.0
):
    """
    Utility function to visualize Gabor weights (useful for debugging).

    Returns:
        2D numpy array of Gabor weights
    """
    radius = size // 2
    gabor_weights = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x = j - radius
            y = i - radius
            gabor_weights[i, j] = gabor_function(
                x, y, sigma_x, sigma_y, theta, lambda_freq, psi, gamma
            )

    return gabor_weights


def create_multi_orientation_gabor_objective(
    layer_name,
    channel_idx,
    orientations=None,
    weights=None,
    **gabor_kwargs,
):
    """
    Create an objective that combines multiple Gabor orientations.

    Args:
        layer_name: Name of the layer
        channel_idx: Channel index
        orientations: List of orientation angles in radians
        weights: Weights for each orientation (if None, uses equal weights)
        **gabor_kwargs: Additional Gabor parameters

    Returns:
        Combined multi-orientation objective
    """
    if orientations is None:
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    if weights is None:
        weights = [1.0] * len(orientations)

    if len(weights) != len(orientations):
        raise ValueError("Number of weights must match number of orientations")

    objectives_list = []

    for i, (theta, weight) in enumerate(zip(orientations, weights)):
        print(
            f"Creating Gabor objective {i+1}/{len(orientations)} with orientation {theta:.2f} rad"
        )

        obj = create_gabor_weighted_objective(
            layer_name, channel_idx, theta=theta, spatial_weight=weight, **gabor_kwargs
        )

        if obj:
            objectives_list.append(obj)

    if objectives_list:
        return sum(objectives_list)
    else:
        print("Failed to create any Gabor objectives")
        return None


# Example usage and testing
if __name__ == "__main__":
    from lucent.modelzoo import inceptionv1

    # Load model
    model = inceptionv1(pretrained=True)

    # Test basic Gabor objective
    print("=== Testing Gabor-weighted Objectives ===")

    # Create a horizontal edge detector
    gabor_obj = create_gabor_weighted_objective(
        "mixed4a",
        42,
        size=(7, 7),
        sigma=(1.5, 1.5),
        theta=0,  # horizontal edges
        lambda_freq=3.0,
    )

    print(f"Horizontal edge Gabor objective created: {gabor_obj is not None}")

    # Test preset objectives
    print("\n=== Testing Gabor Presets ===")
    presets = [
        "edge_detector",
        "vertical_edges",
        "diagonal_edges",
        "texture_fine",
        "blob_detector",
    ]

    for preset in presets:
        obj = create_gabor_preset_objectives("mixed4a", 42, preset=preset)
        print(f"{preset}: {obj is not None}")

    # Test multi-orientation objective
    print("\n=== Testing Multi-orientation Gabor ===")
    multi_obj = create_multi_orientation_gabor_objective(
        "mixed4a",
        42,
        orientations=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        size=7,
        sigma_x=1.5,
        sigma_y=1.5,
        lambda_freq=3.0,
    )
    print(f"Multi-orientation objective created: {multi_obj is not None}")

    # Visualize some Gabor weights
    print("\n=== Visualizing Gabor Weights ===")

    # Horizontal edge detector
    weights_horizontal = visualize_gabor_weights(
        size=7, sigma_x=1.5, sigma_y=1.5, theta=0, lambda_freq=3.0
    )
    print(f"Horizontal edge weights shape: {weights_horizontal.shape}")
    print(
        f"Weight range: [{weights_horizontal.min():.3f}, {weights_horizontal.max():.3f}]"
    )

    # Diagonal edge detector
    weights_diagonal = visualize_gabor_weights(
        size=7, sigma_x=1.5, sigma_y=1.5, theta=np.pi / 4, lambda_freq=3.0
    )
    print(f"Diagonal edge weights shape: {weights_diagonal.shape}")
    print(f"Weight range: [{weights_diagonal.min():.3f}, {weights_diagonal.max():.3f}]")

    print("\n=== Demo Complete ===")
    print("\nTo use with render.render_vis():")
    print("obj = create_gabor_preset_objectives('mixed4a', 42, 'edge_detector')")
    print("render.render_vis(model, obj, show_inline=True)")
