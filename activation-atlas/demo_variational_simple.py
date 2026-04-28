"""
Simplified variational demo that shows the approach more clearly.

This demo uses the variational framework but with a clearer objective:
maximize activations while maintaining uncertainty estimates.
"""

import torch
import torch.nn as nn
from lucent.modelzoo import inceptionv1
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from variational_visualization import VariationalImageParameter


def simple_variational_demo():
    """
    Simple demo showing variational approach with activation maximization.
    """
    print("\n" + "=" * 60)
    print("SIMPLE VARIATIONAL FEATURE VISUALIZATION")
    print("=" * 60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("Loading InceptionV1...")
    model = inceptionv1(pretrained=True)
    model.eval()
    model = model.to(device)

    # Target
    layer_name = "mixed4a_1x1_pre_relu_conv"
    channel_idx = 42

    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations["output"] = output

    # Register hook
    for name, module in model.named_modules():
        if name.replace(".", "_") == layer_name:
            module.register_forward_hook(hook_fn)
            print(f"Registered hook on: {name}")
            break

    # Create variational parameter
    image_param = VariationalImageParameter(
        image_size=224,  # Reduced resolution to avoid OOM
        channels=3,
        initial_log_sigma=-4.0,  # Start with lower variance for more focused optimization
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        image_param.parameters(), lr=0.1
    )  # Higher learning rate

    # Training loop
    n_iterations = 1600  # Even more iterations for highly refined visualization
    n_samples = 64  # Reduced samples to avoid OOM

    # Clear GPU cache before starting
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nOptimizing for {n_iterations} iterations...")
    print(f"Samples per iteration: {n_samples}\n")

    loss_history = []

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Sample images
        images = image_param.sample(n_samples=n_samples)

        # Forward pass
        _ = model(images)
        acts = activations["output"][:, channel_idx]

        # Objective: MAXIMIZE mean activation (so minimize negative)
        mean_activation = acts.mean()
        loss = -mean_activation

        # Add small variance penalty to prevent collapse
        variance_penalty = (
            0.005 * image_param.get_variance().mean()
        )  # Lower penalty for more variation
        total_loss = loss + variance_penalty

        # Backward
        total_loss.backward()
        optimizer.step()

        loss_history.append(
            {
                "iteration": iteration,
                "activation": mean_activation.item(),
                "loss": total_loss.item(),
                "variance": image_param.get_variance().mean().item(),
            }
        )

        if iteration % 50 == 0:
            print(
                f"Iter {iteration:3d}: Activation = {mean_activation.item():8.4f}, "
                f"Variance = {image_param.get_variance().mean().item():.6f}"
            )

    print("\n[OK] Optimization complete!")

    # Get results
    mean_image = image_param.get_mean().squeeze(0).detach().cpu()
    variance = image_param.get_variance().squeeze(0).mean(dim=0).detach().cpu()

    # Normalize mean image for display
    mean_image_np = mean_image.numpy().transpose(1, 2, 0)
    mean_image_np = (mean_image_np - mean_image_np.min()) / (
        mean_image_np.max() - mean_image_np.min()
    )

    # Create output directory
    os.makedirs("screen_captures/variational", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save mean image
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_image_np)
    plt.title(f"Mean Image: {layer_name}, channel {channel_idx}")
    plt.axis("off")
    mean_path = f"screen_captures/variational/{timestamp}_simple_mean.png"
    plt.savefig(
        mean_path, bbox_inches="tight", dpi=300
    )  # Higher DPI for publication quality
    plt.close()
    print(f"\nSaved mean image: {mean_path}")

    # Save variance map
    plt.figure(figsize=(8, 8))
    plt.imshow(variance.numpy(), cmap="hot")
    plt.colorbar(label="Variance")
    plt.title(f"Uncertainty Map: {layer_name}, channel {channel_idx}")
    plt.axis("off")
    var_path = f"screen_captures/variational/{timestamp}_simple_variance.png"
    plt.savefig(
        var_path, bbox_inches="tight", dpi=300
    )  # Higher DPI for publication quality
    plt.close()
    print(f"Saved variance map: {var_path}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = [h["iteration"] for h in loss_history]
    activations_list = [h["activation"] for h in loss_history]
    variances = [h["variance"] for h in loss_history]

    ax1.plot(iterations, activations_list)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Activation")
    ax1.set_title("Activation Growth")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, variances)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean Variance")
    ax2.set_title("Uncertainty Evolution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = f"screen_captures/variational/{timestamp}_simple_curves.png"
    plt.savefig(
        curve_path, bbox_inches="tight", dpi=300
    )  # Higher DPI for publication quality
    plt.close()
    print(f"Saved training curves: {curve_path}")

    # Show samples
    print("\nGenerating samples from learned distribution...")
    with torch.no_grad():
        samples = image_param.sample(n_samples=9)
        samples = samples.cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        img = samples[i].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis("off")

    plt.suptitle(f"Samples from Distribution: {layer_name}, channel {channel_idx}")
    plt.tight_layout()
    samples_path = f"screen_captures/variational/{timestamp}_simple_samples.png"
    plt.savefig(
        samples_path, bbox_inches="tight", dpi=300
    )  # Higher DPI for publication quality
    plt.close()
    print(f"Saved samples: {samples_path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nFinal mean activation: {activations_list[-1]:.4f}")
    print(f"Final variance: {variances[-1]:.6f}")
    print(f"\nCheck screen_captures/variational/ for results")


if __name__ == "__main__":
    simple_variational_demo()
