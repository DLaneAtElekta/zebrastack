"""_generate_random_objective_visualizations.py
Generates and saves random objective visualizations for various CNN models
"""

import datetime
import os
import random
import logging
import yaml
from typing import Optional, Dict, Any, List
import torch
from lucent.optvis import render, param, transform

from lucent.modelzoo import inceptionv1, inception_v3, resnet152, resnext101_64x4d
from lucent_layer_utils import get_visualizable_layers, get_layer_dimensions
from spatial_objectives import create_random_objective
from wrapping_transforms import wrap_transform

# Setup logging
LOG_FILENAME = (
    f"logs\\visualization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)


# %%
# hoping for a favorite!

# Initialize counter for unique image identification
image_counter = 0

GENERATED_IMAGE_SIZE = 384
GENERATE_BASE_IMAGE = False
GENERATE_JITTER_IMAGE = False


def generate_filename(
    objective_type: str,
    sampled_channels: int,
    transform_type: str = "base",
    layer_name: Optional[str] = None,
) -> str:
    """Generate filename with timestamp and parameters"""
    global image_counter
    image_counter += 1
    base_name: str = "_".join(
        [
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            f"img{image_counter:04d}",
            objective_type,
            f"{sampled_channels}ch",
            transform_type,
        ]
    )
    if layer_name:
        layer_name = "\\".join(layer_name.split("_"))
        return f"screen_captures\\{layer_name}\\{base_name}.png"
    return f"screen_captures\\{base_name}.png"


def log_visualization_params(
    objective_type: str,
    sampled_channels: int,
    layer_info: Optional[Dict[str, Any]],
    image_filename: str,
    model: torch.nn.Module,
    transform_type: str = "base",
) -> None:
    """Log all visualization parameters"""
    logger.info("Generated visualization #%d:", image_counter)
    logger.info("  Image: %s", image_filename)
    logger.info("  Objective type: %s", objective_type)
    logger.info("  Sampled channels: %d", sampled_channels)
    logger.info("  Transform type: %s", transform_type)
    if layer_info:
        logger.info("  Layer info: %s", layer_info)
    logger.info("  Model: %s", type(model).__name__)
    logger.info("-" * 40)


def save_visualization_params_to_yaml(
    objective_type: str,
    sampled_channels: int,
    layer_info: Optional[Dict[str, Any]],
    image_filename: str,
    model: torch.nn.Module,
    transform_type: str = "base",
    transform_details: Optional[str] = None,
    all_params: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save visualization parameters to a YAML file alongside the image"""
    # Create YAML filename by replacing .png with .yaml
    yaml_filename = image_filename.replace('.png', '.yaml')
    
    # Prepare data structure matching the log format
    visualization_data = {
        'visualization_info': {
            'image_number': image_counter,
            'image_filename': image_filename,
            'objective_type': objective_type,
            'sampled_channels': sampled_channels,
            'transform_type': transform_type,
            'model_name': type(model).__name__,
            'generation_timestamp': datetime.datetime.now().isoformat(),
        },
        'layer_info': layer_info if layer_info else {},
        'transform_details': transform_details if transform_details else transform_type,
        'objective_parameters': all_params if all_params else [],
    }
    
    # Save to YAML file
    try:
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(visualization_data, yaml_file, default_flow_style=False, indent=2)
        logger.info("  YAML saved: %s", yaml_filename)
    except Exception as e:
        logger.error("Failed to save YAML file %s: %s", yaml_filename, e)


def visualize_to_file(
    model,
    use_objective,
    sampled_channels,
    layer_name,
    obj,
    with_transforms,
    transforms_label,
    transform_details,
    all_params=None,
):
    filename = generate_filename(
        use_objective, sampled_channels, transforms_label, layer_name
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info("Generating %s visualization: %s", transforms_label, filename)

    # TODO: figure out how to morph visualizations to make a "breathing" effect
    _ = render.render_vis(
        model,
        objective_f=obj,
        param_f=lambda: param.image(GENERATED_IMAGE_SIZE),
        transforms=with_transforms,
        show_image=False,
        save_image=True,
        image_name=filename,
    )

    # Log to console/file
    log_visualization_params(
        use_objective,
        sampled_channels,
        transform_details,
        filename,
        model,
        transforms_label,
    )
    
    # Save parameters to YAML file
    save_visualization_params_to_yaml(
        use_objective,
        sampled_channels,
        transform_details,
        filename,
        model,
        transforms_label,
        transform_details,
        all_params,
    )



def generate_for_model(model: torch.nn.Module, layers: List[str]) -> None:
    """Generate visualizations for a specific model and its layers.

    Args:
        model (torch.nn.Module): The model to visualize.
        layers (List[str]): The layers to visualize.
    """
    use_objective = random.choice(
        ["channel", "neuron", "center_3x3", "center_5x5", "center_7x7"]
    )
    use_objective = "neuron"
    use_objective = "gabor"

    sampled_channels = 0 # random.randint(1, 11)

    logger.info(
        "\nStarting new batch with %s objective, %d channels",
        use_objective,
        sampled_channels,
    )

    num_of_points = random.randint(1, 4)
    layer_name = random.choice(layers)
    height, width, total_channels = get_layer_dimensions(
        model, layer_name, input_size=GENERATED_IMAGE_SIZE
    )
    print(f"Selected layer: {layer_name} ({height}x{width}x{total_channels})")

    # select other layers that have the same height, width as this one
    matching_layers = [layer_name]
    for other_layer in random.choices(layers, k=10):
        if other_layer != layer_name:
            h, w, other_channels = get_layer_dimensions(
                model, other_layer, input_size=GENERATED_IMAGE_SIZE
            )
            if h == height and w == width:
                print(f"  Matching layer: {other_layer} ({h}x{w}x{other_channels})")
                matching_layers.append(other_layer)

    all_params = []
    for n in range(num_of_points):
        # for each objective type (channel, neuron, center_, gabor)
        # randomly generate parameters
        #
        for_layer = random.choice(matching_layers)
        # generate random channel index (as float)
        height, width, total_channels = get_layer_dimensions(
            model, for_layer, input_size=GENERATED_IMAGE_SIZE
        )

        params = {
            "objective_type": use_objective,
            "layer": for_layer,
            "channel": random.uniform(0.0, total_channels - 1.0),
        }

        if use_objective in [
            "neuron",
            "center_3x3",
            "center_5x5",
            "center_7x7",
            "gabor",
        ]:
            # random (x,y) within layer dimensions (default to 0,0)
            params["offset"] = (
                (0.0, 0.0)
                if n == 0
                else (
                    random.uniform(-width // 2, width // 2),
                    random.uniform(-height // 2, height // 2),
                )
            )

        if use_objective == "gabor":
            params = {
                **params,
                # random gabor parameters: sigma, lambda, theta, psi, gamma
                "sigma": (random.uniform(0.5, 3.0), random.uniform(0.5, 3.0)),
                "lambda": random.uniform(1.0, 5.0),
                "theta": random.uniform(0.0, 6.28319),  # 0 to 2*pi
                "psi": random.uniform(0.0, 6.28319),  # 0 to 2*pi
                "gamma": random.uniform(0.5, 2.0),
            }

        all_params += [params]

    sampled_channels = random.randint(1, 4)
    obj = create_random_objective(
        model,
        layers,
        layer_name=layer_name,
        objective_types=[use_objective] * num_of_points,
        offsets=[
            (
                random.randint(-width // 2, width // 2),
                random.randint(-height // 6, height // 6),
            )
            for _ in range(num_of_points - 1)
        ],
        sampled_channels=sampled_channels,
    )

    assert obj is not None, "Objective creation failed"

    if GENERATE_BASE_IMAGE:
        # Base visualization (no transforms)
        visualize_to_file(
            model,
            use_objective,
            sampled_channels,
            layer_name,
            obj,
            None,
            "base",
            "base",
            all_params,
        )

    # %%
    # Adding jitter, notice that the visualization is much less noisy!

    if GENERATE_JITTER_IMAGE:
        jitter_only = [transform.jitter(8)]

        visualize_to_file(
            model,
            use_objective,
            sampled_channels,
            layer_name,
            obj,
            jitter_only,
            "jitter",
            "jitter(8)",
            all_params,
        )

    # %%
    # Adding a whole suite of transforms!

    all_transforms = [
        # transform.pad(16),
        wrap_transform(0.20),
        transform.jitter(8),
        transform.random_scale([n / 100.0 for n in range(80, 120)]),
        transform.random_rotate(
            list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))
        ),
        transform.jitter(2),
    ]

    transform_details = ", ".join(
        [
            "pad(16)",
            "jitter(8)",
            "random_scale(0.8-1.2)",
            "random_rotate(-10 to +10)",
            "jitter(2)",
        ]
    )
    for _ in range(3):
        visualize_to_file(
            model,
            use_objective,
            sampled_channels,
            layer_name,
            obj,
            all_transforms,
            "full_transforms",
            transform_details,
            all_params,
        )

    logger.info(
        "Completed batch #%d. Total images generated: %d",
        image_counter // 3,
        image_counter,
    )
    logger.info("=" * 60)


def main(use_model):
    # Log session start
    logger.info("=" * 60)
    logger.info("Starting new visualization generation session")
    logger.info("Log file: %s", LOG_FILENAME)
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = use_model(pretrained=True)
    _ = model.to(device).eval()

    logger.info("Loaded model: %s", type(model).__name__)

    good_layers = get_visualizable_layers(model)
    logger.info("Found %d visualizable layers", len(good_layers))
    logger.info("Sample layers: %s...", good_layers[:5])
    for _ in range(1000):
        generate_for_model(model, good_layers)


if __name__ == "__main__":
    models_to_choose = [inceptionv1, inception_v3, resnet152, resnext101_64x4d]
    main(models_to_choose[0])

# %%
