"""
Batch depth map generation using MiDaS.

Processes visualization images and generates corresponding depth maps.
Runs as a separate batch process, can be executed in parallel with visualization generation.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthMapGenerator:
    """Generate depth maps from images using MiDaS."""

    def __init__(self, model_type: str = "DPT_Large", device: Optional[str] = None):
        """
        Initialize the depth map generator.

        Args:
            model_type: MiDaS model type. Options:
                - "DPT_Large" (highest quality, slower)
                - "DPT_Hybrid" (good quality, balanced speed)
                - "MiDaS_small" (fastest, lower quality)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        logger.info(f"Loading MiDaS model '{model_type}' on {self.device}...")

        # Load MiDaS model from torch hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # Select appropriate transform based on model type
        if model_type in ["MiDaS_small", "midas_v21_small_256"]:
            self.transform = midas_transforms.small_transform
        elif model_type in ["DPT_LeViT_224", "dpt_levit_224"]:
            self.transform = midas_transforms.levit_transform
        elif model_type in ["DPT_BEiT_L_512", "dpt_beit_large_512"]:
            self.transform = midas_transforms.beit512_transform
        elif model_type in ["DPT_SwinV2_T_256"]:
            self.transform = midas_transforms.swin256_transform
        elif model_type in ["DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_Swin_L_384",
                            "DPT_BEiT_L_384", "DPT_BEiT_B_384", "DPT_Next_ViT_L_384"]:
            self.transform = midas_transforms.swin384_transform
        else:
            # DPT_Large, DPT_Hybrid use dpt_transform
            self.transform = midas_transforms.dpt_transform

        logger.info("MiDaS model loaded successfully")

    def generate_depth_map(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Generate a depth map from an image.

        Args:
            image_path: Path to the input image
            output_path: Optional path to save the depth map. If None, uses input path with _depth suffix

        Returns:
            Depth map as numpy array (normalized 0-255)
        """
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # Apply MiDaS transform
        input_batch = self.transform(img_np).to(self.device)

        # Generate depth prediction
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()

        # Normalize to 0-255 range
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)

        # Save depth map
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_depth{input_path.suffix}"

        depth_img = Image.fromarray(depth_normalized)
        depth_img.save(output_path)

        return depth_normalized

    def process_directory(
        self,
        input_dir: str,
        recursive: bool = True,
        skip_existing: bool = True,
        pattern: str = "*.png",
        priority_layers: list = None
    ) -> dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images
            recursive: Whether to process subdirectories
            skip_existing: Skip images that already have depth maps
            pattern: Glob pattern for image files
            priority_layers: List of layer names to process first (e.g., ['mixed5b', 'mixed5a'])

        Returns:
            Statistics dictionary with counts
        """
        input_path = Path(input_dir)

        if recursive:
            image_files = list(input_path.rglob(pattern))
        else:
            image_files = list(input_path.glob(pattern))

        # Filter out depth maps and yaml files from processing
        image_files = [
            f for f in image_files
            if '_depth' not in f.stem and not f.suffix.lower() == '.yaml'
        ]

        # Sort by priority layers first if specified
        if priority_layers:
            def sort_key(f):
                # Check if any part of the path contains the layer name
                path_str = str(f)
                for i, layer in enumerate(priority_layers):
                    if f"\\{layer}\\" in path_str or f"/{layer}/" in path_str:
                        return (i, path_str)
                return (len(priority_layers), path_str)
            image_files.sort(key=sort_key)
            logger.info(f"Prioritizing layers: {priority_layers}")

        stats = {
            'total': len(image_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        logger.info(f"Found {len(image_files)} images to process in {input_dir}")

        for i, image_file in enumerate(image_files):
            try:
                # Check if depth map already exists
                depth_path = image_file.parent / f"{image_file.stem}_depth{image_file.suffix}"

                if skip_existing and depth_path.exists():
                    stats['skipped'] += 1
                    continue

                # Generate depth map
                logger.info(f"[{i+1}/{len(image_files)}] Processing: {image_file.name}")
                self.generate_depth_map(str(image_file), str(depth_path))
                stats['processed'] += 1

            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                stats['errors'] += 1

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from visualization images using MiDaS"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="screen_captures",
        help="Input directory containing images (default: screen_captures)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DPT_Large",
        choices=[
            "DPT_Large", "DPT_Hybrid", "MiDaS_small",
            # MiDaS 3.1 models (use exact names from hubconf)
            "DPT_BEiT_L_512", "DPT_BEiT_L_384", "DPT_BEiT_B_384",
            "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256",
            "DPT_Swin_L_384", "DPT_Next_ViT_L_384", "DPT_LeViT_224"
        ],
        help="MiDaS model type (default: DPT_Large)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't process subdirectories"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate depth maps even if they exist"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Process a single image file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--priority-layers",
        type=str,
        nargs="+",
        help="Layer names to process first (e.g., --priority-layers mixed5b mixed5a)"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = DepthMapGenerator(model_type=args.model, device=args.device)

    if args.single:
        # Process single image
        logger.info(f"Processing single image: {args.single}")
        depth = generator.generate_depth_map(args.single)
        logger.info(f"Depth map generated successfully")
    else:
        # Process directory
        stats = generator.process_directory(
            input_dir=args.input_dir,
            recursive=not args.no_recursive,
            skip_existing=not args.force,
            priority_layers=args.priority_layers
        )

        logger.info("=" * 60)
        logger.info("Depth map generation complete!")
        logger.info(f"  Total images found: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Skipped (existing): {stats['skipped']}")
        logger.info(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
