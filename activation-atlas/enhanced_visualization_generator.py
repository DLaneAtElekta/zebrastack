"""
Enhanced visualization generation with comprehensive logging.

This module extends the existing visualization generation to include:
- Objective parameter generation using the new generators
- Comprehensive logging to pandas DataFrames
- CSV serialization for later analysis
- Metadata tracking for all visualizations
"""

import time
import datetime
import os
import random
import logging
from typing import Optional, Dict, Any, List

from lucent.optvis import render, param, transform
from lucent.modelzoo import inceptionv1

from objective_generators import (
    generate_random_objective_params,
    create_objective_from_params,
    mutate_objective_params
)
from visualization_logger import get_logger
from lucent_layer_utils import get_visualizable_layers
from wrapping_transforms import wrap_transform, tile_transform


class EnhancedVisualizationGenerator:
    """
    Enhanced visualization generator with comprehensive logging and parameter tracking.
    """
    
    def __init__(
        self,
        model_name: str = "inception_v1",
        image_size: int = 384,
        csv_filename: Optional[str] = None,
        enable_wrapping: bool = True
    ):
        """
        Initialize the enhanced generator.
        
        Args:
            model_name: Name of the model to use
            image_size: Size of generated images
            csv_filename: CSV file for logging (auto-generated if None)
            enable_wrapping: Whether to enable wrapping transforms
        """
        self.model_name = model_name
        self.image_size = image_size
        self.enable_wrapping = enable_wrapping
        
        # Initialize logger
        if csv_filename is None:
            csv_filename = f"visualizations_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.logger = get_logger(csv_filename)
        self.generation_logger = logging.getLogger(__name__)
        
        # Load model and get layers
        self.model = inceptionv1(pretrained=True)
        self.model.eval()
        self.visualizable_layers = get_visualizable_layers(self.model)
        
        self.generation_logger.info(f"✅ Initialized enhanced generator for {model_name}")
        self.generation_logger.info(f"📊 Found {len(self.visualizable_layers)} visualizable layers")
        self.generation_logger.info(f"💾 Logging to: {csv_filename}")
    
    def generate_single_visualization(
        self,
        objective_params: Optional[Dict[str, Any]] = None,
        transforms_config: Optional[Dict[str, Any]] = None,
        save_image: bool = True,
        user_rating: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single visualization with comprehensive logging.
        
        Args:
            objective_params: Objective parameters (auto-generated if None)
            transforms_config: Transform configuration
            save_image: Whether to save the generated image
            user_rating: Optional user rating for the visualization
            notes: Optional notes about the visualization
            
        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        
        # Generate objective parameters if not provided
        if objective_params is None:
            objective_types = ["channel", "neuron", "center", "gabor"]
            objective_params_list = generate_random_objective_params(
                self.model,
                self.visualizable_layers,
                objective_types=objective_types,
                num_objectives=1,
                input_size=self.image_size
            )
            objective_params = objective_params_list[0]
        
        # Create objective from parameters
        try:
            obj = create_objective_from_params(objective_params)
        except Exception as e:
            self.generation_logger.error(f"Failed to create objective: {e}")
            return {"success": False, "error": str(e)}
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        obj_type = objective_params.get('objective_type', 'unknown')
        layer_name = objective_params.get('layer_name', 'unknown').replace('_', '\\')
        
        filename = f"screen_captures\\{layer_name}\\{timestamp}_{obj_type}_enhanced.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Setup transforms
        transforms_list = []
        transform_names = []
        
        # Base transforms
        if transforms_config is None:
            transforms_config = {
                "use_padding": True,
                "use_jitter": True,
                "use_scale": True,
                "use_rotate": True,
                "use_wrapping": self.enable_wrapping,
                "wrap_factor": random.uniform(0.2, 0.5) if self.enable_wrapping else None,
                "tile_count": random.randint(2, 4) if self.enable_wrapping and random.choice([True, False]) else None
            }
        
        if transforms_config.get("use_padding", True):
            transforms_list.append(transform.pad(16))
            transform_names.append("pad(16)")
        
        if transforms_config.get("use_jitter", True):
            transforms_list.append(transform.jitter(8))
            transform_names.append("jitter(8)")
        
        if transforms_config.get("use_scale", True):
            transforms_list.append(transform.random_scale([n / 100.0 for n in range(80, 120)]))
            transform_names.append("random_scale(0.8-1.2)")
        
        if transforms_config.get("use_rotate", True):
            transforms_list.append(transform.random_rotate(
                list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))
            ))
            transform_names.append("random_rotate(-10 to +10)")
        
        # Add wrapping transforms if enabled
        if transforms_config.get("use_wrapping", False):
            wrap_factor = transforms_config.get("wrap_factor")
            tile_count = transforms_config.get("tile_count")
            
            if wrap_factor:
                transforms_list.append(wrap_transform(wrap_factor))
                transform_names.append(f"wrap({wrap_factor:.2f})")
            
            if tile_count:
                transforms_list.append(tile_transform(tile_count))
                transform_names.append(f"tile({tile_count})")
        
        if transforms_config.get("use_jitter", True):
            transforms_list.append(transform.jitter(2))
            transform_names.append("jitter(2)")
        
        # Generate visualization
        self.generation_logger.info(f"🎨 Generating {obj_type} visualization: {filename}")
        
        try:
            result = render.render_vis(
                self.model,
                objective_f=obj,
                param_f=lambda: param.image(self.image_size),
                transforms=transforms_list if transforms_list else None,
                show_image=False,
                save_image=save_image,
                image_name=filename if save_image else None,
            )
        except Exception as e:
            self.generation_logger.error(f"Failed to generate visualization: {e}")
            return {"success": False, "error": str(e)}
        
        generation_time = time.time() - start_time
        
        # Prepare logging data
        generation_config = {
            "image_size": self.image_size,
            "sampled_channels": 1,  # For now, single objective
            "optimization_steps": 512,  # Default Lucent steps
            "learning_rate": 0.05,  # Default Lucent learning rate
        }
        
        transform_params = {
            "transform_names": transform_names,
            "wrapping_enabled": transforms_config.get("use_wrapping", False),
            "wrap_factor": transforms_config.get("wrap_factor"),
            "tile_count": transforms_config.get("tile_count"),
        }
        
        # Log to pandas DataFrame
        vis_id = self.logger.log_visualization(
            objective_params=objective_params,
            image_filename=filename,
            model_name=self.model_name,
            generation_config=generation_config,
            transform_params=transform_params,
            generation_time=generation_time,
            user_rating=user_rating,
            notes=notes
        )
        
        self.generation_logger.info(f"✅ Generated visualization {vis_id} in {generation_time:.2f}s")
        
        return {
            "success": True,
            "visualization_id": vis_id,
            "filename": filename,
            "objective_params": objective_params,
            "generation_time": generation_time,
            "transform_params": transform_params,
            "result": result
        }
    
    def generate_batch(
        self,
        num_visualizations: int = 10,
        objective_types: Optional[List[str]] = None,
        enable_mutation: bool = False,
        mutation_rate: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of visualizations.
        
        Args:
            num_visualizations: Number of visualizations to generate
            objective_types: Types of objectives to use
            enable_mutation: Whether to include mutations of previous visualizations
            mutation_rate: Rate of mutation for child visualizations
            
        Returns:
            List of generation results
        """
        self.generation_logger.info(f"🚀 Starting batch generation of {num_visualizations} visualizations")
        
        if objective_types is None:
            objective_types = ["channel", "neuron", "center", "gabor"]
        
        results = []
        parent_candidates = []
        
        for i in range(num_visualizations):
            self.generation_logger.info(f"📸 Generating visualization {i+1}/{num_visualizations}")
            
            # Decide whether to mutate or create new
            if enable_mutation and parent_candidates and random.random() < 0.3:
                # Create mutation
                parent = random.choice(parent_candidates)
                mutated_params = mutate_objective_params(
                    parent["objective_params"],
                    mutation_rate=mutation_rate
                )
                
                result = self.generate_single_visualization(
                    objective_params=mutated_params,
                    notes=f"Mutation of {parent['visualization_id']}"
                )
                
                if result["success"]:
                    # Update the logged visualization with parent info
                    # (This would require extending the logger to update existing entries)
                    pass
                
            else:
                # Create new visualization
                objective_params_list = generate_random_objective_params(
                    self.model,
                    self.visualizable_layers,
                    objective_types=objective_types,
                    num_objectives=1,
                    input_size=self.image_size
                )
                
                result = self.generate_single_visualization(
                    objective_params=objective_params_list[0]
                )
            
            results.append(result)
            
            # Add successful results as potential parents for mutation
            if result["success"]:
                parent_candidates.append(result)
                
                # Keep only recent candidates to avoid memory issues
                if len(parent_candidates) > 20:
                    parent_candidates = parent_candidates[-10:]
        
        successful = sum(1 for r in results if r["success"])
        self.generation_logger.info(f"🎉 Batch complete: {successful}/{num_visualizations} successful")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about generated visualizations."""
        return self.logger.get_visualization_stats()
    
    def export_for_gradio(self, output_dir: str = "gradio_data") -> str:
        """Export data for Gradio interface."""
        return self.logger.export_for_gradio(output_dir)


# Convenience functions for backward compatibility
def enhanced_generate_for_model(model, layers, num_visualizations: int = 1):
    """
    Enhanced version of the original generate_for_model function.
    
    Args:
        model: PyTorch model
        layers: List of layer names
        num_visualizations: Number of visualizations to generate
    """
    generator = EnhancedVisualizationGenerator(
        model_name=type(model).__name__.lower(),
        enable_wrapping=True
    )
    
    return generator.generate_batch(num_visualizations=num_visualizations)


if __name__ == "__main__":
    print("🎨 Enhanced Visualization Generator")
    print("=" * 50)
    
    # Create generator
    generator = EnhancedVisualizationGenerator(
        model_name="inception_v1",
        image_size=256,  # Smaller for demo
        enable_wrapping=True
    )
    
    print("\\n📊 Generating test visualizations...")
    
    # Generate a few test visualizations
    results = generator.generate_batch(
        num_visualizations=3,
        objective_types=["channel", "neuron", "center"],
        enable_mutation=False
    )
    
    print("\\n📈 Generation Statistics:")
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\\n💾 Exporting for Gradio...")
    gradio_dir = generator.export_for_gradio()
    print(f"✅ Data exported to {gradio_dir}")
    
    print("\\n🎉 Demo complete!")