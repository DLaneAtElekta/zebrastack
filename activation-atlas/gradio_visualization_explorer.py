"""
Gradio frontend for browsing and exploring neural network visualizations.

This module provides an interactive web interface for:
- Browsing Inception model architecture layers
- Viewing stored visualizations from CSV database
- Filtering and searching visualizations
- Rating and annotating visualizations
- Generating new visualizations interactively
- Creating mutation children from existing visualizations
"""

import gradio as gr
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import json

from visualization_logger import VisualizationLogger, get_logger
from enhanced_visualization_generator import EnhancedVisualizationGenerator
from objective_generators import mutate_objective_params
from lucent_layer_utils import get_visualizable_layers
from lucent.modelzoo import inceptionv1


class VisualizationExplorer:
    """
    Interactive exploration interface for neural network visualizations.
    """
    
    def __init__(self, csv_filename: Optional[str] = None):
        """
        Initialize the visualization explorer.
        
        Args:
            csv_filename: Path to visualization CSV database
        """
        self.logger = get_logger(csv_filename)
        self.generator = EnhancedVisualizationGenerator(
            model_name="inception_v1",
            image_size=384,
            csv_filename=csv_filename
        )
        
        # Load model for layer information
        self.model = inceptionv1(pretrained=True)
        self.layers = get_visualizable_layers(self.model)
        
        # Cache for loaded images
        self.image_cache = {}
        
        print(f"🎨 Visualization Explorer initialized")
        print(f"📊 Database: {len(self.logger.df)} visualizations")
        print(f"🧠 Model: {len(self.layers)} layers available")
    
    def get_layer_info(self) -> pd.DataFrame:
        """Get summary information about available layers."""
        if len(self.logger.df) == 0:
            # Return basic layer info if no visualizations exist
            layer_data = []
            for layer in self.layers[:20]:  # Show first 20 layers
                layer_data.append({
                    'layer_name': layer,
                    'visualizations_count': 0,
                    'avg_rating': 0.0,
                    'objective_types': []
                })
            return pd.DataFrame(layer_data)
        
        # Get statistics from existing visualizations
        layer_stats = self.logger.df.groupby('layer_name').agg({
            'visualization_id': 'count',
            'objective_type': lambda x: list(x.unique()),
            'user_rating': 'mean',
            'generation_time_seconds': 'mean'
        }).reset_index()
        
        layer_stats.columns = ['layer_name', 'visualizations_count', 'objective_types', 'avg_rating', 'avg_time']
        layer_stats['avg_rating'] = layer_stats['avg_rating'].fillna(0.0)
        
        return layer_stats.sort_values('visualizations_count', ascending=False)
    
    def get_visualizations_for_layer(self, layer_name: str) -> pd.DataFrame:
        """Get all visualizations for a specific layer."""
        if not layer_name or layer_name == "Select a layer":
            return pd.DataFrame()
        
        layer_viz = self.logger.get_visualizations_by_layer(layer_name)
        if len(layer_viz) == 0:
            return pd.DataFrame()
        
        # Select key columns for display
        display_cols = [
            'visualization_id', 'timestamp', 'objective_type', 
            'channel_idx', 'user_rating', 'generation_time_seconds',
            'image_filename'
        ]
        
        available_cols = [col for col in display_cols if col in layer_viz.columns]
        return layer_viz[available_cols].sort_values('timestamp', ascending=False)
    
    def load_visualization_image(self, image_filename: str) -> Optional[Image.Image]:
        """Load and cache visualization image."""
        if not image_filename or not os.path.exists(image_filename):
            return None
        
        if image_filename not in self.image_cache:
            try:
                self.image_cache[image_filename] = Image.open(image_filename)
            except Exception as e:
                print(f"Error loading image {image_filename}: {e}")
                return None
        
        return self.image_cache[image_filename]
    
    def generate_new_visualization(
        self,
        layer_name: str,
        objective_type: str,
        channel_idx: Optional[int] = None,
        use_wrapping: bool = True,
        wrap_factor: float = 0.3
    ) -> Tuple[str, str, Image.Image]:
        """
        Generate a new visualization with specified parameters.
        
        Returns:
            Tuple of (success_message, visualization_id, image)
        """
        try:
            from objective_generators import (
                generate_channel_objective_params,
                generate_neuron_objective_params,
                generate_center_objective_params,
                generate_gabor_objective_params
            )
            
            # Generate parameters based on objective type
            if objective_type == "channel":
                params = generate_channel_objective_params(self.model, [layer_name], layer_name)
            elif objective_type == "neuron":
                params = generate_neuron_objective_params(self.model, [layer_name], layer_name)
            elif objective_type == "center":
                params = generate_center_objective_params(self.model, [layer_name], layer_name)
            elif objective_type == "gabor":
                params = generate_gabor_objective_params(self.model, [layer_name], layer_name)
            else:
                return "Unknown objective type", "", None
            
            # Override channel if specified
            if channel_idx is not None:
                params['channel_idx'] = channel_idx
            
            # Setup transform configuration
            transforms_config = {
                "use_padding": True,
                "use_jitter": True,
                "use_scale": True,
                "use_rotate": True,
                "use_wrapping": use_wrapping,
                "wrap_factor": wrap_factor if use_wrapping else None,
            }
            
            # Generate visualization
            result = self.generator.generate_single_visualization(
                objective_params=params,
                transforms_config=transforms_config,
                save_image=True,
                notes=f"Generated via Gradio interface"
            )
            
            if result["success"]:
                image = self.load_visualization_image(result["filename"])
                return (
                    f"✅ Successfully generated visualization {result['visualization_id']}",
                    result['visualization_id'],
                    image
                )
            else:
                return f"❌ Generation failed: {result['error']}", "", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", "", None
    
    def create_mutation(
        self,
        parent_viz_id: str,
        mutation_strength: float = 0.3,
        mutation_rate: float = 0.4
    ) -> Tuple[str, str, Image.Image]:
        """
        Create a mutated version of an existing visualization.
        
        Returns:
            Tuple of (success_message, new_visualization_id, image)
        """
        try:
            # Find parent visualization
            parent_row = self.logger.df[self.logger.df['visualization_id'] == parent_viz_id]
            if len(parent_row) == 0:
                return "❌ Parent visualization not found", "", None
            
            # Parse parent parameters
            parent_params_json = parent_row.iloc[0]['full_objective_params']
            parent_params = json.loads(parent_params_json)
            
            # Create mutation
            mutated_params = mutate_objective_params(
                parent_params,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength
            )
            
            # Generate mutated visualization
            result = self.generator.generate_single_visualization(
                objective_params=mutated_params,
                save_image=True,
                notes=f"Mutation of {parent_viz_id} (strength={mutation_strength}, rate={mutation_rate})"
            )
            
            if result["success"]:
                image = self.load_visualization_image(result["filename"])
                return (
                    f"✅ Created mutation {result['visualization_id']} from {parent_viz_id}",
                    result['visualization_id'],
                    image
                )
            else:
                return f"❌ Mutation failed: {result['error']}", "", None
                
        except Exception as e:
            return f"❌ Error creating mutation: {str(e)}", "", None
    
    def rate_visualization(self, viz_id: str, rating: float, notes: str = "") -> str:
        """
        Rate a visualization and update the database.
        
        Args:
            viz_id: Visualization ID to rate
            rating: Rating value (1-5)
            notes: Optional notes about the rating
            
        Returns:
            Success/error message
        """
        try:
            # Update the rating in the DataFrame
            mask = self.logger.df['visualization_id'] == viz_id
            if not mask.any():
                return "❌ Visualization not found"
            
            self.logger.df.loc[mask, 'user_rating'] = rating
            if notes:
                existing_notes = self.logger.df.loc[mask, 'notes'].iloc[0]
                if pd.isna(existing_notes):
                    new_notes = notes
                else:
                    new_notes = f"{existing_notes}; Rating: {notes}"
                self.logger.df.loc[mask, 'notes'] = new_notes
            
            # Save to CSV
            self.logger.save_to_csv()
            
            return f"✅ Rated visualization {viz_id}: {rating}/5"
            
        except Exception as e:
            return f"❌ Error rating visualization: {str(e)}"
    
    def search_visualizations(
        self,
        search_terms: str,
        objective_type: str = "All",
        min_rating: float = 0.0,
        max_results: int = 20
    ) -> pd.DataFrame:
        """
        Search visualizations based on various criteria.
        
        Args:
            search_terms: Search terms for layer names or notes
            objective_type: Filter by objective type
            min_rating: Minimum rating filter
            max_results: Maximum number of results
            
        Returns:
            Filtered DataFrame of visualizations
        """
        df = self.logger.df.copy()
        
        # Filter by search terms
        if search_terms:
            mask = (
                df['layer_name'].str.contains(search_terms, case=False, na=False) |
                df['notes'].str.contains(search_terms, case=False, na=False)
            )
            df = df[mask]
        
        # Filter by objective type
        if objective_type != "All":
            df = df[df['objective_type'] == objective_type]
        
        # Filter by rating
        if min_rating > 0:
            df = df[df['user_rating'] >= min_rating]
        
        # Select display columns
        display_cols = [
            'visualization_id', 'timestamp', 'layer_name', 'objective_type',
            'channel_idx', 'user_rating', 'image_filename'
        ]
        available_cols = [col for col in display_cols if col in df.columns]
        
        result = df[available_cols].sort_values('timestamp', ascending=False)
        return result.head(max_results)


def create_gradio_interface(csv_filename: Optional[str] = None) -> gr.Blocks:
    """
    Create the main Gradio interface for visualization exploration.
    
    Args:
        csv_filename: Path to visualization CSV database
        
    Returns:
        Gradio Blocks interface
    """
    explorer = VisualizationExplorer(csv_filename)
    
    with gr.Blocks(title="🎨 Neural Network Visualization Explorer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Neural Network Visualization Explorer")
        gr.Markdown("Explore, generate, and analyze neural network feature visualizations")
        
        # Statistics display
        with gr.Row():
            with gr.Column(scale=1):
                stats_display = gr.JSON(
                    value=explorer.logger.get_visualization_stats(),
                    label="📊 Database Statistics"
                )
            
            with gr.Column(scale=1):
                refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
                refresh_stats_btn.click(
                    lambda: explorer.logger.get_visualization_stats(),
                    outputs=stats_display
                )
        
        with gr.Tabs():
            # Layer Browser Tab
            with gr.TabItem("🧠 Layer Browser"):
                with gr.Row():
                    with gr.Column(scale=2):
                        layer_info_df = gr.Dataframe(
                            value=explorer.get_layer_info(),
                            label="📋 Available Layers",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        selected_layer = gr.Dropdown(
                            choices=["Select a layer"] + explorer.layers[:50],
                            value="Select a layer",
                            label="🎯 Select Layer"
                        )
                        
                        show_layer_viz_btn = gr.Button("👁️ Show Visualizations")
                
                # Layer visualizations display
                layer_viz_df = gr.Dataframe(
                    label="🖼️ Visualizations for Selected Layer",
                    interactive=False
                )
                
                with gr.Row():
                    selected_viz_id = gr.Textbox(label="🆔 Visualization ID", placeholder="Enter or select visualization ID")
                    load_viz_btn = gr.Button("🔍 Load Visualization")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        viz_image = gr.Image(label="🖼️ Visualization", type="pil")
                    
                    with gr.Column(scale=1):
                        viz_details = gr.JSON(label="📋 Visualization Details")
                        
                        # Rating system
                        with gr.Group():
                            gr.Markdown("### ⭐ Rate this Visualization")
                            rating_slider = gr.Slider(
                                minimum=1, maximum=5, step=0.5, value=3,
                                label="Rating (1-5)"
                            )
                            rating_notes = gr.Textbox(
                                label="Notes",
                                placeholder="Optional notes about this visualization"
                            )
                            rate_btn = gr.Button("💾 Save Rating")
                            rating_result = gr.Textbox(label="Result")
            
            # Generator Tab
            with gr.TabItem("🎨 Generate New"):
                with gr.Row():
                    with gr.Column():
                        gen_layer = gr.Dropdown(
                            choices=explorer.layers[:30],
                            value=explorer.layers[0] if explorer.layers else None,
                            label="🧠 Layer"
                        )
                        gen_objective = gr.Dropdown(
                            choices=["channel", "neuron", "center", "gabor"],
                            value="channel",
                            label="🎯 Objective Type"
                        )
                        gen_channel = gr.Number(
                            value=42,
                            label="📡 Channel Index (optional)",
                            precision=0
                        )
                        
                        with gr.Group():
                            gr.Markdown("### 🌊 Wrapping Options")
                            use_wrapping = gr.Checkbox(value=True, label="Enable Wrapping")
                            wrap_factor = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.3,
                                label="Wrap Factor"
                            )
                        
                        generate_btn = gr.Button("🎨 Generate Visualization", variant="primary")
                    
                    with gr.Column():
                        gen_result = gr.Textbox(label="📝 Generation Result")
                        gen_viz_id = gr.Textbox(label="🆔 New Visualization ID")
                        gen_image = gr.Image(label="🖼️ Generated Visualization", type="pil")
            
            # Mutation Tab
            with gr.TabItem("🧬 Create Mutations"):
                with gr.Row():
                    with gr.Column():
                        parent_id = gr.Textbox(
                            label="👨‍👩‍👧‍👦 Parent Visualization ID",
                            placeholder="Enter visualization ID to mutate"
                        )
                        mutation_strength = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.3,
                            label="🎛️ Mutation Strength"
                        )
                        mutation_rate = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.4,
                            label="🎲 Mutation Rate"
                        )
                        mutate_btn = gr.Button("🧬 Create Mutation", variant="primary")
                    
                    with gr.Column():
                        mut_result = gr.Textbox(label="📝 Mutation Result")
                        mut_viz_id = gr.Textbox(label="🆔 New Mutation ID")
                        mut_image = gr.Image(label="🖼️ Mutated Visualization", type="pil")
            
            # Search Tab
            with gr.TabItem("🔍 Search & Filter"):
                with gr.Row():
                    with gr.Column():
                        search_terms = gr.Textbox(
                            label="🔍 Search Terms",
                            placeholder="Search layer names, notes, etc."
                        )
                        search_obj_type = gr.Dropdown(
                            choices=["All", "channel", "neuron", "center", "gabor"],
                            value="All",
                            label="🎯 Objective Type Filter"
                        )
                        min_rating = gr.Slider(
                            minimum=0, maximum=5, value=0,
                            label="⭐ Minimum Rating"
                        )
                        max_results = gr.Slider(
                            minimum=5, maximum=100, value=20,
                            label="📊 Max Results"
                        )
                        search_btn = gr.Button("🔍 Search")
                
                search_results = gr.Dataframe(
                    label="🔍 Search Results",
                    interactive=False
                )
        
        # Event handlers
        def show_layer_visualizations(layer_name):
            return explorer.get_visualizations_for_layer(layer_name)
        
        def load_visualization_details(viz_id):
            if not viz_id:
                return None, {}
            
            # Find visualization
            viz_row = explorer.logger.df[explorer.logger.df['visualization_id'] == viz_id]
            if len(viz_row) == 0:
                return None, {"error": "Visualization not found"}
            
            viz_data = viz_row.iloc[0]
            image = explorer.load_visualization_image(viz_data['image_filename'])
            
            details = {
                "visualization_id": viz_data['visualization_id'],
                "layer_name": viz_data['layer_name'],
                "objective_type": viz_data['objective_type'],
                "channel_idx": viz_data['channel_idx'],
                "timestamp": viz_data['timestamp'],
                "user_rating": viz_data.get('user_rating', 'Not rated'),
                "generation_time": f"{viz_data.get('generation_time_seconds', 0):.2f}s",
                "image_filename": viz_data['image_filename']
            }
            
            return image, details
        
        # Connect event handlers
        show_layer_viz_btn.click(
            show_layer_visualizations,
            inputs=selected_layer,
            outputs=layer_viz_df
        )
        
        load_viz_btn.click(
            load_visualization_details,
            inputs=selected_viz_id,
            outputs=[viz_image, viz_details]
        )
        
        generate_btn.click(
            explorer.generate_new_visualization,
            inputs=[gen_layer, gen_objective, gen_channel, use_wrapping, wrap_factor],
            outputs=[gen_result, gen_viz_id, gen_image]
        )
        
        mutate_btn.click(
            explorer.create_mutation,
            inputs=[parent_id, mutation_strength, mutation_rate],
            outputs=[mut_result, mut_viz_id, mut_image]
        )
        
        rate_btn.click(
            explorer.rate_visualization,
            inputs=[selected_viz_id, rating_slider, rating_notes],
            outputs=rating_result
        )
        
        search_btn.click(
            explorer.search_visualizations,
            inputs=[search_terms, search_obj_type, min_rating, max_results],
            outputs=search_results
        )
    
    return interface


def launch_visualization_explorer(
    csv_filename: Optional[str] = None,
    share: bool = False,
    port: int = 7860
) -> None:
    """
    Launch the Gradio visualization explorer interface.
    
    Args:
        csv_filename: Path to visualization CSV database
        share: Whether to create a public link
        port: Port to run the server on
    """
    print("🚀 Launching Neural Network Visualization Explorer...")
    
    interface = create_gradio_interface(csv_filename)
    
    interface.launch(
        share=share,
        server_port=port,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    # Launch the interface
    launch_visualization_explorer(
        csv_filename=None,  # Will auto-create
        share=False,        # Set to True for public link
        port=7860
    )