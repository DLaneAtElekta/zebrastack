"""
Simplified Gradio frontend for neural network visualization exploration.

A streamlined interface for browsing visualizations and generating new ones.
"""

import gradio as gr
import pandas as pd
import os
from typing import Optional, Tuple
from PIL import Image

from visualization_logger import get_logger
from enhanced_visualization_generator import EnhancedVisualizationGenerator


class SimpleVisualizationExplorer:
    """Simplified visualization explorer for Gradio interface."""
    
    def __init__(self, csv_filename: Optional[str] = None):
        self.logger = get_logger(csv_filename)
        self.generator = EnhancedVisualizationGenerator(
            model_name="inception_v1",
            image_size=256,  # Smaller for faster generation
            csv_filename=csv_filename
        )
        
        print("🎨 Simple Visualization Explorer initialized")
        print(f"📊 Database: {len(self.logger.df)} visualizations")
        print(f"🧠 Model: {len(self.generator.visualizable_layers)} layers available")
    
    def get_layer_choices(self):
        """Get list of available layers for dropdown."""
        return self.generator.visualizable_layers[:30]  # First 30 layers
    
    def generate_visualization(
        self,
        layer_name: str,
        objective_type: str,
        channel_idx: int,
        use_wrapping: bool,
        wrap_factor: float
    ) -> Tuple[str, str]:
        """Generate a new visualization."""
        try:
            from objective_generators import (
                generate_channel_objective_params,
                generate_neuron_objective_params,
                generate_center_objective_params,
                generate_gabor_objective_params
            )
            
            # Generate parameters
            if objective_type == "channel":
                params = generate_channel_objective_params(
                    self.generator.model, [layer_name], layer_name
                )
            elif objective_type == "neuron":
                params = generate_neuron_objective_params(
                    self.generator.model, [layer_name], layer_name
                )
            elif objective_type == "center":
                params = generate_center_objective_params(
                    self.generator.model, [layer_name], layer_name
                )
            elif objective_type == "gabor":
                params = generate_gabor_objective_params(
                    self.generator.model, [layer_name], layer_name
                )
            else:
                return "❌ Unknown objective type", ""
            
            # Override channel
            params['channel_idx'] = channel_idx
            
            # Setup transforms
            transforms_config = {
                "use_padding": True,
                "use_jitter": True,
                "use_scale": True,
                "use_rotate": True,
                "use_wrapping": use_wrapping,
                "wrap_factor": wrap_factor if use_wrapping else None,
            }
            
            # Generate
            result = self.generator.generate_single_visualization(
                objective_params=params,
                transforms_config=transforms_config,
                save_image=True,
                notes="Generated via Gradio interface"
            )
            
            if result["success"]:
                return result["filename"], f"✅ Generated {result['visualization_id']}"
            else:
                return "", f"❌ Failed: {result['error']}"
                
        except (ImportError, AttributeError, ValueError, KeyError, TypeError) as e:
            return "", f"❌ Error: {str(e)}"
    
    def get_recent_visualizations(self, n: int = 10) -> pd.DataFrame:
        """Get recent visualizations."""
        if len(self.logger.df) == 0:
            return pd.DataFrame()
        
        recent = self.logger.df.sort_values('timestamp', ascending=False).head(n)
        
        # Select key columns
        display_cols = [
            'visualization_id', 'timestamp', 'layer_name', 'objective_type',
            'channel_idx', 'user_rating', 'image_filename'
        ]
        available_cols = [col for col in display_cols if col in recent.columns]
        
        return recent[available_cols]
    
    def load_image_from_path(self, image_path: str) -> Optional[Image.Image]:
        """Load image from file path."""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            return Image.open(image_path)
        except (IOError, OSError):
            return None
    
    def rate_visualization(self, viz_id: str, rating: float) -> str:
        """Rate a visualization."""
        try:
            mask = self.logger.df['visualization_id'] == viz_id
            if not mask.any():
                return "❌ Visualization not found"
            
            self.logger.df.loc[mask, 'user_rating'] = rating
            self.logger.save_to_csv()
            
            return f"✅ Rated {viz_id}: {rating}/5"
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return f"❌ Error: {str(e)}"


def create_simple_interface(csv_filename: Optional[str] = None) -> gr.Blocks:
    """Create a simplified Gradio interface."""
    explorer = SimpleVisualizationExplorer(csv_filename)
    
    with gr.Blocks(title="🎨 Neural Network Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Neural Network Visualization Explorer")
        gr.Markdown("Generate and explore neural network feature visualizations")
        
        with gr.Tabs():
            # Generation Tab
            with gr.TabItem("🎨 Generate"):
                with gr.Row():
                    with gr.Column():
                        layer_dropdown = gr.Dropdown(
                            choices=explorer.get_layer_choices(),
                            value=explorer.get_layer_choices()[0] if explorer.get_layer_choices() else None,
                            label="🧠 Layer"
                        )
                        objective_dropdown = gr.Dropdown(
                            choices=["channel", "neuron", "center", "gabor"],
                            value="channel",
                            label="🎯 Objective Type"
                        )
                        channel_number = gr.Number(
                            value=42,
                            label="📡 Channel Index",
                            precision=0
                        )
                        
                        with gr.Group():
                            gr.Markdown("### 🌊 Wrapping Options")
                            wrapping_checkbox = gr.Checkbox(
                                value=True,
                                label="Enable Wrapping"
                            )
                            wrap_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                label="Wrap Factor"
                            )
                        
                        generate_button = gr.Button("🎨 Generate", variant="primary")
                    
                    with gr.Column():
                        result_image = gr.Image(label="🖼️ Generated Visualization")
                        result_text = gr.Textbox(label="📝 Result")
                
                # Connect generation
                generate_button.click(
                    explorer.generate_visualization,
                    inputs=[
                        layer_dropdown,
                        objective_dropdown,
                        channel_number,
                        wrapping_checkbox,
                        wrap_slider
                    ],
                    outputs=[result_image, result_text]
                )
            
            # Browse Tab
            with gr.TabItem("📚 Browse"):
                with gr.Row():
                    refresh_button = gr.Button("🔄 Refresh")
                
                recent_df = gr.Dataframe(
                    value=explorer.get_recent_visualizations(),
                    label="📋 Recent Visualizations",
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column():
                        viz_id_input = gr.Textbox(
                            label="🆔 Visualization ID",
                            placeholder="Enter ID to view/rate"
                        )
                        rating_input = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=0.5,
                            value=3,
                            label="⭐ Rating (1-5)"
                        )
                        rate_button = gr.Button("💾 Save Rating")
                        rating_result = gr.Textbox(label="Rating Result")
                    
                    with gr.Column():
                        browse_image = gr.Image(label="🖼️ Visualization")
                
                # Connect browsing functions
                refresh_button.click(
                    explorer.get_recent_visualizations,
                    outputs=recent_df
                )
                
                def load_viz_image(viz_id):
                    if not viz_id:
                        return None
                    
                    viz_row = explorer.logger.df[
                        explorer.logger.df['visualization_id'] == viz_id
                    ]
                    if len(viz_row) == 0:
                        return None
                    
                    image_path = viz_row.iloc[0]['image_filename']
                    return explorer.load_image_from_path(image_path)
                
                viz_id_input.change(
                    load_viz_image,
                    inputs=viz_id_input,
                    outputs=browse_image
                )
                
                rate_button.click(
                    explorer.rate_visualization,
                    inputs=[viz_id_input, rating_input],
                    outputs=rating_result
                )
            
            # Stats Tab
            with gr.TabItem("📊 Statistics"):
                stats_display = gr.JSON(
                    value=explorer.logger.get_visualization_stats(),
                    label="📈 Database Statistics"
                )
                
                refresh_stats_button = gr.Button("🔄 Refresh Stats")
                refresh_stats_button.click(
                    explorer.logger.get_visualization_stats,
                    outputs=stats_display
                )
    
    return demo


def launch_simple_explorer(
    csv_filename: Optional[str] = None,
    share: bool = False,
    port: int = 7860
) -> None:
    """Launch the simplified Gradio interface."""
    print("🚀 Launching Simple Visualization Explorer...")
    
    interface = create_simple_interface(csv_filename)
    
    interface.launch(
        share=share,
        server_port=port,
        show_error=True
    )


if __name__ == "__main__":
    launch_simple_explorer(
        csv_filename=None,
        share=False,
        port=7860
    )