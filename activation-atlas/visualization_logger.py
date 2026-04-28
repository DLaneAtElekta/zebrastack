"""
Visualization logging system using pandas DataFrames.

This module provides comprehensive logging of visualization parameters,
metadata, and results in pandas DataFrames that can be serialized to CSV
for later analysis, navigation, and mutation.
"""

import pandas as pd
import datetime
import json
import os
from typing import Dict, Any, Optional
import hashlib


class VisualizationLogger:
    """
    Comprehensive logging system for Lucent visualizations.
    
    Stores all parameters, metadata, and results in a pandas DataFrame
    that can be serialized to CSV and used for later analysis.
    """
    
    def __init__(self, csv_filename: Optional[str] = None):
        """
        Initialize the visualization logger.
        
        Args:
            csv_filename: Path to CSV file for persistence
        """
        self.csv_filename = csv_filename or f"visualizations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize empty DataFrame with comprehensive schema
        self.df = pd.DataFrame(columns=[
            # Basic identification
            'visualization_id',
            'timestamp',
            'session_id',
            'image_filename',
            'model_name',
            
            # Objective parameters
            'objective_type',
            'layer_name',
            'channel_idx',
            'total_channels',
            
            # Spatial parameters (for neuron/center objectives)
            'spatial_x',
            'spatial_y',
            'spatial_width', 
            'spatial_height',
            'center_size',
            'offset_x',
            'offset_y',
            
            # Gabor parameters
            'gabor_sigma',
            'gabor_lambda',
            'gabor_theta',
            'gabor_psi',
            'gabor_gamma',
            'gabor_size',
            
            # Generation settings
            'image_size',
            'num_objectives',
            'sampled_channels',
            'transforms_used',
            'optimization_steps',
            'learning_rate',
            
            # Wrapping parameters
            'wrapping_enabled',
            'wrap_factor',
            'tile_count',
            
            # Quality/analysis metrics
            'generation_time_seconds',
            'final_loss',
            'user_rating',
            'notes',
            
            # JSON serialized full parameters
            'full_objective_params',
            'full_transform_params',
            'full_generation_config',
            
            # Parent/mutation tracking
            'parent_id',
            'mutation_generation',
            'mutation_type',
            'mutation_strength',
        ])
        
        # Try to load existing data
        self.load_from_csv()
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.visualization_counter = len(self.df)
    
    def generate_visualization_id(self, params: Dict[str, Any]) -> str:
        """Generate unique ID for a visualization based on parameters."""
        # Create hash from key parameters
        key_data = {
            'objective_type': params.get('objective_type'),
            'layer_name': params.get('layer_name'),
            'channel_idx': params.get('channel_idx'),
            'timestamp': datetime.datetime.now().isoformat(),
            'session': self.session_id,
            'counter': self.visualization_counter
        }
        
        hash_string = json.dumps(key_data, sort_keys=True)
        short_hash = hashlib.md5(hash_string.encode()).hexdigest()[:8]
        
        return f"vis_{self.session_id}_{self.visualization_counter:04d}_{short_hash}"
    
    def log_visualization(
        self,
        objective_params: Dict[str, Any],
        image_filename: str,
        model_name: str,
        generation_config: Dict[str, Any],
        transform_params: Optional[Dict[str, Any]] = None,
        generation_time: Optional[float] = None,
        final_loss: Optional[float] = None,
        parent_id: Optional[str] = None,
        mutation_info: Optional[Dict[str, Any]] = None,
        user_rating: Optional[float] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Log a complete visualization with all parameters.
        
        Args:
            objective_params: Dictionary from objective generators
            image_filename: Path to generated image
            model_name: Name of the model used
            generation_config: Configuration used for generation
            transform_params: Transform parameters used
            generation_time: Time taken to generate (seconds)
            final_loss: Final optimization loss value
            parent_id: ID of parent visualization (for mutations)
            mutation_info: Information about mutation applied
            user_rating: User rating/score for the visualization
            notes: Additional notes
            
        Returns:
            Generated visualization ID
        """
        vis_id = self.generate_visualization_id(objective_params)
        self.visualization_counter += 1
        
        # Extract parameters with safe defaults
        obj_type = objective_params.get('objective_type', 'unknown')
        layer_name = objective_params.get('layer_name', 'unknown')
        channel_idx = objective_params.get('channel_idx', 0)
        total_channels = objective_params.get('total_channels', 0)
        
        # Extract spatial parameters
        spatial = objective_params.get('spatial_params', {})
        spatial_x = spatial.get('x', None)
        spatial_y = spatial.get('y', None)
        spatial_width = spatial.get('width', None)
        spatial_height = spatial.get('height', None)
        center_size = spatial.get('center_size', None)
        offset_x = spatial.get('offset_x', None)
        offset_y = spatial.get('offset_y', None)
        
        # Extract Gabor parameters
        gabor = objective_params.get('gabor_params', {})
        gabor_sigma = gabor.get('sigma', None)
        gabor_lambda = gabor.get('lambda_freq', None)
        gabor_theta = gabor.get('theta', None)
        gabor_psi = gabor.get('psi', None)
        gabor_gamma = gabor.get('gamma', None)
        gabor_size = gabor.get('size', None)
        
        # Extract generation config
        image_size = generation_config.get('image_size', None)
        num_objectives = generation_config.get('num_objectives', 1)
        sampled_channels = generation_config.get('sampled_channels', 1)
        optimization_steps = generation_config.get('optimization_steps', None)
        learning_rate = generation_config.get('learning_rate', None)
        
        # Extract transform information
        transforms_used = None
        wrapping_enabled = False
        wrap_factor = None
        tile_count = None
        
        if transform_params:
            transforms_used = json.dumps(transform_params.get('transform_names', []))
            wrapping_enabled = transform_params.get('wrapping_enabled', False)
            wrap_factor = transform_params.get('wrap_factor', None)
            tile_count = transform_params.get('tile_count', None)
        
        # Extract mutation information
        mutation_generation = 0
        mutation_type = None
        mutation_strength = None
        
        if mutation_info:
            mutation_generation = mutation_info.get('generation', 0)
            mutation_type = mutation_info.get('type', None)
            mutation_strength = mutation_info.get('strength', None)
        
        # Create new row
        new_row = pd.DataFrame([{
            'visualization_id': vis_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': self.session_id,
            'image_filename': image_filename,
            'model_name': model_name,
            
            # Objective parameters
            'objective_type': obj_type,
            'layer_name': layer_name,
            'channel_idx': channel_idx,
            'total_channels': total_channels,
            
            # Spatial parameters
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_width': spatial_width,
            'spatial_height': spatial_height,
            'center_size': center_size,
            'offset_x': offset_x,
            'offset_y': offset_y,
            
            # Gabor parameters
            'gabor_sigma': gabor_sigma,
            'gabor_lambda': gabor_lambda,
            'gabor_theta': gabor_theta,
            'gabor_psi': gabor_psi,
            'gabor_gamma': gabor_gamma,
            'gabor_size': gabor_size,
            
            # Generation settings
            'image_size': image_size,
            'num_objectives': num_objectives,
            'sampled_channels': sampled_channels,
            'transforms_used': transforms_used,
            'optimization_steps': optimization_steps,
            'learning_rate': learning_rate,
            
            # Wrapping parameters
            'wrapping_enabled': wrapping_enabled,
            'wrap_factor': wrap_factor,
            'tile_count': tile_count,
            
            # Quality/analysis metrics
            'generation_time_seconds': generation_time,
            'final_loss': final_loss,
            'user_rating': user_rating,
            'notes': notes,
            
            # JSON serialized full parameters
            'full_objective_params': json.dumps(objective_params),
            'full_transform_params': json.dumps(transform_params) if transform_params else None,
            'full_generation_config': json.dumps(generation_config),
            
            # Parent/mutation tracking
            'parent_id': parent_id,
            'mutation_generation': mutation_generation,
            'mutation_type': mutation_type,
            'mutation_strength': mutation_strength,
        }])
        
        # Add to DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Auto-save if CSV filename is set
        if self.csv_filename:
            self.save_to_csv()
        
        return vis_id
    
    def save_to_csv(self, filename: Optional[str] = None) -> None:
        """Save DataFrame to CSV file."""
        save_path = filename or self.csv_filename
        self.df.to_csv(save_path, index=False)
        print(f"💾 Saved {len(self.df)} visualizations to {save_path}")
    
    def load_from_csv(self, filename: Optional[str] = None) -> None:
        """Load DataFrame from CSV file if it exists."""
        load_path = filename or self.csv_filename
        if os.path.exists(load_path):
            try:
                self.df = pd.read_csv(load_path)
                print(f"📂 Loaded {len(self.df)} visualizations from {load_path}")
            except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
                print(f"⚠️  Warning: Could not load {load_path}: {e}")
    
    def get_visualizations_by_layer(self, layer_name: str) -> pd.DataFrame:
        """Get all visualizations for a specific layer."""
        return self.df[self.df['layer_name'] == layer_name].copy()
    
    def get_visualizations_by_type(self, objective_type: str) -> pd.DataFrame:
        """Get all visualizations of a specific objective type."""
        return self.df[self.df['objective_type'] == objective_type].copy()
    
    def get_best_visualizations(self, n: int = 10, sort_by: str = 'user_rating') -> pd.DataFrame:
        """Get top N visualizations sorted by rating or other metric."""
        valid_data = self.df.dropna(subset=[sort_by])
        return valid_data.nlargest(n, sort_by)
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get summary statistics about logged visualizations."""
        stats = {
            'total_visualizations': len(self.df),
            'unique_layers': self.df['layer_name'].nunique(),
            'objective_type_counts': self.df['objective_type'].value_counts().to_dict(),
            'average_generation_time': self.df['generation_time_seconds'].mean(),
            'sessions': self.df['session_id'].nunique(),
        }
        
        if 'user_rating' in self.df.columns and not self.df['user_rating'].isna().all():
            stats['average_rating'] = self.df['user_rating'].mean()
            stats['top_rated_layer'] = self.df.loc[self.df['user_rating'].idxmax(), 'layer_name']
        
        return stats
    
    def find_similar_visualizations(
        self, 
        target_params: Dict[str, Any], 
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> pd.DataFrame:
        """
        Find visualizations similar to given parameters.
        
        Args:
            target_params: Parameters to find similar visualizations for
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with similar visualizations and similarity scores
        """
        # Simple similarity based on layer and objective type
        similar = self.df[
            (self.df['layer_name'] == target_params.get('layer_name')) &
            (self.df['objective_type'] == target_params.get('objective_type'))
        ].copy()
        
        if len(similar) == 0:
            return pd.DataFrame()
        
        # Calculate channel similarity
        target_channel = target_params.get('channel_idx', 0)
        similar['channel_similarity'] = 1.0 - (
            abs(similar['channel_idx'] - target_channel) / similar['total_channels']
        ).clip(0, 1)
        
        # Filter by similarity threshold and sort
        similar = similar[similar['channel_similarity'] >= similarity_threshold]
        similar = similar.nlargest(max_results, 'channel_similarity')
        
        return similar
    
    def export_for_gradio(self, output_dir: str = "gradio_data") -> str:
        """
        Export data in format suitable for Gradio interface.
        
        Args:
            output_dir: Directory to save exported data
            
        Returns:
            Path to exported data directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main CSV
        main_csv = os.path.join(output_dir, "visualizations.csv")
        self.save_to_csv(main_csv)
        
        # Create layer summary
        layer_summary = self.df.groupby('layer_name').agg({
            'visualization_id': 'count',
            'objective_type': lambda x: list(x.unique()),
            'user_rating': 'mean',
            'generation_time_seconds': 'mean'
        }).reset_index()
        
        layer_summary.columns = ['layer_name', 'count', 'objective_types', 'avg_rating', 'avg_time']
        layer_summary.to_csv(os.path.join(output_dir, "layer_summary.csv"), index=False)
        
        # Create objective type summary  
        obj_summary = self.df.groupby('objective_type').agg({
            'visualization_id': 'count',
            'user_rating': 'mean',
            'generation_time_seconds': 'mean'
        }).reset_index()
        
        obj_summary.columns = ['objective_type', 'count', 'avg_rating', 'avg_time']
        obj_summary.to_csv(os.path.join(output_dir, "objective_summary.csv"), index=False)
        
        print(f"📤 Exported Gradio data to {output_dir}")
        return output_dir


# Global logger instance
_global_logger = None

def get_logger(csv_filename: Optional[str] = None) -> VisualizationLogger:
    """Get or create global visualization logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = VisualizationLogger(csv_filename)
    return _global_logger


def log_visualization_simple(
    objective_params: Dict[str, Any],
    image_filename: str,
    model_name: str = "inception_v1",
    generation_time: Optional[float] = None,
    **kwargs
) -> str:
    """
    Simple function to log a visualization with minimal parameters.
    
    Args:
        objective_params: Parameters from objective generators
        image_filename: Path to generated image
        model_name: Name of model used
        generation_time: Time taken to generate
        **kwargs: Additional parameters
        
    Returns:
        Generated visualization ID
    """
    logger = get_logger()
    
    generation_config = {
        'image_size': kwargs.get('image_size', 384),
        'sampled_channels': kwargs.get('sampled_channels', 1),
    }
    
    return logger.log_visualization(
        objective_params=objective_params,
        image_filename=image_filename,
        model_name=model_name,
        generation_config=generation_config,
        generation_time=generation_time,
        **kwargs
    )


if __name__ == "__main__":
    print("📊 Visualization Logging System")
    print("=" * 40)
    print()
    print("Features:")
    print("• Comprehensive parameter logging")
    print("• CSV serialization")
    print("• Similarity search")
    print("• Gradio export")
    print("• Mutation tracking")
    print()
    print("Example usage:")
    print("logger = VisualizationLogger('my_visualizations.csv')")
    print("vis_id = logger.log_visualization(params, 'image.png', 'inception_v1', config)")
    print("stats = logger.get_visualization_stats()")
    print("similar = logger.find_similar_visualizations(params)")