"""
Visualization mutation and scoring system.

This module provides advanced functionality for:
- Creating mutant children from parent visualizations
- Scoring and rating visualizations
- Evolutionary generation of better visualizations
- Population-based improvement algorithms
"""

import random
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass

from objective_generators import mutate_objective_params
from visualization_logger import get_logger
from enhanced_visualization_generator import EnhancedVisualizationGenerator


@dataclass
class MutationConfig:
    """Configuration for mutation operations."""
    mutation_rate: float = 0.3
    mutation_strength: float = 0.2
    crossover_rate: float = 0.1
    elite_percentage: float = 0.2
    population_size: int = 20
    generations: int = 10


@dataclass
class VisualizationScore:
    """Comprehensive scoring for a visualization."""
    user_rating: float = 0.0
    generation_time: float = 0.0
    novelty_score: float = 0.0
    complexity_score: float = 0.0
    fitness_score: float = 0.0


class VisualizationMutator:
    """
    Advanced mutation system for creating and evolving visualizations.
    """
    
    def __init__(self, csv_filename: Optional[str] = None):
        """
        Initialize the mutation system.
        
        Args:
            csv_filename: CSV file for logging visualizations
        """
        self.logger = get_logger(csv_filename)
        self.generator = EnhancedVisualizationGenerator(
            model_name="inception_v1",
            image_size=256,
            csv_filename=csv_filename
        )
        
        print("🧬 Visualization Mutator initialized")
        print(f"📊 Current database: {len(self.logger.df)} visualizations")
    
    def calculate_visualization_score(self, viz_id: str) -> VisualizationScore:
        """
        Calculate comprehensive score for a visualization.
        
        Args:
            viz_id: Visualization ID to score
            
        Returns:
            VisualizationScore object with all score components
        """
        # Find visualization in database
        viz_row = self.logger.df[self.logger.df['visualization_id'] == viz_id]
        if len(viz_row) == 0:
            return VisualizationScore()
        
        viz_data = viz_row.iloc[0]
        
        # User rating component
        user_rating = viz_data.get('user_rating', 0.0)
        if pd.isna(user_rating):
            user_rating = 0.0
        
        # Generation time component (faster is better, up to a point)
        gen_time = viz_data.get('generation_time_seconds', 60.0)
        if pd.isna(gen_time):
            gen_time = 60.0
        time_score = max(0.0, 1.0 - (gen_time - 10.0) / 50.0)  # Optimal around 10s
        
        # Novelty score based on parameter uniqueness
        novelty_score = self.calculate_novelty_score(viz_id)
        
        # Complexity score based on objective parameters
        complexity_score = self.calculate_complexity_score(viz_id)
        
        # Combined fitness score
        fitness_score = (
            user_rating * 0.4 +
            time_score * 0.1 +
            novelty_score * 0.3 +
            complexity_score * 0.2
        )
        
        return VisualizationScore(
            user_rating=user_rating,
            generation_time=gen_time,
            novelty_score=novelty_score,
            complexity_score=complexity_score,
            fitness_score=fitness_score
        )
    
    def calculate_novelty_score(self, viz_id: str) -> float:
        """
        Calculate novelty score based on parameter uniqueness.
        
        Args:
            viz_id: Visualization ID
            
        Returns:
            Novelty score (0.0-1.0)
        """
        viz_row = self.logger.df[self.logger.df['visualization_id'] == viz_id]
        if len(viz_row) == 0:
            return 0.0
        
        viz_data = viz_row.iloc[0]
        layer_name = viz_data['layer_name']
        obj_type = viz_data['objective_type']
        channel_idx = viz_data['channel_idx']
        
        # Count similar visualizations
        similar_mask = (
            (self.logger.df['layer_name'] == layer_name) &
            (self.logger.df['objective_type'] == obj_type) &
            (abs(self.logger.df['channel_idx'] - channel_idx) < 5)
        )
        
        similar_count = similar_mask.sum()
        total_count = len(self.logger.df)
        
        if total_count == 0:
            return 1.0
        
        # More unique = higher novelty
        novelty = max(0.0, 1.0 - (similar_count / min(total_count, 20)))
        return novelty
    
    def calculate_complexity_score(self, viz_id: str) -> float:
        """
        Calculate complexity score based on objective parameters.
        
        Args:
            viz_id: Visualization ID
            
        Returns:
            Complexity score (0.0-1.0)
        """
        viz_row = self.logger.df[self.logger.df['visualization_id'] == viz_id]
        if len(viz_row) == 0:
            return 0.0
        
        viz_data = viz_row.iloc[0]
        obj_type = viz_data['objective_type']
        
        # Base complexity by objective type
        complexity_map = {
            'channel': 0.2,
            'neuron': 0.4,
            'center': 0.6,
            'gabor': 0.8
        }
        
        base_complexity = complexity_map.get(obj_type, 0.3)
        
        # Bonus for non-default parameters
        bonuses = 0.0
        
        # Check for spatial parameters
        if not pd.isna(viz_data.get('spatial_x', np.nan)):
            bonuses += 0.1
        
        # Check for Gabor parameters
        if not pd.isna(viz_data.get('gabor_sigma', np.nan)):
            bonuses += 0.2
        
        # Check for wrapping
        if viz_data.get('wrapping_enabled', False):
            bonuses += 0.1
        
        return min(1.0, base_complexity + bonuses)
    
    def create_single_mutation(
        self,
        parent_viz_id: str,
        mutation_config: MutationConfig = None,
        save_result: bool = True
    ) -> Dict[str, Any]:
        """
        Create a single mutated visualization from a parent.
        
        Args:
            parent_viz_id: ID of parent visualization
            mutation_config: Configuration for mutation
            save_result: Whether to save the result
            
        Returns:
            Dictionary with mutation results
        """
        if mutation_config is None:
            mutation_config = MutationConfig()
        
        # Find parent visualization
        parent_row = self.logger.df[self.logger.df['visualization_id'] == parent_viz_id]
        if len(parent_row) == 0:
            return {"success": False, "error": "Parent visualization not found"}
        
        try:
            # Parse parent parameters
            import json
            parent_params_json = parent_row.iloc[0]['full_objective_params']
            parent_params = json.loads(parent_params_json)
            
            # Create mutation
            mutated_params = mutate_objective_params(
                parent_params,
                mutation_rate=mutation_config.mutation_rate,
                mutation_strength=mutation_config.mutation_strength
            )
            
            # Generate mutated visualization
            result = self.generator.generate_single_visualization(
                objective_params=mutated_params,
                save_image=save_result,
                notes=f"Mutation of {parent_viz_id}"
            )
            
            if result["success"]:
                # Calculate parent score for comparison
                parent_score = self.calculate_visualization_score(parent_viz_id)
                
                mutation_info = {
                    "parent_id": parent_viz_id,
                    "parent_score": parent_score.fitness_score,
                    "mutation_rate": mutation_config.mutation_rate,
                    "mutation_strength": mutation_config.mutation_strength,
                    "generation": 1  # First generation mutation
                }
                
                result["mutation_info"] = mutation_info
                result["parent_score"] = parent_score
                
                return result
            else:
                return result
                
        except (ImportError, ValueError, KeyError, TypeError, AttributeError) as e:
            return {"success": False, "error": str(e)}
    
    def create_mutation_population(
        self,
        parent_viz_ids: List[str],
        mutation_config: MutationConfig = None,
        generation_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Create a population of mutations from multiple parents.
        
        Args:
            parent_viz_ids: List of parent visualization IDs
            mutation_config: Configuration for mutation
            generation_number: Current generation number
            
        Returns:
            List of mutation results
        """
        if mutation_config is None:
            mutation_config = MutationConfig()
        
        mutations = []
        
        for i in range(mutation_config.population_size):
            # Select random parent
            parent_id = random.choice(parent_viz_ids)
            
            # Create mutation with varied parameters
            varied_config = MutationConfig(
                mutation_rate=mutation_config.mutation_rate + random.uniform(-0.1, 0.1),
                mutation_strength=mutation_config.mutation_strength + random.uniform(-0.05, 0.05)
            )
            
            mutation_result = self.create_single_mutation(
                parent_id,
                varied_config,
                save_result=True
            )
            
            if mutation_result["success"]:
                mutation_result["generation"] = generation_number
                mutations.append(mutation_result)
                
                print(f"🧬 Created mutation {i+1}/{mutation_config.population_size} from {parent_id}")
            else:
                print(f"❌ Mutation {i+1} failed: {mutation_result.get('error', 'Unknown error')}")
        
        return mutations
    
    def evolutionary_generation(
        self,
        num_generations: int = 5,
        mutation_config: MutationConfig = None,
        initial_parents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run evolutionary algorithm to improve visualizations.
        
        Args:
            num_generations: Number of generations to run
            mutation_config: Configuration for mutations
            initial_parents: Initial parent IDs (auto-selected if None)
            
        Returns:
            Dictionary with evolution results
        """
        if mutation_config is None:
            mutation_config = MutationConfig()
        
        print(f"🧬 Starting evolutionary generation ({num_generations} generations)")
        
        # Select initial parents if not provided
        if initial_parents is None:
            initial_parents = self.select_best_parents(mutation_config.population_size // 2)
        
        if not initial_parents:
            return {"success": False, "error": "No suitable parents found"}
        
        generation_results = []
        current_parents = initial_parents
        
        for gen in range(num_generations):
            print(f"\\n🔄 Generation {gen + 1}/{num_generations}")
            
            # Create mutation population
            mutations = self.create_mutation_population(
                current_parents,
                mutation_config,
                generation_number=gen + 1
            )
            
            if not mutations:
                print("❌ No successful mutations in this generation")
                break
            
            # Score all mutations (after generation for user input)
            scored_mutations = []
            for mutation in mutations:
                if mutation["success"]:
                    viz_id = mutation["visualization_id"]
                    score = self.calculate_visualization_score(viz_id)
                    mutation["score"] = score
                    scored_mutations.append(mutation)
            
            # Select best performers for next generation
            scored_mutations.sort(key=lambda x: x["score"].fitness_score, reverse=True)
            
            elite_count = max(1, int(len(scored_mutations) * mutation_config.elite_percentage))
            elites = scored_mutations[:elite_count]
            
            # Update current parents
            current_parents = [m["visualization_id"] for m in elites]
            
            generation_results.append({
                "generation": gen + 1,
                "mutations_created": len(mutations),
                "successful_mutations": len(scored_mutations),
                "best_score": elites[0]["score"].fitness_score if elites else 0.0,
                "elite_ids": current_parents
            })
            
            print(f"✅ Generation {gen + 1} complete:")
            print(f"   Created: {len(mutations)} mutations")
            print(f"   Successful: {len(scored_mutations)} mutations")
            print(f"   Best score: {elites[0]['score'].fitness_score:.3f}" if elites else "   No successful mutations")
        
        return {
            "success": True,
            "generations": generation_results,
            "final_elites": current_parents,
            "total_mutations": sum(gr["mutations_created"] for gr in generation_results)
        }
    
    def select_best_parents(self, n: int = 10) -> List[str]:
        """
        Select best visualizations as parents for evolution.
        
        Args:
            n: Number of parents to select
            
        Returns:
            List of visualization IDs
        """
        if len(self.logger.df) == 0:
            return []
        
        # Calculate scores for all visualizations
        viz_scores = []
        for viz_id in self.logger.df['visualization_id']:
            score = self.calculate_visualization_score(viz_id)
            viz_scores.append((viz_id, score.fitness_score))
        
        # Sort by fitness score
        viz_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [viz_id for viz_id, _ in viz_scores[:n]]
    
    def get_mutation_lineage(self, viz_id: str, max_depth: int = 10) -> List[str]:
        """
        Get the mutation lineage (family tree) of a visualization.
        
        Args:
            viz_id: Visualization ID to trace
            max_depth: Maximum depth to trace back
            
        Returns:
            List of ancestor IDs (oldest first)
        """
        lineage = []
        current_id = viz_id
        depth = 0
        
        while current_id and depth < max_depth:
            viz_row = self.logger.df[self.logger.df['visualization_id'] == current_id]
            if len(viz_row) == 0:
                break
            
            parent_id = viz_row.iloc[0].get('parent_id')
            if pd.isna(parent_id) or not parent_id:
                break
            
            lineage.insert(0, parent_id)  # Add to beginning (oldest first)
            current_id = parent_id
            depth += 1
        
        return lineage
    
    def analyze_mutation_success(self) -> Dict[str, Any]:
        """
        Analyze success patterns in mutations.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.logger.df) == 0:
            return {"error": "No data available"}
        
        # Find mutations (have parent_id)
        mutations = self.logger.df[self.logger.df['parent_id'].notna()]
        
        if len(mutations) == 0:
            return {"error": "No mutations found"}
        
        # Calculate success metrics
        analysis = {
            "total_mutations": len(mutations),
            "original_visualizations": len(self.logger.df) - len(mutations),
            "mutation_rate": len(mutations) / len(self.logger.df),
        }
        
        # Score analysis
        mutation_scores = []
        parent_scores = []
        
        for _, mutation in mutations.iterrows():
            mut_score = self.calculate_visualization_score(mutation['visualization_id'])
            mutation_scores.append(mut_score.fitness_score)
            
            parent_id = mutation['parent_id']
            if parent_id:
                parent_score = self.calculate_visualization_score(parent_id)
                parent_scores.append(parent_score.fitness_score)
        
        if mutation_scores and parent_scores:
            analysis.update({
                "avg_mutation_score": np.mean(mutation_scores),
                "avg_parent_score": np.mean(parent_scores),
                "improvement_rate": np.mean([m > p for m, p in zip(mutation_scores, parent_scores)]),
                "avg_improvement": np.mean([m - p for m, p in zip(mutation_scores, parent_scores)])
            })
        
        # Best mutations
        if mutation_scores:
            best_idx = np.argmax(mutation_scores)
            best_mutation = mutations.iloc[best_idx]
            analysis["best_mutation"] = {
                "id": best_mutation['visualization_id'],
                "score": mutation_scores[best_idx],
                "parent_id": best_mutation['parent_id']
            }
        
        return analysis


# Convenience functions
def create_mutation_from_viz(
    parent_viz_id: str,
    mutation_strength: float = 0.3,
    csv_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simple function to create a single mutation.
    
    Args:
        parent_viz_id: Parent visualization ID
        mutation_strength: Strength of mutation (0.0-1.0)
        csv_filename: CSV file for logging
        
    Returns:
        Mutation result dictionary
    """
    mutator = VisualizationMutator(csv_filename)
    config = MutationConfig(mutation_strength=mutation_strength)
    
    return mutator.create_single_mutation(parent_viz_id, config)


def run_evolution_experiment(
    num_generations: int = 3,
    population_size: int = 10,
    csv_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a simple evolution experiment.
    
    Args:
        num_generations: Number of generations
        population_size: Size of each generation
        csv_filename: CSV file for logging
        
    Returns:
        Evolution results
    """
    mutator = VisualizationMutator(csv_filename)
    config = MutationConfig(
        population_size=population_size,
        mutation_rate=0.3,
        mutation_strength=0.2
    )
    
    return mutator.evolutionary_generation(num_generations, config)


if __name__ == "__main__":
    print("🧬 Visualization Mutation System")
    print("=" * 40)
    
    # Create mutator
    mutator = VisualizationMutator()
    
    print("\\n📊 Current database analysis:")
    analysis = mutator.analyze_mutation_success()
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print("\\n🏆 Best potential parents:")
    best_parents = mutator.select_best_parents(5)
    for i, parent_id in enumerate(best_parents, 1):
        score = mutator.calculate_visualization_score(parent_id)
        print(f"  {i}. {parent_id} (score: {score.fitness_score:.3f})")
    
    print("\\n🧬 Mutation system ready!")
    print("Use create_mutation_from_viz() or run_evolution_experiment() to get started.")