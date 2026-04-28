"""
Comprehensive demo showcasing all new visualization features.

This script demons    print("\n✅ Objective generators demo complete!")rates:
1. Objective parameter generation for all types
2. Enhanced visualization generation with pandas logging
3. Mutation and evolution of visualizations
4. Gradio         print("\n🎉 DEMO COMPLETE!")
        print(f"   Total time: {total_time:.1f} seconds")
        print("   Generated files: demo_visualizations.csv")
        print("   Gradio data: demo_gradio_data/")
        print("   Images: screen_captures/")face        print("\n📊 Final Database Statistics:")for interactive exploration

Run this script to see all the new features in action!
"""

import os
import time
from typing import List

# Import all our new modules
from objective_generators import (
    generate_random_objective_params,
    create_objective_from_params,
    mutate_objective_params
)
from visualization_logger import get_logger
from enhanced_visualization_generator import EnhancedVisualizationGenerator
from visualization_mutator import VisualizationMutator, MutationConfig
from simple_gradio_interface import launch_simple_explorer

from lucent.modelzoo import inceptionv1


def demo_objective_generators():
    """Demonstrate the objective parameter generators."""
    print("🎯 DEMO: Objective Parameter Generators")
    print("=" * 50)
    
    # Load model
    model = inceptionv1(pretrained=True)
    from lucent_layer_utils import get_visualizable_layers
    layers = get_visualizable_layers(model)
    
    print(f"📊 Model loaded: {len(layers)} visualizable layers")
    
    # Generate different types of objectives
    objective_types = ["channel", "neuron", "center", "gabor"]
    
    for obj_type in objective_types:
        print(f"\\n🎯 Generating {obj_type} objective parameters...")
        
        try:
            params_list = generate_random_objective_params(
                model,
                layers,
                objective_types=[obj_type],
                num_objectives=1,
                input_size=256
            )
            
            if params_list:
                params = params_list[0]
                print(f"   ✅ {obj_type} objective generated:")
                print(f"      Layer: {params['layer_name']}")
                print(f"      Channel: {params['channel_idx']}")
                
                if params.get('spatial_params'):
                    spatial = params['spatial_params']
                    if 'x' in spatial and 'y' in spatial:
                        print(f"      Position: ({spatial['x']:.1f}, {spatial['y']:.1f})")
                    if 'center_size' in spatial:
                        print(f"      Center size: {spatial['center_size']}")
                
                if params.get('gabor_params'):
                    gabor = params['gabor_params']
                    print(f"      Gabor σ={gabor['sigma']:.2f}, λ={gabor['lambda_freq']:.2f}")
                
                # Test creating objective from parameters
                _ = create_objective_from_params(params)
                print("      ✅ Successfully created Lucent objective")
                
                # Test mutation
                mutated = mutate_objective_params(params, mutation_rate=0.5)
                print(f"      🧬 Created mutation with channel {mutated['channel_idx']}")
                
            else:
                print(f"   ❌ Failed to generate {obj_type} objective")
                
        except Exception as e:
            print(f"   ❌ Error with {obj_type}: {e}")
    
    print("\\n✅ Objective generators demo complete!")


def demo_enhanced_generation():
    """Demonstrate enhanced visualization generation with logging."""
    print("\\n🎨 DEMO: Enhanced Visualization Generation")
    print("=" * 50)
    
    # Create enhanced generator
    generator = EnhancedVisualizationGenerator(
        model_name="inception_v1",
        image_size=128,  # Small for demo
        csv_filename="demo_visualizations.csv",
        enable_wrapping=True
    )
    
    print("📊 Generator initialized")
    
    # Generate a few test visualizations
    print("\\n🎨 Generating test visualizations...")
    
    results = generator.generate_batch(
        num_visualizations=3,
        objective_types=["channel", "center", "gabor"],
        enable_mutation=False
    )
    
    successful = [r for r in results if r["success"]]
    print(f"✅ Generated {len(successful)}/{len(results)} visualizations successfully")
    
    # Show statistics
    stats = generator.get_stats()
    print("\\n📈 Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Export for Gradio
    gradio_dir = generator.export_for_gradio("demo_gradio_data")
    print(f"\\n💾 Exported data for Gradio: {gradio_dir}")
    
    return [r["visualization_id"] for r in successful]


def demo_mutation_system(viz_ids: List[str]):
    """Demonstrate the mutation and evolution system."""
    print("\\n🧬 DEMO: Mutation and Evolution System")
    print("=" * 50)
    
    if not viz_ids:
        print("❌ No visualizations available for mutation demo")
        return
    
    # Create mutator
    mutator = VisualizationMutator("demo_visualizations.csv")
    
    print(f"📊 Mutator initialized with {len(mutator.logger.df)} visualizations")
    
    # Analyze current database
    print("\\n📊 Current database analysis:")
    analysis = mutator.analyze_mutation_success()
    for key, value in analysis.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Create single mutation
    print("\\n🧬 Creating single mutation...")
    parent_id = viz_ids[0]
    
    mutation_result = mutator.create_single_mutation(
        parent_id,
        MutationConfig(mutation_rate=0.4, mutation_strength=0.3),
        save_result=True
    )
    
    if mutation_result["success"]:
        print(f"✅ Created mutation {mutation_result['visualization_id']} from {parent_id}")
        
        # Show scoring
        child_score = mutator.calculate_visualization_score(mutation_result['visualization_id'])
        parent_score = mutator.calculate_visualization_score(parent_id)
        
        print(f"   Parent score: {parent_score.fitness_score:.3f}")
        print(f"   Child score: {child_score.fitness_score:.3f}")
        print(f"   Improvement: {child_score.fitness_score - parent_score.fitness_score:+.3f}")
    else:
        print(f"❌ Mutation failed: {mutation_result.get('error', 'Unknown error')}")
    
    # Run mini evolution
    if len(viz_ids) >= 2:
        print("\\n🔄 Running mini evolution experiment...")
        
        evolution_result = mutator.evolutionary_generation(
            num_generations=2,
            mutation_config=MutationConfig(
                population_size=4,
                mutation_rate=0.3,
                mutation_strength=0.2
            ),
            initial_parents=viz_ids[:2]
        )
        
        if evolution_result["success"]:
            print("✅ Evolution experiment complete:")
            for gen_result in evolution_result["generations"]:
                print(f"   Generation {gen_result['generation']}: "
                      f"{gen_result['successful_mutations']} mutations, "
                      f"best score: {gen_result['best_score']:.3f}")
        else:
            print(f"❌ Evolution failed: {evolution_result.get('error', 'Unknown error')}")


def demo_gradio_interface():
    """Demonstrate launching the Gradio interface."""
    print("\\n🌐 DEMO: Gradio Interface")
    print("=" * 50)
    
    print("🚀 The Gradio interface provides:")
    print("   • Interactive visualization generation")
    print("   • Browsing and rating existing visualizations")
    print("   • Real-time mutation creation")
    print("   • Database statistics and analysis")
    print("   • Image viewing and parameter exploration")
    
    response = input("\\n🌐 Launch Gradio interface? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("\\n🚀 Launching Gradio interface...")
        print("   This will open a web browser with the interactive interface")
        print("   Press Ctrl+C to stop the server when done")
        
        try:
            launch_simple_explorer(
                csv_filename="demo_visualizations.csv",
                share=False,
                port=7860
            )
        except KeyboardInterrupt:
            print("\\n👋 Gradio interface stopped")
        except Exception as e:
            print(f"\\n❌ Failed to launch Gradio: {e}")
            print("   Note: Make sure gradio is installed: pip install gradio")
    else:
        print("👍 Skipping Gradio interface demo")


def main():
    """Run the comprehensive demo."""
    print("🎨 COMPREHENSIVE VISUALIZATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases all the new features:")
    print("1. 🎯 Objective parameter generators")
    print("2. 🎨 Enhanced visualization generation")
    print("3. 📊 Pandas logging and CSV serialization")
    print("4. 🧬 Mutation and evolution system")
    print("5. 🌐 Gradio interactive interface")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs("screen_captures", exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run demos
        demo_objective_generators()
        viz_ids = demo_enhanced_generation()
        demo_mutation_system(viz_ids)
        demo_gradio_interface()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\\n🎉 DEMO COMPLETE!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Generated files: demo_visualizations.csv")
        print(f"   Gradio data: demo_gradio_data/")
        print(f"   Images: screen_captures/")
        
        # Show final database stats
        logger = get_logger("demo_visualizations.csv")
        final_stats = logger.get_visualization_stats()
        print(f"\\n📊 Final Database Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
        
        print("\\n🚀 Ready for production use!")
        print("   Use enhanced_visualization_generator.py for batch generation")
        print("   Use simple_gradio_interface.py for interactive exploration")
        print("   Use visualization_mutator.py for evolution experiments")
        
    except KeyboardInterrupt:
        print("\\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()