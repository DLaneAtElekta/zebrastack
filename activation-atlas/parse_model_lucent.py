# from lucent.modelzoo import inceptionv1
# from lucent.modelzoo.util import get_model_layers
# from lucent_layer_utils import get_visualizable_layers, get_channels_from_lucent_name


# # Method 1: Get module by name and check its properties
# def get_module_by_name(model, layer_name):
#     """Get a module from the model using dot notation layer name"""
#     # Convert underscore notation (Lucent style) to dot notation (PyTorch style)
#     pytorch_name = layer_name.replace("_", ".")

#     module = model
#     for name in pytorch_name.split("."):
#         if hasattr(module, name):
#             module = getattr(module, name)
#         else:
#             return None
#     return module


# def main():
#     """Demonstrate the layer analysis functions and new center objectives"""
#     print("=== Lucent Layer Analysis Demo ===")

#     # Load a model (using Inception V1 as example)
#     print("\nLoading Inception V1 model...")
#     model = inceptionv1(pretrained=True)
#     print(f"Model type: {type(model).__name__}")

#     # Debug: Check what get_model_layers returns
#     print("\nDebug: Checking get_model_layers output...")
#     all_layers = get_model_layers(model)
#     print(f"Total layers from get_model_layers: {len(all_layers)}")
#     print("First 20 layers:", all_layers[:20])

#     # Get all visualizable layers
#     print("\n1. Getting all visualizable layers...")
#     visualizable = get_visualizable_layers(model)
#     print(f"Found {len(visualizable)} visualizable layers")
#     print("First 10 layers:", visualizable[:10])

#     # Test the new center objective functionality
#     if visualizable:
#         print("\n2. Testing new center objective sizes...")

#         # Test different center objective types
#         objective_types = [
#             "channel",
#             "neuron",
#             "center_3x3",
#             "center_5x5",
#             "center_7x7",
#         ]

#         # Import spatial objectives for testing
#         from spatial_objectives import (
#             create_random_objective,
#             create_center_3x3_objective,
#             create_center_5x5_objective,
#             create_center_7x7_objective,
#             create_center_nxn_objective,
#         )

#         for obj_type in objective_types:
#             print(f"\nTesting {obj_type} objective...")
#             try:
#                 obj = create_random_objective(
#                     model, visualizable, objective_types=[obj_type], sampled_channels=1
#                 )
#                 if obj:
#                     print(f"  ✅ Successfully created {obj_type} objective!")
#                 else:
#                     print(f"  ❌ Failed to create {obj_type} objective")
#             except (RuntimeError, ValueError, AttributeError) as e:
#                 print(f"  ❌ Error creating {obj_type} objective: {e}")

#         # Test direct function calls
#         print("\n3. Testing direct function calls...")
#         test_layer = visualizable[0] if visualizable else "mixed4a_1x1_pre_relu_conv"
#         test_channel = 42

#         center_functions = [
#             ("3x3", create_center_3x3_objective),
#             ("5x5", create_center_5x5_objective),
#             ("7x7", create_center_7x7_objective),
#             (
#                 "NxN (size=9)",
#                 lambda layer, channel: create_center_nxn_objective(
#                     layer, channel, size=9
#                 ),
#             ),
#         ]

#         for size_name, func in center_functions:
#             try:
#                 obj = func(test_layer, test_channel)
#                 if obj:
#                     print(f"  ✅ {size_name} center objective created successfully")
#                 else:
#                     print(f"  ❌ {size_name} center objective creation returned None")
#             except (RuntimeError, ValueError, AttributeError) as e:
#                 print(f"  ❌ {size_name} center objective failed: {e}")

#     # If no visualizable layers found, let's debug further
#     if len(visualizable) == 0:
#         print("\nDebug: Testing specific layers...")
#         test_layers = (
#             all_layers[:10] if all_layers else ["mixed3a", "mixed4a", "mixed5a"]
#         )
#         for layer in test_layers:
#             channels = get_channels_from_lucent_name(model, layer)
#             print(f"  {layer}: {channels} channels")

#     # Get detailed information
#     if visualizable:
#         print("\n4. Getting detailed layer information...")
#         detailed = get_visualizable_layers(model, include_detailed_info=True)
#         print("Sample layer details:")
#         for name, info in list(detailed.items())[:5]:
#             print(f"  {name}: {info}")

#     # Legacy test to see if the basic functions work
#     print("\nLegacy test:")
#     legacy_test()

#     print("\n=== Demo Complete ===")
#     print("\nAvailable objective types:")
#     print("- 'channel': Standard channel objective")
#     print("- 'neuron': Single neuron objective")
#     print("- 'center_3x3': 3x3 center array")
#     print("- 'center_5x5': 5x5 center array")
#     print("- 'center_7x7': 7x7 center array")
#     print(
#         "- Or use create_center_nxn_objective(layer, channel, size=N) for any odd size N"
#     )


# # Legacy test function (keeping for backwards compatibility)
# def legacy_test():
#     model = inceptionv1(pretrained=True)

#     # Test with a few layer names
#     test_layers = ["layer1.0.conv1", "layer2.1.conv2", "layer3.5.conv1", "conv1"]

#     print("Getting channel counts for different layers:")
#     for layer_name in test_layers:
#         module = get_module_by_name(model, layer_name)
#         if module is not None and hasattr(module, "out_channels"):
#             print(f"  {layer_name}: {module.out_channels} channels")
#         else:
#             print(
#                 f"  {layer_name}: Could not determine channels (not a Conv layer or not found)"
#             )

#     # Test with Lucent-style layer names
#     lucent_test_layers = [
#         "layer1_0_conv1",
#         "layer2_1_conv2",
#         "layer3_5_conv1",
#         "layer4_0_conv1",
#     ]

#     print("Channel counts for Lucent-style layer names:")
#     for layer_name in lucent_test_layers:
#         channels = get_channels_from_lucent_name(model, layer_name)
#         if channels is not None:
#             print(f"  {layer_name}: {channels} channels")
#         else:
#             print(f"  {layer_name}: Could not determine channels")


# if __name__ == "__main__":
#     main()
