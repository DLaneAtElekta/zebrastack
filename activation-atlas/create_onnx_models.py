# Create and save Inception model for Netron visualization
import torch
import torch.onnx
import netron
from lucent.modelzoo import inceptionv1

# Create Inception model
inception_model = inceptionv1(pretrained=True).eval()

print(f"Total parameters: {sum(p.numel() for p in inception_model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in inception_model.parameters() if p.requires_grad):,}"
)

# Get model structure overview
print("\n=== High-Level Model Structure ===")
for name, module in inception_model.named_children():
    if hasattr(module, "__len__"):
        print(f"{name}: {type(module).__name__} with {len(module)} blocks")
    else:
        print(f"{name}: {type(module).__name__}")
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            print(
                f"  -> {module.in_channels} → {module.out_channels} channels"
            )  # Install required packages for visualization

# Create dummy input for ONNX export
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX format for Netron
torch.onnx.export(
    inception_model,
    dummy_input,
    "inception_v1.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("Inception model saved as inception_v1.onnx")
