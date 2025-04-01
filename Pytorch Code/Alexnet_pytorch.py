import torch
import torchvision.models as models
import torch.nn as nn

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def append_tensor_to_header(tensor, macro_prefix, header_file, write_once=False, reverse_shape=False):
    shape_vector = list(tensor.shape)
    if reverse_shape:
        shape_vector = shape_vector[::-1]
    values = tensor.flatten().tolist()
    shape_define = f"#define {macro_prefix}_SHAPE {', '.join(map(str, shape_vector))}\n"
    data_define = f"#define {macro_prefix}_DATA {', '.join(map(str, values))}\n"

    mode = "w" if write_once else "a"
    with open(header_file, mode) as f:
        if write_once:
            f.write("// Auto-generated tensor data for AlexNet\n#pragma once\n\n")
        f.write(shape_define)
        f.write(data_define)

# === Load AlexNet and Run === #
model = models.alexnet(pretrained=True)
model.eval()

# Use an example RGB image of size 224x224 (AlexNet input)
example_tensor = torch.randn(1, 3, 224, 224)
output = model(example_tensor)

# Output file
header_file = "test_AlexNet.hpp"

# Input tensor
append_tensor_to_header(example_tensor, "INPUT_TENSOR", header_file, write_once=True)

# Final output
append_tensor_to_header(output, "OUTPUT_TENSOR", header_file)

# Predicted class
predicted_class = torch.argmax(output, dim=1).item()
with open(header_file, "a") as f:
    f.write(f"#define PREDICTED_CLASS {predicted_class}\n")

print("Header file 'test_AlexNet.hpp' generated.")
print("Predicted class:", predicted_class)

# === Export to ONNX === #
onnx_filename = "alexnet.onnx"

# Wrapper for ONNX (no activations used)
class ONNXAlexNetWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

onnx_model = ONNXAlexNetWrapper(model)

torch.onnx.export(
    onnx_model,
    example_tensor,
    onnx_filename,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model exported to ONNX format as '{onnx_filename}'.")
