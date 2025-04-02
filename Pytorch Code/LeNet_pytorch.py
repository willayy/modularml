# This script generates a header file with all the tensors from a custom LeNet model
# The header file can be put in the test directory of the library to test the model
# against Pytorch's implementation.

# Requires PyTorch and ONNX to be installed. ONNX viewer (VS Code extension) is recommended to visualize the model.

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CustomLeNet(nn.Module):
    def __init__(self):
        super(CustomLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, bias = False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias = False)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, bias = False)
        self.fc1 = nn.Linear(120 * 2 * 2, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x, activations):
        activations["input"] = x.clone()
        x = F.relu(self.conv1(x))
        activations["conv1_output"] = x.clone()

        x = F.max_pool2d(x, 2)
        activations["maxpool1_output"] = x.clone()

        x = F.relu(self.conv2(x))
        activations["conv2_output"] = x.clone()

        x = F.max_pool2d(x, 2)
        activations["maxpool2_output"] = x.clone()

        x = F.relu(self.conv3(x))
        activations["conv3_output"] = x.clone()

        x = x.view(-1, 120 * 2 * 2)

        activations["reshape1_output"] = x.clone()

        x = F.relu(self.fc1(x))
        activations["gemm1_output"] = x.clone()

        x = self.fc2(x)
        activations["gemm2_output"] = x.clone()

        x = F.log_softmax(x, dim=1)
        activations["logsoftmax_output"] = x.clone()  # Renamed to avoid overwrite
        return x


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
            f.write("// Auto-generated tensor data for LeNet\n#pragma once\n\n")
        f.write(shape_define)
        f.write(data_define)


# === Run model and dump outputs === #
model = CustomLeNet()
example_tensor = torch.randn(1, 1, 32, 32)
activations = {}
output = model(example_tensor, activations)

# Output file
header_file = "test_LeNet.hpp"

# Input tensor
append_tensor_to_header(example_tensor, "INPUT_TENSOR", header_file, write_once=True)

# Intermediate node outputs
for name, tensor in activations.items():
    macro_name = name.upper().replace("/", "_")
    append_tensor_to_header(tensor.detach(), macro_name, header_file)

# Final output as separate macro
append_tensor_to_header(output, "OUTPUT_TENSOR", header_file)

# FC weights & biases
append_tensor_to_header(model.fc1.weight.detach(), "FC1_WEIGHT", header_file, reverse_shape=True)
append_tensor_to_header(model.fc1.bias.detach(), "FC1_BIAS", header_file)
append_tensor_to_header(model.fc2.weight.detach(), "FC2_WEIGHT", header_file, reverse_shape=True)
append_tensor_to_header(model.fc2.bias.detach(), "FC2_BIAS", header_file)

# Conv weights
conv_layers = {
    "conv1": model.conv1.weight,
    "conv2": model.conv2.weight,
    "conv3": model.conv3.weight,
}

for name, layer in conv_layers.items():
    append_tensor_to_header(layer.detach(), f"{name.upper()}_WEIGHT", header_file)

# Optional: predicted class as macro
predicted_class = torch.argmax(output, dim=1).item()
with open(header_file, "a") as f:
    f.write(f"#define PREDICTED_CLASS {predicted_class}\n")

# Log
print("Header file 'test_LeNet.hpp' generated with all node outputs.")
print("Predicted class:", predicted_class)

# === Export to ONNX === #
onnx_filename = "custom_lenet.onnx"

# Wrapper forward method for ONNX export (no activations dictionary)
class ONNXLeNetWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # Just run the forward pass without recording activations
        return self.base_model(x, activations={})

# Prepare the model and dummy input
onnx_model = ONNXLeNetWrapper(model)
onnx_model.eval()

# Export to ONNX
torch.onnx.export(
    onnx_model,
    example_tensor,                        # example input
    onnx_filename,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model exported to ONNX format as '{onnx_filename}'.")
