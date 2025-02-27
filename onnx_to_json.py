"""
ONNX Node Type documentation
============================

This framework processes various ONNX node types and extracts relevant attributes 
for each. Below is an overview of the primary node types and the information they store.

1. Conv (Convolution)
   - Inputs: Input tensor, Weights, (Optional) Bias
   - Outputs: Feature map
   - Attributes:
     - kernel_shape: List of kernel dimensions
     - strides: List specifying stride for each spatial dimension
     - pads: Padding values
     - dilations: Dilation factor
     - group: Number of groups for grouped convolutions

2. Relu (Rectified Linear Unit)
   - Inputs: Input tensor
   - Outputs: Activated output tensor
   - Attributes: None (ReLU is a simple activation function)

3. MaxPool / Maxpool (Max Pooling)
   - Inputs: Input tensor
   - Outputs: Pooled feature map
   - Attributes:
     - kernel_shape: Size of the pooling window
     - strides: Stride of the pooling operation
     - pads: Padding around the input

4. AveragePool (Average Pooling)
   - Inputs: Input tensor
   - Outputs: Pooled feature map
   - Attributes: Same as MaxPool

5. Flatten
   - Inputs: Input tensor
   - Outputs: Flattened tensor
   - Attributes:
     - axis: Axis from which flattening starts

6. Gemm (General Matrix Multiplication)
   - Inputs: Input tensor, Weights, (Optional) Bias
   - Outputs: Fully connected layer output
   - Attributes:
     - alpha: Scalar multiplier for input matrix multiplication (default: 1.0)
     - beta: Scalar multiplier for bias (default: 1.0)
     - transA: Whether to transpose input matrix A
     - transB: Whether to transpose input matrix B

============================

Notes:

- The framework extracts these attributes when parsing an ONNX model.
- Other node types may be present, but these are the primary ones handled.

"""

import onnx
import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser(description="Process a file")
parser.add_argument("path", type=str, help="Path to a file")

""" Checks if the path provided leads to a onnx file """
def is_onnx(path: str):
    if os.path.isfile(path) and path.lower().endswith(".onnx"):
        return True
    else:
        return False

def convert_initializer(initializer):
    dtype_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64
    }

    dtype = dtype_map.get(initializer.data_type, np.float32)  # Default to float32
    return np.frombuffer(initializer.raw_data, dtype=dtype).tolist()


""" Creates a JSON representation of the model stored in ONNX format """
def onnx_to_json(path: str):
    if is_onnx(path):
        model = onnx.load(path)
        graph = model.graph

        initializers_dict = {initializer.name: initializer for initializer in graph.initializer}

        model_json = {
            "model": {
                "name": model.graph.name,
                "ir_version": model.ir_version,
                "opset_version": model.opset_import[0].version
            },
            "nodes": []
        }

        num_nodes = len(graph.node)
        for index, node in enumerate(graph.node):
            if (index + 1 == num_nodes):
                print(f"\rProcessing nodes: {index + 1}/{num_nodes}", flush=True) # Shows the progress of the script
            else:
                print(f"\rProcessing nodes: {index + 1}/{num_nodes}", end="", flush=True) # Shows the progress of the script

            node_json = {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": [{"name": i} for i in node.input],
                "outputs": [{"name": i} for i in node.output],
                "attributes": {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
                "initializers": [
                    {
                        "name": initializer.name,
                        "shape": [dim for dim in initializer.dims],
                        "data_type": onnx.TensorProto.DataType.Name(initializer.data_type),
                        "values": convert_initializer(initializer)
                    }
                    for inp in node.input if inp in initializers_dict
                    for initializer in [initializers_dict[inp]]
                ]
            }
            model_json["nodes"].append(node_json)


        with open("./model.json", "w") as f:
            json.dump(model_json, f, indent=4)

""" A helper function that can be used to get a overview of a model from the console """
def get_node_op_types(path: str) -> None:
    if is_onnx(path):
        model = onnx.load(path)
        graph = model.graph
        
        for node in graph.node:
            print(node.op_type) 

# Script that reads a onnx file and converts it into a json format.
def main():
    args = parser.parse_args()
    # get_node_op_types(args.path) # Uncomment to print the nodes in the onnx
    onnx_to_json(args.path)

if __name__ == "__main__":
    main()