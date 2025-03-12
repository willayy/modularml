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

7. MatMul (Matrix Multiplication)
   - Inputs: Two matrices A and B
   - Outputs: Matrix product (A × B)
   - Attributes: None (MatMul performs standard matrix multiplication)
   - Notes: 
     - This is often followed by BiasAdd to form a fully connected layer.
     - Often there are weights stored as an attribute making the node have a single input

8. BiasAdd
   - Inputs: Input tensor, Bias tensor
   - Outputs: Input tensor with bias added element-wise
   - Attributes: None (BiasAdd is a simple addition operation)
   - Notes: 
     - In some ONNX models, Gemm replaces MatMul + BiasAdd for efficiency.

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
from onnx import numpy_helper

parser = argparse.ArgumentParser(description="Process a file")
parser.add_argument("path", type=str, help="Path to a file")

model_weight_file_name = "model_weights.bin"

""" Checks if the path provided leads to a onnx file """
def is_onnx(path: str):
    if os.path.isfile(path) and path.lower().endswith(".onnx"):
        return True
    else:
        return False

""" Converts a TensorProto object into a numpy array """
def convert_initializer(initializer):
    return numpy_helper.to_array(initializer)


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
            "nodes": [],
            "weights_file": model_weight_file_name, # Path to the binary file
            "inputs": [convert_tensor_type(i.type) for i in graph.input],
            "outputs": [convert_tensor_type(i.type) for i in graph.output]
        }

        weight_file_path = f"./{model_weight_file_name}"

        # Open the weight binary
        with open(weight_file_path, "wb") as weight_file:

            num_nodes = len(graph.node)

            # Iterate over all nodes in the onnx file, the index is used to keep track of progress
            for index, node in enumerate(graph.node):
                print(f"\rProcessing nodes: {index + 1}/{num_nodes} ({(index + 1) / num_nodes * 100:.2f}%)", end="\r", flush=True)

                # Structure of each node dict
                node_json = {
                    "name": node.name,
                    "op_type": node.op_type,
                    "inputs": [{"name": i} for i in node.input],
                    "outputs": [{"name": i} for i in node.output],
                    "attributes": {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
                    "initializers": []
                }

                # Each node has inputs, we use the name of the input to find the initializer field which stores values for weights and biases
                for i in node.input:
                    if i in initializers_dict:
                        initializer = initializers_dict[i]
                        data = convert_initializer(initializer) # Convert initializer to raw bytes
                        offset = weight_file.tell()
                        size = data.nbytes

                        weight_file.write(data.tobytes())

                        # Structure of each initializer
                        node_json["initializers"].append({
                            "name": initializer.name,
                            "shape": list(initializer.dims),
                            "data_type": onnx.TensorProto.DataType.Name(initializer.data_type),
                            "offset": offset,
                            "size": size
                        })
                        
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

def convert_tensor_type(tensor_type):
    # Print the tensor_type to understand its structure
    print(tensor_type)
    # Check if elem_type is available
    if tensor_type.HasField('tensor_type'):
        element_type = tensor_type.tensor_type.elem_type
        shape = [{"dim_value": dim.dim_value} for dim in tensor_type.tensor_type.shape.dim]
    else:
        element_type = "Unknown"  # Handle missing elem_type gracefully
        shape = []  # If shape or dims are missing, return empty list
    
    return {
        "element_type": element_type,
        "shape": shape
    }


# Script that reads a onnx file and converts it into a json format.
def main():
    args = parser.parse_args()
    # get_node_op_types(args.path) # Uncomment to print the nodes in the onnx
    onnx_to_json(args.path)

if __name__ == "__main__":
    main()
