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
            "nodes": [
                {
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
                            "values": convert_initializer(initializer)  # Store raw values (you might want to handle this differently)
                        }
                        for inp in node.input if inp in initializers_dict
                        for initializer in [initializers_dict[inp]]
                    ]
                } for node in graph.node
            ]
        }

        with open("./model.json", "w") as f:
            json.dump(model_json, f, indent=4)

# Script that reads a onnx file and converts it into a json format.
def main():
    args = parser.parse_args()
    onnx_to_json(args.path)

if __name__ == "__main__":
    main()