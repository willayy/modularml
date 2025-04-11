import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Define the input and expected output shapes:
# Input shape: [1, 1, 4, 5]
# With kernel_shape=[2,2], strides=[1,2], dilations=[2,2], auto_pad="SAME_UPPER", ceil_mode=0,
# the standard ONNX behavior computes:
#   - effective kernel = (2-1)*2 + 1 = 3
#   - out_dim (height) = floor((4-1)/1) + 1 = 4, (width) = floor((5-1)/2) + 1 = 3
# and the computed padding will shift the pooling window so that for output (0,0)
# only one kernel element (from input position (1,1), value 7) is valid.
# Thus, the output will be like:
# [[[[ 7,  9,  9],
#    [12, 14, 14],
#    [17, 19, 19],
#    [12, 14, 14]]]]
# (which might be surprising if you expected simple subsampling).

input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 5])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 3])

# Create the MaxPool node with the given attributes
maxpool_node = helper.make_node(
    'MaxPool',
    inputs=['input'],
    outputs=['output'],
    kernel_shape=[2, 2],
    strides=[1, 2],
    dilations=[2, 2],
    auto_pad="SAME_UPPER",
    ceil_mode=0  # floor mode
)

# Create the graph and model
graph_def = helper.make_graph(
    nodes=[maxpool_node],
    name='MaxPoolGraph',
    inputs=[input_tensor],
    outputs=[output_tensor]
)
model_def = helper.make_model(graph_def, producer_name='onnx-maxpool-example')

# Save the model to file
onnx.save(model_def, 'maxpool_standard.onnx')


# ----------------------------
# Common Input Data and Model
# ----------------------------

# Define the input data (shape [1, 1, 4, 5])
input_data = np.array([[[[ 1,  2,  3,  4,  5],
                          [ 6,  7,  8,  9, 10],
                          [11, 12, 13, 14, 15],
                          [16, 17, 18, 19, 20]]]], dtype=np.float32)
print("Input Data:")
print(input_data)

onnx_model_file = "maxpool_standard.onnx"  # make sure your ONNX model file is in the same directory

# ----------------------------
# 1. ONNX Runtime
# ----------------------------
try:
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_model_file)
    outputs = session.run(None, {"input": input_data})
    print("\nONNX Runtime output:")
    print(outputs[0])
except ImportError as e:
    print("\n[ONNX Runtime] Package not installed:", e)
except Exception as e:
    print("\n[ONNX Runtime] Error:", e)

# ----------------------------
# 2. OpenCV DNN Module
# ----------------------------
try:
    import cv2 as cv
    net = cv.dnn.readNetFromONNX(onnx_model_file)
    net.setInput(input_data)
    output_cv = net.forward()
    print("\nOpenCV DNN output:")
    print(output_cv)
except ImportError as e:
    print("\n[OpenCV DNN] Package not installed:", e)
except Exception as e:
    print("\n[OpenCV DNN] Error:", e)

