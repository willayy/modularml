#include "conv_node.hpp"

ConvNode::ConvNode(std::string X,
                      std::string W,
                      std::string Y,
                      array_mml<int> dilations,
                      array_mml<int> padding,
                      array_mml<int> kernel_shape,
                      array_mml<int> stride,
                      optional<std::string> B,
                      int group)
    : X(X), W(W), B(B), Y(Y), dilations(dilations), padding(padding), kernel_shape(kernel_shape), stride(stride) {
    if (dilations.size() != 2) {
        throw invalid_argument("Invalid dilations size. Expected a vector of size 2, but got: " +
                                std::to_string(dilations.size()) + ".");
    }

    if (padding.size() != 4) {
        throw invalid_argument("Invalid padding vector size. Expected a vector of size 4, but got: " +
                                std::to_string(padding.size()) + ".");
    }

    if (kernel_shape.size() != 2) {
        throw invalid_argument("Invalid kernel_shape vector size. Expected a vector of size 2, but got: " +
                                std::to_string(kernel_shape.size()) + ".");
    }

    if (stride.size() != 2) {
        throw invalid_argument("Invalid stride vector size. Expected a vector of size 2, but got: " +
                                std::to_string(stride.size()) + ".");
    }
}

ConvNode::ConvNode(const json& node) {
    if (node.contains("inputs") && node["inputs"].is_array()) {
        X = node["inputs"][0];
        W = node["inputs"][1];
        if (node["inputs"].size() > 2) {
            B = node["inputs"][2];
        }
    }

    if (node.contains("outputs") && node["outputs"].is_array()) {
        Y = node["outputs"][0];
    }

    group = 1;
    if (node.contains("attributes") && node["attributes"].is_object()) {
        for (const auto& attr : node["attributes"]) {
            if (attr["name"] == "dilations") {
                std::vector<int> dilations_vec = attr["ints"];
                dilations = array_mml<int>(dilations_vec);
            } else if (attr["name"] == "pads") {
                std::vector<int> padding_vec = attr["ints"];
                padding = array_mml<int>(padding_vec);
            } else if (attr["name"] == "kernel_shape") {
                std::vector<int> kernel_vec = attr["ints"];
                kernel_shape = array_mml<int>(kernel_vec);
            } else if (attr["name"] == "strides") {
                std::vector<int> stride_vec = attr["ints"];
                stride = array_mml<int>(stride_vec);
            } else if (attr["name"] == "group") {
                group = attr["i"];
            }
        }
    }
}

void ConvNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    //infer attributes first

    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("ConvNode: Input tensor X not found in iomap");
    }

    const GeneralDataTypes& x_tensor = x_it->second;

    std::visit([&](const auto& x_ptr) {
        using TensorPtr = std::decay_t<decltype(x_ptr)>;
        using TensorType = typename TensorPtr::element_type;
        using ValueType = typename TensorType::value_type;
    
        if constexpr (!is_in_variant_v<shared_ptr<Tensor<ValueType>>, T>) {
          throw std::runtime_error("ConvNode: Unsupported data type for tensor data");
        } else {
            if (x_ptr->get_shape().size() < 1) {
                throw runtime_error("Input tensor must have 4 dimensions: (Features x Channels x Height x Width).");
            }
        
            auto w_it = iomap.find(W);
            if (w_it == iomap.end()) {
              throw std::runtime_error("ConvNode: Input tensor W not found in iomap");
            }
        
            auto w_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(w_it->second);
    
            auto y_it = iomap.find(Y);
            if (y_it == iomap.end()) {
              throw std::runtime_error("ConvNode: Input tensor Y not found in iomap");
            }
        
            auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);
    
            // Create a copy of the input
            auto input_copy = x_ptr->copy();
            
            // Begin by flipping the weight kernel
            T weight_variant = w_ptr;
            flip_kernel(weight_variant);
    
            auto im2col_output_shape = array_mml<int>({get_in_channels() * get_kernel_height() * get_kernel_width(),
                get_batch_size() * get_out_height() * get_out_width()});
    
            auto im2col_output = make_shared<Tensor_mml<ValueType>>(im2col_output_shape);
            
            T input_copy_variant = input_copy;
            T im2col_output_variant = im2col_output;
            im2col(input_copy_variant, im2col_output_variant);
    
            // Flatten the weight tensor to prepare for GEMM
            int flattened_size = get_in_channels() * get_kernel_height() * get_kernel_width();
            w_ptr->reshape({get_out_channels(), flattened_size});
            
            // Prepare the result tensor
            array_mml<int> result_shape({w_ptr->get_shape()[0], im2col_output->get_shape()[1]});
            auto result_ptr = make_shared<Tensor_mml<ValueType>>(result_shape);
    
            auto gemm = make_shared<Gemm_mml<ValueType>>();
            gemm->gemm_inner_product(
                0, 0,
                w_ptr->get_shape()[0], im2col_output->get_shape()[1], w_ptr->get_shape()[1],
                1.0f,
                w_ptr, w_ptr->get_shape()[1],
                im2col_output, im2col_output->get_shape()[1],
                0.0f,
                result_ptr, result_ptr->get_shape()[1]);
    
            result_ptr->reshape({get_batch_size(), get_out_channels(), get_out_height(), get_out_width()});
    
            // Provided a bias, add it to the result tensor across each output feature
            if (B.has_value()) {
                auto b_it = iomap.find(B.value());
                if (b_it == iomap.end()) {
                    throw std::runtime_error("ConvNode: Input tensor B not found in iomap");
                }
                auto b_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(b_it->second);

                T result_variant = result_ptr;
                T b_variant = b_ptr;
                add_bias(result_variant, b_variant);
            }
        
            // Write over the content of the output with the result of the convolution
            *y_ptr = *result_ptr;
        }

    }, x_tensor);
}

void ConvNode::flip_kernel(T& weight_variant) {
    std::visit([this](auto &weight) {
        int height = get_kernel_height();
        int width = get_kernel_width();

        for (int f = 0; f < get_out_channels(); f++) {
            for (int c = 0; c < get_in_channels(); c++) {
                // Flip horizontally
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width / 2; w++) {
                        auto tmp = (*weight)[{f, c, h, w}];
                        (*weight)[{f, c, h, w}] = (*weight)[{f, c, h, width - 1 - w}];
                        (*weight)[{f, c, h, width - 1 - w}] = tmp;
                    }
                }

                // Flip vertically
                for (int w = 0; w < width; w++) {
                    for (int h = 0; h < height / 2; h++) {
                        auto tmp = (*weight)[{f, c, h, w}];
                        (*weight)[{f, c, h, w}] = (*weight)[{f, c, height - h - 1, w}];
                        (*weight)[{f, c, height - 1 - h, w}] = tmp;
                    }
                }
            }
        }
    }, weight_variant);
}

void ConvNode::im2col(T& input_variant, T& output_variant) {
    std::visit([this](auto &input, auto &output) {
        // Iterate over each image in the batch
        for (int n = 0; n < get_batch_size(); ++n) {
            for (int h = 0; h < get_out_height(); ++h) {
                for (int w = 0; w < get_out_width(); ++w) {  // Traverse into each batch

                    int col_index = h * get_out_width() + w;  // Column index in im2col matrix

                    for (int c = 0; c < get_in_channels(); ++c) {  // If the input has multiple channels, iterate over each one

                        // Here we loop over the kernel's height and width, simulating how the kernel moves across the input tensor.
                        // For each position of the kernel, the corresponding input values are extracted and stored in the output tensor.
                        // If the kernel extends beyond the boundaries of the input (due to padding or stride), zero padding is added instead of the input values.
                        for (int kh = 0; kh < get_kernel_height(); ++kh) {
                            for (int kw = 0; kw < get_kernel_width(); ++kw) {
                                int input_h = h * get_stride_height() - get_padding_top() + kh;
                                int input_w = w * get_stride_width() - get_padding_left() + kw;

                                if (input_h < 0 || input_h >= get_in_height() + get_padding_bottom() ||
                                    input_w < 0 || input_w >= get_in_width() + get_padding_right()) {
                                    (*output)[col_index] = 0;  // Padding
                                } else {
                                    int row_index = c * get_kernel_height() * get_kernel_width() + kh * get_kernel_width() + kw;

                                    int output_index = row_index * (get_out_height() * get_out_width()) + col_index;

                                    int input_index = n * (get_in_channels() * get_in_height() * get_in_width()) +
                                                    c * (get_in_height() * get_in_width()) +
                                                    input_h * get_in_width() + input_w;

                                    // Check if input index is valid
                                    if (input_index >= 0 && input_index < get_in_channels() * get_in_height() * get_in_width()) {
                                        (*output)[output_index] = (*input)[input_index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }, input_variant, output_variant);
}

void ConvNode::add_bias(T& result_variant, T& bias_variant) {
    std::visit([this](auto &result, auto &bias) {
        for (int b = 0; b < get_batch_size(); b++) {
            for (int i = 0; i < get_out_channels(); ++i) {
                for (int h = 0; h < get_out_height(); ++h) {
                    for (int w = 0; w < get_out_width(); ++w) {
                        int index = ((b * get_out_channels() + i) * get_out_height() + h) * get_out_width() + w;

                        // Each value in bias vector is added to one entire out feature at a time
                        (*result)[index] += (*bias)[i];
                    }
                }
            }
        }
    }, result_variant, bias_variant);
}

int ConvNode::get_batch_size() const { return batch_size; }

int ConvNode::get_in_channels() const { return in_channels; }

int ConvNode::get_in_height() const { return in_height; }

int ConvNode::get_in_width() const { return in_width; }

int ConvNode::get_kernel_height() const { return kernel_height; }

int ConvNode::get_kernel_width() const { return kernel_width; }

int ConvNode::get_out_channels() const { return out_channels; }

int ConvNode::get_stride_height() const { return stride[0]; }

int ConvNode::get_stride_width() const { return stride[1]; }

int ConvNode::get_padding_top() const { return padding[0]; }

int ConvNode::get_padding_bottom() const { return padding[1]; }

int ConvNode::get_padding_left() const { return padding[2]; }

int ConvNode::get_padding_right() const { return padding[3]; }

int ConvNode::get_out_height() {
    return (get_in_height() + get_padding_top() + get_padding_bottom() - get_kernel_height()) / get_stride_height() + 1;
}

int ConvNode::get_out_width() {
    return (get_in_width() + get_padding_left() + get_padding_right() - get_kernel_width()) / get_stride_width() + 1;
}