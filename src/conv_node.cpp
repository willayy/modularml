#include "conv_node.hpp"

ConvNode::ConvNode(std::string X,
                      std::string W,
                      std::string Y,
                      array_mml<uli> dilations,
                      array_mml<uli> padding,
                      array_mml<uli> kernel_shape,
                      array_mml<uli> stride,
                      optional<std::string> B,
                      uli group)
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
    if (node.contains("input") && node["input"].is_array()) {
        X = node["input"][0];
        W = node["input"][1];
        if (node["input"].size() > 2) {
            B = node["input"][2];
        }
    }

    if (node.contains("output") && node["output"].is_array()) {
        Y = node["output"][0];
    }

    group = 1;
    if (node.contains("attribute") && node["attribute"].is_object()) {
        for (const auto& attr : node["attribute"]) {
            if (attr["name"] == "dilations") {
                std::vector<uli> dilations_vec = attr["ints"];
                dilations = array_mml<uli>(dilations_vec);
            } else if (attr["name"] == "pads") {
                std::vector<uli> padding_vec = attr["ints"];
                padding = array_mml<uli>(padding_vec);
            } else if (attr["name"] == "kernel_shape") {
                std::vector<uli> kernel_vec = attr["ints"];
                kernel_shape = array_mml<uli>(kernel_vec);
            } else if (attr["name"] == "strides") {
                std::vector<uli> stride_vec = attr["ints"];
                stride = array_mml<uli>(stride_vec);
            } else if (attr["name"] == "group") {
                group = attr["i"];
            }
        }
    }
}

void ConvNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("ConvNode: Input tensor X not found in iomap");
    }

    auto w_it = iomap.find(W);
    if (w_it == iomap.end()) {
        throw std::runtime_error("ConvNode: Input tensor W not found in iomap");
    }

    const GeneralDataTypes& x_tensor = x_it->second;
    const GeneralDataTypes& w_tensor = w_it->second;

    std::visit([&](const auto& x_ptr, const auto& w_ptr) {
        using ValueTypeX = typename std::decay_t<decltype(x_ptr)>::element_type::value_type;
        using ValueTypeW = typename std::decay_t<decltype(w_ptr)>::element_type::value_type;
        
    
        if constexpr (!is_in_variant_v<ValueTypeX, T> || !std::is_same_v<ValueTypeX, ValueTypeW>) {
          throw std::runtime_error("ConvNode: Unsupported data type for tensor data");
        } else {
            if (x_ptr->get_shape().size() < 1) {
                throw runtime_error("Input tensor must have 4 dimensions: (Features x Channels x Height x Width).");
            }
    
            auto y_it = iomap.find(Y);
            if (y_it == iomap.end()) {
                // Create output tensor if it doesn't exist
                auto y_ptr = x_ptr->copy();
                // No need to fill with zeros as the convolution function will overwrite the values
                iomap[Y] = y_ptr;
                y_it = iomap.find(Y);
            } else if (!std::holds_alternative<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second)) {
                throw std::runtime_error("ConvNode: Output tensor Y has incorrect type");
            }
        
            auto y_ptr = std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

            //infer and update attributes first
            update_parameters(x_ptr->get_shape(), w_ptr->get_shape());
    
            // Create a copy of the input
            auto input_copy = x_ptr->copy();
            
            // Begin by flipping the weight kernel
            flip_kernel(w_ptr);
    
            auto im2col_output_shape = array_mml<uli>({get_in_channels() * get_kernel_height() * get_kernel_width(),
                get_batch_size() * get_out_height() * get_out_width()});
    
            auto im2col_output = make_shared<Tensor_mml<ValueTypeX>>(im2col_output_shape);
            
            im2col(input_copy, im2col_output);
    
            // Flatten the weight tensor to prepare for GEMM
            uli flattened_size = get_in_channels() * get_kernel_height() * get_kernel_width();
            w_ptr->reshape({get_out_channels(), flattened_size});
            
            // Prepare the result tensor
            array_mml<uli> result_shape({w_ptr->get_shape()[0], im2col_output->get_shape()[1]});
            auto result_ptr = make_shared<Tensor_mml<ValueTypeX>>(result_shape);
    
            auto gemm = make_shared<Gemm_mml<ValueTypeX>>();
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
                auto b_ptr = std::get<std::shared_ptr<Tensor<ValueTypeX>>>(b_it->second);

                add_bias(result_ptr, b_ptr);
            }
        
            // Write over the content of the output with the result of the convolution
            *y_ptr = *result_ptr;
        }

    }, x_tensor, w_tensor);
}

std::vector<std::string> ConvNode::getInputs() {
    if (B.has_value()) {
        return {X, W, B.value()};
    } else {
        return {X, W};
    }
}

std::vector<std::string> ConvNode::getOutputs() {
    return {Y};
}

void ConvNode::flip_kernel(const TensorT& weight_variant) {
    std::visit([this](auto &weight) {
        uli height = get_kernel_height();
        uli width = get_kernel_width();

        for (uli f = 0; f < get_out_channels(); f++) {
            for (uli c = 0; c < get_in_channels(); c++) {
                // Flip horizontally
                for (uli h = 0; h < height; h++) {
                    for (uli w = 0; w < width / 2; w++) {
                        auto tmp = (*weight)[{f, c, h, w}];
                        (*weight)[{f, c, h, w}] = (*weight)[{f, c, h, width - 1 - w}];
                        (*weight)[{f, c, h, width - 1 - w}] = tmp;
                    }
                }

                // Flip vertically
                for (uli w = 0; w < width; w++) {
                    for (uli h = 0; h < height / 2; h++) {
                        auto tmp = (*weight)[{f, c, h, w}];
                        (*weight)[{f, c, h, w}] = (*weight)[{f, c, height - h - 1, w}];
                        (*weight)[{f, c, height - 1 - h, w}] = tmp;
                    }
                }
            }
        }
    }, weight_variant);
}

void ConvNode::im2col(const TensorT& input_variant, const TensorT& output_variant) {
    std::visit([this](auto &input, auto &output) {
        // Iterate over each image in the batch
        for (uli n = 0; n < get_batch_size(); ++n) {
            for (uli h = 0; h < get_out_height(); ++h) {
                for (uli w = 0; w < get_out_width(); ++w) {  // Traverse into each batch

                    uli col_index = h * get_out_width() + w;  // Column index in im2col matrix

                    for (uli c = 0; c < get_in_channels(); ++c) {  // If the input has multiple channels, iterate over each one

                        // Here we loop over the kernel's height and width, simulating how the kernel moves across the input tensor.
                        // For each position of the kernel, the corresponding input values are extracted and stored in the output tensor.
                        // If the kernel extends beyond the boundaries of the input (due to padding or stride), zero padding is added instead of the input values.
                        for (uli kh = 0; kh < get_kernel_height(); ++kh) {
                            for (uli kw = 0; kw < get_kernel_width(); ++kw) {
                                uli input_h = h * get_stride_height() - get_padding_top() + kh;
                                uli input_w = w * get_stride_width() - get_padding_left() + kw;

                                if (input_h < 0 || input_h >= get_in_height() + get_padding_bottom() ||
                                    input_w < 0 || input_w >= get_in_width() + get_padding_right()) {
                                    (*output)[col_index] = 0;  // Padding
                                } else {
                                    uli row_index = c * get_kernel_height() * get_kernel_width() + kh * get_kernel_width() + kw;

                                    uli output_index = row_index * (get_out_height() * get_out_width()) + col_index;

                                    uli input_index = n * (get_in_channels() * get_in_height() * get_in_width()) +
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

void ConvNode::add_bias(const TensorT& result_variant, const TensorT& bias_variant) {
    std::visit([this](auto &result, auto &bias) {
        for (uli b = 0; b < get_batch_size(); b++) {
            for (uli i = 0; i < get_out_channels(); ++i) {
                for (uli h = 0; h < get_out_height(); ++h) {
                    for (uli w = 0; w < get_out_width(); ++w) {
                        uli index = ((b * get_out_channels() + i) * get_out_height() + h) * get_out_width() + w;

                        // Each value in bias vector is added to one entire out feature at a time
                        (*result)[index] += (*bias)[i];
                    }
                }
            }
        }
    }, result_variant, bias_variant);
}

uli ConvNode::get_batch_size() const { return batch_size; }

uli ConvNode::get_in_channels() const { return in_channels; }

uli ConvNode::get_in_height() const { return in_height; }

uli ConvNode::get_in_width() const { return in_width; }

uli ConvNode::get_kernel_height() const { return kernel_height; }

uli ConvNode::get_kernel_width() const { return kernel_width; }

uli ConvNode::get_out_channels() const { return out_channels; }

uli ConvNode::get_stride_height() const { return stride[0]; }

uli ConvNode::get_stride_width() const { return stride[1]; }

uli ConvNode::get_padding_top() const { return padding[0]; }

uli ConvNode::get_padding_bottom() const { return padding[1]; }

uli ConvNode::get_padding_left() const { return padding[2]; }

uli ConvNode::get_padding_right() const { return padding[3]; }

uli ConvNode::get_out_height() {
    return (get_in_height() + get_padding_top() + get_padding_bottom() - get_kernel_height()) / get_stride_height() + 1;
}

uli ConvNode::get_out_width() {
    return (get_in_width() + get_padding_left() + get_padding_right() - get_kernel_width()) / get_stride_width() + 1;
}

void ConvNode::update_parameters(const array_mml<uli>& input_shape, const array_mml<uli>& weight_shape) {
    kernel_height = weight_shape[2];
    kernel_width = weight_shape[3];
    batch_size = input_shape[0];
    in_channels = input_shape[1];

    in_height = input_shape[2];
    in_width = input_shape[3];
    out_channels = weight_shape[0];
}