#include "conv_node.hpp"

template <typename T>
ConvNode<T>::ConvNode(shared_ptr<AbstractTensor> X,
                      shared_ptr<AbstractTensor> W,
                      shared_ptr<AbstractTensor> Y,
                      array_mml<int> dilations,
                      array_mml<int> padding,
                      array_mml<int> kernel_shape,
                      array_mml<int> stride,
                      optional<shared_ptr<AbstractTensor>> B,
                      int group)
    : X(X), W(W), B(B), Y(Y), dilations(dilations), padding(padding), kernel_shape(kernel_shape), stride(stride) {
    kernel_height = W->get_shape()[2];
    kernel_width = W->get_shape()[3];
    batch_size = X->get_shape()[0];
    in_channels = X->get_shape()[1];
    in_height = X->get_shape()[2];
    in_width = X->get_shape()[3];
    out_channels = W->get_shape()[0];
}

template <typename T>
void ConvNode<T>::forward() {
    update_parameters();
    validate_inputs();

    // Create a copy of the input
    auto input_copy = X->copy();

    // Begin by flipping the weight kernel
    flip_kernel();

    auto im2col_output_shape = array_mml<int>({get_in_channels() * get_kernel_height() * get_kernel_width(),
                                               get_batch_size() * get_out_height() * get_out_width()});

    auto im2col_output = make_shared<Tensor_mml<T>>(im2col_output_shape);

    im2col(input_copy, im2col_output);

    // Flatten the weight tensor to prepare for GEMM
    int flattened_size = get_in_channels() * get_kernel_height() * get_kernel_width();
    W->reshape({get_out_channels(), flattened_size});

    // Prepare the result tensor
    array_mml<int> result_shape({W->get_shape()[0], im2col_output->get_shape()[1]});
    auto result_ptr = make_shared<Tensor_mml<T>>(result_shape);

    shared_ptr<GemmModule<T>> gemm = make_shared<Gemm_mml<T>>();
    gemm->gemm_inner_product(
        0, 0,
        W->get_shape()[0], im2col_output->get_shape()[1], W->get_shape()[1],
        1.0f,
        W, W->get_shape()[1],
        im2col_output, im2col_output->get_shape()[1],
        0.0f,
        result_ptr, result_ptr->get_shape()[1]);

    // Reshape the flattened result
    result_ptr->reshape({get_batch_size(), get_out_channels(), get_out_height(), get_out_width()});

    // Provided a bias, add it to the result tensor across each output feature
    if (B.has_value()) {
        add_bias(result_ptr);
    }

    // Write over the content of the output with the result of the convolution
    *Y = *result_ptr;
}

template <typename T>
bool ConvNode<T>::areInputsFilled() const {
    return X && X->get_size() > 0 &&
           W && W->get_size() > 0 &&
           (!B.has_value() || (B.value() && B.value()->get_size() > 0));
}

template <typename T>
void ConvNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
    if (inputs.size() > 0) {
        auto x_value = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
        *X = *x_value;
    }

    if (inputs.size() > 1) {
        auto w_value = std::get<std::shared_ptr<AbstractTensor>>(inputs[1]);
        *W = *w_value;
    }

    if (inputs.size() > 2 && B.has_value()) {
        auto b_value = std::get<std::shared_ptr<AbstractTensor>>(inputs[2]);
        *B.value() = *b_value;
    }
}

template <typename T>
bool ConvNode<T>::areOutputsFilled() const {
    return Y && Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> ConvNode<T>::getOutputs() const {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}

template <typename T>
void ConvNode<T>::flip_kernel() {
    int height = get_kernel_height();
    int width = get_kernel_width();

    for (int f = 0; f < get_out_channels(); f++) {
        for (int c = 0; c < get_in_channels(); c++) {
            // Flip horizontally
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width / 2; w++) {
                    auto tmp = (*W)[{f, c, h, w}];
                    (*W)[{f, c, h, w}] = (*W)[{f, c, h, width - 1 - w}];
                    (*W)[{f, c, h, width - 1 - w}] = tmp;
                }
            }

            // Flip vertically
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height / 2; h++) {
                    auto tmp = (*W)[{f, c, h, w}];
                    (*W)[{f, c, h, w}] = (*W)[{f, c, height - h - 1, w}];
                    (*W)[{f, c, height - 1 - h, w}] = tmp;
                }
            }
        }
    }
}

template <typename T>
void ConvNode<T>::im2col(shared_ptr<Tensor<T>> input, shared_ptr<Tensor<T>> output) {
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
}

template <typename T>
void ConvNode<T>::add_bias(shared_ptr<Tensor<T>> result_ptr) {
    // We first have to retrieve the bias tensor inside the optional
    auto& bias_tensor = *B;

    for (int b = 0; b < get_batch_size(); b++) {
        for (int i = 0; i < get_out_channels(); ++i) {
            for (int h = 0; h < get_out_height(); ++h) {
                for (int w = 0; w < get_out_width(); ++w) {
                    int index = ((b * get_out_channels() + i) * get_out_height() + h) * get_out_width() + w;

                    // Each value in bias vector is added to one entire out feature at a time
                    (*result_ptr)[index] += (*bias_tensor)[i];
                }
            }
        }
    }
}

template <typename T>
void ConvNode<T>::validate_inputs() {
    if (!areInputsFilled())
        throw runtime_error("ConvNode inputs are not fully set.");

    auto x_shape = X->get_shape();
    if (x_shape.size() != 4)
        throw runtime_error("Input tensor must have 4 dimensions: (Features x Channels x Height x Width)");

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

template <typename T>
void ConvNode<T>::update_parameters() {
    kernel_height = W->get_shape()[2];
    kernel_width = W->get_shape()[3];
    batch_size = X->get_shape()[0];
    in_channels = X->get_shape()[1];

    in_height = X->get_shape()[2];
    in_width = X->get_shape()[3];
    out_channels = W->get_shape()[0];
}

template <typename T>
int ConvNode<T>::get_batch_size() const { return batch_size; }

template <typename T>
int ConvNode<T>::get_in_channels() const { return in_channels; }

template <typename T>
int ConvNode<T>::get_in_height() const { return in_height; }

template <typename T>
int ConvNode<T>::get_in_width() const { return in_width; }

template <typename T>
int ConvNode<T>::get_kernel_height() const { return kernel_height; }

template <typename T>
int ConvNode<T>::get_kernel_width() const { return kernel_width; }

template <typename T>
int ConvNode<T>::get_out_channels() const { return out_channels; }

template <typename T>
int ConvNode<T>::get_stride_height() const { return stride[0]; }

template <typename T>
int ConvNode<T>::get_stride_width() const { return stride[1]; }

template <typename T>
int ConvNode<T>::get_padding_top() const { return padding[0]; }

template <typename T>
int ConvNode<T>::get_padding_bottom() const { return padding[1]; }

template <typename T>
int ConvNode<T>::get_padding_left() const { return padding[2]; }

template <typename T>
int ConvNode<T>::get_padding_right() const { return padding[3]; }

template <typename T>
int ConvNode<T>::get_out_height() {
    return (get_in_height() + get_padding_top() + get_padding_bottom() - get_kernel_height()) / get_stride_height() + 1;
}

template <typename T>
int ConvNode<T>::get_out_width() {
    return (get_in_width() + get_padding_left() + get_padding_right() - get_kernel_width()) / get_stride_width() + 1;
}


