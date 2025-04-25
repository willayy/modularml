#include "nodes/conv.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json.hpp"

template <typename T>
class Gemm_mml;
template <typename T>
class Tensor_mml;

ConvNode::ConvNode(const std::string &X, const std::string &W, const std::string &Y,
                   const array_mml<size_t> &dilations, const array_mml<size_t> &padding,
                   const array_mml<size_t> &kernel_shape, const array_mml<size_t> &stride,
                   const std::optional<std::string> &B, size_t group)
    : X(X),
      W(W),
      B(B),
      Y(Y),
      dilations(dilations),
      padding(padding),
      kernel_shape(kernel_shape),
      stride(stride) {
  if (dilations.size() != 2) {
    throw std::invalid_argument(
        "Invalid dilations size. Expected a std::vector of size 2, but got: " +
        std::to_string(dilations.size()) + ".");
  }

  if (padding.size() != 4) {
    throw std::invalid_argument(
        "Invalid padding std::vector size. Expected a "
        "std::vector of size 4, but got: " +
        std::to_string(padding.size()) + ".");
  }

  if (kernel_shape.size() != 2) {
    throw std::invalid_argument(
        "Invalid kernel_shape std::vector size. Expected a "
        "std::vector of size 2, but got: " +
        std::to_string(kernel_shape.size()) + ".");
  }

  if (stride.size() != 2) {
    throw std::invalid_argument(
        "Invalid stride std::vector size. Expected a "
        "std::vector of size 2, but got: " +
        std::to_string(stride.size()) + ".");
  }
}

ConvNode::ConvNode(const nlohmann::json &node) {
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
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "dilations") {
        std::vector<size_t> dilations_vec;
        for (const auto &el : attr["ints"]) {
          dilations_vec.push_back(std::stoul(el.get<std::string>()));
        }
        dilations = array_mml<size_t>(dilations_vec);
      } else if (attr["name"] == "pads") {
        std::vector<size_t> padding_vec;
        for (const auto &el : attr["ints"]) {
          padding_vec.push_back(std::stoul(el.get<std::string>()));
        }
        padding = array_mml<size_t>(padding_vec);
      } else if (attr["name"] == "kernel_shape") {
        std::vector<size_t> kernel_vec;
        for (const auto &el : attr["ints"]) {
          kernel_vec.push_back(std::stoul(el.get<std::string>()));
        }
        kernel_shape = array_mml<size_t>(kernel_vec);
      } else if (attr["name"] == "strides") {
        std::vector<size_t> stride_vec;
        for (const auto &el : attr["ints"]) {
          stride_vec.push_back(std::stoul(el.get<std::string>()));
        }
        stride = array_mml<size_t>(stride_vec);
      } else if (attr["name"] == "group") {
        group = std::stoul(attr["i"].get<std::string>());
      }
    }
  }
}

void ConvNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error("ConvNode: Input tensor X not found in iomap");
  }

  auto w_it = iomap.find(W);
  if (w_it == iomap.end()) {
    throw std::runtime_error("ConvNode: Input tensor W not found in iomap");
  }

  const GeneralDataTypes &x_tensor = x_it->second;
  const GeneralDataTypes &w_tensor = w_it->second;

  std::visit(
      [&](const auto &x_ptr, const auto &w_ptr) {
        using ValueTypeX =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;
        using ValueTypeW =
            typename std::decay_t<decltype(w_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeX, T> ||
                      !std::is_same_v<ValueTypeX, ValueTypeW>) {
          throw std::runtime_error(
              "ConvNode: Unsupported data type for tensor data");
        } else {
          if (x_ptr->get_shape().size() < 1) {
            throw std::runtime_error(
                "Input tensor must have 4 dimensions: "
                "(Features x Channels x Height x Width).");
          }

          auto y_it = iomap.find(Y);
          if (y_it == iomap.end()) {
            // Create output tensor if it doesn't exist
            auto y_ptr = x_ptr->copy();
            // No need to fill with zeros as the convolution std::function will
            // overwrite the values
            iomap[Y] = y_ptr;
            y_it = iomap.find(Y);
          } else if (!std::holds_alternative<
                         std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second)) {
            throw std::runtime_error(
                "ConvNode: Output tensor Y has incorrect type");
          }

          auto y_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

          // infer and update attributes first
          update_parameters(x_ptr->get_shape(), w_ptr->get_shape());

          // Create a std::copy of the input
          auto input_copy = x_ptr->copy();
          std::cout << "input shape: " << input_copy->get_shape() << std::endl; 
          // Begin by flipping the weight kernel
          
          flip_kernel(w_ptr);
          std::cout << "flip kernel done" << std::endl;

          auto im2col_output_shape = array_mml<size_t>(
              {get_in_channels() * get_kernel_height() * get_kernel_width(),
               get_batch_size() * get_out_height() * get_out_width()});

          auto im2col_output =
              std::make_shared<Tensor_mml<ValueTypeX>>(im2col_output_shape);

            
          im2col(input_copy, im2col_output);
          std::cout << "im2col done" << std::endl;
          
          // Flatten the weight tensor to prepare for GEMM
          size_t flattened_size =
              get_in_channels() * get_kernel_height() * get_kernel_width();
          w_ptr->reshape({get_out_channels(), flattened_size});
          std::cout << "weigth reshaping done" << std::endl;
          

          // Prepare the result tensor
          array_mml<size_t> result_shape(
              {w_ptr->get_shape()[0], im2col_output->get_shape()[1]});
          auto result_ptr =
              std::make_shared<Tensor_mml<ValueTypeX>>(result_shape);
          TensorOperations::gemm<ValueTypeX>(
              0, 0, w_ptr->get_shape()[0], im2col_output->get_shape()[1],
              w_ptr->get_shape()[1], 1.0f, w_ptr, w_ptr->get_shape()[1],
              im2col_output, im2col_output->get_shape()[1], 0.0f, result_ptr,
              result_ptr->get_shape()[1]);

          std::cout << "gemm done" << std::endl;

          result_ptr->reshape({get_batch_size(), get_out_channels(),
                               get_out_height(), get_out_width()});
          std::cout << "reshape done" << std::endl;
          
          // Provided a bias, add it to the result tensor across each output
          // feature
          if (B.has_value()) {
            auto b_it = iomap.find(B.value());
            if (b_it == iomap.end()) {
              throw std::runtime_error(
                  "ConvNode: Input tensor B not found in iomap");
            }
            auto b_ptr =
                std::get<std::shared_ptr<Tensor<ValueTypeX>>>(b_it->second);

            add_bias(result_ptr, b_ptr);
          }

          // Write over the content of the output with the result of the
          // convolution
          *y_ptr = *result_ptr;
        }
      },
      x_tensor, w_tensor);
}

std::vector<std::string> ConvNode::getInputs() {
  if (B.has_value()) {
    return {X, W, B.value()};
  } else {
    return {X, W};
  }
}

std::vector<std::string> ConvNode::getOutputs() { return {Y}; }

void ConvNode::flip_kernel(const TensorT &weight_variant) {
  std::cout << "inside flip kernel" << std::endl;
  std::visit(
      [this](auto &weight) {
        size_t height = get_kernel_height();
        size_t width = get_kernel_width();

        for (size_t f = 0; f < get_out_channels(); f++) {
          for (size_t c = 0; c < get_in_channels(); c++) {
            // Flip horizontally
            for (size_t h = 0; h < height; h++) {
              for (size_t w = 0; w < width / 2; w++) {
                auto tmp = (*weight)[{f, c, h, w}];
                (*weight)[{f, c, h, w}] = (*weight)[{f, c, h, width - 1 - w}];
                (*weight)[{f, c, h, width - 1 - w}] = tmp;
              }
            }

            // Flip vertically
            for (size_t w = 0; w < width; w++) {
              for (size_t h = 0; h < height / 2; h++) {
                auto tmp = (*weight)[{f, c, h, w}];
                (*weight)[{f, c, h, w}] = (*weight)[{f, c, height - h - 1, w}];
                (*weight)[{f, c, height - 1 - h, w}] = tmp;
              }
            }
          }
        }
      },
      weight_variant);
}

void ConvNode::im2col(const TensorT &input_variant,
                      const TensorT &output_variant) {
  std::visit(
      [this](auto &input, auto &output) {
        // Iterate over each image in the batch
        for (size_t n = 0; n < get_batch_size(); ++n) {
          for (size_t h = 0; h < get_out_height(); ++h) {
            for (size_t w = 0; w < get_out_width();
                 ++w) {  // Traverse into each batch

              size_t col_index =
                  h * get_out_width() + w;  // Column index in im2col matrix

              for (size_t c = 0; c < get_in_channels();
                   ++c) {  // If the input has multiple channels, iterate over
                           // each one

                // Here we loop over the kernel's height and width, simulating
                // how the kernel moves across the input tensor. For each
                // position of the kernel, the corresponding input values are
                // extracted and stored in the output tensor. If the kernel
                // extends beyond the boundaries of the input (due to padding or
                // stride), zero padding is added instead of the input values.
                for (size_t kh = 0; kh < get_kernel_height(); ++kh) {
                  for (size_t kw = 0; kw < get_kernel_width(); ++kw) {
                    size_t input_h =
                        h * get_stride_height() - get_padding_top() + kh;
                    size_t input_w =
                        w * get_stride_width() - get_padding_left() + kw;

                    if (input_h < 0 ||
                        input_h >= get_in_height() + get_padding_bottom() ||
                        input_w < 0 ||
                        input_w >= get_in_width() + get_padding_right()) {
                      (*output)[col_index] = 0;  // Padding
                    } else {
                      size_t row_index =
                          c * get_kernel_height() * get_kernel_width() +
                          kh * get_kernel_width() + kw;

                      size_t output_index =
                          row_index * (get_out_height() * get_out_width()) +
                          col_index;

                      size_t input_index =
                          n * (get_in_channels() * get_in_height() *
                               get_in_width()) +
                          c * (get_in_height() * get_in_width()) +
                          input_h * get_in_width() + input_w;

                      // Check if input index is valid
                      if (input_index >= 0 &&
                          input_index < get_in_channels() * get_in_height() *
                                            get_in_width()) {
                        (*output)[output_index] = (*input)[input_index];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      input_variant, output_variant);
}

void ConvNode::add_bias(const TensorT &result_variant,
                        const TensorT &bias_variant) {
  std::visit(
      [this](auto &result, auto &bias) {
        for (size_t b = 0; b < get_batch_size(); b++) {
          for (size_t i = 0; i < get_out_channels(); ++i) {
            for (size_t h = 0; h < get_out_height(); ++h) {
              for (size_t w = 0; w < get_out_width(); ++w) {
                size_t index =
                    ((b * get_out_channels() + i) * get_out_height() + h) *
                        get_out_width() +
                    w;

                // Each value in bias std::vector is added to one entire out
                // feature at a time
                (*result)[index] += (*bias)[i];
              }
            }
          }
        }
      },
      result_variant, bias_variant);
}

size_t ConvNode::get_batch_size() const { return batch_size; }

size_t ConvNode::get_in_channels() const { return in_channels; }

size_t ConvNode::get_in_height() const { return in_height; }

size_t ConvNode::get_in_width() const { return in_width; }

size_t ConvNode::get_kernel_height() const { return kernel_height; }

size_t ConvNode::get_kernel_width() const { return kernel_width; }

size_t ConvNode::get_out_channels() const { return out_channels; }

size_t ConvNode::get_stride_height() const { return stride[0]; }

size_t ConvNode::get_stride_width() const { return stride[1]; }

size_t ConvNode::get_padding_top() const { return padding[0]; }

size_t ConvNode::get_padding_bottom() const { return padding[1]; }

size_t ConvNode::get_padding_left() const { return padding[2]; }

size_t ConvNode::get_padding_right() const { return padding[3]; }

size_t ConvNode::get_out_height() {
  return (get_in_height() + get_padding_top() + get_padding_bottom() -
          get_kernel_height()) /
             get_stride_height() +
         1;
}

size_t ConvNode::get_out_width() {
  return (get_in_width() + get_padding_left() + get_padding_right() -
          get_kernel_width()) /
             get_stride_width() +
         1;
}

void ConvNode::update_parameters(const array_mml<size_t> &input_shape,
                                 const array_mml<size_t> &weight_shape) {
  kernel_height = weight_shape[2];
  kernel_width = weight_shape[3];
  batch_size = input_shape[0];
  in_channels = input_shape[1];

  in_height = input_shape[2];
  in_width = input_shape[3];
  out_channels = weight_shape[0];
}