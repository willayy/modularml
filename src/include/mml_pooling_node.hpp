#pragma once

#include <cmath>
#include <stdexcept>

/**
 * @class PoolingNode_mml
 * @brief Base class for pooling nodes (e.g., MaxPooling, AveragePooling).
 * @details The PoolingNode_mml class is the base class for pooling operations
 * such as max pooling and average pooling. It performs a pooling operation on
 * the input tensor, reducing its spatial dimensions (height and width) while
 * retaining the essential features within the defined pooling window or filter.
 * @tparam T The datatype of elements in the input tensor (e.g., float, double).
 */
template <typename T> class PoolingNode_mml : public Node {
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
          std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
      "PoolingNode_mml supports only float, double, int32_t, int64_t");

public:
  /**
   * @brief Base constructor for PoolingNode.
   * @param kernel_shape A 2x2 array_mml of integers representing the kernel
   * shape/pooling window of the layer.
   * @param strides A 2x2 array_mml of integers representing the strides of the
   * layer.
   * @param input Pointer to the input tensor
   * @param auto_pad (OPTIONAL Parameter representing the padding of the
   * layer. It has a default value of "NOTSET" (no padding) and can also accept
   * "VALID", "SAME_UPPER" and "SAME_LOWER"
   * @param ceil_mode (OPTIONAL) Whether the output shape should be
   * calcualted with ceil or floor. Accepted values 1 for ceil and 0 for floor.
   * @param dilations (OPTIONAL) Value for dilution of kernel_shape. Default
   * value {1,1}.
   * @param pads (NOT SUPPORTED)
   */
  PoolingNode_mml(array_mml<uli> kernel_shape, array_mml<uli> strides,
                  shared_ptr<Tensor<T>> input, string auto_pad = "NOTSET",
                  uli ceil_mode = 0, array_mml<uli> dilations = {1, 1},
                  array_mml<uli> pads = {0, 0, 0, 0, 0, 0, 0, 0});

  void forward() override;

  bool areInputsFilled() const override;

  void setInputs(const array_mml<GeneralDataTypes> &inputs) override;

  bool areOutputsFilled() const override;

  array_mml<GeneralDataTypes> getOutputs() const override;

  /**
   * @brief Performs the specific pooling operation of the derived class, e.g.
   * calculate the Maximum or average values within the pooling window.
   * @param t A shared pointer to the input tensor.
   * @param shape A array_mml of integers, representing the shape of the input
   * tensor.
   * @param element Index of the current element of the batch that is being
   * processed.
   * @param channel Index of the current channel being processed.
   * @param in_row_start Index of the first row in the window that is being
   * processed.
   * @param in_col_start Index of the first column in the window that is being
   * processed.
   */

protected:
  virtual void pooling(const shared_ptr<Tensor<T>> t,
                       array_mml<uli> input_shape, array_mml<uli> output_shape,
                       array_mml<uli> effective_kernel_shape, uli pad_h,
                       uli pad_w, string auto_pad) = 0;
  //--------Inputs----------

  ///@brief Input tensor
  shared_ptr<Tensor<T>> input;

  //--------Attributes------
  ///@brief A 2x2 array_mml of integers representing the filter/pooling window.
  array_mml<uli> kernel_shape;
  ///@brief A 2x2 array_mml of integers representing the stride of the window.
  array_mml<uli> strides;
  /// @brief A string representing the padding type applied by the layer.
  /// Can be either "valid" (no padding) or "same" (padding to preserve the
  /// input dimensions).
  string auto_pad;

  ///@brief Output tensors
  array_mml<GeneralDataTypes> output;

  uli ceil_mode;

  array_mml<uli> dilations;
  array_mml<uli> pads;
};
#include "../mml_pooling_node.tpp"