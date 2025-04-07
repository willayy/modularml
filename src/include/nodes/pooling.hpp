#pragma once

#include "nodes/a_node.hpp"

/**
 * @class PoolingNode_mml
 * @brief Base class for pooling nodes (e.g., MaxPooling, AveragePooling).
 * @details The PoolingNode_mml class is the base class for pooling
 * operations such as max pooling and average pooling. It performs a pooling
 * operation on the input tensor, reducing its spatial dimensions (height
 * and width) while retaining the essential features within the defined
 * pooling window or filter.
 */
class PoolingNode_mml : public Node {
public:
  using T = std::variant<float, double, int32_t, int64_t>;
  using TensorT =
      TensorVariant<T>; // Gets std::variant<std::shared_ptr<tensor<T>>,
                        // ...> from T

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
  PoolingNode_mml(std::string input, std::vector<std::string> outputs,
                  array_mml<uli> kernel_shape, array_mml<uli> strides,
                  std::string auto_pad = "NOTSET", uli ceil_mode = 0,
                  array_mml<uli> dilations = {1, 1},
                  array_mml<uli> pads = {0, 0, 0, 0, 0, 0, 0, 0});

  PoolingNode_mml(const nlohmann::json &node);

  void
  forward(std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  std::vector<std::string> getInputs() override;

  std::vector<std::string> getOutputs() override;

protected:
  virtual void
  pooling(const TensorT &t, array_mml<uli> input_shape,
          array_mml<uli> output_shape, array_mml<uli> effective_kernel_shape,
          uli pad_h, uli pad_w, std::string auto_pad,
          std::unordered_map<std::string, GeneralDataTypes> &iomap) = 0;
  //--------Inputs----------

  ///@brief Input tensor
  std::string input;
  ///@brief Output tensors
  std::vector<std::string> outputs;

  //--------Attributes------
  ///@brief A 2x2 array_mml of integers representing the filter/pooling window.
  array_mml<uli> kernel_shape;
  ///@brief A 2x2 array_mml of integers representing the stride of the window.
  array_mml<uli> strides;
  /// @brief A std::string representing the padding type applied by the layer.
  /// Can be either "valid" (no padding) or "same" (padding to preserve the
  /// input dimensions).
  std::string auto_pad;

  uli ceil_mode;

  array_mml<uli> dilations;
  array_mml<uli> pads;
};
