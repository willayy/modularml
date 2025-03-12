#pragma once

#include <cmath>
#include <stdexcept>

#include "a_node.hpp"
#include "string.h"

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
   * @param k A 2x2 vector of integers representing the kernel shape/pooling
   * window of the layer.
   * @param s A 2x2 vector of integers representing the strides of the layer.
   * @param p An optional parameter representing the padding of the layer.
   * It has a default value of "valid" (no padding) and can also accept "same"
   * (padding to preserve the input dimensions).
   */
  PoolingNode_mml(vector<int> kernel_shape, vector<int> strides,
                  shared_ptr<Tensor<T>> input, shared_ptr<Tensor<T>> output,
                  string auto_pad = "NOTSET", int ceiling_mode = 0,
                  vector<int> dilations = {1, 1},
                  vector<int> pads = {0, 0, 0, 0, 0, 0, 0, 0});

  /**
   * @brief Forward function that propogates the input tensor through the
   * pooling operation.
   * @param input A shared pointer to the input tensor.
   * @return Returns a shared pointer to a new tensor with reduced spatial
   * dimensions.
   */
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
   * @returns Returns a value of type T that will be placed at the current index
   * of the output tensor.
   */
  virtual T pooling(const shared_ptr<Tensor<T>> t, array_mml<int> shape,
                    int element, int channel, int in_row_start,
                    int in_col_start) const = 0;

protected:
  //--------Inputs----------

  ///@brief Input tensor
  shared_ptr<Tensor<T>> input;

  ///@brief Output tensor
  shared_ptr<Tensor<T>> output;

  //--------Attributes------
  ///@brief A 2x2 vector of integers representing the filter/pooling window.
  vector<int> kernel_shape;
  ///@brief A 2x2 vector of integers representing the stride of the window.
  vector<int> strides;
  /// @brief A string representing the padding type applied by the layer.
  /// Can be either "valid" (no padding) or "same" (padding to preserve the
  /// input dimensions).
  string auto_pad;

  int ceil_mode;

  vector<int> dilations;
  vector<int> pads;
};
#include "../mml_pooling_node.tpp"