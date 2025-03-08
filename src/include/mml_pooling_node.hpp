#pragma once

#include <cmath>
#include <stdexcept>

#include "a_node.hpp"
#include "string.h"

/**
 * @class PoolingLayer
 * @brief Base class for pooling layers (e.g., MaxPooling, AveragePooling).
 * @details The PoolingLayer class is the base class for pooling operations such as max pooling and average pooling.
 * It performs a pooling operation on the input tensor, reducing its spatial dimensions (height and width)
 * while retaining the essential features within the defined pooling window or filter.
 * @tparam T The datatype of elements in the input tensor (e.g., float, double).
 */
class PoolingNode_mml : public Node {
 public:
  /**
   * @brief Base constructor for PoolingLayer.
   * @param f A 2x2 vector of integers representing the filter or pooling window of the layer.
   * @param s A 2x2 vector of integers representing the stride of the layer.
   * @param p An optional parameter representing the padding of the layer.
   * It has a default value of "valid" (no padding) and can also accept "same" (padding to preserve the input dimensions).
   */
  PoolingLayer(vector<int> f, vector<int> s, string p = "valid");

  /**
   * @brief Forward function that propogates the input tensor through the pooling operation.
   * @param input A shared pointer to the input tensor.
   * @return Returns a shared pointer to a new tensor with reduced spatial dimensions.
   */
  shared_ptr<Tensor<GeneralDataTypes>> forward(const shared_ptr<Tensor<GeneralDataTypes>> input) const override;

  /**
   * @brief Performs the specific pooling operation of the derived class, e.g. calculate the Maximum or average values within the pooling window.
   * @param t A shared pointer to the input tensor.
   * @param shape A array_mml of integers, representing the shape of the input tensor.
   * @param element Index of the current element of the batch that is being processed.
   * @param channel Index of the current channel being processed.
   * @param in_row_start Index of the first row in the window that is being processed.
   * @param in_col_start Index of the first column in the window that is being processed.
   * @returns Returns a value of type T that will be placed at the current index of the output tensor.
   */
  virtual T pooling(const shared_ptr<Tensor<GeneralDataTypes>> t, array_mml<int> shape, int element,
                    int channel, int in_row_start, int in_col_start) const = 0;

 protected:
  ///@brief A 2x2 vector of integers representing the filter/pooling window.
  vector<int> filter;
  ///@brief A 2x2 vector of integers representing the stride of the window.
  vector<int> stride;
  /// @brief A string representing the padding type applied by the layer.
  /// Can be either "valid" (no padding) or "same" (padding to preserve the input dimensions).
  string padding;
};

#include "../mml_pooling_node.tpp"