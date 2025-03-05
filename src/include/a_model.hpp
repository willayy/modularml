#pragma once

#include "a_layer.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"

/**
 * @class Model
 * @brief Abstract Model class/type.
 *
 * This class provides an interface for models
 */

/// @brief Abstract Model class/type.

template <typename T>
class Model {
 protected:
  /// @brief Dynamic array of Layers
  /// @details Structure to be defined for each derived model
  vector<Layer<T>> layers;

  /// @brief Default constructor for Model.
  explicit Model() = default;

 public:
  /// @brief Virtual destructor for Model.
  /// @details Ensures derived class destructors are called properly.
  virtual ~Model() = default;

  /// @brief Make an inference.
  /// @param t Input data
  /// @return Predicted data
  virtual shared_ptr<Tensor<T>> infer(const shared_ptr<Tensor<T>> t) const = 0;

  /// @brief Print overview of model
  /// @details Visualize the model in the command line. Structure might vary
  /// depending on model
  virtual void print() const = 0;
};
