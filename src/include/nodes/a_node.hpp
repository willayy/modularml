#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>  // IWYU pragma: keep

#include "backend/mml_arithmetic.hpp"
#include "backend/mml_gemm.hpp"
#include "datastructures/a_tensor.hpp"
#include "datastructures/mml_tensor.hpp"
#include "datastructures/tensor_operations_module.hpp"
#include "nodes/node_utils.hpp"

template <typename T, typename Variant>
struct is_in_variant;

// Specialization for std::variant types using a fold expression
template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

// Helper variable template for convenience
template <typename T, typename Variant>
constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

template <typename Variant>
struct TensorVariantMaker;

// Specialization for std::variant types:
template <typename... Ts>
struct TensorVariantMaker<std::variant<Ts...>> {
  using type = std::variant<std::shared_ptr<Tensor<Ts>>...>;
};

// Helper type alias for convenience
template <typename Variant>
using TensorVariant = typename TensorVariantMaker<Variant>::type;

// Type constraints: no bfloat16 or float16 for now (not native to c++ 17). Also
// maybe exists more don't know.
using GeneralDataTypes = std::variant<
    std::shared_ptr<Tensor<bool>>, std::shared_ptr<Tensor<double>>,
    std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<int16_t>>,
    std::shared_ptr<Tensor<int32_t>>, std::shared_ptr<Tensor<int64_t>>,
    std::shared_ptr<Tensor<int8_t>>, std::shared_ptr<Tensor<uint16_t>>,
    std::shared_ptr<Tensor<uint32_t>>,
    std::shared_ptr<Tensor<uint64_t>>,  // Think this is meant to be
                                        // unsigned long long or uint64_t
                                        // not unsigned long int
    std::shared_ptr<Tensor<uint8_t>>>;

/**
 * @class Node
 * @brief Abstract base class representing a node in a computational graph.
 *
 * This class defines the interface for nodes in a computational graph,
 * including methods for forward propagation, input/output management,
 * and checking the status of inputs and outputs.
 */
class Node {
 public:
  /**
   * @brief Perform the forward pass computation.
   *
   * This pure virtual std::function must be overridden by derived classes to
   * implement the specific forward pass logic. It modifies the output(s)
   * in place.
   */
  virtual void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) = 0;

  /**
   * @brief Get inputs.
   *
   * This pure virtual std::function must be overridden by derived classes to
   * @return The names of the inputs to the node.
   */
  virtual std::vector<std::string> getInputs() = 0;

  /**
   * @brief Get outputs.
   *
   * This pure virtual std::function must be overridden by derived classes to
   * @return The names of the outputs to the node.
   */
  virtual std::vector<std::string> getOutputs() = 0;

  /**
   * @brief Virtual destructor for the Node class.
   *
   * Ensures derived class destructors are called properly.
   */
  virtual ~Node() = default;
};
