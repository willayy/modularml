#include "nodes/add.hpp"

#include <stddef.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "backend/mml_arithmetic.hpp"
#include "datastructures/mml_array.hpp"
#include "nlohmann/json.hpp"

AddNode::AddNode(std::string A, std::string B, std::string C)
    : A(A), B(B), C(C) {}

AddNode::AddNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    A = node["input"][0];
    B = node["input"][1];
  }

  if (node.contains("output") && node["output"].is_array()) {
    C = node["output"][0];
  }
}

void AddNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto a_it = iomap.find(A);
  if (a_it == iomap.end()) {
    throw std::runtime_error("AddNode: Input tensor A not found in iomap");
  }

  auto b_it = iomap.find(B);
  if (b_it == iomap.end()) {
    throw std::runtime_error("AddNode: Input tensor B not found in iomap");
  }

  const GeneralDataTypes &a_tensor = a_it->second;
  const GeneralDataTypes &b_tensor = b_it->second;

  std::visit(
      [&](const auto &a_ptr, const auto &b_ptr) {
        using ValueTypeA =
            typename std::decay_t<decltype(a_ptr)>::element_type::value_type;
        using ValueTypeB =
            typename std::decay_t<decltype(b_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeA, T> ||
                      !std::is_same_v<ValueTypeA, ValueTypeB>) {
          throw std::runtime_error(
              "AddNode: Unsupported data type for tensors A and B");
        } else {
          auto c_it = iomap.find(C);
          if (c_it == iomap.end()) {
            // Create output tensor if it doesn't exist
            auto c_ptr = a_ptr->copy();
            // No need to fill with zeros as the gemm_inner_product
            // std::function will overwrite the values
            iomap[C] = c_ptr;
            c_it = iomap.find(C);
          } else if (!std::holds_alternative<
                         std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second)) {
            throw std::runtime_error(
                "AddNode: Output tensor C has incorrect type");
          }

          auto c_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second);

          auto A_shape = a_ptr->get_shape();
          auto B_shape = b_ptr->get_shape();
          auto A_rank = A_shape.size();
          auto B_rank = B_shape.size();
          auto max_rank = std::max(A_rank, B_rank);
          bool broadcast_comp = true;

          // Check if broadcasting is possible
          for (size_t i = 0; i < max_rank; i++) {
            size_t dim_A = (i < A_rank) ? A_shape[A_rank - 1 - i] : 1;
            size_t dim_B = (i < B_rank) ? B_shape[B_rank - 1 - i] : 1;

            // Valid if dimensions match or one of them is 1
            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
              broadcast_comp = false;  // Incompatible for broadcasting
            }
          }

          Arithmetic_mml<ValueTypeA> arithmetic;

          // Valid case:
          if (A_shape == B_shape) {
            if (c_ptr->get_shape() != A_shape) {
              c_ptr->reshape(A_shape);  // Reshape output tensor to be the same
                                        // as input tensors
            }
            arithmetic.add(a_ptr, b_ptr, c_ptr);
            // Broadcasting case:
          } else if (broadcast_comp) {
            broadcast_addition(a_ptr, b_ptr, c_ptr);
            // Invalid case:
          } else {
            throw std::runtime_error(
                "Incompatible shapes for addition attempt in AddNode. "
                "Broadcasting impossible.");
          }
        }
      },
      a_tensor, b_tensor);
}

void AddNode::broadcast_addition(const TensorT &a_ptr, const TensorT &b_ptr,
                                 const TensorT &c_ptr) const {
  std::visit(
      [&](const auto &a_ptr, const auto &b_ptr, const auto &c_ptr) {
        auto A_shape = a_ptr->get_shape();
        auto B_shape = b_ptr->get_shape();
        auto A_rank = A_shape.size();
        auto B_rank = B_shape.size();
        auto max_rank = std::max(A_rank, B_rank);

        // Compute output shape based on broadcasting rules
        array_mml<size_t> output_shape(max_rank);
        std::fill(output_shape.begin(), output_shape.end(), 1);
        for (size_t i = 0; i < max_rank; i++) {
          size_t dim_A = (i < A_rank) ? A_shape[A_rank - 1 - i] : 1;
          size_t dim_B = (i < B_rank) ? B_shape[B_rank - 1 - i] : 1;

          switch ((dim_A == dim_B) ? 0
                  : (dim_A == 1)   ? 1
                  : (dim_B == 1)   ? 2
                                   : 3) {
            case 0:
              output_shape[max_rank - 1 - i] = dim_A;
              break;
            case 1:
              output_shape[max_rank - 1 - i] = dim_B;
              break;
            case 2:
              output_shape[max_rank - 1 - i] = dim_A;
              break;
            default:
              throw std::runtime_error("Incompatible shapes for broadcasting.");
          }
        }

        c_ptr->reshape(output_shape);

        std::vector<size_t> A_strides(A_rank, 1);
        std::vector<size_t> B_strides(B_rank, 1);
        std::vector<size_t> output_strides(max_rank, 1);

        // Compute strides for each tensor
        for (size_t i = max_rank - 2; ((int)i) >= 0; --i) {
          output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }
        for (size_t i = A_rank - 2; ((int)i) >= 0; --i) {
          A_strides[i] = A_strides[i + 1] * A_shape[i + 1];
        }
        for (size_t i = B_rank - 2; ((int)i) >= 0; --i) {
          B_strides[i] = B_strides[i + 1] * B_shape[i + 1];
        }

        // Iterate through the output tensor
        for (size_t flat_idx = 0; flat_idx < c_ptr->get_size(); flat_idx++) {
          size_t A_idx = 0, B_idx = 0;
          size_t remaining = flat_idx;

          // Compute multi-dimensional indices on the fly
          for (size_t j = 0; j < max_rank; j++) {
            size_t coord =
                remaining / output_strides[j];  // Extract coordinate for dim j
            remaining %= output_strides[j];

            size_t dim_A = (j < A_rank) ? A_shape[A_rank - max_rank + j] : 1;
            size_t dim_B = (j < B_rank) ? B_shape[B_rank - max_rank + j] : 1;

            if (dim_A > 1) A_idx += coord * A_strides[j];
            if (dim_B > 1) B_idx += coord * B_strides[j];
          }

          // Perform element-wise addition
          auto value_A = (*a_ptr)[A_idx];
          auto value_B = (*b_ptr)[B_idx];
          (*c_ptr)[flat_idx] = value_A + value_B;
        }
      },
      a_ptr, b_ptr, c_ptr);
}

std::vector<std::string> AddNode::getInputs() { return {A, B}; }

std::vector<std::string> AddNode::getOutputs() { return {C}; }