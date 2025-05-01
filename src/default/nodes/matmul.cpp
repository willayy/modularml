#include "nodes/matmul.hpp"

MatMulNode::MatMulNode(std::string A, std::string B, std::string Y)
    : A(A), B(B), Y(Y) {}

MatMulNode::MatMulNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    A = node["input"][0];
    B = node["input"][1];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }
}

void MatMulNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto a_it = iomap.find(A);
  if (a_it == iomap.end()) {
    throw std::runtime_error("MatMul: Input tensor A not found in iomap");
  }

  auto b_it = iomap.find(B);
  if (b_it == iomap.end()) {
    throw std::runtime_error("MatMul: Input tensor B not found in iomap");
  }

  const GeneralDataTypes &a_tensor = a_it->second;
  const GeneralDataTypes &b_tensor = b_it->second;

  std::visit(
      [&](const auto &a_ptr, const auto &b_ptr) {
        using ValueTypeA =
            std::decay_t<decltype(a_ptr)>::element_type::value_type;
        using ValueTypeB =
            std::decay_t<decltype(b_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeA, T> ||
                      !std::is_same_v<ValueTypeA, ValueTypeB>) {
          throw std::runtime_error(
              "MatMul: Unsupported data type for tensor A");
        } else {
          if (!a_ptr->is_matrix() || !b_ptr->is_matrix()) {
            throw std::runtime_error(
                "MatMul: Input tensors must be 2D matrices");
          }

          auto new_a_ptr = a_ptr->copy();
          auto new_b_ptr = b_ptr->copy();

          array_mml<size_t> a_shape = new_a_ptr->get_shape();
          array_mml<size_t> b_shape = new_b_ptr->get_shape();

          size_t M = a_shape[0];
          size_t K_a = a_shape[1];
          size_t K_b = b_shape[0];
          size_t N = b_shape[1];

          if (K_a != K_b) {
            throw std::runtime_error(
                "MatMul: Inner dimensions of A and B must match");
          }

          size_t lda = K_a;
          size_t ldb = N;
          size_t ldc = N;

          std::shared_ptr<Tensor<ValueTypeA>> new_c_ptr;
          auto c_it = iomap.find(Y);
          auto raw_c_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second)
                  ->copy();
          new_c_ptr = raw_c_ptr->broadcast_reshape({M, N});

          TensorOperations<ValueTypeA>::gemm(0, 0, M, N, K_a, 1.0, 0.0, 
                                             new_a_ptr, lda, 
                                             new_b_ptr, ldb, 
                                             new_c_ptr, ldc);

          iomap[Y] = new_c_ptr;
        }
      },
      a_tensor, b_tensor);
}

std::vector<std::string> MatMulNode::getInputs() { return {A, B}; }

std::vector<std::string> MatMulNode::getOutputs() { return {Y}; }