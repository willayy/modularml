#include "include/gemm_node.hpp"

GemmNode::GemmNode(std::string A,
                      std::string B,
                      std::string Y,
                      optional<std::string> C,
                      float alpha, float beta,
                      int transA, int transB)
    : A(A), B(B), C(C), Y(Y), alpha(alpha), beta(beta), transA(transA), transB(transB) {}

GemmNode::GemmNode(const json& node) {
  if (node.contains("input") && node["input"].is_array()) {
    A = node["input"][0];
    B = node["input"][1];
    if (node["input"].size() > 2) {
      C = node["input"][2];
    }
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  alpha = 1.0f;
  beta = 1.0f;
  transA = 0;
  transB = 0;
  if (node.contains("attribute") && node["attribute"].is_object()) {
    for (const auto& attr : node["attribute"]) {
      if (attr["name"] == "alpha") {
        alpha = attr["f"];
      } else if (attr["name"] == "beta") {
        beta = attr["f"];
      } else if (attr["name"] == "transA") {
        transA = attr["i"];
      } else if (attr["name"] == "transB") {
        transB = attr["i"];
      }
    }
  }
}

void GemmNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto a_it = iomap.find(A);
  if (a_it == iomap.end()) {
    throw std::runtime_error("GemmNode: Input tensor A not found in iomap");
  }

  auto b_it = iomap.find(B);
  if (b_it == iomap.end()) {
    throw std::runtime_error("GemmNode: Output tensor Y not found in iomap");
  }

  const GeneralDataTypes& a_tensor = a_it->second;
  const GeneralDataTypes& b_tensor = b_it->second;

  std::visit([&](const auto& a_ptr, const auto& b_ptr) {
    using ValueTypeA = std::decay_t<decltype(a_ptr)>::element_type::value_type;
    using ValueTypeB = std::decay_t<decltype(b_ptr)>::element_type::value_type;
    
    if constexpr (!is_in_variant_v<ValueTypeA, T> || !std::is_same_v<ValueTypeA, ValueTypeB>) {
      throw std::runtime_error("GemmNode: Unsupported data type for tensor A");
    } else {
      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        // Create output tensor if it doesn't exist
        auto y_ptr = a_ptr->copy();
        // No need to fill with zeros as the gemm_inner_product function will overwrite the values
        iomap[Y] = y_ptr;
        y_it = iomap.find(Y);
      } else if (!std::holds_alternative<std::shared_ptr<Tensor<ValueTypeA>>>(y_it->second)) {
        throw std::runtime_error("GemmNode: Output tensor Y has incorrect type");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueTypeA>>>(y_it->second);

      auto shapeA = a_ptr->get_shape();

      if (shapeA.size() < 2) {
        throw runtime_error("Tensor A must be at least 2D.");
      }

      uli M = shapeA[0];  // Number of rows.
      uli K = shapeA[1];  // Number of columns of A.

      auto shapeB = b_ptr->get_shape();
      if (shapeB.size() < 2) {
        throw runtime_error("Tensor B must be at least 2D.");
      }
      if (shapeB[0] != K) {
        throw runtime_error("GemmNode: Dimension mismatch between A and B.");
      }

      uli N = shapeB[1];  // Number of columns of B.

      uli lda = K;
      uli ldb = N;
      uli ldc = N;

      // Handling optional C tensor not implemented directly in gemm_inner_product.
      // Will have to be done here instead by constructing suboptimal concrete tensor.
      // Gemm_inner_product could be modified to handle optional C tensor and take output Y.
      shared_ptr<Tensor_mml<ValueTypeA>> new_c_ptr;
      if (C.has_value()) {
        auto c_it = iomap.find(C.value());
        if (c_it == iomap.end()) {
          throw std::runtime_error("GemmNode: Input tensor C not found in iomap");
        }
        auto c_ptr = std::get<std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second);
        auto c_shape = c_ptr->get_shape();
        
        // Handle broadcasting for C
        if (c_shape.size() == 1 && c_shape[0] == N) {
          // C is a 1D vector [N], reshape to [1, N] for matrix operations
          auto reshaped_c = c_ptr->copy();
          reshaped_c->reshape({1, N});
          
          // Now expand to [M, N] if M > 1 by repeating the vector M times
          array_mml<uli> new_shape = array_mml<uli>({M, N});
          new_c_ptr = make_shared<Tensor_mml<ValueTypeA>>(new_shape);
          for (uli i = 0; i < M; i++) {
            for (uli j = 0; j < N; j++) {
              (*new_c_ptr)[i * N + j] = (*reshaped_c)[j];
            }
          }
        } 
        else if (c_shape.size() == 2 && c_shape[0] == M && c_shape[1] == N) {
          // C is already in the right shape [M, N]
          new_c_ptr = std::dynamic_pointer_cast<Tensor_mml<ValueTypeA>>(c_ptr);
          if (!new_c_ptr) {
            throw runtime_error("GemmNode: Failed to cast optional C to Tensor_mml<T>.");
          }
        } 
        else {
          throw runtime_error("GemmNode: Tensor C must be broadcastable to shape [" + 
                              std::to_string(M) + ", " + std::to_string(N) + "]");
        }
      } else {
        // If C is not provided, use zeros
        Tensor_mml<ValueTypeA> zero_tensor({M, N});
        zero_tensor.fill(static_cast<ValueTypeA>(0));
        new_c_ptr = make_shared<Tensor_mml<ValueTypeA>>(zero_tensor);
      }

      Gemm_mml<ValueTypeA> gemm;
      gemm.gemm_inner_product(0, 0, M, N, K, static_cast<ValueTypeA>(alpha), a_ptr,
                              lda, b_ptr, ldb, static_cast<ValueTypeA>(beta), new_c_ptr, ldc);

      *y_ptr = *new_c_ptr;
    }
  }, a_tensor, b_tensor);
}

std::vector<std::string> GemmNode::getInputs() {
  if (C.has_value()) {
    return {A, B, C.value()};
  } else {
    return {A, B};
  }
}

std::vector<std::string> GemmNode::getOutputs() {
  return {Y};
}