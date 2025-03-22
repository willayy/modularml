#include "include/gemm_node.hpp"

GemmNode::GemmNode(std::string A,
                      std::string B,
                      std::string Y,
                      optional<std::string> C,
                      float alpha, float beta,
                      int transA, int transB)
    : A(A), B(B), C(C), Y(Y), alpha(alpha), beta(beta), transA(transA), transB(transB) {}

GemmNode::GemmNode(const json& node) {
  if (node.contains("inputs") && node["inputs"].is_array()) {
    A = node["inputs"][0];
    B = node["inputs"][1];
    if (node["inputs"].size() > 2) {
      C = node["inputs"][2];
    }
  }

  if (node.contains("outputs") && node["outputs"].is_array()) {
    Y = node["outputs"][0];
  }

  alpha = 1.0f;
  beta = 1.0f;
  transA = 0;
  transB = 0;
  if (node.contains("attributes") && node["attributes"].is_object()) {
    for (const auto& attr : node["attributes"]) {
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
  const GeneralDataTypes& a_tensor = a_it->second;

  std::visit([&](const auto& a_ptr) {
    using TensorPtr = std::decay_t<decltype(a_ptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("GemmNode: Unsupported data type for tensor A");
    } else {
      auto b_it = iomap.find(B);
      if (b_it == iomap.end()) {
        throw std::runtime_error("GemmNode: Output tensor Y not found in iomap");
      }

      auto b_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(b_it->second);

      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        // Create output tensor if it doesn't exist
        auto y_ptr = a_ptr->copy();
        // No need to fill with zeros as the gemm_inner_product function will overwrite the values
        iomap[Y] = y_ptr;
        y_it = iomap.find(Y);
      } else if (!std::holds_alternative<std::shared_ptr<Tensor<ValueType>>>(y_it->second)) {
        throw std::runtime_error("GemmNode: Output tensor Y has incorrect type");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

      auto shapeA = a_ptr->get_shape();

      if (shapeA.size() < 2) {
        throw runtime_error("Tensor A must be at least 2D.");
      }

      int M = shapeA[0];  // Number of rows.
      int K = shapeA[1];  // Number of columns of A.

      auto shapeB = b_ptr->get_shape();
      if (shapeB.size() < 2) {
        throw runtime_error("Tensor B must be at least 2D.");
      }
      if (shapeB[0] != K) {
        throw runtime_error("GemmNode: Dimension mismatch between A and B.");
      }

      int N = shapeB[1];  // Number of columns of B.

      int lda = K;
      int ldb = N;
      int ldc = N;

      // Handling optional C tensor not implemented directly in gemm_inner_product.
      // Will have to be done here instead by constructing suboptimal concrete tensor.
      // Gemm_inner_product could be modified to handle optional C tensor and take output Y.
      shared_ptr<Tensor_mml<ValueType>> new_c_ptr;
      if (C.has_value()) {
        auto c_it = iomap.find(C.value());
        if (c_it == iomap.end()) {
          throw std::runtime_error("GemmNode: Input tensor C not found in iomap");
        }
        auto c_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(c_it->second);
        new_c_ptr = std::dynamic_pointer_cast<Tensor_mml<ValueType>>(c_ptr);
        if (!new_c_ptr) {
          throw runtime_error("GemmNode: Failed to cast optional C to Tensor_mml<T>.");
        }
      } else {
        Tensor_mml<ValueType> zero_tensor({M, N});
        zero_tensor.fill(static_cast<ValueType>(0));
        new_c_ptr = make_shared<Tensor_mml<ValueType>>(zero_tensor);
      }

      Gemm_mml<ValueType> gemm;
      gemm.gemm_inner_product(0, 0, M, N, K, static_cast<ValueType>(alpha), a_ptr,
                              lda, b_ptr, ldb, static_cast<ValueType>(beta), new_c_ptr, ldc);

      *y_ptr = *new_c_ptr;
    }
  }, a_tensor);
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