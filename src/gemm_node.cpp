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

      std::optional<std::shared_ptr<Tensor<ValueType>>> c_data = std::nullopt;
      if (C.has_value()) {
        auto c_it = iomap.find(C.value());
        if (c_it == iomap.end()) {
          throw std::runtime_error("GemmNode: Input tensor C not found in iomap");
        }
        c_data = std::get<std::shared_ptr<Tensor<ValueType>>>(c_it->second);
      }
      
      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        throw std::runtime_error("GemmNode: Output tensor Y not found in iomap");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

      auto shape_a = a_ptr->get_shape();
      auto shape_b = b_ptr->get_shape();
      if (shape_a.size() < 2) throw runtime_error("Tensor A must be at least 2D.");
      if (shape_b.size() < 2) throw runtime_error("Tensor B must be at least 2D.");
      if (shape_b[0] != shape_a[1]) throw runtime_error("GemmNode: Dimension mismatch between A and B.");

      //OnnxGemm_mml<ValueType> gemm;
      //*y_ptr = *gemm.gemm_inner_product(a_ptr, b_ptr, alpha, beta, transA, transB, c_data);
    }
  }, a_tensor);
}