#include "nodes/gemm.hpp"

GemmNode::GemmNode(std::string A, std::string B, std::string Y,
                   std::optional<std::string> C, float alpha, float beta,
                   int transA, int transB)
    : A(A), B(B), C(C), Y(Y), alpha(alpha), beta(beta), transA(transA),
      transB(transB) {}

GemmNode::GemmNode(const nlohmann::json &node) {
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
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "alpha") {
        alpha = attr["f"];
      } else if (attr["name"] == "beta") {
        beta = attr["f"];
      } else if (attr["name"] == "transA") {
        transA = std::stoi(attr["i"].get<std::string>());
      } else if (attr["name"] == "transB") {
        transB = std::stoi(attr["i"].get<std::string>());
      }
    }
  }
}

void GemmNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto a_it = iomap.find(A);
  if (a_it == iomap.end()) {
    throw std::runtime_error("GemmNode: Input tensor A not found in iomap");
  }

  auto b_it = iomap.find(B);
  if (b_it == iomap.end()) {
    throw std::runtime_error("GemmNode: Output tensor Y not found in iomap");
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
              "GemmNode: Unsupported data type for tensor A");
        } else {
          if (!a_ptr->is_matrix() || !b_ptr->is_matrix()) {
            throw std::runtime_error(
                "GemmNode: Input tensors must be 2D matrices");
          }

          auto new_a_ptr = a_ptr->copy();
          auto new_b_ptr = b_ptr->copy();
          if (transA == 1) {
            new_a_ptr = a_ptr->transpose();
          }
          if (transB == 1) {
            new_b_ptr = b_ptr->transpose();
          }

          array_mml<size_t> a_shape = new_a_ptr->get_shape();
          array_mml<size_t> b_shape = new_b_ptr->get_shape();

          size_t M = a_shape[0];
          size_t K_a = a_shape[1];
          size_t K_b = b_shape[0];
          size_t N = b_shape[1];

          if (K_a != K_b) {
            throw std::runtime_error(
                "GemmNode: Inner dimensions of A and B must match");
          }

          std::shared_ptr<Tensor<ValueTypeA>> new_c_ptr;
          if (C.has_value()) {
            auto c_it = iomap.find(C.value());
            if (c_it == iomap.end()) {
              throw std::runtime_error(
                  "GemmNode: Output tensor C not found in iomap");
            }
            auto raw_c_ptr =
                std::get<std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second)
                    ->copy();
            new_c_ptr = raw_c_ptr->broadcast_to({M, N});
          } else {
            new_c_ptr = std::make_shared<Tensor_mml<ValueTypeA>>(
                array_mml<size_t>{M, N});
            new_c_ptr->fill(static_cast<ValueTypeA>(0));
          }

          size_t lda = K_a;
          size_t ldb = N;
          size_t ldc = N;

          Gemm_mml<ValueTypeA> gemm;
          gemm.gemm_inner_product(
              0, 0, M, N, K_a, static_cast<ValueTypeA>(alpha), new_a_ptr, lda,
              new_b_ptr, ldb, static_cast<ValueTypeA>(beta), new_c_ptr, ldc);

          iomap[Y] = new_c_ptr;
        }
      },
      a_tensor, b_tensor);
}

std::vector<std::string> GemmNode::getInputs() {
  if (C.has_value()) {
    return {A, B, C.value()};
  } else {
    return {A, B};
  }
}

std::vector<std::string> GemmNode::getOutputs() { return {Y}; }