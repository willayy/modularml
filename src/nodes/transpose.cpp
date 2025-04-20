#include "nodes/transpose.hpp"


template <typename T>
class Tensor_mml;

TransposeNode::TransposeNode(std::string A, std::string Y, const std::vector<int> perm)
    : A(A),
      Y(Y),
      perm(perm) {}

TransposeNode::TransposeNode(const nlohmann::json &node) {
    if (node.contains("input") && node["input"].is_array()) {
    A = node["input"][0];
    }

    if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
    }

    if (node.contains("attribute") && node["attribute"].is_array()) {
        std::vector<int> perm_vec;
        for (const auto &attr : node["attribute"]) {
            if (attr["name"] == "perm") {
                for (const auto &el : attr["ints"]) {
                    perm_vec.push_back(std::stoul(el.get<std::string>()));
                }
            }
        }
        perm = perm_vec;
    }
}

void TransposeNode::forward(std::unordered_map<std::string, GeneralDataTypes> &iomap) {
    auto a_it = iomap.find(A);
    if (a_it == iomap.end()) {
    throw std::runtime_error("MatMul: Input tensor A not found in iomap");
    }

    const GeneralDataTypes &a_tensor = a_it->second;

    std::visit([&](const auto &a_ptr) {
        using ValueTypeA = std::decay_t<decltype(a_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeA, T>) {
            throw std::runtime_error("MatMul: Unsupported data type for tensor A");
        }

        auto new_a_ptr = a_ptr->copy();

        std::shared_ptr<Tensor<ValueTypeA>> new_c_ptr;
        auto c_it = iomap.find(Y);
        auto raw_c_ptr =
            std::get<std::shared_ptr<Tensor<ValueTypeA>>>(c_it->second)
                ->copy();

        iomap[Y] = new_c_ptr;
    },
    a_tensor);
}

std::vector<std::string> TransposeNode::getInputs() {
    return {A};
}

std::vector<std::string> TransposeNode::getOutputs() { return {Y}; }