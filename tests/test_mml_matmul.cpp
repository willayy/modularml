#include <cassert>
#include <iostream>
#include <modularml>

#include "mml_matmul.hpp"

#define assert_msg(name, condition)                             \
    if (!(condition)) {                                         \
        std::cerr << "Assertion failed: " << name << std::endl; \
    }                                                           \
    assert(condition);                                          \
    std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;

int main() {
    const Tensor<float> t_tensor = tensor_mll({2, 2}, {0.2, 0.4, 0.8, 1.6});

    const Tensor<float> weights = tensor_mll({2, 2}, {0.1, 0.2, 0.3, 0.4});

    MatMul<float> matmul_node = MatMul<float>(weights);

    assert_msg("Constructor successful", (1 == 1));

    Tensor<float> result = matmul_node.forward(t_tensor);

    assert_msg("Check matrix multiplication", (result[{0,0}]));
    
    //assert_msg("Forward calculation", result);
}