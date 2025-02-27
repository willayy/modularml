#include <cassert>
#include <iostream>
#include <modularml>

#define assert_msg(name, condition)                             \
    if (!(condition)) {                                         \
        std::cerr << "Assertion failed: " << name << std::endl; \
    }                                                           \
    assert(condition);                                          \
    std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;

int main() {
    Tensor<float> t_tensor = tensor_mll({2, 2}, {0.2, 0.4, 0.8, 1.6});

    Tensor<float> weights = tensor_mll({2, 2}, {0.1, 0.2, 0.3, 0.4});

    MatMul<float> matmul_node = MatMul<float>(weights);

    assert_msg("Test", (1 == 1));
}