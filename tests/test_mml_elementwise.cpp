#include <cassert>
#include <iostream>
#include <modularml>
#include <vector>
#include <modularml>

#define assert_msg(name, condition)                         \
    if (!(condition)) {                                       \
        std::cerr << "Assertion failed: " << name << std::endl; \
    }                                                         \
    assert(condition);                                        \
    std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;

float square(float x) {
    return x * x;
}

int main() {
    Tensor<float> t1 = tensor_mll({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<float> t2 = tensor_mll({3, 3}, {1, 4, 9, 16, 25, 36, 49, 64, 81});

    // Test elementwise_apply function
    elementwise_apply(t1, square);
    assert_msg("Elementwise apply test", t1 == t2);

    std::cout << "All tests completed!" << std::endl;

    return 0;
}