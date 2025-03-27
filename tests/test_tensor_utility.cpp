#include <gtest/gtest.h>

#include <modularml>

TEST(test_tensor_utility, test_kaiming_uniform_basic) {
    const int in_channels = 3;
    const int kernel_size = 3;
    const size_t num_elements = 27;

    auto tensor = tensor_mml_p<double>({num_elements});
    std::mt19937 gen(42);  // fixed seed for reproducibility

    kaimingUniform(tensor, in_channels, kernel_size, gen);

    const double limit = std::sqrt(6.0 / (in_channels * kernel_size * kernel_size));

    // Check that all values are within [-limit, +limit]
    for (size_t i = 0; i < tensor->get_size(); ++i) {
        double val = (*tensor)[i];
        ASSERT_GE(val, -limit);
        ASSERT_LE(val, limit);
    }

    // Optionally check that values are not all equal
    double first = (*tensor)[0];
    bool all_same = true;
    for (size_t i = 1; i < tensor->get_size(); ++i) {
        if ((*tensor)[i] != first) {
            all_same = false;
            break;
        }
    }
    ASSERT_FALSE(all_same);

    // Try generating another tensor with the same seed
    auto tensor2 = tensor_mml_p<double>({num_elements});
    gen.seed(42);  // Reset the generator to the same initial state
    kaimingUniform(tensor2, in_channels, kernel_size, gen);
    ASSERT_EQ(*tensor, *tensor2);
}

TEST(test_tensor_utility, test_kaiming_uniform_empty_tensor) {
    const auto in_channels = 3;
    const auto kernel_size = 3;

    // Empty tensor (zero elements)
    auto tensor = tensor_mml_p<double>({0});

    // Should not throw or crash
    ASSERT_NO_THROW(kaimingUniform(tensor, in_channels, kernel_size));
    ASSERT_EQ(tensor->get_size(), 0);  // Still zero
}

TEST(test_tensor_utility, test_kaiming_uniform_zero_fan_in) {
    const auto in_channels = 0;
    const auto kernel_size = 3;

    // Empty tensor (zero elements)
    auto tensor = tensor_mml_p<double>({3, 3});

    // Should throw
    ASSERT_THROW(kaimingUniform(tensor, in_channels, kernel_size), std::invalid_argument);
}

TEST(test_tensor_utility, test_kaiming_external_vs_internal) {
    const int in_channels = 3;
    const int kernel_size = 3;
    const size_t num_elements = 27;

    const double limit = std::sqrt(6.0 / (in_channels * kernel_size * kernel_size));

    // Tensor with external RNG (fixed seed)
    auto tensor_ext = tensor_mml_p<double>({num_elements});
    std::mt19937 gen(42);
    kaimingUniform(tensor_ext, in_channels, kernel_size, gen);

    // Tensor with internal RNG (random seed)
    auto tensor_int = tensor_mml_p<double>({num_elements});
    kaimingUniform(tensor_int, in_channels, kernel_size);  // overload with no gen

    // 1. Both tensors should have values within [-limit, limit]
    for (size_t i = 0; i < num_elements; ++i) {
        ASSERT_GE((*tensor_ext)[i], -limit);
        ASSERT_LE((*tensor_ext)[i], limit);

        ASSERT_GE((*tensor_int)[i], -limit);
        ASSERT_LE((*tensor_int)[i], limit);
    }

    // 2. Tensors should likely be different (not always guaranteed, but very likely)
    bool all_same = true;
    for (size_t i = 0; i < num_elements; ++i) {
        if ((*tensor_ext)[i] != (*tensor_int)[i]) {
            all_same = false;
            break;
        }
    }

    ASSERT_FALSE(all_same) << "Expected different outputs from external and internal RNGs";

    // 3. Ensure both aren't all the same value
    auto is_constant = [](const shared_ptr<Tensor<double>>& t) {
        double first = (*t)[0];
        for (size_t i = 1; i < t->get_size(); ++i) {
            if ((*t)[i] != first) return false;
        }
        return true;
    };

    ASSERT_FALSE(is_constant(tensor_ext)) << "External RNG tensor is unexpectedly constant";
    ASSERT_FALSE(is_constant(tensor_int)) << "Internal RNG tensor is unexpectedly constant";
}

