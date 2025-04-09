#include <gtest/gtest.h>

#include <modularml>

// ---------    Kaiming Tests
TEST(test_tensor_utility, test_kaiming_uniform_basic) {
  const int in_channels = 3;
  const int kernel_size = 3;
  const size_t num_elements = 27;

  auto tensor = tensor_mml_p<double>({num_elements});
  std::mt19937 gen(42); // fixed seed for reproducibility

  kaiming_uniform(tensor, in_channels, kernel_size, gen);

  const double limit =
      std::sqrt(6.0 / (in_channels * kernel_size * kernel_size));

  // Check that all values are within [-limit, +limit]
  for (size_t i = 0; i < tensor->get_size(); ++i) {
    double val = (*tensor)[i];
    ASSERT_GE(val, -limit);
    ASSERT_LE(val, limit);
  }

  // Optionally check that values are not all std::equal
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
  gen.seed(42); // Reset the generator to the same initial state
  kaiming_uniform(tensor2, in_channels, kernel_size, gen);
  ASSERT_EQ(*tensor, *tensor2);
}

TEST(test_tensor_utility, test_kaiming_uniform_empty_tensor) {
  const auto in_channels = 3;
  const auto kernel_size = 3;

  // Empty tensor (zero elements)
  auto tensor = tensor_mml_p<double>({0});

  // Should not throw or crash
  ASSERT_NO_THROW(kaiming_uniform(tensor, in_channels, kernel_size));
  ASSERT_EQ(tensor->get_size(), 0); // Still zero
}

TEST(test_tensor_utility, test_kaiming_uniform_zero_fan_in) {
  const auto in_channels = 0;
  const auto kernel_size = 3;

  // Empty tensor (zero elements)
  auto tensor = tensor_mml_p<double>({3, 3});

  // Should throw
  ASSERT_THROW(kaiming_uniform(tensor, in_channels, kernel_size),
               std::invalid_argument);
}

TEST(test_tensor_utility, test_kaiming_external_vs_internal) {
  const int in_channels = 3;
  const int kernel_size = 3;
  const size_t num_elements = 27;

  const double limit =
      std::sqrt(6.0 / (in_channels * kernel_size * kernel_size));

  // Tensor with external RNG (fixed seed)
  auto tensor_ext = tensor_mml_p<double>({num_elements});
  std::mt19937 gen(42);
  kaiming_uniform(tensor_ext, in_channels, kernel_size, gen);

  // Tensor with internal RNG (random seed)
  auto tensor_int = tensor_mml_p<double>({num_elements});
  kaiming_uniform(tensor_int, in_channels, kernel_size); // overload with no gen

  // 1. Both tensors should have values within [-limit, limit]
  for (size_t i = 0; i < num_elements; ++i) {
    ASSERT_GE((*tensor_ext)[i], -limit);
    ASSERT_LE((*tensor_ext)[i], limit);

    ASSERT_GE((*tensor_int)[i], -limit);
    ASSERT_LE((*tensor_int)[i], limit);
  }

  // 2. Tensors should likely be different (not always guaranteed, but very
  // likely)
  bool all_same = true;
  for (size_t i = 0; i < num_elements; ++i) {
    if ((*tensor_ext)[i] != (*tensor_int)[i]) {
      all_same = false;
      break;
    }
  }

  ASSERT_FALSE(all_same)
      << "Expected different outputs from external and internal RNGs";

  // 3. Ensure both aren't all the same value
  auto is_constant = [](const std::shared_ptr<Tensor<double>> &t) {
    double first = (*t)[0];
    for (size_t i = 1; i < t->get_size(); ++i) {
      if ((*t)[i] != first)
        return false;
    }
    return true;
  };

  ASSERT_FALSE(is_constant(tensor_ext))
      << "External RNG tensor is unexpectedly constant";
  ASSERT_FALSE(is_constant(tensor_int))
      << "Internal RNG tensor is unexpectedly constant";
}

// ---------    Random tensor tests

TEST(test_tensor_utility, test_generate_random_tensor_basic) {
  using namespace std;

  // 1. Test with integer type
  {
    array_mml<uli> shape = {2, 3, 4}; // Total: 24 elements
    int lo = 10, hi = 20;
    auto tensor = generate_random_tensor<int>(shape, lo, hi);

    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), 24);

    // Check range
    for (size_t i = 0; i < tensor.get_size(); ++i) {
      int val = tensor[i];
      ASSERT_GE(val, lo);
      ASSERT_LE(val, hi);
    }

    // Check not all same
    int first = tensor[0];
    bool all_same = true;
    for (size_t i = 1; i < tensor.get_size(); ++i) {
      if (tensor[i] != first) {
        all_same = false;
        break;
      }
    }
    ASSERT_FALSE(all_same);
  }

  // 2. Test with floating-point type
  {
    array_mml<uli> shape = {5};
    float lo = -1.5f, hi = 2.5f;
    auto tensor = generate_random_tensor<float>(shape, lo, hi);

    ASSERT_EQ(tensor.get_shape(), shape);
    ASSERT_EQ(tensor.get_size(), 5);

    // Check range
    for (size_t i = 0; i < tensor.get_size(); ++i) {
      float val = tensor[i];
      ASSERT_GE(val, lo);
      ASSERT_LE(val, hi);
    }

    // Check not all same
    float first = tensor[0];
    bool all_same = true;
    for (size_t i = 1; i < tensor.get_size(); ++i) {
      if (tensor[i] != first) {
        all_same = false;
        break;
      }
    }
    ASSERT_FALSE(all_same);
  }
}

// ---------    tesors_are_close tests

TEST(test_tensor_utility, test_tensors_are_close_equal) {
  Tensor_mml<float> a({3}, {1.0f, 2.0f, 3.0f});
  Tensor_mml<float> b({3}, {1.0f, 2.0f, 3.0f});
  ASSERT_TRUE(tensors_are_close(a, b, 0.01f));
}

TEST(test_tensor_utility, test_tensors_are_close_within_tolerance) {
  Tensor_mml<float> a({3}, {1.0f, 2.01f, 2.98f});
  Tensor_mml<float> b({3}, {1.0f, 2.0f, 3.0f});
  ASSERT_TRUE(tensors_are_close(a, b, 0.01f)); // within 1% tolerance
}

TEST(test_tensor_utility, test_tensors_are_close_exceeds_tolerance) {
  Tensor_mml<float> a({3}, {1.0f, 2.2f, 3.1f});
  Tensor_mml<float> b({3}, {1.0f, 2.0f, 3.0f});
  ASSERT_FALSE(tensors_are_close(a, b, 0.01f)); // exceeds tolerance
}

TEST(test_tensor_utility, test_tensors_are_close_different_shapes) {
  Tensor_mml<float> a({2}, {1.0f, 2.0f});
  Tensor_mml<float> b({3}, {1.0f, 2.0f, 3.0f});
  ASSERT_FALSE(tensors_are_close(a, b, 0.01f));
}

TEST(test_tensor_utility, test_tensors_are_close_zero_tolerance) {
  Tensor_mml<float> a({3}, {1.0f, 2.0f, 3.0f});
  Tensor_mml<float> b({3}, {1.0f, 2.0f, 3.001f});
  ASSERT_FALSE(tensors_are_close(a, b, 0.0f)); // no tolerance
}

TEST(test_tensor_utility, test_tensors_are_close_integers_exact) {
  Tensor_mml<int> a({3}, {1, 2, 3});
  Tensor_mml<int> b({3}, {1, 2, 3});
  ASSERT_TRUE(tensors_are_close(a, b, 0)); // exact match for integers
}

TEST(test_tensor_utility, test_tensors_are_close_integers_fail) {
  Tensor_mml<int> a({3}, {1, 2, 4});
  Tensor_mml<int> b({3}, {1, 2, 3});
  ASSERT_FALSE(tensors_are_close(a, b, 0)); // mismatch
}

TEST(test_tensor_utility, test_tensors_are_close_with_negatives) {
  Tensor_mml<float> a({3}, {-1.0f, -2.01f, -3.0f});
  Tensor_mml<float> b({3}, {-1.0f, -2.0f, -3.0f});
  ASSERT_TRUE(
      tensors_are_close(a, b, 0.01f)); // relative tolerance should still apply
}

TEST(test_tensor_utils, test_tensors_are_close_zero_reference_value) {
  // a is slightly off from zero at the middle
  Tensor_mml<float> a({3}, {1.0f, 0.000009f, 3.0f});
  Tensor_mml<float> b({3}, {1.0f, 0.0f, 3.0f});

  // tolerance is 1e-2, but the actual fallback tolerance is 1e-5
  // 0.000009 < 0.00001 ⇒ within tolerance, should pass
  ASSERT_TRUE(tensors_are_close(a, b, 0.01f));
}

TEST(test_tensor_utils, test_tensors_are_close_zero_reference_value_fail) {
  Tensor_mml<float> a({3}, {1.0f, 0.0002f, 3.0f});
  Tensor_mml<float> b({3}, {1.0f, 0.0f, 3.0f});

  // 0.0002 > 0.00001 ⇒ outside fallback tolerance
  ASSERT_FALSE(tensors_are_close(a, b, 0.01f));
}
