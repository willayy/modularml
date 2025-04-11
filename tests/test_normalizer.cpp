#include <gtest/gtest.h>

#include "backend/dataloader/normalizer.hpp"

TEST(normalizer_test, test_normalize_valid_input) {
  Normalize normalizer;
  // Input tensor shape: 1x3x2x2
  array_mml<size_t> input_shape({1, 3, 2, 2});
  array_mml<float> input_values({
      1.0f, 2.0f, 3.0f, 4.0f,    // Channel 1
      5.0f, 6.0f, 7.0f, 8.0f,    // Channel 2
      9.0f, 10.0f, 11.0f, 12.0f  // Channel 3
  });

  auto input_tensor = std::make_shared<Tensor_mml<float>>(input_shape, input_values);

  // Mean and standard deviation for each channel
  std::array<float, 3> mean = {2.0f, 6.0f, 10.0f};
  std::array<float, 3> std = {1.0f, 2.0f, 3.0f};

  auto output_tensor = normalizer.normalize(input_tensor, mean, std);

  // Expected output tensor values
  array_mml<float> expected_values({
      -1.0f, 0.0f, 1.0f, 2.0f,          // Channel 1
      -0.5f, 0.0f, 0.5f, 1.0f,          // Channel 2
      -0.3333f, 0.0f, 0.3333f, 0.6667f  // Channel 3
  });

  for (size_t n = 0; n < input_shape[0]; ++n) {
    for (size_t c = 0; c < input_shape[1]; ++c) {
      for (size_t h = 0; h < input_shape[2]; ++h) {
        for (size_t w = 0; w < input_shape[3]; ++w) {
          float actual = (*output_tensor)[{n, c, h, w}];
          float expected = expected_values[n * input_shape[1] * input_shape[2] * input_shape[3] +
                                           c * input_shape[2] * input_shape[3] +
                                           h * input_shape[3] + w];
          EXPECT_NEAR(actual, expected, 1e-4) << "Mismatch at index (" << n << ", " << c << ", " << h << ", " << w << ")";
        }
      }
    }
  }
}

TEST(normalizer_test, test_invalid_input_dimensions) {
  Normalize normalizer;
  // Input tensor shape: 1x3x2 (invalid, not 4D)
  array_mml<size_t> input_shape({1, 3, 2});
  array_mml<float> input_values({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto input_tensor = std::make_shared<Tensor_mml<float>>(input_shape, input_values);

  std::array<float, 3> mean = {2.0f, 6.0f, 10.0f};
  std::array<float, 3> std = {1.0f, 2.0f, 3.0f};

  EXPECT_THROW(normalizer.normalize(input_tensor, mean, std), std::invalid_argument);
}