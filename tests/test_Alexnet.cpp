/**
 * @file test_Alexnet.cpp
 * @brief Tests for running alexnet and comparing it to the pytorch output.
 */

#include "test_Alexnet.hpp"

#include <gtest/gtest.h>

#include <modularml>

// TODO: Parse the pretrained alexnet model: https://drive.google.com/file/d/1mGgWd8xiXaFFPYnnCGZt0bD0OksuQBz0/view?usp=sharing
//       make sure the weights and bias are loaded correctly.

// TODO: Run the model with the input currently defined inputtensor. Compare against the corrently defined outputtensor. These
//       are the same as the pytorch input and output.

// TODO: Run argMax to make sure that the predicted_class matches the expected class from the pytorch output.

TEST(test_Alexnet, test_Alexnet_parse_run_compare) {
  auto TensorToProcess = tensor_mml_p<double>({INPUT_TENSOR_SHAPE}, {INPUT_TENSOR_DATA});
  auto refrence_output = tensor_mml_p<double>({OUTPUT_TENSOR_SHAPE}, {OUTPUT_TENSOR_DATA});
  auto refrence_predicted_class = PREDICTED_CLASS;

  ASSERT_TRUE(1 + 2 == 3);  // This is just a placeholder to make sure the test is running.
}