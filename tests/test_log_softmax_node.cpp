#include <gtest/gtest.h>

#include <modularml>

TEST(test_log_softmax_node, test_forward_basic) {
  array_mml<size_t> x_shape({3, 3});
  array_mml<float> x_values(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  array_mml<size_t> y_shape({3, 3});

  std::shared_ptr<Tensor<float>> X =
      std::make_shared<Tensor<float>>(x_shape, x_values);
  std::shared_ptr<Tensor<float>> Y =
      std::make_shared<Tensor<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LogSoftMaxNode logsoftmax(x_string, y_string);

  logsoftmax.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  for (size_t b = 0; b < result_ptr->get_shape()[0]; b++) {
    float row_sum = 0;

    for (size_t c = 0; c < result_ptr->get_shape()[1]; c++) {
      row_sum +=
          std::exp((*result_ptr)[{b, c}]);  // exponentiate the result so we
                                            // can check that they sum to 1
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-5);  // Checks that each row sums to 1
  }
}

TEST(test_log_softmax_node, test_forward_large_range_of_values) {
  array_mml<size_t> x_shape({3, 3});
  array_mml<float> x_values({1.0f, 1000.0f, 1000000.0f, 0.001f, 10.0f, 100.0f,
                             -100.0f, -10.0f, -1.0f});

  array_mml<size_t> y_shape({3, 3});

  std::shared_ptr<Tensor<float>> X =
      std::make_shared<Tensor<float>>(x_shape, x_values);
  std::shared_ptr<Tensor<float>> Y =
      std::make_shared<Tensor<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LogSoftMaxNode logsoftmax(x_string, y_string);

  logsoftmax.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  for (size_t b = 0; b < result_ptr->get_shape()[0]; b++) {
    float row_sum = 0;

    for (size_t c = 0; c < result_ptr->get_shape()[1]; c++) {
      row_sum +=
          std::exp((*result_ptr)[{b, c}]);  // exponentiate the result so we
                                            // can check that they sum to 1
    }
    EXPECT_NEAR(row_sum, 1.0f,
                1e-5);  // Checks that each row sums to 1 which it should
  }
}

TEST(test_log_softmax_node, test_forward_handle_zeros) {
  // Should result in an std::equal distribution
  array_mml<size_t> x_shape({3, 3});
  array_mml<float> x_values(
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  array_mml<size_t> y_shape({3, 3});

  std::shared_ptr<Tensor<float>> X =
      std::make_shared<Tensor<float>>(x_shape, x_values);
  std::shared_ptr<Tensor<float>> Y =
      std::make_shared<Tensor<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LogSoftMaxNode logsoftmax(x_string, y_string);

  logsoftmax.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  for (size_t b = 0; b < result_ptr->get_shape()[0]; b++) {
    float row_sum = 0;

    for (size_t c = 0; c < result_ptr->get_shape()[1]; c++) {
      row_sum +=
          std::exp((*result_ptr)[{b, c}]);  // exponentiate the result so we
                                            // can check that they sum to 1
    }
    EXPECT_NEAR(row_sum, 1.0f,
                1e-5);  // Checks that each row sums to 1 which it should
  }
}

TEST(test_log_softmax_node, test_forward_maxfloat_minfloat_values) {
  // Checks that the node can handle very large and very small floats
  array_mml<size_t> x_shape({3, 3});
  array_mml<float> x_values(
      {FLT_MAX, FLT_MIN, 1.0f, FLT_MAX, FLT_MIN, 1.0f, FLT_MAX, FLT_MIN, 1.0f});

  array_mml<size_t> y_shape({3, 3});

  std::shared_ptr<Tensor<float>> X =
      std::make_shared<Tensor<float>>(x_shape, x_values);
  std::shared_ptr<Tensor<float>> Y =
      std::make_shared<Tensor<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LogSoftMaxNode logsoftmax(x_string, y_string);

  logsoftmax.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  for (size_t b = 0; b < result_ptr->get_shape()[0]; b++) {
    float row_sum = 0;

    for (size_t c = 0; c < result_ptr->get_shape()[1]; c++) {
      row_sum +=
          std::exp((*result_ptr)[{b, c}]);  // exponentiate the result so we
                                            // can check that they sum to 1
    }
    EXPECT_NEAR(row_sum, 1.0f,
                1e-5);  // Checks that each row sums to 1 which it should
  }
}
