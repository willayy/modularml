#include <gtest/gtest.h>

#include <modularml>
#include <typeinfo>
// Maxpool
TEST(test_mml_pooling, test_max_pool_auto_pad_NOTSET) {
  std::shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 2, 2}, {6, 8, 14, 16});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices =
      tensor_mml_p<int64_t>({1, 1, 2, 2}, {5, 7, 13, 15});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolNode max_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "NOTSET",                     // auto_pad
    0,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    0,                            // storage_order
    {2, 2}                        // strides
 );

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}

TEST(test_mml_pooling, test_max_pool_auto_pad_SAME_UPPER) {
  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {5, 6, 8, 9, 8, 9});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices =
      tensor_mml_p<int64_t>({1, 1, 3, 2}, {4, 5, 7, 8, 7, 8});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolNode max_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "SAME_UPPER",                     // auto_pad
    1,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    0,                            // storage_order
    {1, 2}                        // strides
  );

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}

TEST(test_mml_pooling, test_max_pool_auto_pad_SAME_UPPER_floor_dilation_col) {
  std::shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 1, 4, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  std::shared_ptr<Tensor<float>> exp_output = tensor_mml_p<float>(
      {1, 1, 4, 3}, {1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices = tensor_mml_p<int64_t>(
      {1, 1, 4, 3}, {0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolingNode_mml max_pool = MaxPoolingNode_mml(
      input_string, std::vector<std::string>{output_string, indices_string},
      array_mml({2UL, 2UL}), array_mml({1UL, 2UL}), "SAME_UPPER", 0UL,
      array_mml({2UL, 2UL}), array_mml({0UL, 0UL, 0UL, 0UL}), 1UL);

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}

TEST(test_mml_pooling, test_max_pool_auto_pad_SAME_LOWER) {
  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {1, 3, 4, 6, 7, 9});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices =
      tensor_mml_p<int64_t>({1, 1, 3, 2}, {0, 2, 3, 5, 6, 8});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolNode max_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "SAME_LOWER",                     // auto_pad
    1,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    0,                            // storage_order
    {1, 2}                        // strides
  );

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}

TEST(test_mml_pooling, test_max_pool_auto_pad_VALID) {
  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 2, 1}, {5, 8});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices =
      tensor_mml_p<int64_t>({1, 1, 2, 1}, {4, 7});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolNode max_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "VALID",                     // auto_pad
    1,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    0,                            // storage_order
    {1, 2}                        // strides
  );

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}

TEST(test_mml_pooling, test_max_pool_custom_pad) {
  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {5, 6, 8, 9, 8, 9});
  std::shared_ptr<Tensor<int64_t>> exp_output_indices =
      tensor_mml_p<int64_t>({1, 1, 3, 2}, {4, 5, 7, 8, 7, 8});

  std::string input_string = "input";
  std::string output_string = "output";
  std::string indices_string = "indices";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  MaxPoolNode max_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "NOTSET",                     // auto_pad
    1,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 1, 0},                 // pads
    0,                            // storage_order
    {1, 2}                        // strides
  );

  max_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  auto indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  auto indices_ptr =
      std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);

  iomap.clear();
  iomap[input_string] = input;

  exp_output = tensor_mml_p<float>({1, 1, 3, 1}, {5, 8, 8});
  exp_output_indices = tensor_mml_p<int64_t>({1, 1, 3, 1}, {4, 7, 7});
 
  max_pool = MaxPoolNode (
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    indices_string,               // indices
    "NOTSET",                     // auto_pad
    0,                            // ceil_mode
    {1, 1},                       // dilations
    {0, 0, 1, 0},                 // pads
    0,                            // storage_order
    {1, 2}                        // strides
  );

  max_pool.forward(iomap);

  output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  indices_it = iomap.find(indices_string);
  ASSERT_NE(output_it, iomap.end())
      << "Indices tensor not found in iomap after forward pass";

  indices_ptr = std::get<std::shared_ptr<Tensor<int64_t>>>(indices_it->second);
  ASSERT_NE(indices_ptr, nullptr) << "Failed to get indices tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
  ASSERT_EQ(*indices_ptr, *exp_output_indices);
}
// Average Pooling
TEST(test_mml_pooling, test_avg_pool_valid) {
  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 2, 1}, {3, 6});

  std::string input_string = "input";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  AvgPoolNode avg_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "VALID",                      // auto_pad
    1,                            // ceil_mode
    0,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
}

TEST(test_mml_pooling, test_avg_pool_same_upper) {

  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {3, 4.5, 6, 7.5, 7.5, 9});

  std::string input_string = "input";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  AvgPoolNode avg_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "SAME_UPPER",                 // auto_pad
    1,                            // ceil_mode
    0,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);

  exp_output = tensor_mml_p<float>({1, 1, 3, 2}, {3, 4.5, 6, 7.5, 7.5, 9});

  avg_pool = AvgPoolNode(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "SAME_UPPER",                 // auto_pad
    0,                            // ceil_mode
    0,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);

  iomap.clear();
  iomap[input_string] = input;

  exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {3, 2.25, 6, 3.75, 3.75, 2.25});

  avg_pool = AvgPoolNode(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "SAME_UPPER",                 // auto_pad
    0,                            // ceil_mode
    1,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
}

TEST(test_mml_pooling, test_avg_pool_same_lower) {

  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {1, 2.5, 2.5, 4, 5.5, 7});

  std::string input_string = "input";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  AvgPoolNode avg_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "SAME_LOWER",                 // auto_pad
    1,                            // ceil_mode
    0,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 0, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
}

TEST(test_mml_pooling, test_avg_pool_custom_pad) {

  std::shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 1, 3, 2}, {3, 2.25, 6, 3.75, 3.75, 2.25});

  std::string input_string = "input";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[input_string] = input;

  AvgPoolNode avg_pool(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "NOTSET",                     // auto_pad
    1,                            // ceil_mode
    1,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 1, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  auto output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);

  iomap.clear();

  input = tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  exp_output = tensor_mml_p<float>({1, 1, 3, 1}, {3, 6, 3.75});

  iomap[input_string] = input;

  avg_pool = AvgPoolNode(
    input_string,                 // X
    output_string,                // Y
    {2, 2},                       // kernel_shape
    "NOTSET",                     // auto_pad
    0,                            // ceil_mode
    1,                            // count_include_pad
    {1, 1},                       // dilations
    {0, 0, 1, 0},                 // pads
    {1, 2}                        // strides
  );

  avg_pool.forward(iomap);

  output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end())
      << "Output tensor not found in iomap after forward pass";

  output_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(output_ptr, nullptr) << "Failed to get output tensor";

  ASSERT_EQ(*output_ptr, *exp_output);
}
