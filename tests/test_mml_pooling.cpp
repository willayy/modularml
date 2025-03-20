#include <gtest/gtest.h>

#include <modularml>
#include <typeinfo>

TEST(test_mml_pooling, test_max_pool_auto_pad_NOTSET) {
  shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 2, 2}, {6, 8, 14, 16});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 2, 2}, {5, 7, 13, 15});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({2, 2}), input, "NOTSET", 0,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  max_pool.forward();

  array_mml<GeneralDataTypes> outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {

    ASSERT_EQ(*(tensor_ptr->get()), *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;
    std::cerr << "Resulting tensor: " << real_output_indices->to_string()
              << std::endl
              << std::flush;
    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_max_pool_auto_pad_SAME_UPPER) {
  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {5, 6, 8, 9, 8, 9});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 3, 2}, {4, 5, 7, 8, 7, 8});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_UPPER", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  max_pool.forward();
  array_mml<GeneralDataTypes> outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;

    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_max_pool_auto_pad_SAME_LOWER) {
  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {1, 3, 4, 6, 7, 9});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 3, 2}, {0, 2, 3, 5, 6, 8});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_LOWER", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  max_pool.forward();
  array_mml<GeneralDataTypes> outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;

    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_max_pool_auto_pad_VALID) {
  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 1, 2, 1}, {5, 8});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 2, 1}, {4, 7});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "VALID", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  max_pool.forward();
  array_mml<GeneralDataTypes> outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;

    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_max_pool_custom_pad) {
  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {5, 6, 8, 9, 8, 9});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 3, 2}, {4, 5, 7, 8, 7, 8});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "NOTSET", 1,
      array_mml({1, 1}), array_mml({0, 1, 0, 0}), 0);

  max_pool.forward();
  array_mml<GeneralDataTypes> outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;

    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }

  output = tensor_mml_p<float>({1, 1, 3, 1}, {5, 8, 8});
  output_indices = tensor_mml_p<float>({1, 1, 3, 1}, {4, 7, 7});

  max_pool = MaxPoolingNode_mml<float>(array_mml({2, 2}), array_mml({1, 2}),
                                       input, "NOTSET", 0, array_mml({1, 1}),
                                       array_mml({0, 1, 0, 0}), 0);

  max_pool.forward();
  outputs = max_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;

    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_avg_pool_valid) {
  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 1, 2, 1}, {3, 6});

  AvgPoolingNode_mml<float> avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "VALID", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  avg_pool.forward();
  array_mml<GeneralDataTypes> outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_avg_pool_same_upper) {

  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {3, 4.5, 6, 7.5, 7.5, 9});

  AvgPoolingNode_mml<float> avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_UPPER", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  avg_pool.forward();
  array_mml<GeneralDataTypes> outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }

  output = tensor_mml_p<float>({1, 1, 3, 2}, {3, 4.5, 6, 7.5, 7.5, 9});

  avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_UPPER", 0,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  avg_pool.forward();
  outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }

  output = tensor_mml_p<float>({1, 1, 3, 2}, {3, 2.25, 6, 3.75, 3.75, 2.25});

  avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_UPPER", 0,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 1);

  avg_pool.forward();
  outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_avg_pool_same_lower) {

  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {1, 2.5, 2.5, 4, 5.5, 7});

  AvgPoolingNode_mml<float> avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "SAME_LOWER", 1,
      array_mml({1, 1}), array_mml({0, 0, 0, 0}), 0);

  avg_pool.forward();
  array_mml<GeneralDataTypes> outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}

TEST(test_mml_pooling, test_avg_pool_custom_pad) {

  shared_ptr<Tensor<float>> input =
      tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 3, 2}, {3, 2.25, 6, 3.75, 3.75, 2.25});

  AvgPoolingNode_mml<float> avg_pool = AvgPoolingNode_mml<float>(
      array_mml({2, 2}), array_mml({1, 2}), input, "NOTSET", 1,
      array_mml({1, 1}), array_mml({0, 1, 0, 0}), 1);

  avg_pool.forward();
  array_mml<GeneralDataTypes> outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }

  input = tensor_mml_p<float>({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output = tensor_mml_p<float>({1, 1, 3, 1}, {3, 6, 3.75});

  avg_pool = AvgPoolingNode_mml<float>(array_mml({2, 2}), array_mml({1, 2}),
                                       input, "NOTSET", 0, array_mml({1, 1}),
                                       array_mml({0, 1, 0, 0}), 1);

  avg_pool.forward();
  outputs = avg_pool.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&outputs[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;

    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}
