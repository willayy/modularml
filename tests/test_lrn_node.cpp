#include <gtest/gtest.h>

#include <modularml>
#include <typeinfo>

TEST(test_lrn, test_lrn_node) {

  shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  shared_ptr<Tensor<float>> exp_output = tensor_mml_p<float>(
      {1, 4, 2, 2},
      {0.5939, 1.1869, 1.7787, 2.3692, 2.9575, 3.5432, 4.1258, 4.7047, 5.2795,
       5.8498, 6.4150, 6.9747, 7.6350, 8.2041, 8.7682, 9.3282});

  LRNNode_mml<float> lrn_node = LRNNode_mml<float>(input, 3, 0.0004, 0.75, 2);
  lrn_node.forward();

  array_mml<GeneralDataTypes> output = lrn_node.getOutputs();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&output[0])) {

    ASSERT_TRUE(tensors_are_close(*(tensor_ptr->get()), *exp_output));
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl
              << std::flush;
  }
}
