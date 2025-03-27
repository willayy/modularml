#include <gtest/gtest.h>

#include <modularml>
#include <typeinfo>

TEST(test_lrn, test_lrn_node) {

  shared_ptr<Tensor<float>> X = tensor_mml_p<float>(
    {1, 4, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  shared_ptr<Tensor<float>> exp_output = tensor_mml_p<float>(
      {1, 4, 2, 2},
      {0.5939, 1.1869, 1.7787, 2.3692, 2.9575, 3.5432, 4.1258, 4.7047, 5.2795,
      5.8498, 6.4150, 6.9747, 7.6350, 8.2041, 8.7682, 9.3282});

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;

  LRNNode_mml lrn_node = LRNNode_mml(x_string, y_string, 3, 0.0004, 0.75, 2);
    
  lrn_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *exp_output));
}
