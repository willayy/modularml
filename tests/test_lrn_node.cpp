#include <gtest/gtest.h>

#include <modularml>
#include <typeinfo>

TEST(test_lrn, test_lrn_node_float) {
  std::shared_ptr<Tensor<float>> X = TensorFactory::create_tensor<float>(
      {1, 4, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  std::shared_ptr<Tensor<float>> exp_output =
      TensorFactory::create_tensor<float>(
          {1, 4, 2, 2},
          {0.5939, 1.1869, 1.7787, 2.3692, 2.9575, 3.5432, 4.1258, 4.7047,
           5.2795, 5.8498, 6.4150, 6.9747, 7.6350, 8.2041, 8.7682, 9.3282});

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
TEST(test_lrn, test_lrn_node_double) {
  std::shared_ptr<Tensor<double>> X = TensorFactory::create_tensor<double>(
      {1, 4, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  std::shared_ptr<Tensor<double>> exp_output =
      TensorFactory::create_tensor<double>(
          {1, 4, 2, 2},
          {0.5939, 1.1869, 1.7787, 2.3692, 2.9575, 3.5432, 4.1258, 4.7047,
           5.2795, 5.8498, 6.4150, 6.9747, 7.6350, 8.2041, 8.7682, 9.3282});

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;

  LRNNode_mml lrn_node = LRNNode_mml(x_string, y_string, 3, 0.0004, 0.75, 2);

  lrn_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<double>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *exp_output));
}

TEST(test_lrn, test_lrn_node_square_sum_0) {
  std::shared_ptr<Tensor<double>> X =
      TensorFactory::create_tensor<double>({1, 4, 2, 2}, {
                                                             0,
                                                             2,
                                                             3,
                                                             4,
                                                             0,
                                                             2,
                                                             3,
                                                             4,
                                                             0,
                                                             2,
                                                             3,
                                                             4,
                                                             0,
                                                             2,
                                                             3,
                                                             4,
                                                         });

  std::shared_ptr<Tensor<double>> exp_output =
      TensorFactory::create_tensor<double>(
          {1, 4, 2, 2},
          {0.0f, 1.9997f, 2.9987f, 3.9970f, 0.0f, 1.9994f, 2.9980f, 3.9952f,
           0.0f, 1.9994f, 2.9980f, 3.9952f, 0.0f, 1.9997f, 2.9987f, 3.9970f});

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;

  LRNNode_mml lrn_node =
      LRNNode_mml(x_string, y_string, 3, 0.0001f, 0.75f, 1.0f);

  lrn_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<double>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";
  ASSERT_TRUE(tensors_are_close(*result_ptr, *exp_output));
}

TEST(test_lrn, test_lrn_node_random_values) {
  std::shared_ptr<Tensor<double>> X = TensorFactory::create_tensor<double>(
      {1, 4, 2, 2}, {
                        0.608746f,
                        9.412452f,
                        -5.879261f,
                        5.789683f,
                        -9.167872f,
                        -8.310365000000001f,
                        -2.689699f,
                        5.539192f,
                        9.300974f,
                        -6.686403f,
                        -0.36053500000000005f,
                        3.802501f,
                        -1.701825f,
                        -1.942637f,
                        7.586148f,
                        -6.068864f,
                    });

  std::shared_ptr<Tensor<double>> exp_output =
      TensorFactory::create_tensor<double>(
          {1, 4, 2, 2},
          {0.6075f, 9.3755f, -5.8731f, 5.7804f, -9.1289f, -8.2686f, -2.6869f,
           5.5283f, 9.2608f, -6.6668f, -0.3600f, 3.7947f, -1.6980f, -1.9403f,
           7.5752f, -6.0611f});

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;

  LRNNode_mml lrn_node =
      LRNNode_mml(x_string, y_string, 3, 0.0001f, 0.75f, 1.0f);

  lrn_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<double>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";
  ASSERT_TRUE(tensors_are_close(*result_ptr, *exp_output));
}

TEST(test_lrn, test_lrn_node_invalid_arguments) {
  std::shared_ptr<Tensor<double>> X =
      TensorFactory::create_tensor<double>({1, 1, 1, 1});

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;

  ASSERT_THROW(LRNNode_mml(x_string, y_string, 0.0f, 0.001f, 0.75f, 1.0f),
               std::invalid_argument);
  ASSERT_THROW(LRNNode_mml(x_string, y_string, 1.0f, 0.001f, 0.75f, 0.00001f),
               std::invalid_argument);
}
