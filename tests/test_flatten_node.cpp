#include <gtest/gtest.h>

#include "nodes/flatten_node.hpp"


TEST(flatten_node_test, test_forward_3d_tensor) {
  array_mml<uli> x_shape({2, 2, 3});
  array_mml<float> x_values(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

  shared_ptr<Tensor_mml<float>> X =
      make_shared<Tensor_mml<float>>(x_shape, x_values);

  // Shape doesnt matter for output
  array_mml<uli> y_shape({1, 1, 1});

  shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  FlattenNode flatten(x_string, y_string, 1);

  flatten.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";
  
  EXPECT_EQ(result_ptr->get_shape(), array_mml<uli>({2, 6}));
}   


TEST(flatten_node_test, test_forward_4d_tensor) {
  array_mml<uli> x_shape({2, 2, 3, 3});
  array_mml<float> x_values({1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f,

                             1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f,

                             1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f

  });

  shared_ptr<Tensor_mml<float>> X =
      make_shared<Tensor_mml<float>>(x_shape, x_values);

  // Shape doesnt matter for output
  array_mml<uli> y_shape({1, 1, 1});

  shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  //iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  // Axis = 2 means that that the shape is flattened as such 2x2, 3x3 = {4, 9} 
  FlattenNode flatten(x_string, y_string, 2);

  flatten.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";
  
  EXPECT_EQ(result_ptr->get_shape(), array_mml<uli>({4, 9}));
}   