#include <gtest/gtest.h>

#include "flatten_node.hpp"


TEST(flatten_node_test, test_forward_3d_tensor) {
  array_mml<uli> x_shape({2, 2, 3});
  array_mml<float> x_values(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

  shared_ptr<Tensor_mml<float>> X =
      make_shared<Tensor_mml<float>>(x_shape, x_values);

  // Shape doesnt matter for output
  array_mml<uli> y_shape({1, 1, 1});

  shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

  FlattenNode<float> flatten(X, Y, 1);

  flatten.forward();

  EXPECT_EQ(Y->get_shape(), array_mml<uli>({2, 6}));
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

  // Axis = 2 means that that the shape is flattened as such 2x2, 3x3 = {4, 9}
  FlattenNode<float> flatten(X, Y, 2);

  flatten.forward();

  EXPECT_EQ(Y->get_shape(), array_mml<uli>({4, 9}));
}   