#include <gtest/gtest.h>

#include <modularml>

// Test the default constructors
TEST(test_mml_tensor, test_default_constructor_1) {
  Tensor_mml<int> t1 = Tensor_mml<int>({3, 3});
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t1.get_shape();
  auto actual_data = t1.get_data();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ(expected_data, actual_data);
}

// Test the copy constructor
TEST(test_mml_tensor, test_copy_constructor_1) {
  Tensor_mml<int> t1 = Tensor_mml<int>({3, 3});
  Tensor_mml<int> t2 = t1;
  ASSERT_EQ(t1, t2);
  // Check if the data is different pointers
  ASSERT_NE(&t1.get_data(), &t2.get_data());
}

// Test the move constructor
TEST(test_mml_tensor, test_move_constructor_1) {
  Tensor_mml<int> t1 = Tensor_mml<int>({3, 3});
  Tensor_mml<int> t2 = move(t1);
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2.get_shape();
  auto actual_data = t2.get_data();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ(expected_data, actual_data);
}

// Test the copy assignment operator
TEST(test_mml_tensor, test_copy_assignment_1) {
  Tensor_mml<int> t1 = Tensor_mml<int>({3, 3});
  Tensor_mml<int> t2 = Tensor_mml<int>({2, 2});
  t2 = t1;
  ASSERT_EQ(t1, t2);
}

// Test the move assignment operator
TEST(test_mml_tensor, test_move_assignment_1) {
  Tensor_mml<int> t1 = Tensor_mml<int>({3, 3});
  Tensor_mml<int> t2 = Tensor_mml<int>({2, 2});
  t2 = move(t1);
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2.get_shape();
  auto actual_data = t2.get_data();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ(expected_data, actual_data);
}

// Test the move assignment using abstract class
TEST(test_mml_tensor, test_move_assignment_2) {
  shared_ptr<Tensor<int>> t1 =
      make_shared<Tensor_mml<int>>(array_mml<int>({3, 3}));
  shared_ptr<Tensor<int>> t2 =
      make_shared<Tensor_mml<int>>(array_mml<int>({2, 2}));
  *t2 = move(*t1);
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);

  // Cast to Tensor_mml to access the data
  auto actual_data = dynamic_pointer_cast<Tensor_mml<int>>(t2)->get_data();
  ASSERT_EQ(expected_data, actual_data);
}

// Test the copy assignment using abstract class
TEST(test_mml_tensor, test_copy_assignment_2) {
  shared_ptr<Tensor<int>> t1 =
      make_shared<Tensor_mml<int>>(array_mml<int>({3, 3}));
  shared_ptr<Tensor<int>> t2 =
      make_shared<Tensor_mml<int>>(array_mml<int>({2, 2}));
  *t2 = *t1;
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);

  // Cast to Tensor_mml to access the data
  auto actual_data = dynamic_pointer_cast<Tensor_mml<int>>(t2)->get_data();
  ASSERT_EQ(expected_data, actual_data);
}

// Generate an arbitrary tensor and check if all elements can be accessed using
// indices
TEST(test_mml_tensor, test_index_1) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>();
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);
    for (int j = 0; j < (*t1).get_size(); j++) {
      (*t1)[j] = 101;
      ASSERT_EQ(101, (*t1)[j]);
    }
  }
}

TEST(test_mml_tensor, test_index_2) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>();
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);
    for (int j = 0; j < (*t1).get_size(); j++) {
      ASSERT_EQ(data[j], (*t1)[j]);
    }
  }
}

TEST(test_mml_tensor, test_indices_1) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>();
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);
    for (int j = 0; j < t1->get_size(); j++) {
      array_mml<uli> indices = array_mml<uli>(shape.size());
      int k = j;

      uli l = shape.size() - 1;
      do {
        indices[l] = k % shape[l];
        k /= shape[l];
      } while (l-- > 0);
      (*t1)[indices] = 101;
      ASSERT_EQ(101, (*t1)[indices]);
    }
  }
}

TEST(test_mml_tensor, test_indices_2) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>();
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);

    auto indices = array_mml<uli>(shape.size());
    indices.fill(0);

    for (uli j = 0; j < t1->get_size(); j++) {
      ASSERT_EQ(data[j], (*t1)[indices]);

      uli k = shape.size() - 1;
      do {
        if (indices[k] < shape[k] - 1) {
          indices[k] = indices[k] + 1;
          break;
        } else {
          indices[k] = 0;
        }
      } while (k-- > 0);
    }
  }
}

// Reshape into 1D tensor
TEST(test_mml_tensor, test_reshape_1) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>();
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);
    t1->reshape({elements});
    auto expected_shape = array_mml<uli>({elements});
    auto actual_shape = t1->get_shape();
    ASSERT_EQ(expected_shape, actual_shape);
    auto expected_data = data;
    for (uli j = 0; i < elements; i++) {
      ASSERT_EQ(expected_data[j], (*t1)[j]);
    }
  }
}

// Reshape into 1D into 2D tensor
TEST(test_mml_tensor, test_reshape_2) {
  for (int i = 0; i < 200; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>(1, 1);
    const auto elements =
        accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data =
        generate_random_array_mml_integral<int>(elements, elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data);
    if (shape[0] % 2 == 0) {
      uli rows = shape[0] / 2;
      uli cols = 2;
      auto new_shape = array_mml<uli>({rows, cols});
      t1->reshape(new_shape);
      for (uli i = 0; i < rows; i++) {
        for (uli j = 0; j < cols; j++) {
          auto expected = data[i * cols + j];
          auto actual = (*t1)[{i, j}];
          ASSERT_EQ(expected, actual);
        }
      }
    } else {
      continue; // Skip odd-sized arrays testing with 200 iterations to get an
                // average of 100 valid tests
    }
  }
}

TEST(test_mml_tensor, test_to_string) {
  const auto t1 = Tensor_mml<int>({3, 3});
  const string ptr_str =
      "Pointer: " + std::to_string(reinterpret_cast<uint64_t>(&t1));
  string expected =
      "Tensor_mml<i> " + ptr_str +
      ", Shape: [3, 3], Size: 9, Data: [0, 0, 0, 0, 0, 0, 0, 0, 0]";
  string actual = t1.to_string();
  ASSERT_EQ(expected, actual);
}

TEST(test_mml_tensor, tensor_utility_tensors_are_close) {
  const auto t1 =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.3f, 0.0f, -3.2f, 5.1f, 2.0f});
  const auto t2 =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.3f, 0.000002f, -3.2f, 5.1f, 2.0f});
  const auto t3 =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.3f, 0.00002f, -3.2f, 5.1f, 2.0f});

  ASSERT_TRUE(tensors_are_close(*t1, *t2));
  ASSERT_FALSE(tensors_are_close(*t1, *t3));
}