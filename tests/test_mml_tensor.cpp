#include <gtest/gtest.h>

#include <modularml>

// Test the default constructors
TEST(test_mml_tensor, test_default_constructor_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p<int>({3, 3});
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_t1 = tensor_mml_p<int>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const& actual_shape = t1->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t1), (*t1));
}

// Test the copy constructor
TEST(test_mml_tensor, test_copy_constructor_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p<int>({3, 3});
  shared_ptr<Tensor<int>> t2 = t1;
  ASSERT_EQ((*t1), (*t2));
}

// Test the move constructor
TEST(test_mml_tensor, test_move_constructor_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p<int>({3, 3});
  shared_ptr<Tensor<int>> t2 = move(t1);
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_t2 = tensor_mml_p<int>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const& actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t2), (*t2));
}

// Test the copy assignment operator
TEST(test_mml_tensor, test_copy_assignment_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p<int>({3, 3});
  shared_ptr<Tensor<int>> t2 = tensor_mml_p<int>({2, 2});
  t2 = t1;
  ASSERT_EQ((*t1), (*t2));
}

// Test the move assignment operator
TEST(test_mml_tensor, test_move_assignment_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p<int>({3, 3});
  shared_ptr<Tensor<int>> t2 = tensor_mml_p<int>({2, 2});
  t2 = move(t1);
  auto expected_shape = array_mml<uli>({3, 3});
  auto expected_t2 = tensor_mml_p<int>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const& actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t2), (*t2));
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

// Test slicing Tensors
TEST(test_mml_tensor, test_slicing_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p(
    {3, 3},
    {1, 2, 3,
     4, 5, 6,
     7, 8, 9});
  shared_ptr<Tensor<int>> t2 = t1->slice({0});
  shared_ptr<Tensor<int>> t3 = t1->slice({1});
  shared_ptr<Tensor<int>> t4 = t1->slice({2});
  shared_ptr<Tensor<int>> expected_t2 = tensor_mml_p({3}, {1, 4, 7});
  shared_ptr<Tensor<int>> expected_t3 = tensor_mml_p({3}, {2, 5, 8});
  shared_ptr<Tensor<int>> expected_t4 = tensor_mml_p({3}, {3, 6, 9});
  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_2) {
  shared_ptr<Tensor<float>> t1 = tensor_mml_p(
    {3, 3},
    {1.0f, 2.0f, 3.0f,
     4.0f, 5.0f, 6.0f,
     7.0f, 8.0f, 9.0f});
  shared_ptr<Tensor<float>> t2 = t1->slice({0});
  shared_ptr<Tensor<float>> t3 = t1->slice({1});
  shared_ptr<Tensor<float>> t4 = t1->slice({2});
  shared_ptr<Tensor<float>> expected_t2 = tensor_mml_p({3}, {1.0f, 4.0f, 7.0f});
  shared_ptr<Tensor<float>> expected_t3 = tensor_mml_p({3}, {2.0f, 5.0f, 8.0f});
  shared_ptr<Tensor<float>> expected_t4 = tensor_mml_p({3}, {3.0f, 6.0f, 9.0f});
  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_3) {
  shared_ptr<Tensor<float>> t1 = tensor_mml_p(
      {3, 3},
      {1.0f, 2.0f, 3.0f,
       4.0f, 5.0f, 6.0f,
       7.0f, 8.0f, 9.0f});
  
  shared_ptr<Tensor<float>> t2 = t1->slice({2});
  // Test indices access
  ASSERT_EQ(3.0f, (*t2)[{0}]);
  ASSERT_EQ(6.0f, (*t2)[{1}]);
  ASSERT_EQ(9.0f, (*t2)[{2}]);
}

TEST(test_mml_tensor, test_slicing_4) {
  shared_ptr<Tensor<float>> t1 = tensor_mml_p(
      {3, 3, 3},
      {1.0f,  2.0f,  3.0f,
       4.0f,  5.0f,  6.0f,
       7.0f,  8.0f,  9.0f,

       10.0f, 11.0f, 12.0f, 
       13.0f, 14.0f, 15.0f,
       16.0f, 17.0f, 18.0f,

       19.0f, 20.0f, 21.0f,
       22.0f, 23.0f, 24.0f,
       25.0f, 26.0f, 27.0f});

  shared_ptr<Tensor<float>> t2 = t1->slice({0});
  shared_ptr<Tensor<float>> t3 = t1->slice({1});
  shared_ptr<Tensor<float>> t4 = t1->slice({2});

  shared_ptr<Tensor<float>> expected_t2 = tensor_mml_p(
      {3, 3},
      {1.0f,  2.0f,  3.0f,
       4.0f,  5.0f,  6.0f,
       7.0f,  8.0f,  9.0f});

  shared_ptr<Tensor<float>> expected_t3 = tensor_mml_p(
      {3, 3},
      {10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f,
       16.0f, 17.0f, 18.0f});

  shared_ptr<Tensor<float>> expected_t4 = tensor_mml_p(
      {3, 3},
      {19.0f, 20.0f, 21.0f,
       22.0f, 23.0f, 24.0f,
       25.0f, 26.0f, 27.0f});

  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_5) {
  shared_ptr<Tensor<float>> t1 = tensor_mml_p(
      {3, 3, 3},
      {1.0f,  2.0f,  3.0f,
       4.0f,  5.0f,  6.0f,
       7.0f,  8.0f,  9.0f,

       10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f,
       16.0f, 17.0f, 18.0f,

       19.0f, 20.0f, 21.0f,
       22.0f, 23.0f, 24.0f,
       25.0f, 26.0f, 27.0f});

  shared_ptr<Tensor<float>> t2 = t1->slice({0,0});
  shared_ptr<Tensor<float>> t3 = t1->slice({1,0});
  shared_ptr<Tensor<float>> t4 = t1->slice({2,0});
  shared_ptr<Tensor<float>> t5 = t1->slice({0,1});
  shared_ptr<Tensor<float>> t6 = t1->slice({1,1});
  shared_ptr<Tensor<float>> t7 = t1->slice({2,1});
  shared_ptr<Tensor<float>> t8 = t1->slice({0,2});
  shared_ptr<Tensor<float>> t9 = t1->slice({1,2});
  shared_ptr<Tensor<float>> t10 = t1->slice({2,2});

  shared_ptr<Tensor<float>> expected_t2 = tensor_mml_p(
      {3},
      {1.0f, 4.0f, 7.0f});

  shared_ptr<Tensor<float>> expected_t3 = tensor_mml_p(
      {3},
      {10.0f, 13.0f, 16.0f});

  shared_ptr<Tensor<float>> expected_t4 = tensor_mml_p(
      {3},
      {19.0f, 22.0f, 25.0f});

  shared_ptr<Tensor<float>> expected_t5 = tensor_mml_p(
      {3},
      {2.0f, 5.0f, 8.0f});

  shared_ptr<Tensor<float>> expected_t6 = tensor_mml_p(
      {3},
      {11.0f, 14.0f, 17.0f});

  shared_ptr<Tensor<float>> expected_t7 = tensor_mml_p(
      {3},
      {20.0f, 23.0f, 26.0f});

  shared_ptr<Tensor<float>> expected_t8 = tensor_mml_p(
      {3},
      {3.0f, 6.0f, 9.0f});

  shared_ptr<Tensor<float>> expected_t9 = tensor_mml_p(
      {3},
      {12.0f, 15.0f, 18.0f});

  shared_ptr<Tensor<float>> expected_t10 = tensor_mml_p(
      {3},
      {21.0f, 24.0f, 27.0f});

  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
  ASSERT_EQ(*expected_t5, *t5);
  ASSERT_EQ(*expected_t6, *t6);
  ASSERT_EQ(*expected_t7, *t7);
  ASSERT_EQ(*expected_t8, *t8);
  ASSERT_EQ(*expected_t9, *t9);
  ASSERT_EQ(*expected_t10, *t10);
}

TEST(test_mml_tensor, test_slicing_6) {
  shared_ptr<Tensor<float>> t1 = tensor_mml_p(
      {3, 3, 3},
      {1.0f, 2.0f, 3.0f,
       4.0f, 5.0f, 6.0f,
       7.0f, 8.0f, 9.0f,

       10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f,
       16.0f, 17.0f, 18.0f,

       19.0f, 20.0f, 21.0f,
       22.0f, 23.0f, 24.0f,
       25.0f, 26.0f, 27.0f});

  // Slice once
  shared_ptr<Tensor<float>> t2 = t1->slice({0});
  shared_ptr<Tensor<float>> t3 = t1->slice({1});
  shared_ptr<Tensor<float>> t4 = t1->slice({2});
  // Slice the slices
  shared_ptr<Tensor<float>> t20 = t2->slice({0});
  shared_ptr<Tensor<float>> t21 = t2->slice({1});
  shared_ptr<Tensor<float>> t22 = t2->slice({2});
  shared_ptr<Tensor<float>> t30 = t3->slice({0});
  shared_ptr<Tensor<float>> t31 = t3->slice({1});
  shared_ptr<Tensor<float>> t32 = t3->slice({2});
  shared_ptr<Tensor<float>> t40 = t4->slice({0});
  shared_ptr<Tensor<float>> t41 = t4->slice({1});
  shared_ptr<Tensor<float>> t42 = t4->slice({2});

  shared_ptr<Tensor<float>> expected_t20 = tensor_mml_p(
      {3},
      {1.0f, 4.0f, 7.0f});

  shared_ptr<Tensor<float>> expected_t21 = tensor_mml_p(
      {3},
      {2.0f, 5.0f, 8.0f});

  shared_ptr<Tensor<float>> expected_t22 = tensor_mml_p(
      {3},
      {3.0f, 6.0f, 9.0f});

  shared_ptr<Tensor<float>> expected_t30 = tensor_mml_p(
      {3},
      {10.0f, 13.0f, 16.0f});

  shared_ptr<Tensor<float>> expected_t31 = tensor_mml_p(
      {3},
      {11.0f, 14.0f, 17.0f});

  shared_ptr<Tensor<float>> expected_t32 = tensor_mml_p(
      {3},
      {12.0f, 15.0f, 18.0f});

  shared_ptr<Tensor<float>> expected_t40 = tensor_mml_p(
      {3},
      {19.0f, 22.0f, 25.0f});

  shared_ptr<Tensor<float>> expected_t41 = tensor_mml_p(
      {3},
      {20.0f, 23.0f, 26.0f});

  shared_ptr<Tensor<float>> expected_t42 = tensor_mml_p(
      {3},
      {21.0f, 24.0f, 27.0f});

  ASSERT_EQ(*expected_t20, *t20);
  ASSERT_EQ(*expected_t21, *t21);
  ASSERT_EQ(*expected_t22, *t22);
  ASSERT_EQ(*expected_t30, *t30);
  ASSERT_EQ(*expected_t31, *t31);
  ASSERT_EQ(*expected_t32, *t32);
  ASSERT_EQ(*expected_t40, *t40);
  ASSERT_EQ(*expected_t41, *t41);
  ASSERT_EQ(*expected_t42, *t42);
}

TEST(test_mml_tensor, test_slicing_7) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p(
      {2, 2, 5},
      {1,  2,  3,  4,  5,
       6,  7,  8,  9,  10,

       11, 12, 13, 14, 15,
       16, 17, 18, 19, 20});

  shared_ptr<Tensor<int>> t2 = t1->slice({0});
  shared_ptr<Tensor<int>> t3 = t1->slice({1});
  
  shared_ptr<Tensor<int>> expected_t2 = tensor_mml_p(
      {2, 5},
      {1,  2,  3,  4,  5,
       6,  7,  8,  9,  10});

  shared_ptr<Tensor<int>> expected_t3 = tensor_mml_p(
      {2, 5},
      {11, 12, 13, 14, 15,
       16, 17, 18, 19, 20});

  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);

  shared_ptr<Tensor<int>> t4 = t1->slice({0, 0});
  shared_ptr<Tensor<int>> t5 = t1->slice({1, 0});
  shared_ptr<Tensor<int>> t6 = t1->slice({0, 1});
  shared_ptr<Tensor<int>> t7 = t1->slice({1, 1});
  shared_ptr<Tensor<int>> t8 = t1->slice({0, 2});
  shared_ptr<Tensor<int>> t9 = t1->slice({1, 2});
  shared_ptr<Tensor<int>> t10 = t1->slice({0, 3});
  shared_ptr<Tensor<int>> t11 = t1->slice({1, 3});
  shared_ptr<Tensor<int>> t12 = t1->slice({0, 4});
  shared_ptr<Tensor<int>> t13 = t1->slice({1, 4});

  shared_ptr<Tensor<int>> expected_t4 = tensor_mml_p(
      {2},
      {1, 6});

  shared_ptr<Tensor<int>> expected_t5 = tensor_mml_p(
      {2},
      {11, 16});

  shared_ptr<Tensor<int>> expected_t6 = tensor_mml_p(
      {2},
      {2, 7});

  shared_ptr<Tensor<int>> expected_t7 = tensor_mml_p(
      {2},
      {12, 17});

  shared_ptr<Tensor<int>> expected_t8 = tensor_mml_p(
      {2},
      {3, 8});

  shared_ptr<Tensor<int>> expected_t9 = tensor_mml_p(
      {2},
      {13, 18});

  shared_ptr<Tensor<int>> expected_t10 = tensor_mml_p(
      {2},
      {4, 9});

  shared_ptr<Tensor<int>> expected_t11 = tensor_mml_p(
      {2},
      {14, 19});

  shared_ptr<Tensor<int>> expected_t12 = tensor_mml_p(
      {2},
      {5, 10});

  shared_ptr<Tensor<int>> expected_t13 = tensor_mml_p(
      {2},
      {15, 20});

  ASSERT_EQ(*expected_t4, *t4);
  ASSERT_EQ(*expected_t5, *t5);
  ASSERT_EQ(*expected_t6, *t6);
  ASSERT_EQ(*expected_t7, *t7);
  ASSERT_EQ(*expected_t8, *t8);
  ASSERT_EQ(*expected_t9, *t9);
  ASSERT_EQ(*expected_t10, *t10);
  ASSERT_EQ(*expected_t11, *t11);
  ASSERT_EQ(*expected_t12, *t12);
  ASSERT_EQ(*expected_t13, *t13);
}

TEST(test_mml_tensor, test_buffer_reverse_1) {
  shared_ptr<Tensor<int>> t1 = tensor_mml_p(
      {3, 3},
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9});

  
  t1->reverse_buffer();

  shared_ptr<Tensor<int>> expected_t1 = tensor_mml_p(
      {3, 3},
      {9, 8, 7,
       6, 5, 4,
       3, 2, 1});

  ASSERT_EQ(*expected_t1, *t1);
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