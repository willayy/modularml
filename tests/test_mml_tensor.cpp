#include <gtest/gtest.h>

#include <modularml>

// Test the default constructors
TEST(test_mml_tensor, test_default_constructor_1) {
  std::shared_ptr<Tensor<int>> t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3});
  auto expected_shape = array_mml<size_t>({3, 3});
  auto expected_t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const &actual_shape = t1->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t1), (*t1));
}

// Test the std::copy constructor
TEST(test_mml_tensor, test_copy_constructor_1) {
  std::shared_ptr<Tensor<int>> t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3});
  std::shared_ptr<Tensor<int>> t2 = t1;
  ASSERT_EQ((*t1), (*t2));
}

// Test the std::move constructor
TEST(test_mml_tensor, test_move_constructor_1) {
  std::shared_ptr<Tensor<int>> t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3});
  std::shared_ptr<Tensor<int>> t2 = std::move(t1);
  auto expected_shape = array_mml<size_t>({3, 3});
  auto expected_t2 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const &actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t2), (*t2));
}

// Test the std::copy assignment operator
TEST(test_mml_tensor, test_copy_assignment_1) {
  std::shared_ptr<Tensor<int>> t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3});
  std::shared_ptr<Tensor<int>> t2 =std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2});
  t2 = t1;
  ASSERT_EQ((*t1), (*t2));
}

// Test the std::move assignment operator
TEST(test_mml_tensor, test_move_assignment_1) {
  std::shared_ptr<Tensor<int>> t1 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3});
  std::shared_ptr<Tensor<int>> t2 =std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2});
  t2 = std::move(t1);
  auto expected_shape = array_mml<size_t>({3, 3});
  auto expected_t2 =std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const &actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);
  ASSERT_EQ((*expected_t2), (*t2));
}

// Test the std::move assignment using abstract class
TEST(test_mml_tensor, test_move_assignment_2) {
  std::shared_ptr<Tensor<int>> t1 =
      std::make_shared<Tensor<int>>(array_mml<size_t>({3, 3}));
  std::shared_ptr<Tensor<int>> t2 =
      std::make_shared<Tensor<int>>(array_mml<size_t>({2, 2}));
  *t2 = std::move(*t1);
  auto expected_shape = array_mml<size_t>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);

  auto actual_data = t2->get_data();
  ASSERT_EQ(expected_data, actual_data);
}

// Test the std::copy assignment using abstract class
TEST(test_mml_tensor, test_copy_assignment_2) {
  std::shared_ptr<Tensor<int>> t1 =
      std::make_shared<Tensor<int>>(array_mml<size_t>({3, 3}));
  std::shared_ptr<Tensor<int>> t2 =
      std::make_shared<Tensor<int>>(array_mml<size_t>({2, 2}));
  *t2 = *t1;
  auto expected_shape = array_mml<size_t>({3, 3});
  auto expected_data = array_mml<int>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto actual_shape = t2->get_shape();
  ASSERT_EQ(expected_shape, actual_shape);

  auto actual_data = t2->get_data();
  ASSERT_EQ(expected_data, actual_data);
}

// Generate an arbitrary tensor and check if all elements can be accessed using
// indices
TEST(test_mml_tensor, test_index_1) {
  for (int i = 0; i < 100; i++) {
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>();
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);
    for (int j = 0; j < (*t1).get_size(); j++) {
      (*t1)[j] = 101;
      ASSERT_EQ(101, (*t1)[j]);
    }
  }
}

TEST(test_mml_tensor, test_index_2) {
  for (int i = 0; i < 100; i++) {
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>();
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);
    for (int j = 0; j < (*t1).get_size(); j++) {
      ASSERT_EQ(data[j], (*t1)[j]);
    }
  }
}

TEST(test_mml_tensor, test_indices_1) {
  for (int i = 0; i < 100; i++) {
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>();
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);
    for (int j = 0; j < t1->get_size(); j++) {
      array_mml<size_t> indices = array_mml<size_t>(shape.size());
      int k = j;

      size_t l = shape.size() - 1;
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
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>();
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);

    auto indices = array_mml<size_t>(shape.size());
    indices.fill(0);

    for (size_t j = 0; j < t1->get_size(); j++) {
      ASSERT_EQ(data[j], (*t1)[indices]);

      size_t k = shape.size() - 1;
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
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>();
    const size_t elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);
    t1->reshape({elements});
    auto expected_shape = array_mml<size_t>({elements});
    auto actual_shape = t1->get_shape();
    ASSERT_EQ(expected_shape, actual_shape);
    auto expected_data = data;
    for (size_t j = 0; i < elements; i++) {
      ASSERT_EQ(expected_data[j], (*t1)[j]);
    }
  }
}

// Reshape into 1D into 2D tensor
TEST(test_mml_tensor, test_reshape_2) {
  for (int i = 0; i < 200; i++) {
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>(1, 1);
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data);
    if (shape[0] % 2 == 0) {
      size_t rows = shape[0] / 2;
      size_t cols = 2;
      auto new_shape = array_mml<size_t>({rows, cols});
      t1->reshape(new_shape);
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
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
  std::shared_ptr<Tensor<int>> t1 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::shared_ptr<Tensor<int>> t2 = t1->slice({0});
  std::shared_ptr<Tensor<int>> t3 = t1->slice({1});
  std::shared_ptr<Tensor<int>> t4 = t1->slice({2});
  std::shared_ptr<Tensor<int>> expected_t2 = std::make_shared<Tensor<int>>(array_mml<size_t>{3}, array_mml<int>{1, 4, 7});
  std::shared_ptr<Tensor<int>> expected_t3 = std::make_shared<Tensor<int>>(array_mml<size_t>{3}, array_mml<int>{2, 5, 8});
  std::shared_ptr<Tensor<int>> expected_t4 = std::make_shared<Tensor<int>>(array_mml<size_t>{3}, array_mml<int>{3, 6, 9});
  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_2) {
  std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3}, array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  std::shared_ptr<Tensor<float>> t2 = t1->slice({0});
  std::shared_ptr<Tensor<float>> t3 = t1->slice({1});
  std::shared_ptr<Tensor<float>> t4 = t1->slice({2});
  std::shared_ptr<Tensor<float>> expected_t2 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{1.0f, 4.0f, 7.0f});
  std::shared_ptr<Tensor<float>> expected_t3 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{2.0f, 5.0f, 8.0f});
  std::shared_ptr<Tensor<float>> expected_t4 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{3.0f, 6.0f, 9.0f});
  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_3) {
  std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3}, array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  std::shared_ptr<Tensor<float>> t2 = t1->slice({2});
  // Test indices access
  ASSERT_EQ(3.0f, (*t2)[{0}]);
  ASSERT_EQ(6.0f, (*t2)[{1}]);
  ASSERT_EQ(9.0f, (*t2)[{2}]);
}

TEST(test_mml_tensor, test_slicing_4) {
  std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3, 3},
    array_mml<float>{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,

                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,

                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});

  std::shared_ptr<Tensor<float>> t2 = t1->slice({0});
  std::shared_ptr<Tensor<float>> t3 = t1->slice({1});
  std::shared_ptr<Tensor<float>> t4 = t1->slice({2});

  std::shared_ptr<Tensor<float>> expected_t2 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3}, array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  std::shared_ptr<Tensor<float>> expected_t3 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3}, array_mml<float>{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});

  std::shared_ptr<Tensor<float>> expected_t4 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3}, array_mml<float>{19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});

  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);
  ASSERT_EQ(*expected_t4, *t4);
}

TEST(test_mml_tensor, test_slicing_5) {
  std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3, 3},
    array_mml<float>{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,

                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,

                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});

  std::shared_ptr<Tensor<float>> t2 = t1->slice({0, 0});
  std::shared_ptr<Tensor<float>> t3 = t1->slice({1, 0});
  std::shared_ptr<Tensor<float>> t4 = t1->slice({2, 0});
  std::shared_ptr<Tensor<float>> t5 = t1->slice({0, 1});
  std::shared_ptr<Tensor<float>> t6 = t1->slice({1, 1});
  std::shared_ptr<Tensor<float>> t7 = t1->slice({2, 1});
  std::shared_ptr<Tensor<float>> t8 = t1->slice({0, 2});
  std::shared_ptr<Tensor<float>> t9 = t1->slice({1, 2});
  std::shared_ptr<Tensor<float>> t10 = t1->slice({2, 2});

  std::shared_ptr<Tensor<float>> expected_t2 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{1.0f, 4.0f, 7.0f});

  std::shared_ptr<Tensor<float>> expected_t3 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{10.0f, 13.0f, 16.0f});

  std::shared_ptr<Tensor<float>> expected_t4 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{19.0f, 22.0f, 25.0f});

  std::shared_ptr<Tensor<float>> expected_t5 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{2.0f, 5.0f, 8.0f});

  std::shared_ptr<Tensor<float>> expected_t6 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{11.0f, 14.0f, 17.0f});

  std::shared_ptr<Tensor<float>> expected_t7 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{20.0f, 23.0f, 26.0f});

  std::shared_ptr<Tensor<float>> expected_t8 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{3.0f, 6.0f, 9.0f});

  std::shared_ptr<Tensor<float>> expected_t9 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{12.0f, 15.0f, 18.0f});

  std::shared_ptr<Tensor<float>> expected_t10 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{21.0f, 24.0f, 27.0f});

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
  std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 3, 3},
    array_mml<float>{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,

                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,

                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});

  // Slice once
  std::shared_ptr<Tensor<float>> t2 = t1->slice({0});
  std::shared_ptr<Tensor<float>> t3 = t1->slice({1});
  std::shared_ptr<Tensor<float>> t4 = t1->slice({2});
  // Slice the slices
  std::shared_ptr<Tensor<float>> t20 = t2->slice({0});
  std::shared_ptr<Tensor<float>> t21 = t2->slice({1});
  std::shared_ptr<Tensor<float>> t22 = t2->slice({2});
  std::shared_ptr<Tensor<float>> t30 = t3->slice({0});
  std::shared_ptr<Tensor<float>> t31 = t3->slice({1});
  std::shared_ptr<Tensor<float>> t32 = t3->slice({2});
  std::shared_ptr<Tensor<float>> t40 = t4->slice({0});
  std::shared_ptr<Tensor<float>> t41 = t4->slice({1});
  std::shared_ptr<Tensor<float>> t42 = t4->slice({2});

  std::shared_ptr<Tensor<float>> expected_t20 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{1.0f, 4.0f, 7.0f});

  std::shared_ptr<Tensor<float>> expected_t21 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{2.0f, 5.0f, 8.0f});

  std::shared_ptr<Tensor<float>> expected_t22 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{3.0f, 6.0f, 9.0f});

  std::shared_ptr<Tensor<float>> expected_t30 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{10.0f, 13.0f, 16.0f});

  std::shared_ptr<Tensor<float>> expected_t31 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{11.0f, 14.0f, 17.0f});

  std::shared_ptr<Tensor<float>> expected_t32 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{12.0f, 15.0f, 18.0f});

  std::shared_ptr<Tensor<float>> expected_t40 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{19.0f, 22.0f, 25.0f});

  std::shared_ptr<Tensor<float>> expected_t41 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{20.0f, 23.0f, 26.0f});

  std::shared_ptr<Tensor<float>> expected_t42 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{21.0f, 24.0f, 27.0f});

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
  std::shared_ptr<Tensor<int>> t1 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2, 5}, 
      array_mml<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  std::shared_ptr<Tensor<int>> t2 = t1->slice({0});
  std::shared_ptr<Tensor<int>> t3 = t1->slice({1});

  std::shared_ptr<Tensor<int>> expected_t2 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{2, 5}, array_mml<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  std::shared_ptr<Tensor<int>> expected_t3 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{2, 5}, array_mml<int>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  ASSERT_EQ(*expected_t2, *t2);
  ASSERT_EQ(*expected_t3, *t3);

  std::shared_ptr<Tensor<int>> t4 = t1->slice({0, 0});
  std::shared_ptr<Tensor<int>> t5 = t1->slice({1, 0});
  std::shared_ptr<Tensor<int>> t6 = t1->slice({0, 1});
  std::shared_ptr<Tensor<int>> t7 = t1->slice({1, 1});
  std::shared_ptr<Tensor<int>> t8 = t1->slice({0, 2});
  std::shared_ptr<Tensor<int>> t9 = t1->slice({1, 2});
  std::shared_ptr<Tensor<int>> t10 = t1->slice({0, 3});
  std::shared_ptr<Tensor<int>> t11 = t1->slice({1, 3});
  std::shared_ptr<Tensor<int>> t12 = t1->slice({0, 4});
  std::shared_ptr<Tensor<int>> t13 = t1->slice({1, 4});

  std::shared_ptr<Tensor<int>> expected_t4 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{1, 6});

  std::shared_ptr<Tensor<int>> expected_t5 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{11, 16});

  std::shared_ptr<Tensor<int>> expected_t6 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{2, 7});

  std::shared_ptr<Tensor<int>> expected_t7 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{12, 17});

  std::shared_ptr<Tensor<int>> expected_t8 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{3, 8});

  std::shared_ptr<Tensor<int>> expected_t9 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{13, 18});

  std::shared_ptr<Tensor<int>> expected_t10 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{4, 9});

  std::shared_ptr<Tensor<int>> expected_t11 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{14, 19});

  std::shared_ptr<Tensor<int>> expected_t12 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{5, 10});

  std::shared_ptr<Tensor<int>> expected_t13 = std::make_shared<Tensor<int>>(array_mml<size_t>{2}, array_mml<int>{15, 20});

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
  std::shared_ptr<Tensor<int>> t1 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{1, 2, 3, 4, 5, 6, 7, 8, 9});

  t1->reverse_buffer();

  std::shared_ptr<Tensor<int>> expected_t1 =
    std::make_shared<Tensor<int>>(array_mml<size_t>{3, 3}, array_mml<int>{9, 8, 7, 6, 5, 4, 3, 2, 1});

  ASSERT_EQ(*expected_t1, *t1);
}

TEST(test_mml_tensor, test_to_string) {
  const auto t1 = Tensor<int>({3, 3});
  const std::string ptr_str =
      "Pointer: " + std::to_string(reinterpret_cast<uint64_t>(&t1));
  std::string expected =
      "Tensor<i> " + ptr_str +
      ", Shape: [3, 3], Size: 9, Data: [0, 0, 0, 0, 0, 0, 0, 0, 0]";
  std::string actual = t1.to_string();
  ASSERT_EQ(expected, actual);
}

TEST(test_mml_tensor, tensor_utility_tensors_are_close) {
  const auto t1 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{1.0f, 2.3f, 0.0f, -3.2f, 5.1f, 2.0f});
  const auto t2 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{1.0f, 2.3f, 0.000002f, -3.2f, 5.1f, 2.0f});
  const auto t3 =
    std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{1.0f, 2.3f, 0.00002f, -3.2f, 5.1f, 2.0f});

  ASSERT_TRUE(TensorUtils::tensors_are_close(*t1, *t2));
  ASSERT_FALSE(TensorUtils::tensors_are_close(*t1, *t3));
}

TEST(test_mml_tensor, broadcast_1D_to_2D) {
  // Given a 1D tensor of shape [3]
  auto tensor1D = std::make_shared<Tensor<float>>(array_mml<size_t>{3}, array_mml<float>{1.0f, 2.0f, 3.0f});

  // Broadcast it to shape [2, 3]
  auto broadcasted = tensor1D->broadcast_to({2, 3});

  // Expected output: row-wise repetition
  auto expected = std::make_shared<Tensor<float>>(
    array_mml<size_t>{2, 3}, array_mml<float>{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});

  ASSERT_EQ(*broadcasted, *expected);
}

TEST(test_mml_tensor, broadcast_general_2D_to_3D) {
  // Shape: [1, 3] → Target: [2, 4, 3]
  auto tensor = std::make_shared<Tensor<int>>(array_mml<size_t>{1, 3}, array_mml<int>{10, 20, 30});
  auto broadcasted = tensor->broadcast_to({2, 4, 3});

  auto expected = std::make_shared<Tensor<int>>(
    array_mml<size_t>{2, 4, 3},
    array_mml<int>{
          10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, // first batch
          10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30  // second batch
      });

  ASSERT_EQ(*broadcasted, *expected);
}

TEST(test_mml_tensor, broadcast_scalar_to_2D) {
  std::cout << "1" << std::endl;
  auto scalar = std::make_shared<Tensor<float>>(array_mml<size_t>{}, array_mml<float>{42.0f});
  std::cout << "2" << std::endl;
  auto broadcasted = scalar->broadcast_to({2, 2});
  std::cout << "3" << std::endl;
  auto expected =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{42.0f, 42.0f, 42.0f, 42.0f});
  std::cout << "4" << std::endl;
  ASSERT_EQ(*broadcasted, *expected);
}

TEST(test_mml_tensor, is_broadcastable_positive) {
  auto tensor = std::make_shared<Tensor<int>>(array_mml<size_t>{1, 3});
  EXPECT_NO_THROW({ auto b = tensor->broadcast_to({2, 4, 3}); });
}

TEST(test_mml_tensor, is_broadcastable_negative) {
  auto tensor = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3});
  EXPECT_THROW(tensor->broadcast_to({2, 4, 3}), std::invalid_argument);
}

TEST(test_mml_tensor, transpose_2D) {
  auto tensor = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});

  auto transposed =
      tensor->transpose(); // Should default to swapping last 2 dims

  auto expected = std::make_shared<Tensor<int>>(array_mml<size_t>{3, 2}, array_mml<int>{1, 4, 2, 5, 3, 6});

  ASSERT_EQ(*transposed, *expected);
}

TEST(test_mml_tensor, transpose_high_rank_dims) {
  auto tensor =
    std::make_shared<Tensor<int>>(array_mml<size_t>{2, 1, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});

  // Swap dim 0 and dim 2 → expected shape: [3, 1, 2]
  auto transposed = tensor->transpose(0, 2);

  auto expected =
    std::make_shared<Tensor<int>>(array_mml<size_t>{3, 1, 2}, array_mml<int>{1, 4, 2, 5, 3, 6});

  ASSERT_EQ(*transposed, *expected);
}