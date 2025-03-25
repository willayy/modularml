#include <gtest/gtest.h>

#include <modularml>

const shared_ptr<ArithmeticModule<float>> am = make_shared<Arithmetic_mml<float>>();

// Test add
TEST(test_mml_arithmetic, test_add_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {5, 7, 9, 11, 13, 15});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_add_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {2, 4, 6, 8, 10, 12, 14, 16, 18});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const float b = 2;
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {2, 4, 6, 8, 10, 12});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const float b = 0.5;
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {-3, -3, -3, -3, -3, -3});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}

float square(float x) { return x * x; }
TEST(test_mml_arithmetic, test_elementwise) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f});
  am->elementwise(a, square, b);
  ASSERT_EQ(*b, *c);
}

TEST(test_mml_arithmetic, test_elementwise_in_place) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f});
  am->elementwise_in_place(a, square);
  ASSERT_EQ(*a, *b);
}

TEST(test_mml_arithmetic, test_elementwise_in_place_many_dimensions_4D) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 3, 2}, 
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  
     7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f,  

    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,  
    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});

  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({2, 2, 3, 2}, 
    {1.0f,   4.0f,   9.0f,  16.0f,  25.0f,  36.0f,  
     49.0f,  64.0f,  81.0f, 100.0f, 121.0f, 144.0f,  

    169.0f, 196.0f, 225.0f, 256.0f, 289.0f, 324.0f,  
    361.0f, 400.0f, 441.0f, 484.0f, 529.0f, 576.0f});

  am->elementwise_in_place(a, square);
  ASSERT_EQ(*a, *b);
}
TEST(test_mml_arithmetic, test_reduce_max_axis0) {
  // Create a 2D tensor (2 rows, 3 columns)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {
      1, 2, 3,  // Row 1
      4, 5, 6   // Row 2
  });

  // Reduce along axis 0 (rows), keeping only the max per column
  const int axis = 0;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({3}, {4, 5, 6});

  auto c = am->reduce_max(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_max) {
  // 2D tensor (2x3)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {
      1, 2, 3,
      4, 5, 6
  });

  // Reduce along axis 1 (columns), keeping only the max per row
  const int axis = 1;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2}, {3, 6});

  auto c = am->reduce_max(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_max_3D) {
  // 3D tensor (2x2x3)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 3}, {
      1,  2,  3,  
      4,  5,  6,  
      7,  8,  9,  
      10, 11, 12   
  });

  // Reduce along axis 2 (depth), keeping only the max per row
  const int axis = 2;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 2}, {
      3, 6,    
      9, 12
  });

  auto c = am->reduce_max(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_max_4D_axis3) {
  // 4D tensor (2x2x2x3)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 2, 3}, {
      1, 2, 3,   4, 5, 6,  
      7, 8, 9,  10, 11, 12,   
      13, 14, 15,  16, 17, 18,  
      19, 20, 21,  22, 23, 24  
  });

  // Reduce along axis 3 (last dimension), keeping only the max per row
  const int axis = 3;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 2, 2}, {
      3, 6, 9, 12,  
      15, 18, 21, 24  
  });

  auto c = am->reduce_max(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_max_3D_large_dimension) {
  // Large 3D tensor (2x3x12)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3, 12}, {
      // First batch (2D slice of 3x12)
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,

      // Second batch (2D slice of 3x12)
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
      49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
  });

  // Reduce along the last axis (axis=2)
  const int axis = 2;

  // Expected shape after reduction: (2, 3, 1)
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 3, 1}, {
      12, 24, 36,  
      48, 60, 72   
  });

  auto c = am->reduce_max(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_3D_axis0) {
  // Summing along the first axis (collapsing the first dimension)
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 3}, {
      1,  2,  3,  4,  5,  6,  
      7,  8,  9, 10, 11, 12   
  });

  const int axis = 0;  
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 3}, {
      8, 10, 12,  
      14, 16, 18   
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_3D_axis1) {
  // Summing along the middle axis
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 3}, {
      1,  2,  3,  4,  5,  6,  
      7,  8,  9, 10, 11, 12   
  });

  const int axis = 1;  
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 3}, {
      5,  7,  9,   
      17, 19, 21   
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_4D_axis3) {
  // Summing along the last axis
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 2, 3}, {
      1, 2, 3,   4, 5, 6,  
      7, 8, 9,  10, 11, 12,   
      13, 14, 15,  16, 17, 18,  
      19, 20, 21,  22, 23, 24  
  });

  const int axis = 3;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 2, 2}, {
      6, 15, 24, 33,  
      42, 51, 60, 69  
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_4D_axis1) {
  // Summing across the second axis
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3, 2, 3}, {
      1, 2, 3,   4, 5, 6,  
      7, 8, 9,  10, 11, 12,   
      13, 14, 15,  16, 17, 18,  
      19, 20, 21,  22, 23, 24,  
      25, 26, 27,  28, 29, 30,  
      31, 32, 33,  34, 35, 36   
  });

  const int axis = 1;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 2, 3}, {
      21, 24, 27,  30, 33, 36,  
      75, 78, 81,  84, 87, 90  
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_negative_values) {
  // Summing with negative values
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {
      -1, -2, -3,  
      -4, -5, -6  
  });

  const int axis = 1;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2}, {-6, -15});

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

TEST(test_mml_arithmetic, test_reduce_sum_floats) {
  // Summing floating-point values
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {
      1.1, 2.2, 3.3,  
      4.4, 5.5, 6.6  
  });

  const int axis = 1;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2}, {6.6, 16.5});

  auto c = am->reduce_sum(a, axis);
  ASSERT_NEAR((*c)[0], (*expected)[0], 1e-5);
  ASSERT_NEAR((*c)[1], (*expected)[1], 1e-5);
}

TEST(test_mml_arithmetic, test_reduce_sum_reduced) {
  // Summing an already reduced tensor
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({1, 3}, {1, 2, 3});

  const int axis = 0;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({3}, {1, 2, 3});

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

// Test reducing along multiple axes
TEST(test_mml_arithmetic, test_reduce_sum_multiple_axes) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 2, 2}, {
      1, 2, 3, 4,  
      5, 6, 7, 8  
  });

  const shared_ptr<Tensor<float>> expected_axis0 = tensor_mml_p<float>({2, 2}, {6, 8, 10, 12});
  const shared_ptr<Tensor<float>> expected_axis1 = tensor_mml_p<float>({2, 2}, {4, 6, 12, 14});
  const shared_ptr<Tensor<float>> expected_axis2 = tensor_mml_p<float>({2, 2}, {3, 7, 11, 15});

  auto sum_axis0 = am->reduce_sum(a, 0);
  auto sum_axis1 = am->reduce_sum(a, 1);
  auto sum_axis2 = am->reduce_sum(a, 2);

  ASSERT_EQ((*sum_axis0), (*expected_axis0));
  ASSERT_EQ((*sum_axis1), (*expected_axis1));
  ASSERT_EQ((*sum_axis2), (*expected_axis2));
}

// Test reducing a large 3D tensor with random values
TEST(test_mml_arithmetic, test_reduce_sum_large_3D) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 5, 10}, {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  
      31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  
      41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  
      
      51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  
      71, 72, 73, 74, 75, 76, 77, 78, 79, 80,  
      81, 82, 83, 84, 85, 86, 87, 88, 89, 90,  
      91, 92, 93, 94, 95, 96, 97, 98, 99, 100  
  });

  // Reduce along axis 2 (last dimension)
  const int axis = 2;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 5}, {
      55, 155, 255, 355, 455,  
      555, 655, 755, 855, 955  
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}

// Test reducing a large 4D tensor
TEST(test_mml_arithmetic, test_reduce_sum_large_4D) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3, 4, 5}, {
      // Batch 1
      1, 2, 3, 4, 5,   6, 7, 8, 9, 10,   11, 12, 13, 14, 15,   16, 17, 18, 19, 20,
      21, 22, 23, 24, 25,   26, 27, 28, 29, 30,   31, 32, 33, 34, 35,   36, 37, 38, 39, 40,
      41, 42, 43, 44, 45,   46, 47, 48, 49, 50,   51, 52, 53, 54, 55,   56, 57, 58, 59, 60,
      
      // Batch 2
      61, 62, 63, 64, 65,   66, 67, 68, 69, 70,   71, 72, 73, 74, 75,   76, 77, 78, 79, 80,
      81, 82, 83, 84, 85,   86, 87, 88, 89, 90,   91, 92, 93, 94, 95,   96, 97, 98, 99, 100,
      101, 102, 103, 104, 105,   106, 107, 108, 109, 110,   111, 112, 113, 114, 115,   116, 117, 118, 119, 120  
  });

  // Reduce along axis 3 (last dimension)
  const int axis = 3;
  const shared_ptr<Tensor<float>> expected = tensor_mml_p<float>({2, 3, 4}, {
      15, 40, 65, 90,  
      115, 140, 165, 190,  
      215, 240, 265, 290,  
      
      315, 340, 365, 390,  
      415, 440, 465, 490,  
      515, 540, 565, 590  
  });

  auto c = am->reduce_sum(a, axis);
  ASSERT_EQ((*c), (*expected));
}
