#include <gtest/gtest.h>

#include "log_softmax_node.hpp"
#include "mml_tensor.hpp"


TEST(test_log_softmax_node, test_forward_basic) {
    array_mml<int> x_shape({3, 3});
    array_mml<float> x_values({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f,
                               7.0f, 8.0f, 9.0f});

    array_mml<int> y_shape({3, 3});

    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(x_shape, x_values);
    shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

    LogSoftMaxNode<float> logsoftmax(X, Y);

    logsoftmax.forward();

    for (int b = 0; b < Y->get_shape()[0]; b++) {

        float row_sum = 0;
        
        for (int c = 0; c < Y->get_shape()[1]; c++) {
            row_sum += std::exp( (*Y)[{b, c}] ); // exponentiate the result so we can check that they sum to 1
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5); // Checks that each row sums to 1 
    }
}

TEST(test_log_softmax_node, test_forward_large_range_of_values) {
    array_mml<int> x_shape({3, 3});
    array_mml<float> x_values({
        1.0f, 1000.0f, 1000000.0f,
        0.001f, 10.0f, 100.0f,
        -100.0f, -10.0f, -1.0f});

    array_mml<int> y_shape({3, 3});

    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(x_shape, x_values);
    shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

    LogSoftMaxNode<float> logsoftmax(X, Y);

    logsoftmax.forward();

    for (int b = 0; b < Y->get_shape()[0]; b++) {
        float row_sum = 0;
        
        for (int c = 0; c < Y->get_shape()[1]; c++) {
            row_sum += std::exp( (*Y)[{b, c}] ); // exponentiate the result so we can check that they sum to 1
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5); // Checks that each row sums to 1 which it should
    }
}

TEST(test_log_softmax_node, test_forward_handle_zeros) {
    // Should result in an equal distribution
    array_mml<int> x_shape({3, 3});
    array_mml<float> x_values({
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    });

    array_mml<int> y_shape({3, 3});

    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(x_shape, x_values);
    shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

    LogSoftMaxNode<float> logsoftmax(X, Y);

    logsoftmax.forward();

    for (int b = 0; b < Y->get_shape()[0]; b++) {
        float row_sum = 0;
        
        for (int c = 0; c < Y->get_shape()[1]; c++) {
            row_sum += std::exp( (*Y)[{b, c}] ); // exponentiate the result so we can check that they sum to 1
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5); // Checks that each row sums to 1 which it should
    }
}

TEST(test_log_softmax_node, test_forward_maxfloat_minfloat_values) {
    // Checks that the node can handle very large and very small floats
    array_mml<int> x_shape({3, 3});
    array_mml<float> x_values({
        FLT_MAX, FLT_MIN, 1.0f,
        FLT_MAX, FLT_MIN, 1.0f,
        FLT_MAX, FLT_MIN, 1.0f
    });

    array_mml<int> y_shape({3, 3});

    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(x_shape, x_values);
    shared_ptr<Tensor_mml<float>> Y = make_shared<Tensor_mml<float>>(y_shape);

    LogSoftMaxNode<float> logsoftmax(X, Y);

    logsoftmax.forward();

    for (int b = 0; b < Y->get_shape()[0]; b++) {
        float row_sum = 0;
        
        for (int c = 0; c < Y->get_shape()[1]; c++) {
            row_sum += std::exp( (*Y)[{b, c}] ); // exponentiate the result so we can check that they sum to 1
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5); // Checks that each row sums to 1 which it should
    }
}


