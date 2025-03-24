#include <gtest/gtest.h>

#include "image_loader.hpp"


TEST(image_loader_test, test_load_mnist_digit) {
    ImageLoader<float> loader;

    std::string path = "test.jpg";

    auto tensor = loader.load(path, 28, 28);
    EXPECT_EQ(tensor->get_shape()[0], 1);
}