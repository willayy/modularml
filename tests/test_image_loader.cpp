#include <gtest/gtest.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <modularml>

namespace fs = std::filesystem;

TEST(test_image_loader, load_mnist_image_jpg_test) {
    const ImageLoaderConfig config("../tests/data/mnist_5.jpg");
    ImageLoader loader;

    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(1), static_cast<unsigned long int>(28), static_cast<unsigned long int>(28)}), image_tensor->get_shape());
}

TEST(test_image_loader, load_rgb_image_png_test) {
    const ImageLoaderConfig config("../tests/data/rgb_test.png");
    ImageLoader loader;

    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(3), static_cast<unsigned long int>(100), static_cast<unsigned long int>(100)}), image_tensor->get_shape());
}

TEST(test_image_loader, load_jpeg_image) {
    const ImageLoaderConfig config("../tests/data/alps.JPEG");
    ImageLoader loader;

    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(3), static_cast<unsigned long int>(375), static_cast<unsigned long int>(500)}), image_tensor->get_shape());
}