#include <gtest/gtest.h>

#include <filesystem>
#include <iomanip>
#include <iostream>

#include "backend/dataloader/image_loader.hpp"

namespace fs = std::filesystem;

TEST(test_image_loader, load_mnist_image_jpg_test) {
    fs::path cwd = fs::current_path();
    std::cout << "Current working directory: " << cwd << std::endl;

    const ImageLoaderConfig config("../tests/data/mnist_5.jpg");
    ImageLoader loader;

    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    // XD
    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(1), static_cast<unsigned long int>(28), static_cast<unsigned long int>(28)}), image_tensor->get_shape());

    // Use this to verify the tensor in the console
    /* for (int i = 0; i < image_tensor->get_size(); i++) {
        if (i % 28 == 0) std::cout << "\n";
        std::cout << std::fixed << std::setprecision(1) << (*image_tensor)[i] << " ";
    } */
}

TEST(test_image_loader, load_rgb_image_png_test) {
    const ImageLoaderConfig config("../tests/data/rgb_test.png");
    ImageLoader loader;

    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    // XD
    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(3), static_cast<unsigned long int>(100), static_cast<unsigned long int>(100)}), image_tensor->get_shape());

    // Used to visualize and verify the expected tensor output
    /* for (int c = 0; c < image_tensor->get_shape()[1]; c++) {
        std::cout << "Channel " << c << ":\n";
        for (int i = 0; i < image_tensor->get_shape()[2]; i++) {
            for (int j = 0; j < image_tensor->get_shape()[3]; j++) {
                int index = c * (image_tensor->get_shape()[2] * image_tensor->get_shape()[3]) + i * image_tensor->get_shape()[3] + j;
                float num = (*image_tensor)[index];
                char cs = ' ';
                if (num > 0.2) cs = '.';
                if (num > 0.4) cs = '>';
                if (num > 0.6) cs = '%';
                if (num > 0.8) cs = '#';

                std::cout << std::fixed << std::setprecision(1) << cs;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    } */
}