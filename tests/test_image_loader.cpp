#include <gtest/gtest.h>

#include "backend/dataloader/image_loader.hpp"
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

TEST(test_image_loader, load_mnist_image_test) {
    fs::path cwd = fs::current_path();
    std::cout << "Current working directory: " << cwd << std::endl;
    
    const ImageLoaderConfig config("../tests/data/mnist_5.jpg");
    ImageLoader loader;
    
    auto image_tensor = loader.load(config);

    std::cout << image_tensor << std::endl;

    // XD
    EXPECT_EQ(array_mml({static_cast<unsigned long int>(1), static_cast<unsigned long int>(1), static_cast<unsigned long int>(28), static_cast<unsigned long int>(28)}), image_tensor->get_shape());
    
    for (int i = 0; i < image_tensor->get_size(); i++) {
        if (i % 28 == 0) std::cout << "\n";
        std::cout << std::fixed << std::setprecision(1) << (*image_tensor)[i] << " ";
    }
    EXPECT_EQ(1, 2);
}