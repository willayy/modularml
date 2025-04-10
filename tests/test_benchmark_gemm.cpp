#include <gtest/gtest.h>
#include <modularml>
#include "fstream"

// Fixture
namespace {
    template <typename T>
    std::tuple<
        std::shared_ptr<Tensor<T>>,
        std::shared_ptr<Tensor<T>>,
        std::shared_ptr<Tensor<T>>
    > generate_random_tensors(unsigned long int M, unsigned long int N, unsigned long int K, T min=0, T max=100) {
        auto a_data = generate_array_random_content_real<T>(M * K, min, max);
        auto b_data = generate_array_random_content_real<T>(K * N, min, max);
        auto a = TensorFactory::create_tensor<T>({M, K}, a_data);
        auto b = TensorFactory::create_tensor<T>({K, N}, b_data);
        auto c = TensorFactory::create_tensor<T>({M, N});
        return {a, b, c};
    }  
}

TEST(test_benchmark_gemm, benchmark_gemm_128x128_float) {
    auto [a, b, c] = generate_random_tensors<float>(128, 128, 128);
    
    TensorOperationsModule::gemm<float>(0, 0, 128, 128, 128, 1, a, 128, b, 128, 0, c, 128);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_256x256_float) {
    auto [a, b, c] = generate_random_tensors<float>(256, 256, 256);

    TensorOperationsModule::gemm<float>(0, 0, 256, 256, 256, 1, a, 256, b, 256, 0, c, 256);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_512x512_float) {
    auto [a, b, c] = generate_random_tensors<float>(512, 512, 512);
    
    TensorOperationsModule::gemm<float>(0, 0, 512, 512, 512, 1, a, 512, b, 512, 0, c, 512);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_1024x1024_float) {
    auto [a, b, c] = generate_random_tensors<float>(1024, 1024, 1024);
    
    TensorOperationsModule::gemm<float>(0, 0, 1024, 1024, 1024, 1, a, 1024, b, 1024, 0, c, 1024);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}

TEST(test_benchmark_gemm, get_benchmark_data) {
    std::string backend = "naive";

    std::ofstream file("results.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
        return;  // Or handle the error as needed
    }
    file << "backend,size,time,flops\n";

    for (int size = 128; size <= 2048; size += 128) {
        auto [a, b, c] = generate_random_tensors<float>(size, size, size);
        
        auto start = std::chrono::high_resolution_clock::now();
        TensorOperationsModule::gemm<float>(0, 0, size, size, size, 1, a, size, b, size, 0, c, size);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        double time = elapsed.count();

        double flops = 2.0 * size * size * size;
        file << backend << "," << size << "," << time << "," << flops << "\n";
    }

    file.close();

    ASSERT_TRUE(1);
}