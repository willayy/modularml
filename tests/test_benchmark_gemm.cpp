#include <gtest/gtest.h>
#include <modularml>


TEST(test_benchmark_gemm, benchmark_gemm_128x128_float) {
    array_mml<float> a_data = generate_array_random_content_real<float>(128 * 128, 0, 100);
    array_mml<float> b_data = generate_array_random_content_real<float>(128 * 128, 0, 100);
    
    shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({128, 128}, a_data);
    shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({128, 128}, b_data);
    shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({128, 128});
    
    TensorOperationsModule::gemm<float>(0, 0, 128, 128, 128, 1, a, 128, b, 128, 0, c, 128);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_256x256_float) {
    array_mml<float> a_data = generate_array_random_content_real<float>(256 * 256, 0, 100);
    array_mml<float> b_data = generate_array_random_content_real<float>(256 * 256, 0, 100);
    
    shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({256, 256}, a_data);
    shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({256, 256}, b_data);
    shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({256, 256});
    
    TensorOperationsModule::gemm<float>(0, 0, 256, 256, 256, 1, a, 256, b, 256, 0, c, 256);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_512x512_float) {
    array_mml<float> a_data = generate_array_random_content_real<float>(512 * 512, 0, 100);
    array_mml<float> b_data = generate_array_random_content_real<float>(512 * 512, 0, 100);
    
    shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({512, 512}, a_data);
    shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({512, 512}, b_data);
    shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({512, 512});
    
    TensorOperationsModule::gemm<float>(0, 0, 512, 512, 512, 1, a, 512, b, 512, 0, c, 512);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}


TEST(test_benchmark_gemm, benchmark_gemm_1024x1024_float) {
    array_mml<float> a_data = generate_array_random_content_real<float>(1024 * 1024, 0, 100);
    array_mml<float> b_data = generate_array_random_content_real<float>(1024 * 1024, 0, 100);
    
    shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({1024, 1024}, a_data);
    shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({1024, 1024}, b_data);
    shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({1024, 1024});
    
    TensorOperationsModule::gemm<float>(0, 0, 1024, 1024, 1024, 1, a, 1024, b, 1024, 0, c, 1024);
    
    ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}