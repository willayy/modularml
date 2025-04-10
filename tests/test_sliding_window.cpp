#include <gtest/gtest.h>
#include <modularml>
#include "nodes/node_utils.hpp"

template <typename T>
function<T(const std::vector<T>&, const std::vector<int64_t>&, int64_t&)> max_pool_reducer() {
    return [](const std::vector<T>& windowValues, const std::vector<int64_t>& wi, int64_t& outIndex) -> T {
        if (windowValues.empty()) {
            return static_cast<T>(0);
        }
        T max_val = windowValues[0];
        outIndex = 0;
        for (size_t i = 1; i < windowValues.size(); ++i) {
            if (windowValues[i] > max_val) {
                max_val = windowValues[i];
                outIndex = static_cast<int64_t>(i);
            }
        }
        // We ignore 'outIndex' for now.
        return max_val;
    };
}
template <typename T>
function<T(const std::vector<T>&, const std::vector<int64_t>&, int64_t&)> avg_pool_reducer() {
    return [](const std::vector<float>& windowValues, const std::vector<int64_t>& wi, int64_t& outIndex) -> float {
        if (windowValues.empty()) {
            return 0.0f;
        }
        
        float sum = 0.0f;
        for (const auto& val : windowValues) {
            sum += val;
        }
        return sum / static_cast<float>(windowValues.size());
    };
}

TEST(test_sliding_window, max_pool_2d) {
    // Create an input tensor of shape [1, 1, 4, 4].
    // For example, the input matrix (4x4) is:
    //  1  2  3  4
    //  5  6  7  8
    //  9  10 11 12
    //  13 14 15 16
    array_mml<uli> input_shape = {1, 1, 4, 4};
    array_mml<float> input_data = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // For max pooling with a 2x2 kernel, stride 2, no dilation, no padding,
    // the output shape should be [1, 1, 2, 2].
    array_mml<uli> output_shape = {1, 1, 2, 2};
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);

    // For this test, we are ignoring indices. Pass std::nullopt.
    std::optional<std::shared_ptr<Tensor<int64_t>>> indices_out = std::nullopt;

    // Define the pooling parameters.
    std::vector<int> kernel_shape = {2, 2};         // 2x2 kernel.
    std::vector<int> strides = {2, 2};              // Stride of 2 in each spatial dimension.
    std::vector<int> dilations = {1, 1};            // No dilation.
    // For NOTSET auto_pad and no padding, explicit pads: { {0,0}, {0,0} }.
    std::vector<std::pair<int, int>> pads = { {0, 0}, {0, 0} };

    // Call the general sliding window function.
    // This function will iterate over the output tensor indices, 
    // extract the corresponding 2x2 window from the input, and apply the max reducer.
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        indices_out,
        kernel_shape,
        strides,
        dilations,
        pads,
        0,
        max_pool_reducer<float>()
    );

    // Expected output values for max pooling.
    // For the 4x4 input above, the 2x2 windows are:
    // Window 1 (top-left): {1, 2, 5, 6} -> max: 6
    // Window 2 (top-right): {3, 4, 7, 8} -> max: 8
    // Window 3 (bottom-left): {9, 10, 13, 14} -> max: 14
    // Window 4 (bottom-right): {11, 12, 15, 16} -> max: 16
    // Flattened as [6, 8, 14, 16].
    array_mml<float> expected_output = {6, 8, 14, 16};
    auto expected_output_tensor = TensorFactory::create_tensor<float>(output_shape, expected_output);

    // Verify the output tensor.
    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_output_tensor));
}

TEST(test_sliding_window, max_pool_1d) {
    array_mml<uli> input_shape = {1, 1, 6};  // Shape: [N, C, L]
    array_mml<float> input_data = {1, 3, 2, 5, 4, 6};
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    array_mml<uli> output_shape = {1, 1, 3};  // Expect 3 windows
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);

    std::vector<int> kernel = {2};
    std::vector<int> strides = {2};
    std::vector<int> dilations = {1};
    std::vector<std::pair<int, int>> pads = {{0, 0}};

    TensorOperationsModule::sliding_window<float>(
        input_tensor, output_tensor, std::nullopt,
        kernel, strides, dilations, pads, 0,
        max_pool_reducer<float>()
    );

    array_mml<float> expected = {3, 5, 6};  // max([1,3], [2,5], [4,6])
    auto expected_tensor = TensorFactory::create_tensor<float>(output_shape, expected);

    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_tensor));
}

TEST(test_sliding_window, max_pool_3d) {
    array_mml<uli> input_shape = {1, 1, 2, 2, 2};  // Shape: [N, C, D, H, W]
    array_mml<float> input_data = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    array_mml<uli> output_shape = {1, 1, 1, 1, 1};  // Just one window
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);

    std::vector<int> kernel = {2, 2, 2};
    std::vector<int> strides = {1, 1, 1};
    std::vector<int> dilations = {1, 1, 1};
    std::vector<std::pair<int, int>> pads = {{0, 0}, {0, 0}, {0, 0}};

    TensorOperationsModule::sliding_window<float>(
        input_tensor, output_tensor, std::nullopt,
        kernel, strides, dilations, pads, 0,
        max_pool_reducer<float>()
    );

    array_mml<float> expected = {8};  // max of all values
    auto expected_tensor = TensorFactory::create_tensor<float>(output_shape, expected);

    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_tensor));
}

TEST(test_sliding_window, max_pool_4d) {
    // Create a 6D tensor: [N, C, D1, D2, D3, D4]
    array_mml<uli> input_shape = {1, 1, 2, 2, 2, 2};
    
    // Create 16 values (2^4) for the 4D spatial dimensions
    array_mml<float> input_data = {
        // D1=0, D2=0
        1, 2,    // D3=0, D4=0-1
        3, 4,    // D3=1, D4=0-1
        
        // D1=0, D2=1
        5, 6,    // D3=0, D4=0-1
        7, 8,    // D3=1, D4=0-1
        
        // D1=1, D2=0
        9, 10,   // D3=0, D4=0-1
        11, 12,  // D3=1, D4=0-1
        
        // D1=1, D2=1
        13, 14,  // D3=0, D4=0-1
        15, 16   // D3=1, D4=0-1
    };
    
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Output shape for a 2x2x2x2 kernel with stride 1 will be 1x1x1x1
    array_mml<uli> output_shape = {1, 1, 1, 1, 1, 1};
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);

    // Configure the kernel parameters
    std::vector<int> kernel = {2, 2, 2, 2};       // 4D kernel
    std::vector<int> strides = {1, 1, 1, 1};      // Stride of 1 in each dim
    std::vector<int> dilations = {1, 1, 1, 1};    // No dilation
    std::vector<std::pair<int, int>> pads = {{0, 0}, {0, 0}, {0, 0}, {0, 0}}; // No padding

    // Apply the sliding window operation
    TensorOperationsModule::sliding_window<float>(
        input_tensor, output_tensor, std::nullopt,
        kernel, strides, dilations, pads, 0,
        max_pool_reducer<float>()
    );

    // Maximum value is 16
    array_mml<float> expected = {16};
    auto expected_tensor = TensorFactory::create_tensor<float>(output_shape, expected);

    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_tensor));
}

TEST(test_sliding_window, simulated_max_pool_with_node_utils) {
    // Create a 4x4 input tensor
    array_mml<uli> input_shape = {1, 1, 4, 4};
    array_mml<float> input_data = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Set up pooling parameters
    std::string auto_pad = "SAME_UPPER";
    int ceil_mode = 1;
    std::vector<int> kernel_shape = {2, 2};
    std::vector<int> strides = {2, 2};
    std::vector<int> pads = {};
    std::vector<int> dilations = {1, 1};

    // Use NodeUtils to compute all the required attributes
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    
    // Get output shape and padding information
    array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    // Create output tensor
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);
    
    // Apply the max pooling operation using sliding_window
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        std::nullopt,  // No indices output for this test
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        max_pool_reducer<float>()
    );

    // For SAME_UPPER with kernel 2x2, stride 2x2, the output should be 2x2
    // Expected: top-left, top-right, bottom-left, bottom-right max values
    array_mml<float> expected_output = {6, 8, 14, 16};
    auto expected_tensor = TensorFactory::create_tensor<float>(output_shape, expected_output);

    // Verify the output matches the expected values
    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_tensor));
    
    // Print shapes for debugging
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Padding pairs: ";
    for (const auto& pair : pad_pairs) {
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << std::endl;
}

TEST(test_sliding_window, max_pool_with_asymmetric_dilations) {
    // Create a 4x4 input tensor
    array_mml<uli> input_shape = {1, 1, 4, 4};
    array_mml<float> input_data = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Use VALID padding with asymmetric dilation values
    std::string auto_pad = "VALID";
    int ceil_mode = 0;
    std::vector<int> kernel_shape = {2, 2};
    std::vector<int> strides = {1, 1};          // Stride of 1 for dense sampling
    std::vector<int> pads = {};
    std::vector<int> dilations = {2, 1};        // Dilated in height dimension only

    // Compute the attributes using NodeUtils
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    
    // The effective kernel is now 3x2 (2-1)*2+1 x (2-1)*1+1
    array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    // Create output tensor
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);
    
    // Apply the max pooling operation with the dilated kernel
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        std::nullopt,
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        max_pool_reducer<float>()
    );

    // With VALID padding, a 2x2 kernel, dilation [2,1], and stride [1,1]
    // The output shape will be [1, 1, 2, 3]
    // The effective kernel looks at:
    // Window 1: [1,2,9,10] → max: 10
    // Window 2: [2,3,10,11] → max: 11
    // Window 3: [3,4,11,12] → max: 12
    // Window 4: [5,6,13,14] → max: 14
    // Window 5: [6,7,14,15] → max: 15
    // Window 6: [7,8,15,16] → max: 16
    
    array_mml<float> expected_output = {10, 11, 12, 14, 15, 16};
    auto expected_tensor = TensorFactory::create_tensor<float>(output_shape, expected_output);

    // Verify the output
    ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_tensor));
    
    // Print debugging information
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Dilations: [" << dilations[0] << "," << dilations[1] << "]" << std::endl;
    std::cout << "Padding pairs: ";
    for (const auto& pair : pad_pairs) {
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << std::endl;
}

TEST(test_sliding_window, max_pool_5d_with_complex_attributes) {
    // Create a 5D spatial tensor (7D including batch and channel dimensions)
    array_mml<uli> input_shape = {1, 1, 3, 3, 3, 3, 3};
    
    // Create data for the 5D tensor (243 values)
    array_mml<float> input_data(243);
    for (uli i = 0; i < 243; ++i) {
        input_data[i] = static_cast<float>(i + 1);  // Values 1-243
    }
    
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Use complex pooling parameters
    std::string auto_pad = "NOTSET";
    int ceil_mode = 1;  // Use ceiling mode for output calculation
    
    // Asymmetric kernel shape
    std::vector<int> kernel_shape = {2, 2, 2, 3, 2};
    
    // Different strides for different dimensions
    std::vector<int> strides = {2, 1, 2, 1, 2};
    
    // Custom padding for each dimension (beginning, end)
    std::vector<int> pads = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
    
    // Mixed dilations
    std::vector<int> dilations = {1, 2, 1, 1, 2};

    // Compute the attributes using NodeUtils
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    
    // Calculate output shape and padding pairs
    array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    // Create output tensor
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);
    
    // Apply max pooling with the complex configuration
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        std::nullopt,
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        max_pool_reducer<float>()
    );

    // Verify the shape of the output tensor matches what we calculated
    ASSERT_EQ(output_tensor->get_shape(), output_shape);
    
    // While we can't easily validate all values, we can check some properties:
    // 1. The maximum value should not exceed the input maximum
    float max_val = 0.0f;
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        max_val = std::max(max_val, (*output_tensor)[i]);
    }
    ASSERT_LE(max_val, 243.0f);
    
    // 2. The output should not be all zeros
    bool all_zeros = true;
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        if ((*output_tensor)[i] > 0) {
            all_zeros = false;
            break;
        }
    }
    ASSERT_FALSE(all_zeros);
    
    // Print debugging information
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Kernel: [";
    for (size_t i = 0; i < kernel_shape.size(); ++i) {
        std::cout << kernel_shape[i] << (i < kernel_shape.size()-1 ? "," : "");
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Strides: [";
    for (size_t i = 0; i < strides.size(); ++i) {
        std::cout << strides[i] << (i < strides.size()-1 ? "," : "");
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Dilations: [";
    for (size_t i = 0; i < dilations.size(); ++i) {
        std::cout << dilations[i] << (i < dilations.size()-1 ? "," : "");
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Padding pairs: ";
    for (const auto& pair : pad_pairs) {
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << std::endl;
}

TEST(test_sliding_window, avg_pool_with_complex_parameters) {
    // Create a 4x4 input tensor with specific pattern for easier verification
    array_mml<uli> input_shape = {1, 1, 4, 4};
    array_mml<float> input_data = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Set up complex pooling parameters
    std::string auto_pad = "SAME_UPPER";
    int ceil_mode = 1;
    std::vector<int> kernel_shape = {3, 2};       // Non-square kernel (3x2)
    std::vector<int> strides = {2, 1};            // Different strides per dimension
    std::vector<int> pads = {};
    std::vector<int> dilations = {1, 2};          // Dilation in second dimension

    // Apply NodeUtils to compute attributes
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    
    // Calculate output shape and padding
    array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    // Create output tensor
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);
    
    // Apply average pooling using sliding window
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        std::nullopt,
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        avg_pool_reducer<float>()
    );

    // Verify output shape matches expected shape
    ASSERT_EQ(output_tensor->get_shape(), output_shape);
    
    // Check that values are within expected range
    float max_val = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float sum_val = 0.0f;
    
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        float val = (*output_tensor)[i];
        max_val = std::max(max_val, val);
        min_val = std::min(min_val, val);
        sum_val += val;
    }
    
    // Average pooling should produce values between min and max of input
    ASSERT_GE(max_val, 1.0f);
    ASSERT_LE(max_val, 16.0f);
    
    // Overall average of output should be similar to input average (8.5) if padding is minimal
    float avg_val = sum_val / static_cast<float>(output_tensor->get_size());
    ASSERT_NEAR(avg_val, 8.5f, 3.0f);  // Allow some deviation due to padding
    
    // Print diagnostic information
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Kernel: [" << kernel_shape[0] << "," << kernel_shape[1] << "]" << std::endl;
    std::cout << "Strides: [" << strides[0] << "," << strides[1] << "]" << std::endl;
    std::cout << "Dilations: [" << dilations[0] << "," << dilations[1] << "]" << std::endl;
    std::cout << "Output min/max/avg: " << min_val << "/" << max_val << "/" << avg_val << std::endl;
    std::cout << "Padding pairs: ";
    for (const auto& pair : pad_pairs) {
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << std::endl;
    
    // Print actual output tensor values for inspection
    std::cout << "Output values: ";
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        std::cout << (*output_tensor)[i] << " ";
    }
    std::cout << std::endl;
}

TEST(test_sliding_window, avg_pool_with_counting_and_edge_handling) {
    // Create a 5x5 input tensor with specific pattern for verification
    array_mml<uli> input_shape = {1, 1, 5, 5};
    array_mml<float> input_data = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Set up complex pooling parameters
    std::string auto_pad = "NOTSET";  // Use explicit padding
    int ceil_mode = 1;                // Use ceiling mode
    std::vector<int> kernel_shape = {3, 4};       // Non-square kernel (3x4)
    std::vector<int> strides = {2, 3};            // Different strides per dimension
    std::vector<int> pads = {1, 1, 2, 1};         // Asymmetric padding [top, bottom, left, right]
    std::vector<int> dilations = {2, 1};          // Mixed dilations

    // Apply NodeUtils to compute attributes
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    
    // Calculate output shape and padding
    array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    
    // Create output tensor
    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);
    
    // Apply average pooling using sliding window
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor, 
        std::nullopt,
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        [](const std::vector<float> windowValues, const std::vector<int64_t> w, int64_t outIndex) -> float {
            if (windowValues.empty()) {
                return 0.0f; // Return 0 for empty windows
            }
            
            float sum = 0.0f;
            for (const auto& val : windowValues) {
                sum += val;
            }
            
            // Here we're calculating true average based on actual elements,
            // not including the padding zeros which don't make it into windowValues
            return sum / static_cast<float>(windowValues.size());
        }
    );

    // Check that the output shape matches expected shape
    ASSERT_EQ(output_tensor->get_shape(), output_shape);
    
    // Calculate and check statistics about the output tensor
    float max_val = std::numeric_limits<float>::lowest();
    float min_val = std::numeric_limits<float>::max();
    float sum_val = 0.0f;
    
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        float val = (*output_tensor)[i];
        max_val = std::max(max_val, val);
        min_val = std::min(min_val, val);
        sum_val += val;
    }
    
    // Average pooling should produce values between min and max of input
    ASSERT_GE(max_val, 1.0f);
    ASSERT_LE(max_val, 25.0f);
    
    // The average should be somewhere in the middle range
    float avg_val = sum_val / static_cast<float>(output_tensor->get_size());
    ASSERT_NEAR(avg_val, 13.0f, 7.0f);  // Allow fairly wide deviation due to complex padding
    
    // Print extensive diagnostic information
    std::cout << "===== Complex Average Pooling Test =====" << std::endl;
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Kernel: [" << kernel_shape[0] << "," << kernel_shape[1] << "]" << std::endl;
    std::cout << "Strides: [" << strides[0] << "," << strides[1] << "]" << std::endl;
    std::cout << "Dilations: [" << dilations[0] << "," << dilations[1] << "]" << std::endl;
    std::cout << "Explicit pads: [" << pads[0] << "," << pads[1] << "," << pads[2] << "," << pads[3] << "]" << std::endl;
    std::cout << "Computed pad pairs: ";
    for (const auto& pair : pad_pairs) {
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << std::endl;
    
    std::cout << "Output statistics: " << std::endl;
    std::cout << "- Min value: " << min_val << std::endl;
    std::cout << "- Max value: " << max_val << std::endl;
    std::cout << "- Average value: " << avg_val << std::endl;
    std::cout << "- Output size: " << output_tensor->get_size() << std::endl;
    
    // Print output values in a 2D grid for better visualization
    std::cout << "Output tensor values:" << std::endl;
    uli rows = output_shape[2];
    uli cols = output_shape[3];
    for (uli r = 0; r < rows; ++r) {
        std::cout << "  ";
        for (uli c = 0; c < cols; ++c) {
            array_mml<uli> indices = {0, 0, r, c};
            std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (*output_tensor)[indices] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================================" << std::endl;
}

TEST(test_sliding_window, max_pool_high_dimensional_7d_tensor) {
    // Input tensor shape: N=1, C=1, D1=3, D2=4, D3=3, D4=2, D5=3
    array_mml<uli> input_shape = {1, 1, 3, 4, 3, 2, 3};
    uli total_size = 1;
    for (auto dim : input_shape) total_size *= dim;

    // Fill input tensor with increasing values
    array_mml<float> input_data(total_size);
    for (uli i = 0; i < total_size; ++i) {
        input_data[i] = static_cast<float>(i + 1);  // values 1..total_size
    }

    auto input_tensor = TensorFactory::create_tensor<float>(input_shape, input_data);

    // Pooling configuration
    std::string auto_pad = "NOTSET";
    int ceil_mode = 1;

    std::vector<int> kernel_shape = {2, 2, 2, 2, 2};
    std::vector<int> strides      = {1, 2, 1, 2, 1};
    std::vector<int> dilations    = {1, 1, 2, 1, 1};
    std::vector<int> pads         = {
        1, 0,   // D1 (pad top, bottom)
        0, 1,   // D2
        1, 0,   // D3
        0, 0,   // D4
        1, 1    // D5
    };

    // Compute attributes via NodeUtils
    NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);
    auto output_shape = NodeUtils::compute_pool_output_shape(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);
    auto pad_pairs = NodeUtils::compute_pool_pad_begin_end(
        input_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

    auto output_tensor = TensorFactory::create_tensor<float>(output_shape);

    // Run max pooling
    TensorOperationsModule::sliding_window<float>(
        input_tensor,
        output_tensor,
        std::nullopt,
        kernel_shape,
        strides,
        dilations,
        pad_pairs,
        0,
        max_pool_reducer<float>()
    );

    // Assertions: Check shape and basic properties
    ASSERT_EQ(output_tensor->get_shape(), output_shape);

    // Max value should not exceed input max
    float max_val = std::numeric_limits<float>::lowest();
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        max_val = std::max(max_val, (*output_tensor)[i]);
    }
    ASSERT_LE(max_val, static_cast<float>(total_size));

    // Ensure output is not all zeros
    bool nonzero_found = false;
    for (uli i = 0; i < output_tensor->get_size(); ++i) {
        if ((*output_tensor)[i] > 0) {
            nonzero_found = true;
            break;
        }
    }
    ASSERT_TRUE(nonzero_found);

    // Debug output
    std::cout << "===== High-Dimensional Max Pool Test =====" << std::endl;
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;
    std::cout << "Kernel shape: ";
    for (auto k : kernel_shape) std::cout << k << " ";
    std::cout << "\nStrides: ";
    for (auto s : strides) std::cout << s << " ";
    std::cout << "\nDilations: ";
    for (auto d : dilations) std::cout << d << " ";
    std::cout << "\nPad pairs: ";
    for (auto p : pad_pairs) std::cout << "(" << p.first << "," << p.second << ") ";
    std::cout << "\nMax value in output: " << max_val << std::endl;
    std::cout << "==========================================" << std::endl;
}