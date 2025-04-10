/* #include "nodes/maxpool.hpp"
#include "datastructures/tensor_operations_module.hpp"

void MaxPoolNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("MaxPoolNode: Input tensor X not found in iomap");
    }
    
    const GeneralDataTypes& x_tensor = x_it->second;

    std::visit([&](const auto& x_ptr) {
        using ValueType = std::decay_t<decltype(x_ptr)>::element_type::value_type;
        
        if constexpr (!is_in_variant_v<ValueType, T>) {
            throw std::runtime_error("MaxPoolNode: Unsupported data type for tensor X");
        } else {
            if (x_ptr->get_shape().size() < 3) {
                throw std::runtime_error("PoolNode: Input tensor must be at least NCL");
            }

            NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);

            array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto pad_pair = NodeUtils::compute_pool_pad_begin_end(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto output_ptr = std::make_shared<Tensor_mml<ValueType>>(output_shape);
            auto indices_ptr = std::make_shared<Tensor_mml<int64_t>>(output_shape);

            // Perform pooling operation
            TensorOperationsModule::sliding_window<ValueType>(x_ptr, output_ptr, indices_ptr, kernel_shape, strides, dilations, pad_pair, );
            
            iomap[Y] = output_ptr;
            //iomap[indices] = indices_ptr;
        }
    }, x_tensor);
} */

