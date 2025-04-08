#include "nodes/maxpool.hpp"

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

            array_mml<uli> output_shape = compute_output_shape(x_ptr->get_shape(), kernel_shape, strides, pads, dilations, ceil_mode);

            auto output_ptr = std::make_shared<Tensor_mml<ValueType>>(output_shape);
            auto indices_ptr = std::make_shared<Tensor_mml<int64_t>>(output_shape);

            Arithmetic_mml<ValueType>::apply_pooling(x_ptr, output_ptr, indices_ptr, kernel_shape, strides, pads, dilations, ceil_mode);
            
            iomap[Y] = output_ptr;
            iomap[indices] = indices_ptr;
        }
    }, x_tensor);
}