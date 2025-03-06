#pragma once

// Type constraints: no bfloat16 or float16 for now (not native to c++ 17).
using DataTypes = variant<
    Tensor_mml<float>,
    Tensor_mml<double>,
    Tensor_mml<int32_t>,
    Tensor_mml<int64_t>,
    Tensor_mml<uint32_t>,
    Tensor_mml<uint64_t>
>;