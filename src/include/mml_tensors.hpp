#pragma once

#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_vector_ds.hpp"
#include "tensor.hpp"

Tensor<float> tensor_mll(vec<int> shape);

Tensor<float> tensor_mll(const vec<int> shape, const vec<float> data);