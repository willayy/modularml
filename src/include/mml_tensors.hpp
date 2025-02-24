#pragma once

#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_vector.hpp"
#include "tensor.hpp"

Tensor<float> tensor_mll(Vec<int> shape);

Tensor<float> tensor_mll(const Vec<int> shape, const Vec<float> data);