#pragma once

#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_vector.hpp"
#include "tensor.hpp"

Tensor<float> tensor_mll(vector<int> shape);

Tensor<float> tensor_mll(const vector<int> shape, const vector<float> data);