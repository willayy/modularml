#pragma once
#include "globals.hpp"
#include "mml_tensor.hpp"

template <typename T>
static shared_ptr<Tensor<T>> mml_constructor_1(const array_mml<uli>& dims, const array_mml<T>& values);

template <typename T>
static shared_ptr<Tensor<T>> mml_constructor_2(const array_mml<uli>& dims);

template <typename T>
static shared_ptr<Tensor<T>> mml_constructor_3(const initializer_list<uli> dims, const initializer_list<T> values);

template <typename T>
static shared_ptr<Tensor<T>> mml_constructor_4(const initializer_list<uli> dims);

#include "../datastructures/tensor_factory_functions.tpp"