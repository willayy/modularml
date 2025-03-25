#pragma once

#include "a_tensor.hpp"
#include "globals.hpp"


template <typename T>
class TensorFactory {

    /**
    * @brief Get the instance of the TensorFactory.
    * @return The instance of the TensorFactory. */
    static TensorFactory& getInstance() {
        static TensorFactory instance;
        return instance;
    }

    // Delete copy constructor and assignment operator.
    TensorFactory(const TensorFactory&) = delete;
    TensorFactory& operator=(const TensorFactory&) = delete;

    /**
     * @brief Creates a tensor with the specified shape and data.
     * @param shape The shape of the tensor to create.
     * @param data The data to fill the tensor with.
     * @return A tensor with the specified shape and data. */
    static shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape, const array_mml<T> &data) const;

    /**
     * @brief Creates a tensor with the specified shape.
     * @param shape The shape of the tensor to create.
     * @return A tensor with the specified shape. */
    static shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape) const;

    /**
     * @brief Creates a tensor with the specified shape and data.
     * @param shape The shape of the tensor to create.
     * @param data The data to fill the tensor with.
     * @return A tensor with the specified shape and data. */
    static shared_ptr<Tensor<T>> create_tensor(initializer_list<uli> &shape, initializer_list<T> &data) const;

    /**
     * @brief Creates a tensor with the specified shape.
     * @param shape The shape of the tensor to create.
     * @return A tensor with the specified shape. */
    static shared_ptr<Tensor<T>> create_tensor(initializer_list<uli> &shape) const;

    /**
     * @brief Creates a tensor with the specified shape and data.
     * @param shape The shape of the tensor to create.
     * @param lo_v The lower bound of the random values.
     * @param hi_v The upper bound of the random values.
     * @return A tensor with the specified shape and data. */
    static shared_ptr<Tensor<T>> random_tensor(const array_mml<uli> &shape, T lo_v = T(0), T hi_v = T(1)) const;
    
    /**
     * @brief Creates a random array with integral values.
     * @param shape The shape of the tensor to create.
     * @param lo_sz The lower bound of the array size.
     * @param hi_sz The upper bound of the array size.
     * @param lo_v The lower bound of the random values.
     * @param hi_v The upper bound of the random values.
     * @return A tensor with the specified shape and data. */
    static array_mml<T> random_array_mml_integral(uli lo_sz = 1, uli hi_sz = 5, T lo_v = 1, T hi_v = 10) const;

    /**
     * @brief Creates a random array with real values.
     * @param shape The shape of the tensor to create.
     * @param lo_sz The lower bound of the array size.
     * @param hi_sz The upper bound of the array size.
     * @param lo_v The lower bound of the random values.
     * @param hi_v The upper bound of the random values.
     * @return A tensor with the specified shape and data. */
    static array_mml<T> random_array_mml_real(uli lo_sz = 1, uli hi_sz = 5, T lo_v = 1, T hi_v = 100) const;

    static void set_tensor_constructor(string id, (void*) constructor) {
        if (id == "tensor_constructor_1") {
            static_assert(
                std::is_same_v<decltype(constructor),
                void (*)(const array_mml<uli> &, const array_mml<T> &) const>,
                "Function signature does not match tensor_constructor_1."
            );
            tensor_constructor_1 = constructor;
        } else if (id == "tensor_constructor_2") {
            static_assert(
                std::is_same_v<decltype(constructor),
                void (*)(const array_mml<uli> &) const>,
                "Function signature does not match tensor_constructor_2."
            );
            tensor_constructor_2 = constructor;
        } else if (id == "tensor_constructor_3") {
            static_assert(
                std::is_same_v<decltype(constructor),
                void (*)(initializer_list<uli> &, initializer_list<T> &) const>,
                "Function signature does not match tensor_constructor_3."
            );
            tensor_constructor_3 = constructor;
        } else if (id == "tensor_constructor_4") {
            static_assert(
                std::is_same_v<decltype(constructor),
                void (*)(initializer_list<uli> &) const>,
                "Function signature does not match tensor_constructor_4."
            );
            tensor_constructor_4 = constructor;
        } else {
            throw invalid_argument("Invalid constructor id.");
        }
    }

    private:
    TensorFactory() {}
    void (tensor_constructor_1*)(const array_mml<uli> &shape, const array_mml<T> &data);
    void (tensor_constructor_2*)(const array_mml<uli> &shape);
    void (tensor_constructor_3*)(initializer_list<uli> &shape, initializer_list<T> &data);
    void (tensor_constructor_4*)(initializer_list<uli> &shape);
};