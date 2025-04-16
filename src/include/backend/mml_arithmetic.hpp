#pragma once

#include <memory>
#include <functional>

#include "datastructures/tensor.hpp"

namespace Arithmetic {
  template <typename T>
  void add(const std::shared_ptr<Tensor<T>> a,
           const std::shared_ptr<Tensor<T>> b,
           std::shared_ptr<Tensor<T>> c);
  
  template <typename T>
  void subtract(const std::shared_ptr<Tensor<T>> a,
                const std::shared_ptr<Tensor<T>> b,
                std::shared_ptr<Tensor<T>> c);
  
  template <typename T>
  void multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                std::shared_ptr<Tensor<T>> c);
  
  template <typename T>
  bool equals(const std::shared_ptr<Tensor<T>> a,
              const std::shared_ptr<Tensor<T>> b);
  
  template <typename T>
  void elementwise(const std::shared_ptr<const Tensor<T>> a,
                   std::function<T(T)> f,
                   const std::shared_ptr<Tensor<T>> c);
  
  template <typename T>
  int arg_max(const std::shared_ptr<const Tensor<T>> a);
  
  template <typename T>
  void elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                            std::function<T(T)> f);

  void sliding_window(
    const array_mml<size_t>& in_shape,
    const array_mml<size_t>& out_shape,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    const std::function<void(const std::vector<std::vector<size_t>>&, const std::vector<size_t>&)> &window_f
  );

};

#define _ARITHMETIC(DT) \
template void Arithmetic::add<DT>(const std::shared_ptr<Tensor<DT>>, const std::shared_ptr<Tensor<DT>>, std::shared_ptr<Tensor<DT>>); \
template void Arithmetic::subtract<DT>(const std::shared_ptr<Tensor<DT>>, const std::shared_ptr<Tensor<DT>>, std::shared_ptr<Tensor<DT>>); \
template void Arithmetic::multiply<DT>(const std::shared_ptr<Tensor<DT>>, DT, std::shared_ptr<Tensor<DT>>);\
template bool Arithmetic::equals<DT>(const std::shared_ptr<Tensor<DT>>, const std::shared_ptr<Tensor<DT>>); \
template int Arithmetic::arg_max<DT>(const std::shared_ptr<const Tensor<DT>>); \
template void Arithmetic::elementwise<DT>(const std::shared_ptr<const Tensor<DT>>, std::function<DT(DT)>, const std::shared_ptr<Tensor<DT>>); \
template void Arithmetic::elementwise_in_place<DT>(const std::shared_ptr<Tensor<DT>>, std::function<DT(DT)>);
