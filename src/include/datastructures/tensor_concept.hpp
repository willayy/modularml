#pragma once
#include <concepts>
#include <memory>
#include <type_traits>

namespace TensorConcept {
template <typename T>
concept Types = std::is_arithmetic_v<T>;
}
