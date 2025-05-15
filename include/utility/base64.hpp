#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "datastructures/mml_array.hpp"

namespace Base64 {
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * Decodes a base64 std::string to a std::vector of elements of type T
 *
 * @tparam T The target element type
 * @param input The base64-encoded std::string
 * @return array_mml<T> array containing the decoded elements
 * @throws std::runtime_error If input contains invalid characters or size
 * doesn't align with T
 */
template <typename T>
inline array_mml<T> decode(const std::string& input) {
  std::vector<unsigned char> bytes;
  int val = 0;
  int val_bits = -8;

  for (unsigned char c : input) {
    if (c == '=') break;  // Padding character
    std::size_t pos = base64_chars.find(c);
    if (pos == std::string::npos)
      throw std::runtime_error("Invalid base64 character");
    val = (val << 6) + static_cast<int>(pos);
    val_bits += 6;
    if (val_bits >= 0) {
      bytes.push_back((val >> val_bits) & 0xFF);
      val_bits -= 8;
    }
  }

  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error(std::format(
        "Decoded data size ({} bytes) is not aligned with sizeof({}) = {}",
        bytes.size(), typeid(T).name(), sizeof(T)));
  }

  size_t element_count = bytes.size() / sizeof(T);

  T* typed_ptr = nullptr;

#ifdef ALIGN_TENSORS
  if (posix_memalign(reinterpret_cast<void**>(&typed_ptr), MEMORY_ALIGNMENT,
                     element_count * sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
  std::memcpy(typed_ptr, bytes.data(), bytes.size());

#else
  typed_ptr = new T[element_count];
  std::memcpy(typed_ptr, bytes.data(), bytes.size());
#endif

  std::shared_ptr<T[]> data_ptr(typed_ptr, [](T* ptr) {
    if (ptr) {
#ifdef ALIGN_TENSORS
      free(ptr);
#else
      delete[] ptr;
#endif
    }
  });
  return array_mml<T>(data_ptr, element_count);
}
}  // namespace Base64
