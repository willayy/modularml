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

/**
 * @namespace Base64
 * @brief Provides utilities for Base64 encoding and decoding operations.
 *
 * This namespace contains functions for decoding Base64-encoded data into typed
 * arrays, which is particularly useful when working with binary data in
 * text-based formats like JSON. Base64 encoding is a common way to represent
 * binary data in formats that only support text.
 */
namespace Base64 {

/**
 * @brief Standard Base64 character set for encoding/decoding
 *
 * This string contains the 64 characters used in the standard Base64 encoding
 * scheme, plus the '=' character used for padding.
 */
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * @brief Decodes a Base64-encoded string into an array of elements of type T.
 *
 * This template function converts a Base64-encoded string into an array of
 * binary data, interpreting the resulting bytes as elements of type T. The
 * function first decodes the Base64 string into raw bytes, then reinterprets
 * those bytes as elements of type T.
 *
 * @tparam T The target element type (e.g., float, int, double)
 * @param input The Base64-encoded string to decode
 * @return An array_mml<T> containing the decoded elements
 * @throws std::runtime_error If the input contains invalid Base64 characters
 * @throws std::runtime_error If the decoded data size is not a multiple of
 * sizeof(T)
 *
 * @note The function expects the input to be properly encoded and padded if
 * necessary.
 * @note The function assumes little-endian byte order for multi-byte types.
 */
template <typename T>
inline array_mml<T> decode(const std::string &input) {
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
  auto data_ptr = std::make_shared<T[]>(element_count);
  std::memcpy(data_ptr.get(), bytes.data(), bytes.size());
  return array_mml<T>(data_ptr, element_count);
}

}  // namespace Base64
