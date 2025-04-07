#pragma once

#include "datastructures/mml_array.hpp"
#include "../utility/uli.hpp"
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
#include <vector>

namespace Base64 {
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * Decodes a base64 std::string to a std::vector of elements of type T
 *
 * @tparam T The target element type
 * @param input The base64-encoded std::string
 * @return array_mml<T> A array containing the decoded elements
 * @throws std::runtime_error If input contains invalid characters or size
 * doesn't align with T
 */
template <typename T> array_mml<T> decode(const std::string &input) {
  std::vector<unsigned char> bytes;
  int val = 0, val_bits = -8;

  for (unsigned char c : input) {
    if (c == '=')
      break; // Padding character
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
    throw std::runtime_error(
        "Decoded data size (" + std::to_string(bytes.size()) +
        " bytes) is not aligned with sizeof(" + typeid(T).name() +
        ") = " + std::to_string(sizeof(T)));
  }

  size_t element_count = bytes.size() / sizeof(T);
  auto data_ptr = std::make_shared<T[]>(element_count);
  std::memcpy(data_ptr.get(), bytes.data(), bytes.size());
  return array_mml<T>(data_ptr, element_count);
}
} // namespace Base64
