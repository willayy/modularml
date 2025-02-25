#pragma once
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Aliases for common types
template <typename T>
using Vec = std::vector<T>;
using String = std::string;
using Data = nlohmann::json;

template <typename T>
using UPtr = std::unique_ptr<T>;
template <typename T>
using SPtr = std::shared_ptr<T>;
template <typename T>
using WPtr = std::weak_ptr<T>;

template <typename T, typename... Args>
UPtr<T> make_UPtr(Args&&... args) {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T>
decltype(auto) move_Ptr(T&& obj) {
  return std::move(obj);
}