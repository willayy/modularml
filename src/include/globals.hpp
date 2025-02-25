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

// utility functions
namespace util {
    /// @brief Makes a unique pointer from a raw pointer.
    /// @tparam T The type of the object to be wrapped.
    /// @param ptr The raw pointer to be wrapped.
    /// @return A unique pointer to the object.
    template <typename T>
    UPtr<T> make_UPr() {
        return std::make_unique<T>();
    }

    /// @brief Moves a unique pointer
    /// @tparam T The type of the object to be moved.
    /// @param ptr The unique pointer to be moved.
    /// @return A unique pointer to the object.
    template <typename T>
    UPtr<T> move_UPr(UPtr<T>& ptr) {
        return std::move(ptr);
    }
}