#pragma once
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

template <typename T>
using Vec = std::vector<T>;
using String = std::string;
using Data = nlohmann::json;