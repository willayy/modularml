#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Aliases for common types
using Data = nlohmann::json;

using std::make_unique;
using std::move;
using std::unique_ptr;
using std::vector;
using std::string;