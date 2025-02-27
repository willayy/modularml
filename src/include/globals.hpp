#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>

using nlohmann::json;
using std::make_unique;
using std::make_shared;
using std::move;
using std::string;
using std::unique_ptr;
using std::vector;
using std::initializer_list;
using std::shared_ptr;
using std::logic_error;
using std::out_of_range;
using std::accumulate;
using std::multiplies;