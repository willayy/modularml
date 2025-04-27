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
#include <vector>  // IWYU pragma: keep

#include "dataloader/data_loader_config.hpp"
#include "datastructures/a_tensor.hpp"

/**
 * @class DataLoader
 * @brief Abstract base class for data loading components.
 *
 * This template class defines the interface for loading external data and
 * translating it into tensor format for use in machine learning models. Data
 * loaders are responsible for reading data from various sources (files,
 * databases, etc.) and converting them into the appropriate tensor format for
 * processing.
 *
 * @tparam T The data type of the tensor elements (e.g., float, int, double)
 * @author Tim Carlsson (timca@chalmers.se)
 */
template <typename T>
class DataLoader {
 public:
  /**
   * @brief Default constructor for DataLoader.
   */
  DataLoader() = default;

  /**
   * @brief Loads external data and translates it to a tensor.
   *
   * This pure virtual function must be implemented by derived classes to define
   * how specific types of data are loaded and converted to tensors. The
   * configuration parameter allows specifying data source location,
   * preprocessing options, and other loading parameters.
   *
   * @param config Configuration parameters for the data loading process
   * @return A shared pointer to a Tensor containing the loaded data
   */
  virtual std::shared_ptr<Tensor<T>> load(
      const DataLoaderConfig &config) const = 0;

  /**
   * @brief Virtual destructor for the DataLoader class.
   *
   * Ensures proper cleanup of derived class objects when deleted through
   * a pointer to the base class.
   */
  virtual ~DataLoader() = default;
};