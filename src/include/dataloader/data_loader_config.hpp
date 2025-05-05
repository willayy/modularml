#pragma once

#include <string>
#include <vector>  // IWYU pragma: keep

/**
 * @class DataLoaderConfig
 * @brief Base configuration for all data loaders in the framework.
 *
 * This abstract base class provides a common interface for all data loader
 * configurations. It establishes a polymorphic framework that allows specific
 * data loaders to define their own configuration parameters through derived
 * classes.
 *
 * @note This class is designed as a base class and contains no member variables
 *       or functionality of its own. Concrete implementations should inherit
 * from this class and add their specific configuration parameters.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
struct DataLoaderConfig {
  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   *
   * This allows safe polymorphic destruction of derived configuration objects
   * when accessed through a pointer to this base class.
   */
  virtual ~DataLoaderConfig() = default;
};

/**
 * @class ImageLoaderConfig
 * @brief Configuration for loading image data in the framework.
 *
 * This class extends the DataLoaderConfig base class to provide specific
 * configuration parameters for loading image data. It includes options to
 * specify the image file path and control how color channels are processed
 * during loading.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
struct ImageLoaderConfig : public DataLoaderConfig {
  /**
   * @brief Path to the image file to be loaded.
   *
   * This should be a valid path to an image file in a format supported by the
   * image loader implementation (typically JPEG, PNG, BMP, etc.).
   */
  std::string image_path;

  /**
   * @brief Flag controlling whether the alpha channel is included when loading
   * images.
   *
   * When set to `true`, the image loader will preserve the alpha channel for
   * images that support transparency (e.g., loading as RGBA instead of RGB).
   * For images without an alpha channel, this parameter has no effect.
   */
  bool include_alpha_channel;

  /**
   * @brief Constructs an ImageLoaderConfig with the specified parameters.
   *
   * @param path The file path to the image to be loaded
   * @param include_alpha_channel Whether to preserve the alpha channel if
   * present (default: `false`, which discards alpha)
   */
  explicit ImageLoaderConfig(const std::string &path,
                             bool include_alpha_channel = false)
      : image_path(path), include_alpha_channel(include_alpha_channel) {}
};