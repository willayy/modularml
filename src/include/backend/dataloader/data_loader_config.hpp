#pragma once

#include <string>

/**
 * @brief Base configuration for the DataLoader.
 *
 * This struct serves as a base class for all DataLoader configurations.
 * It provides a polymorphic interface, allowing derived classes to 
 * define specific configurations.
 *
 * @note This class is meant to be inherited; it does not contain any 
 *       member variables or functionality on its own.
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */
struct DataLoaderConfig {
    /**
     * @brief Virtual destructor for safe polymorphic destruction.
     */
    virtual ~DataLoaderConfig() = default; 
};


/**
 * @brief Configuration for loading images in the DataLoader.
 *
 * This struct extends DataLoaderConfig to provide image-specific 
 * configuration options such as the image path and whether to 
 * include the alpha channel.
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */
struct ImageLoaderConfig : public DataLoaderConfig {
    
    /**
     * @brief Path to the image file to be loaded.
     */
    std::string image_path;

    /**
     * @brief Flag indicating whether to include the alpha channel when loading RGBA images.
     *
     * If set to `true`, the image loader will attempt to load 
     * images with an alpha channel (e.g., RGBA instead of RGB).
     */
    bool include_alpha_channel;

    /**
     * @brief Constructs an ImageLoaderConfig with the specified image path.
     * 
     * @param path The path to the image file.
     * @param include_alpha_channel Whether to load the image with an alpha channel (default: `false`).
     */
    explicit ImageLoaderConfig(const std::string& path, bool include_alpha_channel = false) : image_path(path), include_alpha_channel(include_alpha_channel) {}
};