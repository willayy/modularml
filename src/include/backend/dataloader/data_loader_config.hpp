#pragma once

#include <string>


struct DataLoaderConfig {
    virtual ~DataLoaderConfig() = default; 
};

struct ImageLoaderConfig : public DataLoaderConfig {
    std::string image_path;
    bool include_alpha_channel;
    explicit ImageLoaderConfig(const std::string& path, bool include_alpha_channel = false) : image_path(path) {}
};