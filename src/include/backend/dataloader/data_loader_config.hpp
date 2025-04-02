#pragma once

#include <string>


struct DataLoaderConfig {
    virtual ~DataLoaderConfig() = default; 
};

struct ImageLoaderConfig : public DataLoaderConfig {
    std::string image_path;
    explicit ImageLoaderConfig(const std::string& path) : image_path(path) {}
};