#pragma once

#include <string>

class DataLoader
{  
  public:
    virtual void load(std::string& onnx_path, std::string& destination_path) const = 0;
    virtual ~DataLoader() = default;
};