#pragma once

#include <string>


/**
 * @class DataLoader
 * @brief Abstract class for loading external files.
 */
class DataLoader
{  
  public:
    /**
     * @brief Loads a file and writes the output to the destination path.
     * 
     * @param onnx_path Path to the file that is to be read.abs
     * @param destination_path Path to the where the result is written.
     */
    virtual void load(std::string& onnx_path, std::string& destination_path) const = 0;
    
    /// @brief Virtual destructor for cleanup.
    virtual ~DataLoader() = default;
};