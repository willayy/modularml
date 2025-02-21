#include "a_data_loader.hpp"

#include "onnx-data_pb.h"

class OnnxLoader : public DataLoader
{
  public:
    void load(const std::string& onnx_path, const std::string& destination_path) const override;
};