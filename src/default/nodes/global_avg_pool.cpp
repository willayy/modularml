#include "nodes/global_avg_pool.hpp"

GlobalAvgPoolNode::GlobalAvgPoolNode(std::string X, std::string Y)
    : X(X), Y(Y) {}

GlobalAvgPoolNode::GlobalAvgPoolNode(const nlohmann::json& node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }
}

void GlobalAvgPoolNode::forward(
    std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error(
        "GlobalAvgPoolNode: Input tensor X not found in iomap");
  }

  const GeneralDataTypes& x_tensor = x_it->second;

  std::visit(
      [&](const auto& x_ptr) {
        using ValueType =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueType, T>) {
          throw std::runtime_error(
              "GlobalAvgPoolNode: Unsupported data type for tensor X");
        } else {
          const auto& x_shape = x_ptr->get_shape();
          size_t rank = x_shape.size();
          if (rank < 3) {
            throw std::runtime_error(
                "GlobalAvgPoolNode: Input tensor must have at least 3 "
                "dimensions (N, C, ...)");
          }

          size_t batch = x_shape[0];
          size_t channels = x_shape[1];

          // total elements in each spatial slice
          size_t spatial_size = 1;
          for (size_t i = 2; i < rank; ++i) {
            spatial_size *= x_shape[i];
          }

          // build output shape [N, C, 1, 1, ...]
          std::vector<size_t> y_shape_vec;
          y_shape_vec.reserve(rank);
          y_shape_vec.push_back(batch);
          y_shape_vec.push_back(channels);
          for (size_t i = 2; i < rank; ++i) {
            y_shape_vec.push_back(1);
          }
          array_mml<size_t> y_shape(y_shape_vec);
          auto y_ptr = std::make_shared<Tensor<ValueType>>(y_shape);

          // index buffers
          std::vector<size_t> idx(rank, 0);
          std::vector<size_t> spatial_idx(rank - 2, 0);

          for (size_t n = 0; n < batch; ++n) {
            idx[0] = n;
            for (size_t c = 0; c < channels; ++c) {
              idx[1] = c;
              ValueType sum = 0;

              // iterate through every spatial position
              bool done = false;
              while (!done) {
                // fill idx[2..] from spatial_idx
                for (size_t d = 0; d < spatial_idx.size(); ++d) {
                  idx[d + 2] = spatial_idx[d];
                }

                // *** FIX: use a named array_mml to index ***
                array_mml<size_t> in_index(idx);
                sum += (*x_ptr)[in_index];

                // increment spatial_idx
                for (int d = (int)spatial_idx.size() - 1; d >= 0; --d) {
                  if (++spatial_idx[d] < x_shape[d + 2]) {
                    break;
                  }
                  spatial_idx[d] = 0;
                  if (d == 0) done = true;
                }
              }

              // write output at [n,c,0,...,0]
              std::vector<size_t> out_idx_vec(rank, 0);
              out_idx_vec[0] = n;
              out_idx_vec[1] = c;
              array_mml<size_t> out_index(out_idx_vec);

              (*y_ptr)[out_index] = sum / static_cast<ValueType>(spatial_size);

              // reset for next channel
              std::fill(spatial_idx.begin(), spatial_idx.end(), 0);
            }
          }

          iomap[Y] = y_ptr;
        }
      },
      x_tensor);
}

std::vector<std::string> GlobalAvgPoolNode::getInputs() { return {X}; }

std::vector<std::string> GlobalAvgPoolNode::getOutputs() { return {Y}; }