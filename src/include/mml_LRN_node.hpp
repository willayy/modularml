#pragma once
/**
 * @class LRNNode_mml
 * @brief Performs Local Response Normalizarion
 * @details LRNNode_mml performs Local Response Normalization according to the
 * ONNX specifications. It normalizes the tensor across local input regions. The
 * local region is defined across the channels.
 * @tparam The datatype i the tensor. Accepts float and double.
 */
template <typename T> class LRNNode_mml : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "LRNNode_mml supports only float, double ");

public:
  /**
   * @brief Constructor for LRNNode_mml
   * @param input A shared pointer to the input tensor.
   * @param size (Required) The number of channels to sum over
   * @param alpha (default = 0.0001) Scaling parameter
   * @param beta (default = 0.75) The exponent
   * @param bias (default = 1.0) Bias to avoid division with 0.
   *
   */
  LRNNode_mml(shared_ptr<Tensor<T>> input, int size, float alpha = 0.0001f,
              float beta = 0.75f, float bias = 1.0f);

  void forward() override;

  bool areInputsFilled() const override;

  void setInputs(const array_mml<GeneralDataTypes> &inputs) override;

  bool areOutputsFilled() const override;

  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  ///@brief Shared pointer to the input tensor
  shared_ptr<Tensor<T>> input;

  ///@brief Shared pointer to the output tensor
  shared_ptr<Tensor<T>> output;

  ///@brief Scaling parameter
  float alpha;

  ///@brief The exponent
  float beta;

  ///@brief To avoid division by zero
  float bias;

  ///@brief Number of channels to sum over
  int size;
};
#include "../mml_LRN_node.tpp"