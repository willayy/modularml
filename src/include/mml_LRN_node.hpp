#pragma once

template <typename T> class LRNNode_mml : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "LRNNode_mml supports only float, double ");

public:
  LRNNode_mml(int size, float alpha = 0.0001f, float beta = 0.75f,
              float bias = 1.0f);

  void forward() override;

  bool areInputsFilled() const override;

  void setInputs(const array_mml<GeneralDataTypes> &inputs) override;

  bool areOutputsFilled() const override;

  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  shared_ptr<Tensor<T>> input;
  shared_ptr<Tensor<T>> output;

  float alpha;
  float beta;
  float bias;
  int size;
};
#include "../mml_LRN_node.tpp"