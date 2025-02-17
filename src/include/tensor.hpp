#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <string>
#include <vector>

/// @brief Abstract base class for a tensor object.
/// @tparam T the type of the data contained in the tensor. E.g. int, float,
/// double etc.
template <typename T> class Tensor {
public:
  /// @brief Construct a new Tensor object.
  /// @note This is only intended as a base constructor for derived classes. It
  /// is meant to be extended.
  /// @param shape a vector of integers representing the shape of the tensor.
  explicit Tensor(std::vector<int> &shape) {
    // implemented here because its a template class
    this.shape = shape;
  }

  /// @brief Get the shape of the tensor.
  /// @return a vector of integers representing the shape.
  const std::vector<int> &get_shape() {
    // implemented here because its a template class
    return this.shape;
  }

  /// @brief Get the shape as a string. E.g. [2, 3, 4].
  /// @return a string representation of the shape.
  std::string get_shape_str() {
    std::string shape_str = "[";
    for (int i = 0; i < this.shape.size(); i++) {
      // implemented here because its a template class
      shape_str += std::to_string(this.shape[i]);
      if (i < this.shape.size() - 1) {
        shape_str += ", ";
      }
    }
    shape_str += "]";
    return shape_str;
  }

  /// @brief Get value at some index.
  /// @param indices a tensor index, e.g. [0, 1, 2]
  /// @return a value of type T
  virtual T get(std::vector<int> indices) = 0;

  /// @brief Set value at some index.
  /// @param indices a tensor index, e.g. [0, 1, 2]
  /// @param value a value to set of type T, e.g. 3.14 for float or 42 for int
  virtual void set(std::vector<int> indices, T value) = 0;

  /// @brief Abstract destructor for Tensor class.
  virtual ~Tensor();

  // Operator overloads are required to be overridden in derived classes.
  virtual Tensor<T> operator+(const Tensor<T> &other); // NOSONAR

  virtual Tensor<T> operator-(const Tensor<T> &other); // NOSONAR

  virtual Tensor<T> operator*(const Tensor<T> &other); // NOSONAR

  virtual bool operator==(const Tensor<T> &other); // NOSONAR

private:
  /// @brief Underlying shape data structure for the tensor.
  std::vector<int> shape;
};

#endif // TENSOR_HPP
