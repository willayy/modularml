#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <string>
#include <vector>

/*!
    @brief Abstract base class for an object representing a Tensor.
    @details A tensor is a multi-dimensional array of data.
    This class is a base class for all Tensor implementations within the
    ModularML library. ModularML support is intened to be used with arithmetic
    operators:
    - Addition (+), Tensor-Tensor only
    - Subtraction (-), Tensor-Tensor only
    - Multiplication (*), Tensor-Tensor and Tensor-Scalar
    - Division (/), Tensor-Scalar only
    @tparam T the type of the data contained in the tensor. E.g. int, float,
    double etc.
*/
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

  /// @brief Abstract destructor for Tensor class.
  virtual ~Tensor();

  // Operator overloads are required to be overridden in derived classes.
  virtual Tensor<T> operator+(const Tensor<T> &other); // NOSONAR 

  virtual Tensor<T> operator-(const Tensor<T> &other); // NOSONAR

  virtual Tensor<T> operator*(const Tensor<T> &other); // NOSONAR

  virtual Tensor<T> operator/(const T &other); // NOSONAR

  virtual Tensor<T> operator*(const T &other); // NOSONAR

  virtual bool operator==(const Tensor<T> &other); // NOSONAR

  virtual T &operator[](std::vector<int> indices); // NOSONAR

  virtual const T &operator[](std::vector<int> indices) const; // NOSONAR

private:
  /// @brief Underlying shape data structure for the tensor.
  std::vector<int> shape;
};

#endif // TENSOR_HPP
