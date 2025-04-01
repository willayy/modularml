
template <typename T> class ArrayFactory {
public:
  /**
   * @brief Get the instance of the ArrayFactory.
   * @return The instance of the ArrayFactory.
   */
  static ArrayFactory &get_instance() {
    static ArrayFactory instance;
    return instance;
  }

  // Delete copy constructor and assignment operator.
  ArrayFactory(const ArrayFactory &) = delete;
  ArrayFactory &operator=(const ArrayFactory &) = delete;

  /**
   * @brief Creates a random array with integral values.
   * @param shape The shape of the tensor to create.
   * @param lo_sz The lower bound of the array size.
   * @param hi_sz The upper bound of the array size.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  array_mml<T> random_array_mml_integral(uli lo_sz = 1, uli hi_sz = 5,
                                         T lo_v = 1, T hi_v = 10) const;

  /**
   * @brief Creates a random array with real values.
   * @param shape The shape of the tensor to create.
   * @param lo_sz The lower bound of the array size.
   * @param hi_sz The upper bound of the array size.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  array_mml<T> random_array_mml_real(uli lo_sz = 1, uli hi_sz = 5, T lo_v = 1,
                                     T hi_v = 100) const;
}

#include "../datastructure/array_factory.tpp"