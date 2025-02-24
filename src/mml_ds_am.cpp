#include "mml_ds_am.hpp"

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::add(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
    if (a->get_shape() != b->get_shape()) {
        throw std::invalid_argument(
            "The data structures does not have the same shape");
    }

    auto shape = a->get_shape();
    MML_DataStructure<T>* result = new MML_DataStructure(shape);

    for (int i = 0; i < a->get_size(); ++i) {
        result->set(a->get(i) + b->get(i));
    }
    return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::subtract(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
    if (a->get_shape() != b->get_shape()) {
        throw std::invalid_argument(
            "The data structures does not have the same shape");
    }

    auto shape = a->get_shape();
    MML_DataStructure<T>* result = new MML_DataStructure(shape);

    for (int i = 0; i < a->get_size(); ++i) {
        result->set(a->get(i) - b->get(i));
    }
    return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::multiply(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
    // TODO: Implement multiplication for 3D
    if (a->get_shape().size() != a->get_shape().size()) {
        throw std::invalid_argument(
            "The data structures does not have the same shape");
    }
    if (a->get_shape().size() != 2) {
        throw std::invalid_argument(
            "The data structures does not have the same shape");
    }
    if (a->get_shape().at(1) != b->get_shape().at(0)) {
        throw std::invalid_argument(
            "The inner shape of the data structures are incompatible with matrix multiplication");
    }
    auto result_shape = {a->get_shape().at(0), b->get_shape().at(1)};
    MML_DataStructure<T>* result = new MML_DataStructure(result_shape);

    int m = a->get_shape().at(0); // Number of rows in matrix A
    int n = a->get_shape().at(1); // Number of columns in matrix A
    int p = b->get_shape().at(1); // Number of columns in matrix B

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                int a_value = a->get({i, k});
                int b_value = b->get({k, j});
                result->set({i, j}, result.get({i, j}) + (a_value * b_value));
            }
        }   
    }
    return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::multiply(MML_DataStructure<T>* a, const T b) const {
    if (a->get_shape().size() == 0) {
        throw std::invalid_argument("The data structure shape isn't valid");
    }

    auto shape = a->get_shape();
    MML_DataStructure<T>* result = new MML_DataStructure(shape);

    for (int i = 0; i < a->get_size(); ++i) {
        result->set(i, a->get(i) * b);
    }
    return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::divide(MML_DataStructure<T>* a, const T b) const {
    if (a->get_shape().size() == 0) {
        throw std::invalid_argument("The data structure shape isn't valid");
    }

    auto shape = a->get_shape();
    MML_DataStructure<T>* result = new MML_DataStructure(shape);

    for (int i = 0; i < a->get_size(); ++i) {
        result->set(i, a->get(i) / b);
    }
    return result;
}

template <typename T>
bool MML_DataStructure_AM<T>::equal(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
    if (a->get_shape() != b->get_shape()) {
        return false;
    }
    for (int i = 0; i < a->get_size(); ++i) {
        if (a->get(i) != b->get(i)) {
            return false;
        }
    }
    return true;
}
