#pragma once

#include <immintrin.h>
#include <type_traits>
#include <stdexcept>

template <typename T>
__m256i make_avx2_mask(int elem_left) {
    if constexpr (std::is_same<T, float>::value || std::is_same<T, int>::value) {
        return _mm256_set_epi32(
             0, 
            (7 > elem_left) ? 0 : -1, 
            (6 > elem_left) ? 0 : -1,
            (5 > elem_left) ? 0 : -1, 
            (4 > elem_left) ? 0 : -1,
            (3 > elem_left) ? 0 : -1, 
            (2 > elem_left) ? 0 : -1,
            (1 > elem_left) ? 0 : -1
         );
    } else if constexpr (std::is_same<T, double>::value) {
        return _mm256_set_epi64x(
             0, 
            (3 > elem_left) ? 0 : -1LL, 
            (2 > elem_left) ? 0 : -1LL, 
            (1 > elem_left) ? 0 : -1LL
         );
    } else {
        throw std::runtime_error("Type isn't valid for creating mask");
    }
}