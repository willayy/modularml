#pragma once

#include <immintrin.h>
#include <type_traits>
#include <stdexcept>

/**
 * @brief Creates a bitmask used for loading leftover elements during avx2 gemm
 */
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
        throw std::runtime_error("Type isn't valid for creating an avx2 mask");
    }
}

/**
 * @brief Creates a bitmask used for loading leftover elements during avx512 gemm
 */
template <typename T>
auto make_avx512_mask(int elem_left) {
    if constexpr (std::is_same<T, float>::value || std::is_same<T, int>::value) {
        // 16 elements for __m512 / __m512i
        return static_cast<__mmask16>((1 << elem_left) - 1);
    } else if constexpr (std::is_same<T, double>::value) {
        // 8 elements for __m512d
        return static_cast<__mmask8>((1 << elem_left) - 1);
    } else {
        throw std::runtime_error("Unsupported type for AVX-512 mask.");
    }
}