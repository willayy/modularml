message(STATUS "Using AVX GEMM configuration")
SET(MEMORY_ALIGNMENT 32)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mavx2 -march=native)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/arch:AVX2)
endif()

