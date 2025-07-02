message(STATUS "Using AVX512 GEMM configuration")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx512f -mavx512vl -mavx512dq -mavx512bw -march=native)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE -/arch:AVX512)
endif()