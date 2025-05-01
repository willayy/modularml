cmake_minimum_required(VERSION 3.22)
project(ModularML)

set(PROJECT_NAME "modularml")

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "CMake generator: ${CMAKE_GENERATOR}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMake source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake binary dir: ${CMAKE_BINARY_DIR}")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND (APPLE))
    message(STATUS "Setting macOS SDK path, this fixes a bug with macosx sdk 15.4")
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk")
endif()

# ------------------- Compiler/Standards ------------------- #

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the C version
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

# Check if GCC is the compiler
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
    message(WARNING "WARNING: This project recommends GCC or Clang as the compiler.")
endif()

# Check if system is UNIX
if(NOT (UNIX))
    message(WARNING "WARNING: This project is intended for Linux or MacOS.")
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")


# ------------------- Libs --------------------------------- #
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3  # Use the latest version available
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)

# ==== Header-only library used to load images ==== #
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)


# Dependency directory
set(PROJECT_DEPS ${CMAKE_BINARY_DIR}/_deps)

# ------------------- Binary ------------------------------- #
# Use max 8 threads for building, improve build times
find_package(Threads REQUIRED)
set(CMAKE_BUILD_PARALLEL_LEVEL 8) 

# Binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Library output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# The header file directory
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/include)

# The source code directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# include the headers
include_directories(${INCLUDE_DIR})

# Glob all source files
file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cpp ${SRC_DIR}/*.tpp)

# Glob all header files
file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/*.hpp)

# Add library
add_library(
    ${PROJECT_NAME}
    STATIC
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_BINARY_DIR}/_deps/stb-src
)


# ------------------- Testing ------------------------------ #

enable_testing()

set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

# Get all test source files
file(GLOB TEST_SOURCES ${TESTS_DIR}/*.cpp)

# Add the test executable
add_executable(
    ${PROJECT_NAME}_tests
    ${TEST_SOURCES}
)

# Compiler flags
target_compile_options(${PROJECT_NAME}_tests PRIVATE 
    -funroll-loops
    -O3
    -g 
    -fPIC 
    --coverage 
    -fprofile-abs-path 
    -Wfatal-errors 
    -fmax-errors=1 
    -pedantic-errors
)

target_link_options(${PROJECT_NAME}_tests PRIVATE -g -O0 -fPIC --coverage -fprofile-abs-path)


# Link the test executable with the main project executable
target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${PROJECT_NAME})
# Link gtest
target_link_libraries(${PROJECT_NAME}_tests PRIVATE GTest::gtest_main)
# Link coverage
target_link_libraries(${PROJECT_NAME}_tests PRIVATE --coverage)

# Add the test to gtest
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests DISCOVERY_MODE POST_BUILD)

# ------------------- Executable --------------------- #

set(EXECUTABLE_SRC ${SRC_DIR}/main.cpp)

add_executable(${PROJECT_NAME}_exec ${EXECUTABLE_SRC})

target_link_libraries(${PROJECT_NAME}_exec PRIVATE ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME}_exec PRIVATE 
    -funroll-loops
    -O3
    -g 
    -fPIC 
    -fprofile-abs-path 
    -Wfatal-errors 
    -fmax-errors=1 
    -pedantic-errors
)

# ------------------- GEMM & Optimizations ----------------- #
option(ALIGN_TENSORS "Enable alignment for tensor data for better performance" ON)

option(USE_DEFAULT_GEMM "Use default GEMM implementation" ON)
option(USE_BLOCKED_GEMM "Use blocked GEMM implementation" OFF)
option(USE_AVX_GEMM "Use AVX GEMM implementation" OFF)
option(USE_AVX512_GEMM "Use AVX512 GEMM implementation" OFF)
option(USE_OPENBLAS_GEMM "Use OPENBLAS GEMM implementation" OFF)

if (USE_BLOCKED_GEMM)
    message(STATUS "Using blocked GEMM implementation")
    SET(MEMORY_ALIGNMENT 16)
    add_definitions(-DUSE_BLOCKED_GEMM)

elseif (USE_AVX_GEMM)
    message(STATUS "Using AVX GEMM implementation")
    SET(MEMORY_ALIGNMENT 32)
    add_definitions(-DUSE_AVX_GEMM)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -mfma)
        target_compile_options(${PROJECT_NAME}_tests PRIVATE -mavx2 -mfma)
        target_compile_options(${PROJECT_NAME}_exec PRIVATE -mavx2 -mfma)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT_NAME} PRIVATE -/arch:AVX2)
        target_compile_options(${PROJECT_NAME}_tests PRIVATE -/arch:AVX2)
        target_compile_options(${PROJECT_NAME}_exec PRIVATE -/arch:AVX2)
    endif()

elseif (USE_AVX512_GEMM)
    message(STATUS "Using AVX512 GEMM implementation")
    SET(MEMORY_ALIGNMENT 64)
    add_definitions(-DUSE_AVX512_GEMM)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${PROJECT_NAME} PRIVATE -mavx512f -mavx512vl -mavx512dq -mavx512bw)
        target_compile_options(${PROJECT_NAME}_tests PRIVATE -mavx512f -mavx512vl -mavx512dq -mavx512bw)
        target_compile_options(${PROJECT_NAME}_exec PRIVATE -mavx512f -mavx512vl -mavx512dq -mavx512bw)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/arch:AVX2)
    endif()

elseif (USE_OPENBLAS_GEMM)
    message(STATUS "Using OpenBLAS GEMM implementation")
    SET(MEMORY_ALIGNMENT 64)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-march=native)
    endif()
    add_definitions(-DUSE_OPENBLAS_GEMM)
    find_library(OPENBLAS_LIB openblas REQUIRED)
    if(OPENBLAS_LIB)
        message(STATUS "Found OpenBLAS at ${OPENBLAS_LIB}")
        target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENBLAS_LIB})
    else()
        message(FATAL_ERROR "OpenBLAS not found!")
    endif()

else()
    message(STATUS "Using default GEMM implementation")
    SET(MEMORY_ALIGNMENT 16)
    add_definitions(-DUSE_DEFAULT_GEMM)
    endif()


# ------------------- Coverage (gcovr) ---------------------- #

find_program(GCOVR_EXECUTABLE gcovr)
if(NOT GCOVR_EXECUTABLE)
    message(WARNING "gcovr not found. Skipping coverage report.")
else()
    message(STATUS "gcovr found at ${GCOVR_EXECUTABLE}")
endif()

# If on MacOS, gcovr --gcov-executable gcov-14
if(${GCOVR_EXECUTABLE} STREQUAL /opt/homebrew/bin/gcovr AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(GCOVR_EXECUTABLE ${GCOVR_EXECUTABLE} --gcov-executable gcov-14)
endif()

# Create coverage target
add_custom_target(coverage_report
    COMMAND ${CMAKE_BINARY_DIR}/bin/modularml_tests
    COMMAND ${GCOVR_EXECUTABLE} ${CMAKE_BINARY_DIR}/CMakeFiles/modularml_tests.dir/tests -r ${CMAKE_SOURCE_DIR}/src --html --html-details --csv -o ${CMAKE_BINARY_DIR}/coverage_report/
)

# Add dependencies to the coverage report
add_dependencies(coverage_report ${PROJECT_NAME}_tests)

# ------------------- Doxygen ------------------------------- #

# Find Doxygen and other dependencies
find_package(Doxygen
             REQUIRED dot
             OPTIONAL_COMPONENTS mscgen dia)

if(DOXYGEN_FOUND)
    message(STATUS "Doxygen found at ${DOXYGEN_EXECUTABLE}")
else()
    message(WARNING "Doxygen not found. Skipping documentation generation.")
endif()

# Set the input directory for Doxygen
set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)

# Set the Doxygen input directory
set(DOXYGEN_INPUT ${DOXYGEN_INPUT_DIR})

# Define output directory for Doxygen
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)

# Make doxygen extract all comments
set(DOXYGEN_EXTRACT_ALL YES)

# Exclude libraries from doxygen
set(DOXYGEN_EXCLUDE_PATTERNS ${PROJECT_DEPS}/*)

# Create a custom target to generate docs
add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIRECTORY}
    COMMAND doxygen ${CMAKE_SOURCE_DIR}/Doxyfile
    COMMENT "Generating Doxygen documentation"
)

# ------------------- Include-what-you-use --------------------- #
# Check if include-what-you-use is installed
find_program(iwyu include-what-you-use)

# Change this if you want to enable include-what-you-use
option(USE_IWYU "Use include-what-you-use" OFF)

if(iwyu)
    message(STATUS "include-what-you-use found at ${iwyu}")
    if (USE_IWYU)
        if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
            message(WARNING "include-what-you-use is only supported with Clang or AppleClang")
        endif()
        message(STATUS "Using include-what-you-use")
        set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_INCLUDE_WHAT_YOU_USE "${iwyu}")
    else()
        message(STATUS "Not using include-what-you-use (set to OFF)")
    endif()
else()
    message(WARNING "include-what-you-use not found. Skipping IWYU.")
endif()


