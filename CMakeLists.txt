cmake_minimum_required(VERSION 3.20)
project(ModularML)

set(PROJECT_NAME "modularml")

# ------------------- Compiler/Standards ------------------- #

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the C version
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

# Check if GCC is the compiler
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    message(WARNING "WARNING: This project recommends GCC or Clang as the compiler.")
endif()

# Check if system is UNIX
if(NOT (UNIX))
    message(WARNING "WARNING: This project is intended for Linux or MacOS.")
endif()

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

# Dependency directory
set(PROJECT_DEPS ${CMAKE_BINARY_DIR}/_deps)

# ------------------- Binary ------------------------------- #

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
file(GLOB SOURCES ${SRC_DIR}/*.cpp)

# Glob all header files
file(GLOB HEADERS ${INCLUDE_DIR}/*.hpp)

# Add library
add_library(
    ${PROJECT_NAME}
    STATIC
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

# Add all warning flags if build type is Debug, this could be considered pseudo linting.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")

    target_compile_options(${PROJECT_NAME} PRIVATE 
        -Wall 
        -Wextra 
        -pedantic 
        -Werror
        -Wnon-virtual-dtor
        -Waddress
        -Walloc-zero
        -Walloca
        -Wattributes
        -Warray-bounds
        -Wdeprecated
        -Wif-not-aligned
        -Wmissing-attributes
    )
    
endif()

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
target_compile_options(${PROJECT_NAME}_tests PRIVATE -g -O0 -fPIC --coverage -fprofile-abs-path)
# Linker flags
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

# ------------------- Coverage (gcovr) ---------------------- #

find_program(GCOVR_EXECUTABLE gcovr REQUIRED)

message(STATUS "gcovr found at: ${GCOVR_EXECUTABLE}")

# If on MacOS, gcovr --gcov-executable gcov-14
if(${GCOVR_EXECUTABLE} STREQUAL /opt/homebrew/bin/gcovr)
    set(GCOVR_EXECUTABLE ${GCOVR_EXECUTABLE} --gcov-executable gcov-14)
endif()

# Create coverage target
add_custom_target(coverage_report
    COMMAND ${CMAKE_BINARY_DIR}/bin/modularml_tests
    COMMAND ${GCOVR_EXECUTABLE} ${CMAKE_BINARY_DIR}/CMakeFiles/modularml_tests.dir/tests -r ${CMAKE_SOURCE_DIR}/src --html --html-details -o ${CMAKE_BINARY_DIR}/coverage_report/
)

# Add dependencies to the coverage report
add_dependencies(coverage_report ${PROJECT_NAME}_tests)

# ------------------- Doxygen ------------------------------- #

# Find Doxygen and other dependencies
find_package(Doxygen
             REQUIRED dot
             OPTIONAL_COMPONENTS mscgen dia)

# Set the output directory for Doxygen
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)

# Make doxygen extract all comments
set(DOXYGEN_EXTRACT_ALL YES)

# Exclude libraries from doxygen
set(DOXYGEN_EXCLUDE_PATTERNS ${PROJECT_DEPS}/*)

# Use convenience function to add doxygen docs
add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIRECTORY}
    COMMAND doxygen ${CMAKE_SOURCE_DIR}/Doxyfile
    COMMENT "Generating Doxygen documentation"
)   