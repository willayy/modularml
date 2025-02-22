cmake_minimum_required(VERSION 3.10)
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

# ------------------- Libs --------------------------------- #
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3  # Use the latest version available
)

FetchContent_MakeAvailable(nlohmann_json)

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

# Add executable
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

# Enable testing with CTest
enable_testing()

set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

# Get all test source files
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# Create all tests
FOREACH(TEST_SOURCE ${TEST_SOURCES})
    # Get the name of the test
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Add the test executable
    add_executable(
        ${TEST_NAME}
        ${TEST_SOURCE}
    )

    # Add the test to CTest
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
    )

ENDFOREACH()

# ------------------- Coverage (gcovr) ---------------------- #

# Create binaries with coverage flags
FOREACH(TEST_SOURCE ${TEST_SOURCES})

    # Get the name of the test
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_custom_target(${TEST_NAME}_coverage_binary 
        ALL
        COMMAND g++ -ftest-coverage -fprofile-arcs -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME}_coverage_binary ${TEST_SOURCE}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME}_coverage_binary # Run the test to generate coverage data
    )

ENDFOREACH()

# Create coverage target
add_custom_target(coverage_report
    ALL
    COMMAND gcovr -r ${CMAKE_SOURCE_DIR} --html --html-details -o ${CMAKE_BINARY_DIR}/coverage_report/
    COMMENT "Generating coverage report"
)

# coverage_report depends on all the coverage binaries
FOREACH(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_dependencies(coverage_report ${TEST_NAME}_coverage_binary)

ENDFOREACH()

# ------------------- Doxygen ------------------------------- #

# Find Doxygen and other dependencies
find_package(Doxygen
             REQUIRED dot
             OPTIONAL_COMPONENTS mscgen dia)

# Set the output directory for Doxygen
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)

# Make doxygen extract all comments
set(DOXYGEN_EXTRACT_ALL YES)

# Exlclude libraries from doxygen
set(DOXYGEN_EXCLUDE_PATTERNS ${PROJECT_DEPS}/*)

# Use convenience function to add doxygen docs
doxygen_add_docs(
    ${PROJECT_NAME}_docs
    ALL 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generate Doxygen documentation"
)