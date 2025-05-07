cmake_minimum_required(VERSION 3.30)
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

# ------------------- GEMM & Optimizations ----------------- #
option(ALIGN_TENSORS "Enable alignment for tensor data for better performance" ON)

# default memory alignment
Set(MEMORY_ALIGNMENT 16)

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
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)


# Dependency directory
set(PROJECT_DEPS "${CMAKE_BINARY_DIR}/_deps")

# ------------------- Binary ------------------------------- #
# Use max 8 threads for building, improve build times

find_package(Threads REQUIRED)
set(CMAKE_BUILD_PARALLEL_LEVEL 8) 

# Binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Library output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# ------------------- Source files ------------------------- #

set(BUILD "default" CACHE STRING "Source variant to build")

# Define source and include paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(BASE_DIR "${SRC_DIR}/default")
set(VAR_DIR "${SRC_DIR}/${BUILD}")
set(MAIN_FILE "${SRC_DIR}/main.cpp")

# include the headers
include_directories(${INCLUDE_DIR})

# Glob all source files
file(GLOB_RECURSE DEFAULT_SOURCES RELATIVE "${BASE_DIR}" "${BASE_DIR}/*.cpp")

# Glob all header files
file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.hpp")

# Supported extensions
set(SUPPORTED_EXTS ".cpp" ".cu" ".metal" ".cl")

foreach(REL_PATH IN LISTS DEFAULT_SOURCES)
    string(REGEX REPLACE "\\.cpp$" "" REL_BASE "${REL_PATH}")

    unset(VARIANT_MATCHES)
    set(VARIANT_MATCHES "")
    foreach(EXT IN LISTS SUPPORTED_EXTS)
        file(GLOB MATCH RELATIVE "${VAR_DIR}" "${VAR_DIR}/${REL_BASE}${EXT}")
        list(APPEND VARIANT_MATCHES ${MATCH})
    endforeach()

    if(VARIANT_MATCHES)
        list(LENGTH VARIANT_MATCHES N)
        if(N EQUAL 1)
            list(GET VARIANT_MATCHES 0 MATCHED_REL)
            list(APPEND SOURCES "${VAR_DIR}/${MATCHED_REL}")
        else()
            message(FATAL_ERROR
                "Found ${N} files matching ${VAR_DIR}/${REL_BASE}.*; "
                "expected exactly one variant file.")
        endif()

    else()
        list(APPEND SOURCES "${BASE_DIR}/${REL_PATH}")
    endif()
endforeach()

# Show the final list of source files (for debugging)
message(STATUS "Compiling with variant: ${BUILD}")
foreach(SOURCE_FILE IN LISTS SOURCES)
    message(STATUS "  -> ${SOURCE_FILE}")
endforeach()

# ------------------- Add/link Library ------------------------------ #

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

# ------------------- Compiler configuration options ----------------------- #

if(EXISTS "${VAR_DIR}/CMakeLists.txt")
    message(STATUS "Found CMakeLists.txt in ${VAR_DIR}")
    include("${VAR_DIR}/CMakeLists.txt")
else()
    message(STATUS "No CMakeLists.txt found in ${VAR_DIR}")
endif()

# ------------------- Testing ------------------------------ #

enable_testing()

set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")

# Get all test source files
file(GLOB TEST_SOURCES "${TESTS_DIR}/*.cpp")

# Add the test executable
add_executable(
    ${PROJECT_NAME}_tests
    ${TEST_SOURCES}
)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # Compiler flags
    target_compile_options(${PROJECT_NAME}_tests PRIVATE 
    -g 
    -O0 
    -fPIC 
    --coverage 
    -fprofile-abs-path 
    -Wfatal-errors 
    -fmax-errors=1 
    -pedantic-errors
    )
    # Linker flags
    target_link_options(${PROJECT_NAME}_tests PRIVATE 
    -g 
    -O0 
    -fPIC 
    --coverage 
    -fprofile-abs-path
    )
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
    # Compiler flags
    target_compile_options(${PROJECT_NAME}_tests PRIVATE
    -g
    -O0
    -fPIC
    -fprofile-arcs
    -ftest-coverage
    -ferror-limit=1
    -pedantic-errors
    -Wfatal-errors
    )
    # Linker flags
    target_link_options(${PROJECT_NAME}_tests PRIVATE
    -g
    -O0
    -fPIC
    -fprofile-arcs
    -ftest-coverage
    )
endif()

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
