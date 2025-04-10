# ModularML

![Coverage](https://raw.githubusercontent.com/willayy/modularml/gh-pages/docs/coverage-badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Build](https://github.com/willayy/modularml/actions/workflows/ci_cd.yaml/badge.svg)
![GitHub Contributors](https://img.shields.io/github/contributors/willayy/modularml)

<!-- 
Doesnt work currently issue with services but will try again later
![Visitors](https://shields.io/badge/dynamic/json?label=Visitors&query=value&url=https://api.countapi.xyz/hit/willayy.modularml)
-->


ModularML is a machine learning framework with the aim to let users more easily load, explore and experiment with ML models as to foster a greater understanding behind the underlying processes that make machine learning possible. 

The framework loads already trained models using the onnx format, the onnx file is then translated into JSON which the framework uses during runtime to construct the model internally. The user is then to run inference on the model by feeding the model some input data.


### Backend Implementations
Machine learning entails heavy use of **General Matrix Multiplication (GEMM)** to perform calculations. ModularML provides several interchangeable backend implementations for executing these efficiently.
Current backend implementations include:

- **Naive GEMM Backend**
  A simple reference implementation. Good for understanding the computation step-by-step.
- **Blocked GEMM Backend**
  Optimized for cache locality by processing matrices in smaller blocks.
- **AVX2 GEMM Backend**
  Uses AVX2 SIMD intrinsics to faster computation on supported CPU:s.
- **AVX512 GEMM Backend**
  Uses AVX512 SIMD intrinsics to even faster computation on supported CPU:s.


### Requirements

- **C++ Compiler**  
  - g++ `GCC >= 12`  
  - Clang `>= 10`  
- **Build Tools**  
  - Make  
  - CMake `>= 3.20`  

### Install Dependencies
This command installs the neccessary dependencies needed to build the framework.
```sh
make install
```

### Configure & Build The Framework
This command will configure and build the framework using the default naive GEMM backend.
```sh
make all
```
To compile the framework using the blocked GEMM backend for example:
```sh
make blocked_gemm
```
To check which backends are available on your system:
```sh
make check_backends
```
This will show you a list of available and/or unavailable backends.
### Run Tests
This command will run all the unit and integration tests for the framework using the default naive GEMM backend.
```sh
make test
```
To compile and run the tests using the blocked GEMM backend for example:
```sh
make test_blocked_gemm
```

### Contributing
We welcome contributions!  
Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started.

### License
This project is licensed under the [MIT License](LICENSE).

### Acknowledgments
This framework was developed as part of a Bachelor Thesis at Chalmers University of Technology.