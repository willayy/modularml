# ModularML

![Coverage](https://raw.githubusercontent.com/willayy/modularml/gh-pages/docs/coverage-badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Build](https://github.com/willayy/modularml/actions/workflows/ci_cd.yml/badge.svg)
![GitHub Contributors](https://img.shields.io/github/contributors/willayy/modularml)

ModularML is a machine learning framework with the aim to let users more easily load, explore and experiment with ML models as to foster a greater understanding behind the underlying processes that make machine learning possible. 

The framework loads already trained models using the onnx format, the onnx file is then translated into JSON which the framework uses during runtime to construct the model internally. The user is then to run inference on the model by feeding the model some input data.

### Requirements
    - C++ Compiler (GCC >= 12, Clang >= 10)
    - Make
    - CMake (v3.20)

### Install Dependencies
The rest of the dependencies can be installed as follows.
```sh
make install
```

### Configure & Build The Framework
```sh
make all
```

### Run Tests
```sh
make test
```

### Contributing
We welcome contributions!  
Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started.

### License
This project is licensed under the [MIT License](LICENSE).

### Acknowledgments
This framework was developed as part of a Bachelor Thesis at Chalmers University of Technology.