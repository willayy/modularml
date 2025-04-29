BUILD_DIR := build

CMAKE := $(shell which cmake)
OS := $(shell uname -s)

# Detect which os the user has and set the package manager and dependencies accordingly
# There are slight variations in the dependencies between operating systems.
ifeq ($(OS), Linux)
	PKG_MANAGER := sudo apt-get install -y
	DEPENDENCIES := cmake g++ make python3-pip graphviz gcovr doxygen
endif

ifeq ($(OS), Darwin) # MacOS
	PKG_MANAGER := brew install
	DEPENDENCIES := cmake g++ make graphviz gcovr doxygen include-what-you-use
endif


.PHONY: all default_gemm blocked_gemm blocked_gemm_parallel avx_gemm avx512 build run clean install test coverage docs

all: default_gemm

default_gemm:
	@echo "Configuring the project with default GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=ON -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=OFF -DUSE_AVX512_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8

blocked_gemm:
	@echo "Configuring the project with blocked GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=ON -DUSE_AVX_GEMM=OFF -DUSE_AVX512_GEMM=OFF -DUSE_OPENBLAS_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8

avx_gemm:
	@echo "Configuring the project with AVX2 GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=ON -DUSE_AVX512_GEMM=OFF -DUSE_OPENBLAS_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8

avx512_gemm:
	@echo "Configuring the project with AVX512 GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=OFF -DUSE_AVX512_GEMM=ON -DUSE_OPENBLAS_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8
openblas_gemm:
	@echo "Configuring the project with OpenBLAS GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=OFF -DUSE_AVX512_GEMM=OFF -DUSE_OPENBLAS_GEMM=ON
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8
install:
	@echo "Detected OS: $(OS)"
	@echo "Installing dependencies..."
	@$(PKG_MANAGER) $(DEPENDENCIES)
	@pip install -r requirements.txt
	@echo "Installation complete!"

run:
	@echo "Running main program...\n"
	@cd ./build/bin && ./modularml


test: test_default_gemm


# Leaving this here for now
test_default_gemm: all
	@echo "Running tests...\n"
	@if [ -n "$(TEST_NAME)" ]; then \
		echo "Running test: $(TEST_NAME)"; \
		cd ./build && ctest -R "$(TEST_NAME)" --output-on-failure; \
	else \
		echo "Running all tests..."; \
		cd ./build && ctest --output-on-failure; \
	fi

test_blocked_gemm: blocked_gemm
	@echo "Running tests...\n"
	@if [ -n "$(TEST_NAME)" ]; then \
		echo "Running test: $(TEST_NAME)"; \
		cd ./build && ctest -R "$(TEST_NAME)" --output-on-failure; \
	else \
		echo "Running all tests..."; \
		cd ./build && ctest --output-on-failure; \
	fi

test_avx_gemm: avx_gemm
	@echo "Running tests...\n"
	@if [ -n "$(TEST_NAME)" ]; then \
		echo "Running test: $(TEST_NAME)"; \
		cd ./build && ctest -R "$(TEST_NAME)" --output-on-failure; \
	else \
		echo "Running all tests..."; \
		cd ./build && ctest --output-on-failure; \
	fi

test_avx512_gemm: avx512_gemm
	@echo "Running tests...\n"
	@if [ -n "$(TEST_NAME)" ]; then \
		echo "Running test: $(TEST_NAME)"; \
		cd ./build && ctest -R "$(TEST_NAME)" --output-on-failure; \
	else \
		echo "Running all tests..."; \
		cd ./build && ctest --output-on-failure; \
	fi
	
coverage:
	@echo "Generating coverage...\n"
	@$(CMAKE) --build $(BUILD_DIR) --target coverage_report

docs:
	@echo "Generating Doxygen documentation..."
	@$(CMAKE) --build $(BUILD_DIR) --target docs

check_backends:
	@echo "Checking available backends for your system...\n"
	@echo "Default GEMM: Available! (always)"
	@echo "Blocked GEMM: Available! (always)"
	@if lscpu | grep -q 'avx2'; then \
		echo "AVX2: Available!"; \
	else \
		echo "AVX2: Not available!"; \
	fi
	@if lscpu | grep -q 'avx512'; then \
		echo "AVX-512: Available!"; \
	else \
		echo "AVX-512: Not available!"; \
	fi


clean:
	@echo "Cleaning up...\n"
	@rm -rf $(BUILD_DIR)