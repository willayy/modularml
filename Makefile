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


.PHONY: all default_gemm blocked_gemm avx_gemm build run clean install test coverage docs

all: default_gemm

default_gemm:
	@echo "Configuring the project with default GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=ON -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8

blocked_gemm:
	@echo "Configuring the project with blocked GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=ON -DUSE_AVX_GEMM=OFF
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8
	
avx_gemm:
	@echo "Configuring the project with AVX GEMM implementation..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFAULT_GEMM=OFF -DUSE_BLOCKED_GEMM=OFF -DUSE_AVX_GEMM=ON 
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

test_avx: avx_gemm
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


clean:
	@echo "Cleaning up...\n"
	@rm -rf $(BUILD_DIR)