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
	DEPENDENCIES := cmake g++ make graphviz gcovr doxygen
endif

.PHONY: all config build run clean

all: config build

config:
	@echo "Configuring the project..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug

build:
	@echo "Building the project..."
	@$(CMAKE) --build $(BUILD_DIR) --parallel 8

# Will install dependencies
install:
	@echo "Detected OS: $(OS)"
	@echo "Installing dependencies..."
	@$(PKG_MANAGER) $(DEPENDENCIES)
	@pip install -r requirements.txt
	@echo "Installation complete!"

run:
	@echo "Running main program...\n"
	@cd ./build/bin && ./modularml

test: all
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