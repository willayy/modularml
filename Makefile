BUILD_DIR := build

CMAKE := $(shell which cmake)

.PHONY: all config build run clean

all: config build

config:
	@echo "Configuring the project..."
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug

build:
	@echo "Building the project..."
	@$(CMAKE) --build $(BUILD_DIR)

# Will install dependencies
install:
	@echo "Not set up currently..."


run: build
	@echo "Running main program...\n"
	@$(cd ./build/bin && ./modularml)


clean:
	@echo "Cleaning up...\n"
	@rm -rf $(BUILD_DIR)