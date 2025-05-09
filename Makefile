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

install:
	@echo "Detected OS: $(OS)"
	@echo "Installing dependencies..."
	@$(PKG_MANAGER) $(DEPENDENCIES)
	@pip install -r requirements.txt
	@echo "Installation complete!"

run:
	@echo "Running main program...\n"
	@cd ./build/bin && ./modularml
	
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