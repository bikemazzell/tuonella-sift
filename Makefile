# Makefile for tuonella-sift

.PHONY: all build release debug cuda clean test test-cuda build-auto help install dirs run run-cuda

# Default target
all: dirs release

# Create necessary directories
dirs:
	@mkdir -p ./output
	@mkdir -p ./temp

# Build release version and copy to root
release: dirs
	@echo "Building release version..."
	@cargo build --release
	@cp target/release/tuonella-sift ./tuonella-sift
	@chmod +x ./tuonella-sift
	@echo "✓ Release binary available at ./tuonella-sift"

# Build debug version and copy to root
debug: dirs
	@echo "Building debug version..."
	@cargo build
	@cp target/debug/tuonella-sift ./tuonella-sift
	@chmod +x ./tuonella-sift
	@echo "✓ Debug binary available at ./tuonella-sift"

# Build release version with CUDA support
cuda: dirs
	@echo "Building release version with CUDA support..."
	@cargo build --release --features cuda
	@cp target/release/tuonella-sift ./tuonella-sift
	@chmod +x ./tuonella-sift
	@echo "✓ CUDA-enabled release binary available at ./tuonella-sift"

# Build alias
build: release

# Build with automatic CUDA detection (uses build.sh)
build-auto: dirs
	@echo "Building with automatic CUDA detection..."
	@./build.sh

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -f ./tuonella-sift
	@echo "✓ Clean completed"

# Run tests
test:
	@echo "Running tests..."
	@cargo test --lib

# Run tests with CUDA
test-cuda:
	@echo "Running tests with CUDA support..."
	@cargo test --lib --features cuda

# Run the program with test data
run: dirs release
	@echo "Running tuonella-sift with test data..."
	@./tuonella-sift --input ./test_input --output ./output/deduplicated.csv

# Run with CUDA and test data
run-cuda: dirs cuda
	@echo "Running tuonella-sift with CUDA and test data..."
	@./tuonella-sift --input ./test_input --output ./output/deduplicated_cuda.csv

# Install to /usr/local/bin (requires sudo)
install: release
	@echo "Installing tuonella-sift to /usr/local/bin..."
	@sudo cp ./tuonella-sift /usr/local/bin/tuonella-sift
	@sudo chmod +x /usr/local/bin/tuonella-sift
	@echo "✓ tuonella-sift installed to /usr/local/bin"
	@echo "You can now run 'tuonella-sift' from anywhere"

# Uninstall from /usr/local/bin
uninstall:
	@echo "Removing tuonella-sift from /usr/local/bin..."
	@sudo rm -f /usr/local/bin/tuonella-sift
	@echo "✓ tuonella-sift uninstalled"

# Show help
help:
	@echo "Tuonella Sift Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build release version (default)"
	@echo "  release   - Build optimized release version"
	@echo "  debug     - Build debug version"
	@echo "  cuda      - Build release version with CUDA support"
	@echo "  build     - Alias for release"
	@echo "  build-auto - Build with automatic CUDA detection (uses build.sh)"
	@echo "  test      - Run all tests"
	@echo "  test-cuda - Run all tests with CUDA support"
	@echo "  clean     - Clean build artifacts"
	@echo "  run       - Run with test data"
	@echo "  run-cuda  - Run with CUDA and test data"
	@echo "  install   - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall - Remove from /usr/local/bin (requires sudo)"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build release version"
	@echo "  make build-auto   # Build with automatic CUDA detection"
	@echo "  make cuda         # Build with CUDA support"
	@echo "  make test         # Run all tests"
	@echo "  make test-cuda    # Run all tests with CUDA"
	@echo "  make run          # Run with test data"
	@echo "  make install      # Install system-wide"
	@echo ""
	@echo "After building, run:"
	@echo "  ./tuonella-sift --help" 