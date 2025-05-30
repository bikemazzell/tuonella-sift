#!/bin/bash

# Build script for tuonella-sift
# This script builds the project and copies the binary to the root directory

set -e  # Exit on any error

echo "Building tuonella-sift..."

# Create necessary directories
mkdir -p ./output
mkdir -p ./temp

# Build the release version
if [ "$1" = "--release" ] || [ "$1" = "-r" ]; then
    echo "Building release version..."
    cargo build --release
    
    # Copy the release binary to root
    if [ -f "target/release/tuonella-sift" ]; then
        cp target/release/tuonella-sift ./tuonella-sift
        echo "✓ Release binary copied to ./tuonella-sift"
        
        # Make it executable
        chmod +x ./tuonella-sift
        
        # Show binary info
        echo ""
        echo "Binary information:"
        ls -lh ./tuonella-sift
        echo ""
        echo "You can now run: ./tuonella-sift --help"
    else
        echo "✗ Release binary not found!"
        exit 1
    fi
    
elif [ "$1" = "--cuda" ] || [ "$1" = "-c" ]; then
    echo "Building release version with CUDA support..."
    cargo build --release --features cuda
    
    # Copy the release binary to root
    if [ -f "target/release/tuonella-sift" ]; then
        cp target/release/tuonella-sift ./tuonella-sift
        echo "✓ CUDA-enabled release binary copied to ./tuonella-sift"
        
        # Make it executable
        chmod +x ./tuonella-sift
        
        # Show binary info
        echo ""
        echo "Binary information:"
        ls -lh ./tuonella-sift
        echo ""
        echo "You can now run: ./tuonella-sift --help"
    else
        echo "✗ CUDA-enabled release binary not found!"
        exit 1
    fi
    
else
    echo "Building debug version..."
    cargo build
    
    # Copy the debug binary to root
    if [ -f "target/debug/tuonella-sift" ]; then
        cp target/debug/tuonella-sift ./tuonella-sift
        echo "✓ Debug binary copied to ./tuonella-sift"
        
        # Make it executable
        chmod +x ./tuonella-sift
        
        # Show binary info
        echo ""
        echo "Binary information:"
        ls -lh ./tuonella-sift
        echo ""
        echo "You can now run: ./tuonella-sift --help"
    else
        echo "✗ Debug binary not found!"
        exit 1
    fi
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "Usage:"
echo "  ./tuonella-sift --input ./test_input --output ./output/deduplicated.csv"
echo "  ./tuonella-sift --input ./test_input --output ./output/deduplicated.csv --verbose"
echo "  ./tuonella-sift --input ./test_input --output ./output/deduplicated.csv --config config.json"
echo ""
