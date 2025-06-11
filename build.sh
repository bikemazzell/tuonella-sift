#!/bin/bash

set -e

echo "🔨 Building Tuonella Sift..."

# Check if CUDA feature should be enabled
CUDA_FLAG=""
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit detected, enabling CUDA acceleration"
    CUDA_FLAG="--features cuda"
else
    echo "⚠️ CUDA toolkit not found, building CPU-only version"
fi

# Build the main binary
echo "📦 Compiling tuonella-sift..."
cargo build --release $CUDA_FLAG

# Copy to root directory for easier access
if [ -f "target/release/tuonella-sift" ]; then
    cp target/release/tuonella-sift ./tuonella-sift
    echo "✅ Built successfully: ./tuonella-sift"
    
    # Make executable
    chmod +x ./tuonella-sift
    
    echo ""
    echo "🚀 Usage:"
    echo "  ./tuonella-sift --input ./input_dir --output ./output.csv --config config.json"
    echo ""
    echo "📖 For more options:"
    echo "  ./tuonella-sift --help"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "🧪 Running tests..."
cargo test --lib $CUDA_FLAG

echo ""
echo "✅ Tuonella Sift ready!"