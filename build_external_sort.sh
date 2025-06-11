#!/bin/bash

set -e

echo "🔨 Building External Sort Utility..."

# Check if CUDA feature should be enabled
CUDA_FLAG=""
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit detected, enabling CUDA acceleration"
    CUDA_FLAG="--features cuda"
else
    echo "⚠️ CUDA toolkit not found, building CPU-only version"
fi

# Build the external sort example
echo "📦 Compiling external sort example..."
cargo build --release --bin external_sort_example $CUDA_FLAG

# Copy to root directory for easier access
if [ -f "target/release/external_sort_example" ]; then
    cp target/release/external_sort_example ./external-sort
    echo "✅ Built successfully: ./external-sort"
    
    # Make executable
    chmod +x ./external-sort
    
    echo ""
    echo "🚀 Usage:"
    echo "  ./external-sort --input ./input_dir --output ./output.csv --config external_sort_config.json"
    echo ""
    echo "📖 For more options:"
    echo "  ./external-sort --help"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "🧪 Running tests..."
cargo test --lib external_sort::tests $CUDA_FLAG

echo ""
echo "✅ External Sort Utility ready!"
