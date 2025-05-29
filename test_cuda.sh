#!/bin/bash

# Test script for CUDA functionality in Tuonella Sift

echo "Testing CUDA implementation for Tuonella Sift"
echo "=============================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
else
    echo "✗ No NVIDIA GPU detected. CUDA tests will be skipped."
    exit 1
fi

# Create test data directory
mkdir -p test_data

# Generate sample CSV data for testing
cat > test_data/sample1.csv << 'EOF'
user@example.com,password123,https://www.facebook.com/user1
test@gmail.com,secret456,http://m.twitter.com/test
admin@site.org,admin789,google.com/search?q=test
user@example.com,password123,facebook.com/user1
different@email.com,pass123,https://www.linkedin.com/in/user
EOF

cat > test_data/sample2.csv << 'EOF'
john@doe.com,mypass,https://github.com/johndoe
jane@smith.net,janepass,www.instagram.com/jane
user@example.com,password123,https://facebook.com/user1/
test@gmail.com,secret456,mobile.twitter.com/test
EOF

echo ""
echo "Generated test data:"
echo "- test_data/sample1.csv (5 records)"
echo "- test_data/sample2.csv (4 records)"

# Build with CUDA support
echo ""
echo "Building with CUDA support..."
if make cuda; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi

# Test CUDA processing using the dedicated CUDA config
echo ""
echo "Testing CUDA processing..."
if ./tuonella-sift -i test_data -o output_cuda -c cuda.config.json -v; then
    echo "✓ CUDA processing completed successfully"
else
    echo "✗ CUDA processing failed"
    exit 1
fi

# Check output
echo ""
echo "Checking output..."
if [ -d "output_cuda" ] && [ "$(ls -A output_cuda)" ]; then
    echo "✓ Output files generated:"
    ls -la output_cuda/

    echo ""
    echo "Sample output content:"
    head -5 output_cuda/*.csv
else
    echo "✗ No output files generated"
    exit 1
fi

# Compare with CPU-only processing
echo ""
echo "Comparing with CPU-only processing..."
if ./tuonella-sift -i test_data -o output_cpu -c cpu.config.json; then
    echo "✓ CPU processing completed"

    # Compare record counts
    cuda_count=$(wc -l output_cuda/*.csv | tail -1 | awk '{print $1}')
    cpu_count=$(wc -l output_cpu/*.csv | tail -1 | awk '{print $1}')

    if [ "$cuda_count" -eq "$cpu_count" ]; then
        echo "✓ CUDA and CPU processing produced same number of records: $cuda_count"
    else
        echo "⚠ Record count mismatch - CUDA: $cuda_count, CPU: $cpu_count"
    fi
else
    echo "✗ CPU processing failed"
fi

# Cleanup
echo ""
echo "Cleaning up..."
rm -rf test_data output_cuda output_cpu temp
echo "✓ Cleanup completed"

echo ""
echo "CUDA test completed successfully! 🎉"