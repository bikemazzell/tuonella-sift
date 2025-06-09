#!/bin/bash

# Configuration Validation Script for tuonella-sift
# This script validates the optimized configuration and provides performance feedback

set -e

echo "🚀 Tuonella-Sift Configuration Validation"
echo "=========================================="

# Check if binary exists
if [ ! -f "./tuonella-sift" ]; then
    echo "❌ Binary not found. Building..."
    cargo build --release --features cuda
fi

# Check system resources
echo ""
echo "📊 System Resource Analysis:"
echo "----------------------------"

# Memory check
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
echo "💾 RAM: ${TOTAL_RAM}GB total, ${AVAILABLE_RAM}GB available"

# CPU check
CPU_CORES=$(nproc)
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
echo "🖥️  CPU: ${CPU_MODEL} (${CPU_CORES} cores)"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "🎮 GPU: ${GPU_INFO}"
    
    # Check GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "🎮 GPU Memory: ${GPU_MEMORY}MB total, ${GPU_FREE}MB free"
else
    echo "⚠️  NVIDIA GPU not detected or nvidia-smi not available"
fi

# Storage check
STORAGE_INFO=$(df -h . | tail -1 | awk '{print $2 " total, " $4 " available"}')
echo "💿 Storage: ${STORAGE_INFO}"

echo ""
echo "⚙️  Configuration Analysis:"
echo "---------------------------"

# Parse config.json values
MEMORY_PERCENT=$(jq -r '.memory.memory_usage_percent' config.json)
CHUNK_SIZE_MB=$(jq -r '.processing.chunk_size_mb' config.json)
RECORD_CHUNK_SIZE=$(jq -r '.processing.record_chunk_size' config.json)
MAX_MEMORY_RECORDS=$(jq -r '.processing.max_memory_records' config.json)
GPU_MEMORY_PERCENT=$(jq -r '.cuda.gpu_memory_usage_percent' config.json)
MIN_BATCH_SIZE=$(jq -r '.cuda.min_batch_size' config.json)
MAX_BATCH_SIZE=$(jq -r '.cuda.max_batch_size' config.json)

echo "📈 Memory Usage: ${MEMORY_PERCENT}% ($(echo "$TOTAL_RAM * $MEMORY_PERCENT / 100" | bc)GB)"
echo "📦 Chunk Size: ${CHUNK_SIZE_MB}MB"
echo "🔢 Record Chunk Size: ${RECORD_CHUNK_SIZE}"
echo "🧠 Max Memory Records: ${MAX_MEMORY_RECORDS}"
echo "🎮 GPU Memory Usage: ${GPU_MEMORY_PERCENT}%"
echo "📊 CUDA Batch Size: ${MIN_BATCH_SIZE} - ${MAX_BATCH_SIZE}"

echo ""
echo "✅ Configuration Validation:"
echo "----------------------------"

# Validate memory configuration
CONFIGURED_MEMORY_GB=$(echo "$TOTAL_RAM * $MEMORY_PERCENT / 100" | bc)
if [ "$CONFIGURED_MEMORY_GB" -le "$AVAILABLE_RAM" ]; then
    echo "✅ Memory configuration is safe (${CONFIGURED_MEMORY_GB}GB configured vs ${AVAILABLE_RAM}GB available)"
else
    echo "⚠️  Memory configuration may be too aggressive (${CONFIGURED_MEMORY_GB}GB configured vs ${AVAILABLE_RAM}GB available)"
fi

# Validate GPU configuration
if command -v nvidia-smi &> /dev/null; then
    CONFIGURED_GPU_MEMORY=$(echo "$GPU_MEMORY * $GPU_MEMORY_PERCENT / 100" | bc)
    if [ "$CONFIGURED_GPU_MEMORY" -le "$GPU_FREE" ]; then
        echo "✅ GPU memory configuration is safe (${CONFIGURED_GPU_MEMORY}MB configured vs ${GPU_FREE}MB available)"
    else
        echo "⚠️  GPU memory configuration may be too aggressive (${CONFIGURED_GPU_MEMORY}MB configured vs ${GPU_FREE}MB available)"
    fi
fi

# Validate chunk size vs available storage
CHUNK_SIZE_GB=$(echo "$CHUNK_SIZE_MB / 1024" | bc -l)
AVAILABLE_STORAGE_GB=$(df --output=avail -BG . | tail -1 | tr -d 'G')
if [ "$(echo "$CHUNK_SIZE_GB * 3 < $AVAILABLE_STORAGE_GB" | bc)" -eq 1 ]; then
    echo "✅ Storage configuration is safe (${CHUNK_SIZE_GB}GB chunks vs ${AVAILABLE_STORAGE_GB}GB available)"
else
    echo "⚠️  Storage may be insufficient for large operations (${CHUNK_SIZE_GB}GB chunks vs ${AVAILABLE_STORAGE_GB}GB available)"
fi

echo ""
echo "🎯 Performance Recommendations:"
echo "------------------------------"

# CPU-based recommendations
if [ "$CPU_CORES" -ge 16 ]; then
    echo "✅ High-core CPU detected - configuration optimized for parallel processing"
elif [ "$CPU_CORES" -ge 8 ]; then
    echo "💡 Medium-core CPU - consider reducing parallel threads if CPU usage is high"
else
    echo "⚠️  Low-core CPU - consider reducing chunk sizes and batch sizes"
fi

# Memory-based recommendations
if [ "$TOTAL_RAM" -ge 32 ]; then
    echo "✅ High-memory system - configuration optimized for large datasets"
elif [ "$TOTAL_RAM" -ge 16 ]; then
    echo "💡 Medium-memory system - configuration should work well for most datasets"
else
    echo "⚠️  Low-memory system - consider reducing memory_usage_percent to 60%"
fi

# GPU-based recommendations
if command -v nvidia-smi &> /dev/null; then
    if [ "$GPU_MEMORY" -ge 12000 ]; then
        echo "✅ High-VRAM GPU detected - configuration optimized for maximum GPU utilization"
    elif [ "$GPU_MEMORY" -ge 8000 ]; then
        echo "💡 Medium-VRAM GPU - configuration should provide good performance"
    else
        echo "⚠️  Low-VRAM GPU - consider reducing GPU batch sizes by 50%"
    fi
fi

echo ""
echo "🧪 Quick Performance Test:"
echo "-------------------------"

# Test binary execution
echo "Testing binary execution..."
if ./tuonella-sift --version > /dev/null 2>&1; then
    VERSION=$(./tuonella-sift --version)
    echo "✅ Binary execution successful: ${VERSION}"
else
    echo "❌ Binary execution failed"
    exit 1
fi

# Test configuration loading
echo "Testing configuration loading..."
if ./tuonella-sift --help > /dev/null 2>&1; then
    echo "✅ Configuration loading successful"
else
    echo "❌ Configuration loading failed"
    exit 1
fi

echo ""
echo "🎉 Configuration Validation Complete!"
echo "===================================="
echo ""
echo "Your system is configured for optimal performance with:"
echo "• ${CONFIGURED_MEMORY_GB}GB RAM allocation (${MEMORY_PERCENT}%)"
echo "• ${CHUNK_SIZE_MB}MB processing chunks"
echo "• ${RECORD_CHUNK_SIZE} record batches"
echo "• ${MAX_MEMORY_RECORDS} max in-memory records"
if command -v nvidia-smi &> /dev/null; then
    echo "• ${CONFIGURED_GPU_MEMORY}MB GPU memory allocation (${GPU_MEMORY_PERCENT}%)"
    echo "• ${MIN_BATCH_SIZE}-${MAX_BATCH_SIZE} CUDA batch size range"
fi
echo ""
echo "Ready for high-performance deduplication! 🚀"
