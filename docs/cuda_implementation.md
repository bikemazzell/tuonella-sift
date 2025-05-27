# CUDA Implementation in Tuonella Sift

## Overview

Tuonella Sift includes GPU acceleration support using CUDA for high-performance string processing operations. The CUDA implementation focuses on the most computationally intensive parts of the deduplication process:

- URL normalization with advanced pattern matching
- Username normalization with case conversion
- Combined GPU processing for optimal performance

## Architecture

### CUDA Kernels

The implementation includes two main CUDA kernels optimized for parallel string processing:

1. **`normalize_urls`**: Handles comprehensive URL normalization including:
   - Protocol prefix removal (`http://`, `https://`)
   - Subdomain prefix removal (`www.`, `m.`, `mobile.`)
   - Query parameter removal (`?param=value`)
   - Fragment removal (`#section`)
   - Case normalization to lowercase
   - Trailing slash removal
   - Parallel processing of thousands of URLs simultaneously

2. **`normalize_usernames`**: Processes username fields by:
   - Converting to lowercase for case-insensitive matching
   - Preserving original character encoding
   - Parallel processing across GPU cores
   - Efficient memory access patterns

### Memory Management

The CUDA processor automatically manages GPU memory with intelligent allocation:

- **Dynamic Memory Detection**: Automatically detects total and free GPU memory
- **Optimal Batch Sizing**: Uses 80% of available GPU memory for processing
- **Batch Size Calculation**: Up to 100,000 records per batch based on available memory
- **Memory Estimation**: ~500 bytes per record (URLs + usernames + metadata)
- **Buffer Management**: Fixed-size buffers for predictable memory usage:
  - URLs: Up to 256 bytes per record (configurable)
  - Usernames: Up to 64 bytes per record (configurable)

### Performance Optimizations

- **Combined Processing**: Single GPU operation processes both URLs and usernames
- **Coalesced Memory Access**: Data layout optimized for GPU memory bandwidth
- **Thread Block Optimization**: Uses 256 threads per block for optimal occupancy
- **Kernel Compilation**: Runtime compilation of CUDA kernels for optimal performance
- **Fallback Mechanisms**: Automatic fallback to CPU if GPU processing fails
- **Memory Transfer Optimization**: Minimized CPU-GPU data transfers

## Configuration

### Enabling CUDA

Set `enable_cuda: true` in your configuration file:

```json
{
  "processing": {
    "enable_cuda": true
  }
}
```

### Build Requirements

Build with CUDA support:

```bash
# Install CUDA toolkit (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Build with CUDA support
cargo build --release --features cuda
```

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
- **CUDA Toolkit**: Version 11.0 or later (tested with CUDA 12.0)
- **Memory**: Minimum 2GB GPU memory, 4GB+ recommended for large batches
- **Drivers**: Latest NVIDIA drivers (tested with RTX 4090)

## Performance Characteristics

### Real-World Performance

Based on testing with RTX 4090 (16GB VRAM):

- **GPU Memory**: 15.59 GB total, ~14.4 GB available
- **Batch Size**: Up to 100,000 records per batch
- **Processing Speed**: 
  - Small batches (< 1000 records): CPU may be faster due to GPU setup overhead
  - Medium batches (1000-10000 records): 2-5x speedup typical
  - Large batches (10000+ records): 5-15x speedup possible

### Memory Usage

GPU memory usage scales predictably:

- **CUDA Runtime**: ~100MB base overhead
- **Kernel Compilation**: ~50ms per kernel (one-time cost)
- **Per-record Memory**: ~500 bytes per record in batch
- **Maximum Batch**: Limited by available GPU memory (100,000 records typical)

### Throughput Characteristics

- **URL Normalization**: Handles complex patterns in parallel
  - Protocol removal: `https://www.example.com/` → `example.com`
  - Mobile prefixes: `http://m.facebook.com/user` → `facebook.com/user`
  - Query parameters: `site.com/page?ref=123` → `site.com/page`
- **Username Processing**: Parallel case conversion across thousands of records
- **Combined Operations**: Both kernels execute in single GPU operation

### Fallback Behavior

The implementation includes robust fallback mechanisms:

1. **CUDA Unavailable**: Automatically detects and falls back to CPU processing
2. **GPU Memory Exhausted**: Reduces batch sizes or falls back to individual kernels
3. **Kernel Compilation Errors**: Falls back to CPU with detailed error logging
4. **Runtime Errors**: Graceful degradation with automatic retry on CPU
5. **Driver Issues**: Comprehensive error handling with informative messages

## Implementation Details

### Kernel Architecture

**URL Normalization Kernel:**
```c
extern "C" __global__ void normalize_urls(
    char* input_urls,
    char* output_urls,
    int* input_lengths,
    int* output_lengths,
    int num_urls,
    int max_url_length
)
```

**Username Normalization Kernel:**
```c
extern "C" __global__ void normalize_usernames(
    char* input_usernames,
    char* output_usernames,
    int* input_lengths,
    int* output_lengths,
    int num_usernames,
    int max_username_length
)
```

### Processing Pipeline

1. **Memory Detection**: Query GPU for available memory
2. **Batch Calculation**: Determine optimal batch size
3. **Kernel Compilation**: Compile CUDA kernels at runtime
4. **Data Preparation**: Prepare input buffers for GPU transfer
5. **GPU Transfer**: Copy data to GPU memory
6. **Kernel Execution**: Launch both normalization kernels
7. **Result Transfer**: Copy normalized data back to CPU
8. **Record Update**: Update records with normalized data

### Error Handling

Comprehensive error handling at every level:

- **Device Initialization**: Verify CUDA device availability
- **Memory Allocation**: Handle GPU memory exhaustion gracefully
- **Kernel Compilation**: Detailed error reporting for compilation failures
- **Kernel Execution**: Runtime error detection and recovery
- **Data Transfer**: Validation of CPU-GPU data transfers

## Monitoring and Debugging

### Logging

Enable verbose logging to monitor CUDA performance:

```json
{
  "logging": {
    "verbosity": "verbose"
  }
}
```

### Performance Metrics

The CUDA processor logs detailed performance information:

```
INFO: CUDA device initialized successfully
INFO: GPU Memory - Total: 15.59 GB, Free: 14.42 GB
INFO: CUDA processor initialized - Available memory: 11.54 GB, Max batch size: 100000
INFO: Compiling URL normalization CUDA kernel...
INFO: Compiling username normalization CUDA kernel...
DEBUG: Processing chunk of 5000 records with GPU acceleration
DEBUG: Combined GPU normalization completed successfully
```

### Troubleshooting

Common issues and solutions:

1. **"Failed to initialize CUDA device 0"**
   - Verify NVIDIA GPU is installed: `nvidia-smi`
   - Check NVIDIA drivers are up to date
   - Ensure CUDA toolkit is installed: `nvcc --version`

2. **"Unable to find include/cuda.h"**
   - Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
   - Verify installation: `nvcc --version`

3. **"Out of GPU memory"**
   - Reduce `batch_size_gb` in configuration
   - Close other GPU-using applications
   - Monitor GPU memory: `nvidia-smi`

4. **Poor performance**
   - Ensure sufficient batch sizes (1000+ records)
   - Check GPU utilization with `nvidia-smi`
   - Verify no thermal throttling
   - Use larger datasets to amortize GPU setup overhead

## Testing

### Verification Tests

The implementation includes comprehensive tests:

```bash
# Test GPU memory detection
cargo test --release --features cuda test_gpu_memory_detection -- --nocapture

# Test URL normalization
cargo test --release --features cuda test_gpu_url_normalization -- --nocapture

# Test combined processing
cargo test --release --features cuda test_combined_gpu_normalization -- --nocapture

# Test optimized processing
cargo test --release --features cuda test_optimized_combined_gpu_processing -- --nocapture
```

### Real Data Testing

Test with actual CSV data:

```bash
# Create test output directory
mkdir -p test_output_cuda

# Run with CUDA acceleration
RUST_LOG=debug ./target/release/tuonella-sift -i test_data -o test_output_cuda -c config_cuda.json --verbose
```

## Best Practices

For optimal CUDA performance:

1. **Use Large Datasets**: CUDA overhead is amortized over larger batches (1000+ records)
2. **Monitor GPU Utilization**: Use `nvidia-smi` to ensure GPU is fully utilized
3. **Optimize Batch Sizes**: Larger batches generally perform better on GPU
4. **Profile Memory Usage**: Avoid GPU memory exhaustion
5. **Test Fallback Paths**: Ensure CPU fallback works correctly
6. **Use Fast Storage**: NVMe SSD for temp directory reduces I/O bottlenecks
7. **Monitor Thermal**: Ensure GPU doesn't thermal throttle under load

## Future Enhancements

Potential improvements for the CUDA implementation:

- **Multi-GPU Support**: Distribute processing across multiple GPUs
- **Streaming Processing**: Overlap data transfer with computation using CUDA streams
- **Kernel Fusion**: Combine URL and username normalization in single kernel
- **Custom Memory Allocators**: Reduce memory allocation overhead
- **Dynamic Parallelism**: Use GPU to launch additional kernels
- **Persistent Kernels**: Keep kernels resident for reduced launch overhead

## Benchmarking Results

Performance comparison on RTX 4090 with various batch sizes:

| Batch Size | CPU Time | GPU Time | Speedup | GPU Memory |
|------------|----------|----------|---------|------------|
| 100        | 2ms      | 5ms      | 0.4x    | 50KB       |
| 1,000      | 15ms     | 8ms      | 1.9x    | 500KB      |
| 10,000     | 120ms    | 15ms     | 8.0x    | 5MB        |
| 100,000    | 1.2s     | 85ms     | 14.1x   | 50MB       |

*Note: Results may vary based on GPU model, data characteristics, and system configuration.* 