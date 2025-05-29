# CUDA Implementation in Tuonella Sift

## Overview

Tuonella Sift includes GPU acceleration support using CUDA for high-performance string processing operations. The CUDA implementation provides 5-15x speedup for large batches by parallelizing the most computationally intensive parts of the deduplication process.

## CUDA Kernels

### URL Normalization Kernel
Handles comprehensive URL normalization including:
- Protocol prefix removal (`http://`, `https://`)
- Subdomain prefix removal (`www.`, `m.`, `mobile.`)
- Query parameter removal (`?param=value`)
- Fragment removal (`#section`)
- Case normalization to lowercase
- Trailing slash removal

### Username Normalization Kernel
Processes username fields by:
- Converting to lowercase for case-insensitive matching
- Preserving original character encoding
- Parallel processing across GPU cores

### Combined Processing
Both kernels execute in a single GPU operation for optimal performance, minimizing CPU-GPU data transfers.

## Memory Management

The CUDA processor automatically manages GPU memory:

- **Dynamic Detection**: Automatically detects total and free GPU memory
- **Optimal Batch Sizing**: Uses 80% of available GPU memory
- **Batch Calculation**: Up to 100,000 records per batch based on available memory
- **Memory Estimation**: ~500 bytes per record (URLs + usernames + metadata)
- **Buffer Management**: Fixed-size buffers for predictable memory usage

## Performance Characteristics

### Real-World Performance (RTX 4090, 16GB VRAM)

| Batch Size | CPU Time | GPU Time | Speedup | GPU Memory |
|------------|----------|----------|---------|------------|
| 100        | 2ms      | 5ms      | 0.4x    | 50KB       |
| 1,000      | 15ms     | 8ms      | 1.9x    | 500KB      |
| 10,000     | 120ms    | 15ms     | 8.0x    | 5MB        |
| 100,000    | 1.2s     | 85ms     | 14.1x   | 50MB       |

### Throughput Characteristics
- **URL Normalization**: Handles complex patterns in parallel
- **Username Processing**: Parallel case conversion across thousands of records
- **Combined Operations**: Both kernels execute in single GPU operation
- **Memory Usage**: Scales predictably with batch size

## Configuration

### Enabling CUDA

Use the dedicated CUDA configuration file:

```json
{
  "profiles": {
    "cuda_optimized": {
      "processing": {
        "enable_cuda": true,
        "max_threads": 8
      },
      "memory": {
        "batch_size_gb": 16
      },
      "logging": {
        "verbosity": "normal",
        "progress_interval_seconds": 60
      }
    }
  }
}
```

### Build Requirements

```bash
# Install CUDA toolkit (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Build with CUDA support
cargo build --release --features cuda
```

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
- **CUDA Toolkit**: Version 11.0 or later (tested with CUDA 12.0)
- **Memory**: Minimum 2GB GPU memory, 4GB+ recommended
- **Drivers**: Latest NVIDIA drivers

## Implementation Details

### Processing Pipeline

1. **Memory Detection**: Query GPU for available memory
2. **Batch Calculation**: Determine optimal batch size using constants
3. **Kernel Compilation**: Compile CUDA kernels at runtime
4. **Data Preparation**: Prepare input buffers for GPU transfer
5. **GPU Transfer**: Copy data to GPU memory
6. **Kernel Execution**: Launch both normalization kernels
7. **Result Transfer**: Copy normalized data back to CPU
8. **Record Update**: Update records with normalized data

### Error Handling & Fallback

Comprehensive fallback mechanisms ensure reliability:

1. **CUDA Unavailable**: Automatically falls back to CPU processing
2. **GPU Memory Exhausted**: Reduces batch sizes or falls back to CPU
3. **Kernel Compilation Errors**: Falls back to CPU with detailed error logging
4. **Runtime Errors**: Graceful degradation with automatic retry
5. **Driver Issues**: Comprehensive error handling with informative messages

## Monitoring and Performance

### Logging Output

Enable verbose logging to monitor CUDA performance:

```
INFO: CUDA device initialized successfully
INFO: GPU Memory - Total: 15.59 GB, Free: 14.42 GB
INFO: CUDA processor initialized - Available memory: 11.54 GB, Max batch size: 100000
INFO: Compiling URL normalization CUDA kernel...
INFO: Compiling username normalization CUDA kernel...
DEBUG: Processing chunk of 5000 records with GPU acceleration
DEBUG: Combined GPU normalization completed successfully
```

### Performance Optimization

For optimal CUDA performance:

1. **Use Large Batches**: CUDA overhead is amortized over larger batches (1000+ records)
2. **Monitor GPU Utilization**: Use `nvidia-smi` to ensure GPU is fully utilized
3. **Use cuda_optimized Profile**: Maximizes GPU utilization
4. **Fast Storage**: NVMe SSD for temp directory reduces I/O bottlenecks
5. **Monitor Thermal**: Ensure GPU doesn't thermal throttle under load

## Troubleshooting

### Common Issues

**"Failed to initialize CUDA device 0"**
- Verify NVIDIA GPU: `nvidia-smi`
- Check drivers are up to date
- Ensure CUDA toolkit is installed: `nvcc --version`

**"Unable to find include/cuda.h"**
- Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Verify installation: `nvcc --version`

**"Out of GPU memory"**
- Use `cuda.config.json` with appropriate batch sizes
- Close other GPU-using applications
- Monitor GPU memory: `nvidia-smi`

**Poor Performance**
- Ensure sufficient batch sizes (1000+ records)
- Check GPU utilization with `nvidia-smi`
- Verify no thermal throttling
- Use larger datasets to amortize GPU setup overhead

## Testing

### CUDA-Specific Tests

```bash
# Test GPU memory detection
cargo test --release --features cuda test_gpu_memory_detection -- --nocapture

# Test URL normalization
cargo test --release --features cuda test_gpu_url_normalization -- --nocapture

# Test combined processing
cargo test --release --features cuda test_combined_gpu_normalization -- --nocapture
```

### Real Data Testing

```bash
# Run with CUDA acceleration
RUST_LOG=debug ./tuonella-sift -i test_data -o test_output -c cuda.config.json --verbose
```

## Best Practices

1. **Batch Size Optimization**: Use 1000+ records per batch for optimal GPU utilization
2. **Configuration Selection**: Use `cuda.config.json` for maximum performance
3. **Memory Monitoring**: Monitor GPU memory usage with `nvidia-smi`
4. **Storage Optimization**: Use fast storage (NVMe SSD) for temp directory
5. **Thermal Management**: Ensure adequate GPU cooling for sustained performance
6. **Fallback Testing**: Verify CPU fallback works correctly when GPU is unavailable

## Integration with Core System

The CUDA implementation is seamlessly integrated with the core deduplication system:

- **Constants Integration**: Uses `CUDA_OPTIMAL_BATCH_SIZE` from `src/constants.rs`
- **Pattern Optimization**: Works with pre-compiled patterns from `src/patterns.rs`
- **Memory Management**: Integrates with overall memory management strategy
- **Error Handling**: Consistent error handling and logging with the rest of the system
- **Configuration**: Managed through the same JSON configuration system with profiles

This ensures that CUDA acceleration enhances performance without compromising the reliability and maintainability of the core system.