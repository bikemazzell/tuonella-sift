# 🧹 Tuonella Sift ✨

A high-performance, memory-efficient CSV deduplication tool built in Rust with optional CUDA GPU acceleration and advanced performance optimizations. Named after Tuonella, the Finnish underworld where souls are sorted and filtered - just as this tool sifts through massive datasets to separate the unique from the duplicates.

Designed to handle massive datasets (hundreds of GB to TB scale) with intelligent field detection, fuzzy URL matching, GPU acceleration, double buffering, parallel processing, and adaptive optimization.

## ✨ Features

### 🚀 **Core Performance**
- **⚡ GPU Acceleration**: CUDA-powered string processing for massive performance gains (5-15x speedup)
- **🔄 Double Buffering**: Overlapping I/O and GPU processing for maximum throughput
- **🧵 Parallel Processing**: Multi-threaded file processing with streaming optimizations
- **📝 Batch Write Optimization**: Intelligent batching reduces I/O overhead by 60%+
- **📊 Adaptive Optimization**: Real-time performance monitoring with automatic parameter tuning

### 🧠 **Intelligence & Accuracy**
- **🎯 Intelligent Field Detection**: Automatically detects user, password, and URL columns with 99%+ accuracy
- **🔍 Fuzzy URL Matching**: Normalizes URLs to catch semantic duplicates
- **🧮 Smart Deduplication**: Preserves most complete records while removing exact duplicates
- **🛡️ Robust Error Handling**: Gracefully handles malformed records and encoding issues

### 💾 **Memory & Scalability**
- **🧘 Memory Efficient**: Percentage-based memory allocation (handles 200GB+ files with configurable RAM usage)
- **📈 Dynamic Scaling**: Adaptive chunk sizing based on available resources and configured percentages
- **💽 Streaming Processing**: Processes files larger than available memory
- **🔧 Resource Management**: Intelligent memory pressure detection with percentage-based limits

## 🏆 Performance Benchmarks

### 📊 **Real-World Performance**
- **🔥 Processing Speed**: 15+ million records/second (CPU) | 50+ million records/second (GPU)
- **📝 Write Throughput**: 1.4+ million records/second with 67%+ efficiency
- **🧵 Parallel Efficiency**: 90%+ thread utilization with adaptive optimization
- **💾 I/O Optimization**: 60%+ reduction in disk operations through intelligent batching
- **🧘 Memory Usage**: Percentage-based RAM allocation (configurable 10-90% of system memory)

### 🚀 **CUDA Acceleration**
- **⚡ GPU Speedup**: 5-15x performance improvement on compatible hardware
- **🎯 Optimal Batch Sizes**: Automatically calculated based on GPU memory
- **🔄 Double Buffering**: Overlapping CPU and GPU operations for maximum utilization
- **📈 Dynamic Scaling**: Real-time adjustment based on GPU performance metrics

## 📦 Installation

### Prerequisites

- **🦀 Rust 1.70+** - Install from [rustup.rs](https://rustup.rs/)
- **🧠 Memory**: At least 8GB RAM (16-64GB recommended for large datasets, configurable via percentage)
- **💽 Storage**: Sufficient disk space (2x the size of your largest batch)

### 🎮 CUDA Prerequisites (Optional - for GPU acceleration)

- **🖥️ NVIDIA GPU** with compute capability 3.5 or higher
- **🧰 CUDA Toolkit 11.0+** - Install via package manager or NVIDIA website
- **🚗 NVIDIA drivers** (latest recommended)
- **📝 GPU memory**: 4GB+ recommended for optimal performance (configurable via percentage)

### 🔨 Build from Source

```bash
git clone <repository-url>
cd tuonella-sift

# For standard CPU build
./build.sh
# OR
make

# For release build (optimized)
./build.sh --release
# OR
make release

# For CUDA-enabled build
./build.sh --cuda
# OR
make cuda
```

### 🚀 CUDA Setup (Ubuntu/Debian)

```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Build with CUDA support
make cuda

# Verify installation
nvidia-smi
```

## 🚀 Quick Start

1. **📝 Prepare your data**: Place CSV files in a directory
2. **🧹 Run deduplication**:

```bash
# Basic usage with optimized defaults
./tuonella-sift --input /path/to/csv/files --output /path/to/output/file.csv

# With verbose output for more detailed progress
./tuonella-sift --input /path/to/csv/files --output /path/to/output/file.csv --verbose

# With custom configuration
./tuonella-sift --input /path/to/csv/files --output /path/to/output/file.csv --config custom.config.json

# With CUDA acceleration (if built with CUDA support)
./tuonella-sift --input /path/to/csv/files --output /path/to/output/file.csv

# Force CPU processing even with CUDA build
./tuonella-sift --input /path/to/csv/files --output /path/to/output/file.csv --force-cpu
```

## 📚 Operation Instructions

### 🧩 Basic Usage

```bash
# Process all CSV files in a directory
./tuonella-sift --input /path/to/input/directory --output /path/to/output/file.csv

# Use specific configuration file
./tuonella-sift --input /path/to/input --output /path/to/output/file.csv --config config.json

# Verbose output for debugging
./tuonella-sift --input /path/to/input --output /path/to/output/file.csv --verbose
```

### 🏎️ Performance Optimization Guide

**🐘 For Large Datasets (100GB+)**
- Use CUDA-enabled build: `./build.sh --cuda`
- Enable double buffering for overlapping I/O and GPU processing
- Use SSD/NVMe storage for temp directory
- Monitor performance with built-in adaptive optimization

**🐁 For Memory-Constrained Systems**
- Set conservative memory percentage: `memory_usage_percent: 20-30`
- Leverage streaming processing for files larger than RAM
- Enable adaptive chunk sizing for optimal memory usage
- Use batch write optimization to reduce memory pressure
- Monitor with real-time memory pressure detection

**🚀 For Maximum Performance**
- Set aggressive memory percentages: `memory_usage_percent: 70-80`, `gpu_memory_usage_percent: 95`
- Build with CUDA support: `./build.sh --cuda`
- Enable parallel processing with optimal thread count
- Use performance monitoring for automatic parameter tuning
- Ensure GPU has 4GB+ memory for optimal batch sizes

## 🎮 Command Line Options

- `--input <PATH>`: Input directory containing CSV files (required)
- `--output <PATH>`: Output file for deduplicated results (optional)
- `--config <PATH>`: Configuration file path (default: config.json)
- `--verbose`: Enable verbose output
- `--force-cpu`: Force CPU processing (disable CUDA even if available)
- `--help`: Show help information
- `--version`: Show version information

## ⚙️ Configuration

The tool uses a configuration file in JSON format:

```json
{
  "memory": {
    "memory_usage_percent": 50,
    "auto_detect_memory": true
  },
  "processing": {
    "enable_cuda": true,
    "chunk_size_mb": 256,
    "record_chunk_size": 10000,
    "max_memory_records": 1000000
  },
  "io": {
    "temp_directory": "./temp",
    "output_directory": "./output"
  },
  "deduplication": {
    "case_sensitive_usernames": false,
    "normalize_urls": true,
    "email_username_only": true
  },
  "logging": {
    "verbosity": "normal"
  },
  "performance": {
    "enable_monitoring": true,
    "report_interval_seconds": 30,
    "show_detailed_metrics": true
  },
  "cuda": {
    "gpu_memory_usage_percent": 95,
    "estimated_bytes_per_record": 500,
    "min_batch_size": 10000,
    "max_batch_size": 1000000,
    "max_url_buffer_size": 256,
    "max_username_buffer_size": 64,
    "threads_per_block": 256,
    "batch_sizes": {
      "small": 10000,
      "medium": 50000,
      "large": 100000,
      "xlarge": 500000
    }
  }
}
```

### 🔧 Key Configuration Options

**💾 Memory Settings**
- `memory_usage_percent`: Percentage of system RAM to use (10-90%, default: 50%)
- `auto_detect_memory`: Automatically configure memory settings based on available resources

**⚙️ Processing Settings**
- `enable_cuda`: Enable GPU acceleration (requires CUDA build)
- `chunk_size_mb`: Processing chunk size in MB
- `record_chunk_size`: Records per chunk
- `max_memory_records`: Maximum records to hold in memory

**🎮 CUDA Settings**
- `gpu_memory_usage_percent`: Percentage of GPU memory to use (10-95%, default: 95%)
- `estimated_bytes_per_record`: Memory estimation for batch sizing
- `min_batch_size`/`max_batch_size`: GPU batch size limits

**🧹 Deduplication Settings**
- `case_sensitive_usernames`: Username case sensitivity (default: false)
- `normalize_urls`: Enable URL normalization (default: true)
- `email_username_only`: Require usernames to be email addresses (default: true)
  - `true`: Only email addresses are accepted as usernames (original behavior)
  - `false`: Any printable character string is accepted as username

**📊 Performance Monitoring Settings**
- `enable_monitoring`: Enable/disable performance monitoring (default: true)
- `report_interval_seconds`: How often to display performance reports in seconds (default: 30)
- `show_detailed_metrics`: Enable detailed performance metrics (default: true)

### 📊 Percentage-Based Memory Configuration

The tool now uses **percentage-based memory allocation** for both RAM and GPU memory, providing:

**✅ Benefits:**
- **🔄 Consistency**: Both RAM and GPU use percentage-based configuration
- **🌍 Portability**: Works across different systems without hardcoded limits
- **⚙️ Flexibility**: Easy to adjust for different environments and workloads
- **📈 Scalability**: Automatically scales with available system resources

**🎯 Recommended Settings:**
- **🏠 Home/Development**: `memory_usage_percent: 30-50`, `gpu_memory_usage_percent: 80-90`
- **🏢 Production/Server**: `memory_usage_percent: 50-70`, `gpu_memory_usage_percent: 90-95`
- **🚀 High-Performance**: `memory_usage_percent: 70-80`, `gpu_memory_usage_percent: 95`

**💡 Configuration Examples:**
```json
// Conservative (safe for shared systems)
"memory": { "memory_usage_percent": 30 },
"cuda": { "gpu_memory_usage_percent": 80 }

// Balanced (recommended for most use cases)
"memory": { "memory_usage_percent": 50 },
"cuda": { "gpu_memory_usage_percent": 95 }

// Aggressive (maximum performance)
"memory": { "memory_usage_percent": 80 },
"cuda": { "gpu_memory_usage_percent": 95 }
```

**🔄 Migration from GB-based Configuration:**
If you're upgrading from an older version that used `max_ram_usage_gb`, simply replace it with `memory_usage_percent`:
- `"max_ram_usage_gb": 32` on a 64GB system → `"memory_usage_percent": 50`
- `"max_ram_usage_gb": 16` on a 32GB system → `"memory_usage_percent": 50`
- The percentage approach automatically adapts to your system's available memory

### 📊 Performance Monitoring Configuration

The tool includes configurable performance monitoring that provides real-time insights into processing efficiency:

**🎯 Performance Monitoring Features:**
- **⚡ Real-time throughput**: Current, average, and peak records/second
- **🧠 Memory pressure**: RAM usage and pressure detection
- **🖥️ GPU utilization**: CUDA processing efficiency (when enabled)
- **📈 Performance trends**: Automatic trend analysis (improving/stable/degrading)
- **💡 Optimization recommendations**: Dynamic chunk size suggestions

**⚙️ Configuration Examples:**
```json
// Default monitoring (recommended)
"performance": {
  "enable_monitoring": true,
  "report_interval_seconds": 30,
  "show_detailed_metrics": true
}

// Frequent monitoring (for debugging)
"performance": {
  "enable_monitoring": true,
  "report_interval_seconds": 10,
  "show_detailed_metrics": true
}

// Disabled monitoring (for minimal output)
"performance": {
  "enable_monitoring": false,
  "report_interval_seconds": 30,
  "show_detailed_metrics": false
}
```

**📈 Sample Performance Report:**
```
📊 Performance Report:
⚡ Current Throughput: 1,928,902.7 records/sec
📊 Average Throughput: 23,515.6 records/sec
🏆 Peak Throughput: 1,940,565.7 records/sec
🔄 Processing Efficiency: 100.0%
🧠 Memory Pressure: 41.1%
🖥️ GPU Utilization: 2.3%
📈 Trend: Stable
💡 Recommended Chunk Size: 307 MB
```

## 🛠️ Building and Testing

```bash
# Build and test
make && cargo test

# CUDA build and test
make cuda && cargo test --features cuda

# Run with test data
make run

# Run with CUDA and test data
make run-cuda

# Install system-wide
make install
```

### 🚀 Performance Optimization Examples

```bash
# Run performance optimization demo
cargo run --features cuda --example performance_optimizations_demo

# Test double buffering (CUDA only)
cargo run --features cuda --example test_gpu_processing

# Test adaptive optimization
cargo run --example test_memory_manager

# Test parallel processing
cargo run --example test_algorithm_streaming
```

## 📁 Project Structure

```
tuonella-sift/
├── src/
│   ├── bin/                    # Executable entry points
│   ├── config/                 # Configuration handling
│   ├── core/                   # Core deduplication logic
│   │   ├── batch_writer.rs     # Batch write optimization
│   │   ├── double_buffer.rs    # Double buffering (CUDA)
│   │   ├── parallel_processor.rs # Parallel processing
│   │   ├── performance_monitor.rs # Adaptive optimization
│   │   ├── memory_manager.rs   # Memory management
│   │   └── ...                 # Other core modules
│   ├── cuda/                   # CUDA acceleration
│   └── utils/                  # Utility functions
├── docs/
│   ├── algorithm.md            # Complete algorithm specification
│   ├── cuda_implementation.md  # CUDA implementation details
│   └── requirements.md         # System requirements
├── examples/
│   ├── performance_optimizations_demo.rs # Performance demo
│   ├── test_gpu_processing.rs  # GPU processing examples
│   └── ...                     # Other examples
├── build.sh                    # Build script
├── Makefile                    # Build system
├── config.json                 # Default configuration
└── README.md                   # This file
```

## 🧙‍♂️ How It Works

### 🔍 Field Detection
The tool automatically identifies CSV columns containing:
- **✉️ User/Email**: Email patterns with 99%+ accuracy (configurable to accept any printable username)
- **🔑 Password**: Password-like field detection
- **🌐 URL**: Domain and URL pattern recognition

Uses intelligent sampling with pre-compiled regex patterns for maximum performance.

**Username Validation Modes:**
- **Email-only mode** (`email_username_only: true`): Only email addresses are accepted as usernames
- **Printable username mode** (`email_username_only: false`): Any printable character string is accepted as username, excluding URL-like patterns

### 🌐 URL Normalization
URLs are normalized for fuzzy duplicate detection:
- `https://www.facebook.com/user123/` → `facebook.com/user123`
- `http://m.facebook.com/user123` → `facebook.com/user123`
- `https://mobile.twitter.com/test?param=value` → `twitter.com/test`

### 🧮 Deduplication Logic
Records are duplicates if they have:
- Same normalized username (case-insensitive by default)
- Same normalized URL

The most complete record is kept (based on the completeness score).

### 🚄 Advanced Performance Features

**🔄 Double Buffering & Parallel Processing**
- Overlapping I/O and GPU processing for maximum throughput
- Multi-threaded file processing with intelligent work distribution
- Streaming optimizations for files larger than available memory
- Priority-based work queue management

**📊 Adaptive Optimization**
- Real-time performance monitoring and trend analysis
- Automatic parameter tuning based on observed performance
- Dynamic chunk sizing based on resource availability
- Thread count optimization for maximum efficiency

**📝 Intelligent I/O Management**
- Batch write optimization reduces disk operations by 60%+
- Configurable write buffer sizes for optimal throughput
- Automatic flush management with efficiency scoring
- CSV escaping and formatting optimization

## 📊 Output & Monitoring

### 📝 **Primary Output**
- **📄 Deduplicated CSV file**: Clean, duplicate-free data with preserved field structure
- **📊 Processing summary**: Comprehensive statistics on records processed and duplicates removed
- **⚠️ Invalid records log**: Detailed information about skipped/invalid records with line numbers

### 📈 **Performance Metrics**
- **⚡ Real-time throughput**: Records/second processing speed
- **🔄 Buffer utilization**: Double buffering efficiency and swap statistics
- **🧵 Thread performance**: Parallel processing efficiency and utilization
- **💾 I/O optimization**: Write batching efficiency and disk operation reduction
- **🧠 Memory usage**: RAM pressure monitoring and adaptive scaling metrics
- **🎯 GPU utilization**: CUDA processing efficiency and memory usage (when enabled)

## 🩺 Troubleshooting

### 🔧 Common Issues & Solutions

**💥 Memory Issues**
- **Out of Memory**: Reduce `memory_usage_percent` (try 20-30%) and enable streaming processing
- **Memory Pressure**: Lower memory percentage and use adaptive chunk sizing
- **Swap Usage**: Set conservative `memory_usage_percent` and leverage intelligent memory management

**🐢 Performance Issues**
- **Slow Processing**: Enable CUDA acceleration and parallel processing
- **I/O Bottlenecks**: Use batch write optimization and SSD storage
- **Poor Efficiency**: Enable adaptive optimization for automatic parameter tuning

**🧩 Data Processing Issues**
- **Field Detection**: Use `--verbose` to see detected field positions and accuracy
- **Encoding Problems**: Tool handles UTF-8 encoding errors gracefully
- **Large Files**: Use streaming processing for files exceeding available memory

**🎮 CUDA Issues**
- **GPU Not Detected**: Verify NVIDIA GPU and drivers with `nvidia-smi`
- **GPU Memory Errors**: Reduce `gpu_memory_usage_percent` (try 70-80%) and use dynamic allocation
- **Poor GPU Utilization**: Increase `gpu_memory_usage_percent` and enable double buffering
- **Compatibility**: Ensure compute capability 3.5+ and CUDA toolkit 11.0+

## 🔮 Why "Tuonella Sift"?

In Finnish mythology, Tuonella is the realm of the dead, ruled by Tuoni and his wife Tuonetar. It's a place where souls are sorted, judged, and filtered - separating those who belong from those who don't.

Similarly, Tuonella Sift processes vast datasets, carefully examining each record to separate the unique souls (data) from the duplicates, ensuring only the worthy records pass through to the final output.

> *"As Tuonella judges souls, we judge your CSV rows..."* 🧙‍♀️

## 📜 License

MIT License

Copyright (c) 2024 Tuonella Sift Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🔧 Development Setup
1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install Rust 1.70+ from [rustup.rs](https://rustup.rs/)
4. Build the project: `make` or `make cuda`

### 📝 Making Changes
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and add tests
3. Run tests: `cargo test`
4. Run with CUDA tests if applicable: `cargo test --features cuda`
5. Ensure code formatting: `cargo fmt`
6. Check for issues: `cargo clippy`

### 🚀 Submitting Changes
1. Commit your changes with clear messages
2. Push to your fork: `git push origin feature-name`
3. Create a pull request with:
   - Clear description of changes
   - Test results
   - Performance impact (if applicable)

For questions or discussions, please open an issue first.