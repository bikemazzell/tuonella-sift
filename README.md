# 🧹 Tuonella Sift ✨

> *"In the realm of Tuonella, every soul is judged... just like your CSV data!"*

A high-performance, memory-efficient CSV deduplication tool built in Rust with optional CUDA GPU acceleration. Named after Tuonella, the Finnish underworld where souls are sorted and filtered - just as this tool sifts through massive datasets to separate the unique from the duplicates.

Designed to handle massive datasets (hundreds of GB to TB scale) with intelligent field detection, fuzzy URL matching, GPU acceleration, and configurable processing parameters.

## ✨ Features

- **🚀 High Performance**: Multi-threaded processing with async I/O and optimized pattern matching
- **⚡ GPU Acceleration**: CUDA-powered string processing for massive performance gains (5-15x speedup)
- **💾 Memory Efficient**: Configurable batch processing to control RAM usage
- **🧠 Intelligent Field Detection**: Automatically detects user, password, and URL columns with 99%+ accuracy
- **🔍 Fuzzy URL Matching**: Normalizes URLs to catch semantic duplicates
- **⚙️ Performance Optimized**: Pre-compiled regex patterns and centralized constants for maximum speed
- **🛡️ Robust Error Handling**: Gracefully handles malformed records and encoding issues
- **📊 Progress Reporting**: Real-time progress updates with ETA calculations
- **⚙️ Configurable**: JSON-based configuration with multiple profiles for different scenarios

### 🏆 Performance Highlights

- **🔥 Pattern Matching**: 15+ million records per second with pre-compiled regex patterns
- **🎯 Field Detection**: 99%+ accuracy with enhanced validation
- **🧘 Memory Usage**: Constant memory usage regardless of dataset size
- **🚄 CUDA Acceleration**: 5-15x speedup for large batches on compatible GPUs
- **🏗️ Optimized Architecture**: Centralized constants and fast-path processing

## 📦 Installation

### Prerequisites

- **🦀 Rust 1.70+** - Install from [rustup.rs](https://rustup.rs/)
- **🧠 Memory**: At least 8GB RAM (32-64GB recommended for large datasets)
- **💽 Storage**: Sufficient disk space (2x the size of your largest batch)

### 🎮 CUDA Prerequisites (Optional - for GPU acceleration)

- **🖥️ NVIDIA GPU** with compute capability 3.5 or higher
- **🧰 CUDA Toolkit 11.0+** - Install via package manager or NVIDIA website
- **🚗 NVIDIA drivers** (latest recommended)
- **📝 GPU memory**: 4GB+ recommended for optimal performance

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

### 🏎️ Performance Recommendations

**For Large Datasets (100GB+) 🐘**
- Use CUDA-enabled build with `./build.sh --cuda`
- Ensure SSD storage for temp directory
- Monitor GPU utilization with `nvidia-smi`

**For Memory-Constrained Systems 🐁**
- Adjust memory settings in `config.json`
- Reduce record_chunk_size in configuration
- Monitor memory usage during processing

**For Maximum Speed 🚀**
- Build with CUDA support: `make cuda`
- Use SSD or NVMe storage for temp files
- Ensure GPU has 4GB+ memory

## 🎮 Command Line Options

- `--input <PATH>`: Input directory containing CSV files (required)
- `--output <PATH>`: Output file for deduplicated results (optional)
- `--config <PATH>`: Configuration file path (default: config.json)
- `--verbose`: Enable verbose output
- `--force-cpu`: Force CPU processing (disable CUDA even if available)
- `--help`: Show help information
- `--version`: Show version information

### 📝 Examples

```bash
# Minimal command with default configuration
./tuonella-sift --input ./data --output ./output/deduplicated.csv

# Custom configuration with verbose output
./tuonella-sift --input ./data --output ./output/deduplicated.csv --config custom.config.json --verbose

# Force CPU processing even with CUDA build
./tuonella-sift --input ./data --output ./output/deduplicated.csv --force-cpu
```

## ⚙️ Configuration

The tool uses a configuration file in JSON format:

```json
{
  "memory": {
    "max_ram_usage_gb": 32,
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
    "normalize_urls": true
  },
  "logging": {
    "verbosity": "normal"
  },
  "cuda": {
    "gpu_memory_usage_percent": 80,
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
- `max_ram_usage_gb`: Maximum RAM usage in GB
- `auto_detect_memory`: Automatically configure memory settings

**⚙️ Processing Settings**
- `enable_cuda`: Enable GPU acceleration (requires CUDA build)
- `chunk_size_mb`: Processing chunk size in MB
- `record_chunk_size`: Records per chunk
- `max_memory_records`: Maximum records to hold in memory

**🧹 Deduplication Settings**
- `case_sensitive_usernames`: Username case sensitivity (default: false)
- `normalize_urls`: Enable URL normalization (default: true)

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

## 📁 Project Structure

```
tuonella-sift/
├── src/
│   ├── bin/            # Executable entry points
│   ├── config/         # Configuration handling
│   ├── core/           # Core deduplication logic
│   ├── cuda/           # CUDA acceleration
│   └── utils/          # Utility functions
├── docs/               # Documentation
├── examples/           # Example code
├── tests/              # Integration tests
├── build.sh            # Build script
├── Makefile            # Build system
├── config.json         # Default configuration
└── README.md           # This file
```

## 🧙‍♂️ How It Works

### 🔍 Field Detection
The tool automatically identifies CSV columns containing:
- **✉️ User/Email**: Email patterns with 99%+ accuracy
- **🔑 Password**: Password-like field detection
- **🌐 URL**: Domain and URL pattern recognition

Uses intelligent sampling with pre-compiled regex patterns for maximum performance.

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

### 🚄 Performance Features
- **⚡ Pre-compiled patterns**: 15+ million records/second processing
- **🔥 GPU acceleration**: 5-15x speedup with CUDA for large batches
- **🧠 Memory efficiency**: Constant RAM usage via configurable batching
- **⚙️ Optimized architecture**: Fast-path processing and centralized constants

## 📊 Output

The tool produces:
- **📝 Deduplicated CSV file**: Clean, duplicate-free data
- **📊 Processing summary**: Statistics on records processed and duplicates removed
- **⏱️ Performance metrics**: Processing speed and time taken
- **⚠️ Invalid records log**: Detailed information about invalid records

## 🩺 Troubleshooting

### 🔧 Common Issues

**💥 Out of Memory Errors**
- Reduce `max_ram_usage_gb` in config.json
- Decrease `record_chunk_size`
- Ensure sufficient swap space

**🐢 Slow Processing**
- Use SSD storage for temp directory
- Enable CUDA if you have an NVIDIA GPU
- Increase `record_chunk_size` for better batching

**🧩 Field Detection Issues**
- Use `--verbose` to see detected field positions
- Verify sample data format and encoding

**🎮 CUDA Issues**
- Verify NVIDIA GPU and drivers with `nvidia-smi`
- Install CUDA toolkit if missing
- Ensure batch sizes are large enough
- Reduce GPU memory usage if needed

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