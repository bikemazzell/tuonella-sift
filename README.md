# Tuonella Sift

A high-performance, memory-efficient CSV deduplication tool built in Rust with optional CUDA GPU acceleration. Named after Tuonella, the Finnish underworld where souls are sorted and filtered - just as this tool sifts through massive datasets to separate the unique from the duplicates.

Designed to handle massive datasets (hundreds of GB to TB scale) with intelligent field detection, fuzzy URL matching, GPU acceleration, and configurable processing parameters.

## Features

- **High Performance**: Multi-threaded processing with async I/O and optimized pattern matching
- **GPU Acceleration**: CUDA-powered string processing for massive performance gains (5-15x speedup)
- **Memory Efficient**: Configurable batch processing to control RAM usage
- **Intelligent Field Detection**: Automatically detects user, password, and URL columns with 99%+ accuracy
- **Fuzzy URL Matching**: Normalizes URLs to catch semantic duplicates
- **Performance Optimized**: Pre-compiled regex patterns and centralized constants for maximum speed
- **Robust Error Handling**: Gracefully handles malformed records and encoding issues
- **Progress Reporting**: Real-time progress updates with ETA calculations
- **Checkpointing**: Resume interrupted processing (optional)
- **Configurable**: JSON-based configuration with multiple profiles for different scenarios

### Performance Highlights

- **Pattern Matching**: 15+ million records per second with pre-compiled regex patterns
- **Field Detection**: 99%+ accuracy with enhanced validation
- **Memory Usage**: Constant memory usage regardless of dataset size
- **CUDA Acceleration**: 5-15x speedup for large batches on compatible GPUs
- **Optimized Architecture**: Centralized constants and fast-path processing

## Installation

### Prerequisites

- **Rust 1.70+** - Install from [rustup.rs](https://rustup.rs/)
- **Memory**: At least 8GB RAM (32-64GB recommended for large datasets)
- **Storage**: Sufficient disk space (2x the size of your largest batch)

### CUDA Prerequisites (Optional - for GPU acceleration)

- **NVIDIA GPU** with compute capability 3.5 or higher
- **CUDA Toolkit 11.0+** - Install via package manager or NVIDIA website
- **NVIDIA drivers** (latest recommended)
- **GPU memory**: 4GB+ recommended for optimal performance

### Build from Source

```bash
git clone <repository-url>
cd tuonella-sift
make  # Creates ./tuonella-sift binary
```

### CUDA Setup (Ubuntu/Debian)

```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Build with CUDA support
make cuda

# Verify installation
nvidia-smi
```

## Quick Start

1. **Prepare your data**: Place CSV files in a directory
2. **Run deduplication**:

```bash
# Basic usage with optimized defaults
./tuonella-sift -i /path/to/csv/files -o /path/to/output

# Use CPU configuration (conservative, works on most machines)
./tuonella-sift -i /path/to/csv/files -o /path/to/output -c cpu.config.json

# With CUDA acceleration for maximum performance (if built with CUDA support)
./tuonella-sift -i /path/to/csv/files -o /path/to/output -c cuda.config.json
```

## Operation Instructions

### Basic Usage

```bash
# Process all CSV files in a directory
./tuonella-sift -i /path/to/input/directory -o /path/to/output/directory

# Use specific configuration file
./tuonella-sift -i /path/to/input -o /path/to/output -c cuda.config.json

# Verbose output for debugging
./tuonella-sift -i /path/to/input -o /path/to/output --verbose

# Resume from checkpoint
./tuonella-sift -i /path/to/input -o /path/to/output --resume
```

### Performance Recommendations

**For Large Datasets (100GB+)**
- Use CUDA-enabled build with `cuda.config.json`
- Ensure SSD storage for temp directory
- Monitor GPU utilization with `nvidia-smi`

**For Memory-Constrained Systems**
- Use `cpu.config.json` for conservative settings
- Reduce batch sizes in configuration
- Monitor memory usage during processing

**For Maximum Speed**
- Build with CUDA support: `make cuda`
- Use `cuda.config.json` configuration
- Use NVMe SSD storage
- Ensure GPU has 4GB+ memory

## Command Line Options

- `-i, --input <PATH>`: Input directory containing CSV files (required)
- `-o, --output <PATH>`: Output directory for deduplicated files (optional, uses config default)
- `-c, --config <PATH>`: Configuration file path (default: cpu.config.json)
- `-v, --verbose`: Enable verbose output
- `--resume`: Resume from checkpoint if available

### Examples

```bash
# Minimal command - uses default CPU configuration
./tuonella-sift -i ./data -o ./output

# Custom configuration with verbose output
./tuonella-sift -i ./data -o ./output -c custom.config.json --verbose

# Resume interrupted processing
./tuonella-sift -i ./data -o ./output --resume

# Maximum performance with CUDA
./tuonella-sift -i ./data -o ./output -c cuda.config.json
```

## Configuration

The tool uses two main configuration profiles:

- **cpu.config.json**: Conservative settings for broad compatibility (default)
- **cuda.config.json**: High-performance settings for CUDA-enabled systems

### Configuration Profiles

**CPU Profile (cpu.config.json)** - Conservative, works on most machines:
- 30% RAM usage, 1GB batches
- CPU-only processing
- 32MB chunks, moderate threading
- Normal logging every 30 seconds

**CUDA Profile (cuda.config.json)** - Maximum performance:
- 70% RAM usage, 32GB batches
- CUDA acceleration enabled
- 256MB chunks, aggressive threading
- Verbose logging every 20 seconds

### Key Configuration Options

**Memory Settings**
- `max_ram_usage_percent`: RAM usage limit (30% for CPU, 70% for CUDA)
- `batch_size_gb`: Processing batch size (1GB for CPU, 32GB for CUDA)
- `auto_detect_memory`: Automatically configure memory settings

**Processing Settings**
- `max_threads`: Number of threads (0 = auto-detect CPU cores)
- `enable_cuda`: Enable GPU acceleration (false for CPU, true for CUDA)
- `chunk_size_mb`: Processing chunk size (32MB for CPU, 256MB for CUDA)
- `record_chunk_size`: Records per chunk (2000 for CPU, 20000 for CUDA)

**Deduplication Settings**
- `case_sensitive_usernames`: Username case sensitivity (default: false)
- `normalize_urls`: Enable URL normalization (default: true)
- `strip_url_params`: Remove URL query parameters (default: true)
- `completeness_strategy`: Record completeness method ("character_count" or "field_count")
- `field_detection_sample_percent`: File sampling percentage (default: 5.0%)

## Building and Testing

```bash
# Build and test
make && cargo test

# CUDA build and test
make cuda && cargo test --features cuda

# Install system-wide
make install
```

## How It Works

### Field Detection
The tool automatically identifies CSV columns containing:
- **User/Email**: Email patterns with 99%+ accuracy
- **Password**: Password-like field detection
- **URL**: Domain and URL pattern recognition

Uses intelligent sampling (5% of file by default) with pre-compiled regex patterns for maximum performance.

### URL Normalization
URLs are normalized for fuzzy duplicate detection:
- `https://www.facebook.com/user123/` → `facebook.com/user123`
- `http://m.facebook.com/user123` → `facebook.com/user123`
- `https://mobile.twitter.com/test?param=value` → `twitter.com/test`

### Deduplication Logic
Records are duplicates if they have:
- Same normalized username (case-insensitive by default)
- Same normalized URL

The most complete record is kept (by character count or field count).

### Performance Features
- **Pre-compiled patterns**: 15+ million records/second processing
- **GPU acceleration**: 5-15x speedup with CUDA for large batches
- **Memory efficiency**: Constant RAM usage via configurable batching
- **Optimized architecture**: Fast-path processing and centralized constants

## Output

The tool produces:
- **Deduplicated CSV files**: Clean, duplicate-free data maintaining original structure
- **Processing summary**: Statistics on records processed and duplicates removed
- **Performance metrics**: Processing speed and field detection accuracy
- **Log files**: Detailed processing information (configurable)

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Use `cpu.config.json` or reduce `batch_size_gb`
- Lower `max_ram_usage_percent`
- Ensure sufficient swap space

**Slow Processing**
- Use SSD storage for temp directory
- Use `cuda.config.json` for optimized settings
- Enable CUDA if you have an NVIDIA GPU
- Increase `max_threads` if CPU usage is low

**Field Detection Issues**
- Use `--verbose` to see detected field positions
- Verify sample data format and encoding
- Adjust `field_detection_sample_percent` if needed

**CUDA Issues**
- Verify NVIDIA GPU and drivers with `nvidia-smi`
- Install CUDA toolkit if missing
- Ensure batch sizes are large enough (1000+ records)
- Reduce `batch_size_gb` for GPU memory errors

## Why "Tuonella Sift"?

In Finnish mythology, Tuonella is the realm of the dead, ruled by Tuoni and his wife Tuonetar. It's a place where souls are sorted, judged, and filtered - separating those who belong from those who don't. Similarly, Tuonella Sift processes vast datasets, carefully examining each record to separate the unique souls (data) from the duplicates, ensuring only the worthy records pass through to the final output.

## License

MIT License

Copyright (c) 2024 Tuonella Sift Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install Rust 1.70+ from [rustup.rs](https://rustup.rs/)
4. Build the project: `make` or `make cuda`

### Making Changes
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and add tests
3. Run tests: `cargo test`
4. Run with CUDA tests if applicable: `cargo test --features cuda`
5. Ensure code formatting: `cargo fmt`
6. Check for issues: `cargo clippy`

### Submitting Changes
1. Commit your changes with clear messages
2. Push to your fork: `git push origin feature-name`
3. Create a pull request with:
   - Clear description of changes
   - Test results
   - Performance impact (if applicable)

### Areas for Contribution
- Performance optimizations
- Additional field detection patterns
- New configuration options
- Documentation improvements
- Test coverage expansion
- CUDA optimizations

For questions or discussions, please open an issue first.