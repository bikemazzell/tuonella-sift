# 🧹 Tuonella Sift ✨

A high-performance, memory-efficient CSV deduplication tool built in Rust with optional CUDA GPU acceleration and advanced performance optimizations. Named after Tuonella, the Finnish underworld where souls are sorted and filtered - just as this tool sifts through massive datasets to separate the unique from the duplicates.

Designed to handle massive datasets (hundreds of GB to TB scale) with intelligent field detection, external sorting, GPU acceleration, parallel processing, and smart checkpointing.

## ✨ Features

- **🗄️ External Sort**: Handles datasets larger than available memory using disk-based sorting
- **⚡ CUDA Acceleration**: Optional GPU acceleration for string normalization and processing  
- **🧵 Parallel Processing**: Multi-threaded file processing with configurable thread pools
- **📝 Smart Checkpointing**: Resume processing from interruptions with minimal data loss
- **🔍 Cross-File Deduplication**: Efficiently removes duplicates across multiple input files
- **💾 Memory Management**: Configurable memory usage with automatic spillover to disk
- **📈 Scalable**: Linear complexity O(N log N) for optimal performance

## 🔨 Building

### Prerequisites
- **🦀 Rust 1.70+** - Install from [rustup.rs](https://rustup.rs/)
- **🧠 Memory**: At least 8GB RAM (16-64GB recommended for large datasets)
- **💽 Storage**: Sufficient disk space (~2x the size of your input data)

### Optional CUDA Prerequisites
- **🖥️ NVIDIA GPU** with compute capability 3.5 or higher
- **🧰 CUDA Toolkit 11.0+** - Install via package manager or NVIDIA website
- **🚗 NVIDIA drivers** (latest recommended)

### Build Commands

```bash
# Clone the repository
git clone https://github.com/bikemazzell/tuonella-sift
cd tuonella-sift

# Standard CPU build (will build with GPU support if CUDA toolkit detected)
./build.sh
```

The build script automatically detects CUDA availability and enables GPU acceleration if possible.

## 🚀 Usage

### Basic Usage

```bash
# Process CSV files in a directory
./tuonella-sift --input /path/to/csv/files --output /path/to/output.csv

# With verbose output
./tuonella-sift --input /path/to/csv/files --output /path/to/output.csv --verbose

# Resume from checkpoint after interruption
./tuonella-sift --input /path/to/csv/files --output /path/to/output.csv --resume

# Use custom configuration
./tuonella-sift --input /path/to/csv/files --output /path/to/output.csv --config my-config.json
```

### Command Line Options

- `--input <PATH>`: Input directory containing CSV files (required)
- `--output <PATH>`: Output file for deduplicated results (required)  
- `--config <PATH>`: Configuration file path (default: config.json)
- `--resume`: Resume from checkpoint if available
- `--verbose`: Enable verbose output
- `--help`: Show help information

## ⚙️ Configuration

The tool uses a JSON configuration file (default: `config.json`):

```json
{
  "memory_usage_percent": 50.0,
  "chunk_size_mb": 1024,
  "io_buffer_size_kb": 128,
  "processing_threads": 8,
  "enable_cuda": true,
  "cuda_batch_size": 200000,
  "cuda_memory_percent": 50.0,
  "temp_directory": "./temp",
  "enable_compression": false,
  "merge_buffer_size_kb": 512,
  "case_sensitive": false,
  "normalize_urls": true,
  "email_only_usernames": false,
  "verbose": true,
  "merge_progress_interval_seconds": 15
}
```

### Key Configuration Options

**💾 Memory & Performance**
- `memory_usage_percent`: Percentage of system RAM to use (10-90%, default: 50%)
- `chunk_size_mb`: Processing chunk size in MB (64-4096, default: 1024)
- `processing_threads`: Number of parallel processing threads (1-32, default: 8)

**🎮 CUDA Settings** 
- `enable_cuda`: Enable GPU acceleration (default: true)
- `cuda_batch_size`: GPU batch size (1000-1000000, default: 200000)
- `cuda_memory_percent`: Percentage of GPU memory to use (10-90%, default: 50%)

**🧹 Deduplication Settings**
- `case_sensitive`: Username case sensitivity (default: false)  
- `normalize_urls`: Enable URL normalization (default: true)
- `email_only_usernames`: Require usernames to be email addresses (default: false)

**📁 I/O Settings**
- `temp_directory`: Directory for temporary files (default: "./temp")
- `io_buffer_size_kb`: I/O buffer size in KB (default: 128)
- `merge_buffer_size_kb`: Merge buffer size in KB (default: 512)


## 💾 Checkpointing & Resume

The tool includes robust checkpointing for long-running operations:

- **⏰ Auto-save**: Checkpoints saved automatically during processing
- **🛑 Graceful shutdown**: Press `Ctrl+C` to interrupt and save checkpoint
- **🚀 Resume**: Use `--resume` flag to continue from where you left off
- **📊 Progress tracking**: Shows completion percentage and statistics

```bash
# If processing is interrupted, resume with:
./tuonella-sift --input /path/to/csv/files --output /path/to/output.csv --resume
```

## 🧙‍♂️ How It Works

### Architecture

**Phase 1: File Processing**
- Process multiple files in parallel using thread pool
- Use CUDA for email/URL normalization within each thread  
- Create sorted chunks with intra-chunk deduplication
- Checkpoint progress after each file completion

**Phase 2: K-Way Merge**
- Merge all sorted chunks simultaneously using priority queue
- Apply final deduplication during merge
- Stream output to prevent memory buildup
- Checkpoint merge progress by chunk ranges

### Field Detection & Deduplication

The tool automatically identifies CSV columns containing usernames, passwords, and URLs using intelligent pattern recognition. Records are considered duplicates if they have the same normalized username and URL. URL normalization handles common variations (www, mobile, https/http) to catch semantic duplicates.

## 📜 License

MIT License

Copyright (c) 2024 Tuonella Sift Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install Rust 1.70+ from [rustup.rs](https://rustup.rs/)
4. Build the project: `./build.sh`

### Making Changes
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and add tests
3. Run tests: `cargo test --lib external_sort::tests`
4. Ensure code formatting: `cargo fmt`
5. Check for issues: `cargo clippy`

### Submitting Changes
1. Commit your changes with clear messages
2. Push to your fork: `git push origin feature-name`  
3. Create a pull request with:
   - Clear description of changes
   - Test results
   - Performance impact (if applicable)

For questions or discussions, please open an issue first.

---

> *"As Tuonella judges souls, we judge your CSV rows..."* 🧙‍♀️