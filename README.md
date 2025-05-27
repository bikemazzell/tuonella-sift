# Tuonella Sift

A high-performance, memory-efficient CSV deduplication tool built in Rust. Named after Tuonella, the Finnish underworld where souls are sorted and filtered - just as this tool sifts through massive datasets to separate the unique from the duplicates.

Designed to handle massive datasets (hundreds of GB to TB scale) with intelligent field detection, fuzzy URL matching, and configurable processing parameters.

## Why "Tuonella Sift"?

In Finnish mythology, Tuonella is the realm of the dead, ruled by Tuoni and his wife Tuonetar. It's a place where souls are sorted, judged, and filtered - separating those who belong from those who don't. Similarly, Tuonella Sift processes vast datasets, carefully examining each record to separate the unique souls (data) from the duplicates, ensuring only the worthy records pass through to the final output.

## Features

- **High Performance**: Multi-threaded processing with async I/O
- **Memory Efficient**: Configurable batch processing to control RAM usage
- **Intelligent Field Detection**: Automatically detects user, password, and URL columns
- **Fuzzy URL Matching**: Normalizes URLs to catch semantic duplicates
- **Robust Error Handling**: Gracefully handles malformed records and encoding issues
- **Progress Reporting**: Real-time progress updates with ETA calculations
- **Checkpointing**: Resume interrupted processing (optional)
- **Configurable**: JSON-based configuration for all processing parameters

## Installation

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- At least 8GB RAM (32-64GB recommended for large datasets)
- Sufficient disk space (2x the size of your largest batch)

### Build from Source

```bash
git clone <repository-url>
cd tuonella-sift
cargo build --release
```

The compiled binary will be available at `target/release/tuonella-sift`.

### Optional: CUDA Support

For CUDA acceleration (if available):

```bash
cargo build --release --features cuda
```

## Quick Start

1. **Prepare your data**: Place CSV files in a directory
2. **Configure settings**: Edit `config.json` (optional, defaults provided)
3. **Run deduplication**:

```bash
./target/release/tuonella-sift -i /path/to/csv/files -o /path/to/output
```

## Configuration

The tool uses a `config.json` file for configuration. Here's the default configuration:

```json
{
  "memory": {
    "max_ram_usage_percent": 50,
    "batch_size_gb": 8,
    "auto_detect_memory": true
  },
  "processing": {
    "max_threads": 0,
    "enable_cuda": true,
    "chunk_size_mb": 64,
    "max_output_file_size_gb": 32
  },
  "io": {
    "temp_directory": "./temp",
    "output_directory": "./output",
    "enable_memory_mapping": true,
    "parallel_io": true
  },
  "deduplication": {
    "case_sensitive_usernames": false,
    "normalize_urls": true,
    "strip_url_params": true,
    "strip_url_prefixes": true,
    "completeness_strategy": "character_count"
  },
  "logging": {
    "verbosity": "normal",
    "progress_interval_seconds": 30,
    "log_file": "dedup.log"
  },
  "recovery": {
    "enable_checkpointing": true,
    "checkpoint_interval_records": 1000000
  }
}
```

### Configuration Options

#### Memory Settings
- `max_ram_usage_percent`: Maximum percentage of available RAM to use (default: 50%)
- `batch_size_gb`: Size of each processing batch in GB (auto-adjusted based on available memory)
- `auto_detect_memory`: Automatically detect and configure memory settings

#### Processing Settings
- `max_threads`: Number of processing threads (0 = auto-detect CPU cores)
- `enable_cuda`: Enable CUDA acceleration if available
- `chunk_size_mb`: Size of processing chunks in MB
- `max_output_file_size_gb`: Maximum size of output files before splitting

#### I/O Settings
- `temp_directory`: Directory for temporary files during processing
- `output_directory`: Default output directory
- `enable_memory_mapping`: Use memory-mapped I/O for large files
- `parallel_io`: Enable parallel I/O operations

#### Deduplication Settings
- `case_sensitive_usernames`: Whether usernames should be case-sensitive
- `normalize_urls`: Enable URL normalization for fuzzy matching
- `strip_url_params`: Remove query parameters from URLs
- `strip_url_prefixes`: Remove www, m, mobile prefixes from URLs
- `completeness_strategy`: How to determine record completeness ("character_count" or "field_count")

## Usage

### Basic Usage

```bash
# Process all CSV files in a directory
./tuonella-sift -i /path/to/input/directory -o /path/to/output/directory

# Use custom configuration
./tuonella-sift -i /path/to/input -o /path/to/output -c custom_config.json

# Verbose output
./tuonella-sift -i /path/to/input -o /path/to/output --verbose

# Resume from checkpoint
./tuonella-sift -i /path/to/input -o /path/to/output --resume
```

### Command Line Options

- `-i, --input <PATH>`: Input directory containing CSV files (required)
- `-o, --output <PATH>`: Output directory for deduplicated files (optional, uses config default)
- `-c, --config <PATH>`: Configuration file path (default: config.json)
- `-v, --verbose`: Enable verbose output
- `--resume`: Resume from checkpoint if available

## How It Works

### Field Detection

The tool automatically detects which columns contain:
- **User/Email**: Uses pattern matching for email addresses
- **Password**: Identifies password-like fields
- **URL**: Detects URL patterns and domains

### URL Normalization

URLs are normalized for fuzzy matching:
- `https://www.facebook.com/user123/` → `facebook.com/user123`
- `http://m.facebook.com/user123` → `facebook.com/user123`
- `facebook.com/user123?ref=123` → `facebook.com/user123`

### Deduplication Logic

Records are considered duplicates if they have:
- Same normalized username (case-insensitive by default)
- Same normalized URL

When duplicates are found, the most complete record is kept (based on total character count or field count).

### Memory Management

The tool processes data in configurable batches to control memory usage:
1. Files are grouped into batches based on size limits
2. Each batch is processed independently
3. Results are merged progressively
4. Temporary files are cleaned up automatically

## Performance Tips

### For Large Datasets (100GB+)
- Increase `batch_size_gb` if you have more RAM available
- Use SSD storage for temp directory
- Enable `parallel_io` for faster I/O
- Consider using `max_threads` = CPU cores × 2

### For Memory-Constrained Systems
- Reduce `max_ram_usage_percent` to 30-40%
- Decrease `batch_size_gb`
- Disable `enable_memory_mapping`

### For Maximum Speed
- Use NVMe SSD for temp directory
- Enable CUDA if available
- Set `verbosity` to "silent"
- Disable checkpointing for shorter runs

## Output

The tool produces:
- **Deduplicated CSV files**: Clean, duplicate-free data
- **Processing summary**: Statistics on records processed, duplicates removed, etc.
- **Log file**: Detailed processing information (if enabled)

### Output Format

Output files maintain the original CSV structure while removing duplicates. Additional fields beyond user/password/URL are preserved.

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce `batch_size_gb` in config
- Lower `max_ram_usage_percent`
- Ensure sufficient swap space

**Slow Processing**
- Check if temp directory is on fast storage (SSD)
- Increase `max_threads` if CPU usage is low
- Enable `parallel_io`

**Field Detection Issues**
- Use verbose mode to see detected field positions
- Manually verify sample data format
- Check for unusual delimiters or encoding

### Getting Help

For issues or questions:
1. Check the log file for detailed error messages
2. Run with `--verbose` for more information
3. Verify your CSV files are properly formatted

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 