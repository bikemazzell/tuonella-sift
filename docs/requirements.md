# Tuonella Sift - Requirements & Solution

## Problem Overview

- Multiple large CSV files (hundreds of GBs, up to 1.2TB total)
- Files contain records with user, password, and URL fields
- Duplicates may exist across files and within files
- Need to remove duplicates without using excessive disk space
- Files have varying formats, delimiters, and data quality issues
- Processing must be done in secure, offline environment

## Requirements Analysis

### Security & Data Handling
- **Environment**: Secure, offline processing only
- **Data Protection**: No retention requirements, no audit logging needed
- **Input Validation**: File size limits based on available system memory and disk space
- **File Types**: CSV files only, but must handle malformed/corrupted records gracefully

### Performance & Resource Constraints
- **System Specs**: 32-64GB RAM available
- **Disk Space**: Must handle files in hundreds of GB range
- **Processing**: Single machine only (no distributed processing)
- **Language**: Prioritize raw speed, parallelism, and memory efficiency over ease of development
- **Progress Reporting**: Non-real-time progress updates (every 30-60 seconds) with ETA

### Data Quality & Deduplication Logic
- **Fuzzy Matching**: URLs should be normalized for semantic matching
  - Strip prefixes: `www.`, `m.`, `mobile.`, `http://`, `https://`
  - Remove query parameters: `site.com/page?ref=123` → `site.com/page`
  - Case-insensitive domain matching
- **Username Handling**: Case-insensitive by default, email+tag treated as different users
- **Password Handling**: Different passwords = different records (no similarity matching)
- **Record Selection**: When duplicates found, keep most complete record (by character count)
- **Duplicate Key**: user + normalized_url combination

### Architecture & Configuration
- **Configuration**: External JSON config file for batch sizes, memory limits, parallelism settings
- **Error Handling**: Skip corrupted/unprocessable records, configurable verbosity for logging
- **Recovery**: Checkpointing capability for long-running operations (optional)
- **Memory Management**: Auto-detect available RAM, configurable percentage usage (default 50%)

### Input/Output Requirements
- **Input Discovery**: Scan directory recursively for .csv files
- **Output Format**: Standardized CSV structure, preserve additional fields beyond user/password/url
- **File Handling**: Support files with varying numbers of columns, maintain original data
- **Output Organization**: Configurable output file size limits (e.g., 32GB chunks)
- **File Management**: Never overwrite originals, clean temp files on completion
- **Summary Reporting**: Statistics on records processed, duplicates removed, etc.

### Deployment & Usage
- **Deployment**: Local-only command-line tool
- **No Dependencies**: No Docker, cloud deployment, or web services
- **Interface**: Simple CLI with directory input/output parameters
- **Testing**: Unit tests for individual components

## Solution: High-Performance Rust Implementation

### Technology Choice: Rust
- **Performance**: Superior to Python/Go for CPU-intensive data processing
- **Memory Safety**: Zero-cost abstractions with guaranteed memory safety
- **Parallelism**: Excellent async/await and multi-threading support
- **Ecosystem**: Robust CSV processing and data manipulation libraries

### Key Architecture Components

#### 1. Intelligent Field Detection (`src/record.rs`)
- **Pattern Matching**: Regex-based detection for emails, URLs, passwords
- **Smart Sampling**: Configurable percentage-based sampling distributed across entire file
- **Robust Analysis**: Handles concatenated files with varying formats throughout
- **Flexible Parsing**: Handles varying column orders and formats
- **Delimiter Detection**: Auto-detects comma, semicolon, tab, pipe delimiters

#### 2. Advanced URL Normalization
```rust
// Examples of normalization:
"https://www.facebook.com/user123/" → "facebook.com/user123"
"http://m.facebook.com/user123" → "facebook.com/user123"  
"facebook.com/user123?ref=123" → "facebook.com/user123"
```

#### 3. Memory-Efficient Processing (`src/deduplicator.rs`)
- **Batch Processing**: Configurable batch sizes based on available memory
- **Progressive Merging**: Constant memory usage regardless of dataset size
- **Parallel Processing**: Multi-threaded file processing with async I/O
- **Smart Batching**: Large files processed individually, smaller files grouped

#### 4. Configuration Management (`src/config.rs`)
- **Auto-Detection**: Automatically detects system memory and CPU cores
- **JSON Configuration**: All settings externalized and configurable
- **Validation**: Input validation with sensible defaults
- **Memory Optimization**: Dynamic adjustment based on available resources

#### 5. Advanced Sampling Strategy
- **Distributed Sampling**: Samples records from across the entire file, not just the beginning
- **Configurable Percentage**: Default 5% sampling rate, adjustable via configuration
- **Adaptive Sample Size**: Respects minimum (50) and maximum (1000) sample limits
- **Concatenated File Support**: Handles files that may have different formats in different sections
- **Deterministic Sampling**: Uses consistent sampling pattern for reproducible results

#### 6. Robust Error Handling
- **Graceful Degradation**: Skip malformed records without stopping
- **Encoding Support**: Handles UTF-8 and Latin-1 encoding issues
- **Detailed Logging**: Configurable verbosity levels (silent/normal/verbose)
- **Progress Tracking**: Real-time progress with ETA calculations

### Configuration Options

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
    "max_output_file_size_gb": 32
  },
  "deduplication": {
    "case_sensitive_usernames": false,
    "normalize_urls": true,
    "strip_url_params": true,
    "strip_url_prefixes": true,
    "completeness_strategy": "character_count",
    "field_detection_sample_percent": 5.0,
    "min_sample_size": 50,
    "max_sample_size": 1000
  },
  "logging": {
    "verbosity": "normal",
    "progress_interval_seconds": 30
  },
  "recovery": {
    "enable_checkpointing": true,
    "checkpoint_interval_records": 1000000
  }
}
```

### Performance Characteristics

#### Memory Usage
- **Configurable**: 30-70% of available RAM (default 50%)
- **Batch Processing**: Processes data in memory-controlled chunks
- **Constant Memory**: Memory usage independent of total dataset size
- **Auto-Scaling**: Automatically adjusts batch sizes based on available memory

#### Processing Speed
- **Multi-Threading**: Utilizes all available CPU cores
- **Async I/O**: Parallel file reading and writing
- **Optimized Data Structures**: High-performance hash maps with custom hashing
- **SIMD Optimizations**: Compiler-level vectorization for string processing

#### Scalability
- **Linear Scaling**: Processing time scales linearly with data size
- **Disk I/O Optimization**: Memory-mapped files for large datasets
- **Temporary Storage**: Requires ~1x largest batch size in temp space
- **Resume Capability**: Checkpoint-based recovery for interrupted processing

### Usage Examples

```bash
# Basic usage
./tuonella-sift -i /path/to/csv/files -o /path/to/output

# Custom configuration
./tuonella-sift -i input_dir -o output_dir -c custom_config.json

# Verbose output with progress
./tuonella-sift -i input_dir -o output_dir --verbose

# Resume interrupted processing
./tuonella-sift -i input_dir -o output_dir --resume
```

### Output and Reporting

#### Deduplication Results
- **Preserved Format**: Maintains original CSV structure and additional fields
- **File Organization**: Configurable output file size limits for manageability
- **Clean Output**: Standardized format while preserving all original data

#### Processing Statistics
```
Files processed: 1,247
Total records: 15,432,891
Unique records: 12,891,445
Duplicates removed: 2,541,446
Processing time: 1,847.3s
```

#### Progress Reporting
```
Progress: 847/1247 files (67.9%) - ETA: 12.4m
Batch 3/8: Processing 156 files (2.3 GB)
```

### Key Improvements Over Initial Requirements

#### Enhanced Sampling Strategy
The implementation addresses the critical issue of concatenated files with varying formats:
- **Distributed Sampling**: Instead of sampling only the first 100 lines, the tool now samples a configurable percentage (default 5%) of records distributed across the entire file
- **Adaptive Limits**: Respects minimum (50) and maximum (1000) sample sizes to balance accuracy with performance
- **Concatenated File Support**: Handles files where different sections may have different formats or delimiters
- **Configurable Parameters**: All sampling parameters are externalized in the JSON configuration

This ensures accurate field detection even for large files that may have been created by concatenating multiple sources with different formats.

### Future Extensibility

The modular architecture supports future enhancements:
- **CUDA Acceleration**: GPU processing for massive datasets
- **Distributed Processing**: Multi-machine coordination (if needed)
- **Advanced Fuzzy Matching**: Levenshtein distance for usernames
- **Custom Field Detection**: User-defined column mapping
- **Output Formats**: Support for other formats beyond CSV

This solution provides a production-ready, high-performance tool optimized for the specific requirements of large-scale CSV deduplication while maintaining flexibility for future needs.