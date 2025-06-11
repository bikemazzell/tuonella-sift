# External Sort Utility

High-performance external sort with CUDA acceleration for processing large CSV datasets that exceed available memory.

## Features

- **External Sorting**: Handles datasets larger than available memory using disk-based sorting
- **CUDA Acceleration**: Optional GPU acceleration for string normalization and processing
- **Parallel Processing**: Multi-threaded file processing with configurable thread pools
- **Smart Checkpointing**: Resume processing from interruptions with minimal data loss
- **Cross-File Deduplication**: Efficiently removes duplicates across multiple input files
- **Memory Management**: Configurable memory usage with automatic spillover to disk
- **Scalable**: Linear complexity O(N log N) for optimal performance

## Architecture

### Phase 1: File Processing
- Process multiple files in parallel using thread pool
- Use CUDA for email/URL normalization within each thread
- Create sorted chunks with intra-chunk deduplication
- Checkpoint progress after each file completion

### Phase 2: K-Way Merge
- Merge all sorted chunks simultaneously using priority queue
- Apply final deduplication during merge
- Stream output to prevent memory buildup
- Checkpoint merge progress by chunk ranges

## Configuration

Create `external_sort_config.json`:

```json
{
  "memory_usage_percent": 60.0,
  "chunk_size_mb": 512,
  "io_buffer_size_kb": 64,
  "processing_threads": 4,
  "enable_cuda": true,
  "cuda_batch_size": 100000,
  "cuda_memory_percent": 80.0,
  "temp_directory": "./temp/external_sort",
  "enable_compression": false,
  "merge_buffer_size_kb": 256,
  "case_sensitive": false,
  "normalize_urls": true,
  "email_only_usernames": false,
  "verbose": true
}
```

## Usage

### As Library

```rust
use tuonella_sift::external_sort::{ExternalSortConfig, sort_and_deduplicate};

#[tokio::main]
async fn main() -> Result<()> {
    let config = ExternalSortConfig::from_file("config.json")?;
    let input_files = vec!["file1.csv".into(), "file2.csv".into()];
    let output_file = "deduplicated.csv".into();
    
    let stats = sort_and_deduplicate(&input_files, &output_file, config).await?;
    
    println!("Processed {} records, {} unique", stats.total_records, stats.unique_records);
    Ok(())
}
```

### As Binary

```bash
cargo run --bin external_sort_example -- \
    --input ./input_directory \
    --output ./output/deduplicated.csv \
    --config external_sort_config.json \
    --verbose
```

## Performance Characteristics

| Dataset Size | Memory Usage | Disk Usage | Time Complexity | CUDA Speedup |
|--------------|--------------|------------|-----------------|---------------|
| 100GB        | ~30GB        | ~200GB     | O(N log N)      | 2-4x          |
| 500GB        | ~30GB        | ~1TB       | O(N log N)      | 2-4x          |
| 1TB+         | ~30GB        | ~2TB       | O(N log N)      | 2-4x          |

## Checkpointing

The utility automatically saves checkpoints during processing:

- **File Progress**: Tracks which files have been completely processed
- **Chunk Metadata**: Records created chunks and their locations
- **Merge Progress**: Tracks merge completion by chunk ranges
- **Resume Logic**: Automatically resumes from last complete state

## Error Handling

- **Graceful Shutdown**: Responds to CTRL+C with checkpoint save
- **Memory Pressure**: Automatically adjusts chunk sizes under memory pressure
- **Disk Space**: Monitors available disk space and warns before exhaustion
- **Corruption Recovery**: Validates chunk integrity before merge operations

## Testing

```bash
cargo test --lib external_sort
```

## Limitations

- Requires temporary disk space ~2x input size
- CUDA acceleration requires NVIDIA GPU with CUDA toolkit
- Single-machine processing (no distributed support)
- CSV format only (no other delimited formats)
