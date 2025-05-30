// Memory constants
pub const BYTES_PER_KB: usize = 1024;
pub const BYTES_PER_MB: usize = 1024 * 1024;
pub const BYTES_PER_GB: usize = 1024 * 1024 * 1024;
pub const PERCENT_100: f64 = 100.0;
pub const PERCENT_95: f64 = 0.95;

// Default values
pub const DEFAULT_CPU_SEGMENT_MIN_MB: usize = 16;
pub const DEFAULT_RECORD_ESTIMATED_BYTES: usize = 500;
pub const CPU_SEGMENT_MEMORY_DIVISOR: usize = 4;

// GPU constants
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_BUS_WIDTH_NORMALIZATION: f64 = 256.0;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_L2_CACHE_NORMALIZATION_MB: f64 = 4.0 * BYTES_PER_MB as f64;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_SEGMENT_BASE_SIZE_MB: f64 = 64.0;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_MEMORY_USAGE_PERCENT: f64 = 0.8;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_MEMORY_HEADROOM_PERCENT: f64 = 0.2;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_COMPUTE_CAPABILITY_FACTOR: f64 = 0.1;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_MIN_SEGMENT_SIZE_MB: usize = 16;
#[cfg(feature = "cuda")]
pub const CONSERVATIVE_RECORD_BYTES_MULTIPLIER: usize = 3;

// Batch constants
pub const DEFAULT_SAMPLE_SIZE: usize = 1000;
pub const DEFAULT_MAX_GPU_BATCH_SIZE: usize = 1_000_000;
pub const MAX_FIELD_LENGTH_BYTES: usize = 4096;
pub const BINARY_HEADER_SIZE_BYTES: usize = 16;

// Processing constants
pub const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
pub const VERBOSE_PROGRESS_INTERVAL_SECONDS: u64 = 5;

// Algorithm-specific memory allocation constants (as per docs/algorithm.md)
pub const ALGORITHM_RAM_ALLOCATION_PERCENT: f64 = 0.90;  // 90% of available RAM
pub const ALGORITHM_GPU_ALLOCATION_PERCENT: f64 = 0.90;  // 90% of available GPU memory
pub const MEMORY_SAFETY_MARGIN: f64 = 0.95;  // Additional safety margin
pub const DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS: usize = 1000;  // Check memory every N records

// Safety limits to prevent OOM during resource querying
pub const MAX_RAM_BUFFER_SIZE_GB: f64 = 8.0;  // Maximum 8GB RAM buffer
pub const MAX_GPU_BUFFER_SIZE_GB: f64 = 4.0;  // Maximum 4GB GPU buffer

// Dynamic chunk size adjustment constants (Section 4: Memory Management)
pub const MEMORY_PRESSURE_THRESHOLD_PERCENT: f64 = 80.0;  // Consider 80% as pressure threshold
pub const MEMORY_CRITICAL_THRESHOLD_PERCENT: f64 = 90.0;  // Consider 90% as critical threshold
pub const CHUNK_SIZE_REDUCTION_FACTOR: f64 = 0.75;  // Reduce chunk size by 25% under pressure
pub const CHUNK_SIZE_INCREASE_FACTOR: f64 = 1.25;  // Increase chunk size by 25% when memory is available
pub const MIN_CHUNK_SIZE_REDUCTION_LIMIT: f64 = 0.25;  // Don't reduce below 25% of original size
pub const MAX_CHUNK_SIZE_INCREASE_LIMIT: f64 = 2.0;  // Don't increase above 200% of original size
pub const CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS: usize = 5000;  // Wait N records between adjustments

// Error handling and recovery constants (Section 5: Error Handling)
pub const MAX_RETRY_ATTEMPTS: usize = 3;  // Maximum number of retry attempts for failed operations
pub const RETRY_DELAY_MS: u64 = 100;  // Initial delay between retries in milliseconds
pub const RETRY_BACKOFF_MULTIPLIER: f64 = 2.0;  // Exponential backoff multiplier
pub const CHUNK_SPLIT_FACTOR: f64 = 0.5;  // Split chunks to 50% of original size on error
pub const MIN_CHUNK_SIZE_RECORDS: usize = 10;  // Minimum chunk size in records
pub const ERROR_LOG_BUFFER_SIZE: usize = 1000;  // Buffer size for error logging
pub const RECOVERY_CHECKPOINT_INTERVAL: usize = 10000;  // Save recovery checkpoint every N records

// GPU processing constants for algorithm step 2.2
pub const GPU_CHUNK_PROCESSING_BATCH_SIZE: usize = 10000;  // Records per GPU batch
pub const GPU_TEMP_FILE_READ_CHUNK_SIZE_MB: usize = 64;    // MB to read from temp files at once
pub const GPU_STRING_BUFFER_PADDING: usize = 256;         // Extra bytes per string for GPU processing