// Unified constants file for tuonella-sift
// Contains all constants used across external_sort and CUDA modules

// CSV extension
pub const CSV_EXTENSION: &str = "csv";

// External Sort Configuration Constants
pub const DEFAULT_MEMORY_USAGE_PERCENT: f64 = 60.0;
pub const DEFAULT_CHUNK_SIZE_MB: usize = 512;
pub const DEFAULT_IO_BUFFER_SIZE_KB: usize = 64;
pub const DEFAULT_PROCESSING_THREADS: usize = 4;
pub const DEFAULT_CUDA_BATCH_SIZE: usize = 100000;
pub const DEFAULT_MERGE_BUFFER_SIZE_KB: usize = 256;
pub const DEFAULT_MERGE_PROCESS_INTERVAL_SECS: u64 = 10;

// External Sort Validation Limits
pub const MIN_MEMORY_USAGE_PERCENT: f64 = 10.0;
pub const MAX_MEMORY_USAGE_PERCENT: f64 = 90.0;
pub const MIN_CHUNK_SIZE_MB: usize = 64;
pub const MAX_CHUNK_SIZE_MB: usize = 4096;
pub const MIN_PROCESSING_THREADS: usize = 1;
pub const MAX_PROCESSING_THREADS: usize = 32;
pub const MIN_CUDA_BATCH_SIZE: usize = 1000;
pub const MAX_CUDA_BATCH_SIZE: usize = 1000000;

// External Sort File and Directory Names
pub const CHECKPOINT_FILE_NAME: &str = "external_sort_checkpoint.json";
pub const CHUNK_FILE_PREFIX: &str = "chunk_";
pub const CHUNK_FILE_EXTENSION: &str = ".csv";
pub const TEMP_DIR_NAME: &str = "external_sort_temp";

// External Sort Processing Constants
pub const PROGRESS_REPORT_INTERVAL_RECORDS: usize = 100000;
pub const CHECKPOINT_SAVE_INTERVAL_MS: u64 = 30000;
pub const MEMORY_CHECK_INTERVAL_RECORDS: usize = 10000;

// CSV Processing Constants
pub const CSV_FIELD_SEPARATOR: char = ',';
pub const CSV_QUOTE_CHAR: char = '"';
pub const CSV_ESCAPE_CHAR: char = '\\';
pub const MIN_RECORD_FIELDS: usize = 2;
pub const MAX_FIELD_LENGTH: usize = 4096;
pub const ESTIMATED_RECORD_SIZE_BYTES: usize = 256;

// External Sort CUDA Constants
pub const CUDA_STREAM_COUNT: usize = 4;
pub const CUDA_MEMORY_USAGE_PERCENT: f64 = 80.0;
pub const CUDA_MAX_STRING_LENGTH: usize = 1024;

// External Sort Merge Constants
pub const MERGE_HEAP_INITIAL_CAPACITY: usize = 1024;
pub const OUTPUT_BUFFER_SIZE_KB: usize = 512;

// External Sort Shutdown Constants
pub const SHUTDOWN_CHECK_INTERVAL_MS: u64 = 100;
pub const GRACEFUL_SHUTDOWN_TIMEOUT_MS: u64 = 5000;

// Derived constant for external_sort
pub const KB_PER_MB: usize = 1024;

// Basic numeric constants
pub const ZERO_USIZE: usize = 0;
pub const ZERO_U32: u32 = 0;
pub const ZERO_U64: u64 = 0;
pub const ZERO_F64: f64 = 0.0;
pub const ONE_F64: f64 = 1.0;

// Memory size constants (integer)
pub const BYTES_PER_KB: usize = 1024;
pub const BYTES_PER_MB: usize = BYTES_PER_KB * 1024;
pub const BYTES_PER_GB: usize = BYTES_PER_MB * 1024;
pub const BYTES_PER_TB: usize = BYTES_PER_GB * 1024;

// Memory size constants (float)
pub const KB_AS_F64: f64 = 1024.0;
pub const MB_AS_F64: f64 = KB_AS_F64 * 1024.0;
pub const GB_AS_F64: f64 = MB_AS_F64 * 1024.0;
pub const TB_AS_F64: f64 = GB_AS_F64 * 1024.0;

// Percentage constants
pub const PERCENT_100: f64 = 100.0;
pub const PERCENT_98: f64 = 98.0;
pub const PERCENT_95: f64 = 95.0;
pub const PERCENT_90: f64 = 90.0;
pub const PERCENT_80: f64 = 80.0;
pub const PERCENT_75: f64 = 75.0;
pub const PERCENT_60: f64 = 60.0;
pub const PERCENT_50: f64 = 50.0;
pub const PERCENT_25: f64 = 25.0;

// Percentage multipliers (0.0-1.0)
pub const MULTIPLIER_95: f64 = 0.95;
pub const MULTIPLIER_90: f64 = 0.90;
pub const MULTIPLIER_80: f64 = 0.80;
pub const MULTIPLIER_60: f64 = 0.60;
pub const MULTIPLIER_50: f64 = 0.50;
pub const MULTIPLIER_25: f64 = 0.25;
pub const MULTIPLIER_20: f64 = 0.20;

// Time constants
pub const SECONDS_PER_MINUTE: u64 = 60;
pub const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * 60;
pub const ONE_HOUR_SECONDS: u64 = 3600;

// Duration constants
pub const ZERO_DURATION_SECS: u64 = 0;
pub const ZERO_DURATION_NANOS: u32 = 0;

// Memory pressure thresholds
pub const MEMORY_PRESSURE_THRESHOLD_PERCENT: f64 = PERCENT_80;
pub const MEMORY_CRITICAL_THRESHOLD_PERCENT: f64 = PERCENT_90;
pub const LOW_MEMORY_THRESHOLD_FACTOR: f64 = MULTIPLIER_60;

// CUDA constants
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_BLOCK_SIZE: usize = 256;
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_GRID_DIM: u32 = 1;
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_SHARED_MEM_BYTES: u32 = 0;
#[cfg(feature = "cuda")]
pub const CUDA_VECTOR_SIZE: usize = 4;
#[cfg(feature = "cuda")]
pub const DEFAULT_MIN_BUFFER_SIZE: usize = 64;
#[cfg(feature = "cuda")]
pub const CUDA_MEMORY_POOL_SIZE_MB: usize = 512;
#[cfg(feature = "cuda")]
pub const CUDA_WARP_SIZE: usize = 32;
#[cfg(feature = "cuda")]
pub const CUDA_MAX_THREADS_PER_BLOCK: usize = 1024;
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_BATCH_SIZE_FALLBACK: usize = 1000;
#[cfg(feature = "cuda")]
pub const GPU_UTILIZATION_MAX_ESTIMATE: f64 = 85.0;
#[cfg(feature = "cuda")]
pub const CUDA_SHARED_MEMORY_SIZE_BYTES: u32 = 16384;
#[cfg(feature = "cuda")]
pub const MAX_REGISTERS_PER_SM_TYPICAL: u32 = 65536;

// Pinned memory constants
#[cfg(feature = "cuda")]
pub const PINNED_BUFFER_SMALL_BYTES: usize = 65536;
#[cfg(feature = "cuda")]
pub const PINNED_BUFFER_MEDIUM_BYTES: usize = 1048576;
#[cfg(feature = "cuda")]
pub const PINNED_BUFFER_LARGE_BYTES: usize = 16777216;
#[cfg(feature = "cuda")]
pub const PINNED_BUFFER_XLARGE_BYTES: usize = 67108864;
#[cfg(feature = "cuda")]
pub const PINNED_BUFFER_MAX_AGE_SECS: u64 = 3600;
#[cfg(feature = "cuda")]
pub const PINNED_POOL_SIZE_SMALL: usize = 64;
#[cfg(feature = "cuda")]
pub const PINNED_POOL_SIZE_MEDIUM: usize = 32;
#[cfg(feature = "cuda")]
pub const PINNED_POOL_SIZE_LARGE: usize = 16;
#[cfg(feature = "cuda")]
pub const PINNED_POOL_SIZE_XLARGE: usize = 8;
#[cfg(feature = "cuda")]
pub const FRAGMENTATION_THRESHOLD_PERCENT: f64 = 25.0;

// Adaptive memory management constants
#[cfg(feature = "cuda")]
pub const MEMORY_HISTORY_MAX_SIZE: usize = 50;
#[cfg(feature = "cuda")]
pub const MEMORY_PREDICTION_WINDOW_SECS: u64 = 30;
#[cfg(feature = "cuda")]
pub const MEMORY_ADJUSTMENT_COOLDOWN_SECS: u64 = 10;
#[cfg(feature = "cuda")]
pub const MEMORY_USAGE_INCREASE_THRESHOLD: f64 = 70.0;
#[cfg(feature = "cuda")]
pub const MEMORY_USAGE_MAINTAIN_THRESHOLD: f64 = 85.0;
#[cfg(feature = "cuda")]
pub const MEMORY_USAGE_REDUCE_THRESHOLD: f64 = 95.0;
#[cfg(feature = "cuda")]
pub const MEMORY_INCREASE_FACTOR: f64 = 1.2;
#[cfg(feature = "cuda")]
pub const MEMORY_REDUCE_FACTOR: f64 = 0.8;
#[cfg(feature = "cuda")]
pub const MEMORY_EMERGENCY_FACTOR: f64 = 0.6;

// Additional numeric constants for CUDA modules
pub const NUMERIC_ONE: usize = 1;
pub const NUMERIC_TWO: usize = 2;
pub const NUMERIC_THREE: usize = 3;
pub const NUMERIC_FOUR: usize = 4;
pub const NUMERIC_FIVE: usize = 5;
pub const NUMERIC_EIGHT: usize = 8;
pub const NUMERIC_TEN: usize = 10;
pub const NUMERIC_SIXTEEN: usize = 16;
pub const NUMERIC_THIRTY_TWO: usize = 32;
pub const NUMERIC_SIXTY_FOUR: usize = 64;

// Index constants
pub const INDEX_ZERO: usize = 0;
pub const INDEX_ONE: usize = 1;

// Decimal constants
pub const DECIMAL_ZERO: f64 = 0.0;
pub const DECIMAL_ONE: f64 = 1.0;

// Error handling constants
#[cfg(feature = "cuda")]
pub const MAX_ALLOCATION_HISTORY: usize = 1000;

// Memory sizes (MB)
pub const MEMORY_64MB: usize = 64;
pub const MEMORY_16MB: usize = 16;
pub const MEMORY_256MB: usize = 256;

// Test memory values (bytes)
#[cfg(test)]
pub const TEST_MEMORY_8GB: usize = 8000000000;
#[cfg(test)]
pub const TEST_MEMORY_2GB: usize = 2000000000;
#[cfg(test)]
pub const TEST_MEMORY_6GB: usize = 6000000000;
#[cfg(test)]
pub const TEST_MEMORY_4GB: usize = 4000000000;
#[cfg(test)]
pub const TEST_MEMORY_500MB: usize = 500000000;

// Test values for CUDA device properties
#[cfg(test)]
pub const TEST_MAX_THREADS_PER_BLOCK: u32 = 1024;
#[cfg(test)]
pub const TEST_MAX_SHARED_MEMORY_PER_BLOCK: u32 = 49152;
#[cfg(test)]
pub const TEST_MAX_SHARED_MEMORY_PER_SM: u32 = 98304;
#[cfg(test)]
pub const TEST_MAX_BLOCKS_PER_SM: u32 = 16;
#[cfg(test)]
pub const TEST_MAX_THREADS_PER_SM: u32 = 2048;
#[cfg(test)]
pub const TEST_WARP_SIZE: u32 = 32;
#[cfg(test)]
pub const TEST_COMPUTE_CAPABILITY_MAJOR: u32 = 7;
#[cfg(test)]
pub const TEST_COMPUTE_CAPABILITY_MINOR: u32 = 5;
#[cfg(test)]
pub const TEST_MULTIPROCESSOR_COUNT: u32 = 20;
#[cfg(test)]
pub const TEST_MAX_REGISTERS_PER_BLOCK: u32 = 65536;