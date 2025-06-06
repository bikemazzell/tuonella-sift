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
pub const DEFAULT_GPU_MEMORY_USAGE_PERCENT: f64 = MULTIPLIER_80;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_MEMORY_HEADROOM_PERCENT: f64 = MULTIPLIER_20;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_COMPUTE_CAPABILITY_FACTOR: f64 = 0.1;
#[cfg(feature = "cuda")]
pub const DEFAULT_GPU_MIN_SEGMENT_SIZE_MB: usize = 16;
#[cfg(feature = "cuda")]
pub const CONSERVATIVE_RECORD_BYTES_MULTIPLIER: usize = 3;

// Memory allocation and pressure thresholds
pub const ALGORITHM_RAM_ALLOCATION_PERCENT: f64 = MULTIPLIER_90;
pub const ALGORITHM_GPU_ALLOCATION_PERCENT: f64 = MULTIPLIER_90;
pub const MEMORY_SAFETY_MARGIN: f64 = MULTIPLIER_95;
pub const DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS: usize = 1000;
pub const DOUBLE_BUFFER_SIZE_RATIO: f64 = MULTIPLIER_50;
pub const BUFFER_SWAP_THRESHOLD_PERCENT: f64 = PERCENT_80;
pub const ASYNC_IO_TIMEOUT_SECONDS: u64 = 30;
pub const MEMORY_PRESSURE_THRESHOLD_PERCENT: f64 = PERCENT_80;
pub const MEMORY_CRITICAL_THRESHOLD_PERCENT: f64 = PERCENT_90;
pub const LOW_MEMORY_THRESHOLD_FACTOR: f64 = MULTIPLIER_60;

// Chunk size adjustment factors
pub const CHUNK_SIZE_REDUCTION_FACTOR: f64 = 0.75;
pub const CHUNK_SIZE_INCREASE_FACTOR: f64 = 1.25;
pub const MIN_CHUNK_SIZE_REDUCTION_LIMIT: f64 = MULTIPLIER_25;
pub const MAX_CHUNK_SIZE_INCREASE_LIMIT: f64 = 2.0;
pub const CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS: usize = 5000;

// Batch constants
pub const DEFAULT_SAMPLE_SIZE: usize = 1000;
pub const DEFAULT_MAX_GPU_BATCH_SIZE: usize = 1_000_000;
pub const MAX_FIELD_LENGTH_BYTES: usize = 4096;
pub const BINARY_HEADER_SIZE_BYTES: usize = 16;

// Processing constants
pub const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
pub const VERBOSE_PROGRESS_INTERVAL_SECONDS: u64 = 5;

// Batch processing constants
pub const WRITE_BATCH_SIZE_RECORDS: usize = 5000;
pub const WRITE_BUFFER_SIZE_MB: usize = 64;
pub const MAX_WRITE_BATCH_SIZE_RECORDS: usize = 50000;
pub const MIN_BATCH_SIZE: usize = 100;
pub const MIN_OPERATIONS_FOR_OPTIMIZATION: usize = 3;
pub const LOW_THROUGHPUT_THRESHOLD: f64 = 1000.0;
pub const HIGH_THROUGHPUT_THRESHOLD: f64 = 5000.0;
pub const IO_THROUGHPUT_THRESHOLD: f64 = 50.0;
pub const BATCH_SIZE_REDUCTION_FACTOR: f64 = 0.8;
pub const BATCH_SIZE_INCREASE_FACTOR: f64 = 1.2;
pub const THROUGHPUT_EFFICIENCY_THRESHOLD: f64 = 10000.0;
pub const BYTES_PER_MB_FLOAT: f64 = MB_AS_F64;
pub const EFFICIENCY_COMPONENTS_COUNT: f64 = 3.0;

// Parallel processing constants
pub const PARALLEL_FILE_PROCESSING_THREADS: usize = 4;
pub const STREAMING_CHUNK_SIZE_MB: usize = 128;
pub const PARALLEL_IO_QUEUE_SIZE: usize = 16;

// Error handling and recovery constants
pub const MAX_RETRY_ATTEMPTS: usize = 3;
pub const RETRY_DELAY_MS: u64 = 100;
pub const RETRY_BACKOFF_MULTIPLIER: f64 = 2.0;
pub const CHUNK_SPLIT_FACTOR: f64 = MULTIPLIER_50;
pub const MIN_CHUNK_SIZE_RECORDS: usize = 10;
pub const ERROR_LOG_BUFFER_SIZE: usize = 1000;
pub const RECOVERY_CHECKPOINT_INTERVAL: usize = 10000;

// Validation constants
pub const PRINTABLE_USERNAME_MIN_LENGTH: usize = 1;
pub const PRINTABLE_USERNAME_MAX_LENGTH: usize = 320;
pub const MIN_FIELD_COUNT: usize = 3;
pub const LONG_PASSWORD_HEURISTIC_LENGTH: usize = 50;
pub const REVERSE_DOMAIN_MIN_LENGTH: usize = 30;
pub const REVERSE_DOMAIN_MIN_PARTS: usize = 4;
pub const MIN_DOMAIN_PART_LENGTH: usize = 2;
pub const MIN_DOMAIN_PARTS: usize = 2;
pub const MIN_EMAIL_LENGTH: usize = 5;

// Protocol and URL constants
pub const PROTOCOL_HTTP: &str = "http://";
pub const PROTOCOL_HTTPS: &str = "https://";
pub const PROTOCOL_ANDROID: &str = "android://";
pub const PROTOCOL_FTP: &str = "ftp://";
pub const PROTOCOL_MAILTO: &str = "mailto://";
pub const URL_WWW_PREFIX: &str = "www.";

// File extension constants
pub const CSV_EXTENSION: &str = "csv";
pub const TEMP_FILE_EXTENSION: &str = ".tmp";


// CSV header constants
pub const DEFAULT_USERNAME_HEADER: &str = "username";
pub const DEFAULT_PASSWORD_HEADER: &str = "password";
pub const DEFAULT_URL_HEADER: &str = "url";
pub const FIELD_NAME_TEMPLATE: &str = "field_{}";  // Template for additional field names

// Buffer and memory constants
pub const NEWLINE_BYTE: u8 = b'\n';  // Newline character as byte

// Error message constants
pub const ERROR_NO_PRINTABLE_CHARS: &str = "no printable characters";
pub const ERROR_NO_DELIMITER: &str = "no delimiter found";
pub const ERROR_NO_EMAIL: &str = "no email address found";
pub const ERROR_NO_VALID_USERNAME: &str = "no valid username found";
pub const ERROR_FEWER_THAN_MIN_FIELDS: &str = "fewer than 3 fields";
pub const ERROR_INVALID_FIELD_POSITIONS: &str = "invalid field positions detected";
pub const ERROR_URL_PROTOCOL_IN_NON_URL_FIELD: &str = "URL protocol found in non-URL field";
pub const ERROR_BUFFER_OVERFLOW: &str = "buffer overflow";

// Temporary file naming constants
pub const TEMP_FILE_PREFIX: &str = "temp_";
pub const FINAL_DEDUPLICATED_FILENAME: &str = "final_deduplicated.csv";
pub const VALIDATION_ERRORS_FILENAME: &str = "validation_errors.log";

// Field index constants for CSV processing
pub const CORE_FIELD_COUNT: usize = 3;  // Core fields: username, password, URL
pub const USERNAME_FIELD_INDEX: usize = 0;  // Index of username field in output
pub const PASSWORD_FIELD_INDEX: usize = 1;  // Index of password field in output
pub const URL_FIELD_INDEX: usize = 2;  // Index of URL field in output
pub const EXTRA_FIELDS_START_INDEX: usize = 3;  // Starting index for extra fields

// Test configuration constants
pub const TEST_CHUNK_SIZE_MB: usize = 1;  // Test chunk size in MB
pub const TEST_RECORD_CHUNK_SIZE: usize = 10;  // Test record chunk size
pub const TEST_MAX_MEMORY_RECORDS: usize = 1000;  // Test memory record limit
pub const TEST_SLEEP_DURATION_MS: u64 = 100;  // Test sleep duration in milliseconds
pub const TEST_SMALL_BYTES: usize = 512;  // Test small bytes size
pub const TEST_MEDIUM_BYTES: usize = 1536;  // Test medium bytes size
pub const TEST_TOTAL_WORK: usize = 100;  // Test total work amount
pub const TEST_PARTIAL_WORK: usize = 10;  // Test partial work amount
pub const TEST_BATCH_SIZE: usize = 100;  // Test batch size
pub const TEST_COMPLETENESS_SCORE: f32 = 0.8;  // Test completeness score
pub const TEST_FIELD_COUNT: usize = 5;  // Test field count

// Performance monitoring constants
pub const PERFORMANCE_SAMPLE_WINDOW_RECORDS: usize = 10000;
pub const ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS: usize = 50000;
pub const MIN_THROUGHPUT_RECORDS_PER_SECOND: f64 = 1000.0;
pub const THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 10.0;

// Memory and buffer management defaults
pub const DEFAULT_CHUNK_ADJUSTMENT_FACTOR: f64 = ONE_F64;
pub const MIN_CHUNK_SIZE_LIMIT: usize = BYTES_PER_KB;
pub const DEFAULT_GPU_MEMORY_FREE: usize = ZERO_USIZE;
pub const DEFAULT_GPU_MEMORY_TOTAL: usize = ZERO_USIZE;
pub const DEFAULT_GPU_PRESSURE: bool = false;
pub const BUFFER_SIZE_RESET_VALUE: usize = ZERO_USIZE;
pub const INITIAL_BUFFER_A_ACTIVE: bool = true;

// Test duration constants
pub const TEST_DURATION_ZERO_SECS: u64 = 0;
pub const TEST_DURATION_61_SECS: u64 = 61;
pub const TEST_DURATION_3661_SECS: u64 = 3661;
pub const TEST_RAM_LIMIT_GB: usize = 1;

// Formatting constants
pub const DECIMAL_PLACES: usize = 2;

// Performance monitoring thresholds
pub const MEMORY_PRESSURE_HIGH_THRESHOLD: f64 = MULTIPLIER_80;
pub const MEMORY_PRESSURE_LOW_THRESHOLD: f64 = MULTIPLIER_50;
pub const THROUGHPUT_REDUCTION_FACTOR: f64 = 0.7;
pub const IO_EFFICIENCY_HIGH_THRESHOLD: f64 = MULTIPLIER_60;
pub const CHUNK_SIZE_MIN_MB: usize = 64;
pub const CHUNK_SIZE_MAX_MB: usize = 2048;
pub const GPU_UTILIZATION_LOW_THRESHOLD: f64 = 70.0;
pub const GPU_UTILIZATION_HIGH_THRESHOLD: f64 = PERCENT_95;
pub const BATCH_SIZE_INCREASE_FACTOR_GPU: f64 = 1.2;
pub const BATCH_SIZE_DECREASE_FACTOR_GPU: f64 = 0.9;
pub const MIN_PARALLEL_THREADS: usize = NUMERIC_ONE;
pub const MAX_PARALLEL_THREADS: usize = NUMERIC_SIXTEEN;
pub const MIN_BATCH_SIZE_RECORDS: usize = 1000;
pub const MAX_BATCH_SIZE_RECORDS: usize = 100000;
pub const MIN_BUFFER_SIZE_MB: usize = 256;
pub const MAX_BUFFER_SIZE_MB: usize = 4096;

// Default optimization parameters
pub const DEFAULT_CHUNK_SIZE_MB: usize = 256;
pub const DEFAULT_BATCH_SIZE: usize = 10000;
pub const DEFAULT_PARALLEL_THREADS: usize = NUMERIC_FOUR;
pub const DEFAULT_BUFFER_SIZE_MB: usize = 512;
pub const MIN_THREAD_COUNT: usize = NUMERIC_ONE;
pub const MAX_THREAD_COUNT: usize = NUMERIC_THIRTY_TWO;
pub const PERFORMANCE_SAMPLE_WINDOW_DIVISOR: usize = 1000;
pub const PERCENTAGE_MULTIPLIER: f64 = PERCENT_100;


// Record scoring constants
pub const BASE_FIELD_SCORE: f64 = ONE_F64;
pub const FIELD_LENGTH_WEIGHT: f64 = 0.1;

// CUDA kernel configuration constants
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_BLOCK_SIZE: usize = 256;  // Default CUDA block size (threads per block)
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_GRID_DIM: u32 = 1;  // Default CUDA grid dimension
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_SHARED_MEM_BYTES: u32 = 0;  // Default shared memory bytes
#[cfg(feature = "cuda")]
pub const CUDA_VECTOR_SIZE: usize = 4;  // Vector size for CUDA operations
#[cfg(feature = "cuda")]
pub const DEFAULT_MIN_BUFFER_SIZE: usize = 64;  // Minimum buffer size for CUDA operations

// URL and protocol length constants
pub const MAX_URL_LENGTH: usize = 255;  // Maximum URL length for processing
pub const MAX_PROTOCOL_LENGTH: usize = 10;  // Maximum protocol length
pub const PROTOCOL_COUNT: usize = 5;  // Number of supported protocols
pub const PROTOCOL_HTTP_LENGTH: usize = 7;  // Length of "http://"
pub const PROTOCOL_HTTPS_LENGTH: usize = 8;  // Length of "https://"
pub const PROTOCOL_ANDROID_LENGTH: usize = 10;  // Length of "android://"
pub const PROTOCOL_FTP_LENGTH: usize = 6;  // Length of "ftp://"
pub const PROTOCOL_MAILTO_LENGTH: usize = 8;  // Length of "mailto://"
pub const WWW_PREFIX_LENGTH: usize = 4;  // Length of "www."

// CUDA optimization constants
#[cfg(feature = "cuda")]
pub const CUDA_WARP_SIZE: usize = 32;  // CUDA warp size
#[cfg(feature = "cuda")]
pub const CUDA_MAX_THREADS_PER_BLOCK: usize = 1024;  // Maximum threads per block
#[cfg(feature = "cuda")]
pub const CUDA_MEMORY_COALESCING_SIZE: usize = 128;  // Memory coalescing size in bytes
#[cfg(feature = "cuda")]
pub const CUDA_STREAM_COUNT: usize = 4;  // Number of CUDA streams for async operations
#[cfg(feature = "cuda")]
pub const CUDA_MEMORY_POOL_SIZE_MB: usize = 512;  // Memory pool size in MB
#[cfg(feature = "cuda")]
pub const CUDA_BATCH_MERGE_THRESHOLD: usize = 1000;  // Threshold for merging small batches

// Missing GPU constants for deduplication
pub const GPU_CHUNK_PROCESSING_BATCH_SIZE: usize = 50000;  // GPU chunk processing batch size
pub const GPU_TEMP_FILE_READ_CHUNK_SIZE_MB: usize = 128;  // GPU temp file read chunk size in MB

pub const MAX_RAM_BUFFER_SIZE_GB: f64 = 8.0;
pub const MAX_GPU_BUFFER_SIZE_GB: f64 = 8.0;

// SIMD processing constants
pub const SIMD_AVX512_WIDTH_BYTES: usize = 64;
pub const SIMD_AVX2_WIDTH_BYTES: usize = 32;
pub const SIMD_SSE_NEON_WIDTH_BYTES: usize = 16;
pub const SIMD_SCALAR_WIDTH_BYTES: usize = 1;
pub const SIMD_AVX512_CHUNK_SIZE: usize = 8192;
pub const SIMD_AVX2_CHUNK_SIZE: usize = 4096;
pub const SIMD_SSE_NEON_CHUNK_SIZE: usize = 2048;
pub const SIMD_SCALAR_CHUNK_SIZE: usize = 1024;
pub const SIMD_AVX2_PROCESS_SIZE: usize = 32;
pub const SIMD_NEON_PROCESS_SIZE: usize = 16;
pub const SIMD_CACHE_SIZE_LIMIT: usize = 8192;

// FNV-1a hash algorithm constants
pub const FNV1A_OFFSET_BASIS: u64 = 14695981039346656037;
pub const FNV1A_PRIME: u64 = 1099511628211;

// External sort constants
pub const EXTERNAL_SORT_MEMORY_FACTOR: f64 = 0.8;
pub const EXTERNAL_SORT_MIN_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
pub const EXTERNAL_SORT_BUFFER_SIZE_BYTES: usize = 64 * 1024 * 1024;
pub const EXTERNAL_SORT_MIN_THREADS: usize = 4;
pub const EXTERNAL_SORT_PROGRESS_INTERVAL: usize = 100_000;
pub const EXTERNAL_SORT_MERGE_PROGRESS_INTERVAL: usize = 1_000_000;
pub const EXTERNAL_SORT_RECORD_OVERHEAD_BYTES: usize = 64;
pub const EXTERNAL_SORT_DISK_SPACE_FACTOR: f64 = 2.2;
pub const EXTERNAL_SORT_MEMORY_THRESHOLD_FACTOR: f64 = 0.6;

// CUDA optimization constants
#[cfg(feature = "cuda")]
pub const CUDA_SHARED_MEMORY_SIZE_BYTES: u32 = 16384;
#[cfg(feature = "cuda")]
pub const CUDA_MAX_BLOCKS_PER_SM: u32 = 16;
#[cfg(feature = "cuda")]
pub const CUDA_UPPERCASE_MASK_A: u32 = 0x40404040;
#[cfg(feature = "cuda")]
pub const CUDA_UPPERCASE_MASK_Z: u32 = 0x5B5B5B5B;
#[cfg(feature = "cuda")]
pub const CUDA_LOWERCASE_OFFSET: u32 = 0x20202020;
#[cfg(feature = "cuda")]
pub const DEFAULT_CUDA_BATCH_SIZE_FALLBACK: usize = 1000;
#[cfg(feature = "cuda")]
pub const UTILIZATION_CALCULATION_FACTOR: f64 = 10.0;

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

// Buffer pool constants
pub const BUFFER_CAPACITY_MULTIPLIER: usize = 4;
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 8;
pub const ESTIMATED_RECORD_SIZE_BYTES: usize = 128;

// Async processing constants
pub const MIN_CONCURRENT_TASKS: usize = 1;
pub const MAX_CONCURRENT_TASKS: usize = 64;
pub const RESULT_CHANNEL_CAPACITY: usize = 1000;
pub const PROGRESS_REPORT_INTERVAL: usize = 100;

// Error handling constants
pub const ERROR_LOG_FLUSH_INTERVAL_SECS: u64 = 5;
pub const MAX_RECOVERY_CHECKPOINTS: usize = 10;
pub const MAX_ALLOCATION_HISTORY: usize = 1000;

// ASCII constants
pub const ASCII_CONTROL_LIMIT: u8 = 32;
pub const ASCII_PRINTABLE_LIMIT: u8 = 126;

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

// Memory division factors
pub const MEMORY_DIVISION_FACTOR_QUARTER: usize = 4;
pub const MEMORY_DIVISION_FACTOR_HALF: usize = 2;

// Threshold percentages for GPU utilization
pub const GPU_MEMORY_USAGE_MIN_THRESHOLD: f64 = 5.0;
pub const GPU_UTILIZATION_MAX_ESTIMATE: f64 = 85.0;


// String processing constants
pub const STRING_LENGTH_THRESHOLD: usize = NUMERIC_THIRTY_TWO;
pub const MAX_CHARS_PER_THREAD: usize = 256;

// Test memory values (bytes)
pub const TEST_GPU_TOTAL_MEMORY_8GB: usize = 8000000000;
pub const TEST_GPU_FREE_MEMORY_2GB: usize = 2000000000;

// GPU hardware limits
#[cfg(feature = "cuda")]
pub const MAX_REGISTERS_PER_SM_TYPICAL: u32 = 65536;

// Test values for CUDA device properties
#[cfg(feature = "cuda")]
pub const TEST_MAX_THREADS_PER_BLOCK: u32 = 1024;
#[cfg(feature = "cuda")]
pub const TEST_MAX_SHARED_MEMORY_PER_BLOCK: u32 = 49152;
#[cfg(feature = "cuda")]
pub const TEST_MAX_SHARED_MEMORY_PER_SM: u32 = 98304;
#[cfg(feature = "cuda")]
pub const TEST_MAX_BLOCKS_PER_SM: u32 = 16;
#[cfg(feature = "cuda")]
pub const TEST_MAX_THREADS_PER_SM: u32 = 2048;
#[cfg(feature = "cuda")]
pub const TEST_WARP_SIZE: u32 = 32;
#[cfg(feature = "cuda")]
pub const TEST_COMPUTE_CAPABILITY_MAJOR: u32 = 7;
#[cfg(feature = "cuda")]
pub const TEST_COMPUTE_CAPABILITY_MINOR: u32 = 5;
#[cfg(feature = "cuda")]
pub const TEST_MULTIPROCESSOR_COUNT: u32 = 20;
#[cfg(feature = "cuda")]
pub const TEST_MAX_REGISTERS_PER_BLOCK: u32 = 65536;

// Test workload values
#[cfg(feature = "cuda")]
pub const TEST_WORKLOAD_NUM_ELEMENTS: u32 = 10000;
#[cfg(feature = "cuda")]
pub const TEST_WORKLOAD_ELEMENT_SIZE_BYTES: u32 = 64;
#[cfg(feature = "cuda")]
pub const TEST_WORKLOAD_AVG_STRING_LENGTH: u32 = 32;
#[cfg(feature = "cuda")]
pub const TEST_WORKLOAD_MAX_STRING_LENGTH: u32 = 255;

// Additional decimal constants
pub const DECIMAL_ZERO: f64 = ZERO_F64;
pub const DECIMAL_ONE: f64 = ONE_F64;
pub const DECIMAL_ONE_POINT_TWO: f64 = 1.2;
pub const DECIMAL_ZERO_POINT_EIGHT: f64 = MULTIPLIER_80;
pub const DECIMAL_ZERO_POINT_SIX: f64 = MULTIPLIER_60;

// Memory sizes for testing (bytes)
pub const TEST_MEMORY_8GB: usize = 8000000000;
pub const TEST_MEMORY_2GB: usize = 2000000000;
pub const TEST_MEMORY_6GB: usize = 6000000000;
pub const TEST_MEMORY_4GB: usize = 4000000000;
pub const TEST_MEMORY_500MB: usize = 500000000;

// Memory sizes (MB)
pub const MEMORY_64MB: usize = 64;
pub const MEMORY_16MB: usize = 16;
pub const MEMORY_256MB: usize = 256;

// Array/collection index constants
pub const INDEX_ZERO: usize = ZERO_USIZE;
pub const INDEX_ONE: usize = NUMERIC_ONE;

