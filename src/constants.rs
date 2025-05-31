// Basic numeric constants
pub const ZERO_USIZE: usize = 0;
pub const ZERO_U32: u32 = 0;
pub const ZERO_U64: u64 = 0;
pub const ZERO_F64: f64 = 0.0;

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
pub const PERCENT_95: f64 = 95.0;
pub const PERCENT_90: f64 = 90.0;
pub const PERCENT_80: f64 = 80.0;
pub const PERCENT_60: f64 = 60.0;
pub const PERCENT_50: f64 = 50.0;
pub const PERCENT_25: f64 = 25.0;

// Percentage multipliers (0.0-1.0)
pub const MULTIPLIER_95: f64 = 0.95;  // Memory safety margin
pub const MULTIPLIER_90: f64 = 0.90;  // RAM/GPU allocation
pub const MULTIPLIER_80: f64 = 0.80;  // Memory pressure threshold
pub const MULTIPLIER_60: f64 = 0.60;  // Low memory threshold
pub const MULTIPLIER_50: f64 = 0.50;  // Buffer split factor
pub const MULTIPLIER_25: f64 = 0.25;  // Minimum chunk size
pub const MULTIPLIER_20: f64 = 0.20;  // GPU memory headroom

// Time constants
pub const SECONDS_PER_MINUTE: u64 = 60;
pub const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * 60;

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

// Algorithm-specific memory allocation constants
pub const ALGORITHM_RAM_ALLOCATION_PERCENT: f64 = MULTIPLIER_90;  // 90% of available RAM
pub const ALGORITHM_GPU_ALLOCATION_PERCENT: f64 = MULTIPLIER_90;  // 90% of available GPU memory
pub const MEMORY_SAFETY_MARGIN: f64 = MULTIPLIER_95;  // Additional safety margin
pub const DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS: usize = 1000;  // Check memory every N records

// Safety limits to prevent OOM during resource querying
pub const MAX_RAM_BUFFER_SIZE_GB: f64 = 8.0;  // Maximum 8GB RAM buffer
pub const MAX_GPU_BUFFER_SIZE_GB: f64 = 4.0;  // Maximum 4GB GPU buffer

// Double buffering for overlapping I/O and GPU processing
pub const DOUBLE_BUFFER_SIZE_RATIO: f64 = MULTIPLIER_50;  // Each buffer gets 50% of available memory
pub const BUFFER_SWAP_THRESHOLD_PERCENT: f64 = PERCENT_80;  // Swap when buffer is 80% full
pub const ASYNC_IO_TIMEOUT_SECONDS: u64 = 30;  // Timeout for async I/O operations

// Memory pressure thresholds
pub const MEMORY_PRESSURE_THRESHOLD_PERCENT: f64 = PERCENT_80;  // Consider 80% as pressure threshold
pub const MEMORY_CRITICAL_THRESHOLD_PERCENT: f64 = PERCENT_90;  // Consider 90% as critical threshold
pub const LOW_MEMORY_THRESHOLD_FACTOR: f64 = 0.6;  // 60% memory threshold

// Chunk size adjustment factors
pub const CHUNK_SIZE_REDUCTION_FACTOR: f64 = 0.75;  // Reduce chunk size by 25% under pressure
pub const CHUNK_SIZE_INCREASE_FACTOR: f64 = 1.25;  // Increase chunk size by 25% when memory is available
pub const MIN_CHUNK_SIZE_REDUCTION_LIMIT: f64 = MULTIPLIER_25;  // Don't reduce below 25% of original size
pub const MAX_CHUNK_SIZE_INCREASE_LIMIT: f64 = 2.0;  // Don't increase above 200% of original size
pub const CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS: usize = 5000;  // Wait N records between adjustments

// Batch constants
pub const DEFAULT_SAMPLE_SIZE: usize = 1000;
pub const DEFAULT_MAX_GPU_BATCH_SIZE: usize = 1_000_000;
pub const MAX_FIELD_LENGTH_BYTES: usize = 4096;
pub const BINARY_HEADER_SIZE_BYTES: usize = 16;

// Processing constants
pub const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
pub const VERBOSE_PROGRESS_INTERVAL_SECONDS: u64 = 5;

// Batch writer constants
pub const WRITE_BATCH_SIZE_RECORDS: usize = 5000;  // Batch multiple records for single write
pub const WRITE_BUFFER_SIZE_MB: usize = 64;  // Write buffer size in MB
pub const MAX_WRITE_BATCH_SIZE_RECORDS: usize = 50000;  // Maximum write batch size
pub const MIN_BATCH_SIZE: usize = 100;  // Minimum batch size for writer
pub const MIN_OPERATIONS_FOR_OPTIMIZATION: usize = 3;  // Minimum operations before optimization
pub const LOW_THROUGHPUT_THRESHOLD: f64 = 1000.0;  // Low throughput threshold (records/sec)
pub const HIGH_THROUGHPUT_THRESHOLD: f64 = 5000.0;  // High throughput threshold (records/sec)
pub const IO_THROUGHPUT_THRESHOLD: f64 = 50.0;  // I/O throughput threshold (MB/sec)
pub const BATCH_SIZE_REDUCTION_FACTOR: f64 = 0.8;  // Factor to reduce batch size
pub const BATCH_SIZE_INCREASE_FACTOR: f64 = 1.2;  // Factor to increase batch size
pub const THROUGHPUT_EFFICIENCY_THRESHOLD: f64 = 10000.0;  // Throughput efficiency threshold
pub const BYTES_PER_MB_FLOAT: f64 = MB_AS_F64;  // Bytes per MB as float
pub const EFFICIENCY_COMPONENTS_COUNT: f64 = 3.0;  // Number of efficiency components

// Streaming and parallel processing
pub const PARALLEL_FILE_PROCESSING_THREADS: usize = 4;  // Number of parallel file processing threads
pub const STREAMING_CHUNK_SIZE_MB: usize = 128;  // Streaming chunk size for large files
pub const PARALLEL_IO_QUEUE_SIZE: usize = 16;  // Queue size for parallel I/O operations

// Error handling and recovery constants
pub const MAX_RETRY_ATTEMPTS: usize = 3;  // Maximum number of retry attempts for failed operations
pub const RETRY_DELAY_MS: u64 = 100;  // Initial delay between retries in milliseconds
pub const RETRY_BACKOFF_MULTIPLIER: f64 = 2.0;  // Exponential backoff multiplier
pub const CHUNK_SPLIT_FACTOR: f64 = 0.5;  // Split chunks to 50% of original size on error
pub const MIN_CHUNK_SIZE_RECORDS: usize = 10;  // Minimum chunk size in records
pub const ERROR_LOG_BUFFER_SIZE: usize = 1000;  // Buffer size for error logging
pub const RECOVERY_CHECKPOINT_INTERVAL: usize = 10000;  // Save recovery checkpoint every N records

// Username validation constants
pub const PRINTABLE_USERNAME_MIN_LENGTH: usize = 1;  // Minimum length for printable username
pub const PRINTABLE_USERNAME_MAX_LENGTH: usize = 320;  // Maximum length for printable username (RFC 5321 limit)

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

// Field validation constants
pub const MIN_FIELD_COUNT: usize = 3;  // Minimum fields required for a valid record
pub const LONG_PASSWORD_HEURISTIC_LENGTH: usize = 50;  // Length threshold for password detection heuristic
pub const REVERSE_DOMAIN_MIN_LENGTH: usize = 30;  // Minimum length for reverse domain notation detection
pub const REVERSE_DOMAIN_MIN_PARTS: usize = 4;  // Minimum parts for reverse domain notation

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
pub const TEST_MAX_RAM_USAGE_GB: usize = 1;  // Test RAM limit in GB
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

// Performance monitoring and adaptive optimization
pub const PERFORMANCE_SAMPLE_WINDOW_RECORDS: usize = 10000;  // Sample window for performance metrics
pub const ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS: usize = 50000;  // Adjust parameters every N records
pub const MIN_THROUGHPUT_RECORDS_PER_SECOND: f64 = 1000.0;  // Minimum acceptable throughput
pub const THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 10.0;  // Minimum improvement to keep changes

// Memory management defaults
pub const DEFAULT_CHUNK_ADJUSTMENT_FACTOR: f64 = 1.0;  // No adjustment
pub const MIN_CHUNK_SIZE_LIMIT: usize = 1024;  // 1KB minimum chunk size
pub const DEFAULT_GPU_MEMORY_FREE: usize = 0;  // Default free GPU memory
pub const DEFAULT_GPU_MEMORY_TOTAL: usize = 0;  // Default total GPU memory
pub const DEFAULT_GPU_PRESSURE: bool = false;  // Default GPU pressure state

// Buffer management
pub const BUFFER_SIZE_RESET_VALUE: usize = 0;
pub const INITIAL_BUFFER_A_ACTIVE: bool = true;

// Test duration constants
pub const TEST_DURATION_ZERO_SECS: u64 = 0;
pub const TEST_DURATION_61_SECS: u64 = 61;
pub const TEST_DURATION_3661_SECS: u64 = 3661;
pub const TEST_RAM_LIMIT_GB: usize = 1;

// Formatting constants
pub const DECIMAL_PLACES: usize = 2;

// Performance monitoring thresholds
pub const MEMORY_PRESSURE_HIGH_THRESHOLD: f64 = 0.8;  // 80% memory pressure threshold
pub const MEMORY_PRESSURE_LOW_THRESHOLD: f64 = 0.5;  // 50% memory pressure threshold
pub const THROUGHPUT_REDUCTION_FACTOR: f64 = 0.7;  // 70% throughput reduction factor
pub const IO_EFFICIENCY_HIGH_THRESHOLD: f64 = 0.6;  // 60% I/O efficiency threshold
pub const CHUNK_SIZE_MIN_MB: usize = 64;  // Minimum chunk size in MB
pub const CHUNK_SIZE_MAX_MB: usize = 2048;  // Maximum chunk size in MB
pub const GPU_UTILIZATION_LOW_THRESHOLD: f64 = 70.0;  // Low GPU utilization threshold
pub const GPU_UTILIZATION_HIGH_THRESHOLD: f64 = 95.0;  // High GPU utilization threshold
pub const BATCH_SIZE_INCREASE_FACTOR_GPU: f64 = 1.2;  // Increase batch size by 20% when GPU utilization is low
pub const BATCH_SIZE_DECREASE_FACTOR_GPU: f64 = 0.9;  // Decrease batch size by 10% when GPU utilization is high
pub const MIN_PARALLEL_THREADS: usize = 1;  // Minimum number of parallel threads
pub const MAX_PARALLEL_THREADS: usize = 16;  // Maximum number of parallel threads
pub const MIN_BATCH_SIZE_RECORDS: usize = 1000;  // Minimum batch size in records
pub const MAX_BATCH_SIZE_RECORDS: usize = 100000;  // Maximum batch size in records
pub const MIN_BUFFER_SIZE_MB: usize = 256;  // Minimum buffer size in MB
pub const MAX_BUFFER_SIZE_MB: usize = 4096;  // Maximum buffer size in MB

// Default optimization parameters
pub const DEFAULT_CHUNK_SIZE_MB: usize = 256;  // Default chunk size in MB
pub const DEFAULT_BATCH_SIZE: usize = 10000;  // Default batch size
pub const DEFAULT_PARALLEL_THREADS: usize = 4;  // Default number of parallel threads
pub const DEFAULT_BUFFER_SIZE_MB: usize = 512;  // Default buffer size in MB
pub const MIN_THREAD_COUNT: usize = 1;  // Minimum thread count
pub const MAX_THREAD_COUNT: usize = 32;  // Maximum thread count
pub const PERFORMANCE_SAMPLE_WINDOW_DIVISOR: usize = 1000;  // Divisor for sample window size
pub const PERCENTAGE_MULTIPLIER: f64 = 100.0;  // Multiplier to convert ratio to percentage

// Record validation constants
pub const MIN_DOMAIN_PART_LENGTH: usize = 2;  // Minimum length for each domain part
pub const MIN_DOMAIN_PARTS: usize = 2;  // Minimum number of domain parts (e.g. example.com)

// Record scoring constants
pub const BASE_FIELD_SCORE: f64 = 1.0;  // Base score for each field
pub const FIELD_LENGTH_WEIGHT: f64 = 0.1;  // Weight for field length in scoring

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