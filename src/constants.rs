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

// Section 6: Performance Optimization Constants
// Double buffering for overlapping I/O and GPU processing
pub const DOUBLE_BUFFER_SIZE_RATIO: f64 = 0.5;  // Each buffer gets 50% of available memory
pub const BUFFER_SWAP_THRESHOLD_PERCENT: f64 = 80.0;  // Swap when buffer is 80% full
pub const ASYNC_IO_TIMEOUT_SECONDS: u64 = 30;  // Timeout for async I/O operations

// Performance monitoring and adaptive optimization
pub const PERFORMANCE_SAMPLE_WINDOW_RECORDS: usize = 10000;  // Sample window for performance metrics
pub const ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS: usize = 50000;  // Adjust parameters every N records
pub const MIN_THROUGHPUT_RECORDS_PER_SECOND: f64 = 1000.0;  // Minimum acceptable throughput
pub const THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 10.0;  // Minimum improvement to keep changes

// Batch processing optimizations
pub const WRITE_BATCH_SIZE_RECORDS: usize = 5000;  // Batch multiple records for single write
pub const WRITE_BUFFER_SIZE_MB: usize = 64;  // Write buffer size in MB
pub const MAX_WRITE_BATCH_SIZE_RECORDS: usize = 50000;  // Maximum write batch size

// Streaming and parallel processing
pub const PARALLEL_FILE_PROCESSING_THREADS: usize = 4;  // Number of parallel file processing threads
pub const STREAMING_CHUNK_SIZE_MB: usize = 128;  // Streaming chunk size for large files
pub const PARALLEL_IO_QUEUE_SIZE: usize = 16;  // Queue size for parallel I/O operations

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

// Username validation constants
pub const PRINTABLE_USERNAME_MIN_LENGTH: usize = 1;  // Minimum length for printable username
pub const PRINTABLE_USERNAME_MAX_LENGTH: usize = 320;  // Maximum length for printable username (RFC 5321 limit)

// GPU processing constants for algorithm step 2.2
pub const GPU_CHUNK_PROCESSING_BATCH_SIZE: usize = 10000;  // Records per GPU batch
pub const GPU_TEMP_FILE_READ_CHUNK_SIZE_MB: usize = 64;    // MB to read from temp files at once
pub const GPU_STRING_BUFFER_PADDING: usize = 256;         // Extra bytes per string for GPU processing

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
pub const MIN_FIELD_COUNT: usize = 3;                     // Minimum fields required for a valid record
pub const LONG_PASSWORD_HEURISTIC_LENGTH: usize = 50;     // Length threshold for password detection heuristic
pub const REVERSE_DOMAIN_MIN_LENGTH: usize = 30;          // Minimum length for reverse domain notation detection
pub const REVERSE_DOMAIN_MIN_PARTS: usize = 4;            // Minimum parts for reverse domain notation

// CSV header constants
pub const DEFAULT_USERNAME_HEADER: &str = "username";
pub const DEFAULT_PASSWORD_HEADER: &str = "password";
pub const DEFAULT_URL_HEADER: &str = "url";
pub const FIELD_NAME_TEMPLATE: &str = "field_{}";         // Template for additional field names

// Buffer and memory constants
pub const NEWLINE_BYTE: u8 = b'\n';                       // Newline character as byte

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
pub const CORE_FIELD_COUNT: usize = 3;                        // Core fields: username, password, URL
pub const USERNAME_FIELD_INDEX: usize = 0;                    // Index of username field in output
pub const PASSWORD_FIELD_INDEX: usize = 1;                    // Index of password field in output
pub const URL_FIELD_INDEX: usize = 2;                         // Index of URL field in output
pub const EXTRA_FIELDS_START_INDEX: usize = 3;                // Starting index for extra fields

// Buffer size calculation constants
pub const NEWLINE_SIZE: usize = 1;                            // Size of newline character in bytes

// Loop and iteration constants
pub const FIRST_RECORD_INDEX: usize = 0;                      // Index of first record for header writing
pub const TEMP_FILE_COUNT_INITIAL: usize = 0;                 // Initial temp file counter value
pub const CHUNK_SIZE_RESET_VALUE: usize = 0;                  // Value to reset chunk size counter

// Test configuration constants (used in unit tests)
pub const TEST_MAX_RAM_USAGE_GB: usize = 1;                   // Test RAM limit in GB
pub const TEST_CHUNK_SIZE_MB: usize = 1;                      // Test chunk size in MB
pub const TEST_RECORD_CHUNK_SIZE: usize = 10;                 // Test record chunk size
pub const TEST_MAX_MEMORY_RECORDS: usize = 1000;              // Test memory record limit

// Batch writer constants
pub const ZERO_DURATION_SECS: u64 = 0;                        // Zero seconds for duration
pub const ZERO_DURATION_NANOS: u32 = 0;                       // Zero nanoseconds for duration
pub const ZERO_FLOAT: f64 = 0.0;                              // Zero float value
pub const ZERO_COUNT: usize = 0;                              // Zero count value
pub const MIN_BATCH_SIZE: usize = 100;                        // Minimum batch size for writer
pub const MIN_OPERATIONS_FOR_OPTIMIZATION: usize = 3;         // Minimum operations before optimization
pub const LOW_THROUGHPUT_THRESHOLD: f64 = 1000.0;             // Low throughput threshold (records/sec)
pub const HIGH_THROUGHPUT_THRESHOLD: f64 = 5000.0;            // High throughput threshold (records/sec)
pub const IO_THROUGHPUT_THRESHOLD: f64 = 50.0;                // I/O throughput threshold (MB/sec)
pub const BATCH_SIZE_REDUCTION_FACTOR: f64 = 0.8;             // Factor to reduce batch size
pub const BATCH_SIZE_INCREASE_FACTOR: f64 = 1.2;              // Factor to increase batch size
pub const THROUGHPUT_EFFICIENCY_THRESHOLD: f64 = 10000.0;     // Throughput efficiency threshold
pub const BYTES_PER_MB_FLOAT: f64 = 1024.0 * 1024.0;         // Bytes per MB as float
pub const EFFICIENCY_COMPONENTS_COUNT: f64 = 3.0;             // Number of efficiency components

// Test constants for batch writer
pub const TEST_BATCH_SIZE: usize = 3;                         // Test batch size
pub const TEST_COMPLETENESS_SCORE: f32 = 3.0;                 // Test completeness score
pub const TEST_FIELD_COUNT: usize = 3;                        // Test field count