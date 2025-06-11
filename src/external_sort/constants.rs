pub const DEFAULT_MEMORY_USAGE_PERCENT: f64 = 60.0;
pub const DEFAULT_CHUNK_SIZE_MB: usize = 512;
pub const DEFAULT_IO_BUFFER_SIZE_KB: usize = 64;
pub const DEFAULT_PROCESSING_THREADS: usize = 4;
pub const DEFAULT_CUDA_BATCH_SIZE: usize = 100000;
pub const DEFAULT_MERGE_BUFFER_SIZE_KB: usize = 256;

pub const MIN_MEMORY_USAGE_PERCENT: f64 = 10.0;
pub const MAX_MEMORY_USAGE_PERCENT: f64 = 90.0;
pub const MIN_CHUNK_SIZE_MB: usize = 64;
pub const MAX_CHUNK_SIZE_MB: usize = 4096;
pub const MIN_PROCESSING_THREADS: usize = 1;
pub const MAX_PROCESSING_THREADS: usize = 32;
pub const MIN_CUDA_BATCH_SIZE: usize = 1000;
pub const MAX_CUDA_BATCH_SIZE: usize = 1000000;

pub const BYTES_PER_KB: usize = 1024;
pub const BYTES_PER_MB: usize = 1024 * 1024;
pub const KB_PER_MB: usize = 1024;

pub const CHECKPOINT_FILE_NAME: &str = "external_sort_checkpoint.json";
pub const CHUNK_FILE_PREFIX: &str = "chunk_";
pub const CHUNK_FILE_EXTENSION: &str = ".csv";
pub const TEMP_DIR_NAME: &str = "external_sort_temp";

pub const PROGRESS_REPORT_INTERVAL_RECORDS: usize = 100000;
pub const CHECKPOINT_SAVE_INTERVAL_MS: u64 = 30000;
pub const MEMORY_CHECK_INTERVAL_RECORDS: usize = 10000;

pub const CSV_FIELD_SEPARATOR: char = ',';
pub const CSV_QUOTE_CHAR: char = '"';
pub const CSV_ESCAPE_CHAR: char = '\\';

pub const MIN_RECORD_FIELDS: usize = 2;
pub const MAX_FIELD_LENGTH: usize = 4096;
pub const ESTIMATED_RECORD_SIZE_BYTES: usize = 256;

pub const CUDA_STREAM_COUNT: usize = 4;
pub const CUDA_MEMORY_USAGE_PERCENT: f64 = 80.0;
pub const CUDA_MAX_STRING_LENGTH: usize = 1024;

pub const MERGE_HEAP_INITIAL_CAPACITY: usize = 1024;
pub const OUTPUT_BUFFER_SIZE_KB: usize = 512;

pub const SHUTDOWN_CHECK_INTERVAL_MS: u64 = 100;
pub const GRACEFUL_SHUTDOWN_TIMEOUT_MS: u64 = 5000;
