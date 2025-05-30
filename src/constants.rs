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