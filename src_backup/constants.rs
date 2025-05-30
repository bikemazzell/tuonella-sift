pub const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
pub const VERBOSE_PROGRESS_INTERVAL_SECONDS: u64 = 5;
pub const BYTES_PER_GB: f64 = 1_073_741_824.0;
pub const BYTES_PER_KB: f64 = 1024.0;
pub const PERCENT_100: f64 = 100.0;
pub const PERCENT_95: f64 = 0.95;

pub const DEFAULT_SAMPLE_SIZE: usize = 100;
pub const DEFAULT_RECORD_ESTIMATED_BYTES: usize = 200;
pub const DEFAULT_MAX_GPU_BATCH_SIZE: usize = 100000;
pub const DEFAULT_CPU_SEGMENT_MIN_MB: usize = 64;
pub const DEFAULT_MAX_URL_LENGTH_FOR_FAST_NORMALIZATION: usize = 512;

// Memory and processing constants
pub const BYTES_PER_MB: usize = 1024 * 1024;
pub const MAX_FIELD_LENGTH_BYTES: usize = 1024 * 1024; // 1MB max per field
pub const CONSERVATIVE_RECORD_BYTES_MULTIPLIER: usize = 3;
pub const CPU_SEGMENT_MEMORY_DIVISOR: usize = 4;
pub const BINARY_HEADER_SIZE_BYTES: usize = 6;

// GPU-related constants
pub const DEFAULT_GPU_BUS_WIDTH_NORMALIZATION: f64 = 256.0;
pub const DEFAULT_GPU_L2_CACHE_NORMALIZATION_MB: f64 = 2.0 * 1024.0 * 1024.0;
pub const DEFAULT_GPU_SEGMENT_BASE_SIZE_MB: f64 = 128.0 * 1024.0 * 1024.0;
pub const DEFAULT_GPU_MEMORY_USAGE_PERCENT: f64 = 0.2;
pub const DEFAULT_GPU_MEMORY_HEADROOM_PERCENT: f64 = 0.8;
pub const DEFAULT_GPU_COMPUTE_CAPABILITY_FACTOR: f64 = 2.0;
pub const DEFAULT_GPU_MIN_SEGMENT_SIZE_MB: usize = 1;

// Test constants for CUDA processor tests
#[cfg(test)]
pub const TEST_TOTAL_MEMORY: usize = 8 * 1024 * 1024 * 1024; // 8GB
#[cfg(test)]
pub const TEST_FREE_MEMORY: usize = 8 * 1024 * 1024 * 1024; // 8GB (match total for tests)
#[cfg(test)]
pub const TEST_COMPUTE_CAPABILITY_MAJOR: i32 = 7;
#[cfg(test)]
pub const TEST_COMPUTE_CAPABILITY_MINOR: i32 = 5;
#[cfg(test)]
pub const TEST_MAX_THREADS_PER_BLOCK: i32 = 1024;
#[cfg(test)]
pub const TEST_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 48 * 1024; // 48KB
#[cfg(test)]
pub const TEST_MEMORY_BUS_WIDTH: i32 = 256;
#[cfg(test)]
pub const TEST_L2_CACHE_SIZE: i32 = 4 * 1024 * 1024; // 4MB
#[cfg(test)]
pub const TEST_GPU_MEMORY_PERCENT: u8 = 80;
#[cfg(test)]
pub const TEST_BYTES_PER_RECORD: usize = 500;
#[cfg(test)]
pub const TEST_MIN_BATCH_SIZE: usize = 1000;
#[cfg(test)]
pub const TEST_MAX_BATCH_SIZE: usize = 10000;
#[cfg(test)]
pub const TEST_URL_BUFFER_SIZE: usize = 256;
#[cfg(test)]
pub const TEST_USERNAME_BUFFER_SIZE: usize = 64;
#[cfg(test)]
pub const TEST_THREADS_PER_BLOCK: usize = 256;
#[cfg(test)]
pub const TEST_BATCH_SIZE_SMALL: usize = 3;
#[cfg(test)]
pub const TEST_BATCH_SIZE_MEDIUM: usize = 5;
#[cfg(test)]
pub const TEST_BATCH_SIZE_LARGE: usize = 8;
#[cfg(test)]
pub const TEST_BATCH_SIZE_XLARGE: usize = 10;
#[cfg(test)]
pub const TEST_TOTAL_MEMORY_8GB: usize = 8 * 1024 * 1024 * 1024;
#[cfg(test)]
pub const TEST_TOTAL_MEMORY_SMALL: usize = 500 * 1024; // 500KB
#[cfg(test)]
pub const TEST_USABLE_MEMORY_8GB: usize = 6_528_350_289; // match calculation in test
#[cfg(test)]
pub const TEST_USABLE_MEMORY_SMALL: usize = 389_120; // 500KB * 0.8 * 0.95
#[cfg(test)]
pub const TEST_CALCULATED_BATCH_SIZE_8GB: usize = 13_056_700; // match calculation in test
#[cfg(test)]
pub const TEST_CALCULATED_BATCH_SIZE_SMALL: usize = 778; // 389,120 / 500
#[cfg(test)]
pub const TEST_OPTIMAL_BATCH_SIZE_SMALL: usize = 3;
