pub const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
pub const VERBOSE_PROGRESS_INTERVAL_SECONDS: u64 = 5;
pub const BYTES_PER_KB: usize = 1024;
pub const BYTES_PER_MB: usize = 1_048_576;
pub const BYTES_PER_GB: f64 = 1_073_741_824.0;
pub const PERCENT_100: f64 = 100.0;
pub const PERCENT_95: f64 = 0.95;

// Test constants for CUDA properties
pub const TEST_TOTAL_MEMORY: usize = 8_000_000_000;
pub const TEST_FREE_MEMORY: usize = 6_000_000_000;
pub const TEST_COMPUTE_CAPABILITY_MAJOR: i32 = 7;
pub const TEST_COMPUTE_CAPABILITY_MINOR: i32 = 5;
pub const TEST_MAX_THREADS_PER_BLOCK: i32 = 1024;
pub const TEST_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 49152;
pub const TEST_MEMORY_BUS_WIDTH: i32 = 256;
pub const TEST_L2_CACHE_SIZE: i32 = 4194304;

// Test constants for batch size calculations
pub const TEST_TOTAL_MEMORY_8GB: u64 = 8_000_000_000;
pub const TEST_TOTAL_MEMORY_SMALL: u64 = 500_000;
pub const TEST_GPU_MEMORY_PERCENT: u8 = 80;
pub const TEST_USABLE_MEMORY_8GB: usize = 6_400_000_000;
pub const TEST_USABLE_MEMORY_SMALL: usize = 400_000;
pub const TEST_CALCULATED_BATCH_SIZE_8GB: usize = 12_800_000;
pub const TEST_CALCULATED_BATCH_SIZE_SMALL: usize = 800;
pub const TEST_OPTIMAL_BATCH_SIZE_SMALL: usize = 3;

// Test constants for CUDA configuration
pub const TEST_BYTES_PER_RECORD: usize = 500;
pub const TEST_MIN_BATCH_SIZE: usize = 1000;
pub const TEST_MAX_BATCH_SIZE: usize = 10000;
pub const TEST_URL_BUFFER_SIZE: usize = 256;
pub const TEST_USERNAME_BUFFER_SIZE: usize = 64;
pub const TEST_THREADS_PER_BLOCK: usize = 256;
pub const TEST_BATCH_SIZE_SMALL: usize = 1000;
pub const TEST_BATCH_SIZE_MEDIUM: usize = 2500;
pub const TEST_BATCH_SIZE_LARGE: usize = 5000;
pub const TEST_BATCH_SIZE_XLARGE: usize = 10000;

pub const DEFAULT_SAMPLE_SIZE: usize = 100;
pub const DEFAULT_RECORD_ESTIMATED_BYTES: usize = 200;
pub const DEFAULT_MAX_GPU_BATCH_SIZE: usize = 100000;
pub const DEFAULT_CPU_SEGMENT_MIN_MB: usize = 64;
pub const DEFAULT_MAX_URL_LENGTH_FOR_FAST_NORMALIZATION: usize = 512;
