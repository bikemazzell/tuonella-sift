// Minimal config model needed for CUDA support in external_sort

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaConfig {
    pub gpu_memory_usage_percent: u8,
    pub estimated_bytes_per_record: usize,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub max_url_buffer_size: usize,
    pub max_username_buffer_size: usize,
    pub threads_per_block: usize,
    pub batch_sizes: BatchSizes,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizes {
    pub small: usize,
    pub medium: usize,
    pub large: usize,
    pub xlarge: usize,
}

#[cfg(feature = "cuda")]
impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            gpu_memory_usage_percent: 50,
            estimated_bytes_per_record: 500,
            min_batch_size: 10000,
            max_batch_size: 1000000,
            max_url_buffer_size: 256,
            max_username_buffer_size: 64,
            threads_per_block: 256,
            batch_sizes: BatchSizes::default(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Default for BatchSizes {
    fn default() -> Self {
        Self {
            small: 10000,
            medium: 50000,
            large: 100000,
            xlarge: 500000,
        }
    }
}