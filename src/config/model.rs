use serde::{Deserialize, Serialize};
use sysinfo::System;
use anyhow::Result;
use crate::constants::BYTES_PER_GB;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub memory: MemoryConfig,
    pub processing: ProcessingConfig,
    pub io: IoConfig,
    pub deduplication: DeduplicationConfig,
    pub logging: LoggingConfig,
    pub performance: PerformanceConfig,
    #[cfg(feature = "cuda")]
    pub cuda: CudaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub memory_usage_percent: u8,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_usage_percent: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub enable_cuda: bool,
    pub chunk_size_mb: usize,
    pub record_chunk_size: usize,
    pub max_memory_records: usize,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_cuda: true,
            chunk_size_mb: 8192,
            record_chunk_size: 500000,
            max_memory_records: 40000000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    pub temp_directory: String,
    pub output_directory: String,
    pub checkpoint_auto_save_interval_seconds: u64,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            temp_directory: "./temp".to_string(),
            output_directory: "./output".to_string(),
            checkpoint_auto_save_interval_seconds: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    pub case_sensitive_usernames: bool,
    pub normalize_urls: bool,
    pub email_username_only: bool,
    pub allow_two_field_lines: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: false,
            allow_two_field_lines: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub verbosity: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            verbosity: "normal".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_monitoring: bool,
    pub report_interval_seconds: u64,
    pub show_detailed_metrics: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            report_interval_seconds: 30,
            show_detailed_metrics: true,
        }
    }
}

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
impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            gpu_memory_usage_percent: 80,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizes {
    pub small: usize,
    pub medium: usize,
    pub large: usize,
    pub xlarge: usize,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            memory: MemoryConfig::default(),
            processing: ProcessingConfig::default(),
            io: IoConfig::default(),
            deduplication: DeduplicationConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
            #[cfg(feature = "cuda")]
            cuda: CudaConfig::default(),
        }
    }
}

impl Config {
    pub fn get_memory_info(&self) -> Result<(f64, f64, f64)> {
        let mut system = System::new_all();
        system.refresh_all();

        let total_ram_bytes = system.total_memory() as usize;
        let available_ram_bytes = system.available_memory() as usize;

        let total_ram_gb = total_ram_bytes as f64 / BYTES_PER_GB as f64;
        let available_ram_gb = available_ram_bytes as f64 / BYTES_PER_GB as f64;

        let user_ram_limit = ((available_ram_bytes as f64) * (self.memory.memory_usage_percent as f64 / 100.0)) as usize;
        let max_usable_gb = user_ram_limit as f64 / BYTES_PER_GB as f64;

        Ok((total_ram_gb, available_ram_gb, max_usable_gb))
    }
}