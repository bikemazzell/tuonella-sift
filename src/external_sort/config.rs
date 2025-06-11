use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;
use crate::external_sort::constants::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSortConfig {
    pub memory_usage_percent: f64,
    pub chunk_size_mb: usize,
    pub io_buffer_size_kb: usize,
    pub processing_threads: usize,
    pub enable_cuda: bool,
    pub cuda_batch_size: usize,
    pub cuda_memory_percent: f64,
    pub temp_directory: PathBuf,
    pub enable_compression: bool,
    pub merge_buffer_size_kb: usize,
    pub case_sensitive: bool,
    pub normalize_urls: bool,
    pub email_only_usernames: bool,
    pub verbose: bool,
    pub merge_progress_interval_seconds: u64,
}

impl Default for ExternalSortConfig {
    fn default() -> Self {
        Self {
            memory_usage_percent: DEFAULT_MEMORY_USAGE_PERCENT,
            chunk_size_mb: DEFAULT_CHUNK_SIZE_MB,
            io_buffer_size_kb: DEFAULT_IO_BUFFER_SIZE_KB,
            processing_threads: DEFAULT_PROCESSING_THREADS,
            enable_cuda: true,
            cuda_batch_size: DEFAULT_CUDA_BATCH_SIZE,
            cuda_memory_percent: CUDA_MEMORY_USAGE_PERCENT,
            temp_directory: std::env::temp_dir().join(TEMP_DIR_NAME),
            enable_compression: false,
            merge_buffer_size_kb: DEFAULT_MERGE_BUFFER_SIZE_KB,
            case_sensitive: false,
            normalize_urls: true,
            email_only_usernames: false,
            verbose: false,
            merge_progress_interval_seconds: 10, // Default to 10 seconds
        }
    }
}

impl ExternalSortConfig {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.memory_usage_percent < MIN_MEMORY_USAGE_PERCENT 
            || self.memory_usage_percent > MAX_MEMORY_USAGE_PERCENT {
            return Err(anyhow::anyhow!(
                "Memory usage percent must be between {} and {}", 
                MIN_MEMORY_USAGE_PERCENT, MAX_MEMORY_USAGE_PERCENT
            ));
        }

        if self.chunk_size_mb < MIN_CHUNK_SIZE_MB || self.chunk_size_mb > MAX_CHUNK_SIZE_MB {
            return Err(anyhow::anyhow!(
                "Chunk size must be between {} and {} MB", 
                MIN_CHUNK_SIZE_MB, MAX_CHUNK_SIZE_MB
            ));
        }

        if self.processing_threads < MIN_PROCESSING_THREADS 
            || self.processing_threads > MAX_PROCESSING_THREADS {
            return Err(anyhow::anyhow!(
                "Processing threads must be between {} and {}", 
                MIN_PROCESSING_THREADS, MAX_PROCESSING_THREADS
            ));
        }

        if self.cuda_batch_size < MIN_CUDA_BATCH_SIZE 
            || self.cuda_batch_size > MAX_CUDA_BATCH_SIZE {
            return Err(anyhow::anyhow!(
                "CUDA batch size must be between {} and {}", 
                MIN_CUDA_BATCH_SIZE, MAX_CUDA_BATCH_SIZE
            ));
        }

        Ok(())
    }

    pub fn memory_limit_bytes(&self) -> usize {
        use sysinfo::System;
        let mut system = System::new_all();
        system.refresh_memory();

        let total_memory = system.total_memory() as f64;
        (total_memory * self.memory_usage_percent / 100.0) as usize
    }

    pub fn chunk_size_bytes(&self) -> usize {
        self.chunk_size_mb * BYTES_PER_MB
    }

    pub fn io_buffer_size_bytes(&self) -> usize {
        self.io_buffer_size_kb * BYTES_PER_KB
    }

    pub fn merge_buffer_size_bytes(&self) -> usize {
        self.merge_buffer_size_kb * BYTES_PER_KB
    }
}
