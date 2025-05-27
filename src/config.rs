use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use sysinfo::System;
use tokio::fs;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub memory: MemoryConfig,
    pub processing: ProcessingConfig,
    pub io: IoConfig,
    pub deduplication: DeduplicationConfig,
    pub logging: LoggingConfig,
    pub recovery: RecoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_ram_usage_percent: u8,
    pub batch_size_gb: f64,
    pub auto_detect_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub max_threads: usize,
    pub enable_cuda: bool,
    pub chunk_size_mb: usize,
    pub max_output_file_size_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    pub temp_directory: String,
    pub output_directory: String,
    pub enable_memory_mapping: bool,
    pub parallel_io: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    pub case_sensitive_usernames: bool,
    pub normalize_urls: bool,
    pub strip_url_params: bool,
    pub strip_url_prefixes: bool,
    pub completeness_strategy: String,
    pub field_detection_sample_percent: f64,
    pub min_sample_size: usize,
    pub max_sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub verbosity: String,
    pub progress_interval_seconds: u64,
    pub log_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    pub enable_checkpointing: bool,
    pub checkpoint_interval_records: usize,
}

impl Config {
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).await?;
        let mut config: Config = serde_json::from_str(&content)?;
        
        config.auto_configure_memory().await?;
        config.validate()?;
        
        Ok(config)
    }

    async fn auto_configure_memory(&mut self) -> Result<()> {
        if !self.memory.auto_detect_memory {
            return Ok(());
        }

        let mut system = System::new_all();
        system.refresh_memory();
        
        let total_memory_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_memory_gb = system.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        info!("Detected system memory: {:.2} GB total, {:.2} GB available", 
              total_memory_gb, available_memory_gb);

        let max_usable_gb = available_memory_gb * (self.memory.max_ram_usage_percent as f64 / 100.0);
        
        if self.memory.batch_size_gb > max_usable_gb {
            warn!("Configured batch size ({:.2} GB) exceeds available memory, adjusting to {:.2} GB", 
                  self.memory.batch_size_gb, max_usable_gb * 0.8);
            self.memory.batch_size_gb = max_usable_gb * 0.8;
        }

        if self.processing.max_threads == 0 {
            self.processing.max_threads = num_cpus::get();
            info!("Auto-detected {} CPU cores", self.processing.max_threads);
        }

        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.memory.max_ram_usage_percent > 90 {
            anyhow::bail!("max_ram_usage_percent cannot exceed 90%");
        }

        if self.memory.batch_size_gb < 0.1 {
            anyhow::bail!("batch_size_gb must be at least 0.1 GB");
        }

        if !matches!(self.deduplication.completeness_strategy.as_str(), 
                    "character_count" | "field_count") {
            anyhow::bail!("completeness_strategy must be 'character_count' or 'field_count'");
        }

        if self.deduplication.field_detection_sample_percent <= 0.0 || 
           self.deduplication.field_detection_sample_percent > 100.0 {
            anyhow::bail!("field_detection_sample_percent must be between 0.1 and 100.0");
        }

        if self.deduplication.min_sample_size == 0 {
            anyhow::bail!("min_sample_size must be at least 1");
        }

        if self.deduplication.max_sample_size < self.deduplication.min_sample_size {
            anyhow::bail!("max_sample_size must be >= min_sample_size");
        }

        if !matches!(self.logging.verbosity.as_str(), "silent" | "normal" | "verbose") {
            anyhow::bail!("verbosity must be 'silent', 'normal', or 'verbose'");
        }

        Ok(())
    }

    pub fn get_max_memory_bytes(&self) -> usize {
        (self.memory.batch_size_gb * 1024.0 * 1024.0 * 1024.0) as usize
    }

    pub fn get_chunk_size_bytes(&self) -> usize {
        self.processing.chunk_size_mb * 1024 * 1024
    }

    pub fn get_max_output_file_bytes(&self) -> u64 {
        (self.processing.max_output_file_size_gb * 1024.0 * 1024.0 * 1024.0) as u64
    }
} 