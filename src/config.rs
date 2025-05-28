use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use sysinfo::System;
use tokio::fs;
use tracing::{info, warn};
use regex::Regex;
use serde::de::Deserializer;
use std::sync::Arc;
use crate::constants::BYTES_PER_GB;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub memory: MemoryConfig,
    pub processing: ProcessingConfig,
    pub io: IoConfig,
    pub deduplication: DeduplicationConfig,
    pub logging: LoggingConfig,
    pub recovery: RecoveryConfig,
    #[serde(default)]
    pub cuda: CudaConfig,
    #[serde(skip_serializing, default, deserialize_with = "deserialize_url_normalization_config_from_strings")]
    pub url_normalization: UrlNormalizationConfig,
    #[serde(default)]
    pub field_detection: FieldDetectionConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizes {
    pub small: usize,
    pub medium: usize,
    pub large: usize,
    pub xlarge: usize,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            gpu_memory_usage_percent: 80,
            estimated_bytes_per_record: 500,
            min_batch_size: 1000000,
            max_batch_size: 10000000,
            max_url_buffer_size: 256,
            max_username_buffer_size: 64,
            threads_per_block: 256,
            batch_sizes: BatchSizes {
                small: 1000000,
                medium: 2500000,
                large: 5000000,
                xlarge: 10000000,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct UrlNormalizationConfig {
    pub protocol_patterns: Vec<Arc<Regex>>,
    pub subdomain_removal_patterns: Vec<Arc<Regex>>,
    pub path_cleanup_patterns: Vec<Arc<Regex>>,
    pub android_uri_cleanup: bool,
    pub remove_query_params: bool,
    pub remove_fragments: bool,
    pub normalize_case: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UrlNormalizationConfigStrings {
    pub protocol_patterns: Vec<String>,
    pub subdomain_removal_patterns: Vec<String>,
    pub path_cleanup_patterns: Vec<String>,
    pub android_uri_cleanup: bool,
    pub remove_query_params: bool,
    pub remove_fragments: bool,
    pub normalize_case: bool,
}

impl From<UrlNormalizationConfigStrings> for UrlNormalizationConfig {
    fn from(str_config: UrlNormalizationConfigStrings) -> Self {
        Self {
            protocol_patterns: str_config.protocol_patterns.into_iter()
                .map(|s| Arc::new(Regex::new(&s).expect("Invalid protocol regex pattern")))
                .collect(),
            subdomain_removal_patterns: str_config.subdomain_removal_patterns.into_iter()
                .map(|s| Arc::new(Regex::new(&s).expect("Invalid subdomain regex pattern")))
                .collect(),
            path_cleanup_patterns: str_config.path_cleanup_patterns.into_iter()
                .map(|s| Arc::new(Regex::new(&s).expect("Invalid path cleanup regex pattern")))
                .collect(),
            android_uri_cleanup: str_config.android_uri_cleanup,
            remove_query_params: str_config.remove_query_params,
            remove_fragments: str_config.remove_fragments,
            normalize_case: str_config.normalize_case,
        }
    }
}

fn deserialize_url_normalization_config_from_strings<'de, D>(deserializer: D) -> Result<UrlNormalizationConfig, D::Error>
where
    D: Deserializer<'de>,
{
    let str_config = UrlNormalizationConfigStrings::deserialize(deserializer)?;
    Ok(UrlNormalizationConfig::from(str_config))
}

impl Default for UrlNormalizationConfig {
    fn default() -> Self {
        let defaults_str = UrlNormalizationConfigStrings {
            protocol_patterns: vec![
                r"^[a-zA-Z][a-zA-Z0-9+.-]*://".to_string(),
            ],
            subdomain_removal_patterns: vec![
                r"^([a-zA-Z0-9-]+)\.([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$".to_string(),
            ],
            path_cleanup_patterns: vec![
                "/.*$".to_string(),
            ],
            android_uri_cleanup: true,
            remove_query_params: true,
            remove_fragments: true,
            normalize_case: true,
        };
        UrlNormalizationConfig::from(defaults_str)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDetectionConfig {
    pub url_patterns: Vec<String>,
    pub email_patterns: Vec<String>,
    pub password_patterns: Vec<String>,
}

impl Default for FieldDetectionConfig {
    fn default() -> Self {
        Self {
            url_patterns: vec![
                "^[a-zA-Z][a-zA-Z0-9+.-]*://".to_string(),
                "\\.[a-zA-Z]{2,}".to_string(),
            ],
            email_patterns: vec![
                "@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$".to_string(),
            ],
            password_patterns: vec![
                "^[a-zA-Z0-9!@#$%^&*()_+=\\-\\[\\]{};':,.<>/?]{4,}$".to_string(),
            ],
        }
    }
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
        
        let total_memory_gb = system.total_memory() as f64 / BYTES_PER_GB;
        let available_memory_gb = system.available_memory() as f64 / BYTES_PER_GB;
        
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
        (self.memory.batch_size_gb * BYTES_PER_GB) as usize
    }

    #[allow(dead_code)]
    pub fn get_chunk_size_bytes(&self) -> usize {
        self.processing.chunk_size_mb * 1_048_576
    }

    #[allow(dead_code)]
    pub fn get_max_output_file_bytes(&self) -> u64 {
        (self.processing.max_output_file_size_gb * BYTES_PER_GB) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_config_loading_with_cuda() {
        // Test that the unified config.json loads correctly
        let config_result = Config::load("config.json").await;
        assert!(config_result.is_ok(), "Should be able to load config.json");
        
        if let Ok(config) = config_result {
            // Verify CUDA settings are loaded
            assert_eq!(config.cuda.gpu_memory_usage_percent, 80);
            assert_eq!(config.cuda.estimated_bytes_per_record, 500);
            assert_eq!(config.cuda.min_batch_size, 1000000);
            assert_eq!(config.cuda.max_batch_size, 10000000);
            assert_eq!(config.cuda.max_url_buffer_size, 256);
            assert_eq!(config.cuda.max_username_buffer_size, 64);
            assert_eq!(config.cuda.threads_per_block, 256);
            
            println!("✓ Unified config.json loaded successfully");
            println!("✓ CUDA settings: {}% GPU memory, {} bytes/record", 
                     config.cuda.gpu_memory_usage_percent, 
                     config.cuda.estimated_bytes_per_record);
            println!("✓ Batch size range: {} - {}", 
                     config.cuda.min_batch_size, 
                     config.cuda.max_batch_size);
        }
    }

    #[tokio::test]
    async fn test_config_loading_without_cuda() {
        // Test that config files without CUDA section work (backward compatibility)
        let minimal_config = r#"{
            "memory": {
                "max_ram_usage_percent": 30,
                "batch_size_gb": 1,
                "auto_detect_memory": false
            },
            "processing": {
                "max_threads": 2,
                "enable_cuda": false,
                "chunk_size_mb": 16,
                "max_output_file_size_gb": 1
            },
            "io": {
                "temp_directory": "./temp",
                "output_directory": "./output",
                "enable_memory_mapping": true,
                "parallel_io": true
            },
            "deduplication": {
                "case_sensitive_usernames": false,
                "normalize_urls": true,
                "strip_url_params": true,
                "strip_url_prefixes": true,
                "completeness_strategy": "character_count",
                "field_detection_sample_percent": 5.0,
                "min_sample_size": 50,
                "max_sample_size": 1000
            },
            "logging": {
                "verbosity": "normal",
                "progress_interval_seconds": 30,
                "log_file": "tuonella-sift.log"
            },
            "recovery": {
                "enable_checkpointing": true,
                "checkpoint_interval_records": 100000
            }
        }"#;
        
        let config_result: Result<Config> = serde_json::from_str(minimal_config)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e));
        assert!(config_result.is_ok(), "Should be able to load config without CUDA section");
        
        if let Ok(config) = config_result {
            // Verify CUDA settings use defaults when not specified
            assert_eq!(config.cuda.gpu_memory_usage_percent, 80); // Default value
            assert_eq!(config.cuda.estimated_bytes_per_record, 500); // Default value
            assert_eq!(config.cuda.min_batch_size, 1000000); // Default value
            assert_eq!(config.cuda.max_batch_size, 10000000); // Default value
            assert_eq!(config.cuda.max_url_buffer_size, 256); // Default value
            assert_eq!(config.cuda.max_username_buffer_size, 64); // Default value
            assert_eq!(config.cuda.threads_per_block, 256); // Default value
            
            // Verify other settings are loaded correctly
            assert_eq!(config.processing.enable_cuda, false);
            assert_eq!(config.logging.verbosity, "normal");
            
            println!("✓ Config without CUDA section loaded successfully");
            println!("✓ CUDA defaults applied: {}% GPU memory, {} bytes/record", 
                     config.cuda.gpu_memory_usage_percent, 
                     config.cuda.estimated_bytes_per_record);
            println!("✓ CUDA disabled: {}", !config.processing.enable_cuda);
        }
    }
} 