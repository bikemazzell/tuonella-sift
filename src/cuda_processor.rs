use anyhow::Result;
use cudarc::driver::{CudaDevice, sys::CUresult};
use tracing::info;
use crate::record::Record;
use crate::config::{CudaConfig, UrlNormalizationConfig};
use crate::patterns::normalize_url_fast;
use crate::constants::BYTES_PER_GB; // Added import
use std::sync::Arc;

// Removed const BYTES_PER_GB definition from here

pub struct CudaProcessor {
    device: Arc<CudaDevice>,
    max_batch_size: usize,
    config: CudaConfig,
    optimal_batch_size: usize,
}

impl CudaProcessor {
    pub fn new(config: CudaConfig) -> Result<Self> { // Removed _url_config parameter
        // Initialize CUDA device
        let device = CudaDevice::new(0)
            .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device 0: {}. Ensure NVIDIA drivers and CUDA toolkit are installed.", e))?;
        
        info!("CUDA device initialized successfully");
        
        // Get GPU memory information
        let (free_memory, total_memory) = Self::get_gpu_memory_info()?;
        
        info!("GPU Memory - Total: {:.2} GB, Free: {:.2} GB", 
              total_memory as f64 / BYTES_PER_GB,
              free_memory as f64 / BYTES_PER_GB);
        
        // Calculate available memory with safety margin
        let available_memory_bytes = (free_memory as f64 * (config.gpu_memory_usage_percent as f64 / 100.0) * 0.95) as usize;
        
        // Calculate optimal batch size based on memory and configuration
        let max_batch_size = (available_memory_bytes / config.estimated_bytes_per_record)
            .max(config.min_batch_size)
            .min(config.max_batch_size);
            
        // Calculate optimal batch size based on GPU characteristics
        let optimal_batch_size = Self::calculate_optimal_batch_size(
            max_batch_size,
            config.threads_per_block,
            &config
        );
        
        info!("CUDA processor initialized - Available memory: {:.2} GB, Max batch size: {}, Optimal batch size: {}", 
              available_memory_bytes as f64 / BYTES_PER_GB,
              max_batch_size,
              optimal_batch_size);
        
        Ok(Self {
            device,
            max_batch_size,
            config,
            optimal_batch_size,
        })
    }

    fn calculate_optimal_batch_size(max_size: usize, threads_per_block: usize, config: &CudaConfig) -> usize {
        // Calculate optimal batch size based on GPU characteristics
        let mut size = max_size;
        
        // Ensure batch size is a multiple of threads per block for optimal performance
        size = (size / threads_per_block) * threads_per_block;
        
        // Apply batch size tiers based on available memory
        if size >= config.batch_sizes.xlarge {
            size = config.batch_sizes.xlarge;
        } else if size >= config.batch_sizes.large {
            size = config.batch_sizes.large;
        } else if size >= config.batch_sizes.medium {
            size = config.batch_sizes.medium;
        } else {
            size = size.max(config.batch_sizes.small);
        }
        
        size
    }

    pub fn process_batch(&self, records: &mut [Record], case_sensitive_usernames: bool) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Process records in chunks that fit in GPU memory
        let chunk_size = self.optimal_batch_size.min(records.len());
        for chunk in records.chunks_mut(chunk_size) {
            self.process_chunk(chunk, case_sensitive_usernames)?;
        }

        Ok(())
    }

    fn process_chunk(&self, records: &mut [Record], case_sensitive_usernames: bool) -> Result<()> {
        // Prepare data for GPU processing
        let mut urls: Vec<String> = records.iter().map(|r| r.url.clone()).collect();
        let mut usernames: Vec<String> = records.iter().map(|r| r.user.clone()).collect();

        // Process URLs and usernames on GPU
        self.normalize_urls_gpu(&mut urls)?;
        if !case_sensitive_usernames {
            self.normalize_usernames_gpu(&mut usernames)?;
        }

        // Update records with normalized data
        for (i, record) in records.iter_mut().enumerate() {
            record.normalized_url = urls[i].clone();
            record.normalized_user = if case_sensitive_usernames {
                record.user.clone()
            } else {
                usernames[i].clone()
            };
        }

        Ok(())
    }

    fn normalize_urls_gpu(&self, urls: &mut [String]) -> Result<()> {
        // TODO: Implement actual GPU normalization
        // For now, use CPU implementation
        for url in urls.iter_mut() {
            *url = normalize_url_fast(url);
        }
        Ok(())
    }

    fn normalize_usernames_gpu(&self, usernames: &mut [String]) -> Result<()> {
        // TODO: Implement actual GPU normalization
        // For now, use CPU implementation
        for username in usernames.iter_mut() {
            *username = username.to_lowercase();
        }
        Ok(())
    }

    pub fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }
    
    pub fn get_memory_info(&self) -> (usize, usize) {
        // Return (available_memory, max_batch_size)
        // Calculate available memory from GPU memory usage percent
        if let Ok((free_memory, _total_memory)) = Self::get_gpu_memory_info() {
            let available_memory_bytes = (free_memory as f64 * (self.config.gpu_memory_usage_percent as f64 / 100.0)) as usize;
            (available_memory_bytes, self.max_batch_size)
        } else {
            (0, self.max_batch_size)
        }
    }
    
    pub fn get_max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size
    }

    fn get_gpu_memory_info() -> Result<(usize, usize)> {
        unsafe {
            let mut free = 0;
            let mut total = 0;
            let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow::anyhow!("Failed to get GPU memory info: error code {:?}", result));
            }
            Ok((free as usize, total as usize))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_processor_creation() {
        if CudaProcessor::is_available() {
            let processor = CudaProcessor::new(CudaConfig::default()); // Removed UrlNormalizationConfig::default()
            assert!(processor.is_ok());
            
            if let Ok(processor) = processor {
                let (available_memory, max_batch_size) = processor.get_memory_info();
                println!("Available GPU memory: {:.2} GB", available_memory as f64 / BYTES_PER_GB);
                println!("Max batch size: {}", max_batch_size);
                
                // Basic sanity checks
                assert!(available_memory > 0, "Available memory should be greater than 0");
                assert!(max_batch_size >= 1_000_000, "Batch size should be at least 1,000,000");
                assert!(max_batch_size <= 10_000_000, "Batch size should not exceed 10,000,000");
            }
        } else {
            println!("CUDA not available, skipping test");
        }
    }
    
    #[test]
    fn test_gpu_memory_detection() {
        if CudaProcessor::is_available() {
            let _device = CudaDevice::new(0).unwrap();
            let memory_info = CudaProcessor::get_gpu_memory_info();
            
            assert!(memory_info.is_ok(), "Memory detection should succeed");
            
            if let Ok((free, total)) = memory_info {
                println!("GPU Memory - Free: {} bytes, Total: {} bytes", free, total);
                assert!(free > 0, "Free memory should be greater than 0");
                assert!(total > 0, "Total memory should be greater than 0");
                assert!(free <= total, "Free memory should not exceed total memory");
            }
        } else {
            println!("CUDA not available, skipping memory detection test");
        }
    }
}
