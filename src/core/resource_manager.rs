use anyhow::Result;
use crate::core::memory_manager::{MemoryManager, ResourceUsageStats};

#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaProcessor;

pub struct ResourceManager<'a> {
    memory_manager: &'a mut MemoryManager,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<&'a CudaProcessor>,
}

impl<'a> ResourceManager<'a> {
    #[cfg(feature = "cuda")]
    pub fn new(
        memory_manager: &'a mut MemoryManager,
        cuda_processor: Option<&'a CudaProcessor>,
    ) -> Self {
        Self {
            memory_manager,
            cuda_processor,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(memory_manager: &'a mut MemoryManager) -> Self {
        Self { memory_manager }
    }

    pub fn release_chunk_resources(&mut self) -> Result<()> {
        self.memory_manager.release_chunk_resources()?;

        #[cfg(feature = "cuda")]
        if let Some(processor) = self.cuda_processor {
            processor.release_gpu_resources()?;
        }

        Ok(())
    }

    pub fn get_resource_usage_stats(&self) -> Result<ResourceUsageStats> {
        #[cfg(feature = "cuda")]
        {
            self.memory_manager
                .get_resource_usage_stats_with_gpu(self.cuda_processor)
        }

        #[cfg(not(feature = "cuda"))]
        {
            self.memory_manager.get_resource_usage_stats()
        }
    }

    pub fn check_resource_pressure(&self) -> Result<bool> {
        let stats = self.get_resource_usage_stats()?;
        Ok(stats.cpu_memory_pressure || stats.gpu_memory_pressure)
    }

    pub fn perform_adaptive_management(&mut self) -> Result<AdaptiveManagementResult> {
        let initial_chunk_size = self.memory_manager.get_current_chunk_size();
        let resource_pressure = self.check_resource_pressure()?;
        
        let chunk_adjusted = self.memory_manager.adjust_chunk_size_if_needed()?;
        let new_chunk_size = self.memory_manager.get_current_chunk_size();
        
        let resources_released = if resource_pressure {
            self.release_chunk_resources()?;
            true
        } else {
            false
        };

        Ok(AdaptiveManagementResult {
            resource_pressure,
            chunk_size_adjusted: chunk_adjusted,
            resources_released,
            initial_chunk_size,
            new_chunk_size,
        })
    }

    pub fn memory_manager(&self) -> &MemoryManager {
        self.memory_manager
    }

    pub fn memory_manager_mut(&mut self) -> &mut MemoryManager {
        self.memory_manager
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveManagementResult {
    pub resource_pressure: bool,
    pub chunk_size_adjusted: bool,
    pub resources_released: bool,
    pub initial_chunk_size: usize,
    pub new_chunk_size: usize,
}

impl AdaptiveManagementResult {
    pub fn format_summary(&self) -> String {
        let mut summary = String::new();
        
        if self.resource_pressure {
            summary.push_str("⚠️ Resource pressure detected\n");
        } else {
            summary.push_str("✅ Resources operating normally\n");
        }
        
        if self.chunk_size_adjusted {
            let factor = self.new_chunk_size as f64 / self.initial_chunk_size as f64;
            if factor > 1.0 {
                summary.push_str(&format!(
                    "📈 Chunk size increased: {:.2} MB → {:.2} MB ({:.2}x)\n",
                    self.initial_chunk_size as f64 / crate::constants::BYTES_PER_MB as f64,
                    self.new_chunk_size as f64 / crate::constants::BYTES_PER_MB as f64,
                    factor
                ));
            } else if factor < 1.0 {
                summary.push_str(&format!(
                    "📉 Chunk size reduced: {:.2} MB → {:.2} MB ({:.2}x)\n",
                    self.initial_chunk_size as f64 / crate::constants::BYTES_PER_MB as f64,
                    self.new_chunk_size as f64 / crate::constants::BYTES_PER_MB as f64,
                    factor
                ));
            }
        } else {
            summary.push_str("➡️ Chunk size unchanged\n");
        }
        
        if self.resources_released {
            summary.push_str("🧹 Resources released due to pressure\n");
        }
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_manager::MemoryManager;

    #[test]
    fn test_resource_manager_creation() {
        let mut memory_manager = MemoryManager::new(Some(1)).unwrap();
        
        #[cfg(feature = "cuda")]
        let resource_manager = ResourceManager::new(&mut memory_manager, None);
        
        #[cfg(not(feature = "cuda"))]
        let resource_manager = ResourceManager::new(&mut memory_manager);
        
        assert!(resource_manager.memory_manager().is_initialized());
    }

    #[test]
    fn test_resource_cleanup() {
        let mut memory_manager = MemoryManager::new(Some(1)).unwrap();
        
        #[cfg(feature = "cuda")]
        let mut resource_manager = ResourceManager::new(&mut memory_manager, None);
        
        #[cfg(not(feature = "cuda"))]
        let mut resource_manager = ResourceManager::new(&mut memory_manager);
        
        let test_data = b"test data";
        resource_manager.memory_manager_mut().add_to_ram_buffer(test_data).unwrap();
        
        assert!(resource_manager.memory_manager().get_ram_buffer_contents().len() > 0);
        
        let result = resource_manager.release_chunk_resources();
        assert!(result.is_ok());
        
        assert_eq!(resource_manager.memory_manager().get_ram_buffer_contents().len(), 0);
    }

    #[test]
    fn test_adaptive_management() {
        let mut memory_manager = MemoryManager::new(Some(1)).unwrap();
        
        #[cfg(feature = "cuda")]
        let mut resource_manager = ResourceManager::new(&mut memory_manager, None);
        
        #[cfg(not(feature = "cuda"))]
        let mut resource_manager = ResourceManager::new(&mut memory_manager);
        
        let result = resource_manager.perform_adaptive_management();
        assert!(result.is_ok());
        
        let management_result = result.unwrap();
        let summary = management_result.format_summary();
        assert!(summary.len() > 0);
    }
}
