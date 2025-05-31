use anyhow::Result;
use crate::constants::{
    DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS, BYTES_PER_MB,
    MEMORY_PRESSURE_THRESHOLD_PERCENT, MEMORY_CRITICAL_THRESHOLD_PERCENT,
    CHUNK_SIZE_REDUCTION_FACTOR, CHUNK_SIZE_INCREASE_FACTOR,
    MIN_CHUNK_SIZE_REDUCTION_LIMIT, MAX_CHUNK_SIZE_INCREASE_LIMIT,
    CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS, ZERO_USIZE, PERCENT_100,
    ZERO_F64, LOW_MEMORY_THRESHOLD_FACTOR, DEFAULT_CHUNK_ADJUSTMENT_FACTOR,
    MIN_CHUNK_SIZE_LIMIT, DEFAULT_GPU_MEMORY_FREE, DEFAULT_GPU_MEMORY_TOTAL,
    DEFAULT_GPU_PRESSURE
};
use crate::utils::system::SystemResources;

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext;

#[derive(Debug)]
pub struct MemoryManager {
    resources: SystemResources,

    ram_buffer: Vec<u8>,

    ram_buffer_position: usize,

    ram_buffer_capacity: usize,

    #[cfg(feature = "cuda")]
    gpu_buffer_capacity: usize,

    #[cfg(feature = "cuda")]
    gpu_context: Option<Arc<CudaContext>>,

    record_counter: usize,

    buffers_initialized: bool,

    original_chunk_size_bytes: usize,

    current_chunk_size_bytes: usize,

    last_chunk_adjustment_record: usize,
}

impl MemoryManager {
    pub fn new(user_ram_limit_gb: Option<usize>) -> Result<Self> {
        let resources = SystemResources::query_system_resources(user_ram_limit_gb)?;

        println!("🧠 Initializing Memory Manager...");
        println!("{}", resources.format_summary());

        let ram_buffer_capacity = resources.ram_buffer_size_bytes;
        println!("📦 Preallocating RAM buffer: {:.2} MB", ram_buffer_capacity as f64 / BYTES_PER_MB as f64);

        let ram_buffer = Vec::with_capacity(ram_buffer_capacity);

        #[cfg(feature = "cuda")]
        let gpu_buffer_capacity = resources.gpu_buffer_size_bytes;

        #[cfg(feature = "cuda")]
        let gpu_context = if gpu_buffer_capacity > ZERO_USIZE && resources.gpu_properties.is_some() {
            println!("🚀 GPU buffer capacity reserved: {:.2} MB", gpu_buffer_capacity as f64 / BYTES_PER_MB as f64);
            match CudaContext::new(0) {
                Ok(context) => {
                    println!("✅ CUDA context initialized for buffer management");
                    Some(context)
                }
                Err(e) => {
                    println!("⚠️ Failed to create CUDA context: {}, GPU buffer disabled", e);
                    None
                }
            }
        } else {
            println!("ℹ️ GPU buffer not available");
            None
        };

        let initial_chunk_size = ram_buffer_capacity;

        Ok(Self {
            resources,
            ram_buffer,
            ram_buffer_position: ZERO_USIZE,
            ram_buffer_capacity,
            #[cfg(feature = "cuda")]
            gpu_buffer_capacity,
            #[cfg(feature = "cuda")]
            gpu_context,
            record_counter: ZERO_USIZE,
            buffers_initialized: true,
            original_chunk_size_bytes: initial_chunk_size,
            current_chunk_size_bytes: initial_chunk_size,
            last_chunk_adjustment_record: ZERO_USIZE,
        })
    }

    pub fn ram_buffer_available_space(&self) -> usize {
        self.ram_buffer_capacity.saturating_sub(self.ram_buffer_position)
    }

    pub fn can_fit_in_ram_buffer(&self, data_size: usize) -> bool {
        self.ram_buffer_available_space() >= data_size
    }

    pub fn add_to_ram_buffer(&mut self, data: &[u8]) -> Result<bool> {
        if !self.can_fit_in_ram_buffer(data.len()) {
            return Ok(false);
        }

        self.ram_buffer.extend_from_slice(data);
        self.ram_buffer_position += data.len();

        self.record_counter += 1;
        if self.record_counter % DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS == 0 {
            self.check_memory_pressure()?;
        }

        Ok(true)
    }

    pub fn get_ram_buffer_contents(&self) -> &[u8] {
        &self.ram_buffer[..self.ram_buffer_position]
    }

    pub fn clear_ram_buffer(&mut self) {
        self.ram_buffer.clear();
        self.ram_buffer_position = ZERO_USIZE;
    }

    #[cfg(feature = "cuda")]
    pub fn get_gpu_buffer_capacity(&self) -> usize {
        self.gpu_buffer_capacity
    }

    #[cfg(feature = "cuda")]
    pub fn get_gpu_context(&self) -> Option<&Arc<CudaContext>> {
        self.gpu_context.as_ref()
    }

    #[cfg(feature = "cuda")]
    pub fn has_gpu_buffer(&self) -> bool {
        self.gpu_context.is_some() && self.gpu_buffer_capacity > ZERO_USIZE
    }

    pub fn check_memory_pressure(&self) -> Result<bool> {
        let is_pressure = self.resources.is_memory_pressure()?;
        if is_pressure {
            let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
            println!("⚠️  Memory pressure detected: {:.2}% usage ({} bytes)",
                    usage_percent, current_usage);
        }
        Ok(is_pressure)
    }

    pub fn get_current_chunk_size(&self) -> usize {
        self.current_chunk_size_bytes
    }

    pub fn adjust_chunk_size_if_needed(&mut self) -> Result<bool> {
        if self.record_counter - self.last_chunk_adjustment_record < CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS {
            return Ok(false);
        }

        let (_current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
        let mut adjusted = false;

        if usage_percent >= MEMORY_CRITICAL_THRESHOLD_PERCENT {
            let new_size = (self.current_chunk_size_bytes as f64 * CHUNK_SIZE_REDUCTION_FACTOR) as usize;
            let min_allowed = (self.original_chunk_size_bytes as f64 * MIN_CHUNK_SIZE_REDUCTION_LIMIT) as usize;

            if new_size >= min_allowed && new_size < self.current_chunk_size_bytes {
                self.current_chunk_size_bytes = new_size;
                self.last_chunk_adjustment_record = self.record_counter;
                println!("🔽 Critical memory pressure ({:.1}%) - reducing chunk size to {:.2} MB",
                        usage_percent, new_size as f64 / BYTES_PER_MB as f64);
                adjusted = true;
            }
        } else if usage_percent >= MEMORY_PRESSURE_THRESHOLD_PERCENT {
            let new_size = (self.current_chunk_size_bytes as f64 * CHUNK_SIZE_REDUCTION_FACTOR) as usize;
            let min_allowed = (self.original_chunk_size_bytes as f64 * MIN_CHUNK_SIZE_REDUCTION_LIMIT) as usize;

            if new_size >= min_allowed && new_size < self.current_chunk_size_bytes {
                self.current_chunk_size_bytes = new_size;
                self.last_chunk_adjustment_record = self.record_counter;
                println!("🔽 Memory pressure ({:.1}%) - reducing chunk size to {:.2} MB",
                        usage_percent, new_size as f64 / BYTES_PER_MB as f64);
                adjusted = true;
            }
        } else if usage_percent < MEMORY_PRESSURE_THRESHOLD_PERCENT * LOW_MEMORY_THRESHOLD_FACTOR {
            let new_size = (self.current_chunk_size_bytes as f64 * CHUNK_SIZE_INCREASE_FACTOR) as usize;
            let max_allowed = (self.original_chunk_size_bytes as f64 * MAX_CHUNK_SIZE_INCREASE_LIMIT) as usize;

            if new_size <= max_allowed && new_size > self.current_chunk_size_bytes {
                self.current_chunk_size_bytes = new_size;
                self.last_chunk_adjustment_record = self.record_counter;
                println!("🔼 Low memory usage ({:.1}%) - increasing chunk size to {:.2} MB",
                        usage_percent, new_size as f64 / BYTES_PER_MB as f64);
                adjusted = true;
            }
        }

        Ok(adjusted)
    }

    pub fn get_memory_stats(&self) -> Result<MemoryStats> {
        let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
        let is_pressure = self.resources.is_memory_pressure()?;

        let chunk_adjustment_factor = if self.original_chunk_size_bytes > 0 {
            self.current_chunk_size_bytes as f64 / self.original_chunk_size_bytes as f64
        } else {
            DEFAULT_CHUNK_ADJUSTMENT_FACTOR
        };

        Ok(MemoryStats {
            total_ram_bytes: self.resources.total_ram_bytes,
            available_ram_bytes: self.resources.available_ram_bytes,
            ram_limit_bytes: self.resources.ram_limit_bytes,
            ram_buffer_capacity_bytes: self.ram_buffer_capacity,
            ram_buffer_used_bytes: self.ram_buffer_position,
            current_process_usage_bytes: current_usage,
            usage_percent,
            memory_pressure: is_pressure,
            #[cfg(feature = "cuda")]
            gpu_buffer_capacity_bytes: self.gpu_buffer_capacity,
            original_chunk_size_bytes: self.original_chunk_size_bytes,
            current_chunk_size_bytes: self.current_chunk_size_bytes,
            chunk_size_adjustment_factor: chunk_adjustment_factor,
        })
    }

    pub fn get_system_resources(&self) -> &SystemResources {
        &self.resources
    }

    pub fn is_initialized(&self) -> bool {
        self.buffers_initialized
    }

    pub fn record_processed(&mut self, count: usize) {
        self.record_counter += count;
    }

    pub fn force_chunk_size_adjustment(&mut self, new_size: usize) {
        self.current_chunk_size_bytes = new_size.max(MIN_CHUNK_SIZE_LIMIT);
        self.last_chunk_adjustment_record = self.record_counter;
    }

    pub fn release_chunk_resources(&mut self) -> Result<()> {
        self.clear_ram_buffer();

        std::hint::black_box(());

        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn get_resource_usage_stats_with_gpu(
        &self,
        cuda_processor: Option<&crate::cuda::processor::CudaProcessor>
    ) -> Result<ResourceUsageStats> {
        let (cpu_usage, cpu_percent) = self.resources.get_current_memory_usage()?;
        let cpu_pressure = self.resources.is_memory_pressure()?;

        let (gpu_free, gpu_total, gpu_pressure) = if let Some(processor) = cuda_processor {
            let (free, total) = processor.get_gpu_memory_usage()
                .map_err(|e| anyhow::anyhow!("Failed to get GPU memory usage: {}", e))?;
            let pressure = processor.check_gpu_memory_pressure()?;
            (free, total, pressure)
        } else {
            (DEFAULT_GPU_MEMORY_FREE, DEFAULT_GPU_MEMORY_TOTAL, DEFAULT_GPU_PRESSURE)
        };

        Ok(ResourceUsageStats {
            cpu_memory_used_bytes: cpu_usage,
            cpu_memory_usage_percent: cpu_percent,
            cpu_memory_pressure: cpu_pressure,
            ram_buffer_used_bytes: self.ram_buffer_position,
            ram_buffer_capacity_bytes: self.ram_buffer_capacity,
            gpu_memory_free_bytes: gpu_free,
            gpu_memory_total_bytes: gpu_total,
            gpu_memory_pressure: gpu_pressure,
            current_chunk_size_bytes: self.current_chunk_size_bytes,
            records_processed: self.record_counter,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn get_resource_usage_stats(&self) -> Result<ResourceUsageStats> {
        let (cpu_usage, cpu_percent) = self.resources.get_current_memory_usage()?;
        let cpu_pressure = self.resources.is_memory_pressure()?;

        Ok(ResourceUsageStats {
            cpu_memory_used_bytes: cpu_usage,
            cpu_memory_usage_percent: cpu_percent,
            cpu_memory_pressure: cpu_pressure,
            ram_buffer_used_bytes: self.ram_buffer_position,
            ram_buffer_capacity_bytes: self.ram_buffer_capacity,
            gpu_memory_free_bytes: DEFAULT_GPU_MEMORY_FREE,
            gpu_memory_total_bytes: DEFAULT_GPU_MEMORY_TOTAL,
            gpu_memory_pressure: DEFAULT_GPU_PRESSURE,
            current_chunk_size_bytes: self.current_chunk_size_bytes,
            records_processed: self.record_counter,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ResourceUsageStats {
    pub cpu_memory_used_bytes: usize,
    pub cpu_memory_usage_percent: f64,
    pub cpu_memory_pressure: bool,
    pub ram_buffer_used_bytes: usize,
    pub ram_buffer_capacity_bytes: usize,
    pub gpu_memory_free_bytes: usize,
    pub gpu_memory_total_bytes: usize,
    pub gpu_memory_pressure: bool,
    pub current_chunk_size_bytes: usize,
    pub records_processed: usize,
}

impl ResourceUsageStats {
    pub fn format_summary(&self) -> String {
        let ram_buffer_percent = if self.ram_buffer_capacity_bytes > ZERO_USIZE {
            (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * PERCENT_100
        } else {
            ZERO_F64
        };

        let gpu_usage_percent = if self.gpu_memory_total_bytes > ZERO_USIZE {
            let used = self.gpu_memory_total_bytes - self.gpu_memory_free_bytes;
            (used as f64 / self.gpu_memory_total_bytes as f64) * PERCENT_100
        } else {
            ZERO_F64
        };

        format!(
            "Resource Usage Statistics:\n\
             CPU Memory: {:.2} GB ({:.1}%) - Pressure: {}\n\
             RAM Buffer: {:.2} MB / {:.2} MB ({:.1}%)\n\
             GPU Memory: {:.2} GB / {:.2} GB ({:.1}%) - Pressure: {}\n\
             Current Chunk Size: {:.2} MB\n\
             Records Processed: {}",
            self.cpu_memory_used_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
            self.cpu_memory_usage_percent,
            if self.cpu_memory_pressure { "YES" } else { "NO" },
            self.ram_buffer_used_bytes as f64 / BYTES_PER_MB as f64,
            self.ram_buffer_capacity_bytes as f64 / BYTES_PER_MB as f64,
            ram_buffer_percent,
            (self.gpu_memory_total_bytes - self.gpu_memory_free_bytes) as f64 / crate::constants::BYTES_PER_GB as f64,
            self.gpu_memory_total_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
            gpu_usage_percent,
            if self.gpu_memory_pressure { "YES" } else { "NO" },
            self.current_chunk_size_bytes as f64 / BYTES_PER_MB as f64,
            self.records_processed
        )
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_ram_bytes: usize,
    pub available_ram_bytes: usize,
    pub ram_limit_bytes: usize,
    pub ram_buffer_capacity_bytes: usize,
    pub ram_buffer_used_bytes: usize,
    pub current_process_usage_bytes: usize,
    pub usage_percent: f64,
    pub memory_pressure: bool,
    #[cfg(feature = "cuda")]
    pub gpu_buffer_capacity_bytes: usize,
    pub original_chunk_size_bytes: usize,
    pub current_chunk_size_bytes: usize,
    pub chunk_size_adjustment_factor: f64,
}

impl MemoryStats {
    pub fn format_summary(&self) -> String {
        #[cfg(feature = "cuda")]
        {
            let mut summary = format!(
                "Memory Statistics:\n\
                 Total RAM: {:.2} GB\n\
                 Available RAM: {:.2} GB\n\
                 RAM Limit: {:.2} GB\n\
                 RAM Buffer: {:.2} MB capacity, {:.2} MB used ({:.1}%)\n\
                 Process Usage: {:.2} GB ({:.1}%)\n\
                 Memory Pressure: {}",
                self.total_ram_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.available_ram_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.ram_limit_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.ram_buffer_capacity_bytes as f64 / BYTES_PER_MB as f64,
                self.ram_buffer_used_bytes as f64 / BYTES_PER_MB as f64,
                (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * PERCENT_100,
                self.current_process_usage_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.usage_percent,
                if self.memory_pressure { "YES" } else { "NO" }
            );

            summary.push_str(&format!(
                "\nGPU Buffer: {:.2} MB capacity",
                self.gpu_buffer_capacity_bytes as f64 / BYTES_PER_MB as f64
            ));

            summary.push_str(&format!(
                "\nDynamic Chunk Sizing: {:.2} MB original → {:.2} MB current (factor: {:.2}x)",
                self.original_chunk_size_bytes as f64 / BYTES_PER_MB as f64,
                self.current_chunk_size_bytes as f64 / BYTES_PER_MB as f64,
                self.chunk_size_adjustment_factor
            ));

            summary
        }

        #[cfg(not(feature = "cuda"))]
        {
            format!(
                "Memory Statistics:\n\
                 Total RAM: {:.2} GB\n\
                 Available RAM: {:.2} GB\n\
                 RAM Limit: {:.2} GB\n\
                 RAM Buffer: {:.2} MB capacity, {:.2} MB used ({:.1}%)\n\
                 Process Usage: {:.2} GB ({:.1}%)\n\
                 Memory Pressure: {}\n\
                 Dynamic Chunk Sizing: {:.2} MB original → {:.2} MB current (factor: {:.2}x)",
                self.total_ram_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.available_ram_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.ram_limit_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.ram_buffer_capacity_bytes as f64 / BYTES_PER_MB as f64,
                self.ram_buffer_used_bytes as f64 / BYTES_PER_MB as f64,
                (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * PERCENT_100,
                self.current_process_usage_bytes as f64 / crate::constants::BYTES_PER_GB as f64,
                self.usage_percent,
                if self.memory_pressure { "YES" } else { "NO" },
                self.original_chunk_size_bytes as f64 / BYTES_PER_MB as f64,
                self.current_chunk_size_bytes as f64 / BYTES_PER_MB as f64,
                self.chunk_size_adjustment_factor
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new(None);
        assert!(manager.is_ok(), "Should create memory manager successfully");

        let manager = manager.unwrap();
        assert!(manager.is_initialized(), "Manager should be initialized");
        assert!(manager.ram_buffer_capacity > 0, "RAM buffer should have capacity");
        assert_eq!(manager.ram_buffer_position, ZERO_USIZE, "RAM buffer should start empty");
    }

    #[test]
    fn test_ram_buffer_operations() {
        let mut manager = MemoryManager::new(Some(1)).unwrap(); // 1GB limit for testing

        let test_data = b"test data for buffer";

        let result = manager.add_to_ram_buffer(test_data);
        assert!(result.is_ok(), "Should add data successfully");
        assert!(result.unwrap(), "Should return true for successful add");

        let contents = manager.get_ram_buffer_contents();
        assert_eq!(contents, test_data, "Buffer contents should match added data");

        manager.clear_ram_buffer();
        assert_eq!(manager.ram_buffer_position, ZERO_USIZE, "Buffer position should reset");
        assert_eq!(manager.get_ram_buffer_contents().len(), 0, "Buffer should be empty");
    }

    #[test]
    fn test_memory_stats() {
        let manager = MemoryManager::new(None).unwrap();
        let stats = manager.get_memory_stats();
        assert!(stats.is_ok(), "Should get memory stats successfully");

        let stats = stats.unwrap();
        assert!(stats.total_ram_bytes > 0, "Total RAM should be positive");
        assert!(stats.ram_buffer_capacity_bytes > 0, "Buffer capacity should be positive");

        let summary = stats.format_summary();
        assert!(summary.contains("Memory Statistics"), "Summary should contain header");
    }

    #[test]
    fn test_dynamic_chunk_sizing() {
        let mut manager = MemoryManager::new(Some(1)).unwrap(); // 1GB limit for testing

        let initial_size = manager.get_current_chunk_size();
        assert!(initial_size > 0, "Initial chunk size should be positive");
        assert_eq!(initial_size, manager.original_chunk_size_bytes, "Initial size should equal original size");

        let adjusted = manager.adjust_chunk_size_if_needed().unwrap();
        assert!(!adjusted, "Should not adjust chunk size during cooldown period");

        manager.record_counter = CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS + 1;

        let result = manager.adjust_chunk_size_if_needed();
        assert!(result.is_ok(), "Chunk size adjustment should not fail");

        let stats = manager.get_memory_stats().unwrap();
        assert_eq!(stats.original_chunk_size_bytes, manager.original_chunk_size_bytes);
        assert_eq!(stats.current_chunk_size_bytes, manager.current_chunk_size_bytes);
        assert!(stats.chunk_size_adjustment_factor > 0.0, "Adjustment factor should be positive");

        let summary = stats.format_summary();
        assert!(summary.contains("Dynamic Chunk Sizing"), "Summary should contain chunk sizing info");
    }

    #[test]
    fn test_resource_cleanup() {
        let mut manager = MemoryManager::new(Some(1)).unwrap(); // 1GB limit for testing

        let test_data = b"test data for cleanup";
        let result = manager.add_to_ram_buffer(test_data);
        assert!(result.is_ok() && result.unwrap(), "Should add data successfully");

        assert!(manager.get_ram_buffer_contents().len() > 0, "Buffer should contain data");

        let cleanup_result = manager.release_chunk_resources();
        assert!(cleanup_result.is_ok(), "Resource cleanup should succeed");

        assert_eq!(manager.get_ram_buffer_contents().len(), 0, "Buffer should be empty after cleanup");
        assert_eq!(manager.ram_buffer_position, ZERO_USIZE, "Buffer position should be reset");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_resource_usage_stats_cpu_only() {
        let manager = MemoryManager::new(Some(1)).unwrap();

        let stats = manager.get_resource_usage_stats();
        assert!(stats.is_ok(), "Should get resource usage stats successfully");

        let stats = stats.unwrap();
        assert!(stats.cpu_memory_used_bytes > 0, "CPU memory usage should be positive");
        assert!(stats.cpu_memory_usage_percent >= 0.0, "CPU usage percent should be non-negative");
        assert_eq!(stats.gpu_memory_total_bytes, DEFAULT_GPU_MEMORY_TOTAL, "GPU memory should be 0 in CPU-only mode");
        assert!(!stats.gpu_memory_pressure, "GPU pressure should be false in CPU-only mode");

        let summary = stats.format_summary();
        assert!(summary.contains("Resource Usage Statistics"), "Summary should contain header");
        assert!(summary.contains("CPU Memory"), "Summary should contain CPU info");
    }
}
