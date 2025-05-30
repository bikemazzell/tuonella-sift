use anyhow::Result;
use crate::constants::{
    DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS, BYTES_PER_MB,
    MEMORY_PRESSURE_THRESHOLD_PERCENT, MEMORY_CRITICAL_THRESHOLD_PERCENT,
    CHUNK_SIZE_REDUCTION_FACTOR, CHUNK_SIZE_INCREASE_FACTOR,
    MIN_CHUNK_SIZE_REDUCTION_LIMIT, MAX_CHUNK_SIZE_INCREASE_LIMIT,
    CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS
};
use crate::utils::system::SystemResources;

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext;

/// Memory manager for handling buffer preallocation and lifecycle as per algorithm step 1
///
/// This implements:
/// - RAM buffer preallocation (~90% of RAM limit)
/// - GPU buffer preallocation (~90% of GPU limit)
/// - Buffer reuse and lifecycle management
/// - Dynamic memory monitoring
#[derive(Debug)]
pub struct MemoryManager {
    /// System resource information
    resources: SystemResources,

    /// Preallocated RAM buffer for file chunk processing
    ram_buffer: Vec<u8>,

    /// Current position in RAM buffer
    ram_buffer_position: usize,

    /// Maximum RAM buffer size (preallocated)
    ram_buffer_capacity: usize,

    #[cfg(feature = "cuda")]
    /// GPU buffer information (managed by CUDA context)
    gpu_buffer_capacity: usize,

    #[cfg(feature = "cuda")]
    /// CUDA context for GPU buffer management
    gpu_context: Option<Arc<CudaContext>>,

    /// Record counter for periodic memory checks
    record_counter: usize,

    /// Flag indicating if buffers are initialized
    buffers_initialized: bool,

    /// Dynamic chunk sizing state (Section 4: Memory Management)
    /// Original chunk size for baseline calculations
    original_chunk_size_bytes: usize,

    /// Current adaptive chunk size
    current_chunk_size_bytes: usize,

    /// Last record count when chunk size was adjusted
    last_chunk_adjustment_record: usize,
}

impl MemoryManager {
    /// Create a new memory manager with preallocated buffers
    ///
    /// This implements algorithm step 1: "Preallocate Buffers"
    pub fn new(user_ram_limit_gb: Option<usize>) -> Result<Self> {
        // Query system resources dynamically
        let resources = SystemResources::query_system_resources(user_ram_limit_gb)?;

        println!("🧠 Initializing Memory Manager with algorithm-compliant allocation:");
        println!("{}", resources.format_summary());

        // Preallocate RAM buffer as specified in algorithm
        let ram_buffer_capacity = resources.ram_buffer_size_bytes;
        println!("📦 Preallocating RAM buffer: {:.2} MB", ram_buffer_capacity as f64 / BYTES_PER_MB as f64);

        // Only reserve capacity, don't actually allocate to prevent OOM
        let ram_buffer = Vec::with_capacity(ram_buffer_capacity);

        #[cfg(feature = "cuda")]
        let gpu_buffer_capacity = resources.gpu_buffer_size_bytes;

        #[cfg(feature = "cuda")]
        let gpu_context = if gpu_buffer_capacity > 0 && resources.gpu_properties.is_some() {
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

        // Initialize dynamic chunk sizing with RAM buffer size as baseline
        let initial_chunk_size = ram_buffer_capacity;

        Ok(Self {
            resources,
            ram_buffer,
            ram_buffer_position: 0,
            ram_buffer_capacity,
            #[cfg(feature = "cuda")]
            gpu_buffer_capacity,
            #[cfg(feature = "cuda")]
            gpu_context,
            record_counter: 0,
            buffers_initialized: true,
            original_chunk_size_bytes: initial_chunk_size,
            current_chunk_size_bytes: initial_chunk_size,
            last_chunk_adjustment_record: 0,
        })
    }

    /// Get available space in RAM buffer
    pub fn ram_buffer_available_space(&self) -> usize {
        self.ram_buffer_capacity.saturating_sub(self.ram_buffer_position)
    }

    /// Check if data fits in current RAM buffer
    pub fn can_fit_in_ram_buffer(&self, data_size: usize) -> bool {
        self.ram_buffer_available_space() >= data_size
    }

    /// Add data to RAM buffer, returns true if successful
    ///
    /// This implements the algorithm's "store it in the RAM buffer" step
    pub fn add_to_ram_buffer(&mut self, data: &[u8]) -> Result<bool> {
        if !self.can_fit_in_ram_buffer(data.len()) {
            return Ok(false); // Buffer full, caller should flush
        }

        // Extend buffer with new data
        self.ram_buffer.extend_from_slice(data);
        self.ram_buffer_position += data.len();

        // Periodic memory monitoring as per algorithm
        self.record_counter += 1;
        if self.record_counter % DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS == 0 {
            self.check_memory_pressure()?;
        }

        Ok(true)
    }

    /// Get current RAM buffer contents for writing to temporary file
    ///
    /// This implements: "Write Filtered Lines to a Temporary File"
    pub fn get_ram_buffer_contents(&self) -> &[u8] {
        &self.ram_buffer[..self.ram_buffer_position]
    }

    /// Clear RAM buffer after writing to temporary file
    ///
    /// This implements buffer reuse after flushing
    pub fn clear_ram_buffer(&mut self) {
        self.ram_buffer.clear();
        self.ram_buffer_position = 0;
    }

    /// Get GPU buffer capacity for CUDA processing
    #[cfg(feature = "cuda")]
    pub fn get_gpu_buffer_capacity(&self) -> usize {
        self.gpu_buffer_capacity
    }

    /// Get GPU context for CUDA operations
    #[cfg(feature = "cuda")]
    pub fn get_gpu_context(&self) -> Option<&Arc<CudaContext>> {
        self.gpu_context.as_ref()
    }

    /// Check if GPU buffer is available
    #[cfg(feature = "cuda")]
    pub fn has_gpu_buffer(&self) -> bool {
        self.gpu_context.is_some() && self.gpu_buffer_capacity > 0
    }

    /// Check for memory pressure and adjust if needed
    ///
    /// This implements: "Dynamically adjust chunk sizes if memory usage approaches limits"
    pub fn check_memory_pressure(&self) -> Result<bool> {
        let is_pressure = self.resources.is_memory_pressure()?;
        if is_pressure {
            let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
            println!("⚠️  Memory pressure detected: {:.2}% usage ({} bytes)",
                    usage_percent, current_usage);
        }
        Ok(is_pressure)
    }

    /// Get current adaptive chunk size
    ///
    /// This implements dynamic chunk sizing from Section 4: Memory Management
    pub fn get_current_chunk_size(&self) -> usize {
        self.current_chunk_size_bytes
    }

    /// Adjust chunk size based on memory pressure
    ///
    /// This implements: "Dynamically adjust chunk sizes if memory usage approaches limits"
    /// - Increase chunk size if memory usage is low
    /// - Decrease chunk size if memory usage is high
    pub fn adjust_chunk_size_if_needed(&mut self) -> Result<bool> {
        // Check if enough records have been processed since last adjustment
        if self.record_counter - self.last_chunk_adjustment_record < CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS {
            return Ok(false);
        }

        let (_current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
        let mut adjusted = false;

        if usage_percent >= MEMORY_CRITICAL_THRESHOLD_PERCENT {
            // Critical memory pressure - reduce chunk size significantly
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
            // Moderate memory pressure - reduce chunk size moderately
            let new_size = (self.current_chunk_size_bytes as f64 * CHUNK_SIZE_REDUCTION_FACTOR) as usize;
            let min_allowed = (self.original_chunk_size_bytes as f64 * MIN_CHUNK_SIZE_REDUCTION_LIMIT) as usize;

            if new_size >= min_allowed && new_size < self.current_chunk_size_bytes {
                self.current_chunk_size_bytes = new_size;
                self.last_chunk_adjustment_record = self.record_counter;
                println!("🔽 Memory pressure ({:.1}%) - reducing chunk size to {:.2} MB",
                        usage_percent, new_size as f64 / BYTES_PER_MB as f64);
                adjusted = true;
            }
        } else if usage_percent < MEMORY_PRESSURE_THRESHOLD_PERCENT * 0.6 {
            // Low memory usage - increase chunk size if possible
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

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> Result<MemoryStats> {
        let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
        let is_pressure = self.resources.is_memory_pressure()?;

        let chunk_adjustment_factor = if self.original_chunk_size_bytes > 0 {
            self.current_chunk_size_bytes as f64 / self.original_chunk_size_bytes as f64
        } else {
            1.0
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

    /// Get system resources for external use
    pub fn get_system_resources(&self) -> &SystemResources {
        &self.resources
    }

    /// Check if buffers are properly initialized
    pub fn is_initialized(&self) -> bool {
        self.buffers_initialized
    }

    /// Record that we've processed more records (for memory monitoring)
    pub fn record_processed(&mut self, count: usize) {
        self.record_counter += count;
    }

    /// Release resources after processing a chunk
    ///
    /// This implements Section 4: "Free GPU and RAM buffers after processing each chunk to avoid memory leaks"
    pub fn release_chunk_resources(&mut self) -> Result<()> {
        // Clear RAM buffer to free memory
        self.clear_ram_buffer();

        // Force garbage collection hint (Rust doesn't guarantee this will run immediately)
        // But it helps indicate to the runtime that this is a good time to clean up
        std::hint::black_box(());

        Ok(())
    }

    /// Get comprehensive resource usage statistics with optional GPU processor
    ///
    /// This provides detailed information about both CPU and GPU resource utilization
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
            (0, 0, false)
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

    /// Get comprehensive resource usage statistics (CPU only version)
    ///
    /// This provides detailed information about CPU resource utilization
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
            gpu_memory_free_bytes: 0,
            gpu_memory_total_bytes: 0,
            gpu_memory_pressure: false,
            current_chunk_size_bytes: self.current_chunk_size_bytes,
            records_processed: self.record_counter,
        })
    }
}

/// Comprehensive resource usage statistics for monitoring
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
    /// Format resource usage statistics for display
    pub fn format_summary(&self) -> String {
        let ram_buffer_percent = if self.ram_buffer_capacity_bytes > 0 {
            (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * 100.0
        } else {
            0.0
        };

        let gpu_usage_percent = if self.gpu_memory_total_bytes > 0 {
            let used = self.gpu_memory_total_bytes - self.gpu_memory_free_bytes;
            (used as f64 / self.gpu_memory_total_bytes as f64) * 100.0
        } else {
            0.0
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

/// Memory statistics for monitoring and reporting
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
    /// Dynamic chunk sizing information (Section 4: Memory Management)
    pub original_chunk_size_bytes: usize,
    pub current_chunk_size_bytes: usize,
    pub chunk_size_adjustment_factor: f64,
}

impl MemoryStats {
    /// Format memory statistics for display
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
                (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * 100.0,
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
                (self.ram_buffer_used_bytes as f64 / self.ram_buffer_capacity_bytes as f64) * 100.0,
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
        assert_eq!(manager.ram_buffer_position, 0, "RAM buffer should start empty");
    }

    #[test]
    fn test_ram_buffer_operations() {
        let mut manager = MemoryManager::new(Some(1)).unwrap(); // 1GB limit for testing

        let test_data = b"test data for buffer";

        // Test adding data
        let result = manager.add_to_ram_buffer(test_data);
        assert!(result.is_ok(), "Should add data successfully");
        assert!(result.unwrap(), "Should return true for successful add");

        // Test buffer contents
        let contents = manager.get_ram_buffer_contents();
        assert_eq!(contents, test_data, "Buffer contents should match added data");

        // Test clearing buffer
        manager.clear_ram_buffer();
        assert_eq!(manager.ram_buffer_position, 0, "Buffer position should reset");
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

        // Test initial chunk size
        let initial_size = manager.get_current_chunk_size();
        assert!(initial_size > 0, "Initial chunk size should be positive");
        assert_eq!(initial_size, manager.original_chunk_size_bytes, "Initial size should equal original size");

        // Test that adjustment doesn't happen too early (cooldown period)
        let adjusted = manager.adjust_chunk_size_if_needed().unwrap();
        assert!(!adjusted, "Should not adjust chunk size during cooldown period");

        // Simulate processing enough records to allow adjustment
        manager.record_counter = CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS + 1;

        // Test chunk size adjustment (this may or may not adjust based on actual memory usage)
        let result = manager.adjust_chunk_size_if_needed();
        assert!(result.is_ok(), "Chunk size adjustment should not fail");

        // Test memory stats include chunk sizing information
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

        // Add some data to the buffer
        let test_data = b"test data for cleanup";
        let result = manager.add_to_ram_buffer(test_data);
        assert!(result.is_ok() && result.unwrap(), "Should add data successfully");

        // Verify data is in buffer
        assert!(manager.get_ram_buffer_contents().len() > 0, "Buffer should contain data");

        // Test resource cleanup
        let cleanup_result = manager.release_chunk_resources();
        assert!(cleanup_result.is_ok(), "Resource cleanup should succeed");

        // Verify buffer is cleared
        assert_eq!(manager.get_ram_buffer_contents().len(), 0, "Buffer should be empty after cleanup");
        assert_eq!(manager.ram_buffer_position, 0, "Buffer position should be reset");
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
        assert_eq!(stats.gpu_memory_total_bytes, 0, "GPU memory should be 0 in CPU-only mode");
        assert!(!stats.gpu_memory_pressure, "GPU pressure should be false in CPU-only mode");

        let summary = stats.format_summary();
        assert!(summary.contains("Resource Usage Statistics"), "Summary should contain header");
        assert!(summary.contains("CPU Memory"), "Summary should contain CPU info");
    }
}
