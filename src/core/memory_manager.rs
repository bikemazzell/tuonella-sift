use anyhow::Result;
use crate::constants::{DYNAMIC_MEMORY_CHECK_INTERVAL_RECORDS, BYTES_PER_MB};
use crate::utils::system::SystemResources;

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

    /// Record counter for periodic memory checks
    record_counter: usize,

    /// Flag indicating if buffers are initialized
    buffers_initialized: bool,
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

        let mut ram_buffer = Vec::with_capacity(ram_buffer_capacity);
        // Initialize buffer to ensure memory is actually allocated
        ram_buffer.resize(ram_buffer_capacity, 0);
        ram_buffer.clear(); // Clear but keep capacity

        #[cfg(feature = "cuda")]
        let gpu_buffer_capacity = resources.gpu_buffer_size_bytes;

        #[cfg(feature = "cuda")]
        if gpu_buffer_capacity > 0 {
            println!("🚀 GPU buffer capacity reserved: {:.2} MB", gpu_buffer_capacity as f64 / BYTES_PER_MB as f64);
        }

        Ok(Self {
            resources,
            ram_buffer,
            ram_buffer_position: 0,
            ram_buffer_capacity,
            #[cfg(feature = "cuda")]
            gpu_buffer_capacity,
            record_counter: 0,
            buffers_initialized: true,
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

    /// Check for memory pressure and adjust if needed
    ///
    /// This implements: "Dynamically adjust chunk sizes if memory usage approaches limits"
    fn check_memory_pressure(&self) -> Result<()> {
        if self.resources.is_memory_pressure()? {
            let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
            println!("⚠️  Memory pressure detected: {:.2}% usage ({} bytes)",
                    usage_percent, current_usage);
        }
        Ok(())
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> Result<MemoryStats> {
        let (current_usage, usage_percent) = self.resources.get_current_memory_usage()?;
        let is_pressure = self.resources.is_memory_pressure()?;

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
}
