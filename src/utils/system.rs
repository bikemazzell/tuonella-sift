use sysinfo::{System, Pid};
use std::time::{Duration, Instant};
use crate::constants::{
    BYTES_PER_GB, ALGORITHM_RAM_ALLOCATION_PERCENT, MEMORY_SAFETY_MARGIN,
    MEMORY_PRESSURE_THRESHOLD_PERCENT,
    SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
    KB_AS_F64, MB_AS_F64, GB_AS_F64, TB_AS_F64,
    DECIMAL_PLACES, ZERO_F64, PERCENT_100
};

#[cfg(feature = "cuda")]
use crate::constants::{ALGORITHM_GPU_ALLOCATION_PERCENT};
use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaDeviceProperties;

/// Comprehensive system resource information following algorithm specifications
#[derive(Debug, Clone)]
pub struct SystemResources {
    pub total_ram_bytes: usize,
    pub available_ram_bytes: usize,
    pub ram_limit_bytes: usize,  // 90% of available RAM as per algorithm
    pub ram_buffer_size_bytes: usize,  // Buffer size for processing
    #[cfg(feature = "cuda")]
    pub gpu_properties: Option<CudaDeviceProperties>,
    #[cfg(feature = "cuda")]
    pub gpu_limit_bytes: usize,  // 90% of available GPU memory as per algorithm
    #[cfg(feature = "cuda")]
    pub gpu_buffer_size_bytes: usize,  // GPU buffer size for processing
}

impl SystemResources {
    /// Query system resources dynamically as specified in algorithm step 1
    pub fn query_system_resources(
        ram_memory_usage_percent: Option<f64>,
        _gpu_memory_usage_percent: Option<f64>
    ) -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        let total_ram_bytes = system.total_memory() as usize;
        let available_ram_bytes = system.available_memory() as usize;

        // Calculate RAM limit: Use percentage of available RAM as per algorithm
        let ram_usage_percent = ram_memory_usage_percent.unwrap_or(50.0) / 100.0; // Default to 50%
        let ram_limit_bytes = ((available_ram_bytes as f64) * ram_usage_percent * MEMORY_SAFETY_MARGIN) as usize;

        // RAM Buffer Size: Allocate ~90% of the RAM_limit for buffering file chunks
        // Use configured RAM limit as the maximum buffer size (with safety margin)
        let calculated_ram_buffer = (ram_limit_bytes as f64 * ALGORITHM_RAM_ALLOCATION_PERCENT) as usize;
        let max_ram_buffer_bytes = (ram_limit_bytes as f64 * 0.8) as usize; // 80% of configured limit as buffer
        let ram_buffer_size_bytes = calculated_ram_buffer.min(max_ram_buffer_bytes);

        #[cfg(feature = "cuda")]
        let (gpu_properties, gpu_limit_bytes, gpu_buffer_size_bytes) = {
            // Try to get GPU properties - this will fail gracefully if no CUDA device
            match Self::query_gpu_resources(_gpu_memory_usage_percent) {
                Ok((props, limit, buffer)) => (Some(props), limit, buffer),
                Err(_) => (None, 0, 0),
            }
        };

        Ok(SystemResources {
            total_ram_bytes,
            available_ram_bytes,
            ram_limit_bytes,
            ram_buffer_size_bytes,
            #[cfg(feature = "cuda")]
            gpu_properties,
            #[cfg(feature = "cuda")]
            gpu_limit_bytes,
            #[cfg(feature = "cuda")]
            gpu_buffer_size_bytes,
        })
    }

    #[cfg(feature = "cuda")]
    fn query_gpu_resources(gpu_memory_usage_percent: Option<f64>) -> Result<(CudaDeviceProperties, usize, usize)> {
        use cudarc::driver::result;

        // Query GPU memory directly without creating full context to avoid memory allocation
        let (free_memory, total_memory) = result::mem_get_info()?;

        // Create a minimal context just to query device attributes
        use cudarc::driver::safe::CudaContext;
        use cudarc::driver::sys::CUdevice_attribute_enum;
        use std::sync::Arc;

        let context = Arc::new(CudaContext::new(0)?);
        let compute_capability_major = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let compute_capability_minor = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let max_threads_per_block = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        let max_shared_memory_per_block = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let memory_bus_width = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let l2_cache_size = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;

        let props = CudaDeviceProperties {
            total_memory,
            free_memory,
            compute_capability_major,
            compute_capability_minor,
            max_threads_per_block,
            max_shared_memory_per_block,
            memory_bus_width,
            l2_cache_size,
        };

        // Calculate GPU limit: Use configured percentage of available GPU memory
        let gpu_usage_percent = gpu_memory_usage_percent.unwrap_or(90.0) / 100.0; // Default to 90%
        let gpu_limit_bytes = ((props.free_memory as f64) * gpu_usage_percent * MEMORY_SAFETY_MARGIN) as usize;

        // GPU Chunk Size: Allocate ~80% of the GPU_limit for processing buffer
        // Use configured limit as maximum (with safety margin)
        let calculated_gpu_buffer = (gpu_limit_bytes as f64 * ALGORITHM_GPU_ALLOCATION_PERCENT) as usize;
        let max_gpu_buffer_bytes = (gpu_limit_bytes as f64 * 0.8) as usize; // 80% of configured limit as buffer
        let gpu_buffer_size_bytes = calculated_gpu_buffer.min(max_gpu_buffer_bytes);

        Ok((props, gpu_limit_bytes, gpu_buffer_size_bytes))
    }

    /// Get current memory usage for monitoring
    pub fn get_current_memory_usage(&self) -> Result<(usize, f64)> {
        let current_usage = get_process_memory_usage();
        let usage_percent = (current_usage as f64 / self.ram_limit_bytes as f64) * PERCENT_100;
        Ok((current_usage, usage_percent))
    }

    /// Check if we're approaching memory limits
    pub fn is_memory_pressure(&self) -> Result<bool> {
        let (_, usage_percent) = self.get_current_memory_usage()?;
        Ok(usage_percent > MEMORY_PRESSURE_THRESHOLD_PERCENT)
    }

    /// Get formatted resource summary for logging
    pub fn format_summary(&self) -> String {
        #[cfg(feature = "cuda")]
        {
            let mut summary = format!(
                "System Resources:\n  RAM: {:.DECIMAL_PLACES$} GB total, {:.DECIMAL_PLACES$} GB available, {:.DECIMAL_PLACES$} GB limit, {:.DECIMAL_PLACES$} GB buffer",
                self.total_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.available_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_limit_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_buffer_size_bytes as f64 / BYTES_PER_GB as f64
            );

            if let Some(ref props) = self.gpu_properties {
                summary.push_str(&format!(
                    "\n  GPU: {:.DECIMAL_PLACES$} GB total, {:.DECIMAL_PLACES$} GB free, {:.DECIMAL_PLACES$} GB limit, {:.DECIMAL_PLACES$} GB buffer",
                    props.total_memory as f64 / BYTES_PER_GB as f64,
                    props.free_memory as f64 / BYTES_PER_GB as f64,
                    self.gpu_limit_bytes as f64 / BYTES_PER_GB as f64,
                    self.gpu_buffer_size_bytes as f64 / BYTES_PER_GB as f64
                ));
            } else {
                summary.push_str("\n  GPU: Not available");
            }

            summary
        }

        #[cfg(not(feature = "cuda"))]
        {
            format!(
                "System Resources:\n  RAM: {:.DECIMAL_PLACES$} GB total, {:.DECIMAL_PLACES$} GB available, {:.DECIMAL_PLACES$} GB limit, {:.DECIMAL_PLACES$} GB buffer",
                self.total_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.available_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_limit_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_buffer_size_bytes as f64 / BYTES_PER_GB as f64
            )
        }
    }
}

/// Get information about the system's memory
///
/// Returns (total_memory_gb, available_memory_gb)
pub fn get_memory_info() -> (f64, f64) {
    let mut system = System::new_all();
    system.refresh_all();

    let total_memory = system.total_memory() as f64 / BYTES_PER_GB as f64;
    let available_memory = system.available_memory() as f64 / BYTES_PER_GB as f64;

    (total_memory, available_memory)
}

/// Get the memory usage of the current process
///
/// Returns memory usage in bytes
pub fn get_process_memory_usage() -> usize {
    let mut system = System::new_all();
    system.refresh_all();

    let pid = Pid::from_u32(std::process::id());
    if let Some(process) = system.process(pid) {
        process.memory() as usize // sysinfo already returns bytes, no conversion needed
    } else {
        0
    }
}

/// Check if the system has enough memory available
///
/// Ensures there's at least the required amount of memory available
pub fn check_memory_available(required_gb: f64) -> bool {
    let (_, available_gb) = get_memory_info();
    available_gb >= required_gb
}

/// Format a duration as a human-readable string
///
/// Formats the duration in the form "HH:MM:SS"
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / SECONDS_PER_HOUR;
    let minutes = (total_seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let seconds = total_seconds % SECONDS_PER_MINUTE;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format bytes as a human-readable string
///
/// Formats bytes as KB, MB, GB, etc.
pub fn format_bytes(bytes: usize) -> String {
    let bytes = bytes as f64;
    if bytes < KB_AS_F64 {
        format!("{:.0} B", bytes)
    } else if bytes < MB_AS_F64 {
        format!("{:.DECIMAL_PLACES$} KB", bytes / KB_AS_F64)
    } else if bytes < GB_AS_F64 {
        format!("{:.DECIMAL_PLACES$} MB", bytes / MB_AS_F64)
    } else if bytes < TB_AS_F64 {
        format!("{:.DECIMAL_PLACES$} GB", bytes / GB_AS_F64)
    } else {
        format!("{:.DECIMAL_PLACES$} TB", bytes / TB_AS_F64)
    }
}

/// Estimate the remaining time for a task
///
/// Based on the amount of work done and the time elapsed
pub fn estimate_remaining_time(
    start_time: Instant,
    total_work: usize,
    work_done: usize,
) -> Option<Duration> {
    if work_done == 0 {
        return None;
    }

    let elapsed = start_time.elapsed();
    let progress = work_done as f64 / total_work as f64;
    if progress <= ZERO_F64 {
        return None;
    }

    let total_estimated = elapsed.as_secs_f64() / progress;
    let remaining_secs = total_estimated - elapsed.as_secs_f64();

    if remaining_secs <= ZERO_F64 {
        return Some(Duration::from_secs(0));
    }

    Some(Duration::from_secs_f64(remaining_secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use crate::constants::{
        TEST_TOTAL_WORK, TEST_PARTIAL_WORK, TEST_DURATION_ZERO_SECS,
        TEST_RAM_LIMIT_GB, ZERO_USIZE, ZERO_F64, TEST_DURATION_61_SECS,
        TEST_DURATION_3661_SECS, TEST_SMALL_BYTES, TEST_MEDIUM_BYTES,
        BYTES_PER_KB, BYTES_PER_MB, TEST_SLEEP_DURATION_MS, BYTES_PER_GB,
        MAX_RAM_BUFFER_SIZE_GB
    };

    #[cfg(feature = "cuda")]
    use crate::constants::{
        MAX_GPU_BUFFER_SIZE_GB
    };

    #[test]
    fn test_memory_info() {
        let (total, available) = get_memory_info();
        assert!(total > ZERO_F64, "Total memory should be positive");
        assert!(available > ZERO_F64, "Available memory should be positive");
        assert!(available <= total, "Available memory should not exceed total");
    }

    #[test]
    fn test_process_memory_usage() {
        let usage = get_process_memory_usage();
        assert!(usage > ZERO_USIZE, "Process memory usage should be positive");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(TEST_DURATION_ZERO_SECS)), "00:00:00");
        assert_eq!(format_duration(Duration::from_secs(TEST_DURATION_61_SECS)), "00:01:01");
        assert_eq!(format_duration(Duration::from_secs(TEST_DURATION_3661_SECS)), "01:01:01");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(TEST_SMALL_BYTES), "512 B");
        assert_eq!(format_bytes(TEST_MEDIUM_BYTES), "1.50 KB");
        assert_eq!(format_bytes(TEST_MEDIUM_BYTES * BYTES_PER_KB), "1.50 MB");
        assert_eq!(format_bytes(TEST_MEDIUM_BYTES * BYTES_PER_MB), "1.50 GB");
    }

    #[test]
    fn test_estimate_remaining_time() {
        let start = Instant::now();

        // No progress yet
        assert_eq!(estimate_remaining_time(start, TEST_TOTAL_WORK, ZERO_USIZE), None);

        // Some progress
        sleep(Duration::from_millis(TEST_SLEEP_DURATION_MS));
        let remaining = estimate_remaining_time(start, TEST_TOTAL_WORK, TEST_PARTIAL_WORK);
        assert!(remaining.is_some());

        // Complete
        let remaining = estimate_remaining_time(start, TEST_TOTAL_WORK, TEST_TOTAL_WORK);
        assert_eq!(remaining.unwrap(), Duration::from_secs(TEST_DURATION_ZERO_SECS));
    }

    #[test]
    fn test_system_resources_query() {
        // Test basic resource querying with conservative limits
        let resources = SystemResources::query_system_resources(Some(TEST_RAM_LIMIT_GB as f64), None);
        assert!(resources.is_ok(), "Should be able to query system resources");

        let resources = resources.unwrap();
        assert!(resources.total_ram_bytes > ZERO_USIZE, "Total RAM should be positive");
        assert!(resources.available_ram_bytes > ZERO_USIZE, "Available RAM should be positive");
        assert!(resources.ram_limit_bytes > ZERO_USIZE, "RAM limit should be positive");
        assert!(resources.ram_buffer_size_bytes > ZERO_USIZE, "RAM buffer size should be positive");

        // Ensure safety limits are applied
        let max_ram_buffer = (MAX_RAM_BUFFER_SIZE_GB * BYTES_PER_GB as f64) as usize;
        assert!(resources.ram_buffer_size_bytes <= max_ram_buffer, "RAM buffer should not exceed safety limit");

        #[cfg(feature = "cuda")]
        if resources.gpu_properties.is_some() {
            let max_gpu_buffer = (MAX_GPU_BUFFER_SIZE_GB * BYTES_PER_GB as f64) as usize;
            assert!(resources.gpu_buffer_size_bytes <= max_gpu_buffer, "GPU buffer should not exceed safety limit");
        }

        // Test memory usage monitoring (should be lightweight)
        let (usage, percent) = resources.get_current_memory_usage().unwrap();
        assert!(usage > ZERO_USIZE, "Current memory usage should be positive");
        assert!(percent >= ZERO_F64, "Memory usage percent should be non-negative");

        // Test format summary (should not allocate memory)
        let summary = resources.format_summary();
        assert!(summary.contains("System Resources"), "Summary should contain header");
        assert!(summary.contains("RAM:"), "Summary should contain RAM info");
    }
}