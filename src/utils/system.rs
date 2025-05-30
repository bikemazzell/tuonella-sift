use sysinfo::{System, Pid};
use std::time::{Duration, Instant};
use crate::constants::{
    BYTES_PER_GB, ALGORITHM_RAM_ALLOCATION_PERCENT, MEMORY_SAFETY_MARGIN,
    MAX_RAM_BUFFER_SIZE_GB, MAX_GPU_BUFFER_SIZE_GB
};

#[cfg(feature = "cuda")]
use crate::constants::ALGORITHM_GPU_ALLOCATION_PERCENT;
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
    pub fn query_system_resources(user_ram_limit_gb: Option<usize>) -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        let total_ram_bytes = system.total_memory() as usize;
        let available_ram_bytes = system.available_memory() as usize;

        // Calculate RAM limit: min(available_ram, user_limit) * 90% as per algorithm
        let user_ram_limit_bytes = user_ram_limit_gb
            .map(|gb| gb * BYTES_PER_GB as usize)
            .unwrap_or(available_ram_bytes);

        let ram_limit_bytes = ((available_ram_bytes.min(user_ram_limit_bytes) as f64)
            * ALGORITHM_RAM_ALLOCATION_PERCENT * MEMORY_SAFETY_MARGIN) as usize;

        // RAM Buffer Size: Allocate ~90% of the RAM_limit for buffering file chunks
        // Apply safety limit to prevent OOM during testing
        let calculated_ram_buffer = (ram_limit_bytes as f64 * ALGORITHM_RAM_ALLOCATION_PERCENT) as usize;
        let max_ram_buffer_bytes = (MAX_RAM_BUFFER_SIZE_GB * BYTES_PER_GB as f64) as usize;
        let ram_buffer_size_bytes = calculated_ram_buffer.min(max_ram_buffer_bytes);

        #[cfg(feature = "cuda")]
        let (gpu_properties, gpu_limit_bytes, gpu_buffer_size_bytes) = {
            // Try to get GPU properties - this will fail gracefully if no CUDA device
            match Self::query_gpu_resources() {
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
    fn query_gpu_resources() -> Result<(CudaDeviceProperties, usize, usize)> {
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

        // Calculate GPU limit: 90% of available GPU memory as per algorithm
        let gpu_limit_bytes = ((props.free_memory as f64)
            * ALGORITHM_GPU_ALLOCATION_PERCENT * MEMORY_SAFETY_MARGIN) as usize;

        // GPU Chunk Size: Allocate ~90% of the GPU_limit for processing
        // Apply safety limit to prevent OOM during testing
        let calculated_gpu_buffer = (gpu_limit_bytes as f64 * ALGORITHM_GPU_ALLOCATION_PERCENT) as usize;
        let max_gpu_buffer_bytes = (MAX_GPU_BUFFER_SIZE_GB * BYTES_PER_GB as f64) as usize;
        let gpu_buffer_size_bytes = calculated_gpu_buffer.min(max_gpu_buffer_bytes);

        Ok((props, gpu_limit_bytes, gpu_buffer_size_bytes))
    }

    /// Get current memory usage for monitoring
    pub fn get_current_memory_usage(&self) -> Result<(usize, f64)> {
        let current_usage = get_process_memory_usage();
        let usage_percent = (current_usage as f64 / self.ram_limit_bytes as f64) * 100.0;
        Ok((current_usage, usage_percent))
    }

    /// Check if we're approaching memory limits
    pub fn is_memory_pressure(&self) -> Result<bool> {
        let (_, usage_percent) = self.get_current_memory_usage()?;
        Ok(usage_percent > 80.0)  // Consider 80% as pressure threshold
    }

    /// Get formatted resource summary for logging
    pub fn format_summary(&self) -> String {
        #[cfg(feature = "cuda")]
        {
            let mut summary = format!(
                "System Resources:\n  RAM: {:.2} GB total, {:.2} GB available, {:.2} GB limit, {:.2} GB buffer",
                self.total_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.available_ram_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_limit_bytes as f64 / BYTES_PER_GB as f64,
                self.ram_buffer_size_bytes as f64 / BYTES_PER_GB as f64
            );

            if let Some(ref props) = self.gpu_properties {
                summary.push_str(&format!(
                    "\n  GPU: {:.2} GB total, {:.2} GB free, {:.2} GB limit, {:.2} GB buffer",
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
                "System Resources:\n  RAM: {:.2} GB total, {:.2} GB available, {:.2} GB limit, {:.2} GB buffer",
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
        process.memory() as usize * 1024 // Convert KB to bytes
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
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format bytes as a human-readable string
///
/// Formats bytes as KB, MB, GB, etc.
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let bytes = bytes as f64;
    if bytes < KB {
        format!("{:.0} B", bytes)
    } else if bytes < MB {
        format!("{:.2} KB", bytes / KB)
    } else if bytes < GB {
        format!("{:.2} MB", bytes / MB)
    } else if bytes < TB {
        format!("{:.2} GB", bytes / GB)
    } else {
        format!("{:.2} TB", bytes / TB)
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
    if progress <= 0.0 {
        return None;
    }

    let total_estimated = elapsed.as_secs_f64() / progress;
    let remaining_secs = total_estimated - elapsed.as_secs_f64();

    if remaining_secs <= 0.0 {
        return Some(Duration::from_secs(0));
    }

    Some(Duration::from_secs_f64(remaining_secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_memory_info() {
        let (total, available) = get_memory_info();
        assert!(total > 0.0, "Total memory should be positive");
        assert!(available > 0.0, "Available memory should be positive");
        assert!(available <= total, "Available memory should not exceed total");
    }

    #[test]
    fn test_process_memory_usage() {
        let usage = get_process_memory_usage();
        assert!(usage > 0, "Process memory usage should be positive");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
        assert_eq!(format_duration(Duration::from_secs(61)), "00:01:01");
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
        assert_eq!(format_bytes(1536 * 1024 * 1024), "1.50 GB");
    }

    #[test]
    fn test_estimate_remaining_time() {
        let start = Instant::now();

        // No progress yet
        assert_eq!(estimate_remaining_time(start, 100, 0), None);

        // Some progress
        sleep(Duration::from_millis(100));
        let remaining = estimate_remaining_time(start, 100, 10);
        assert!(remaining.is_some());

        // Complete
        let remaining = estimate_remaining_time(start, 100, 100);
        assert_eq!(remaining.unwrap(), Duration::from_secs(0));
    }

    #[test]
    fn test_system_resources_query() {
        // Test basic resource querying with conservative limits
        let resources = SystemResources::query_system_resources(Some(1)); // Limit to 1GB to be safe
        assert!(resources.is_ok(), "Should be able to query system resources");

        let resources = resources.unwrap();
        assert!(resources.total_ram_bytes > 0, "Total RAM should be positive");
        assert!(resources.available_ram_bytes > 0, "Available RAM should be positive");
        assert!(resources.ram_limit_bytes > 0, "RAM limit should be positive");
        assert!(resources.ram_buffer_size_bytes > 0, "RAM buffer size should be positive");

        // Ensure safety limits are applied
        let max_ram_buffer = (crate::constants::MAX_RAM_BUFFER_SIZE_GB * crate::constants::BYTES_PER_GB as f64) as usize;
        assert!(resources.ram_buffer_size_bytes <= max_ram_buffer, "RAM buffer should not exceed safety limit");

        #[cfg(feature = "cuda")]
        if resources.gpu_properties.is_some() {
            let max_gpu_buffer = (crate::constants::MAX_GPU_BUFFER_SIZE_GB * crate::constants::BYTES_PER_GB as f64) as usize;
            assert!(resources.gpu_buffer_size_bytes <= max_gpu_buffer, "GPU buffer should not exceed safety limit");
        }

        // Test memory usage monitoring (should be lightweight)
        let (usage, percent) = resources.get_current_memory_usage().unwrap();
        assert!(usage > 0, "Current memory usage should be positive");
        assert!(percent >= 0.0, "Memory usage percent should be non-negative");

        // Test format summary (should not allocate memory)
        let summary = resources.format_summary();
        assert!(summary.contains("System Resources"), "Summary should contain header");
        assert!(summary.contains("RAM:"), "Summary should contain RAM info");
    }
}