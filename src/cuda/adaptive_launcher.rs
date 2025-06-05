#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaContext, CudaDevice, CudaFunction};
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, DeviceAttribute};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use crate::constants::{CUDA_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub max_threads_per_block: u32,
    pub max_shared_memory_per_block: u32,
    pub max_shared_memory_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub max_threads_per_sm: u32,
    pub warp_size: u32,
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub multiprocessor_count: u32,
    pub max_registers_per_block: u32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub num_elements: u32,
    pub element_size_bytes: u32,
    pub avg_string_length: u32,
    pub max_string_length: u32,
    pub memory_access_pattern: MemoryAccessPattern,
    pub computation_intensity: ComputationIntensity,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided { stride: u32 },
    Coalesced,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub enum ComputationIntensity {
    MemoryBound,
    ComputeBound,
    Balanced,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct OptimalLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub occupancy_percentage: f32,
    pub estimated_performance: f32,
    pub reasoning: String,
}

#[cfg(feature = "cuda")]
pub struct AdaptiveKernelLauncher {
    device_properties: DeviceProperties,
    context: Arc<CudaContext>,
    device: Arc<CudaDevice>,
    kernel_cache: HashMap<String, KernelMetrics>,
    performance_history: Vec<PerformanceRecord>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct KernelMetrics {
    register_usage: u32,
    shared_memory_usage: u32,
    local_memory_usage: u32,
    theoretical_occupancy: f32,
    measured_performance: Option<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct PerformanceRecord {
    kernel_name: String,
    launch_config: LaunchConfig,
    workload_size: u32,
    execution_time_ms: f32,
    throughput_elements_per_sec: f32,
    timestamp: std::time::Instant,
}

#[cfg(feature = "cuda")]
impl AdaptiveKernelLauncher {
    pub fn new(context: Arc<CudaContext>, device: Arc<CudaDevice>) -> Result<Self> {
        let device_properties = Self::query_device_properties(&device)?;
        
        Ok(Self {
            device_properties,
            context,
            device,
            kernel_cache: HashMap::new(),
            performance_history: Vec::new(),
        })
    }
    
    fn query_device_properties(device: &CudaDevice) -> Result<DeviceProperties> {
        Ok(DeviceProperties {
            max_threads_per_block: device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32,
            max_shared_memory_per_block: device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)? as u32,
            max_shared_memory_per_sm: device.get_attribute(DeviceAttribute::MaxSharedMemoryPerMultiprocessor)? as u32,
            max_blocks_per_sm: device.get_attribute(DeviceAttribute::MaxBlocksPerMultiprocessor)? as u32,
            max_threads_per_sm: device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)? as u32,
            warp_size: device.get_attribute(DeviceAttribute::WarpSize)? as u32,
            compute_capability_major: device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)? as u32,
            compute_capability_minor: device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)? as u32,
            multiprocessor_count: device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32,
            max_registers_per_block: device.get_attribute(DeviceAttribute::MaxRegistersPerBlock)? as u32,
        })
    }
    
    /// Calculate optimal launch configuration for a given workload
    pub fn calculate_optimal_config(
        &self,
        kernel_name: &str,
        workload: &WorkloadCharacteristics,
        kernel_function: &CudaFunction,
    ) -> Result<OptimalLaunchConfig> {
        // Get or estimate kernel metrics
        let kernel_metrics = self.get_or_estimate_kernel_metrics(kernel_name, kernel_function)?;
        
        // Calculate optimal block size based on occupancy
        let optimal_block_size = self.calculate_optimal_block_size(&kernel_metrics, workload)?;
        
        // Calculate grid size based on workload
        let grid_size = self.calculate_optimal_grid_size(optimal_block_size, workload)?;
        
        // Determine shared memory requirements
        let shared_mem_bytes = self.calculate_shared_memory_requirements(workload, optimal_block_size)?;
        
        // Calculate actual occupancy
        let occupancy = self.calculate_occupancy(&kernel_metrics, optimal_block_size, shared_mem_bytes)?;
        
        // Estimate performance
        let estimated_performance = self.estimate_performance(&kernel_metrics, optimal_block_size, grid_size, occupancy, workload)?;
        
        // Generate reasoning
        let reasoning = self.generate_config_reasoning(optimal_block_size, grid_size, occupancy, workload);
        
        Ok(OptimalLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (optimal_block_size, 1, 1),
            shared_mem_bytes,
            occupancy_percentage: occupancy * 100.0,
            estimated_performance,
            reasoning,
        })
    }
    
    fn calculate_optimal_block_size(
        &self,
        kernel_metrics: &KernelMetrics,
        workload: &WorkloadCharacteristics,
    ) -> Result<u32> {
        let max_threads = self.device_properties.max_threads_per_block;
        let warp_size = self.device_properties.warp_size;
        
        // Try different block sizes (multiples of warp size)
        let mut best_block_size = warp_size;
        let mut best_occupancy = 0.0;
        
        for block_size in (warp_size..=max_threads).step_by(warp_size as usize) {
            // Check register constraints
            let registers_per_thread = kernel_metrics.register_usage;
            let total_registers = block_size * registers_per_thread;
            if total_registers > self.device_properties.max_registers_per_block {
                continue;
            }
            
            // Check shared memory constraints
            let shared_mem_per_block = self.calculate_shared_memory_requirements(workload, block_size)?;
            if shared_mem_per_block > self.device_properties.max_shared_memory_per_block {
                continue;
            }
            
            // Calculate occupancy for this block size
            let occupancy = self.calculate_occupancy(kernel_metrics, block_size, shared_mem_per_block)?;
            
            // Consider memory access pattern efficiency
            let efficiency_factor = match workload.memory_access_pattern {
                MemoryAccessPattern::Coalesced => 1.0,
                MemoryAccessPattern::Sequential => 0.9,
                MemoryAccessPattern::Strided { stride } => {
                    if stride <= warp_size { 0.8 } else { 0.6 }
                }
                MemoryAccessPattern::Random => 0.5,
            };
            
            let adjusted_occupancy = occupancy * efficiency_factor;
            
            if adjusted_occupancy > best_occupancy {
                best_occupancy = adjusted_occupancy;
                best_block_size = block_size;
            }
        }
        
        Ok(best_block_size)
    }
    
    fn calculate_optimal_grid_size(
        &self,
        block_size: u32,
        workload: &WorkloadCharacteristics,
    ) -> Result<u32> {
        let elements_per_block = match workload.computation_intensity {
            ComputationIntensity::MemoryBound => block_size, // One element per thread
            ComputationIntensity::ComputeBound => block_size * 2, // More work per thread
            ComputationIntensity::Balanced => block_size,
        };
        
        let grid_size = (workload.num_elements + elements_per_block - 1) / elements_per_block;
        
        // Limit grid size to reasonable bounds
        let max_grid_size = self.device_properties.multiprocessor_count * 32; // 32 blocks per SM max
        Ok(grid_size.min(max_grid_size))
    }
    
    fn calculate_shared_memory_requirements(
        &self,
        workload: &WorkloadCharacteristics,
        block_size: u32,
    ) -> Result<u32> {
        // Base shared memory for cooperative processing
        let base_shared_mem = match workload.memory_access_pattern {
            MemoryAccessPattern::Coalesced => 0,
            MemoryAccessPattern::Sequential => block_size * workload.element_size_bytes,
            MemoryAccessPattern::Random => block_size * workload.element_size_bytes * 2,
            MemoryAccessPattern::Strided { .. } => block_size * workload.element_size_bytes,
        };
        
        // Additional shared memory for string processing
        let string_buffer_size = if workload.avg_string_length > 32 {
            block_size * workload.avg_string_length.min(256) // Cap at 256 chars per thread
        } else {
            0
        };
        
        let total_shared_mem = base_shared_mem + string_buffer_size;
        
        // Ensure we don't exceed limits
        Ok(total_shared_mem.min(self.device_properties.max_shared_memory_per_block))
    }
    
    fn calculate_occupancy(
        &self,
        kernel_metrics: &KernelMetrics,
        block_size: u32,
        shared_mem_bytes: u32,
    ) -> Result<f32> {
        let max_threads_per_sm = self.device_properties.max_threads_per_sm;
        let max_blocks_per_sm = self.device_properties.max_blocks_per_sm;
        let max_shared_mem_per_sm = self.device_properties.max_shared_memory_per_sm;
        let max_registers_per_sm = 65536u32; // Typical for modern GPUs
        
        // Calculate limiting factors
        let blocks_limited_by_threads = max_threads_per_sm / block_size;
        let blocks_limited_by_max_blocks = max_blocks_per_sm;
        let blocks_limited_by_shared_mem = if shared_mem_bytes > 0 {
            max_shared_mem_per_sm / shared_mem_bytes
        } else {
            max_blocks_per_sm
        };
        let blocks_limited_by_registers = if kernel_metrics.register_usage > 0 {
            max_registers_per_sm / (kernel_metrics.register_usage * block_size)
        } else {
            max_blocks_per_sm
        };
        
        // The most restrictive factor determines actual occupancy
        let actual_blocks_per_sm = blocks_limited_by_threads
            .min(blocks_limited_by_max_blocks)
            .min(blocks_limited_by_shared_mem)
            .min(blocks_limited_by_registers);
        
        let actual_threads_per_sm = actual_blocks_per_sm * block_size;
        let occupancy = actual_threads_per_sm as f32 / max_threads_per_sm as f32;
        
        Ok(occupancy.min(1.0))
    }
    
    fn estimate_performance(
        &self,
        kernel_metrics: &KernelMetrics,
        block_size: u32,
        grid_size: u32,
        occupancy: f32,
        workload: &WorkloadCharacteristics,
    ) -> Result<f32> {
        // Base performance from occupancy
        let base_performance = occupancy;
        
        // Memory bandwidth efficiency
        let memory_efficiency = match workload.memory_access_pattern {
            MemoryAccessPattern::Coalesced => 1.0,
            MemoryAccessPattern::Sequential => 0.8,
            MemoryAccessPattern::Strided { stride } => {
                if stride <= self.device_properties.warp_size { 0.7 } else { 0.4 }
            }
            MemoryAccessPattern::Random => 0.3,
        };
        
        // Computation intensity factor
        let compute_factor = match workload.computation_intensity {
            ComputationIntensity::MemoryBound => 0.6, // Limited by memory bandwidth
            ComputationIntensity::ComputeBound => 1.0, // Can fully utilize compute
            ComputationIntensity::Balanced => 0.8,
        };
        
        // Workload scaling factor
        let total_threads = block_size * grid_size;
        let parallelism_efficiency = if total_threads >= workload.num_elements {
            1.0 // Sufficient parallelism
        } else {
            total_threads as f32 / workload.num_elements as f32
        };
        
        let estimated_performance = base_performance * memory_efficiency * compute_factor * parallelism_efficiency;
        
        Ok(estimated_performance)
    }
    
    fn generate_config_reasoning(
        &self,
        block_size: u32,
        grid_size: u32,
        occupancy: f32,
        workload: &WorkloadCharacteristics,
    ) -> String {
        format!(
            "Optimal config: {}x{} blocks, {:.1}% occupancy. Chosen for {:?} access pattern with {:?} intensity. Total parallelism: {} threads for {} elements.",
            grid_size, block_size, occupancy * 100.0,
            workload.memory_access_pattern, workload.computation_intensity,
            block_size * grid_size, workload.num_elements
        )
    }
    
    fn get_or_estimate_kernel_metrics(
        &self,
        kernel_name: &str,
        _kernel_function: &CudaFunction,
    ) -> Result<KernelMetrics> {
        // Check cache first
        if let Some(metrics) = self.kernel_cache.get(kernel_name) {
            return Ok(metrics.clone());
        }
        
        // Estimate metrics based on kernel type
        let metrics = match kernel_name {
            name if name.contains("normalize") => KernelMetrics {
                register_usage: 24,
                shared_memory_usage: 16384, // 16KB for vectorized processing
                local_memory_usage: 0,
                theoretical_occupancy: 0.75,
                measured_performance: None,
            },
            name if name.contains("validate") => KernelMetrics {
                register_usage: 20,
                shared_memory_usage: 8192, // 8KB for validation buffers
                local_memory_usage: 0,
                theoretical_occupancy: 0.80,
                measured_performance: None,
            },
            name if name.contains("hash") => KernelMetrics {
                register_usage: 16,
                shared_memory_usage: 0, // No shared memory needed
                local_memory_usage: 0,
                theoretical_occupancy: 0.85,
                measured_performance: None,
            },
            _ => KernelMetrics {
                register_usage: 32, // Conservative estimate
                shared_memory_usage: 4096,
                local_memory_usage: 0,
                theoretical_occupancy: 0.60,
                measured_performance: None,
            },
        };
        
        Ok(metrics)
    }
    
    /// Record performance for future optimization
    pub fn record_performance(
        &mut self,
        kernel_name: String,
        config: LaunchConfig,
        workload_size: u32,
        execution_time_ms: f32,
    ) {
        let throughput = workload_size as f32 / (execution_time_ms / 1000.0);
        
        let record = PerformanceRecord {
            kernel_name: kernel_name.clone(),
            launch_config: config,
            workload_size,
            execution_time_ms,
            throughput_elements_per_sec: throughput,
            timestamp: std::time::Instant::now(),
        };
        
        self.performance_history.push(record);
        
        // Update kernel cache with measured performance
        if let Some(metrics) = self.kernel_cache.get_mut(&kernel_name) {
            metrics.measured_performance = Some(throughput);
        }
    }
    
    /// Get device properties
    pub fn get_device_properties(&self) -> &DeviceProperties {
        &self.device_properties
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> AdaptiveLauncherStats {
        let total_launches = self.performance_history.len();
        let avg_occupancy = self.performance_history.iter()
            .map(|r| {
                let block_threads = r.launch_config.block_dim.0 * r.launch_config.block_dim.1 * r.launch_config.block_dim.2;
                block_threads as f32 / self.device_properties.max_threads_per_block as f32
            })
            .sum::<f32>() / total_launches.max(1) as f32;
        
        let avg_throughput = self.performance_history.iter()
            .map(|r| r.throughput_elements_per_sec)
            .sum::<f32>() / total_launches.max(1) as f32;
        
        AdaptiveLauncherStats {
            total_launches,
            average_occupancy: avg_occupancy,
            average_throughput_elements_per_sec: avg_throughput,
            cached_kernels: self.kernel_cache.len(),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct AdaptiveLauncherStats {
    pub total_launches: usize,
    pub average_occupancy: f32,
    pub average_throughput_elements_per_sec: f32,
    pub cached_kernels: usize,
}

#[cfg(not(feature = "cuda"))]
pub struct AdaptiveKernelLauncher;

#[cfg(not(feature = "cuda"))]
impl AdaptiveKernelLauncher {
    pub fn new(_context: std::sync::Arc<()>, _device: std::sync::Arc<()>) -> anyhow::Result<Self> {
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "cuda")]
    #[test]
    fn test_device_properties() {
        let props = DeviceProperties {
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            max_shared_memory_per_sm: 98304,
            max_blocks_per_sm: 16,
            max_threads_per_sm: 2048,
            warp_size: 32,
            compute_capability_major: 7,
            compute_capability_minor: 5,
            multiprocessor_count: 20,
            max_registers_per_block: 65536,
        };
        
        assert_eq!(props.max_threads_per_block, 1024);
        assert_eq!(props.warp_size, 32);
        assert_eq!(props.compute_capability_major, 7);
    }
    
    #[cfg(feature = "cuda")]
    #[test]
    fn test_workload_characteristics() {
        let workload = WorkloadCharacteristics {
            num_elements: 10000,
            element_size_bytes: 64,
            avg_string_length: 32,
            max_string_length: 255,
            memory_access_pattern: MemoryAccessPattern::Coalesced,
            computation_intensity: ComputationIntensity::Balanced,
        };
        
        assert_eq!(workload.num_elements, 10000);
        assert_eq!(workload.memory_access_pattern, MemoryAccessPattern::Coalesced);
        assert_eq!(workload.computation_intensity, ComputationIntensity::Balanced);
    }
    
    #[test]
    fn test_adaptive_launcher_creation() {
        #[cfg(not(feature = "cuda"))]
        {
            let launcher = AdaptiveKernelLauncher::new(std::sync::Arc::new(()), std::sync::Arc::new(()));
            assert!(launcher.is_ok());
        }
    }
}