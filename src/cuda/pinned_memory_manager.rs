#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use parking_lot::Mutex;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferSize {
    Small,   // Up to 64KB
    Medium,  // Up to 1MB  
    Large,   // Up to 16MB
    XLarge,  // Up to 64MB
}

#[cfg(feature = "cuda")]
impl BufferSize {
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=65536 => BufferSize::Small,
            65537..=1048576 => BufferSize::Medium, 
            1048577..=16777216 => BufferSize::Large,
            _ => BufferSize::XLarge,
        }
    }
    
    pub fn to_bytes(&self) -> usize {
        match self {
            BufferSize::Small => 65536,    // 64KB
            BufferSize::Medium => 1048576, // 1MB
            BufferSize::Large => 16777216, // 16MB
            BufferSize::XLarge => 67108864, // 64MB
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PinnedBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub buffer_size_category: BufferSize,
    pub allocated_at: Instant,
    pub last_used: Instant,
    pub use_count: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for PinnedBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedBuffer {}

#[cfg(feature = "cuda")]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // CUDA pinned memory should be freed when dropped
        if !self.ptr.is_null() {
            unsafe {
                // Use CUDA driver API to free pinned memory
                let result = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut _);
                if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    eprintln!("Warning: Failed to free pinned memory: {:?}", result);
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PinnedMemoryPool {
    available_buffers: Vec<PinnedBuffer>,
    max_pool_size: usize,
    buffer_size: BufferSize,
    total_allocated: usize,
    allocation_count: usize,
    deallocation_count: usize,
}

#[cfg(feature = "cuda")]
impl PinnedMemoryPool {
    pub fn new(buffer_size: BufferSize, max_pool_size: usize) -> Self {
        Self {
            available_buffers: Vec::with_capacity(max_pool_size),
            max_pool_size,
            buffer_size,
            total_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    pub fn get_buffer(&mut self) -> Result<PinnedBuffer> {
        // Try to reuse an existing buffer
        if let Some(mut buffer) = self.available_buffers.pop() {
            buffer.last_used = Instant::now();
            buffer.use_count += 1;
            return Ok(buffer);
        }

        // Allocate a new pinned buffer
        self.allocate_new_buffer()
    }

    pub fn return_buffer(&mut self, buffer: PinnedBuffer) {
        // Only return buffers that match our size category and aren't too old
        if buffer.buffer_size_category == self.buffer_size 
           && buffer.allocated_at.elapsed().as_secs() < 3600 // Don't keep buffers older than 1 hour
           && self.available_buffers.len() < self.max_pool_size {
            self.available_buffers.push(buffer);
            self.deallocation_count += 1;
        }
        // Otherwise, buffer will be dropped and memory freed
    }

    fn allocate_new_buffer(&mut self) -> Result<PinnedBuffer> {
        let size = self.buffer_size.to_bytes();
        
        // Allocate pinned host memory using CUDA driver API
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        
        unsafe {
            let result = cudarc::driver::sys::cuMemAllocHost_v2(
                &mut ptr,
                size
            );
            
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow::anyhow!("Failed to allocate {} bytes of pinned memory: {:?}", size, result));
            }
        }

        self.total_allocated += size;
        self.allocation_count += 1;

        Ok(PinnedBuffer {
            ptr: ptr as *mut u8,
            size,
            buffer_size_category: self.buffer_size,
            allocated_at: Instant::now(),
            last_used: Instant::now(),
            use_count: 1,
        })
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            buffer_size_category: self.buffer_size,
            available_buffers: self.available_buffers.len(),
            max_pool_size: self.max_pool_size,
            total_allocated_bytes: self.total_allocated,
            allocation_count: self.allocation_count,
            deallocation_count: self.deallocation_count,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub buffer_size_category: BufferSize,
    pub available_buffers: usize,
    pub max_pool_size: usize,
    pub total_allocated_bytes: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct AllocationTracker {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
    pub allocation_history: Vec<(Instant, usize, BufferSize)>,
}

#[cfg(feature = "cuda")]
impl AllocationTracker {
    pub fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            allocation_history: Vec::new(),
        }
    }

    pub fn track_allocation(&mut self, size: usize, buffer_size: BufferSize) {
        self.total_allocations += 1;
        self.current_memory_usage += size;
        self.peak_memory_usage = self.peak_memory_usage.max(self.current_memory_usage);
        
        self.allocation_history.push((Instant::now(), size, buffer_size));
        
        // Keep only recent history (last 1000 allocations)
        if self.allocation_history.len() > 1000 {
            self.allocation_history.remove(0);
        }
    }

    pub fn track_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.current_memory_usage = self.current_memory_usage.saturating_sub(size);
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct FragmentationMonitor {
    pub fragmentation_percentage: f64,
    pub largest_available_block: usize,
    pub total_available_memory: usize,
    pub fragmentation_events: usize,
}

#[cfg(feature = "cuda")]
impl FragmentationMonitor {
    pub fn new() -> Self {
        Self {
            fragmentation_percentage: 0.0,
            largest_available_block: 0,
            total_available_memory: 0,
            fragmentation_events: 0,
        }
    }

    pub fn update_fragmentation_stats(&mut self, pools: &HashMap<BufferSize, PinnedMemoryPool>) {
        let mut total_available = 0;
        let mut largest_block = 0;

        for (buffer_size, pool) in pools {
            let available_bytes = pool.available_buffers.len() * buffer_size.to_bytes();
            total_available += available_bytes;
            largest_block = largest_block.max(buffer_size.to_bytes());
        }

        self.total_available_memory = total_available;
        self.largest_available_block = largest_block;

        // Simple fragmentation calculation
        if total_available > 0 {
            self.fragmentation_percentage = (1.0 - (largest_block as f64 / total_available as f64)) * 100.0;
        } else {
            self.fragmentation_percentage = 0.0;
        }
    }

    pub fn should_defragment(&self) -> bool {
        self.fragmentation_percentage > 25.0 // Defragment if more than 25% fragmented
    }
}

#[cfg(feature = "cuda")]
pub struct PinnedMemoryManager {
    context: Arc<CudaContext>,
    pinned_pools: Arc<Mutex<HashMap<BufferSize, PinnedMemoryPool>>>,
    allocation_tracker: Arc<Mutex<AllocationTracker>>,
    fragmentation_monitor: Arc<Mutex<FragmentationMonitor>>,
    max_total_memory: usize,
    defragmentation_enabled: bool,
}

#[cfg(feature = "cuda")]
impl PinnedMemoryManager {
    pub fn new(context: Arc<CudaContext>, max_total_memory_mb: usize) -> Result<Self> {
        let max_total_memory = max_total_memory_mb * 1024 * 1024; // Convert MB to bytes
        
        // Initialize pools for different buffer sizes
        let mut pools = HashMap::new();
        pools.insert(BufferSize::Small, PinnedMemoryPool::new(BufferSize::Small, 64));
        pools.insert(BufferSize::Medium, PinnedMemoryPool::new(BufferSize::Medium, 32));
        pools.insert(BufferSize::Large, PinnedMemoryPool::new(BufferSize::Large, 16));
        pools.insert(BufferSize::XLarge, PinnedMemoryPool::new(BufferSize::XLarge, 8));
        
        println!("Pinned memory manager initialized with {} MB limit", max_total_memory_mb);
        
        Ok(Self {
            context,
            pinned_pools: Arc::new(Mutex::new(pools)),
            allocation_tracker: Arc::new(Mutex::new(AllocationTracker::new())),
            fragmentation_monitor: Arc::new(Mutex::new(FragmentationMonitor::new())),
            max_total_memory,
            defragmentation_enabled: true,
        })
    }

    pub fn allocate_buffer(&self, size: usize) -> Result<PinnedBuffer> {
        let buffer_size = BufferSize::from_size(size);
        
        // Check if we're approaching memory limits
        {
            let tracker = self.allocation_tracker.lock();
            if tracker.current_memory_usage + buffer_size.to_bytes() > self.max_total_memory {
                return Err(anyhow::anyhow!("Would exceed pinned memory limit: {} bytes", self.max_total_memory));
            }
        }

        let buffer = {
            let mut pools = self.pinned_pools.lock();
            let pool = pools.get_mut(&buffer_size)
                .ok_or_else(|| anyhow::anyhow!("No pool available for buffer size: {:?}", buffer_size))?;
            
            pool.get_buffer()?
        };

        // Track allocation
        {
            let mut tracker = self.allocation_tracker.lock();
            tracker.track_allocation(buffer.size, buffer_size);
        }

        // Update fragmentation stats periodically
        if self.defragmentation_enabled {
            let pools = self.pinned_pools.lock();
            let mut monitor = self.fragmentation_monitor.lock();
            monitor.update_fragmentation_stats(&pools);
            
            if monitor.should_defragment() {
                monitor.fragmentation_events += 1;
                // Note: In a full implementation, we'd implement defragmentation here
                println!("Warning: Pinned memory fragmentation detected: {:.1}%", monitor.fragmentation_percentage);
            }
        }

        Ok(buffer)
    }

    pub fn deallocate_buffer(&self, buffer: PinnedBuffer) {
        let buffer_size = buffer.buffer_size_category;
        let size = buffer.size;

        // Return buffer to pool
        {
            let mut pools = self.pinned_pools.lock();
            if let Some(pool) = pools.get_mut(&buffer_size) {
                pool.return_buffer(buffer);
            }
        }

        // Track deallocation
        {
            let mut tracker = self.allocation_tracker.lock();
            tracker.track_deallocation(size);
        }
    }

    pub fn get_statistics(&self) -> PinnedMemoryStats {
        let pools = self.pinned_pools.lock();
        let tracker = self.allocation_tracker.lock();
        let monitor = self.fragmentation_monitor.lock();

        let pool_stats: Vec<PoolStats> = pools.values().map(|pool| pool.stats()).collect();
        
        PinnedMemoryStats {
            pool_stats,
            total_allocations: tracker.total_allocations,
            total_deallocations: tracker.total_deallocations,
            current_memory_usage: tracker.current_memory_usage,
            peak_memory_usage: tracker.peak_memory_usage,
            max_total_memory: self.max_total_memory,
            fragmentation_percentage: monitor.fragmentation_percentage,
            fragmentation_events: monitor.fragmentation_events,
        }
    }

    pub fn cleanup_old_buffers(&self) -> usize {
        let mut cleaned_count = 0;
        let mut pools = self.pinned_pools.lock();
        
        for pool in pools.values_mut() {
            let initial_count = pool.available_buffers.len();
            
            // Remove buffers older than 1 hour
            pool.available_buffers.retain(|buffer| {
                buffer.allocated_at.elapsed().as_secs() < 3600
            });
            
            cleaned_count += initial_count - pool.available_buffers.len();
        }
        
        cleaned_count
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct PinnedMemoryStats {
    pub pool_stats: Vec<PoolStats>,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_memory_usage: usize,
    pub peak_memory_usage: usize,
    pub max_total_memory: usize,
    pub fragmentation_percentage: f64,
    pub fragmentation_events: usize,
}

#[cfg(feature = "cuda")]
impl PinnedMemoryStats {
    pub fn format_summary(&self) -> String {
        format!(
            "Pinned Memory Manager Stats:\n\
             💾 Current Usage: {:.2} MB / {:.2} MB ({:.1}%)\n\
             📈 Peak Usage: {:.2} MB\n\
             🔄 Allocations: {} | Deallocations: {}\n\
             📊 Fragmentation: {:.1}% ({} events)\n\
             🏊 Pool Stats:\n{}",
            self.current_memory_usage as f64 / (1024.0 * 1024.0),
            self.max_total_memory as f64 / (1024.0 * 1024.0),
            (self.current_memory_usage as f64 / self.max_total_memory as f64) * 100.0,
            self.peak_memory_usage as f64 / (1024.0 * 1024.0),
            self.total_allocations,
            self.total_deallocations,
            self.fragmentation_percentage,
            self.fragmentation_events,
            self.format_pool_stats()
        )
    }
    
    fn format_pool_stats(&self) -> String {
        self.pool_stats.iter().map(|stats| {
            format!("   {:?}: {}/{} buffers ({:.2} MB allocated)", 
                    stats.buffer_size_category,
                    stats.available_buffers,
                    stats.max_pool_size,
                    stats.total_allocated_bytes as f64 / (1024.0 * 1024.0))
        }).collect::<Vec<_>>().join("\n")
    }
}

// Stub implementations for when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct PinnedMemoryManager;

#[cfg(not(feature = "cuda"))]
pub struct PinnedBuffer;

#[cfg(not(feature = "cuda"))]
impl PinnedMemoryManager {
    pub fn new(_: (), _: usize) -> Result<Self, ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_buffer_size_categorization() {
        assert_eq!(BufferSize::from_size(1024), BufferSize::Small);
        assert_eq!(BufferSize::from_size(65536), BufferSize::Small);
        assert_eq!(BufferSize::from_size(65537), BufferSize::Medium);
        assert_eq!(BufferSize::from_size(1048576), BufferSize::Medium);
        assert_eq!(BufferSize::from_size(1048577), BufferSize::Large);
        assert_eq!(BufferSize::from_size(16777216), BufferSize::Large);
        assert_eq!(BufferSize::from_size(16777217), BufferSize::XLarge);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_allocation_tracker() {
        let mut tracker = AllocationTracker::new();
        
        tracker.track_allocation(1024, BufferSize::Small);
        assert_eq!(tracker.current_memory_usage, 1024);
        assert_eq!(tracker.peak_memory_usage, 1024);
        assert_eq!(tracker.total_allocations, 1);
        
        tracker.track_allocation(2048, BufferSize::Medium);
        assert_eq!(tracker.current_memory_usage, 3072);
        assert_eq!(tracker.peak_memory_usage, 3072);
        
        tracker.track_deallocation(1024);
        assert_eq!(tracker.current_memory_usage, 2048);
        assert_eq!(tracker.peak_memory_usage, 3072); // Peak should remain
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_fragmentation_monitor() {
        let mut monitor = FragmentationMonitor::new();
        let pools = HashMap::new(); // Empty pools
        
        monitor.update_fragmentation_stats(&pools);
        assert_eq!(monitor.fragmentation_percentage, 0.0);
        assert!(!monitor.should_defragment());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pinned_memory_manager_creation() {
        // This test requires actual CUDA context
        // Will fail gracefully if CUDA is not available
        
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return; // Skip if no CUDA
        }

        match cudarc::driver::safe::CudaContext::new(0) {
            Ok(context) => {
                match PinnedMemoryManager::new(context, 64) {
                    Ok(manager) => {
                        let stats = manager.get_statistics();
                        assert_eq!(stats.max_total_memory, 64 * 1024 * 1024);
                        println!("Pinned memory manager test passed");
                    },
                    Err(e) => {
                        println!("Failed to create pinned memory manager: {}", e);
                    }
                }
            },
            Err(e) => {
                println!("CUDA context creation failed: {}", e);
            }
        }
    }
}