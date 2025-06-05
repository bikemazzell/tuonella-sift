use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use bytes::BytesMut;

/// A thread-safe buffer pool for reusing BytesMut instances
/// to reduce memory allocations and improve performance
#[derive(Debug, Clone)]
pub struct BufferPool {
    pool: Arc<Mutex<VecDeque<BytesMut>>>,
    default_capacity: usize,
    max_pool_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool with specified default capacity and max pool size
    pub fn new(default_capacity: usize, max_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::with_capacity(max_pool_size))),
            default_capacity,
            max_pool_size,
        }
    }

    /// Get a buffer from the pool, or create a new one if pool is empty
    pub fn get(&self) -> BytesMut {
        let mut pool = self.pool.lock().unwrap();
        
        if let Some(mut buf) = pool.pop_front() {
            buf.clear(); // Clear existing data but keep capacity
            buf
        } else {
            BytesMut::with_capacity(self.default_capacity)
        }
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buf: BytesMut) {
        // Only return buffers that aren't too large to avoid memory bloat
        if buf.capacity() <= self.default_capacity * 4 {
            let mut pool = self.pool.lock().unwrap();
            
            // Only keep up to max_pool_size buffers
            if pool.len() < self.max_pool_size {
                pool.push_back(buf);
            }
        }
        // If buffer is too large or pool is full, just drop it
    }

    /// Get current pool size for monitoring
    pub fn pool_size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Get pool statistics
    pub fn stats(&self) -> BufferPoolStats {
        let pool = self.pool.lock().unwrap();
        let total_capacity: usize = pool.iter().map(|buf| buf.capacity()).sum();
        
        BufferPoolStats {
            buffers_in_pool: pool.len(),
            total_capacity,
            default_capacity: self.default_capacity,
            max_pool_size: self.max_pool_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub buffers_in_pool: usize,
    pub total_capacity: usize,
    pub default_capacity: usize,
    pub max_pool_size: usize,
}

impl BufferPoolStats {
    pub fn format_summary(&self) -> String {
        format!(
            "Buffer Pool Stats:\n\
             🏊 Buffers in Pool: {}\n\
             💾 Total Capacity: {:.2} KB\n\
             📏 Default Capacity: {:.2} KB\n\
             🔢 Max Pool Size: {}",
            self.buffers_in_pool,
            self.total_capacity as f64 / 1024.0,
            self.default_capacity as f64 / 1024.0,
            self.max_pool_size
        )
    }
}

/// RAII wrapper for automatic buffer return to pool
pub struct PooledBuffer {
    buffer: Option<BytesMut>,
    pool: BufferPool,
}

impl PooledBuffer {
    pub fn new(pool: BufferPool) -> Self {
        let buffer = pool.get();
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get mutable access to the buffer
    pub fn get_mut(&mut self) -> &mut BytesMut {
        self.buffer.as_mut().unwrap()
    }

    /// Get immutable access to the buffer
    pub fn get(&self) -> &BytesMut {
        self.buffer.as_ref().unwrap()
    }

    /// Take ownership of the buffer (prevents automatic return to pool)
    pub fn take(mut self) -> BytesMut {
        self.buffer.take().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_basic_operations() {
        let pool = BufferPool::new(1024, 5);
        
        // Get a buffer
        let buf1 = pool.get();
        assert_eq!(buf1.capacity(), 1024);
        
        // Return it to pool
        pool.return_buffer(buf1);
        assert_eq!(pool.pool_size(), 1);
        
        // Get it back - should be the same buffer
        let buf2 = pool.get();
        assert_eq!(buf2.capacity(), 1024);
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let pool = BufferPool::new(1024, 2);
        
        // Fill pool to capacity
        let buf1 = pool.get();
        let buf2 = pool.get();
        let buf3 = pool.get();
        
        pool.return_buffer(buf1);
        pool.return_buffer(buf2);
        pool.return_buffer(buf3); // This should be dropped due to max_pool_size
        
        assert_eq!(pool.pool_size(), 2);
    }

    #[test]
    fn test_pooled_buffer_raii() {
        let pool = BufferPool::new(1024, 5);
        
        {
            let mut pooled = PooledBuffer::new(pool.clone());
            pooled.get_mut().extend_from_slice(b"test data");
            assert_eq!(pooled.get().len(), 9);
            // Buffer should be returned to pool when pooled goes out of scope
        }
        
        assert_eq!(pool.pool_size(), 1);
        
        // Next buffer should be cleared but have same capacity
        let buf = pool.get();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 1024);
    }

    #[test]
    fn test_large_buffer_not_returned() {
        let pool = BufferPool::new(1024, 5);
        
        // Create a buffer with large capacity
        let mut large_buf = BytesMut::with_capacity(8192); // 8x default capacity
        large_buf.extend_from_slice(b"large data");
        
        pool.return_buffer(large_buf);
        assert_eq!(pool.pool_size(), 0); // Should not be added to pool
    }
}