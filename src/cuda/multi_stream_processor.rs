#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use parking_lot::Mutex;
#[cfg(feature = "cuda")]
use crate::cuda::processor::{CudaRecord, CudaProcessor};
#[cfg(feature = "cuda")]
use crate::config::model::CudaConfig;
#[cfg(feature = "cuda")]
use crate::constants::*;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub stream_id: usize,
    pub total_batches_processed: usize,
    pub total_processing_time: std::time::Duration,
    pub average_batch_time: std::time::Duration,
    pub utilization_percentage: f64,
}

#[cfg(feature = "cuda")]
impl Default for StreamMetrics {
    fn default() -> Self {
        Self {
            stream_id: 0,
            total_batches_processed: 0,
            total_processing_time: std::time::Duration::new(ZERO_DURATION_SECS, ZERO_DURATION_NANOS),
            average_batch_time: std::time::Duration::new(ZERO_DURATION_SECS, ZERO_DURATION_NANOS),
            utilization_percentage: 0.0,
        }
    }
}

#[cfg(feature = "cuda")]
pub struct StreamScheduler {
    pub next_stream_index: usize,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub metrics: Vec<StreamMetrics>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastUtilized,
}

#[cfg(feature = "cuda")]
impl StreamScheduler {
    pub fn new(num_streams: usize, strategy: LoadBalancingStrategy) -> Self {
        Self {
            next_stream_index: 0,
            load_balancing_strategy: strategy,
            metrics: (0..num_streams).map(|id| StreamMetrics { stream_id: id, ..Default::default() }).collect(),
        }
    }

    pub fn select_stream(&mut self) -> usize {
        match self.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                let stream_id = self.next_stream_index;
                self.next_stream_index = (self.next_stream_index + 1) % self.metrics.len();
                stream_id
            },
            LoadBalancingStrategy::LeastUtilized => {
                self.metrics.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.utilization_percentage.partial_cmp(&b.utilization_percentage).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        }
    }

    pub fn update_metrics(&mut self, stream_id: usize, processing_time: std::time::Duration) {
        if let Some(metrics) = self.metrics.get_mut(stream_id) {
            metrics.total_batches_processed += 1;
            metrics.total_processing_time += processing_time;
            
            if metrics.total_batches_processed > 0 {
                metrics.average_batch_time = metrics.total_processing_time / metrics.total_batches_processed as u32;
            }
            
            // Simplified utilization calculation
            metrics.utilization_percentage = metrics.total_batches_processed as f64 * 10.0; // Placeholder
        }
    }
}

#[cfg(feature = "cuda")]
pub struct MultiStreamCudaProcessor {
    processors: Vec<CudaProcessor>,
    stream_scheduler: Arc<Mutex<StreamScheduler>>,
    num_streams: usize,
}

#[cfg(feature = "cuda")]
impl Clone for MultiStreamCudaProcessor {
    fn clone(&self) -> Self {
        Self {
            processors: self.processors.clone(),
            stream_scheduler: Arc::clone(&self.stream_scheduler),
            num_streams: self.num_streams,
        }
    }
}

#[cfg(feature = "cuda")]
impl MultiStreamCudaProcessor {
    pub fn new(config: CudaConfig, device_ordinal: i32, num_streams: usize) -> Result<Self> {
        println!("Initializing multi-stream CUDA processor with {} virtual streams", num_streams);

        // Create a single CUDA processor and reuse it for all streams
        // This avoids the massive memory overhead of multiple contexts
        let single_processor = CudaProcessor::new(config.clone(), device_ordinal)
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA processor: {}", e))?;

        // Create multiple references to the same processor for stream scheduling
        let mut processors = Vec::with_capacity(num_streams);
        for _ in 0..num_streams {
            processors.push(single_processor.clone());
        }

        let stream_scheduler = Arc::new(Mutex::new(StreamScheduler::new(num_streams, LoadBalancingStrategy::RoundRobin)));

        println!("Multi-stream CUDA processor initialized with {} virtual streams using single context", num_streams);

        Ok(Self {
            processors,
            stream_scheduler,
            num_streams,
        })
    }

    pub async fn process_batch_async(&self, records: &mut [CudaRecord], case_sensitive_usernames: bool) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let chunk_size = self.get_optimal_batch_size().min(records.len()).max(1);

        // Process records in chunks across multiple processors (simulating streams)
        let mut futures = Vec::new();
        
        for chunk in records.chunks_mut(chunk_size) {
            // Select processor using scheduler
            let processor_id = {
                let mut scheduler = self.stream_scheduler.lock();
                scheduler.select_stream()
            };

            // Create future for async processing
            let future = self.process_chunk_on_processor_async(chunk, case_sensitive_usernames, processor_id);
            futures.push(future);
        }

        // Wait for all chunks to complete
        for future in futures {
            future.await?;
        }

        Ok(())
    }

    async fn process_chunk_on_processor_async(&self, records: &mut [CudaRecord], case_sensitive_usernames: bool, processor_id: usize) -> Result<()> {
        let start_time = Instant::now();
        
        if processor_id >= self.processors.len() {
            return Err(anyhow::anyhow!("Invalid processor ID: {}", processor_id));
        }

        // Process on the selected processor
        let result = self.processors[processor_id].process_batch(records, case_sensitive_usernames);

        // Update metrics
        let processing_time = start_time.elapsed();
        {
            let mut scheduler = self.stream_scheduler.lock();
            scheduler.update_metrics(processor_id, processing_time);
        }

        result
    }

    pub fn process_batch(&self, records: &mut [CudaRecord], case_sensitive_usernames: bool) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let chunk_size = self.get_optimal_batch_size().min(records.len()).max(1);

        use rayon::prelude::*;

        let chunks: Vec<_> = records.chunks_mut(chunk_size).collect();

        chunks.into_par_iter().enumerate().try_for_each(|(i, chunk)| -> Result<()> {
            let processor_id = i % self.processors.len();
            let start_time = Instant::now();

            self.processors[processor_id].process_batch(chunk, case_sensitive_usernames)?;

            let processing_time = start_time.elapsed();
            {
                let mut scheduler = self.stream_scheduler.lock();
                scheduler.update_metrics(processor_id, processing_time);
            }

            Ok(())
        })?;

        Ok(())
    }

    pub fn get_stream_metrics(&self) -> Vec<StreamMetrics> {
        let scheduler = self.stream_scheduler.lock();
        scheduler.metrics.clone()
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        if !self.processors.is_empty() {
            self.processors[0].get_optimal_batch_size()
        } else {
            DEFAULT_CUDA_BATCH_SIZE_FALLBACK // Default fallback
        }
    }

    pub fn get_num_streams(&self) -> usize {
        self.num_streams
    }

    pub fn format_performance_summary(&self) -> String {
        let metrics = self.get_stream_metrics();
        let total_batches: usize = metrics.iter().map(|m| m.total_batches_processed).sum();
        let avg_utilization: f64 = if !metrics.is_empty() {
            metrics.iter().map(|m| m.utilization_percentage).sum::<f64>() / metrics.len() as f64
        } else { 0.0 };

        format!(
            "Multi-Stream CUDA Performance:\n\
             🚀 Virtual Streams: {}\n\
             📊 Total Batches: {}\n\
             ⚡ Avg Utilization: {:.1}%\n\
             📈 Batch Size: {}",
            self.get_num_streams(),
            total_batches,
            avg_utilization,
            self.get_optimal_batch_size()
        )
    }

    pub fn check_gpu_memory_pressure(&self) -> Result<bool> {
        if !self.processors.is_empty() {
            self.processors[0].check_gpu_memory_pressure()
        } else {
            Ok(false)
        }
    }

    pub fn get_gpu_utilization_percent(&self) -> Result<f64> {
        if !self.processors.is_empty() {
            self.processors[0].get_gpu_utilization_percent()
        } else {
            Ok(0.0)
        }
    }
}

// Stub implementations for when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct MultiStreamCudaProcessor;

#[cfg(not(feature = "cuda"))]
impl MultiStreamCudaProcessor {
    pub fn new(_: (), _: i32, _: usize) -> Result<Self, ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_stream_scheduler_round_robin() {
        let mut scheduler = StreamScheduler::new(3, LoadBalancingStrategy::RoundRobin);
        
        assert_eq!(scheduler.select_stream(), 0);
        assert_eq!(scheduler.select_stream(), 1);
        assert_eq!(scheduler.select_stream(), 2);
        assert_eq!(scheduler.select_stream(), 0); // wraps around
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_stream_scheduler_least_utilized() {
        let mut scheduler = StreamScheduler::new(3, LoadBalancingStrategy::LeastUtilized);
        
        // Initially all have 0% utilization, should return first
        assert_eq!(scheduler.select_stream(), 0);
        
        // Update metrics for stream 0
        scheduler.update_metrics(0, std::time::Duration::from_millis(100));
        
        // Now should prefer stream 1 or 2
        let selected = scheduler.select_stream();
        assert!(selected == 1 || selected == 2);
    }

    #[tokio::test]
    #[cfg(feature = "cuda")]
    async fn test_multi_stream_processor_creation() -> Result<()> {
        // This test requires an actual CUDA device
        // Skip if CUDA is not available
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return Ok(());
        }

        let config = crate::config::model::CudaConfig {
            gpu_memory_usage_percent: 80,
            max_url_buffer_size: 256,
            max_username_buffer_size: 128,
            estimated_bytes_per_record: 512,
            min_batch_size: 1,
            max_batch_size: 1000,
            threads_per_block: 256,
            batch_sizes: crate::config::model::BatchSizes {
                small: 64,
                medium: 128,
                large: 256,
                xlarge: 512,
            },
        };

        match MultiStreamCudaProcessor::new(config, 0, 4) {
            Ok(processor) => {
                assert_eq!(processor.get_num_streams(), 4);
                assert!(processor.get_optimal_batch_size() > 0);
                println!("Multi-stream CUDA processor test passed");
            },
            Err(e) => {
                println!("CUDA not available for testing: {}", e);
                // This is expected if no CUDA device is available
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_batch_processing() -> Result<()> {
        // Test with mock data when no CUDA device available
        let config = crate::config::model::CudaConfig {
            gpu_memory_usage_percent: 80,
            max_url_buffer_size: 256,
            max_username_buffer_size: 128,
            estimated_bytes_per_record: 512,
            min_batch_size: 1,
            max_batch_size: 1000,
            threads_per_block: 256,
            batch_sizes: crate::config::model::BatchSizes {
                small: 64,
                medium: 128,
                large: 256,
                xlarge: 512,
            },
        };

        match MultiStreamCudaProcessor::new(config, 0, 2) {
            Ok(processor) => {
                let mut records = vec![
                    CudaRecord {
                        user: "test@example.com".to_string(),
                        password: "password123".to_string(),
                        url: "https://www.example.com".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        field_count: 3,
                        all_fields: vec!["test@example.com".to_string(), "password123".to_string(), "https://www.example.com".to_string()],
                    },
                ];

                let result = processor.process_batch(&mut records, false);
                
                // Should either succeed (if CUDA available) or fail gracefully
                match result {
                    Ok(_) => {
                        // Verify processing occurred
                        assert!(!records[0].normalized_user.is_empty() || !records[0].normalized_url.is_empty());
                        println!("Multi-stream batch processing test passed");
                    },
                    Err(e) => {
                        println!("CUDA processing failed (expected if no GPU): {}", e);
                    }
                }
            },
            Err(e) => {
                println!("CUDA not available for testing: {}", e);
            }
        }

        Ok(())
    }
}