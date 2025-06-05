use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::{mpsc, Semaphore};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use crossbeam::queue::SegQueue;
use crate::core::record::Record;
use crate::core::buffer_pool::{BufferPool, PooledBuffer};
use crate::constants::{
    PARALLEL_FILE_PROCESSING_THREADS, STREAMING_CHUNK_SIZE_MB,
    BYTES_PER_MB
};

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub file_path: PathBuf,
    pub chunk_offset: usize,
    pub chunk_size: usize,
    pub priority: u8,
    pub file_size: usize,
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub work_item: WorkItem,
    pub records: Vec<Record>,
    pub processing_time: Duration,
    pub errors: Vec<String>,
    pub bytes_processed: usize,
}

#[derive(Debug, Clone)]
pub struct AsyncParallelProcessingMetrics {
    pub total_files_processed: usize,
    pub total_chunks_processed: usize,
    pub total_records_processed: usize,
    pub total_bytes_processed: usize,
    pub total_processing_time: Duration,
    pub average_chunk_processing_time: Duration,
    pub average_records_per_chunk: f64,
    pub throughput_mb_per_sec: f64,
    pub concurrent_tasks: usize,
    pub queue_utilization: f64,
    pub parallel_efficiency: f64,
}

impl Default for AsyncParallelProcessingMetrics {
    fn default() -> Self {
        Self {
            total_files_processed: 0,
            total_chunks_processed: 0,
            total_records_processed: 0,
            total_bytes_processed: 0,
            total_processing_time: Duration::new(0, 0),
            average_chunk_processing_time: Duration::new(0, 0),
            average_records_per_chunk: 0.0,
            throughput_mb_per_sec: 0.0,
            concurrent_tasks: 0,
            queue_utilization: 0.0,
            parallel_efficiency: 1.0,
        }
    }
}

pub struct AsyncParallelProcessor {
    max_concurrent_tasks: usize,
    streaming_chunk_size: usize,
    buffer_pool: BufferPool,
    metrics: AsyncParallelProcessingMetrics,
}

impl AsyncParallelProcessor {
    pub fn new() -> Self {
        Self::with_max_concurrent_tasks(PARALLEL_FILE_PROCESSING_THREADS)
    }

    pub fn with_max_concurrent_tasks(max_concurrent_tasks: usize) -> Self {
        let max_tasks = max_concurrent_tasks.max(1).min(64); // Reasonable bounds
        let chunk_size = STREAMING_CHUNK_SIZE_MB * BYTES_PER_MB;
        
        // Create buffer pool for chunk reading
        let buffer_pool = BufferPool::new(chunk_size, max_tasks * 2);

        Self {
            max_concurrent_tasks: max_tasks,
            streaming_chunk_size: chunk_size,
            buffer_pool,
            metrics: AsyncParallelProcessingMetrics::default(),
        }
    }

    pub async fn process_files_parallel<F, Fut>(
        &mut self,
        file_paths: &[PathBuf],
        processor_fn: F,
    ) -> Result<Vec<Record>>
    where
        F: Fn(Vec<u8>, PathBuf) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Vec<Record>>> + Send,
    {
        let start_time = Instant::now();
        
        // Create work items for all files
        let work_items = self.create_work_items_for_files(file_paths).await?;
        let total_work_items = work_items.len();
        
        // Use lock-free queue for better performance
        let work_queue = Arc::new(SegQueue::new());
        for item in work_items {
            work_queue.push(item);
        }

        // Channel for collecting results
        let (result_sender, mut result_receiver) = mpsc::channel::<ProcessingResult>(1000);
        
        // Semaphore to control concurrency
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_tasks));
        
        // Spawn worker tasks
        let mut task_handles = Vec::new();
        
        for task_id in 0..self.max_concurrent_tasks {
            let queue = Arc::clone(&work_queue);
            let sender = result_sender.clone();
            let semaphore = Arc::clone(&semaphore);
            let processor = processor_fn.clone();
            let buffer_pool = self.buffer_pool.clone();
            
            let handle = tokio::spawn(async move {
                Self::worker_task(
                    task_id,
                    queue,
                    sender,
                    semaphore,
                    processor,
                    buffer_pool,
                ).await
            });
            
            task_handles.push(handle);
        }

        // Drop the original sender so the channel can close when all tasks finish
        drop(result_sender);

        // Collect results
        let mut all_records = Vec::new();
        let mut results_received = 0;

        while let Some(mut result) = result_receiver.recv().await {
            all_records.extend(result.records.drain(..));
            self.update_metrics_from_result(&result);
            results_received += 1;
            
            // Update progress
            if results_received % 100 == 0 {
                println!("Processed {}/{} chunks", results_received, total_work_items);
            }
        }

        // Wait for all tasks to complete
        for handle in task_handles {
            if let Err(e) = handle.await {
                eprintln!("Worker task failed: {:?}", e);
            }
        }

        // Update final metrics
        self.metrics.total_files_processed = file_paths.len();
        self.metrics.total_records_processed = all_records.len();
        
        let total_time = start_time.elapsed();
        self.calculate_parallel_efficiency(total_time);
        
        println!("Parallel processing completed: {} records from {} files in {:.2}s", 
                all_records.len(), file_paths.len(), total_time.as_secs_f64());

        Ok(all_records)
    }

    async fn create_work_items_for_files(&self, file_paths: &[PathBuf]) -> Result<Vec<WorkItem>> {
        let mut all_work_items = Vec::new();
        
        for file_path in file_paths {
            let file_size = tokio::fs::metadata(file_path).await?.len() as usize;
            let mut offset = 0;
            
            while offset < file_size {
                let chunk_size = (file_size - offset).min(self.streaming_chunk_size);
                all_work_items.push(WorkItem {
                    file_path: file_path.clone(),
                    chunk_offset: offset,
                    chunk_size,
                    priority: 1,
                    file_size,
                });
                offset += chunk_size;
            }
        }
        
        Ok(all_work_items)
    }

    async fn worker_task<F, Fut>(
        _task_id: usize,
        work_queue: Arc<SegQueue<WorkItem>>,
        result_sender: mpsc::Sender<ProcessingResult>,
        semaphore: Arc<Semaphore>,
        processor_fn: F,
        buffer_pool: BufferPool,
    ) where
        F: Fn(Vec<u8>, PathBuf) -> Fut + Send,
        Fut: std::future::Future<Output = Result<Vec<Record>>> + Send,
    {
        while let Some(work_item) = work_queue.pop() {
            // Acquire semaphore permit (limit concurrency)
            let _permit = semaphore.acquire().await.unwrap();
            
            let start_time = Instant::now();
            
            // Read chunk data using pooled buffer
            let chunk_data = match Self::read_file_chunk_async(&work_item, &buffer_pool).await {
                Ok(data) => data,
                Err(e) => {
                    let result = ProcessingResult {
                        work_item,
                        records: Vec::new(),
                        processing_time: start_time.elapsed(),
                        errors: vec![format!("Failed to read chunk: {}", e)],
                        bytes_processed: 0,
                    };
                    let _ = result_sender.send(result).await;
                    continue;
                }
            };

            let bytes_processed = chunk_data.len();

            // Process chunk
            let records = match processor_fn(chunk_data, work_item.file_path.clone()).await {
                Ok(records) => records,
                Err(e) => {
                    let result = ProcessingResult {
                        work_item,
                        records: Vec::new(),
                        processing_time: start_time.elapsed(),
                        errors: vec![format!("Failed to process chunk: {}", e)],
                        bytes_processed,
                    };
                    let _ = result_sender.send(result).await;
                    continue;
                }
            };

            let result = ProcessingResult {
                work_item,
                records,
                processing_time: start_time.elapsed(),
                errors: Vec::new(),
                bytes_processed,
            };

            // Send result (non-blocking)
            if result_sender.send(result).await.is_err() {
                // Receiver has been dropped, exit
                break;
            }
        }
    }

    async fn read_file_chunk_async(
        work_item: &WorkItem,
        buffer_pool: &BufferPool,
    ) -> Result<Vec<u8>> {
        let mut file = File::open(&work_item.file_path).await?;
        file.seek(SeekFrom::Start(work_item.chunk_offset as u64)).await?;

        // Use pooled buffer for reading
        let mut pooled_buffer = PooledBuffer::new(buffer_pool.clone());
        let buffer = pooled_buffer.get_mut();
        
        // Ensure buffer has enough capacity
        if buffer.capacity() < work_item.chunk_size {
            buffer.reserve(work_item.chunk_size - buffer.capacity());
        }
        
        // Read directly into buffer
        let mut temp_buffer = vec![0u8; work_item.chunk_size];
        let bytes_read = file.read(&mut temp_buffer).await?;
        temp_buffer.truncate(bytes_read);
        
        Ok(temp_buffer)
    }

    fn update_metrics_from_result(&mut self, result: &ProcessingResult) {
        self.metrics.total_chunks_processed += 1;
        self.metrics.total_processing_time += result.processing_time;
        self.metrics.total_bytes_processed += result.bytes_processed;

        if self.metrics.total_chunks_processed > 0 {
            self.metrics.average_chunk_processing_time =
                self.metrics.total_processing_time / self.metrics.total_chunks_processed as u32;

            self.metrics.average_records_per_chunk =
                self.metrics.total_records_processed as f64 / self.metrics.total_chunks_processed as f64;
        }
    }

    fn calculate_parallel_efficiency(&mut self, total_wall_time: Duration) {
        let wall_time_secs = total_wall_time.as_secs_f64();
        
        if wall_time_secs > 0.0 {
            // Calculate throughput in MB/s
            self.metrics.throughput_mb_per_sec = 
                (self.metrics.total_bytes_processed as f64 / (1024.0 * 1024.0)) / wall_time_secs;
            
            if self.max_concurrent_tasks > 1 {
                let total_cpu_time = self.metrics.total_processing_time.as_secs_f64();
                let theoretical_speedup = self.max_concurrent_tasks as f64;
                let actual_speedup = total_cpu_time / wall_time_secs;

                self.metrics.parallel_efficiency = (actual_speedup / theoretical_speedup).min(1.0);
                self.metrics.concurrent_tasks = self.max_concurrent_tasks;
            }
        }
    }

    pub fn get_metrics(&self) -> &AsyncParallelProcessingMetrics {
        &self.metrics
    }

    pub fn get_buffer_pool_stats(&self) -> crate::core::buffer_pool::BufferPoolStats {
        self.buffer_pool.stats()
    }
}

impl AsyncParallelProcessingMetrics {
    pub fn format_summary(&self) -> String {
        format!(
            "Async Parallel Processing Performance:\n\
             📁 Files Processed: {}\n\
             🔄 Chunks Processed: {}\n\
             📊 Records Processed: {}\n\
             💾 Bytes Processed: {:.2} MB\n\
             ⚡ Avg Chunk Time: {:.2}ms\n\
             📈 Avg Records/Chunk: {:.1}\n\
             🚀 Throughput: {:.2} MB/s\n\
             🧵 Concurrent Tasks: {}\n\
             📊 Parallel Efficiency: {:.1}%\n\
             ⏱️  Total Processing Time: {:.2}s",
            self.total_files_processed,
            self.total_chunks_processed,
            self.total_records_processed,
            self.total_bytes_processed as f64 / (1024.0 * 1024.0),
            self.average_chunk_processing_time.as_millis(),
            self.average_records_per_chunk,
            self.throughput_mb_per_sec,
            self.concurrent_tasks,
            self.parallel_efficiency * 100.0,
            self.total_processing_time.as_secs_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_async_parallel_processor_creation() {
        let processor = AsyncParallelProcessor::new();
        assert!(processor.max_concurrent_tasks > 0);
        assert_eq!(processor.get_metrics().total_files_processed, 0);
    }

    #[tokio::test]
    async fn test_work_item_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.csv");
        
        // Create a test file
        let mut file = File::create(&file_path).await?;
        file.write_all(b"user,password,url\ntest@example.com,pass123,https://example.com\n").await?;
        file.sync_all().await?;
        
        let mut processor = AsyncParallelProcessor::with_max_concurrent_tasks(2);
        let work_items = processor.create_work_items_for_files(&[file_path]).await?;
        
        assert!(!work_items.is_empty());
        assert_eq!(work_items[0].chunk_offset, 0);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_processing() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.csv");
        
        // Create a test file with multiple records
        let mut file = File::create(&file_path).await?;
        let content = "user,password,url\n".to_string() +
                      &(0..100).map(|i| format!("user{}@example.com,pass{},https://site{}.com", i, i, i))
                      .collect::<Vec<_>>()
                      .join("\n");
        file.write_all(content.as_bytes()).await?;
        file.sync_all().await?;
        
        let mut processor = AsyncParallelProcessor::with_max_concurrent_tasks(2);
        
        // Mock processor function
        let processor_fn = |_data: Vec<u8>, _path: PathBuf| async move {
            // Simulate some processing
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![Record {
                user: "test@example.com".to_string(),
                password: "pass123".to_string(),
                url: "https://example.com".to_string(),
                normalized_user: "test@example.com".to_string(),
                normalized_url: "example.com".to_string(),
                completeness_score: 1.0,
                field_count: 3,
                all_fields: vec!["test@example.com".to_string(), "pass123".to_string(), "https://example.com".to_string()],
            }])
        };
        
        let records = processor.process_files_parallel(&[file_path], processor_fn).await?;
        
        assert!(!records.is_empty());
        assert!(processor.get_metrics().total_chunks_processed > 0);
        assert!(processor.get_metrics().throughput_mb_per_sec > 0.0);
        
        Ok(())
    }
}