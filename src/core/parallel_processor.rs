use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use anyhow::Result;
use crate::core::record::Record;
use crate::core::batch_writer::BatchWriter;
use crate::constants::{
    PARALLEL_FILE_PROCESSING_THREADS, STREAMING_CHUNK_SIZE_MB, PARALLEL_IO_QUEUE_SIZE,
    BYTES_PER_MB
};

#[derive(Debug)]
pub struct ParallelProcessor {
    thread_count: usize,
    streaming_chunk_size: usize,
    io_queue_size: usize,
    metrics: ParallelProcessingMetrics,
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub file_path: PathBuf,
    pub chunk_offset: usize,
    pub chunk_size: usize,
    pub priority: u8,
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub work_item: WorkItem,
    pub records: Vec<Record>,
    pub processing_time: Duration,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParallelProcessingMetrics {
    pub total_files_processed: usize,
    pub total_chunks_processed: usize,
    pub total_records_processed: usize,
    pub total_processing_time: Duration,
    pub average_chunk_processing_time: Duration,
    pub average_records_per_chunk: f64,
    pub thread_utilization: f64,
    pub io_queue_utilization: f64,
    pub parallel_efficiency: f64,
}

impl Default for ParallelProcessingMetrics {
    fn default() -> Self {
        Self {
            total_files_processed: 0,
            total_chunks_processed: 0,
            total_records_processed: 0,
            total_processing_time: Duration::new(0, 0),
            average_chunk_processing_time: Duration::new(0, 0),
            average_records_per_chunk: 0.0,
            thread_utilization: 0.0,
            io_queue_utilization: 0.0,
            parallel_efficiency: 1.0,
        }
    }
}

#[derive(Debug)]
struct WorkQueue {
    items: Arc<Mutex<VecDeque<WorkItem>>>,
    max_size: usize,
    utilization_samples: Arc<Mutex<VecDeque<f64>>>,
}

impl WorkQueue {
    fn new(max_size: usize) -> Self {
        Self {
            items: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            utilization_samples: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn push(&self, item: WorkItem) -> Result<()> {
        let mut queue = self.items.lock().unwrap();
        if queue.len() >= self.max_size {
            return Err(anyhow::anyhow!("Work queue is full"));
        }

        let insert_pos = queue.iter().position(|existing| existing.priority < item.priority)
            .unwrap_or(queue.len());
        queue.insert(insert_pos, item);

        let utilization = queue.len() as f64 / self.max_size as f64;
        let mut samples = self.utilization_samples.lock().unwrap();
        samples.push_back(utilization);
        if samples.len() > 100 {
            samples.pop_front();
        }

        Ok(())
    }

    fn pop(&self) -> Option<WorkItem> {
        let mut queue = self.items.lock().unwrap();
        queue.pop_front()
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.items.lock().unwrap().is_empty()
    }

    fn get_average_utilization(&self) -> f64 {
        let samples = self.utilization_samples.lock().unwrap();
        if samples.is_empty() {
            0.0
        } else {
            samples.iter().sum::<f64>() / samples.len() as f64
        }
    }
}

impl ParallelProcessor {
    pub fn new() -> Self {
        Self::with_thread_count(PARALLEL_FILE_PROCESSING_THREADS)
    }

    pub fn with_thread_count(thread_count: usize) -> Self {
        Self {
            thread_count: thread_count.max(1).min(32), // Reasonable bounds
            streaming_chunk_size: STREAMING_CHUNK_SIZE_MB * BYTES_PER_MB,
            io_queue_size: PARALLEL_IO_QUEUE_SIZE,
            metrics: ParallelProcessingMetrics::default(),
        }
    }

    pub fn process_files_parallel<F>(
        &mut self,
        file_paths: &[PathBuf],
        processor_fn: F,
        output_writer: &mut BatchWriter,
    ) -> Result<()>
    where
        F: Fn(&[u8], &Path) -> Result<Vec<Record>> + Send + Sync + 'static,
    {
        let start_time = Instant::now();
        let work_queue = Arc::new(WorkQueue::new(self.io_queue_size));
        let results_queue = Arc::new(Mutex::new(VecDeque::new()));

        for file_path in file_paths {
            let work_items = self.create_work_items_for_file(file_path)?;
            for item in work_items {
                work_queue.push(item)?;
            }
        }

        let processor_fn = Arc::new(processor_fn);
        let mut handles = Vec::new();

        for thread_id in 0..self.thread_count {
            let queue = Arc::clone(&work_queue);
            let results = Arc::clone(&results_queue);
            let processor = Arc::clone(&processor_fn);

            let handle = thread::spawn(move || {
                Self::worker_thread(thread_id, queue, results, processor)
            });
            handles.push(handle);
        }

        let mut total_records = 0;
        let mut active_threads = self.thread_count;

        while active_threads > 0 || !results_queue.lock().unwrap().is_empty() {
            {
                let mut results = results_queue.lock().unwrap();
                while let Some(mut result) = results.pop_front() {
                    total_records += result.records.len();
                    self.update_metrics_from_result(&result);
                    output_writer.add_records(std::mem::take(&mut result.records))?;
                }
            }

            active_threads = handles.iter().filter(|h| !h.is_finished()).count();

            thread::sleep(Duration::from_millis(10));
        }

        for handle in handles {
            if let Err(e) = handle.join() {
                eprintln!("Worker thread panicked: {:?}", e);
            }
        }

        {
            let mut results = results_queue.lock().unwrap();
            while let Some(mut result) = results.pop_front() {
                total_records += result.records.len();
                self.update_metrics_from_result(&result);
                output_writer.add_records(std::mem::take(&mut result.records))?;
            }
        }

        self.metrics.total_files_processed = file_paths.len();
        self.metrics.total_records_processed = total_records;
        self.metrics.io_queue_utilization = work_queue.get_average_utilization();

        let total_time = start_time.elapsed();
        self.calculate_parallel_efficiency(total_time);

        Ok(())
    }

    fn create_work_items_for_file(&self, file_path: &Path) -> Result<Vec<WorkItem>> {
        let file_size = std::fs::metadata(file_path)?.len() as usize;
        let mut work_items = Vec::new();
        let mut offset = 0;

        while offset < file_size {
            let chunk_size = (file_size - offset).min(self.streaming_chunk_size);
            work_items.push(WorkItem {
                file_path: file_path.to_path_buf(),
                chunk_offset: offset,
                chunk_size,
                priority: 1,
            });
            offset += chunk_size;
        }

        Ok(work_items)
    }

    fn worker_thread<F>(
        _thread_id: usize,
        work_queue: Arc<WorkQueue>,
        results_queue: Arc<Mutex<VecDeque<ProcessingResult>>>,
        processor_fn: Arc<F>,
    ) where
        F: Fn(&[u8], &Path) -> Result<Vec<Record>> + Send + Sync,
    {
        while let Some(work_item) = work_queue.pop() {
            let start_time = Instant::now();

            let chunk_data = match Self::read_file_chunk(&work_item) {
                Ok(data) => data,
                Err(e) => {
                    let result = ProcessingResult {
                        work_item,
                        records: Vec::new(),
                        processing_time: start_time.elapsed(),
                        errors: vec![format!("Failed to read chunk: {}", e)],
                    };
                    results_queue.lock().unwrap().push_back(result);
                    continue;
                }
            };

            let records = match processor_fn(&chunk_data, &work_item.file_path) {
                Ok(records) => records,
                Err(e) => {
                    let result = ProcessingResult {
                        work_item,
                        records: Vec::new(),
                        processing_time: start_time.elapsed(),
                        errors: vec![format!("Failed to process chunk: {}", e)],
                    };
                    results_queue.lock().unwrap().push_back(result);
                    continue;
                }
            };

            let result = ProcessingResult {
                work_item,
                records,
                processing_time: start_time.elapsed(),
                errors: Vec::new(),
            };

            results_queue.lock().unwrap().push_back(result);
        }
    }

    fn read_file_chunk(work_item: &WorkItem) -> Result<Vec<u8>> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(&work_item.file_path)?;
        file.seek(SeekFrom::Start(work_item.chunk_offset as u64))?;

        let mut buffer = vec![0u8; work_item.chunk_size];
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);

        Ok(buffer)
    }

    fn update_metrics_from_result(&mut self, result: &ProcessingResult) {
        self.metrics.total_chunks_processed += 1;
        self.metrics.total_processing_time += result.processing_time;

        if self.metrics.total_chunks_processed > 0 {
            self.metrics.average_chunk_processing_time =
                self.metrics.total_processing_time / self.metrics.total_chunks_processed as u32;

            self.metrics.average_records_per_chunk =
                self.metrics.total_records_processed as f64 / self.metrics.total_chunks_processed as f64;
        }
    }

    fn calculate_parallel_efficiency(&mut self, total_wall_time: Duration) {
        if total_wall_time.as_secs_f64() > 0.0 && self.thread_count > 1 {
            let total_cpu_time = self.metrics.total_processing_time.as_secs_f64();
            let theoretical_speedup = self.thread_count as f64;
            let actual_speedup = total_cpu_time / total_wall_time.as_secs_f64();

            self.metrics.parallel_efficiency = (actual_speedup / theoretical_speedup).min(1.0);
            self.metrics.thread_utilization = (actual_speedup / self.thread_count as f64) * 100.0;
        }
    }

    pub fn get_metrics(&self) -> &ParallelProcessingMetrics {
        &self.metrics
    }

    pub fn optimize_thread_count(&mut self) -> usize {
        if self.metrics.parallel_efficiency < 0.7 && self.thread_count > 1 {
            self.thread_count = (self.thread_count - 1).max(1);
        } else if self.metrics.parallel_efficiency > 0.9 && self.thread_count < 16 {
            self.thread_count += 1;
        }

        self.thread_count
    }
}

impl ParallelProcessingMetrics {
    pub fn format_summary(&self) -> String {
        format!(
            "Parallel Processing Performance:\n\
             📁 Files Processed: {}\n\
             🔄 Chunks Processed: {}\n\
             📊 Records Processed: {}\n\
             ⚡ Avg Chunk Time: {:.2}ms\n\
             📈 Avg Records/Chunk: {:.1}\n\
             🧵 Thread Utilization: {:.1}%\n\
             📋 Queue Utilization: {:.1}%\n\
             🚀 Parallel Efficiency: {:.1}%\n\
             ⏱️  Total CPU Time: {:.2}s",
            self.total_files_processed,
            self.total_chunks_processed,
            self.total_records_processed,
            self.average_chunk_processing_time.as_millis(),
            self.average_records_per_chunk,
            self.thread_utilization,
            self.io_queue_utilization * 100.0,
            self.parallel_efficiency * 100.0,
            self.total_processing_time.as_secs_f64()
        )
    }
}
