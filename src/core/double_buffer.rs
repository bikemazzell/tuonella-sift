#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, Condvar};
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};
#[cfg(feature = "cuda")]
use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaRecord;

#[cfg(feature = "cuda")]
use crate::constants::{
    BUFFER_SWAP_THRESHOLD_PERCENT, ASYNC_IO_TIMEOUT_SECONDS,
    BYTES_PER_MB
};

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DoubleBuffer {
    buffer_a: Arc<Mutex<BufferState>>,
    buffer_b: Arc<Mutex<BufferState>>,
    buffer_ready: Arc<Condvar>,
    active_buffer: Arc<Mutex<bool>>,
    buffer_capacity: usize,
    metrics: Arc<Mutex<DoubleBufferMetrics>>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct BufferState {    
    data: Vec<CudaRecord>,
    size_bytes: usize,
    ready_for_processing: bool,
    being_processed: bool,
    ready_for_io: bool,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct DoubleBufferMetrics {
    pub total_records: usize,
    pub total_processing_time: Duration,
    pub total_io_time: Duration,
    pub buffer_swaps: usize,
    pub average_throughput: f64,
    pub gpu_utilization_percent: f64,
    pub io_wait_percent: f64,
}

#[cfg(feature = "cuda")]
impl Default for BufferState {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            size_bytes: 0,
            ready_for_processing: false,
            being_processed: false,
            ready_for_io: true,
        }
    }
}

#[cfg(feature = "cuda")]
impl Default for DoubleBufferMetrics {
    fn default() -> Self {
        Self {
            total_records: 0,
            total_processing_time: Duration::new(0, 0),
            total_io_time: Duration::new(0, 0),
            buffer_swaps: 0,
            average_throughput: 0.0,
            gpu_utilization_percent: 0.0,
            io_wait_percent: 0.0,
        }
    }
}

#[cfg(feature = "cuda")]
impl DoubleBuffer {
    pub fn new(buffer_capacity_mb: usize) -> Result<Self> {
        let buffer_capacity = (buffer_capacity_mb * BYTES_PER_MB) / 2;

        Ok(Self {
            buffer_a: Arc::new(Mutex::new(BufferState::default())),
            buffer_b: Arc::new(Mutex::new(BufferState::default())),
            buffer_ready: Arc::new(Condvar::new()),
            active_buffer: Arc::new(Mutex::new(true)), // Start with buffer A
            buffer_capacity,
            metrics: Arc::new(Mutex::new(DoubleBufferMetrics::default())),
        })
    }

    pub fn add_records(&self, records: Vec<CudaRecord>) -> Result<bool> {
        let io_start = Instant::now();

        let active = *self.active_buffer.lock().unwrap();
        let (io_buffer, _processing_buffer) = if active {
            (&self.buffer_a, &self.buffer_b)
        } else {
            (&self.buffer_b, &self.buffer_a)
        };

        let mut buffer = io_buffer.lock().unwrap();

        let estimated_size = records.len() * std::mem::size_of::<CudaRecord>();
        if buffer.size_bytes + estimated_size > self.buffer_capacity {
            drop(buffer);
            self.update_io_metrics(io_start.elapsed());
            return Ok(true);
        }

        buffer.data.extend(records);
        buffer.size_bytes += estimated_size;

        let usage_percent = (buffer.size_bytes as f64 / self.buffer_capacity as f64) * 100.0;
        if usage_percent >= BUFFER_SWAP_THRESHOLD_PERCENT {
            buffer.ready_for_processing = true;
            buffer.ready_for_io = false;
            self.buffer_ready.notify_one();
            drop(buffer);
            self.update_io_metrics(io_start.elapsed());
            return Ok(true);
        }

        drop(buffer);
        self.update_io_metrics(io_start.elapsed());
        Ok(false)
    }

    pub fn swap_buffers(&self) -> Result<()> {
        let mut active = self.active_buffer.lock().unwrap();
        let mut metrics = self.metrics.lock().unwrap();

        *active = !*active;
        metrics.buffer_swaps += 1;

        let (new_io_buffer, new_processing_buffer) = if *active {
            (&self.buffer_a, &self.buffer_b)
        } else {
            (&self.buffer_b, &self.buffer_a)
        };

        {
            let mut io_buf = new_io_buffer.lock().unwrap();
            io_buf.ready_for_io = true;
            io_buf.ready_for_processing = false;
            io_buf.being_processed = false;
        }

        {
            let mut proc_buf = new_processing_buffer.lock().unwrap();
            proc_buf.ready_for_processing = true;
            proc_buf.ready_for_io = false;
            proc_buf.being_processed = true;
        }

        self.buffer_ready.notify_all();
        Ok(())
    }

    pub fn get_processing_records(&self) -> Result<Vec<CudaRecord>> {
        let processing_start = Instant::now();

        let active = *self.active_buffer.lock().unwrap();
        let processing_buffer = if active {
            &self.buffer_b
        } else {
            &self.buffer_a
        };

        let mut buffer = processing_buffer.lock().unwrap();

        while !buffer.ready_for_processing {
            let timeout = Duration::from_secs(ASYNC_IO_TIMEOUT_SECONDS);
            let (buf, timeout_result) = self.buffer_ready.wait_timeout(buffer, timeout).unwrap();
            buffer = buf;

            if timeout_result.timed_out() {
                return Err(anyhow::anyhow!("Timeout waiting for processing buffer"));
            }
        }

        let records = std::mem::take(&mut buffer.data);
        buffer.size_bytes = 0;
        buffer.ready_for_processing = false;
        buffer.being_processed = false;
        buffer.ready_for_io = true;

        drop(buffer);
        self.update_processing_metrics(processing_start.elapsed(), records.len());
        Ok(records)
    }

    pub fn is_io_buffer_available(&self) -> bool {
        let active = *self.active_buffer.lock().unwrap();
        let io_buffer = if active {
            &self.buffer_a
        } else {
            &self.buffer_b
        };

        let buffer = io_buffer.lock().unwrap();
        buffer.ready_for_io && !buffer.being_processed
    }

    pub fn get_metrics(&self) -> DoubleBufferMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn update_io_metrics(&self, io_time: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_io_time += io_time;
        self.calculate_utilization(&mut metrics);
    }

    fn update_processing_metrics(&self, processing_time: Duration, record_count: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_processing_time += processing_time;
        metrics.total_records += record_count;

        let total_time_secs = (metrics.total_processing_time + metrics.total_io_time).as_secs_f64();
        if total_time_secs > 0.0 {
            metrics.average_throughput = metrics.total_records as f64 / total_time_secs;
        }

        self.calculate_utilization(&mut metrics);
    }

    fn calculate_utilization(&self, metrics: &mut DoubleBufferMetrics) {
        let total_time = metrics.total_processing_time + metrics.total_io_time;
        if total_time.as_secs_f64() > 0.0 {
            metrics.gpu_utilization_percent =
                (metrics.total_processing_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
            metrics.io_wait_percent =
                (metrics.total_io_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
        }
    }

    pub fn flush_remaining(&self) -> Result<Vec<CudaRecord>> {
        let _active = *self.active_buffer.lock().unwrap();
        let mut all_records = Vec::new();

        for buffer_ref in [&self.buffer_a, &self.buffer_b] {
            let mut buffer = buffer_ref.lock().unwrap();
            if !buffer.data.is_empty() {
                all_records.extend(std::mem::take(&mut buffer.data));
                buffer.size_bytes = 0;
                buffer.ready_for_processing = false;
                buffer.being_processed = false;
                buffer.ready_for_io = true;
            }
        }

        Ok(all_records)
    }
}

#[cfg(feature = "cuda")]
impl DoubleBufferMetrics {
    pub fn format_summary(&self) -> String {
        format!(
            "Double Buffer Performance:\n\
             📊 Records Processed: {}\n\
             ⚡ Average Throughput: {:.1} records/sec\n\
             🔄 Buffer Swaps: {}\n\
             🖥️  GPU Utilization: {:.1}%\n\
             💾 I/O Wait Time: {:.1}%\n\
             ⏱️  Total Processing: {:.2}s\n\
             📁 Total I/O: {:.2}s",
            self.total_records,
            self.average_throughput,
            self.buffer_swaps,
            self.gpu_utilization_percent,
            self.io_wait_percent,
            self.total_processing_time.as_secs_f64(),
            self.total_io_time.as_secs_f64()
        )
    }
}
