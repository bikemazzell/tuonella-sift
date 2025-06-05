use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use anyhow::Result;
use crate::core::record::Record;
use crate::core::buffer_pool::{BufferPool, PooledBuffer};
use bytes::{BytesMut, BufMut};
use crate::constants::{
    WRITE_BATCH_SIZE_RECORDS, WRITE_BUFFER_SIZE_MB, MAX_WRITE_BATCH_SIZE_RECORDS,
    BYTES_PER_MB, ZERO_DURATION_SECS, ZERO_DURATION_NANOS, ZERO_F64, ZERO_USIZE,
    MIN_BATCH_SIZE, MIN_OPERATIONS_FOR_OPTIMIZATION, LOW_THROUGHPUT_THRESHOLD,
    HIGH_THROUGHPUT_THRESHOLD, IO_THROUGHPUT_THRESHOLD, BATCH_SIZE_REDUCTION_FACTOR,
    BATCH_SIZE_INCREASE_FACTOR, THROUGHPUT_EFFICIENCY_THRESHOLD, BYTES_PER_MB_FLOAT,
    EFFICIENCY_COMPONENTS_COUNT
};

#[derive(Debug)]
pub struct BatchWriter {
    record_buffer: Vec<Record>,
    writer: BufWriter<File>,
    batch_size: usize,
    max_batch_size: usize,
    #[allow(dead_code)]
    write_buffer_size: usize,
    metrics: BatchWriteMetrics,
    last_flush: Instant,
    // Buffer pool for efficient memory reuse
    buffer_pool: BufferPool,
    // Estimated bytes per record for buffer sizing
    estimated_record_size: usize,
}

#[derive(Debug, Clone)]
pub struct BatchWriteMetrics {
    pub total_records_written: usize,
    pub total_write_operations: usize,
    pub total_write_time: Duration,
    pub total_bytes_written: usize,
    pub average_records_per_write: f64,
    pub average_write_throughput: f64,
    pub average_io_throughput: f64,
    pub forced_flushes: usize,
    pub automatic_flushes: usize,
}

impl Default for BatchWriteMetrics {
    fn default() -> Self {
        Self {
            total_records_written: ZERO_USIZE,
            total_write_operations: ZERO_USIZE,
            total_write_time: Duration::new(ZERO_DURATION_SECS, ZERO_DURATION_NANOS),
            total_bytes_written: ZERO_USIZE,
            average_records_per_write: ZERO_F64,
            average_write_throughput: ZERO_F64,
            average_io_throughput: ZERO_F64,
            forced_flushes: ZERO_USIZE,
            automatic_flushes: ZERO_USIZE,
        }
    }
}

impl BatchWriter {
    pub fn new<P: AsRef<Path>>(output_path: P) -> Result<Self> {
        Self::with_batch_size(output_path, WRITE_BATCH_SIZE_RECORDS)
    }

    pub fn with_batch_size<P: AsRef<Path>>(output_path: P, batch_size: usize) -> Result<Self> {
        let file = File::create(output_path)?;
        let write_buffer_size = WRITE_BUFFER_SIZE_MB * BYTES_PER_MB;
        let writer = BufWriter::with_capacity(write_buffer_size, file);

        // Estimate average record size for buffer pre-allocation
        let estimated_record_size = 128; // Conservative estimate: 50 chars avg per field * 3 fields + overhead
        let buffer_capacity = batch_size * estimated_record_size;
        
        // Create buffer pool with reasonable defaults
        let buffer_pool = BufferPool::new(buffer_capacity, 8); // Pool up to 8 buffers
        
        Ok(Self {
            record_buffer: Vec::with_capacity(batch_size),
            writer,
            batch_size: batch_size.min(MAX_WRITE_BATCH_SIZE_RECORDS),
            max_batch_size: MAX_WRITE_BATCH_SIZE_RECORDS,
            write_buffer_size,
            metrics: BatchWriteMetrics::default(),
            last_flush: Instant::now(),
            buffer_pool,
            estimated_record_size,
        })
    }

    pub fn add_record(&mut self, record: Record) -> Result<()> {
        self.record_buffer.push(record);

        if self.record_buffer.len() >= self.batch_size {
            self.flush_batch(false)?;
        }

        Ok(())
    }

    pub fn add_records(&mut self, records: Vec<Record>) -> Result<()> {
        for record in records {
            self.record_buffer.push(record);

            if self.record_buffer.len() >= self.batch_size {
                self.flush_batch(false)?;
            }
        }

        Ok(())
    }

    pub fn flush_batch(&mut self, force: bool) -> Result<()> {
        if self.record_buffer.is_empty() {
            return Ok(());
        }

        let write_start = Instant::now();
        let record_count = self.record_buffer.len();
        let mut bytes_written = 0;

        if self.metrics.total_write_operations == 0 {
            let header = "user,password,url\n";
            self.writer.write_all(header.as_bytes())?;
            bytes_written += header.len();
        }

        // Get a buffer from the pool
        let mut pooled_buffer = PooledBuffer::new(self.buffer_pool.clone());
        let write_buf = pooled_buffer.get_mut();
        
        // Pre-allocate buffer capacity based on estimated size
        let estimated_total_size = record_count * self.estimated_record_size;
        if write_buf.capacity() < estimated_total_size {
            write_buf.reserve(estimated_total_size - write_buf.capacity());
        }
        
        // Write all records to buffer in one go
        let records_to_write: Vec<Record> = self.record_buffer.drain(..).collect();
        for record in records_to_write {
            Self::format_record_as_csv_bytes(&record, write_buf);
        }
        
        // Single write operation
        bytes_written += write_buf.len();
        self.writer.write_all(write_buf)?;
        
        // Buffer will be automatically returned to pool when pooled_buffer goes out of scope

        self.writer.flush()?;

        let write_time = write_start.elapsed();

        self.update_metrics(record_count, bytes_written, write_time, force);

        self.last_flush = Instant::now();
        Ok(())
    }

    // Zero-copy CSV formatting directly to byte buffer
    fn format_record_as_csv_bytes(record: &Record, buf: &mut BytesMut) {
        Self::write_csv_field_bytes(&record.user, buf);
        buf.put_u8(b',');
        Self::write_csv_field_bytes(&record.password, buf);
        buf.put_u8(b',');
        Self::write_csv_field_bytes(&record.url, buf);
        buf.put_u8(b'\n');
    }
    
    fn write_csv_field_bytes(field: &str, buf: &mut BytesMut) {
        // Fast path: no special characters
        if !field.bytes().any(|b| b == b',' || b == b'"' || b == b'\n' || b == b'\r') {
            buf.extend_from_slice(field.as_bytes());
            return;
        }
        
        // Slow path: escape special characters
        buf.put_u8(b'"');
        for byte in field.bytes() {
            if byte == b'"' {
                buf.put_u8(b'"');
                buf.put_u8(b'"');
            } else {
                buf.put_u8(byte);
            }
        }
        buf.put_u8(b'"');
    }


    fn update_metrics(&mut self, record_count: usize, bytes_written: usize, write_time: Duration, force: bool) {
        self.metrics.total_records_written += record_count;
        self.metrics.total_write_operations += 1;
        self.metrics.total_write_time += write_time;
        self.metrics.total_bytes_written += bytes_written;

        if force {
            self.metrics.forced_flushes += 1;
        } else {
            self.metrics.automatic_flushes += 1;
        }

        if self.metrics.total_write_operations > ZERO_USIZE {
            self.metrics.average_records_per_write =
                self.metrics.total_records_written as f64 / self.metrics.total_write_operations as f64;
        }

        if self.metrics.total_write_time.as_secs_f64() > ZERO_F64 {
            self.metrics.average_write_throughput =
                self.metrics.total_records_written as f64 / self.metrics.total_write_time.as_secs_f64();

            self.metrics.average_io_throughput =
                (self.metrics.total_bytes_written as f64 / BYTES_PER_MB_FLOAT) / self.metrics.total_write_time.as_secs_f64();
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.record_buffer.is_empty() {
            self.flush_batch(true)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn set_batch_size(&mut self, new_size: usize) {
        self.batch_size = new_size.min(self.max_batch_size).max(MIN_BATCH_SIZE); // Minimum batch size
    }

    pub fn get_buffer_utilization(&self) -> f64 {
        self.record_buffer.len() as f64 / self.batch_size as f64
    }

    pub fn should_auto_flush(&self, max_time_since_flush: Duration) -> bool {
        !self.record_buffer.is_empty() && self.last_flush.elapsed() > max_time_since_flush
    }

    pub fn auto_flush_if_needed(&mut self, max_time_since_flush: Duration) -> Result<bool> {
        if self.should_auto_flush(max_time_since_flush) {
            self.flush_batch(false)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_metrics(&self) -> &BatchWriteMetrics {
        &self.metrics
    }

    pub fn get_buffer_pool_stats(&self) -> crate::core::buffer_pool::BufferPoolStats {
        self.buffer_pool.stats()
    }

    pub fn optimize_batch_size(&mut self) -> Result<usize> {
        if self.metrics.total_write_operations < MIN_OPERATIONS_FOR_OPTIMIZATION {
            return Ok(self.batch_size);
        }

        let current_throughput = self.metrics.average_write_throughput;
        let current_batch_size = self.batch_size;

        if current_throughput < LOW_THROUGHPUT_THRESHOLD {
            let new_size = (current_batch_size as f64 * BATCH_SIZE_REDUCTION_FACTOR) as usize;
            self.set_batch_size(new_size);
        }
        else if current_throughput > HIGH_THROUGHPUT_THRESHOLD && self.metrics.average_io_throughput < IO_THROUGHPUT_THRESHOLD {
            let new_size = (current_batch_size as f64 * BATCH_SIZE_INCREASE_FACTOR) as usize;
            self.set_batch_size(new_size);
        }

        Ok(self.batch_size)
    }
}

impl BatchWriteMetrics {
    pub fn format_summary(&self) -> String {
        format!(
            "Batch Write Performance:\n\
             📝 Records Written: {}\n\
             🔄 Write Operations: {}\n\
             ⚡ Avg Records/Write: {:.1}\n\
             📊 Write Throughput: {:.1} records/sec\n\
             💾 I/O Throughput: {:.1} MB/sec\n\
             ⏱️  Total Write Time: {:.2}s\n\
             📁 Total Bytes: {:.2} MB\n\
             🔄 Auto Flushes: {}\n\
             ⚠️  Forced Flushes: {}",
            self.total_records_written,
            self.total_write_operations,
            self.average_records_per_write,
            self.average_write_throughput,
            self.average_io_throughput,
            self.total_write_time.as_secs_f64(),
            self.total_bytes_written as f64 / BYTES_PER_MB_FLOAT,
            self.automatic_flushes,
            self.forced_flushes
        )
    }

    pub fn get_efficiency_score(&self) -> f64 {
        if self.total_write_operations == ZERO_USIZE {
            return ZERO_F64;
        }

        let batching_efficiency = (self.average_records_per_write / MAX_WRITE_BATCH_SIZE_RECORDS as f64).min(1.0);

        let throughput_efficiency = (self.average_write_throughput / THROUGHPUT_EFFICIENCY_THRESHOLD).min(1.0);

        let flush_efficiency = if self.total_write_operations > ZERO_USIZE {
            1.0 - (self.forced_flushes as f64 / self.total_write_operations as f64)
        } else {
            1.0
        };

        (batching_efficiency + throughput_efficiency + flush_efficiency) / EFFICIENCY_COMPONENTS_COUNT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;
    use crate::constants::{TEST_COMPLETENESS_SCORE, TEST_FIELD_COUNT};

    #[test]
    fn test_batch_writer_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let output_path = temp_dir.path().join("test_output.csv");

        let writer = BatchWriter::new(&output_path)?;
        assert_eq!(writer.get_batch_size(), WRITE_BATCH_SIZE_RECORDS);
        assert_eq!(writer.get_buffer_utilization(), 0.0);

        Ok(())
    }

    #[test]
    fn test_record_batching() -> Result<()> {
        let temp_dir = tempdir()?;
        let output_path = temp_dir.path().join("test_output.csv");

        // Use a small batch size (2) to ensure multiple write operations with 5 records
        let mut writer = BatchWriter::with_batch_size(&output_path, 2)?;

        for i in 0..5 {
            let record = Record {
                user: format!("user{}@example.com", i),
                password: format!("password{}", i),
                url: format!("https://example{}.com", i),
                normalized_user: format!("user{}@example.com", i),
                normalized_url: format!("example{}.com", i),
                completeness_score: TEST_COMPLETENESS_SCORE,
                field_count: TEST_FIELD_COUNT,
                all_fields: vec![
                    format!("user{}@example.com", i),
                    format!("password{}", i),
                    format!("https://example{}.com", i)
                ],
            };
            writer.add_record(record)?;
        }

        writer.finish()?;

        let content = fs::read_to_string(&output_path)?;
        assert!(content.contains("user,password,url")); // Header
        assert!(content.contains("user0@example.com"));
        assert!(content.contains("user4@example.com"));

        let metrics = writer.get_metrics();
        assert_eq!(metrics.total_records_written, 5);
        assert!(metrics.total_write_operations >= 2);

        Ok(())
    }

    #[test]
    fn test_csv_escaping() -> Result<()> {
        let temp_dir = tempdir()?;
        let output_path = temp_dir.path().join("test_output.csv");

        let mut writer = BatchWriter::with_batch_size(&output_path, 1)?;

        let record = Record {
            user: "user,with,commas@example.com".to_string(),
            password: "password\"with\"quotes".to_string(),
            url: "https://example.com/path,with,commas".to_string(),
            normalized_user: "user,with,commas@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            completeness_score: TEST_COMPLETENESS_SCORE,
            field_count: TEST_FIELD_COUNT,
            all_fields: vec![
                "user,with,commas@example.com".to_string(),
                "password\"with\"quotes".to_string(),
                "https://example.com/path,with,commas".to_string()
            ],
        };

        writer.add_record(record)?;
        writer.finish()?;

        let content = fs::read_to_string(&output_path)?;
        assert!(content.contains("\"user,with,commas@example.com\""));
        assert!(content.contains("\"password\"\"with\"\"quotes\""));

        Ok(())
    }
}
