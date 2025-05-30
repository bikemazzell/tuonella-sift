use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use anyhow::Result;
use crate::core::record::Record;
use crate::constants::{
    WRITE_BATCH_SIZE_RECORDS, WRITE_BUFFER_SIZE_MB, MAX_WRITE_BATCH_SIZE_RECORDS,
    BYTES_PER_MB
};

/// Batch writer for optimized I/O operations
///
/// This implements Section 6: "Batch multiple processed records into a single write operation to reduce I/O overhead"
#[derive(Debug)]
pub struct BatchWriter {
    /// Internal buffer for batching records
    record_buffer: Vec<Record>,
    /// Write buffer for disk I/O
    writer: BufWriter<File>,
    /// Current batch size configuration
    batch_size: usize,
    /// Maximum batch size allowed
    max_batch_size: usize,
    /// Write buffer size in bytes
    #[allow(dead_code)]
    write_buffer_size: usize,
    /// Performance metrics
    metrics: BatchWriteMetrics,
    /// Last flush time for automatic flushing
    last_flush: Instant,
}

/// Performance metrics for batch writing
#[derive(Debug, Clone)]
pub struct BatchWriteMetrics {
    /// Total records written
    pub total_records_written: usize,
    /// Total number of write operations
    pub total_write_operations: usize,
    /// Total time spent writing
    pub total_write_time: Duration,
    /// Total bytes written
    pub total_bytes_written: usize,
    /// Average records per write operation
    pub average_records_per_write: f64,
    /// Average write throughput (records/second)
    pub average_write_throughput: f64,
    /// Average I/O throughput (MB/second)
    pub average_io_throughput: f64,
    /// Number of forced flushes
    pub forced_flushes: usize,
    /// Number of automatic flushes
    pub automatic_flushes: usize,
}

impl Default for BatchWriteMetrics {
    fn default() -> Self {
        Self {
            total_records_written: 0,
            total_write_operations: 0,
            total_write_time: Duration::new(0, 0),
            total_bytes_written: 0,
            average_records_per_write: 0.0,
            average_write_throughput: 0.0,
            average_io_throughput: 0.0,
            forced_flushes: 0,
            automatic_flushes: 0,
        }
    }
}

impl BatchWriter {
    /// Create a new batch writer
    pub fn new<P: AsRef<Path>>(output_path: P) -> Result<Self> {
        Self::with_batch_size(output_path, WRITE_BATCH_SIZE_RECORDS)
    }

    /// Create a new batch writer with custom batch size
    pub fn with_batch_size<P: AsRef<Path>>(output_path: P, batch_size: usize) -> Result<Self> {
        let file = File::create(output_path)?;
        let write_buffer_size = WRITE_BUFFER_SIZE_MB * BYTES_PER_MB;
        let writer = BufWriter::with_capacity(write_buffer_size, file);

        Ok(Self {
            record_buffer: Vec::with_capacity(batch_size),
            writer,
            batch_size: batch_size.min(MAX_WRITE_BATCH_SIZE_RECORDS),
            max_batch_size: MAX_WRITE_BATCH_SIZE_RECORDS,
            write_buffer_size,
            metrics: BatchWriteMetrics::default(),
            last_flush: Instant::now(),
        })
    }

    /// Add a record to the batch
    ///
    /// Automatically flushes when batch size is reached
    pub fn add_record(&mut self, record: Record) -> Result<()> {
        self.record_buffer.push(record);

        // Check if we need to flush
        if self.record_buffer.len() >= self.batch_size {
            self.flush_batch(false)?;
        }

        Ok(())
    }

    /// Add multiple records to the batch
    ///
    /// More efficient than adding records one by one
    pub fn add_records(&mut self, records: Vec<Record>) -> Result<()> {
        for record in records {
            self.record_buffer.push(record);

            // Check if we need to flush during the process
            if self.record_buffer.len() >= self.batch_size {
                self.flush_batch(false)?;
            }
        }

        Ok(())
    }

    /// Flush the current batch to disk
    ///
    /// force: whether this is a forced flush (affects metrics)
    pub fn flush_batch(&mut self, force: bool) -> Result<()> {
        if self.record_buffer.is_empty() {
            return Ok(());
        }

        let write_start = Instant::now();
        let record_count = self.record_buffer.len();
        let mut bytes_written = 0;

        // Write CSV header if this is the first write
        if self.metrics.total_write_operations == 0 {
            let header = "user,password,url\n";
            self.writer.write_all(header.as_bytes())?;
            bytes_written += header.len();
        }

        // Write all records in the batch
        let records_to_write: Vec<Record> = self.record_buffer.drain(..).collect();
        for record in records_to_write {
            let line = self.format_record_as_csv(&record);
            self.writer.write_all(line.as_bytes())?;
            bytes_written += line.len();
        }

        // Force flush to disk for immediate persistence
        self.writer.flush()?;

        let write_time = write_start.elapsed();

        // Update metrics
        self.update_metrics(record_count, bytes_written, write_time, force);

        self.last_flush = Instant::now();
        Ok(())
    }

    /// Format a record as CSV line
    fn format_record_as_csv(&self, record: &Record) -> String {
        // Escape any commas or quotes in the fields
        let user = self.escape_csv_field(&record.user);
        let password = self.escape_csv_field(&record.password);
        let url = self.escape_csv_field(&record.url);

        format!("{},{},{}\n", user, password, url)
    }

    /// Escape CSV field (handle commas, quotes, newlines)
    fn escape_csv_field(&self, field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
            // Escape quotes by doubling them and wrap in quotes
            let escaped = field.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        } else {
            field.to_string()
        }
    }

    /// Update performance metrics
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

        // Calculate averages
        if self.metrics.total_write_operations > 0 {
            self.metrics.average_records_per_write =
                self.metrics.total_records_written as f64 / self.metrics.total_write_operations as f64;
        }

        if self.metrics.total_write_time.as_secs_f64() > 0.0 {
            self.metrics.average_write_throughput =
                self.metrics.total_records_written as f64 / self.metrics.total_write_time.as_secs_f64();

            self.metrics.average_io_throughput =
                (self.metrics.total_bytes_written as f64 / (1024.0 * 1024.0)) / self.metrics.total_write_time.as_secs_f64();
        }
    }

    /// Force flush any remaining records
    pub fn finish(&mut self) -> Result<()> {
        if !self.record_buffer.is_empty() {
            self.flush_batch(true)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    /// Get current batch size
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set new batch size (takes effect on next batch)
    pub fn set_batch_size(&mut self, new_size: usize) {
        self.batch_size = new_size.min(self.max_batch_size).max(100); // Minimum 100 records
    }

    /// Get current buffer utilization (0.0 to 1.0)
    pub fn get_buffer_utilization(&self) -> f64 {
        self.record_buffer.len() as f64 / self.batch_size as f64
    }

    /// Check if automatic flush is needed based on time
    pub fn should_auto_flush(&self, max_time_since_flush: Duration) -> bool {
        !self.record_buffer.is_empty() && self.last_flush.elapsed() > max_time_since_flush
    }

    /// Perform automatic flush if needed
    pub fn auto_flush_if_needed(&mut self, max_time_since_flush: Duration) -> Result<bool> {
        if self.should_auto_flush(max_time_since_flush) {
            self.flush_batch(false)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &BatchWriteMetrics {
        &self.metrics
    }

    /// Optimize batch size based on performance
    pub fn optimize_batch_size(&mut self) -> Result<usize> {
        if self.metrics.total_write_operations < 3 {
            // Not enough data to optimize
            return Ok(self.batch_size);
        }

        let current_throughput = self.metrics.average_write_throughput;
        let current_batch_size = self.batch_size;

        // If throughput is very low, try smaller batches for better responsiveness
        if current_throughput < 1000.0 {
            let new_size = (current_batch_size as f64 * 0.8) as usize;
            self.set_batch_size(new_size);
        }
        // If throughput is good but I/O throughput is low, try larger batches
        else if current_throughput > 5000.0 && self.metrics.average_io_throughput < 50.0 {
            let new_size = (current_batch_size as f64 * 1.2) as usize;
            self.set_batch_size(new_size);
        }

        Ok(self.batch_size)
    }
}

impl BatchWriteMetrics {
    /// Format metrics for display
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
            self.total_bytes_written as f64 / (1024.0 * 1024.0),
            self.automatic_flushes,
            self.forced_flushes
        )
    }

    /// Get efficiency score (0.0 to 1.0)
    pub fn get_efficiency_score(&self) -> f64 {
        if self.total_write_operations == 0 {
            return 0.0;
        }

        // Higher score for fewer write operations (better batching)
        let batching_efficiency = (self.average_records_per_write / MAX_WRITE_BATCH_SIZE_RECORDS as f64).min(1.0);

        // Higher score for higher throughput
        let throughput_efficiency = (self.average_write_throughput / 10000.0).min(1.0);

        // Higher score for fewer forced flushes
        let flush_efficiency = if self.total_write_operations > 0 {
            1.0 - (self.forced_flushes as f64 / self.total_write_operations as f64)
        } else {
            1.0
        };

        (batching_efficiency + throughput_efficiency + flush_efficiency) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

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

        let mut writer = BatchWriter::with_batch_size(&output_path, 3)?;

        // Add records one by one
        for i in 0..5 {
            let record = Record {
                user: format!("user{}@example.com", i),
                password: format!("password{}", i),
                url: format!("https://example{}.com", i),
                normalized_user: format!("user{}@example.com", i),
                normalized_url: format!("example{}.com", i),
                completeness_score: 3.0,
                field_count: 3,
                all_fields: vec![
                    format!("user{}@example.com", i),
                    format!("password{}", i),
                    format!("https://example{}.com", i)
                ],
            };
            writer.add_record(record)?;
        }

        writer.finish()?;

        // Check that file was created and has content
        let content = fs::read_to_string(&output_path)?;
        assert!(content.contains("user,password,url")); // Header
        assert!(content.contains("user0@example.com"));
        assert!(content.contains("user4@example.com"));

        let metrics = writer.get_metrics();
        assert_eq!(metrics.total_records_written, 5);
        assert!(metrics.total_write_operations >= 2); // Should have batched

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
            completeness_score: 3.0,
            field_count: 3,
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
