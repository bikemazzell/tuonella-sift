use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, BufWriter};
use chrono;

use crate::constants::*;

#[derive(Error, Debug)]
pub enum DeduplicationError {
    #[error("Memory exhaustion: {message}")]
    MemoryExhaustion { message: String },

    #[error("GPU error: {message}")]
    GpuError { message: String },

    #[error("File corruption detected: {file_path} at line {line_number}")]
    FileCorruption { file_path: String, line_number: usize },

    #[error("Invalid data format: {message}")]
    InvalidDataFormat { message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Recovery failed: {message}")]
    RecoveryFailed { message: String },

    #[error("Chunk processing failed: {message}")]
    ChunkProcessingFailed { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,      // Recoverable errors (invalid lines, minor format issues)
    Medium,   // Retryable errors (temporary resource issues)
    High,     // Serious errors requiring intervention (memory exhaustion)
    Critical, // System-level errors (GPU failures, file system issues)
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error_type: String,
    pub severity: ErrorSeverity,
    pub file_path: Option<PathBuf>,
    pub line_number: Option<usize>,
    pub chunk_id: Option<String>,
    pub timestamp: Instant,
    pub retry_count: usize,
    pub additional_info: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RecoveryCheckpoint {
    pub file_index: usize,
    pub line_number: usize,
    pub records_processed: usize,
    pub chunk_size: usize,
    pub timestamp: Instant,
    pub temp_files_created: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct ErrorLogEntry {
    pub error_message: String,
    pub context: ErrorContext,
}

pub struct AsyncErrorHandler {
    error_sender: mpsc::UnboundedSender<ErrorLogEntry>,
    recovery_checkpoints: Vec<RecoveryCheckpoint>,
    current_checkpoint: Option<RecoveryCheckpoint>,
    stats: ErrorStats,
    _error_task_handle: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Default, Clone)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub recoverable_errors: usize,
    pub retry_attempts: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub corrupted_lines: usize,
    pub invalid_format_lines: usize,
    pub memory_pressure_events: usize,
    pub gpu_errors: usize,
}

impl AsyncErrorHandler {
    pub async fn new(error_log_path: PathBuf) -> Result<Self> {
        let (error_sender, mut error_receiver) = mpsc::unbounded_channel::<ErrorLogEntry>();
        
        // Spawn background task for async error logging
        let error_log_path_clone = error_log_path.clone();
        let error_task_handle = tokio::spawn(async move {
            // Open file for appending
            let error_file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&error_log_path_clone)
                .await
            {
                Ok(file) => file,
                Err(e) => {
                    eprintln!("Failed to open error log file: {}", e);
                    return;
                }
            };

            let mut error_writer = BufWriter::new(error_file);
            let mut batch_buffer = Vec::with_capacity(ERROR_LOG_BUFFER_SIZE);
            let mut last_flush = Instant::now();
            
            // Process error log entries
            while let Some(entry) = error_receiver.recv().await {
                let log_entry = format!(
                    "[{}] {:?} - {} - File: {:?}, Line: {:?}, Chunk: {:?}, Retry: {}\n",
                    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
                    entry.context.severity,
                    entry.error_message,
                    entry.context.file_path,
                    entry.context.line_number,
                    entry.context.chunk_id,
                    entry.context.retry_count
                );

                batch_buffer.push(log_entry);

                // Batch write for efficiency
                if batch_buffer.len() >= ERROR_LOG_BUFFER_SIZE 
                   || last_flush.elapsed() > Duration::from_secs(5) {
                    
                    for log_line in &batch_buffer {
                        if let Err(e) = error_writer.write_all(log_line.as_bytes()).await {
                            eprintln!("Failed to write error log: {}", e);
                        }
                    }
                    
                    if let Err(e) = error_writer.flush().await {
                        eprintln!("Failed to flush error log: {}", e);
                    }
                    
                    batch_buffer.clear();
                    last_flush = Instant::now();
                }
            }
            
            // Final flush on shutdown
            for log_line in &batch_buffer {
                let _ = error_writer.write_all(log_line.as_bytes()).await;
            }
            let _ = error_writer.flush().await;
        });

        Ok(Self {
            error_sender,
            recovery_checkpoints: Vec::new(),
            current_checkpoint: None,
            stats: ErrorStats::default(),
            _error_task_handle: error_task_handle,
        })
    }

    pub fn log_error(&mut self, error: &DeduplicationError, context: ErrorContext) -> Result<()> {
        self.stats.total_errors += 1;

        match context.severity {
            ErrorSeverity::Low => self.stats.recoverable_errors += 1,
            ErrorSeverity::Medium => self.stats.retry_attempts += 1,
            ErrorSeverity::High => self.stats.memory_pressure_events += 1,
            ErrorSeverity::Critical => self.stats.gpu_errors += 1,
        }

        let entry = ErrorLogEntry {
            error_message: error.to_string(),
            context,
        };

        // Send to background task (non-blocking)
        if let Err(_) = self.error_sender.send(entry) {
            // Channel is closed, which means the background task has shut down
            eprintln!("Warning: Error logging channel is closed");
        }

        Ok(())
    }

    pub fn create_checkpoint(&mut self, checkpoint: RecoveryCheckpoint) {
        self.recovery_checkpoints.push(checkpoint.clone());
        self.current_checkpoint = Some(checkpoint);

        // Keep only the last 10 checkpoints to prevent memory bloat
        if self.recovery_checkpoints.len() > 10 {
            self.recovery_checkpoints.remove(0);
        }
    }

    pub fn get_latest_checkpoint(&self) -> Option<&RecoveryCheckpoint> {
        self.current_checkpoint.as_ref()
    }

    pub async fn retry_with_backoff<T, F, Fut>(&mut self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut delay = Duration::from_millis(RETRY_DELAY_MS);

        for attempt in 0..MAX_RETRY_ATTEMPTS {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        self.stats.successful_recoveries += 1;
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if attempt == MAX_RETRY_ATTEMPTS - 1 {
                        self.stats.failed_recoveries += 1;
                        return Err(e);
                    }

                    let context = ErrorContext {
                        error_type: "RetryAttempt".to_string(),
                        severity: ErrorSeverity::Medium,
                        file_path: None,
                        line_number: None,
                        chunk_id: None,
                        timestamp: Instant::now(),
                        retry_count: attempt + 1,
                        additional_info: HashMap::new(),
                    };

                    let retry_error = DeduplicationError::ChunkProcessingFailed {
                        message: format!("Retry attempt {} failed: {}", attempt + 1, e),
                    };

                    self.log_error(&retry_error, context)?;

                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * RETRY_BACKOFF_MULTIPLIER) as u64
                    );
                }
            }
        }

        unreachable!()
    }

    pub fn calculate_recovery_chunk_size(&self, original_size: usize) -> usize {
        let new_size = (original_size as f64 * CHUNK_SPLIT_FACTOR) as usize;
        new_size.max(MIN_CHUNK_SIZE_RECORDS)
    }

    pub fn is_recoverable_error(&self, error: &DeduplicationError) -> bool {
        match error {
            DeduplicationError::MemoryExhaustion { .. } => true,
            DeduplicationError::GpuError { .. } => true,
            DeduplicationError::ChunkProcessingFailed { .. } => true,
            DeduplicationError::IoError { .. } => true,
            DeduplicationError::FileCorruption { .. } => false,
            DeduplicationError::InvalidDataFormat { .. } => false,
            DeduplicationError::RecoveryFailed { .. } => false,
        }
    }

    pub fn get_stats(&self) -> &ErrorStats {
        &self.stats
    }

    pub fn format_error_stats(&self) -> String {
        format!(
            "Async Error Handling Statistics:\n\
             🚨 Total Errors: {}\n\
             ✅ Recoverable Errors: {}\n\
             🔄 Retry Attempts: {}\n\
             ✅ Successful Recoveries: {}\n\
             ❌ Failed Recoveries: {}\n\
             💥 Corrupted Lines: {}\n\
             ⚠️  Invalid Format Lines: {}\n\
             🧠 Memory Pressure Events: {}\n\
             🎮 GPU Errors: {}\n\
             📊 Recovery Success Rate: {:.1}%",
            self.stats.total_errors,
            self.stats.recoverable_errors,
            self.stats.retry_attempts,
            self.stats.successful_recoveries,
            self.stats.failed_recoveries,
            self.stats.corrupted_lines,
            self.stats.invalid_format_lines,
            self.stats.memory_pressure_events,
            self.stats.gpu_errors,
            if self.stats.retry_attempts > 0 {
                (self.stats.successful_recoveries as f64 / self.stats.retry_attempts as f64) * 100.0
            } else {
                100.0
            }
        )
    }

    /// Shutdown the async error handler gracefully
    pub async fn shutdown(self) {
        // Close the channel to signal the background task to finish
        drop(self.error_sender);
        
        // Wait for the background task to complete
        let _ = self._error_task_handle.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_async_error_handler_creation() -> Result<()> {
        let temp_dir = tempdir()?;
        let error_log_path = temp_dir.path().join("test_errors.log");

        let handler = AsyncErrorHandler::new(error_log_path.clone()).await?;
        assert_eq!(handler.get_stats().total_errors, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_error_logging() -> Result<()> {
        let temp_dir = tempdir()?;
        let error_log_path = temp_dir.path().join("test_errors.log");

        let mut handler = AsyncErrorHandler::new(error_log_path.clone()).await?;

        let error = DeduplicationError::InvalidDataFormat {
            message: "Test error".to_string(),
        };

        let context = ErrorContext {
            error_type: "TestError".to_string(),
            severity: ErrorSeverity::Low,
            file_path: Some(PathBuf::from("test.csv")),
            line_number: Some(42),
            chunk_id: Some("chunk_1".to_string()),
            timestamp: Instant::now(),
            retry_count: 0,
            additional_info: HashMap::new(),
        };

        handler.log_error(&error, context)?;
        assert_eq!(handler.get_stats().total_errors, 1);
        assert_eq!(handler.get_stats().recoverable_errors, 1);

        // Give some time for background task to process
        sleep(Duration::from_millis(100)).await;
        handler.shutdown().await;

        // Check that log file was created
        assert!(error_log_path.exists());

        Ok(())
    }

    #[tokio::test]
    async fn test_recovery_checkpoint() -> Result<()> {
        let temp_dir = tempdir()?;
        let error_log_path = temp_dir.path().join("test_errors.log");

        let mut handler = AsyncErrorHandler::new(error_log_path).await?;

        let checkpoint = RecoveryCheckpoint {
            file_index: 0,
            line_number: 100,
            records_processed: 50,
            chunk_size: 1000,
            timestamp: Instant::now(),
            temp_files_created: vec![PathBuf::from("temp1.csv")],
        };

        handler.create_checkpoint(checkpoint.clone());
        assert!(handler.get_latest_checkpoint().is_some());
        assert_eq!(handler.get_latest_checkpoint().unwrap().line_number, 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_retry_with_backoff() -> Result<()> {
        let temp_dir = tempdir()?;
        let error_log_path = temp_dir.path().join("test_errors.log");

        let mut handler = AsyncErrorHandler::new(error_log_path).await?;

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();
        
        let result = handler.retry_with_backoff(move || {
            let count = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 2 {
                    Err(anyhow::anyhow!("Simulated failure"))
                } else {
                    Ok("Success".to_string())
                }
            }
        }).await?;

        assert_eq!(result, "Success");
        assert_eq!(handler.get_stats().successful_recoveries, 1);

        Ok(())
    }
}