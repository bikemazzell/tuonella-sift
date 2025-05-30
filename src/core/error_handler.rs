use anyhow::{Result, Context};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::constants::{
    MAX_RETRY_ATTEMPTS, RETRY_DELAY_MS, RETRY_BACKOFF_MULTIPLIER,
    CHUNK_SPLIT_FACTOR, MIN_CHUNK_SIZE_RECORDS, ERROR_LOG_BUFFER_SIZE
};

/// Comprehensive error types for the deduplication system
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

/// Error severity levels for prioritized handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,      // Recoverable errors (invalid lines, minor format issues)
    Medium,   // Retryable errors (temporary resource issues)
    High,     // Serious errors requiring intervention (memory exhaustion)
    Critical, // System-level errors (GPU failures, file system issues)
}

/// Error context information for detailed logging and recovery
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

/// Recovery checkpoint for resuming processing after failures
#[derive(Debug, Clone)]
pub struct RecoveryCheckpoint {
    pub file_index: usize,
    pub line_number: usize,
    pub records_processed: usize,
    pub chunk_size: usize,
    pub timestamp: Instant,
    pub temp_files_created: Vec<PathBuf>,
}

/// Advanced error handler implementing Section 5: Error Handling
pub struct ErrorHandler {
    error_log_path: PathBuf,
    error_writer: BufWriter<File>,
    error_buffer: Vec<ErrorContext>,
    recovery_checkpoints: Vec<RecoveryCheckpoint>,
    current_checkpoint: Option<RecoveryCheckpoint>,
    stats: ErrorStats,
}

/// Error handling statistics
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

impl ErrorHandler {
    /// Create a new error handler with logging capabilities
    pub fn new(error_log_path: PathBuf) -> Result<Self> {
        let error_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&error_log_path)
            .context("Failed to create error log file")?;

        let error_writer = BufWriter::new(error_file);

        Ok(Self {
            error_log_path,
            error_writer,
            error_buffer: Vec::with_capacity(ERROR_LOG_BUFFER_SIZE),
            recovery_checkpoints: Vec::new(),
            current_checkpoint: None,
            stats: ErrorStats::default(),
        })
    }

    /// Log an error with context information
    pub fn log_error(&mut self, error: &DeduplicationError, context: ErrorContext) -> Result<()> {
        self.stats.total_errors += 1;

        match context.severity {
            ErrorSeverity::Low => self.stats.recoverable_errors += 1,
            ErrorSeverity::Medium => self.stats.retry_attempts += 1,
            ErrorSeverity::High => self.stats.memory_pressure_events += 1,
            ErrorSeverity::Critical => self.stats.gpu_errors += 1,
        }

        // Add to buffer
        self.error_buffer.push(context.clone());

        // Write to log file
        let log_entry = format!(
            "[{}] {:?} - {} - File: {:?}, Line: {:?}, Chunk: {:?}, Retry: {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            context.severity,
            error,
            context.file_path,
            context.line_number,
            context.chunk_id,
            context.retry_count
        );

        self.error_writer.write_all(log_entry.as_bytes())?;

        // Flush buffer if full
        if self.error_buffer.len() >= ERROR_LOG_BUFFER_SIZE {
            self.flush_error_buffer()?;
        }

        Ok(())
    }

    /// Flush error buffer to disk
    pub fn flush_error_buffer(&mut self) -> Result<()> {
        self.error_writer.flush()?;
        self.error_buffer.clear();
        Ok(())
    }

    /// Create a recovery checkpoint
    pub fn create_checkpoint(&mut self, checkpoint: RecoveryCheckpoint) {
        self.recovery_checkpoints.push(checkpoint.clone());
        self.current_checkpoint = Some(checkpoint);

        // Keep only recent checkpoints to avoid memory bloat
        if self.recovery_checkpoints.len() > 10 {
            self.recovery_checkpoints.remove(0);
        }
    }

    /// Get the latest recovery checkpoint
    pub fn get_latest_checkpoint(&self) -> Option<&RecoveryCheckpoint> {
        self.current_checkpoint.as_ref()
    }

    /// Retry operation with exponential backoff
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

                    // Log retry attempt
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

                    // Wait before retry
                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * RETRY_BACKOFF_MULTIPLIER) as u64
                    );
                }
            }
        }

        unreachable!()
    }

    /// Split chunk size for recovery from memory errors
    pub fn calculate_recovery_chunk_size(&self, original_size: usize) -> usize {
        let new_size = (original_size as f64 * CHUNK_SPLIT_FACTOR) as usize;
        new_size.max(MIN_CHUNK_SIZE_RECORDS)
    }

    /// Check if error is recoverable
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

    /// Get error handling statistics
    pub fn get_stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Format error statistics for display
    pub fn format_error_stats(&self) -> String {
        format!(
            "Error Handling Statistics:\n\
             Total Errors: {}\n\
             Recoverable Errors: {}\n\
             Retry Attempts: {}\n\
             Successful Recoveries: {}\n\
             Failed Recoveries: {}\n\
             Corrupted Lines: {}\n\
             Invalid Format Lines: {}\n\
             Memory Pressure Events: {}\n\
             GPU Errors: {}\n\
             Recovery Success Rate: {:.1}%",
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
}

impl Drop for ErrorHandler {
    fn drop(&mut self) {
        let _ = self.flush_error_buffer();
    }
}
