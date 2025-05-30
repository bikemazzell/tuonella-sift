use anyhow::Result;
use std::path::PathBuf;
use std::fs::OpenOptions;
use std::io::Write;

use crate::core::error_handler::{ErrorHandler, DeduplicationError, ErrorContext};
use crate::core::memory_manager::MemoryManager;
use crate::constants::MIN_CHUNK_SIZE_RECORDS;

#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaProcessor;

#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    ReduceChunkSize { new_size: usize },
    FallbackToCpu,
    SkipAndContinue { skip_count: usize },
    SplitChunk { sub_chunk_count: usize },
    RestartFromCheckpoint,
}

#[derive(Debug)]
pub struct RecoveryResult {
    pub strategy_used: RecoveryStrategy,
    pub success: bool,
    pub records_recovered: usize,
    pub records_lost: usize,
    pub new_chunk_size: Option<usize>,
}

pub struct RecoveryManager<'a> {
    error_handler: &'a mut ErrorHandler,
    memory_manager: &'a mut MemoryManager,
    #[cfg(feature = "cuda")]
    _cuda_processor: Option<&'a CudaProcessor>,
    recovery_log_path: PathBuf,
    current_strategy: Option<RecoveryStrategy>,
    degradation_level: u8,
}

impl<'a> RecoveryManager<'a> {
    #[cfg(feature = "cuda")]
    pub fn new(
        error_handler: &'a mut ErrorHandler,
        memory_manager: &'a mut MemoryManager,
        cuda_processor: Option<&'a CudaProcessor>,
        recovery_log_path: PathBuf,
    ) -> Self {
        Self {
            error_handler,
            memory_manager,
            _cuda_processor: cuda_processor,
            recovery_log_path,
            current_strategy: None,
            degradation_level: 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(
        error_handler: &'a mut ErrorHandler,
        memory_manager: &'a mut MemoryManager,
        recovery_log_path: PathBuf,
    ) -> Self {
        Self {
            error_handler,
            memory_manager,
            recovery_log_path,
            current_strategy: None,
            degradation_level: 0,
        }
    }

    pub fn attempt_recovery(&mut self, error: &DeduplicationError, context: &ErrorContext) -> Result<RecoveryResult> {
        let strategy = self.determine_recovery_strategy(error, context);
        self.current_strategy = Some(strategy.clone());

        let result = match strategy {
            RecoveryStrategy::ReduceChunkSize { new_size } => {
                self.recover_with_reduced_chunk_size(new_size, context)
            }
            RecoveryStrategy::FallbackToCpu => {
                self.recover_with_cpu_fallback(context)
            }
            RecoveryStrategy::SkipAndContinue { skip_count } => {
                self.recover_by_skipping(skip_count, context)
            }
            RecoveryStrategy::SplitChunk { sub_chunk_count } => {
                self.recover_with_chunk_splitting(sub_chunk_count, context)
            }
            RecoveryStrategy::RestartFromCheckpoint => {
                self.recover_from_checkpoint(context)
            }
        };

        self.log_recovery_attempt(&strategy, &result)?;

        if result.success {
            self.degradation_level = (self.degradation_level.saturating_sub(1)).max(0);
        } else {
            self.degradation_level = (self.degradation_level + 1).min(5);
        }

        Ok(result)
    }

    fn determine_recovery_strategy(&self, error: &DeduplicationError, context: &ErrorContext) -> RecoveryStrategy {
        match error {
            DeduplicationError::MemoryExhaustion { .. } => {
                let current_size = self.memory_manager.get_current_chunk_size();
                let new_size = self.error_handler.calculate_recovery_chunk_size(current_size);
                RecoveryStrategy::ReduceChunkSize { new_size }
            }
            DeduplicationError::GpuError { .. } => {
                if self.degradation_level < 3 {
                    RecoveryStrategy::FallbackToCpu
                } else {
                    RecoveryStrategy::ReduceChunkSize {
                        new_size: MIN_CHUNK_SIZE_RECORDS
                    }
                }
            }
            DeduplicationError::FileCorruption { .. } => {
                RecoveryStrategy::SkipAndContinue { skip_count: 1 }
            }
            DeduplicationError::ChunkProcessingFailed { .. } => {
                if context.retry_count < 2 {
                    RecoveryStrategy::SplitChunk { sub_chunk_count: 2 }
                } else {
                    RecoveryStrategy::RestartFromCheckpoint
                }
            }
            _ => RecoveryStrategy::ReduceChunkSize {
                new_size: MIN_CHUNK_SIZE_RECORDS
            }
        }
    }

    fn recover_with_reduced_chunk_size(&mut self, new_size: usize, _context: &ErrorContext) -> RecoveryResult {
        let _old_size = self.memory_manager.get_current_chunk_size();

        self.memory_manager.force_chunk_size_adjustment(new_size);

        RecoveryResult {
            strategy_used: RecoveryStrategy::ReduceChunkSize { new_size },
            success: true,
            records_recovered: 0,
            records_lost: 0,
            new_chunk_size: Some(new_size),
        }
    }

    fn recover_with_cpu_fallback(&mut self, _context: &ErrorContext) -> RecoveryResult {
        RecoveryResult {
            strategy_used: RecoveryStrategy::FallbackToCpu,
            success: true,
            records_recovered: 0,
            records_lost: 0,
            new_chunk_size: None,
        }
    }

    fn recover_by_skipping(&mut self, skip_count: usize, _context: &ErrorContext) -> RecoveryResult {
        RecoveryResult {
            strategy_used: RecoveryStrategy::SkipAndContinue { skip_count },
            success: true,
            records_recovered: 0,
            records_lost: skip_count,
            new_chunk_size: None,
        }
    }

    fn recover_with_chunk_splitting(&mut self, sub_chunk_count: usize, _context: &ErrorContext) -> RecoveryResult {
        let current_size = self.memory_manager.get_current_chunk_size();
        let new_size = current_size / sub_chunk_count;

        self.memory_manager.force_chunk_size_adjustment(new_size.max(MIN_CHUNK_SIZE_RECORDS));

        RecoveryResult {
            strategy_used: RecoveryStrategy::SplitChunk { sub_chunk_count },
            success: true,
            records_recovered: 0,
            records_lost: 0,
            new_chunk_size: Some(new_size),
        }
    }

    fn recover_from_checkpoint(&mut self, _context: &ErrorContext) -> RecoveryResult {
        let checkpoint_available = self.error_handler.get_latest_checkpoint().is_some();

        RecoveryResult {
            strategy_used: RecoveryStrategy::RestartFromCheckpoint,
            success: checkpoint_available,
            records_recovered: 0,
            records_lost: 0,
            new_chunk_size: None,
        }
    }

    fn log_recovery_attempt(&mut self, strategy: &RecoveryStrategy, result: &RecoveryResult) -> Result<()> {
        let log_entry = format!(
            "[{}] Recovery attempt: {:?} - Success: {} - Records recovered: {} - Records lost: {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            strategy,
            result.success,
            result.records_recovered,
            result.records_lost
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.recovery_log_path)?;

        file.write_all(log_entry.as_bytes())?;
        Ok(())
    }

    pub fn should_degrade_performance(&self) -> bool {
        self.degradation_level > 2
    }

    pub fn get_degradation_level(&self) -> u8 {
        self.degradation_level
    }

    pub fn get_current_strategy(&self) -> Option<&RecoveryStrategy> {
        self.current_strategy.as_ref()
    }

    pub fn reset_degradation(&mut self) {
        self.degradation_level = 0;
        self.current_strategy = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::collections::HashMap;
    use std::time::Instant;
    use crate::core::error_handler::ErrorSeverity;

    #[test]
    fn test_recovery_strategy_determination() {
        let temp_dir = tempdir().unwrap();
        let error_log_path = temp_dir.path().join("error.log");
        let recovery_log_path = temp_dir.path().join("recovery.log");

        let mut error_handler = ErrorHandler::new(error_log_path).unwrap();
        let mut memory_manager = MemoryManager::new(Some(1)).unwrap();

        #[cfg(feature = "cuda")]
        let recovery_manager = RecoveryManager::new(
            &mut error_handler,
            &mut memory_manager,
            None,
            recovery_log_path,
        );

        #[cfg(not(feature = "cuda"))]
        let recovery_manager = RecoveryManager::new(
            &mut error_handler,
            &mut memory_manager,
            recovery_log_path,
        );

        let error = DeduplicationError::MemoryExhaustion {
            message: "Test memory exhaustion".to_string(),
        };

        let context = ErrorContext {
            error_type: "Test".to_string(),
            severity: ErrorSeverity::High,
            file_path: None,
            line_number: None,
            chunk_id: None,
            timestamp: Instant::now(),
            retry_count: 0,
            additional_info: HashMap::new(),
        };

        let strategy = recovery_manager.determine_recovery_strategy(&error, &context);

        match strategy {
            RecoveryStrategy::ReduceChunkSize { .. } => {
                // Expected for memory exhaustion
            }
            _ => panic!("Unexpected strategy for memory exhaustion"),
        }
    }
}
