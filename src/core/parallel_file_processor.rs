use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs::File;
use std::io::BufWriter;
use crate::config::model::Config;
use crate::core::memory_manager::MemoryManager;
use crate::core::checkpoint_handler::CheckpointHandler;
use crate::core::deduplication::process_single_csv_with_checkpoint;
use crate::constants::{
    TEMP_FILE_PREFIX
};

pub struct ParallelFileProcessor {
    config: Config,
}

impl ParallelFileProcessor {
    pub fn new(config: Config) -> Self {
        Self {
            config,
        }
    }

    pub fn process_files_parallel(
        &self,
        csv_files: &[PathBuf],
        temp_dir: &Path,
        verbose: bool,
        shutdown_flag: Option<Arc<AtomicBool>>,
        checkpoint_handler: Option<Arc<CheckpointHandler>>,
    ) -> Result<Vec<PathBuf>> {
        if csv_files.is_empty() {
            return Ok(Vec::new());
        }

        let mut temp_files = Vec::new();
        let mut shared_memory_manager = MemoryManager::from_config(&self.config)?;

        for (i, csv_file) in csv_files.iter().enumerate() {
            if let Some(ref flag) = shutdown_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let temp_file = temp_dir.join(format!("{}{}.csv", TEMP_FILE_PREFIX, i));

            if self.process_single_file(
                csv_file,
                &temp_file,
                temp_dir,
                &mut shared_memory_manager,
                verbose,
                shutdown_flag.clone(),
                checkpoint_handler.clone(),
            )? {
                if temp_file.exists() &&
                   temp_file.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                    temp_files.push(temp_file);
                }
            }
        }

        Ok(temp_files)
    }

    fn process_single_file(
        &self,
        csv_file: &Path,
        temp_file: &Path,
        temp_dir: &Path,
        memory_manager: &mut MemoryManager,
        verbose: bool,
        shutdown_flag: Option<Arc<AtomicBool>>,
        checkpoint_handler: Option<Arc<CheckpointHandler>>,
    ) -> Result<bool> {
        if let Some(ref flag) = shutdown_flag {
            if flag.load(Ordering::Relaxed) {
                return Ok(false);
            }
        }

        if verbose {
            let file_name = csv_file.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            println!("📂 Processing: {}", file_name);
        }

        let error_log_path = temp_dir.join(format!("errors_{}.log",
            csv_file.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")));

        let error_file = File::create(&error_log_path)?;
        let mut error_writer = BufWriter::new(error_file);

        process_single_csv_with_checkpoint(
            csv_file,
            temp_file,
            &mut error_writer,
            memory_manager,
            &self.config,
            verbose,
            shutdown_flag,
            checkpoint_handler,
        )?;

        Ok(true)
    }
}

impl Default for ParallelFileProcessor {
    fn default() -> Self {
        Self {
            config: Config::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parallel_file_processor_creation() {
        let config = Config::default();
        let _processor = ParallelFileProcessor::new(config);
    }

    #[test]
    fn test_empty_file_list() -> Result<()> {
        let config = Config::default();
        let processor = ParallelFileProcessor::new(config);
        let temp_dir = tempdir()?;

        let result = processor.process_files_parallel(
            &[],
            temp_dir.path(),
            false,
            None,
            None,
        )?;

        assert!(result.is_empty());
        Ok(())
    }
}
