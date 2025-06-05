use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::Utc;

/// Represents the current state of processing that can be saved and resumed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingState {
    /// Timestamp when the checkpoint was created
    pub timestamp: String,
    /// Input directory being processed
    pub input_directory: PathBuf,
    /// Output file path
    pub output_path: PathBuf,
    /// Configuration file used
    pub config_path: PathBuf,
    /// List of CSV files discovered in input directory
    pub discovered_files: Vec<PathBuf>,
    /// Index of the current file being processed
    pub current_file_index: usize,
    /// Number of lines processed in the current file
    pub current_file_lines_processed: usize,
    /// Total lines in the current file
    pub current_file_total_lines: usize,
    /// List of temporary files created so far
    pub temp_files_created: Vec<PathBuf>,
    /// Total records processed across all files
    pub total_records_processed: usize,
    /// Total unique records found so far
    pub unique_records_count: usize,
    /// Total duplicates removed so far
    pub duplicates_removed: usize,
    /// Total invalid records encountered
    pub invalid_records: usize,
    /// Processing start time (for elapsed time calculation)
    pub processing_start_time: u64,
    /// Whether CUDA was enabled for this session
    pub cuda_enabled: bool,
    /// Whether verbose mode was enabled
    pub verbose_mode: bool,
    /// Current chunk size being used
    pub current_chunk_size: usize,
    /// Memory usage configuration
    pub memory_usage_percent: u8,
}

impl ProcessingState {
    /// Create a new processing state
    pub fn new(
        input_directory: PathBuf,
        output_path: PathBuf,
        config_path: PathBuf,
        cuda_enabled: bool,
        verbose_mode: bool,
        memory_usage_percent: u8,
    ) -> Self {
        let now = Utc::now();
        let timestamp = now.format("%Y-%m-%d %H:%M:%S UTC").to_string();
        let processing_start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            timestamp,
            input_directory,
            output_path,
            config_path,
            discovered_files: Vec::new(),
            current_file_index: 0,
            current_file_lines_processed: 0,
            current_file_total_lines: 0,
            temp_files_created: Vec::new(),
            total_records_processed: 0,
            unique_records_count: 0,
            duplicates_removed: 0,
            invalid_records: 0,
            processing_start_time,
            cuda_enabled,
            verbose_mode,
            current_chunk_size: 0,
            memory_usage_percent,
        }
    }

    /// Update file processing progress
    pub fn update_file_progress(&mut self, file_index: usize, lines_processed: usize, total_lines: usize) {
        self.current_file_index = file_index;
        self.current_file_lines_processed = lines_processed;
        self.current_file_total_lines = total_lines;
    }

    /// Add a temporary file to the list
    pub fn add_temp_file(&mut self, temp_file: PathBuf) {
        self.temp_files_created.push(temp_file);
    }

    /// Update processing statistics
    pub fn update_stats(&mut self, total_processed: usize, unique: usize, duplicates: usize, invalid: usize) {
        self.total_records_processed = total_processed;
        self.unique_records_count = unique;
        self.duplicates_removed = duplicates;
        self.invalid_records = invalid;
    }

    /// Calculate processing progress as a percentage
    pub fn calculate_progress(&self) -> f64 {
        if self.discovered_files.is_empty() {
            return 0.0;
        }

        let files_completed = self.current_file_index;
        let current_file_progress = if self.current_file_total_lines > 0 {
            self.current_file_lines_processed as f64 / self.current_file_total_lines as f64
        } else {
            0.0
        };

        let total_progress = (files_completed as f64 + current_file_progress) / self.discovered_files.len() as f64;
        (total_progress * 100.0).min(100.0)
    }

    /// Get elapsed processing time in seconds
    pub fn get_elapsed_time(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(self.processing_start_time)
    }
}

/// Manages checkpoint creation, saving, and loading
pub struct CheckpointManager {
    checkpoint_path: PathBuf,
    temp_directory: PathBuf,
    auto_save_interval: u64, // seconds
    last_save_time: SystemTime,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(temp_directory: &Path, auto_save_interval: u64) -> Self {
        let checkpoint_path = temp_directory.join("checkpoint.json");
        
        Self {
            checkpoint_path,
            temp_directory: temp_directory.to_path_buf(),
            auto_save_interval,
            last_save_time: SystemTime::now(),
        }
    }

    /// Check if a checkpoint file exists
    pub fn checkpoint_exists(&self) -> bool {
        self.checkpoint_path.exists()
    }

    /// Load processing state from checkpoint file
    pub fn load_checkpoint(&self) -> Result<ProcessingState> {
        if !self.checkpoint_exists() {
            return Err(anyhow::anyhow!("No checkpoint file found at {}", self.checkpoint_path.display()));
        }

        let file = File::open(&self.checkpoint_path)?;
        let reader = BufReader::new(file);
        let state: ProcessingState = serde_json::from_reader(reader)?;
        
        Ok(state)
    }

    /// Save processing state to checkpoint file
    pub fn save_checkpoint(&mut self, state: &ProcessingState) -> Result<()> {
        // Ensure temp directory exists
        fs::create_dir_all(&self.temp_directory)?;

        let file = File::create(&self.checkpoint_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, state)?;
        
        self.last_save_time = SystemTime::now();
        Ok(())
    }

    /// Check if it's time for an automatic save
    pub fn should_auto_save(&self) -> bool {
        if let Ok(elapsed) = self.last_save_time.elapsed() {
            elapsed.as_secs() >= self.auto_save_interval
        } else {
            true // If we can't determine elapsed time, save to be safe
        }
    }

    /// Remove the checkpoint file (called on successful completion)
    pub fn cleanup_checkpoint(&self) -> Result<()> {
        if self.checkpoint_exists() {
            fs::remove_file(&self.checkpoint_path)?;
        }
        Ok(())
    }

    /// Validate that a checkpoint is compatible with current run parameters
    pub fn validate_checkpoint(&self, state: &ProcessingState, input_dir: &Path, output_path: &Path, config_path: &Path) -> Result<()> {
        // Check if input directory matches
        if state.input_directory != input_dir {
            return Err(anyhow::anyhow!(
                "Checkpoint input directory mismatch: expected {}, found {}",
                input_dir.display(),
                state.input_directory.display()
            ));
        }

        // Check if output path matches
        if state.output_path != output_path {
            return Err(anyhow::anyhow!(
                "Checkpoint output path mismatch: expected {}, found {}",
                output_path.display(),
                state.output_path.display()
            ));
        }

        // Check if config path matches
        if state.config_path != config_path {
            return Err(anyhow::anyhow!(
                "Checkpoint config path mismatch: expected {}, found {}",
                config_path.display(),
                state.config_path.display()
            ));
        }

        // Verify that temp files still exist
        for temp_file in &state.temp_files_created {
            if !temp_file.exists() {
                return Err(anyhow::anyhow!(
                    "Checkpoint temp file missing: {}",
                    temp_file.display()
                ));
            }
        }

        Ok(())
    }

    /// Get the checkpoint file path
    pub fn get_checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_processing_state_creation() {
        let input_dir = PathBuf::from("/test/input");
        let output_path = PathBuf::from("/test/output.csv");
        let config_path = PathBuf::from("/test/config.json");
        
        let state = ProcessingState::new(
            input_dir.clone(),
            output_path.clone(),
            config_path.clone(),
            true,
            false,
            50,
        );

        assert_eq!(state.input_directory, input_dir);
        assert_eq!(state.output_path, output_path);
        assert_eq!(state.config_path, config_path);
        assert_eq!(state.cuda_enabled, true);
        assert_eq!(state.verbose_mode, false);
        assert_eq!(state.memory_usage_percent, 50);
    }

    #[test]
    fn test_checkpoint_manager() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = CheckpointManager::new(temp_dir.path(), 30);

        assert!(!manager.checkpoint_exists());

        let state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            true,
            75,
        );

        manager.save_checkpoint(&state)?;
        assert!(manager.checkpoint_exists());

        let loaded_state = manager.load_checkpoint()?;
        assert_eq!(loaded_state.input_directory, state.input_directory);
        assert_eq!(loaded_state.cuda_enabled, state.cuda_enabled);

        manager.cleanup_checkpoint()?;
        assert!(!manager.checkpoint_exists());

        Ok(())
    }
}
