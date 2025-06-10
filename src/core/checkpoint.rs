use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::Utc;
use sha2::{Sha256, Digest};

/// Represents the processing phase when checkpoint was created
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessingPhase {
    FileDiscovery,
    FileProcessing { file_index: usize },
    Deduplication,
    FinalMerge,
    Completed,
}

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
    /// Exact byte offset in the current file for precise resume
    pub current_file_byte_offset: u64,
    /// Total lines in the current file
    pub current_file_total_lines: usize,
    /// List of temporary files created so far
    pub temp_files_created: Vec<PathBuf>,
    /// Checksums of temp files for integrity verification
    pub temp_file_checksums: HashMap<PathBuf, String>,
    /// List of temp files already processed during deduplication
    pub temp_files_processed: Vec<PathBuf>,
    /// Number of records in existing output file (for resumption)
    pub output_records_count: usize,
    /// Checksum of existing output file (for integrity verification)
    pub output_file_checksum: Option<String>,
    /// Processing phase when checkpoint was created
    pub processing_phase: ProcessingPhase,
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
    /// Files that have been completely processed (for skipping)
    pub completed_files: Vec<PathBuf>,
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
            current_file_byte_offset: 0,
            current_file_total_lines: 0,
            temp_files_created: Vec::new(),
            temp_file_checksums: HashMap::new(),
            temp_files_processed: Vec::new(),
            output_records_count: 0,
            output_file_checksum: None,
            processing_phase: ProcessingPhase::FileDiscovery,
            total_records_processed: 0,
            unique_records_count: 0,
            duplicates_removed: 0,
            invalid_records: 0,
            processing_start_time,
            cuda_enabled,
            verbose_mode,
            current_chunk_size: 0,
            memory_usage_percent,
            completed_files: Vec::new(),
        }
    }

    /// Update file processing progress with byte offset
    pub fn update_file_progress(&mut self, file_index: usize, lines_processed: usize, total_lines: usize) {
        self.current_file_index = file_index;
        self.current_file_lines_processed = lines_processed;
        self.current_file_total_lines = total_lines;
    }

    /// Update file processing progress with precise byte offset
    pub fn update_file_progress_with_offset(&mut self, file_index: usize, lines_processed: usize, total_lines: usize, byte_offset: u64) {
        self.current_file_index = file_index;
        self.current_file_lines_processed = lines_processed;
        self.current_file_total_lines = total_lines;
        self.current_file_byte_offset = byte_offset;
    }

    /// Update processing phase
    pub fn update_phase(&mut self, phase: ProcessingPhase) {
        self.processing_phase = phase;
    }

    /// Mark a file as completely processed
    pub fn mark_file_completed(&mut self, file_path: PathBuf) {
        if !self.completed_files.contains(&file_path) {
            self.completed_files.push(file_path);
        }
    }

    /// Add a temporary file to the list
    pub fn add_temp_file(&mut self, temp_file: PathBuf) {
        self.temp_files_created.push(temp_file);
    }

    /// Add a temporary file with checksum for integrity verification
    pub fn add_temp_file_with_checksum(&mut self, temp_file: PathBuf) -> Result<()> {
        let checksum = Self::calculate_file_checksum(&temp_file)?;
        self.temp_file_checksums.insert(temp_file.clone(), checksum);
        self.temp_files_created.push(temp_file);
        Ok(())
    }

    /// Calculate SHA256 checksum of a file
    fn calculate_file_checksum(file_path: &Path) -> Result<String> {
        let mut file = File::open(file_path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Verify integrity of temp files using checksums
    pub fn verify_temp_file_integrity(&self) -> Result<Vec<PathBuf>> {
        let mut corrupted_files = Vec::new();
        
        for (file_path, expected_checksum) in &self.temp_file_checksums {
            if !file_path.exists() {
                corrupted_files.push(file_path.clone());
                continue;
            }
            
            match Self::calculate_file_checksum(file_path) {
                Ok(actual_checksum) => {
                    if actual_checksum != *expected_checksum {
                        corrupted_files.push(file_path.clone());
                    }
                }
                Err(_) => {
                    corrupted_files.push(file_path.clone());
                }
            }
        }
        
        Ok(corrupted_files)
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
    incremental_save_enabled: bool,
    records_processed_since_save: usize,
    auto_save_record_threshold: usize,
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
            incremental_save_enabled: true,
            records_processed_since_save: 0,
            auto_save_record_threshold: 100_000, // Save every 100k records
        }
    }

    /// Create a new checkpoint manager with custom thresholds
    pub fn new_with_thresholds(
        temp_directory: &Path, 
        auto_save_interval: u64, 
        record_threshold: usize
    ) -> Self {
        let checkpoint_path = temp_directory.join("checkpoint.json");
        
        Self {
            checkpoint_path,
            temp_directory: temp_directory.to_path_buf(),
            auto_save_interval,
            last_save_time: SystemTime::now(),
            incremental_save_enabled: true,
            records_processed_since_save: 0,
            auto_save_record_threshold: record_threshold,
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
        self.reset_records_counter();
        Ok(())
    }

    /// Save checkpoint if incremental conditions are met
    pub fn auto_save_if_needed(&mut self, state: &ProcessingState) -> Result<bool> {
        if self.should_incremental_save() {
            self.save_checkpoint(state)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if it's time for an automatic save
    pub fn should_auto_save(&self) -> bool {
        if let Ok(elapsed) = self.last_save_time.elapsed() {
            elapsed.as_secs() >= self.auto_save_interval
        } else {
            true // If we can't determine elapsed time, save to be safe
        }
    }

    /// Track records processed and check if incremental save is needed
    pub fn track_records_processed(&mut self, count: usize) -> bool {
        if !self.incremental_save_enabled {
            return false;
        }
        
        self.records_processed_since_save += count;
        self.records_processed_since_save >= self.auto_save_record_threshold
    }

    /// Check if incremental save should be triggered (time or record count)
    pub fn should_incremental_save(&self) -> bool {
        self.should_auto_save() || 
        (self.incremental_save_enabled && self.records_processed_since_save >= self.auto_save_record_threshold)
    }

    /// Reset the records counter after save
    pub fn reset_records_counter(&mut self) {
        self.records_processed_since_save = 0;
    }

    /// Enable or disable incremental saving
    pub fn set_incremental_save(&mut self, enabled: bool) {
        self.incremental_save_enabled = enabled;
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

        // Verify temp file integrity using checksums
        let corrupted_files = state.verify_temp_file_integrity()?;
        if !corrupted_files.is_empty() {
            return Err(anyhow::anyhow!(
                "Checkpoint temp file integrity check failed for {} files: {:?}",
                corrupted_files.len(),
                corrupted_files
            ));
        }

        Ok(())
    }

    /// Get files that can be skipped (already completed)
    pub fn get_remaining_files(&self, state: &ProcessingState) -> Vec<PathBuf> {
        // If we're in the middle of processing a file (has byte offset), 
        // include it in remaining files so it can be resumed
        let include_current_file = state.current_file_byte_offset > 0 && 
                                   state.current_file_lines_processed > 0 &&
                                   state.current_file_index < state.discovered_files.len();
        
        if include_current_file {
            // Include the current file being processed plus all subsequent files
            state.discovered_files
                .iter()
                .skip(state.current_file_index)
                .filter(|file| !state.completed_files.contains(file))
                .cloned()
                .collect()
        } else {
            // Skip the current file index (which should be completed or not started)
            // and include all subsequent files
            state.discovered_files
                .iter()
                .skip(state.current_file_index + 1)
                .filter(|file| !state.completed_files.contains(file))
                .cloned()
                .collect()
        }
    }

    /// Resume from a specific byte offset in a file
    pub fn seek_to_checkpoint_position(&self, state: &ProcessingState, file_path: &Path) -> Result<File> {
        let mut file = File::open(file_path)?;
        file.seek(SeekFrom::Start(state.current_file_byte_offset))?;
        Ok(file)
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

    #[test]
    fn test_checkpoint_resume_temp_file_naming() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = CheckpointManager::new(temp_dir.path(), 30);

        // Create initial state with existing temp files
        let mut state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            true,
            75,
        );

        // Simulate having processed some files and created temp files
        state.temp_files_created = vec![
            PathBuf::from("./temp/temp_0.csv"),
            PathBuf::from("./temp/temp_1.csv"),
            PathBuf::from("./temp/temp_2.csv"),
        ];
        state.current_file_index = 3;
        state.processing_phase = ProcessingPhase::FileProcessing { file_index: 3 };

        // Save checkpoint
        manager.save_checkpoint(&state)?;

        // Load checkpoint to simulate resume
        let loaded_state = manager.load_checkpoint()?;
        
        // Verify temp file count and naming logic
        assert_eq!(loaded_state.temp_files_created.len(), 3);
        assert_eq!(loaded_state.current_file_index, 3);
        
        // The next temp file should be temp_3.csv (using temp_files_created.len())
        let next_temp_file_index = loaded_state.temp_files_created.len();
        assert_eq!(next_temp_file_index, 3);

        manager.cleanup_checkpoint()?;
        Ok(())
    }

    #[test]
    fn test_checkpoint_processing_phase_transitions() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = CheckpointManager::new(temp_dir.path(), 30);

        let mut state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            true,
            75,
        );

        // Test FileProcessing phase
        state.processing_phase = ProcessingPhase::FileProcessing { file_index: 5 };
        manager.save_checkpoint(&state)?;
        let loaded_state = manager.load_checkpoint()?;
        if let ProcessingPhase::FileProcessing { file_index } = loaded_state.processing_phase {
            assert_eq!(file_index, 5);
        } else {
            panic!("Expected FileProcessing phase");
        }

        // Test Deduplication phase
        state.processing_phase = ProcessingPhase::Deduplication;
        manager.save_checkpoint(&state)?;
        let loaded_state = manager.load_checkpoint()?;
        assert!(matches!(loaded_state.processing_phase, ProcessingPhase::Deduplication));

        manager.cleanup_checkpoint()?;
        Ok(())
    }

    #[test]
    fn test_checkpoint_temp_file_tracking() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = CheckpointManager::new(temp_dir.path(), 30);

        let mut state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            true,
            75,
        );

        // Add temp files progressively and test persistence
        state.temp_files_created.push(PathBuf::from("./temp/temp_0.csv"));
        manager.save_checkpoint(&state)?;
        
        let loaded_state = manager.load_checkpoint()?;
        assert_eq!(loaded_state.temp_files_created.len(), 1);
        assert_eq!(loaded_state.temp_files_created[0], PathBuf::from("./temp/temp_0.csv"));

        // Add more temp files
        state.temp_files_created.push(PathBuf::from("./temp/temp_1.csv"));
        state.temp_files_created.push(PathBuf::from("./temp/temp_1_1.csv")); // numbered temp file
        state.temp_files_created.push(PathBuf::from("./temp/temp_1_2.csv")); // numbered temp file
        state.temp_files_created.push(PathBuf::from("./temp/temp_2.csv"));
        
        manager.save_checkpoint(&state)?;
        let loaded_state = manager.load_checkpoint()?;
        assert_eq!(loaded_state.temp_files_created.len(), 5);
        
        // Verify numbered temp files are tracked
        assert!(loaded_state.temp_files_created.contains(&PathBuf::from("./temp/temp_1_1.csv")));
        assert!(loaded_state.temp_files_created.contains(&PathBuf::from("./temp/temp_1_2.csv")));

        manager.cleanup_checkpoint()?;
        Ok(())
    }

    #[test]
    fn test_checkpoint_file_progress_tracking() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = CheckpointManager::new(temp_dir.path(), 30);

        let mut state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            true,
            75,
        );

        // Set up file progress tracking
        state.discovered_files = vec![
            PathBuf::from("/test/file1.csv"),
            PathBuf::from("/test/file2.csv"),
            PathBuf::from("/test/file3.csv"),
        ];
        state.completed_files = vec![PathBuf::from("/test/file1.csv")];
        state.current_file_index = 1;
        state.current_file_lines_processed = 50000;
        state.current_file_total_lines = 100000;
        state.current_file_byte_offset = 2500000;

        manager.save_checkpoint(&state)?;
        let loaded_state = manager.load_checkpoint()?;

        assert_eq!(loaded_state.discovered_files.len(), 3);
        assert_eq!(loaded_state.completed_files.len(), 1);
        assert_eq!(loaded_state.current_file_index, 1);
        assert_eq!(loaded_state.current_file_lines_processed, 50000);
        assert_eq!(loaded_state.current_file_total_lines, 100000);
        assert_eq!(loaded_state.current_file_byte_offset, 2500000);

        manager.cleanup_checkpoint()?;
        Ok(())
    }

    #[test]
    fn test_get_remaining_files_with_partial_processing() -> Result<()> {
        let temp_dir = tempdir()?;
        let manager = CheckpointManager::new(temp_dir.path(), 30);

        let mut state = ProcessingState::new(
            PathBuf::from("/test/input"),
            PathBuf::from("/test/output.csv"),
            PathBuf::from("/test/config.json"),
            false,
            false,
            75,
        );

        // Set up discovered files
        state.discovered_files = vec![
            PathBuf::from("/test/file0.csv"),
            PathBuf::from("/test/file1.csv"),
            PathBuf::from("/test/file2.csv"),
            PathBuf::from("/test/file3.csv"),
        ];

        // Set up scenario: files 0-1 completed, file 2 partially processed, file 3 not started
        state.completed_files = vec![
            PathBuf::from("/test/file0.csv"),
            PathBuf::from("/test/file1.csv"),
        ];
        state.current_file_index = 2;
        state.current_file_byte_offset = 50000; // Partially processed
        state.current_file_lines_processed = 1000;

        let remaining_files = manager.get_remaining_files(&state);
        
        // Should include the partially processed file (index 2) and subsequent file (index 3)
        assert_eq!(remaining_files.len(), 2);
        assert_eq!(remaining_files[0], PathBuf::from("/test/file2.csv"));
        assert_eq!(remaining_files[1], PathBuf::from("/test/file3.csv"));

        // Test scenario where current file is completed (no byte offset)
        state.current_file_byte_offset = 0;
        state.current_file_lines_processed = 0;
        state.current_file_index = 2;
        state.completed_files.push(PathBuf::from("/test/file2.csv"));

        let remaining_files = manager.get_remaining_files(&state);
        
        // Should only include file 3 (file 2 is completed)
        assert_eq!(remaining_files.len(), 1);
        assert_eq!(remaining_files[0], PathBuf::from("/test/file3.csv"));

        Ok(())
    }
}
