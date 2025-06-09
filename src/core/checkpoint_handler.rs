/// Checkpoint handler for managing checkpoint saves during processing
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::path::{Path, PathBuf};
use anyhow::Result;
use crate::core::checkpoint::{ProcessingState, CheckpointManager, ProcessingPhase};

/// Thread-safe checkpoint handler that can be shared across processing functions
pub struct CheckpointHandler {
    checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    processing_state: Arc<Mutex<ProcessingState>>,
    shutdown_flag: Option<Arc<AtomicBool>>,
    verbose: bool,
}

impl CheckpointHandler {
    /// Create a new checkpoint handler
    pub fn new(
        temp_dir: &Path,
        auto_save_interval: u64,
        processing_state: ProcessingState,
        shutdown_flag: Option<Arc<AtomicBool>>,
        verbose: bool,
    ) -> Self {
        let checkpoint_manager = Arc::new(Mutex::new(
            CheckpointManager::new(temp_dir, auto_save_interval)
        ));
        let processing_state = Arc::new(Mutex::new(processing_state));

        Self {
            checkpoint_manager,
            processing_state,
            shutdown_flag,
            verbose,
        }
    }

    /// Update the current file being processed
    pub fn update_current_file(&self, file_index: usize, _file_path: &Path, total_lines: usize) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        state.current_file_index = file_index;
        state.current_file_lines_processed = 0;
        state.current_file_total_lines = total_lines;
        state.current_file_byte_offset = 0;
        state.processing_phase = ProcessingPhase::FileProcessing { file_index };
        
        if self.verbose {
            println!("📍 Updated checkpoint state: Processing file {} of {}", 
                file_index + 1, state.discovered_files.len());
        }
        
        Ok(())
    }

    /// Update progress within current file
    pub fn update_file_progress(&self, lines_processed: usize, byte_offset: u64, records_processed: usize) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        state.current_file_lines_processed = lines_processed;
        state.current_file_byte_offset = byte_offset;
        state.total_records_processed += records_processed;
        Ok(())
    }

    /// Add a temp file that was created
    pub fn add_temp_file(&self, temp_file: PathBuf) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        // Only add if not already in the list to prevent duplicates
        if !state.temp_files_created.contains(&temp_file) {
            state.temp_files_created.push(temp_file.clone());
            if self.verbose {
                println!("📄 Added temp file to checkpoint: {}", temp_file.display());
            }
        } else if self.verbose {
            println!("⚠️ Temp file already in checkpoint, skipping: {}", temp_file.display());
        }
        Ok(())
    }

    /// Update deduplication stats
    pub fn update_stats(&self, total_records: usize, unique_records: usize, duplicates_removed: usize, invalid_records: usize) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        state.update_stats(total_records, unique_records, duplicates_removed, invalid_records);
        Ok(())
    }

    /// Check if shutdown was requested and save checkpoint if so
    pub fn check_shutdown_and_save(&self) -> Result<bool> {
        if let Some(ref flag) = self.shutdown_flag {
            if flag.load(Ordering::Relaxed) {
                if self.verbose {
                    println!("\n🛑 Shutdown signal detected - saving checkpoint immediately...");
                }
                
                // Save the checkpoint
                self.force_save_checkpoint()?;
                
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Force save checkpoint (called during shutdown or periodically)
    pub fn force_save_checkpoint(&self) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        let mut checkpoint_manager = self.checkpoint_manager.lock().unwrap();
        
        // Clean up duplicate temp files before saving
        self.deduplicate_temp_files(&mut state);
        
        checkpoint_manager.save_checkpoint(&*state)?;
        
        if self.verbose {
            println!("✅ Checkpoint saved successfully at: {}", 
                checkpoint_manager.get_checkpoint_path().display());
            println!("📊 Progress: {:.1}% complete", state.calculate_progress());
            println!("📈 Records processed: {}", state.total_records_processed);
            println!("📁 Temp files: {} unique files", state.temp_files_created.len());
        }
        
        Ok(())
    }
    
    /// Clean up duplicate temp files in the state
    fn deduplicate_temp_files(&self, state: &mut crate::core::checkpoint::ProcessingState) {
        let mut unique_temp_files = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        for temp_file in &state.temp_files_created {
            if seen.insert(temp_file.clone()) {
                unique_temp_files.push(temp_file.clone());
            } else if self.verbose {
                println!("🧹 Removing duplicate temp file from checkpoint: {}", temp_file.display());
            }
        }
        
        let original_count = state.temp_files_created.len();
        state.temp_files_created = unique_temp_files;
        
        if self.verbose && original_count != state.temp_files_created.len() {
            println!("🧹 Cleaned temp files: {} -> {} (removed {} duplicates)", 
                original_count, state.temp_files_created.len(), 
                original_count - state.temp_files_created.len());
        }
    }

    /// Check if auto-save is needed and save if so
    pub fn auto_save_if_needed(&self) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        let mut checkpoint_manager = self.checkpoint_manager.lock().unwrap();
        
        // Clean up duplicate temp files before checking if save is needed
        self.deduplicate_temp_files(&mut state);
        
        if checkpoint_manager.auto_save_if_needed(&*state)? {
            if self.verbose {
                println!("💾 Auto-saving checkpoint...");
            }
        }
        
        Ok(())
    }

    /// Track records processed and trigger auto-save if threshold reached
    pub fn track_records_and_save(&self, record_count: usize) -> Result<()> {
        {
            let mut state = self.processing_state.lock().unwrap();
            state.total_records_processed += record_count;
        }

        let mut checkpoint_manager = self.checkpoint_manager.lock().unwrap();
        if checkpoint_manager.track_records_processed(record_count) {
            let mut state = self.processing_state.lock().unwrap();
            // Clean up duplicate temp files before saving
            self.deduplicate_temp_files(&mut state);
            checkpoint_manager.save_checkpoint(&*state)?;
        }
        
        Ok(())
    }

    /// Get a copy of the current processing state
    pub fn get_state(&self) -> ProcessingState {
        self.processing_state.lock().unwrap().clone()
    }

    /// Set the processing phase
    pub fn set_phase(&self, phase: ProcessingPhase) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        state.processing_phase = phase;
        Ok(())
    }

    /// Mark a temp file as processed during deduplication
    pub fn mark_temp_file_processed(&self, temp_file: PathBuf) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        if !state.temp_files_processed.contains(&temp_file) {
            state.temp_files_processed.push(temp_file);
        }
        Ok(())
    }

    /// Update output file information for resumption
    pub fn update_output_file_info(&self, records_count: usize, checksum: String) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        state.output_records_count = records_count;
        state.output_file_checksum = Some(checksum);
        Ok(())
    }

    /// Get list of unprocessed temp files
    pub fn get_unprocessed_temp_files(&self) -> Vec<PathBuf> {
        let state = self.processing_state.lock().unwrap();
        state.temp_files_created
            .iter()
            .filter(|file| !state.temp_files_processed.contains(file))
            .cloned()
            .collect()
    }
    
    /// Mark a CSV file as completed and update current file index
    pub fn mark_file_completed(&self, file_path: PathBuf) -> Result<()> {
        let mut state = self.processing_state.lock().unwrap();
        if !state.completed_files.contains(&file_path) {
            state.completed_files.push(file_path.clone());
            
            // Update current_file_index to reflect the completed file
            if let Some(index) = state.discovered_files.iter().position(|p| p == &file_path) {
                state.current_file_index = index;
                
                if self.verbose {
                    println!("📍 Updated current_file_index to {} after completing file: {}", 
                        index, file_path.display());
                }
            }
        }
        Ok(())
    }
}