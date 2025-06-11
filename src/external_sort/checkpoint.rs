use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use crate::constants::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortCheckpoint {
    pub version: u32,
    pub timestamp: u64,
    pub phase: ProcessingPhase,
    pub input_files: Vec<PathBuf>,
    pub output_file: PathBuf,
    pub temp_directory: PathBuf,
    pub completed_files: Vec<PathBuf>,
    pub current_file_index: usize,
    pub current_file_progress: FileProgress,
    pub created_chunks: Vec<ChunkMetadata>,
    pub merge_progress: MergeProgress,
    pub stats: CheckpointStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPhase {
    Initialization,
    FileProcessing,
    ChunkSorting,
    Merging,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileProgress {
    pub file_path: PathBuf,
    pub bytes_processed: u64,
    pub lines_processed: usize,
    pub records_processed: usize,
    pub current_chunk_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub file_path: PathBuf,
    pub record_count: usize,
    pub file_size_bytes: u64,
    pub is_sorted: bool,
    pub source_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeProgress {
    pub started: bool,
    pub completed_chunks: Vec<usize>,
    pub current_output_size: u64,
    pub records_written: usize,
    pub duplicates_removed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointStats {
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub files_processed: usize,
    pub chunks_created: usize,
    pub processing_time_ms: u64,
    pub peak_memory_mb: f64,
    pub disk_usage_mb: f64,
}

impl Default for SortCheckpoint {
    fn default() -> Self {
        Self {
            version: 1,
            timestamp: current_timestamp(),
            phase: ProcessingPhase::Initialization,
            input_files: Vec::new(),
            output_file: PathBuf::new(),
            temp_directory: PathBuf::new(),
            completed_files: Vec::new(),
            current_file_index: 0,
            current_file_progress: FileProgress::default(),
            created_chunks: Vec::new(),
            merge_progress: MergeProgress::default(),
            stats: CheckpointStats::default(),
        }
    }
}

impl Default for FileProgress {
    fn default() -> Self {
        Self {
            file_path: PathBuf::new(),
            bytes_processed: 0,
            lines_processed: 0,
            records_processed: 0,
            current_chunk_id: 0,
        }
    }
}

impl Default for MergeProgress {
    fn default() -> Self {
        Self {
            started: false,
            completed_chunks: Vec::new(),
            current_output_size: 0,
            records_written: 0,
            duplicates_removed: 0,
        }
    }
}

impl Default for CheckpointStats {
    fn default() -> Self {
        Self {
            total_records: 0,
            unique_records: 0,
            duplicates_removed: 0,
            files_processed: 0,
            chunks_created: 0,
            processing_time_ms: 0,
            peak_memory_mb: 0.0,
            disk_usage_mb: 0.0,
        }
    }
}

impl SortCheckpoint {
    pub fn new(input_files: Vec<PathBuf>, output_file: PathBuf, temp_directory: PathBuf) -> Self {
        Self {
            input_files,
            output_file,
            temp_directory,
            ..Default::default()
        }
    }

    pub fn save(&self, checkpoint_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(checkpoint_dir)?;
        let checkpoint_path = checkpoint_dir.join(CHECKPOINT_FILE_NAME);
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(checkpoint_path, content)?;
        Ok(())
    }

    pub fn load(checkpoint_dir: &Path) -> Result<Self> {
        let checkpoint_path = checkpoint_dir.join(CHECKPOINT_FILE_NAME);
        let content = std::fs::read_to_string(checkpoint_path)?;
        let checkpoint: Self = serde_json::from_str(&content)?;
        Ok(checkpoint)
    }

    pub fn exists(checkpoint_dir: &Path) -> bool {
        checkpoint_dir.join(CHECKPOINT_FILE_NAME).exists()
    }

    pub fn update_timestamp(&mut self) {
        self.timestamp = current_timestamp();
    }

    pub fn progress_percentage(&self) -> f64 {
        if self.input_files.is_empty() {
            return 0.0;
        }

        match self.phase {
            ProcessingPhase::Initialization => 0.0,
            ProcessingPhase::FileProcessing => {
                let file_progress = self.completed_files.len() as f64 / self.input_files.len() as f64;
                file_progress * 80.0
            }
            ProcessingPhase::ChunkSorting => 80.0,
            ProcessingPhase::Merging => {
                if self.created_chunks.is_empty() {
                    return 85.0;
                }
                let merge_progress = self.merge_progress.completed_chunks.len() as f64 
                    / self.created_chunks.len() as f64;
                80.0 + (merge_progress * 15.0)
            }
            ProcessingPhase::Completed => 100.0,
            ProcessingPhase::Failed(_) => 0.0,
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
