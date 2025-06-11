pub mod config;
pub mod constants;
pub mod processor;
pub mod checkpoint;
pub mod record;
pub mod chunk;
pub mod merger;

#[cfg(test)]
mod tests;

pub use config::ExternalSortConfig;
pub use processor::ExternalSortProcessor;
pub use record::SortRecord;
pub use checkpoint::SortCheckpoint;

use anyhow::Result;
use std::path::{Path, PathBuf};

pub struct ExternalSortStats {
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub chunks_created: usize,
    pub files_processed: usize,
    pub processing_time_ms: u64,
    pub sort_time_ms: u64,
    pub merge_time_ms: u64,
    pub disk_usage_mb: f64,
    pub peak_memory_mb: f64,
}

impl Default for ExternalSortStats {
    fn default() -> Self {
        Self {
            total_records: 0,
            unique_records: 0,
            duplicates_removed: 0,
            chunks_created: 0,
            files_processed: 0,
            processing_time_ms: 0,
            sort_time_ms: 0,
            merge_time_ms: 0,
            disk_usage_mb: 0.0,
            peak_memory_mb: 0.0,
        }
    }
}

pub async fn sort_and_deduplicate(
    input_files: &[PathBuf],
    output_file: &Path,
    config: ExternalSortConfig,
) -> Result<ExternalSortStats> {
    let mut processor = ExternalSortProcessor::new(config)?;
    processor.process(input_files, output_file).await
}
