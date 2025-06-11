use anyhow::Result;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, BufRead, Write};
use std::path::Path;
use std::cmp::Reverse;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::external_sort::record::SortRecord;
use crate::external_sort::checkpoint::{ChunkMetadata, SortCheckpoint};
use crate::external_sort::constants::*;

pub struct ChunkMerger {
    io_buffer_size: usize,
    case_sensitive: bool,
    progress_interval_seconds: u64,
}

#[derive(Debug)]
struct MergeEntry {
    record: SortRecord,
    chunk_id: usize,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.dedup_key(false) == other.record.dedup_key(false)
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.record.dedup_key(false).cmp(&self.record.dedup_key(false))
    }
}

impl ChunkMerger {
    pub fn new(io_buffer_size: usize, case_sensitive: bool, progress_interval_seconds: u64) -> Self {
        Self {
            io_buffer_size,
            case_sensitive,
            progress_interval_seconds,
        }
    }

    pub async fn merge_chunks(
        &self,
        chunks: &[ChunkMetadata],
        output_file: &Path,
        checkpoint: &mut SortCheckpoint,
        shutdown_flag: Arc<AtomicBool>,
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        checkpoint.merge_progress.started = true;

        let mut chunk_readers = Vec::new();
        for chunk in chunks {
            let file = File::open(&chunk.file_path)?;
            let reader = BufReader::with_capacity(self.io_buffer_size, file);
            chunk_readers.push(reader);
        }

        let output = File::create(output_file)?;
        let mut writer = BufWriter::with_capacity(
            OUTPUT_BUFFER_SIZE_KB * BYTES_PER_KB,
            output,
        );

        let mut merge_heap = BinaryHeap::with_capacity(MERGE_HEAP_INITIAL_CAPACITY);

        for (chunk_id, reader) in chunk_readers.iter_mut().enumerate() {
            if let Some(record) = self.read_next_record(reader)? {
                merge_heap.push(Reverse(MergeEntry { record, chunk_id }));
            }
        }

        let mut last_dedup_key: Option<String> = None;
        let mut records_written = 0;
        let mut duplicates_removed = 0;
        let mut progress_counter = 0;
        let total_estimated_records: usize = chunks.iter().map(|c| c.record_count).sum();
        let mut last_progress_time = std::time::Instant::now();
        let progress_interval = std::time::Duration::from_secs(self.progress_interval_seconds);

        while let Some(Reverse(merge_entry)) = merge_heap.pop() {
            // Check for shutdown signal every 1000 records for responsiveness
            if progress_counter % 1000 == 0 && shutdown_flag.load(Ordering::Relaxed) {
                println!("🛑 Merge interrupted at {:.1}% progress",
                    if total_estimated_records > 0 {
                        (progress_counter as f64 / total_estimated_records as f64 * 100.0).min(100.0)
                    } else {
                        0.0
                    });
                break;
            }

            let current_key = merge_entry.record.dedup_key(self.case_sensitive);

            if last_dedup_key.as_ref() != Some(&current_key) {
                writeln!(writer, "{}", merge_entry.record.to_csv_line())?;
                records_written += 1;
                last_dedup_key = Some(current_key);
            } else {
                duplicates_removed += 1;
            }

            if let Some(next_record) = self.read_next_record(&mut chunk_readers[merge_entry.chunk_id])? {
                merge_heap.push(Reverse(MergeEntry {
                    record: next_record,
                    chunk_id: merge_entry.chunk_id,
                }));
            }

            progress_counter += 1;

            // Show progress based on time interval instead of record count
            if last_progress_time.elapsed() >= progress_interval {
                checkpoint.merge_progress.records_written = records_written;
                checkpoint.merge_progress.duplicates_removed = duplicates_removed;
                checkpoint.merge_progress.current_output_size = writer.get_ref().metadata()?.len();

                let progress_pct = if total_estimated_records > 0 {
                    (progress_counter as f64 / total_estimated_records as f64 * 100.0).min(100.0)
                } else {
                    0.0
                };

                println!("🔗 Merge progress: {:.1}% ({} unique, {} duplicates removed)",
                    progress_pct, records_written, duplicates_removed);

                last_progress_time = std::time::Instant::now();
            }
        }

        writer.flush()?;

        checkpoint.merge_progress.records_written = records_written;
        checkpoint.merge_progress.duplicates_removed = duplicates_removed;
        checkpoint.merge_progress.current_output_size = std::fs::metadata(output_file)?.len();
        checkpoint.stats.unique_records = records_written;
        checkpoint.stats.duplicates_removed = duplicates_removed;

        Ok(())
    }

    fn read_next_record(&self, reader: &mut BufReader<File>) -> Result<Option<SortRecord>> {
        let mut line = String::new();
        loop {
            match reader.read_line(&mut line) {
                Ok(0) => return Ok(None), // EOF
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        line.clear();
                        continue;
                    }

                    if let Some(record) = SortRecord::from_csv_line(trimmed) {
                        return Ok(Some(record));
                    } else {
                        // Skip invalid lines and continue
                        line.clear();
                        continue;
                    }
                }
                Err(_) => {
                    // Handle UTF-8 errors gracefully - skip invalid lines
                    line.clear();
                    continue;
                }
            }
        }
    }

    pub fn estimate_merge_time(&self, chunks: &[ChunkMetadata]) -> u64 {
        let total_records: usize = chunks.iter().map(|c| c.record_count).sum();
        let estimated_ms = (total_records as f64 * 0.001) as u64;
        estimated_ms.max(1000)
    }

    pub fn validate_chunks(&self, chunks: &[ChunkMetadata]) -> Result<()> {
        for chunk in chunks {
            if !chunk.file_path.exists() {
                return Err(anyhow::anyhow!(
                    "Chunk file does not exist: {}",
                    chunk.file_path.display()
                ));
            }

            if !chunk.is_sorted {
                return Err(anyhow::anyhow!(
                    "Chunk {} is not sorted",
                    chunk.chunk_id
                ));
            }
        }
        Ok(())
    }
}
