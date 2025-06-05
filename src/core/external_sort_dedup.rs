use anyhow::Result;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::cmp::{Ordering, Reverse};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use rayon::prelude::*;
use crate::core::record::Record;
use crate::core::validation::parse_csv_line;
use crate::utils::system::get_memory_info;
use crate::constants::*;

#[derive(Debug, Clone)]
pub struct ExternalSortConfig {
    /// Maximum memory to use for in-memory sorting (bytes)
    pub max_memory_usage_bytes: usize,
    /// Directory for temporary files
    pub temp_directory: PathBuf,
    /// Enable compression for temporary files
    pub compression_enabled: bool,
    /// Buffer size for file I/O operations (bytes)
    pub io_buffer_size_bytes: usize,
    /// Number of parallel threads for sorting
    pub sort_threads: usize,
    /// Automatically clean up temp files when done
    pub cleanup_temp_files: bool,
    /// Enable progress reporting
    pub verbose: bool,
}

impl Default for ExternalSortConfig {
    fn default() -> Self {
        let memory_info = get_memory_info();
        let max_memory = (memory_info.1 * EXTERNAL_SORT_MEMORY_FACTOR * MB_AS_F64 * KB_AS_F64) as usize;
        
        Self {
            max_memory_usage_bytes: max_memory.max(EXTERNAL_SORT_MIN_MEMORY_BYTES),
            temp_directory: std::env::temp_dir().join("tuonella_sift_external_sort"),
            compression_enabled: true,
            io_buffer_size_bytes: EXTERNAL_SORT_BUFFER_SIZE_BYTES,
            sort_threads: num_cpus::get().max(EXTERNAL_SORT_MIN_THREADS),
            cleanup_temp_files: true,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExternalSortStats {
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub chunks_created: usize,
    pub temp_files_created: usize,
    pub total_memory_used_mb: f64,
    pub total_disk_used_mb: f64,
    pub sort_phase_duration_ms: u64,
    pub merge_phase_duration_ms: u64,
    pub total_duration_ms: u64,
}

#[derive(Debug)]
struct RecordChunk {
    records: Vec<Record>,
    estimated_size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct SortedChunk {
    pub file_path: PathBuf,
    pub record_count: usize,
    pub file_size_bytes: u64,
    pub is_compressed: bool,
}

#[derive(Debug)]
struct MergeEntry {
    record: Record,
    chunk_id: usize,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.dedup_key() == other.record.dedup_key()
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.record.dedup_key().cmp(&self.record.dedup_key())
    }
}

pub struct ExternalSortDeduplicator {
    config: ExternalSortConfig,
    stats: ExternalSortStats,
    temp_chunks: Vec<SortedChunk>,
    shutdown_flag: Option<Arc<AtomicBool>>,
}

impl ExternalSortDeduplicator {
    pub fn new(config: ExternalSortConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.temp_directory)?;
        
        Ok(Self {
            config,
            stats: ExternalSortStats::default(),
            temp_chunks: Vec::new(),
            shutdown_flag: None,
        })
    }

    pub fn with_shutdown_signal(config: ExternalSortConfig, shutdown_flag: Arc<AtomicBool>) -> Result<Self> {
        let mut dedup = Self::new(config)?;
        dedup.shutdown_flag = Some(shutdown_flag);
        Ok(dedup)
    }

    pub async fn deduplicate_large_dataset(&mut self, input_files: &[PathBuf], output_file: &Path) -> Result<ExternalSortStats> {
        let start_time = std::time::Instant::now();
        
        if self.config.verbose {
            println!("🔄 Starting external sort deduplication for {} input files", input_files.len());
            println!("   Max memory usage: {} MB", self.config.max_memory_usage_bytes / (1024 * 1024));
            println!("   Temp directory: {}", self.config.temp_directory.display());
            println!("   Sort threads: {}", self.config.sort_threads);
        }

        let sort_start = std::time::Instant::now();
        self.sort_chunks_parallel(input_files).await?;
        self.stats.sort_phase_duration_ms = sort_start.elapsed().as_millis() as u64;
        
        if self.shutdown_requested() {
            return Ok(self.stats.clone());
        }

        let merge_start = std::time::Instant::now();
        self.merge_with_deduplication(output_file).await?;
        self.stats.merge_phase_duration_ms = merge_start.elapsed().as_millis() as u64;
        
        if self.config.cleanup_temp_files {
            self.cleanup_temp_files()?;
        }

        self.stats.total_duration_ms = start_time.elapsed().as_millis() as u64;
        
        if self.config.verbose {
            self.print_final_stats();
        }

        Ok(self.stats.clone())
    }

    /// Sort chunks in parallel
    async fn sort_chunks_parallel(&mut self, input_files: &[PathBuf]) -> Result<()> {
        let mut current_chunk = RecordChunk {
            records: Vec::new(),
            estimated_size_bytes: 0,
        };

        let mut chunk_id = 0;

        for input_file in input_files {
            if self.shutdown_requested() {
                break;
            }

            if self.config.verbose {
                println!("📁 Processing file: {}", input_file.display());
            }

            let file = File::open(input_file)?;
            let reader = BufReader::with_capacity(self.config.io_buffer_size_bytes, file);

            for (line_num, line_result) in reader.lines().enumerate() {
                if self.shutdown_requested() {
                    break;
                }

                let line = line_result?;
                if line.trim().is_empty() {
                    continue;
                }

                // Parse line into record
                let fields = parse_csv_line(&line);
                if fields.len() < 2 {
                    continue; // Skip invalid lines
                }

                let record = if let Some(record) = Record::from_fields_with_config(fields, false) {
                    record
                } else {
                    continue; // Skip invalid records
                };
                let record_size = self.estimate_record_size(&record);

                // Check if we need to flush current chunk
                if current_chunk.estimated_size_bytes + record_size > self.config.max_memory_usage_bytes {
                    if !current_chunk.records.is_empty() {
                        // Sort and write current chunk
                        let sorted_chunk = self.sort_and_write_chunk(chunk_id, current_chunk).await?;
                        self.temp_chunks.push(sorted_chunk);
                        chunk_id += 1;

                        // Start new chunk
                        current_chunk = RecordChunk {
                            records: Vec::new(),
                            estimated_size_bytes: 0,
                        };
                    }
                }

                current_chunk.records.push(record);
                current_chunk.estimated_size_bytes += record_size;
                self.stats.total_records += 1;

                // Progress reporting
                if self.config.verbose && line_num % 100_000 == 0 && line_num > 0 {
                    println!("   Processed {} lines, {} chunks created", line_num, chunk_id);
                }
            }
        }

        // Handle final chunk
        if !current_chunk.records.is_empty() {
            let sorted_chunk = self.sort_and_write_chunk(chunk_id, current_chunk).await?;
            self.temp_chunks.push(sorted_chunk);
        }

        self.stats.chunks_created = self.temp_chunks.len();
        
        if self.config.verbose {
            println!("✅ Phase 1 complete: {} chunks created", self.stats.chunks_created);
        }

        Ok(())
    }

    /// Sort a chunk and write to temporary file
    async fn sort_and_write_chunk(&mut self, chunk_id: usize, mut chunk: RecordChunk) -> Result<SortedChunk> {
        let start_time = std::time::Instant::now();
        
        // Sort records in parallel
        chunk.records.par_sort_by(|a, b| a.dedup_key().cmp(&b.dedup_key()));
        
        // Write to temporary file
        let temp_file_path = self.config.temp_directory.join(format!("chunk_{:06}.csv", chunk_id));
        let file = File::create(&temp_file_path)?;
        let mut writer = BufWriter::with_capacity(self.config.io_buffer_size_bytes, file);

        let mut last_key: Option<String> = None;
        let mut records_written = 0;
        let total_records = chunk.records.len();

        for record in chunk.records {
            let current_key = record.dedup_key();
            
            // Skip duplicates within chunk
            if last_key.as_ref() != Some(&current_key) {
                writeln!(writer, "{},{},{}", record.user, record.password, record.url)?;
                records_written += 1;
                last_key = Some(current_key);
            }
        }

        writer.flush()?;
        let file_size = std::fs::metadata(&temp_file_path)?.len();
        
        self.stats.total_disk_used_mb += file_size as f64 / (1024.0 * 1024.0);
        self.stats.temp_files_created += 1;

        if self.config.verbose {
            println!("   Chunk {} sorted: {} records -> {} unique ({:.1}ms)", 
                chunk_id, total_records, records_written, start_time.elapsed().as_millis());
        }

        Ok(SortedChunk {
            file_path: temp_file_path,
            record_count: records_written,
            file_size_bytes: file_size,
            is_compressed: self.config.compression_enabled,
        })
    }

    /// Perform k-way merge with deduplication
    async fn merge_with_deduplication(&mut self, output_file: &Path) -> Result<()> {
        if self.config.verbose {
            println!("🔗 Starting k-way merge of {} chunks", self.temp_chunks.len());
        }

        // Open all chunk files for reading
        let mut chunk_readers: Vec<BufReader<File>> = Vec::new();
        for chunk in &self.temp_chunks {
            let file = File::open(&chunk.file_path)?;
            chunk_readers.push(BufReader::with_capacity(self.config.io_buffer_size_bytes, file));
        }

        // Create output writer
        let output = File::create(output_file)?;
        let mut output_writer = BufWriter::with_capacity(self.config.io_buffer_size_bytes, output);

        // Initialize priority queue with first record from each chunk
        let mut merge_heap: BinaryHeap<Reverse<MergeEntry>> = BinaryHeap::new();
        
        for (chunk_id, reader) in chunk_readers.iter_mut().enumerate() {
            if let Some(record) = self.read_next_record(reader, chunk_id)? {
                merge_heap.push(Reverse(record));
            }
        }

        // Perform k-way merge
        let mut last_dedup_key: Option<String> = None;
        let mut progress_counter = 0;

        while let Some(Reverse(merge_entry)) = merge_heap.pop() {
            if self.shutdown_requested() {
                break;
            }

            let current_key = merge_entry.record.dedup_key();
            
            // Only write if this is not a duplicate
            if last_dedup_key.as_ref() != Some(&current_key) {
                writeln!(output_writer, "{},{},{}", 
                    merge_entry.record.user,
                    merge_entry.record.password, 
                    merge_entry.record.url)?;
                
                self.stats.unique_records += 1;
                last_dedup_key = Some(current_key);
            } else {
                self.stats.duplicates_removed += 1;
            }

            // Read next record from the same chunk
            if let Some(next_record) = self.read_next_record(&mut chunk_readers[merge_entry.chunk_id], merge_entry.chunk_id)? {
                merge_heap.push(Reverse(next_record));
            }

            // Progress reporting
            progress_counter += 1;
            if self.config.verbose && progress_counter % 1_000_000 == 0 {
                println!("   Merged {} records, {} unique, {} duplicates", 
                    progress_counter, self.stats.unique_records, self.stats.duplicates_removed);
            }
        }

        output_writer.flush()?;

        if self.config.verbose {
            println!("✅ Phase 2 complete: {} unique records written", self.stats.unique_records);
        }

        Ok(())
    }

    /// Read next record from a chunk file
    fn read_next_record(&self, reader: &mut BufReader<File>, chunk_id: usize) -> Result<Option<MergeEntry>> {
        let mut line = String::new();
        match reader.read_line(&mut line)? {
            0 => Ok(None), // EOF
            _ => {
                let line = line.trim();
                if line.is_empty() {
                    return self.read_next_record(reader, chunk_id); // Skip empty lines
                }

                let fields = parse_csv_line(line);
                if fields.len() >= 3 {
                    if let Some(record) = Record::from_fields_with_config(fields, false) {
                        Ok(Some(MergeEntry {
                            record,
                            chunk_id,
                        }))
                    } else {
                        self.read_next_record(reader, chunk_id) // Skip invalid records
                    }
                } else {
                    self.read_next_record(reader, chunk_id) // Skip invalid lines
                }
            }
        }
    }

    /// Estimate memory size of a record
    fn estimate_record_size(&self, record: &Record) -> usize {
        // Rough estimate: string lengths + overhead
        record.user.len() + record.password.len() + record.url.len() + 64 // 64 bytes overhead
    }

    /// Check if shutdown was requested
    fn shutdown_requested(&self) -> bool {
        self.shutdown_flag.as_ref().map_or(false, |flag| flag.load(AtomicOrdering::Relaxed))
    }

    /// Clean up temporary files
    fn cleanup_temp_files(&self) -> Result<()> {
        if self.config.verbose {
            println!("🧹 Cleaning up {} temporary files", self.temp_chunks.len());
        }

        for chunk in &self.temp_chunks {
            if chunk.file_path.exists() {
                if let Err(e) = std::fs::remove_file(&chunk.file_path) {
                    eprintln!("⚠️ Failed to remove temp file {}: {}", chunk.file_path.display(), e);
                }
            }
        }

        // Remove temp directory if empty
        if let Err(_) = std::fs::remove_dir(&self.config.temp_directory) {
            // Ignore error - directory might not be empty
        }

        Ok(())
    }

    /// Print final statistics
    fn print_final_stats(&self) {
        println!("\n🎉 External sort deduplication completed! 🎉");
        println!("===============================================");
        println!("📊 Total records processed: {}", self.stats.total_records);
        println!("✨ Unique records preserved: {}", self.stats.unique_records);
        println!("🗑️ Duplicates removed: {} ({:.2}%)", 
            self.stats.duplicates_removed,
            100.0 * self.stats.duplicates_removed as f64 / self.stats.total_records.max(1) as f64);
        println!("📁 Chunks created: {}", self.stats.chunks_created);
        println!("📂 Temp files created: {}", self.stats.temp_files_created);
        println!("💾 Total memory used: {:.1} MB", self.stats.total_memory_used_mb);
        println!("💿 Total disk used: {:.1} MB", self.stats.total_disk_used_mb);
        println!("⏱️ Sort phase: {:.2}s", self.stats.sort_phase_duration_ms as f64 / 1000.0);
        println!("⏱️ Merge phase: {:.2}s", self.stats.merge_phase_duration_ms as f64 / 1000.0);
        println!("⏱️ Total time: {:.2}s", self.stats.total_duration_ms as f64 / 1000.0);
        
        let throughput = self.stats.total_records as f64 / (self.stats.total_duration_ms as f64 / 1000.0);
        println!("🔄 Processing rate: {:.0} records/sec", throughput);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &ExternalSortStats {
        &self.stats
    }

    /// Get temporary chunks information
    pub fn get_temp_chunks(&self) -> &[SortedChunk] {
        &self.temp_chunks
    }

    /// Estimate required disk space for processing
    pub fn estimate_required_disk_space(input_files: &[PathBuf]) -> Result<u64> {
        let mut total_input_size = 0u64;
        
        for file in input_files {
            if file.exists() {
                total_input_size += std::fs::metadata(file)?.len();
            }
        }
        
        // Estimate: input_size * 2 (temp files + output) + 20% overhead
        let estimated_space = (total_input_size as f64 * 2.2) as u64;
        Ok(estimated_space)
    }

    /// Check if external sort is recommended for given input size
    pub fn is_external_sort_recommended(input_files: &[PathBuf]) -> Result<(bool, String)> {
        let memory_info = get_memory_info(); // Returns (total_gb, available_gb) directly
        let available_memory_bytes = (memory_info.1 * 1024.0 * 1024.0) as u64;
        
        let total_input_size = input_files.iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum::<u64>();
        
        // Rule of thumb: use external sort if input is > 60% of available memory
        let memory_threshold = (available_memory_bytes as f64 * 0.6) as u64;
        
        if total_input_size > memory_threshold {
            Ok((true, format!(
                "External sort recommended: input size ({:.1} MB) > {:.0}% of available memory ({:.1} MB)",
                total_input_size as f64 / MB_AS_F64,
                EXTERNAL_SORT_MEMORY_THRESHOLD_FACTOR * PERCENT_100,
                memory_threshold as f64 / MB_AS_F64
            )))
        } else {
            Ok((false, format!(
                "In-memory processing sufficient: input size ({:.1} MB) < {:.0}% of available memory ({:.1} MB)",
                total_input_size as f64 / MB_AS_F64,
                EXTERNAL_SORT_MEMORY_THRESHOLD_FACTOR * PERCENT_100,
                memory_threshold as f64 / MB_AS_F64
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_external_sort_config_default() {
        let config = ExternalSortConfig::default();
        assert!(config.max_memory_usage_bytes >= 1024 * 1024 * 1024); // At least 1GB
        assert!(config.sort_threads >= 4);
        assert!(config.io_buffer_size_bytes > 0);
        assert!(config.cleanup_temp_files);
    }

    #[test]
    fn test_external_sort_deduplicator_creation() {
        let temp_dir = tempdir().unwrap();
        let config = ExternalSortConfig {
            temp_directory: temp_dir.path().to_path_buf(),
            ..ExternalSortConfig::default()
        };
        
        let dedup = ExternalSortDeduplicator::new(config);
        assert!(dedup.is_ok());
        
        let dedup = dedup.unwrap();
        assert_eq!(dedup.temp_chunks.len(), 0);
        assert_eq!(dedup.stats.total_records, 0);
    }

    #[test]
    fn test_record_size_estimation() {
        let temp_dir = tempdir().unwrap();
        let config = ExternalSortConfig {
            temp_directory: temp_dir.path().to_path_buf(),
            ..ExternalSortConfig::default()
        };
        
        let dedup = ExternalSortDeduplicator::new(config).unwrap();
        
        let record = Record::new(
            "test@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            false
        ).unwrap();
        
        let size = dedup.estimate_record_size(&record);
        assert!(size > 50); // Should be reasonable size estimate
        assert!(size < 200); // Should not be excessive
    }

    #[test]
    fn test_disk_space_estimation() {
        let temp_dir = tempdir().unwrap();
        
        // Create a test file
        let test_file = temp_dir.path().join("test.csv");
        std::fs::write(&test_file, "test,data,here\nmore,test,data\n").unwrap();
        
        let estimated = ExternalSortDeduplicator::estimate_required_disk_space(&[test_file]).unwrap();
        assert!(estimated > 0);
        assert!(estimated > 50); // Should account for overhead
    }

    #[test]
    fn test_external_sort_recommendation() {
        let temp_dir = tempdir().unwrap();
        
        // Create a small test file
        let test_file = temp_dir.path().join("small.csv");
        std::fs::write(&test_file, "small,test,file\n").unwrap();
        
        let (recommended, reason) = ExternalSortDeduplicator::is_external_sort_recommended(&[test_file]).unwrap();
        
        // Small file should not require external sort
        assert!(!recommended);
        assert!(reason.contains("In-memory processing sufficient"));
    }

    #[tokio::test]
    async fn test_empty_input_handling() {
        let temp_dir = tempdir().unwrap();
        let config = ExternalSortConfig {
            temp_directory: temp_dir.path().to_path_buf(),
            verbose: false,
            ..ExternalSortConfig::default()
        };
        
        let mut dedup = ExternalSortDeduplicator::new(config).unwrap();
        let output_file = temp_dir.path().join("output.csv");
        
        // Test with empty input
        let result = dedup.deduplicate_large_dataset(&[], &output_file).await;
        assert!(result.is_ok());
        
        let stats = result.unwrap();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.unique_records, 0);
        assert_eq!(stats.chunks_created, 0);
    }
}