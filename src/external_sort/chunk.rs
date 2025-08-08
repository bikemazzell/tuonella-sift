use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use rayon::prelude::*;
use crate::external_sort::record::SortRecord;
use crate::external_sort::checkpoint::{ChunkMetadata, SortCheckpoint};
use crate::constants::*;

pub struct ChunkProcessor {
    chunk_size_bytes: usize,
    io_buffer_size: usize,
    temp_directory: PathBuf,
    case_sensitive: bool,
}

impl ChunkProcessor {
    pub fn new(
        chunk_size_bytes: usize,
        io_buffer_size: usize,
        temp_directory: PathBuf,
        case_sensitive: bool,
    ) -> Self {
        Self {
            chunk_size_bytes,
            io_buffer_size,
            temp_directory,
            case_sensitive,
        }
    }

    pub async fn process_file_to_chunks(
        &self,
        file_path: &Path,
        checkpoint: &mut SortCheckpoint,
    ) -> Result<Vec<ChunkMetadata>> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::with_capacity(self.io_buffer_size, file);
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut chunk_id = checkpoint.created_chunks.len();
        let mut line_count = 0;
        let mut total_records = 0;

        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    line_count += 1;

                    if let Some(record) = SortRecord::from_csv_line(&line) {
                        let record_size = record.estimated_size();

                        if current_size + record_size > self.chunk_size_bytes && !current_chunk.is_empty() {
                            let chunk_metadata = self.sort_and_write_chunk(
                                chunk_id,
                                current_chunk,
                                vec![file_path.to_path_buf()],
                            ).await?;
                            chunks.push(chunk_metadata);
                            chunk_id += 1;
                            current_chunk = Vec::new();
                            current_size = 0;
                        }

                        current_chunk.push(record);
                        current_size += record_size;
                        total_records += 1;
                    }

                    if line_count % PROGRESS_REPORT_INTERVAL_RECORDS == 0 {
                        checkpoint.current_file_progress.lines_processed = line_count;
                        checkpoint.current_file_progress.records_processed = current_chunk.len();
                    }
                }
                Err(_) => {
                    // Try to recover by reading raw bytes and skipping invalid sequences
                    if let Err(_) = self.skip_invalid_line(&mut reader) {
                        break; // If we can't recover, stop processing this file
                    }
                    line_count += 1;
                }
            }
        }

        if !current_chunk.is_empty() {
            let chunk_metadata = self.sort_and_write_chunk(
                chunk_id,
                current_chunk,
                vec![file_path.to_path_buf()],
            ).await?;
            chunks.push(chunk_metadata);
        }

        checkpoint.stats.total_records += total_records;
        Ok(chunks)
    }

    pub async fn process_file_to_chunks_with_counter(
        &self,
        file_path: &Path,
        checkpoint: &mut SortCheckpoint,
        shutdown_flag: Arc<AtomicBool>,
        chunk_counter: Arc<AtomicUsize>,
    ) -> Result<Vec<ChunkMetadata>> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let file = File::open(file_path)?;
        let mut reader = BufReader::with_capacity(self.io_buffer_size, file);
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut line_count = 0;
        let mut total_records = 0;

        let mut line = String::new();

        loop {
            // Check shutdown signal every 100 lines for better responsiveness
            if line_count % 100 == 0 && shutdown_flag.load(Ordering::Relaxed) {
                // Save current chunk if it has data
                if !current_chunk.is_empty() {
                    let chunk_id = chunk_counter.fetch_add(1, Ordering::Relaxed);
                    let chunk_metadata = self.sort_and_write_chunk(
                        chunk_id,
                        std::mem::take(&mut current_chunk),
                        vec![file_path.to_path_buf()],
                    ).await?;
                    chunks.push(chunk_metadata);
                }
                println!("🛑 {} interrupted at line {}", file_name, line_count);
                break;
            }

            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    line_count += 1;

                    if let Some(record) = SortRecord::from_csv_line(&line) {
                        let record_size = record.estimated_size();

                        if current_size + record_size > self.chunk_size_bytes && !current_chunk.is_empty() {
                            let chunk_id = chunk_counter.fetch_add(1, Ordering::Relaxed);
                            let chunk_metadata = self.sort_and_write_chunk(
                                chunk_id,
                                std::mem::take(&mut current_chunk),
                                vec![file_path.to_path_buf()],
                            ).await?;
                            chunks.push(chunk_metadata);
                            current_chunk = Vec::new();
                            current_size = 0;
                        }

                        current_chunk.push(record);
                        current_size += record_size;
                        total_records += 1;
                    }

                    if line_count % PROGRESS_REPORT_INTERVAL_RECORDS == 0 {
                        checkpoint.current_file_progress.lines_processed = line_count;
                        checkpoint.current_file_progress.records_processed = current_chunk.len();
                    }
                }
                Err(_) => {
                    // Try to recover by reading raw bytes and skipping invalid sequences
                    if let Err(_) = self.skip_invalid_line(&mut reader) {
                        break; // If we can't recover, stop processing this file
                    }
                    line_count += 1;
                }
            }
        }

        if !current_chunk.is_empty() {
            let chunk_id = chunk_counter.fetch_add(1, Ordering::Relaxed);
            let chunk_metadata = self.sort_and_write_chunk(
                chunk_id,
                current_chunk,
                vec![file_path.to_path_buf()],
            ).await?;
            chunks.push(chunk_metadata);
        }

        println!("✅ {} completed: {} lines, {} records, {} chunks",
            file_name, line_count, total_records, chunks.len());

        checkpoint.stats.total_records += total_records;
        Ok(chunks)
    }

    pub async fn process_file_to_chunks_with_shutdown(
        &self,
        file_path: &Path,
        checkpoint: &mut SortCheckpoint,
        shutdown_flag: Arc<AtomicBool>,
    ) -> Result<Vec<ChunkMetadata>> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let file = File::open(file_path)?;
        let mut reader = BufReader::with_capacity(self.io_buffer_size, file);
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut chunk_id = checkpoint.created_chunks.len();
        let mut line_count = 0;
        let mut total_records = 0;

        let mut line = String::new();

        loop {
            // Check shutdown signal every 100 lines for better responsiveness
            if line_count % 100 == 0 && shutdown_flag.load(Ordering::Relaxed) {
                // Save current chunk if it has data
                if !current_chunk.is_empty() {
                    let chunk_metadata = self.sort_and_write_chunk(
                        chunk_id,
                        std::mem::take(&mut current_chunk),
                        vec![file_path.to_path_buf()],
                    ).await?;
                    chunks.push(chunk_metadata);
                }
                println!("🛑 {} interrupted at line {}", file_name, line_count);
                break;
            }

            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    line_count += 1;

                    if let Some(record) = SortRecord::from_csv_line(&line) {
                        let record_size = record.estimated_size();

                        if current_size + record_size > self.chunk_size_bytes && !current_chunk.is_empty() {
                            let chunk_metadata = self.sort_and_write_chunk(
                                chunk_id,
                                current_chunk,
                                vec![file_path.to_path_buf()],
                            ).await?;
                            chunks.push(chunk_metadata);
                            chunk_id += 1;
                            current_chunk = Vec::new();
                            current_size = 0;
                        }

                        current_chunk.push(record);
                        current_size += record_size;
                        total_records += 1;
                    }

                    if line_count % PROGRESS_REPORT_INTERVAL_RECORDS == 0 {
                        checkpoint.current_file_progress.lines_processed = line_count;
                        checkpoint.current_file_progress.records_processed = current_chunk.len();
                    }
                }
                Err(_) => {
                    // Try to recover by reading raw bytes and skipping invalid sequences
                    if let Err(_) = self.skip_invalid_line(&mut reader) {
                        break; // If we can't recover, stop processing this file
                    }
                    line_count += 1;
                }
            }
        }

        if !current_chunk.is_empty() {
            let chunk_metadata = self.sort_and_write_chunk(
                chunk_id,
                current_chunk,
                vec![file_path.to_path_buf()],
            ).await?;
            chunks.push(chunk_metadata);
        }

        println!("✅ {} completed: {} lines, {} records, {} chunks",
            file_name, line_count, total_records, chunks.len());

        checkpoint.stats.total_records += total_records;
        Ok(chunks)
    }

    pub async fn sort_and_write_chunk(
        &self,
        chunk_id: usize,
        mut records: Vec<SortRecord>,
        source_files: Vec<PathBuf>,
    ) -> Result<ChunkMetadata> {
        if self.case_sensitive {
            records.par_sort_by(|a, b| a.dedup_key(true).cmp(&b.dedup_key(true)));
        } else {
            records.par_sort_by(|a, b| a.dedup_key(false).cmp(&b.dedup_key(false)));
        }

        let chunk_file = self.temp_directory.join(format!(
            "{}{}{}",
            CHUNK_FILE_PREFIX,
            chunk_id,
            CHUNK_FILE_EXTENSION
        ));

        let file = File::create(&chunk_file)?;
        let mut writer = BufWriter::with_capacity(self.io_buffer_size, file);

        let mut last_key: Option<String> = None;
        let mut records_written = 0;

        for record in records {
            let current_key = record.dedup_key(self.case_sensitive);
            
            if last_key.as_ref() != Some(&current_key) {
                writeln!(writer, "{}", record.to_csv_line())?;
                records_written += 1;
                last_key = Some(current_key);
            }
        }

        writer.flush()?;
        let file_size = std::fs::metadata(&chunk_file)?.len();

        Ok(ChunkMetadata {
            chunk_id,
            file_path: chunk_file,
            record_count: records_written,
            file_size_bytes: file_size,
            is_sorted: true,
            source_files,
        })
    }

    pub fn read_chunk_records(&self, chunk: &ChunkMetadata) -> Result<Vec<SortRecord>> {
        let file = File::open(&chunk.file_path)?;
        let reader = BufReader::with_capacity(self.io_buffer_size, file);
        let mut records = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;
            if let Some(record) = SortRecord::from_csv_line(&line) {
                records.push(record);
            }
        }

        Ok(records)
    }

    pub fn cleanup_chunk(&self, chunk: &ChunkMetadata) -> Result<()> {
        if chunk.file_path.exists() {
            std::fs::remove_file(&chunk.file_path)?;
        }
        Ok(())
    }

    pub fn cleanup_all_chunks(&self, chunks: &[ChunkMetadata]) -> Result<()> {
        for chunk in chunks {
            if let Err(e) = self.cleanup_chunk(chunk) {
                eprintln!("Warning: Failed to cleanup chunk {}: {}", chunk.chunk_id, e);
            }
        }
        Ok(())
    }

    pub fn estimate_chunk_count(&self, file_size: u64) -> usize {
        // Avoid overflow by checking if file_size can be safely cast
        if file_size == 0 {
            return 1;
        }
        
        // Use saturating operations to prevent overflow
        let file_size_usize = file_size.min(usize::MAX as u64) as usize;
        let estimated_chunks = file_size_usize.saturating_add(self.chunk_size_bytes - 1) / self.chunk_size_bytes;
        estimated_chunks.max(1)
    }

    fn skip_invalid_line(&self, reader: &mut BufReader<File>) -> Result<()> {
        use std::io::Read;

        let mut byte_buffer = [0u8; 1];
        loop {
            match reader.read_exact(&mut byte_buffer) {
                Ok(_) => {
                    if byte_buffer[0] == b'\n' {
                        break;
                    }
                }
                Err(_) => break, // EOF or other error
            }
        }
        Ok(())
    }
}
