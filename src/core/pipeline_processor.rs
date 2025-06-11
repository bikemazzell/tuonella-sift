use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use crate::config::model::Config;
use crate::core::record::Record;
use crate::core::validation::{parse_csv_line, detect_field_positions_with_config};
use crate::constants::{
    PIPELINE_CHANNEL_CAPACITY, MIN_FIELD_COUNT
};

#[derive(Debug, Clone)]
pub struct ProcessingChunk {
    pub file_path: PathBuf,
    pub lines: Vec<String>,
    pub chunk_id: usize,
    pub file_id: usize,
}

#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    pub records: Vec<Record>,
    pub chunk_id: usize,
    pub file_id: usize,
    pub invalid_count: usize,
}

pub struct PipelineProcessor {
    config: Config,
}

impl PipelineProcessor {
    pub fn new(config: Config) -> Self {
        Self {
            config,
        }
    }

    pub async fn process_files_pipeline(
        &self,
        csv_files: &[PathBuf],
        verbose: bool,
        shutdown_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Vec<Record>> {
        if csv_files.is_empty() {
            return Ok(Vec::new());
        }

        let (read_tx, read_rx) = mpsc::channel::<ProcessingChunk>(PIPELINE_CHANNEL_CAPACITY);
        let (process_tx, mut process_rx) = mpsc::channel::<ProcessedChunk>(PIPELINE_CHANNEL_CAPACITY);

        let config_clone = self.config.clone();
        let shutdown_clone = shutdown_flag.clone();
        let csv_files_owned = csv_files.to_vec();

        let reader_task = tokio::spawn(async move {
            Self::reader_task(&csv_files_owned, read_tx, verbose, shutdown_clone).await
        });

        let processor_task = tokio::spawn(async move {
            Self::processor_task(read_rx, process_tx, config_clone, verbose, shutdown_flag).await
        });

        let mut all_records = Vec::new();
        let mut chunks_received = 0;

        while let Some(processed_chunk) = process_rx.recv().await {
            all_records.extend(processed_chunk.records);
            chunks_received += 1;

            if verbose && chunks_received % 100 == 0 {
                println!("📦 Processed {} chunks", chunks_received);
            }
        }

        let _ = tokio::try_join!(reader_task, processor_task)?;

        if verbose {
            println!("🔄 Pipeline processing complete: {} records from {} chunks", 
                     all_records.len(), chunks_received);
        }

        Ok(all_records)
    }

    async fn reader_task(
        csv_files: &[PathBuf],
        read_tx: mpsc::Sender<ProcessingChunk>,
        verbose: bool,
        shutdown_flag: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        for (file_id, file_path) in csv_files.iter().enumerate() {
            if let Some(ref flag) = shutdown_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            if verbose {
                println!("📖 Reading file {}/{}: {}", file_id + 1, csv_files.len(), 
                         file_path.file_name().unwrap_or_default().to_string_lossy());
            }

            let file = File::open(file_path).await?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            let mut chunk_lines = Vec::new();
            let mut chunk_id = 0;

            while let Some(line) = lines.next_line().await? {
                chunk_lines.push(line);

                if chunk_lines.len() >= 10000 {
                    let chunk = ProcessingChunk {
                        file_path: file_path.clone(),
                        lines: std::mem::take(&mut chunk_lines),
                        chunk_id,
                        file_id,
                    };

                    if read_tx.send(chunk).await.is_err() {
                        break;
                    }

                    chunk_id += 1;
                }
            }

            if !chunk_lines.is_empty() {
                let chunk = ProcessingChunk {
                    file_path: file_path.clone(),
                    lines: chunk_lines,
                    chunk_id,
                    file_id,
                };

                let _ = read_tx.send(chunk).await;
            }
        }

        Ok(())
    }

    async fn processor_task(
        mut read_rx: mpsc::Receiver<ProcessingChunk>,
        process_tx: mpsc::Sender<ProcessedChunk>,
        config: Config,
        verbose: bool,
        shutdown_flag: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        while let Some(chunk) = read_rx.recv().await {
            if let Some(ref flag) = shutdown_flag {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let processed = Self::process_chunk(&chunk, &config)?;

            if verbose && processed.records.len() > 1000 {
                println!("⚡ Processed chunk {}: {} records", 
                         processed.chunk_id, processed.records.len());
            }

            if process_tx.send(processed).await.is_err() {
                break;
            }
        }

        Ok(())
    }

    fn process_chunk(chunk: &ProcessingChunk, config: &Config) -> Result<ProcessedChunk> {
        let mut records = Vec::new();
        let mut invalid_count = 0;

        for line in &chunk.lines {
            if line.trim().is_empty() {
                continue;
            }

            let fields = parse_csv_line(line);
            if fields.len() < MIN_FIELD_COUNT {
                invalid_count += 1;
                continue;
            }

            let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(
                &fields,
                config.deduplication.email_username_only,
                config.deduplication.allow_two_field_lines,
            );

            if let Some(record) = Record::new_from_fields(
                fields,
                user_idx,
                password_idx,
                url_idx,
                config.deduplication.case_sensitive_usernames,
            ) {
                records.push(record);
            } else {
                invalid_count += 1;
            }
        }

        Ok(ProcessedChunk {
            records,
            chunk_id: chunk.chunk_id,
            file_id: chunk.file_id,
            invalid_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_pipeline_processor_creation() {
        let config = Config::default();
        let _processor = PipelineProcessor::new(config);
    }

    #[tokio::test]
    async fn test_empty_file_list() -> Result<()> {
        let config = Config::default();
        let processor = PipelineProcessor::new(config);
        
        let result = processor.process_files_pipeline(&[], false, None).await?;
        assert!(result.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_single_file_processing() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.csv");
        
        let mut file = File::create(&file_path).await?;
        file.write_all(b"user@example.com,password123,https://example.com\n").await?;
        file.write_all(b"test@test.com,pass456,https://test.com\n").await?;
        file.sync_all().await?;
        
        let config = Config::default();
        let processor = PipelineProcessor::new(config);
        
        let result = processor.process_files_pipeline(&[file_path], false, None).await?;
        assert_eq!(result.len(), 2);
        
        Ok(())
    }
}
