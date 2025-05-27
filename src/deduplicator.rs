use anyhow::Result;
use csv::{ReaderBuilder, WriterBuilder};
use futures::stream::{self, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::record::{DeduplicationMap, FieldDetector, Record};
use crate::utils::{discover_csv_files, estimate_remaining_time, format_bytes, format_duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub files_processed: usize,
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub processing_time_seconds: f64,
    pub files_skipped: usize,
    pub errors_encountered: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Checkpoint {
    processed_files: Vec<PathBuf>,
    current_batch: usize,
    stats: ProcessingStats,
    timestamp: u64,
}

pub struct Deduplicator {
    config: Config,
    field_detector: FieldDetector,
    temp_dir: PathBuf,
    output_dir: PathBuf,
    checkpoint_path: PathBuf,
}

impl Deduplicator {
    pub async fn new(config: Config) -> Result<Self> {
        let temp_dir = PathBuf::from(&config.io.temp_directory);
        let output_dir = PathBuf::from(&config.io.output_directory);
        let checkpoint_path = temp_dir.join("checkpoint.json");

        fs::create_dir_all(&temp_dir)?;
        fs::create_dir_all(&output_dir)?;

        Ok(Self {
            config,
            field_detector: FieldDetector::new(),
            temp_dir,
            output_dir,
            checkpoint_path,
        })
    }

    pub async fn process_directory(
        &mut self,
        input_dir: &Path,
        output_dir: &Path,
    ) -> Result<ProcessingStats> {
        info!("Discovering CSV files in: {}", input_dir.display());
        let csv_files = discover_csv_files(input_dir)?;
        
        if csv_files.is_empty() {
            warn!("No CSV files found in directory");
            return Ok(ProcessingStats::default());
        }

        info!("Found {} CSV files", csv_files.len());
        let total_size: u64 = csv_files
            .iter()
            .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
            .sum();
        info!("Total size: {}", format_bytes(total_size));

        let batches = self.create_batches(&csv_files)?;
        let total_batches = batches.len();
        info!("Created {} processing batches", total_batches);

        let start_time = Instant::now();
        let mut stats = ProcessingStats::default();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            info!("Processing batch {}/{}", batch_idx + 1, total_batches);
            
            let batch_stats = self.process_batch(batch, batch_idx).await?;
            stats.merge(batch_stats);

            if self.config.recovery.enable_checkpointing {
                self.save_checkpoint(&stats, batch_idx).await?;
            }
        }

        stats.processing_time_seconds = start_time.elapsed().as_secs_f64();
        
        let final_output = self.merge_batch_outputs(output_dir).await?;
        info!("Final output written to: {}", final_output.display());

        self.cleanup_temp_files().await?;

        Ok(stats)
    }

    pub async fn resume_processing(
        &mut self,
        input_dir: &Path,
        output_dir: &Path,
    ) -> Result<ProcessingStats> {
        if let Ok(checkpoint) = self.load_checkpoint().await {
            info!("Resuming from checkpoint at batch {}", checkpoint.current_batch);
            // Implementation for resume would go here
            // For now, fall back to full processing
        }
        
        self.process_directory(input_dir, output_dir).await
    }

    fn create_batches(&self, files: &[PathBuf]) -> Result<Vec<Vec<PathBuf>>> {
        let max_batch_size = self.config.get_max_memory_bytes() as u64;
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_size = 0u64;

        for file_path in files {
            let file_size = fs::metadata(file_path)?.len();
            
            if file_size > max_batch_size {
                if !current_batch.is_empty() {
                    batches.push(current_batch);
                    current_batch = Vec::new();
                    current_size = 0;
                }
                batches.push(vec![file_path.clone()]);
                continue;
            }

            if current_size + file_size > max_batch_size && !current_batch.is_empty() {
                batches.push(current_batch);
                current_batch = vec![file_path.clone()];
                current_size = file_size;
            } else {
                current_batch.push(file_path.clone());
                current_size += file_size;
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }

    async fn process_batch(
        &self,
        batch_files: Vec<PathBuf>,
        batch_idx: usize,
    ) -> Result<ProcessingStats> {
        let start_time = Instant::now();
        let mut stats = ProcessingStats::default();
        
        let progress_counter = Arc::new(AtomicUsize::new(0));
        let error_counter = Arc::new(AtomicUsize::new(0));
        let dedup_map = Arc::new(Mutex::new(DeduplicationMap::new()));

        let progress_task = self.spawn_progress_reporter(
            progress_counter.clone(),
            batch_files.len(),
            start_time,
        );

        let results: Vec<_> = stream::iter(batch_files.into_iter().enumerate())
            .map(|(file_idx, file_path)| {
                let dedup_map = dedup_map.clone();
                let progress_counter = progress_counter.clone();
                let error_counter = error_counter.clone();
                let config = self.config.clone();
                let field_detector = &self.field_detector;

                async move {
                    let result = self.process_single_file(
                        &file_path,
                        file_idx,
                        &config,
                        field_detector,
                        dedup_map,
                    ).await;

                    progress_counter.fetch_add(1, Ordering::Relaxed);

                    match result {
                        Ok(file_stats) => file_stats,
                        Err(e) => {
                            error!("Error processing {}: {}", file_path.display(), e);
                            error_counter.fetch_add(1, Ordering::Relaxed);
                            ProcessingStats::default()
                        }
                    }
                }
            })
            .buffer_unordered(self.config.processing.max_threads)
            .collect()
            .await;

        progress_task.abort();

        for file_stats in results {
            stats.merge(file_stats);
        }

        stats.errors_encountered = error_counter.load(Ordering::Relaxed);
        stats.processing_time_seconds = start_time.elapsed().as_secs_f64();

        let unique_records = {
            let map = dedup_map.lock();
            map.len()
        };
        stats.unique_records = unique_records;
        stats.duplicates_removed = stats.total_records.saturating_sub(unique_records);

        self.write_batch_output(batch_idx, dedup_map).await?;

        Ok(stats)
    }

    async fn process_single_file(
        &self,
        file_path: &Path,
        _file_idx: usize,
        config: &Config,
        field_detector: &FieldDetector,
        dedup_map: Arc<Mutex<DeduplicationMap>>,
    ) -> Result<ProcessingStats> {
        debug!("Processing file: {}", file_path.display());
        
        let mut stats = ProcessingStats::default();
        stats.files_processed = 1;

        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        
        let delimiter = self.detect_delimiter(&mut reader)?;

        let sample_records = self.sample_records(file_path, delimiter)?;
        let (user_idx, password_idx, url_idx) = field_detector.detect_fields(&sample_records);

        debug!("Detected field positions - user: {}, password: {}, url: {}", 
               user_idx, password_idx, url_idx);

        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .flexible(true)
            .from_reader(reader);

        let mut line_number = 0;
        for result in csv_reader.records() {
            line_number += 1;
            
            match result {
                Ok(record) => {
                    let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                    
                    if let Some(parsed_record) = Record::new(
                        fields,
                        user_idx,
                        password_idx,
                        url_idx,
                        file_path.to_string_lossy().to_string(),
                        line_number,
                        config.deduplication.case_sensitive_usernames,
                    ) {
                        let mut map = dedup_map.lock();
                        map.insert(parsed_record);
                        stats.total_records += 1;
                    }
                }
                Err(e) => {
                    if config.logging.verbosity == "verbose" {
                        warn!("Skipping malformed record at line {} in {}: {}", 
                              line_number, file_path.display(), e);
                    }
                }
            }
        }

        Ok(stats)
    }

    fn detect_delimiter(&self, reader: &mut BufReader<File>) -> Result<u8> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        
        let comma_count = line.matches(',').count();
        let semicolon_count = line.matches(';').count();
        let tab_count = line.matches('\t').count();
        let pipe_count = line.matches('|').count();

        let delimiter = if comma_count >= semicolon_count.max(tab_count).max(pipe_count) {
            b','
        } else if semicolon_count >= tab_count.max(pipe_count) {
            b';'
        } else if tab_count >= pipe_count {
            b'\t'
        } else {
            b'|'
        };

        Ok(delimiter)
    }

    fn sample_records(
        &self,
        file_path: &Path,
        delimiter: u8,
    ) -> Result<Vec<Vec<String>>> {
        use std::io::BufRead;
        
        // First pass: count total lines
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let total_lines = reader.lines().count();
        
        if total_lines == 0 {
            return Ok(Vec::new());
        }

        // Calculate sample size
        let sample_percent = self.config.deduplication.field_detection_sample_percent / 100.0;
        let target_sample_size = ((total_lines as f64) * sample_percent).ceil() as usize;
        let sample_size = target_sample_size
            .max(self.config.deduplication.min_sample_size)
            .min(self.config.deduplication.max_sample_size)
            .min(total_lines);

        // Generate deterministic sample indices distributed across the file
        use std::collections::HashSet;
        let mut sample_indices: HashSet<usize> = HashSet::new();
        
        // Use a simple deterministic sampling based on file size for reproducibility
        let step = if sample_size >= total_lines {
            1
        } else {
            total_lines / sample_size
        };
        
        for i in 0..sample_size {
            let index = (i * step + (total_lines / (sample_size * 2))) % total_lines;
            sample_indices.insert(index);
        }

        // Second pass: collect sampled records
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .flexible(true)
            .from_reader(&mut reader);

        let mut samples = Vec::new();
        for (line_idx, result) in csv_reader.records().enumerate() {
            if sample_indices.contains(&line_idx) {
                if let Ok(record) = result {
                    let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                    if fields.len() >= 3 {
                        samples.push(fields);
                    }
                }
            }
            
            // Early exit if we have enough samples
            if samples.len() >= sample_size {
                break;
            }
        }

        debug!("Sampled {} records from {} total lines ({:.1}% of file)", 
               samples.len(), total_lines, 
               (samples.len() as f64 / total_lines as f64) * 100.0);

        Ok(samples)
    }

    async fn write_batch_output(
        &self,
        batch_idx: usize,
        dedup_map: Arc<Mutex<DeduplicationMap>>,
    ) -> Result<()> {
        let output_path = self.temp_dir.join(format!("batch_{}.csv", batch_idx));
        let records = {
            let map = dedup_map.lock();
            let mut temp_map = DeduplicationMap::new();
            for (key, record) in map.records.iter() {
                temp_map.records.insert(key.clone(), record.clone());
            }
            temp_map.into_records()
        };

        let file = File::create(&output_path)?;
        let mut writer = WriterBuilder::new().from_writer(file);

        for record in records {
            writer.write_record(&record.fields)?;
        }

        writer.flush()?;
        info!("Batch {} output written to: {}", batch_idx, output_path.display());

        Ok(())
    }

    async fn merge_batch_outputs(&self, output_dir: &Path) -> Result<PathBuf> {
        let batch_files: Vec<_> = fs::read_dir(&self.temp_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.file_name()?.to_str()?.starts_with("batch_") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if batch_files.is_empty() {
            anyhow::bail!("No batch files found to merge");
        }

        let output_path = output_dir.join("deduplicated_output.csv");
        let output_file = File::create(&output_path)?;
        let mut writer = WriterBuilder::new().from_writer(output_file);

        for batch_file in batch_files {
            let file = File::open(&batch_file)?;
            let mut reader = ReaderBuilder::new()
                .has_headers(false)
                .from_reader(file);

            for result in reader.records() {
                if let Ok(record) = result {
                    writer.write_record(&record)?;
                }
            }
        }

        writer.flush()?;
        Ok(output_path)
    }

    fn spawn_progress_reporter(
        &self,
        progress_counter: Arc<AtomicUsize>,
        total_files: usize,
        start_time: Instant,
    ) -> tokio::task::JoinHandle<()> {
        let interval_duration = Duration::from_secs(self.config.logging.progress_interval_seconds);
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let processed = progress_counter.load(Ordering::Relaxed);
                let elapsed = start_time.elapsed().as_secs_f64();
                
                if let Some(remaining_time) = estimate_remaining_time(processed, total_files, elapsed) {
                    info!("Progress: {}/{} files ({:.1}%) - ETA: {}", 
                          processed, total_files, 
                          (processed as f64 / total_files as f64) * 100.0,
                          format_duration(remaining_time));
                } else {
                    info!("Progress: {}/{} files ({:.1}%)", 
                          processed, total_files,
                          (processed as f64 / total_files as f64) * 100.0);
                }
            }
        })
    }

    async fn save_checkpoint(&self, stats: &ProcessingStats, batch_idx: usize) -> Result<()> {
        let checkpoint = Checkpoint {
            processed_files: Vec::new(), // Would track processed files in full implementation
            current_batch: batch_idx,
            stats: stats.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        tokio::fs::write(&self.checkpoint_path, checkpoint_json).await?;

        Ok(())
    }

    async fn load_checkpoint(&self) -> Result<Checkpoint> {
        let content = tokio::fs::read_to_string(&self.checkpoint_path).await?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;
        Ok(checkpoint)
    }

    async fn cleanup_temp_files(&self) -> Result<()> {
        if self.temp_dir.exists() {
            for entry in fs::read_dir(&self.temp_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self {
            files_processed: 0,
            total_records: 0,
            unique_records: 0,
            duplicates_removed: 0,
            processing_time_seconds: 0.0,
            files_skipped: 0,
            errors_encountered: 0,
        }
    }
}

impl ProcessingStats {
    fn merge(&mut self, other: ProcessingStats) {
        self.files_processed += other.files_processed;
        self.total_records += other.total_records;
        self.unique_records += other.unique_records;
        self.duplicates_removed += other.duplicates_removed;
        self.files_skipped += other.files_skipped;
        self.errors_encountered += other.errors_encountered;
    }
} 