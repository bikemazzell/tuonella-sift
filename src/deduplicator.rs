use crate::record::{Record, DeduplicationMap, FieldDetector};
use crate::config::Config;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::Mutex;
use anyhow::Result;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::fs::{self, File};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use csv::{ReaderBuilder, WriterBuilder};
use futures::stream::{self, StreamExt};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use serde::{Serialize, Deserialize};
use rand::Rng;
use std::collections::HashSet;

#[cfg(feature = "cuda")]
use crate::cuda_processor::CudaProcessor;
use crate::constants::{DEFAULT_PROGRESS_INTERVAL_SECONDS, VERBOSE_PROGRESS_INTERVAL_SECONDS};
use crate::utils::{discover_csv_files, estimate_remaining_time, format_bytes, format_duration};
#[cfg(feature = "cuda")]
use crate::patterns::normalize_url_fast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub files_processed: usize,
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub processing_time_seconds: f64,
    pub files_skipped: usize,
    pub errors_encountered: usize,
    pub invalid_records: usize,
    pub error_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    checkpoint_path: PathBuf,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<CudaProcessor>,
}

impl Deduplicator {
    pub async fn new(config: Config) -> Result<Self> {
        let temp_dir = PathBuf::from(&config.io.temp_directory);
        let output_dir = PathBuf::from(&config.io.output_directory);
        let checkpoint_path = temp_dir.join("checkpoint.json");

        fs::create_dir_all(&temp_dir)?;
        fs::create_dir_all(&output_dir)?;

        #[cfg(feature = "cuda")]
        let cuda_processor = if config.processing.enable_cuda {
            match CudaProcessor::new(config.cuda.clone()) { // Removed config.url_normalization.clone()
                Ok(processor) => {
                    info!("CUDA acceleration enabled");
                    Some(processor)
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA processor: {}. Falling back to CPU processing.", e);
                    None
                }
            }
        } else {
            info!("CUDA acceleration disabled in configuration");
            None
        };

        Ok(Self {
            config: config.clone(),
            field_detector: FieldDetector::from_config(&config.field_detection),
            temp_dir,
            checkpoint_path,
            #[cfg(feature = "cuda")]
            cuda_processor,
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
        let mut all_processed_files = Vec::new();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            info!("Processing batch {}/{}", batch_idx + 1, total_batches);
            
            let batch_stats = self.process_batch(batch.clone(), batch_idx).await?;
            stats.merge(batch_stats);
            
            // Track processed files for checkpointing
            all_processed_files.extend(batch.clone());

            if self.config.recovery.enable_checkpointing {
                self.save_checkpoint_with_files(&stats, batch_idx, all_processed_files.clone()).await?;
            }
        }

        stats.processing_time_seconds = start_time.elapsed().as_secs_f64();
        
        let final_output = self.merge_batch_outputs(output_dir).await?;
        info!("Final output written to: {}", final_output.display());

        self.cleanup_temp_files().await?;
        
        // Clean up checkpoint file after successful completion
        if self.checkpoint_path.exists() {
            tokio::fs::remove_file(&self.checkpoint_path).await?;
            info!("Checkpoint file cleaned up after successful completion");
        }

        Ok(stats)
    }

    pub async fn resume_processing(
        &mut self,
        input_dir: &Path,
        output_dir: &Path,
    ) -> Result<ProcessingStats> {
        if let Ok(checkpoint) = self.load_checkpoint().await {
            info!("Resuming from checkpoint at batch {}", checkpoint.current_batch);
            
            // Discover all CSV files
            let csv_files = discover_csv_files(input_dir)?;
            
            if csv_files.is_empty() {
                warn!("No CSV files found in directory");
                return Ok(ProcessingStats::default());
            }

            // Filter out already processed files
            let remaining_files: Vec<PathBuf> = csv_files
                .into_iter()
                .filter(|file| !checkpoint.processed_files.contains(file))
                .collect();

            if remaining_files.is_empty() {
                info!("All files have been processed. Loading existing results.");
                return Ok(checkpoint.stats);
            }

            info!("Found {} remaining files to process", remaining_files.len());
            
            // Process remaining files starting from the checkpoint batch
            let batches = self.create_batches(&remaining_files)?;
            let total_batches = batches.len();
            let start_batch = checkpoint.current_batch;
            
            info!("Resuming processing from batch {} of {}", start_batch + 1, total_batches + start_batch);

            let start_time = Instant::now();
            let mut stats = checkpoint.stats.clone();

            for (batch_idx, batch) in batches.into_iter().enumerate() {
                let actual_batch_idx = start_batch + batch_idx + 1;
                info!("Processing batch {}/{}", actual_batch_idx + 1, total_batches + start_batch);
                
                let batch_stats = self.process_batch(batch.clone(), actual_batch_idx).await?;
                stats.merge(batch_stats);

                // Update checkpoint with processed files
                let mut updated_checkpoint = checkpoint.clone();
                updated_checkpoint.processed_files.extend(batch);
                updated_checkpoint.current_batch = actual_batch_idx;
                updated_checkpoint.stats = stats.clone();
                
                if self.config.recovery.enable_checkpointing {
                    self.save_checkpoint(&updated_checkpoint.stats, actual_batch_idx).await?;
                }
            }

            stats.processing_time_seconds += start_time.elapsed().as_secs_f64();
            
            let final_output = self.merge_batch_outputs(output_dir).await?;
            info!("Final output written to: {}", final_output.display());

            self.cleanup_temp_files().await?;
            
            // Clean up checkpoint file after successful completion
            if self.checkpoint_path.exists() {
                tokio::fs::remove_file(&self.checkpoint_path).await?;
                info!("Checkpoint file cleaned up after successful completion");
            }

            Ok(stats)
        } else {
            info!("No valid checkpoint found, starting fresh processing");
            self.process_directory(input_dir, output_dir).await
        }
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
        stats.files_processed = progress_counter.load(Ordering::Relaxed);

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
        let mut stats = ProcessingStats::default();

        #[cfg(feature = "cuda")]
        if let Some(_) = &self.cuda_processor {
            // Use CUDA processor if available
            let file = File::open(file_path)?;
            let reader = ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .from_reader(BufReader::new(file));

            let sample_records = self.sample_records_robust(file_path)?;
            let (user_idx, password_idx, url_idx) = field_detector.detect_fields(&sample_records);

            return self.process_file_with_cuda(
                reader,
                user_idx,
                password_idx,
                url_idx,
                file_path,
                config,
                dedup_map,
            ).await;
        }

        // Fallback to CPU processing
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        
        let sample_records = self.sample_records_robust(file_path)?;
        let (user_idx, password_idx, url_idx) = field_detector.detect_fields(&sample_records);

        debug!(
            "Detected field indices for {}: user={}, password={}, url={}",
            file_path.display(),
            user_idx,
            password_idx,
            url_idx
        );

        // Process file line by line
        let mut line_num = 0;
        for line_result in reader.lines() {
            line_num += 1;
            
            match line_result {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    let fields = self.parse_csv_line_robust(&line);
                    
                    if fields.len() < 3 {
                        debug!("Skipping line {}:{} (insufficient fields: {})", file_path.display(), line_num, fields.len());
                        continue;
                    }

                    stats.total_records += 1;

                    let max_idx = user_idx.max(password_idx).max(url_idx);
                    if fields.len() <= max_idx {
                        debug!("Skipping line {}:{} (not enough fields for indices)", file_path.display(), line_num);
                        continue;
                    }

                    if let Some(record) = Record::new(
                        fields,
                        user_idx,
                        password_idx,
                        url_idx,
                        file_path.to_string_lossy().into_owned(),
                        line_num,
                        config.deduplication.case_sensitive_usernames,
                        &config.url_normalization,
                    ) {
                        let mut map = dedup_map.lock();
                        if map.insert(record) {
                            stats.unique_records += 1;
                        } else {
                            stats.duplicates_removed += 1;
                        }
                    } else {
                        debug!("Skipping line {}:{} (failed to create record)", file_path.display(), line_num);
                    }
                }
                Err(e) => {
                    warn!("Error reading line from {}:{} - {}", file_path.display(), line_num, e);
                    stats.errors_encountered += 1;
                }
            }
        }

        stats.files_processed = 1;
        debug!(
            "Processed {} total records from {}",
            stats.total_records,
            file_path.display()
        );
        Ok(stats)
    }

    #[cfg(feature = "cuda")]
    async fn process_file_with_cuda(
        &self,
        mut csv_reader: csv::Reader<BufReader<File>>,
        user_idx: usize,
        password_idx: usize,
        url_idx: usize,
        file_path: &Path,
        config: &Config,
        dedup_map: Arc<Mutex<DeduplicationMap>>,
    ) -> Result<ProcessingStats> {
        let mut stats = ProcessingStats::default();
        let mut line_number = 0;
        let mut record_batch = Vec::new();
        
        let cuda_processor = self.cuda_processor.as_ref().unwrap();
        let batch_size = cuda_processor.get_optimal_batch_size();
        
        debug!("Processing file {} with CUDA batch size {}", file_path.display(), batch_size);
        
        for result in csv_reader.records() {
            line_number += 1;
            
            match result {
                Ok(string_record) => {
                    // Apply line filters and create record
                    if let Some(record) = Record::new(
                        string_record.iter().map(|s| s.to_string()).collect(),
                        user_idx,
                        password_idx,
                        url_idx,
                        file_path.display().to_string(),
                        line_number,
                        config.deduplication.case_sensitive_usernames,
                        &config.url_normalization, // Added url_config
                    ) {
                        record_batch.push(record);
                        
                        // Process batch when it reaches optimal size
                        if record_batch.len() >= batch_size {
                            let batch_stats = self.process_cuda_batch(&mut record_batch, config, &dedup_map).await?;
                            stats.merge(batch_stats);
                            record_batch.clear();
                        }
                    } else {
                        stats.invalid_records += 1;
                    }
                }
                Err(e) => {
                    warn!("Error reading line {}: {}", line_number, e);
                    stats.error_records += 1;
                }
            }
        }
        
        // Process remaining records
        if !record_batch.is_empty() {
            let batch_stats = self.process_cuda_batch(&mut record_batch, config, &dedup_map).await?;
            stats.merge(batch_stats);
        }
        
        Ok(stats)
    }

    #[cfg(feature = "cuda")]
    async fn process_cuda_batch(
        &self,
        records: &mut [Record],
        config: &Config,
        dedup_map: &Arc<Mutex<DeduplicationMap>>,
    ) -> Result<ProcessingStats> {
        let mut stats = ProcessingStats::default();
        
        // Process records in CUDA
        if let Some(cuda_processor) = &self.cuda_processor {
            cuda_processor.process_batch(records, config.deduplication.case_sensitive_usernames)?;
        } else {
            // Fallback to CPU processing
            for record in records.iter_mut() {
                // Normalize URL if present
                if !record.url.is_empty() {
                    record.normalized_url = normalize_url_fast(&record.url);
                }
                
                // Normalize username based on case sensitivity setting
                record.normalized_user = if config.deduplication.case_sensitive_usernames {
                    record.user.clone()
                } else {
                    record.user.to_lowercase()
                };
            }
        }
        
        // Update deduplication map
        let mut map = dedup_map.lock();
        for record in records.iter() {
            if map.insert(record.clone()) {
                stats.duplicates_removed += 1;
            } else {
                stats.unique_records += 1;
            }
            stats.total_records += 1;
        }
        
        Ok(stats)
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
            // Write only the essential fields in a consistent format
            let output_record = vec![&record.user, &record.password, &record.url];
            writer.write_record(&output_record)?;
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
                .flexible(true)
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
        let interval_duration = if self.config.logging.verbosity == "verbose" {
            Duration::from_secs(VERBOSE_PROGRESS_INTERVAL_SECONDS)
        } else {
            Duration::from_secs(self.config.logging.progress_interval_seconds.max(DEFAULT_PROGRESS_INTERVAL_SECONDS))
        };
        
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
        // This is a simplified checkpoint - in a full implementation, we'd track more state
        let checkpoint = Checkpoint {
            processed_files: Vec::new(), // Would need to track this properly in process_batch
            current_batch: batch_idx,
            stats: stats.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        tokio::fs::write(&self.checkpoint_path, checkpoint_json).await?;
        
        debug!("Checkpoint saved at batch {}", batch_idx);
        Ok(())
    }

    async fn save_checkpoint_with_files(&self, stats: &ProcessingStats, batch_idx: usize, processed_files: Vec<PathBuf>) -> Result<()> {
        let checkpoint = Checkpoint {
            processed_files,
            current_batch: batch_idx,
            stats: stats.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        tokio::fs::write(&self.checkpoint_path, checkpoint_json).await?;
        
        debug!("Checkpoint saved at batch {} with {} processed files", batch_idx, checkpoint.processed_files.len());
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

    fn sample_records_robust(&self, file_path: &Path) -> Result<Vec<Vec<String>>> {
        let mut file = File::open(file_path)?;
        let reader = BufReader::new(&mut file);

        // 1. Count total lines
        let total_lines = reader.lines().count();
        if total_lines == 0 {
            return Ok(Vec::new());
        }

        let sample_size = 100;
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();
        let mut selected_line_indices = HashSet::new();

        // Generate unique random indices
        if total_lines <= sample_size {
            // If fewer lines than sample_size, just take all valid lines
            for i in 0..total_lines {
                selected_line_indices.insert(i);
            }
        } else {
            while selected_line_indices.len() < sample_size {
                let random_line_num = rng.gen_range(0..total_lines);
                selected_line_indices.insert(random_line_num);
            }
        }

        // 2. Read selected lines
        file.seek(SeekFrom::Start(0))?; // Reset file pointer to beginning
        let reader = BufReader::new(&mut file);
        let mut current_line_num = 0;

        for line_result in reader.lines() {
            if selected_line_indices.contains(&current_line_num) {
                if let Ok(line) = line_result {
                    let line = line.trim();
                    if !line.is_empty() {
                        let fields = self.parse_csv_line_robust(&line);
                        if fields.len() >= 3 {
                            samples.push(fields);
                        }
                    }
                }
            }
            current_line_num += 1;
            if samples.len() == sample_size || current_line_num >= total_lines {
                break;
            }
        }
        
        debug!("Sampled {} records from {} lines in {}", samples.len(), total_lines, file_path.display());
        Ok(samples)
    }

    fn parse_csv_line_robust(&self, line: &str) -> Vec<String> {
        let mut fields = Vec::new();
        let mut current_field = String::new();
        let mut in_quotes = false;
        let mut escaped_quote = false;
        let mut chars = line.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    if escaped_quote {
                        // This is an escaped quote ("""), add a single quote
                        current_field.push('"');
                        escaped_quote = false;
                    } else if in_quotes {
                        // Check if this is an escaped quote
                        if chars.peek() == Some(&'"') {
                            escaped_quote = true;
                            chars.next(); // Skip the next quote
                        } else {
                            // End of quoted section
                            in_quotes = false;
                        }
                    } else {
                        // Start of quoted section
                        in_quotes = true;
                    }
                }
                ',' if !in_quotes => {
                    // Field separator when not in quotes
                    fields.push(current_field.trim().to_string());
                    current_field.clear();
                }
                '\\' if in_quotes => {
                    // Handle escaped characters in quotes
                    if let Some(&next_ch) = chars.peek() {
                        match next_ch {
                            'n' => {
                                current_field.push('\n');
                                chars.next();
                            }
                            'r' => {
                                current_field.push('\r');
                                chars.next();
                            }
                            't' => {
                                current_field.push('\t');
                                chars.next();
                            }
                            _ => current_field.push(ch),
                        }
                    } else {
                        current_field.push(ch);
                    }
                }
                _ => {
                    current_field.push(ch);
                }
            }
        }
        
        // Add the last field
        if !current_field.is_empty() || fields.len() > 0 {
            fields.push(current_field.trim().to_string());
        }
        
        // Handle empty fields
        if fields.len() < 3 {
            // Try to split by comma and preserve empty fields
            let simple_fields: Vec<String> = line.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            
            if simple_fields.len() >= 3 {
                return simple_fields;
            }
        }
        
        // Handle line continuations in URLs
        if fields.len() >= 3 {
            let url = &fields[2];
            if url.ends_with('\\') || url.ends_with('-') || url.ends_with('_') {
                // Remove the continuation character
                fields[2] = url.trim_end_matches(|c| c == '\\' || c == '-' || c == '_').to_string();
            }
        }
        
        fields
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
            invalid_records: 0,
            error_records: 0,
        }
    }
}

impl ProcessingStats {
    pub fn merge(&mut self, other: ProcessingStats) {
        self.files_processed += other.files_processed;
        self.total_records += other.total_records;
        self.unique_records += other.unique_records;
        self.duplicates_removed += other.duplicates_removed;
        self.processing_time_seconds += other.processing_time_seconds;
        self.files_skipped += other.files_skipped;
        self.errors_encountered += other.errors_encountered;
        self.invalid_records += other.invalid_records;
        self.error_records += other.error_records;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;
    use crate::config::{Config, FieldDetectionConfig, UrlNormalizationConfig, DeduplicationConfig, IoConfig, LoggingConfig, ProcessingConfig, CudaConfig, MemoryConfig, RecoveryConfig};

    // Helper function to create a dummy Deduplicator for testing
    async fn create_dummy_deduplicator() -> Deduplicator {
        let config = Config {
            memory: MemoryConfig::default(),
            processing: ProcessingConfig::default(),
            io: IoConfig::default(),
            deduplication: DeduplicationConfig::default(),
            logging: LoggingConfig::default(),
            recovery: RecoveryConfig::default(),
            cuda: CudaConfig::default(),
            url_normalization: UrlNormalizationConfig::default(),
            field_detection: FieldDetectionConfig::default(),
        };
        Deduplicator::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_random_sampling_large_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("large_test.csv");
        let mut file = File::create(&file_path).unwrap();

        // Create a file with more than 100 lines
        let total_lines = 500;
        for i in 0..total_lines {
            writeln!(file, "user{},pass{},url{}.com", i, i, i).unwrap();
        }
        file.flush().unwrap();

        let deduplicator = create_dummy_deduplicator().await;
        let samples = deduplicator.sample_records_robust(&file_path).unwrap();

        // Assert that exactly 100 samples are returned
        assert_eq!(samples.len(), 100);

        // Assert that all sampled records are valid (have 3 fields)
        for record in &samples {
            assert_eq!(record.len(), 3);
        }

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_random_sampling_small_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("small_test.csv");
        let mut file = File::create(&file_path).unwrap();

        // Create a file with fewer than 100 lines
        let total_lines = 50;
        for i in 0..total_lines {
            writeln!(file, "user{},pass{},url{}.com", i, i, i).unwrap();
        }
        file.flush().unwrap();

        let deduplicator = create_dummy_deduplicator().await;
        let samples = deduplicator.sample_records_robust(&file_path).unwrap();

        // Assert that all lines are sampled
        assert_eq!(samples.len(), total_lines);

        // Assert that all sampled records are valid (have 3 fields)
        for record in &samples {
            assert_eq!(record.len(), 3);
        }

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_random_sampling_empty_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty_test.csv");
        let mut file = File::create(&file_path).unwrap();
        file.flush().unwrap();

        let deduplicator = create_dummy_deduplicator().await;
        let samples = deduplicator.sample_records_robust(&file_path).unwrap();

        // Assert that no samples are returned
        assert!(samples.is_empty());

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_random_sampling_invalid_lines() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("invalid_lines_test.csv");
        let mut file = File::create(&file_path).unwrap();

        // Create a file with some invalid lines (fewer than 3 fields, empty)
        writeln!(file, "user1,pass1").unwrap(); // Invalid
        writeln!(file, "").unwrap(); // Empty
        writeln!(file, "user2,pass2,url2.com").unwrap(); // Valid
        writeln!(file, "user3,pass3").unwrap(); // Invalid
        writeln!(file, "user4,pass4,url4.com").unwrap(); // Valid
        file.flush().unwrap();

        let deduplicator = create_dummy_deduplicator().await;
        let samples = deduplicator.sample_records_robust(&file_path).unwrap();

        // Assert that only valid lines are sampled
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0][0], "user2");
        assert_eq!(samples[1][0], "user4");

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }
}
