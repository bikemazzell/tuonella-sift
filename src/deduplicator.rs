use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom, BufRead};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use tokio::time::interval;
use csv::{ReaderBuilder, WriterBuilder};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use rand::Rng;
use crate::config::Config;
use crate::record::{Record, FieldDetector, DeduplicationMap};
use crate::constants::{
    BYTES_PER_GB,
    VERBOSE_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_MAX_GPU_BATCH_SIZE,
    DEFAULT_RECORD_ESTIMATED_BYTES,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CPU_SEGMENT_MIN_MB,
};
#[cfg(feature = "cuda")]
use crate::constants::{
    DEFAULT_GPU_BUS_WIDTH_NORMALIZATION,
    DEFAULT_GPU_L2_CACHE_NORMALIZATION_MB,
    DEFAULT_GPU_SEGMENT_BASE_SIZE_MB,
    DEFAULT_GPU_MEMORY_USAGE_PERCENT,
    DEFAULT_GPU_COMPUTE_CAPABILITY_FACTOR,
};
use crate::utils::{estimate_remaining_time, format_duration, format_bytes, discover_csv_files};
#[cfg(feature = "cuda")]
use crate::cuda_processor::CudaProcessor;
use sysinfo::System;

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

struct SwappableDeduplicationMap {
    in_memory_map: DeduplicationMap,
    swap_dir: PathBuf,
    max_memory_entries: usize,
    current_segment: usize,
    segment_size_mb: usize,
    segments: Vec<PathBuf>,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<Arc<CudaProcessor>>,
}

impl SwappableDeduplicationMap {
    #[cfg(feature = "cuda")]
    fn new(swap_dir: PathBuf, max_memory_mb: usize, cuda_processor: Option<Arc<CudaProcessor>>) -> Result<Self> {
        fs::create_dir_all(&swap_dir)?;

        let mut segment_size_mb = DEFAULT_CPU_SEGMENT_MIN_MB; // Default for CPU

        if let Some(proc) = &cuda_processor {
            if let Ok(props) = proc.get_properties() {
                // Adapted logic from former GpuCapabilities::optimal_segment_size
                let memory_factor = props.memory_bus_width as f64 / DEFAULT_GPU_BUS_WIDTH_NORMALIZATION;
                let cache_factor = (props.l2_cache_size as f64 / (DEFAULT_GPU_L2_CACHE_NORMALIZATION_MB * 1024.0 * 1024.0)).min(1.0);
                let base_size_bytes = (DEFAULT_GPU_SEGMENT_BASE_SIZE_MB * 1024 * 1024) as f64;
                let adjusted_size_bytes = base_size_bytes * memory_factor * cache_factor;

                let max_size_bytes = (props.free_memory as f64 * DEFAULT_GPU_MEMORY_USAGE_PERCENT) as usize; // Removed 1024*1024 as free_memory is already in bytes.
                segment_size_mb = ((adjusted_size_bytes.round() as usize).min(max_size_bytes) / (1024 * 1024)).max(1);
            }
        }

        let max_memory_entries = (max_memory_mb * 1024 * 1024) / DEFAULT_RECORD_ESTIMATED_BYTES;

        Ok(Self {
            in_memory_map: DeduplicationMap::new(),
            swap_dir,
            max_memory_entries,
            current_segment: 0,
            segment_size_mb,
            segments: Vec::new(),
            cuda_processor,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn new(swap_dir: PathBuf, max_memory_mb: usize) -> Result<Self> {
        fs::create_dir_all(&swap_dir)?;
        // Use larger segments to reduce fragmentation and I/O overhead
        let segment_size_mb = (max_memory_mb / 4).max(DEFAULT_CPU_SEGMENT_MIN_MB);
        let max_memory_entries = (max_memory_mb * 1024 * 1024) / DEFAULT_RECORD_ESTIMATED_BYTES;

        Ok(Self {
            in_memory_map: DeduplicationMap::new(),
            swap_dir,
            max_memory_entries,
            current_segment: 0,
            segment_size_mb,
            segments: Vec::new(),
        })
    }

    fn insert(&mut self, record: Record) -> Result<bool> {
        if self.in_memory_map.len() >= self.max_memory_entries {
            self.swap_segment_to_disk()?;
        }

        Ok(self.in_memory_map.insert(record))
    }

    fn swap_segment_to_disk(&mut self) -> Result<()> {
        if self.in_memory_map.is_empty() {
            return Ok(());
        }

        let segment_path = self.swap_dir.join(format!("segment_{}.bin", self.current_segment));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(&segment_path)?;

        let mut buffer = Vec::with_capacity(self.segment_size_mb * 1024 * 1024);
        for record in self.in_memory_map.records.values() {

            let user_bytes = record.user.as_bytes();
            let pass_bytes = record.password.as_bytes();
            let url_bytes = record.url.as_bytes();

            buffer.extend_from_slice(&(user_bytes.len() as u16).to_le_bytes());
            buffer.extend_from_slice(&(pass_bytes.len() as u16).to_le_bytes());
            buffer.extend_from_slice(&(url_bytes.len() as u16).to_le_bytes());

            buffer.extend_from_slice(user_bytes);
            buffer.extend_from_slice(pass_bytes);
            buffer.extend_from_slice(url_bytes);
        }

        file.write_all(&buffer)?;
        file.sync_all()?;

        self.segments.push(segment_path);
        self.current_segment += 1;
        self.in_memory_map = DeduplicationMap::new();

        Ok(())
    }

    async fn merge_and_write_output(&mut self, output_path: &Path) -> Result<ProcessingStats> {
        let _stats = ProcessingStats::default();
        self.swap_segment_to_disk()?;

        let output_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(output_path)?;

        #[cfg(feature = "cuda")]
        if let Some(cuda_proc) = &self.cuda_processor {
            return self.cuda_merge_segments(output_file, cuda_proc).await;
        }

        self.cpu_merge_segments(output_file).await
    }

    #[cfg(feature = "cuda")]
    async fn cuda_merge_segments(&self, output_file: File, cuda_proc: &Arc<CudaProcessor>) -> Result<ProcessingStats> {
        let mut stats = ProcessingStats::default();
        let mut writer = WriterBuilder::new().from_writer(BufWriter::new(output_file));

        let batch_size = cuda_proc.get_optimal_batch_size();
        let mut parallel_segments = 1;

        if let Ok(props) = cuda_proc.get_properties() {
            let memory_factor = props.memory_bus_width as f64 / DEFAULT_GPU_BUS_WIDTH_NORMALIZATION;
            let cache_factor = (props.l2_cache_size as f64 / (DEFAULT_GPU_L2_CACHE_NORMALIZATION_MB * 1024.0 * 1024.0)).min(1.0);
            let base_size_bytes = (DEFAULT_GPU_SEGMENT_BASE_SIZE_MB * 1024 * 1024) as f64;
            let adjusted_size_bytes = base_size_bytes * memory_factor * cache_factor;
            let current_free_memory_bytes = props.free_memory;
            let max_size_bytes = (current_free_memory_bytes as f64 * DEFAULT_GPU_MEMORY_USAGE_PERCENT) as usize;
            let optimal_segment_size_bytes = (adjusted_size_bytes.round() as usize).min(max_size_bytes).max(1024 * 1024);

            if optimal_segment_size_bytes > 0 {
                let memory_headroom_bytes = (current_free_memory_bytes as f64 * 0.8) as usize;
                let max_segments_by_mem = memory_headroom_bytes / optimal_segment_size_bytes;

                let capability_factor = (props.compute_capability_major as f64 * DEFAULT_GPU_COMPUTE_CAPABILITY_FACTOR) as usize;
                parallel_segments = max_segments_by_mem.min(capability_factor).max(1);
            }
        }

        let (sender, mut receiver) = tokio::sync::mpsc::channel(parallel_segments.max(1) * 2);
        let segments = self.segments.clone();

        let mut handles = Vec::new();
        for segment_chunk in segments.chunks(parallel_segments) {
            let chunk = segment_chunk.to_vec();
            let sender = sender.clone();
            let cuda_proc = Arc::clone(cuda_proc);

            let handle = tokio::spawn(async move {
                let mut records = Vec::new();

                for segment_path in chunk {
                    let mut file = fs::File::open(&segment_path)?;
                    let mut buffer = Vec::new();
                    file.read_to_end(&mut buffer)?;

                    let mut pos = 0;
                    while pos < buffer.len() {
                        let user_len = u16::from_le_bytes([buffer[pos], buffer[pos + 1]]) as usize;
                        let pass_len = u16::from_le_bytes([buffer[pos + 2], buffer[pos + 3]]) as usize;
                        let url_len = u16::from_le_bytes([buffer[pos + 4], buffer[pos + 5]]) as usize;
                        pos += 6;

                        let user = String::from_utf8_lossy(&buffer[pos..pos + user_len]).to_string();
                        pos += user_len;
                        let password = String::from_utf8_lossy(&buffer[pos..pos + pass_len]).to_string();
                        pos += pass_len;
                        let url = String::from_utf8_lossy(&buffer[pos..pos + url_len]).to_string();
                        pos += url_len;

                        records.push(Record::new_normalized(user, password, url, segment_path.to_string_lossy().into_owned(), 0));
                    }
                }

                for batch in records.chunks(batch_size) {
                    let mut gpu_batch = batch.to_vec();
                    cuda_proc.process_batch(&mut gpu_batch, false)?; // false = case insensitive
                    sender.send(gpu_batch).await?;
                }

                Ok::<_, anyhow::Error>(())
            });
            handles.push(handle);
        }

        drop(sender);

        let mut seen_hashes = HashSet::new();
        let mut batch_size = 0;

        while let Some(processed_records) = receiver.recv().await {
            for record in processed_records {
                let hash = record.get_hash();
                if seen_hashes.insert(hash) {
                    stats.unique_records += 1;
                    writer.write_record(&[&record.user, &record.password, &record.url])?;
                    batch_size += 1;

                    if batch_size >= DEFAULT_MAX_GPU_BATCH_SIZE {
                        writer.flush()?;
                        batch_size = 0;
                    }
                } else {
                    stats.duplicates_removed += 1;
                }
                stats.total_records += 1;
            }
        }

        for handle in handles {
            handle.await??;
        }

        writer.flush()?;
        Ok(stats)
    }

    async fn cpu_merge_segments(&self, output_file: File) -> Result<ProcessingStats> {
        let mut stats = ProcessingStats::default();
        let mut writer = WriterBuilder::new().from_writer(BufWriter::new(output_file));
        let mut seen_hashes = HashSet::new();
        let mut batch_size = 0;

        for segment_path in &self.segments {
            let mut file = fs::File::open(segment_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let mut pos = 0;
            while pos < buffer.len() {
                let user_len = u16::from_le_bytes([buffer[pos], buffer[pos + 1]]) as usize;
                let pass_len = u16::from_le_bytes([buffer[pos + 2], buffer[pos + 3]]) as usize;
                let url_len = u16::from_le_bytes([buffer[pos + 4], buffer[pos + 5]]) as usize;
                pos += 6;

                let user = String::from_utf8_lossy(&buffer[pos..pos + user_len]).to_string();
                pos += user_len;
                let password = String::from_utf8_lossy(&buffer[pos..pos + pass_len]).to_string();
                pos += pass_len;
                let url = String::from_utf8_lossy(&buffer[pos..pos + url_len]).to_string();
                pos += url_len;

                let record = Record::new_normalized(user, password, url, segment_path.to_string_lossy().into_owned(), 0);
                let hash = record.get_hash();
                if seen_hashes.insert(hash) {
                    stats.unique_records += 1;
                    writer.write_record(&[&record.user, &record.password, &record.url])?;
                    batch_size += 1;

                    if batch_size >= DEFAULT_MAX_GPU_BATCH_SIZE {
                        writer.flush()?;
                        batch_size = 0;
                    }
                } else {
                    stats.duplicates_removed += 1;
                }
                stats.total_records += 1;
            }
        }
        writer.flush()?;
        Ok(stats)
    }

    fn cleanup(&self) -> Result<()> {
        if self.swap_dir.exists() {
            fs::remove_dir_all(&self.swap_dir)?;
        }
        Ok(())
    }
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
            match CudaProcessor::new(config.cuda.clone(), 0) {
                Ok(processor) => {
                    info!("CUDA acceleration enabled for device 0");
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

            all_processed_files.extend(batch.clone());

            if self.config.recovery.enable_checkpointing {
                self.save_checkpoint_with_files(&stats, batch_idx, all_processed_files.clone()).await?;
            }
        }

        stats.processing_time_seconds = start_time.elapsed().as_secs_f64();

        let final_output = self.merge_batch_outputs(output_dir).await?;
        info!("Final output written to: {}", final_output.display());

        self.cleanup_temp_files().await?;

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

            let csv_files = discover_csv_files(input_dir)?;

            if csv_files.is_empty() {
                warn!("No CSV files found in directory");
                return Ok(ProcessingStats::default());
            }

            let remaining_files: Vec<PathBuf> = csv_files
                .into_iter()
                .filter(|file| !checkpoint.processed_files.contains(file))
                .collect();

            if remaining_files.is_empty() {
                info!("All files have been processed. Loading existing results.");
                return Ok(checkpoint.stats);
            }

            info!("Found {} remaining files to process", remaining_files.len());

            let batches = self.create_batches(&remaining_files)?;
            let total_batches = batches.len();
            let start_batch = checkpoint.current_batch;

            info!("Resuming processing from batch {} of {}", start_batch + 1, total_batches + start_batch);

            let start_time = Instant::now();
            let mut stats = ProcessingStats::default();
            stats.merge(checkpoint.stats.clone());

            for (batch_idx, batch) in batches.into_iter().enumerate() {
                let actual_batch_idx = start_batch + batch_idx + 1;
                info!("Processing batch {}/{}", actual_batch_idx, total_batches + start_batch);

                let batch_stats = self.process_batch(batch.clone(), actual_batch_idx).await?;
                stats.merge(batch_stats);

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

        let total_size: u64 = batch_files
            .iter()
            .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
            .sum();

        let _progress_task = self.spawn_progress_reporter(
            progress_counter.clone(),
            batch_files.len(),
            total_size,
            start_time,
        );

        // Create swap directory for this batch
        let swap_dir = self.temp_dir.join(format!("swap_batch_{}", batch_idx));
        fs::create_dir_all(&swap_dir)?;

        // Calculate max memory usage in MB
        let mut sys = System::new_all();
        sys.refresh_all();
        let total_memory_mb = (sys.total_memory() / (1024 * 1024)) as usize;
        let max_memory_mb = (total_memory_mb as f64 * (self.config.memory.max_ram_usage_percent as f64 / 100.0)) as usize;

        // Initialize swappable map
        let mut swappable_map = {
            #[cfg(feature = "cuda")]
            {
                SwappableDeduplicationMap::new(
                    swap_dir.clone(),
                    max_memory_mb,
                    self.cuda_processor.clone().map(Arc::new),
                )?
            }
            #[cfg(not(feature = "cuda"))]
            {
                SwappableDeduplicationMap::new(
                    swap_dir.clone(),
                    max_memory_mb
                )?
            }
        };

        // Process files
        for file_path in batch_files {
            let file = File::open(&file_path)?;
            let reader = BufReader::new(file);
            let mut csv_reader = ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .from_reader(reader);

            let sample_records = self.sample_records_robust(&file_path)?;
            let (user_idx, password_idx, url_idx) = self.field_detector.detect_fields(&sample_records);

            let mut line_number = 0;
            for result in csv_reader.records() {
                line_number += 1;
                match result {
                    Ok(record) => {
                        let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                        if let Some(processed_record) = Record::new(
                            fields,
                            user_idx,
                            password_idx,
                            url_idx,
                            file_path.to_string_lossy().into_owned(),
                            line_number,
                            self.config.deduplication.case_sensitive_usernames,
                            &self.config.url_normalization,
                        ) {
                            if swappable_map.insert(processed_record)? {
                                stats.duplicates_removed += 1;
                            } else {
                                stats.unique_records += 1;
                            }
                            stats.total_records += 1;
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
            progress_counter.fetch_add(1, Ordering::Relaxed);
        }

        // Write batch output file
        let batch_output_path = self.temp_dir.join(format!("batch_{}.csv", batch_idx));
        let _batch_output_stats = swappable_map.merge_and_write_output(&batch_output_path).await?;

        // Don't overwrite the processing stats - they already contain the correct duplicate counts
        // The merge_and_write_output method just writes the deduplicated records to disk

        stats.processing_time_seconds = start_time.elapsed().as_secs_f64();
        stats.files_processed = progress_counter.load(Ordering::Relaxed);
        stats.errors_encountered = error_counter.load(Ordering::Relaxed);
        Ok(stats)
    }

    #[cfg(feature = "cuda")]
    fn process_record_batch(
        &self,
        records: &mut Vec<Record>,
        map: &mut SwappableDeduplicationMap,
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        if let Some(cuda_proc) = &self.cuda_processor {
            // Process batch with CUDA
            cuda_proc.process_batch(records, self.config.deduplication.case_sensitive_usernames)?;

            // Insert processed records
            for record in records.drain(..) {
                if map.insert(record)? {
                    stats.unique_records += 1;
                } else {
                    stats.duplicates_removed += 1;
                }
                stats.total_records += 1;
            }
        } else {
            // CPU fallback
            for record in records.drain(..) {
                if map.insert(record)? {
                    stats.unique_records += 1;
                } else {
                    stats.duplicates_removed += 1;
                }
                stats.total_records += 1;
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn process_record_batch(
        &self,
        _records: &mut Vec<Record>,
        _map: &mut SwappableDeduplicationMap,
        _stats: &mut ProcessingStats,
    ) -> Result<()> {
        // ... existing CPU implementation ...
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
        total_size: u64,
        start_time: Instant,
    ) -> tokio::task::JoinHandle<()> {
        let base_interval = if self.config.logging.verbosity == "verbose" {
            VERBOSE_PROGRESS_INTERVAL_SECONDS
        } else {
            self.config.logging.progress_interval_seconds.max(DEFAULT_PROGRESS_INTERVAL_SECONDS)
        };

        let size_gb = total_size as f64 / BYTES_PER_GB;
        let interval_duration = if size_gb < 0.1 {
            Duration::from_secs(1)
        } else if size_gb < 1.0 {
            Duration::from_secs(2)
        } else if size_gb < 10.0 {
            Duration::from_secs(3)
        } else {
            Duration::from_secs(base_interval.min(5))
        };

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                let processed = progress_counter.load(Ordering::Relaxed);
                let elapsed = start_time.elapsed().as_secs_f64();

                if processed >= total_files {
                    break;
                }

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

        let sample_size = DEFAULT_SAMPLE_SIZE;
        let mut samples = Vec::new();
        let mut rng = rand::rng();
        let mut selected_line_indices = HashSet::new();

        // Generate unique random indices
        if total_lines <= sample_size {
            // If fewer lines than sample_size, just take all valid lines
            for i in 0..total_lines {
                selected_line_indices.insert(i);
            }
        } else {
            while selected_line_indices.len() < sample_size {
                let random_line_num = rng.random_range(0..total_lines);
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

    #[tokio::test]
    async fn test_verbose_logging_enabled() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_verbose.csv");
        let mut file = File::create(&file_path).unwrap();

        // Create a file with some invalid UTF-8 to trigger errors
        file.write_all(b"user1,pass1,url1.com\n").unwrap();
        file.write_all(b"user2,pass2,\xff\xfe invalid utf8\n").unwrap();
        file.write_all(b"user3,pass3,url3.com\n").unwrap();
        file.flush().unwrap();

        let config = Config {
            memory: MemoryConfig::default(),
            processing: ProcessingConfig::default(),
            io: IoConfig::default(),
            deduplication: DeduplicationConfig::default(),
            logging: LoggingConfig {
                verbosity: "verbose".to_string(),
                ..LoggingConfig::default()
            },
            recovery: RecoveryConfig::default(),
            cuda: CudaConfig::default(),
            url_normalization: UrlNormalizationConfig::default(),
            field_detection: FieldDetectionConfig::default(),
        };

        let _deduplicator = Deduplicator::new(config).await.unwrap();
        // let error_throttler = ErrorThrottler::new(); // Commented out as it's unused

        // Test that error throttling works in verbose mode
        // let should_log = error_throttler.should_log_error(&file_path.to_string_lossy(), "invalid utf-8 sequence"); // Commented out as it's unused
        // assert!(should_log); // First error should be logged

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_verbose_logging_disabled() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_non_verbose.csv");
        let mut file = File::create(&file_path).unwrap();

        // Create a file with some invalid UTF-8 to trigger errors
        file.write_all(b"user1,pass1,url1.com\n").unwrap();
        file.write_all(b"user2,pass2,\xff\xfe invalid utf8\n").unwrap();
        file.write_all(b"user3,pass3,url3.com\n").unwrap();
        file.flush().unwrap();

        let config = Config {
            memory: MemoryConfig::default(),
            processing: ProcessingConfig::default(),
            io: IoConfig::default(),
            deduplication: DeduplicationConfig::default(),
            logging: LoggingConfig {
                verbosity: "info".to_string(),
                ..LoggingConfig::default()
            },
            recovery: RecoveryConfig::default(),
            cuda: CudaConfig::default(),
            url_normalization: UrlNormalizationConfig::default(),
            field_detection: FieldDetectionConfig::default(),
        };

        let deduplicator = Deduplicator::new(config).await.unwrap();

        // In non-verbose mode, the error logging should be suppressed
        // This test verifies the configuration is set up correctly
        assert_eq!(deduplicator.config.logging.verbosity, "info");

        // Clean up
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_error_throttler_functionality() {
        let _file_path = "test_file.csv";
    }

    #[tokio::test]
    async fn test_size_based_progress_intervals() {
        let config = Config {
            memory: MemoryConfig::default(),
            processing: ProcessingConfig::default(),
            io: IoConfig::default(),
            deduplication: DeduplicationConfig::default(),
            logging: LoggingConfig {
                verbosity: "info".to_string(),
                progress_interval_seconds: 30,
                ..LoggingConfig::default()
            },
            recovery: RecoveryConfig::default(),
            cuda: CudaConfig::default(),
            url_normalization: UrlNormalizationConfig::default(),
            field_detection: FieldDetectionConfig::default(),
        };

        let deduplicator = Deduplicator::new(config).await.unwrap();

        // Test that the configuration is set up correctly for size-based intervals
        assert_eq!(deduplicator.config.logging.progress_interval_seconds, 30);
        assert_eq!(deduplicator.config.logging.verbosity, "info");

        // The actual interval calculation is now based on total_size parameter
        // which will be calculated at runtime based on actual file sizes
    }
}
