use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use tokio::task::JoinSet;

use crate::external_sort::{ExternalSortConfig, ExternalSortStats};
use crate::external_sort::checkpoint::{SortCheckpoint, ProcessingPhase, FileProgress};
use crate::external_sort::chunk::ChunkProcessor;
use crate::external_sort::merger::ChunkMerger;
use crate::external_sort::constants::*;

#[cfg(feature = "cuda")]
use crate::cuda::multi_stream_processor::MultiStreamCudaProcessor;

pub struct ExternalSortProcessor {
    config: ExternalSortConfig,
    checkpoint: SortCheckpoint,
    chunk_processor: ChunkProcessor,
    merger: ChunkMerger,
    shutdown_flag: Arc<AtomicBool>,
    chunk_counter: Arc<AtomicUsize>,
    start_time: Instant,

    #[cfg(feature = "cuda")]
    cuda_processor: Option<MultiStreamCudaProcessor>,
}

impl ExternalSortProcessor {
    pub fn new(config: ExternalSortConfig) -> Result<Self> {
        config.validate()?;
        std::fs::create_dir_all(&config.temp_directory)?;

        let chunk_processor = ChunkProcessor::new(
            config.chunk_size_bytes(),
            config.io_buffer_size_bytes(),
            config.temp_directory.clone(),
            config.case_sensitive,
        );

        let merger = ChunkMerger::new(
            config.merge_buffer_size_bytes(),
            config.case_sensitive,
            config.merge_progress_interval_seconds,
        );

        #[cfg(feature = "cuda")]
        let cuda_processor = if config.enable_cuda {
            match Self::create_optimized_cuda_processor(&config) {
                Ok(processor) => Some(processor),
                Err(e) => {
                    if config.verbose {
                        eprintln!("CUDA initialization failed: {}", e);
                        println!("Falling back to CPU processing");
                    }
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            checkpoint: SortCheckpoint::default(),
            chunk_processor,
            merger,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            chunk_counter: Arc::new(AtomicUsize::new(0)),
            start_time: Instant::now(),

            #[cfg(feature = "cuda")]
            cuda_processor,
        })
    }

    pub fn with_checkpoint(mut self, checkpoint: SortCheckpoint) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    pub fn with_shutdown_signal(mut self, shutdown_flag: Arc<AtomicBool>) -> Self {
        self.shutdown_flag = shutdown_flag;
        self
    }

    pub async fn process(
        &mut self,
        input_files: &[PathBuf],
        output_file: &Path,
    ) -> Result<ExternalSortStats> {
        self.checkpoint = SortCheckpoint::new(
            input_files.to_vec(),
            output_file.to_path_buf(),
            self.config.temp_directory.clone(),
        );

        if self.config.verbose {
            println!("🚀 Starting external sort processing");
            println!("📁 Input files: {}", input_files.len());
            println!("💾 Memory limit: {:.1} MB", self.config.memory_limit_bytes() as f64 / BYTES_PER_MB as f64);
            println!("🔧 Processing threads: {}", self.config.processing_threads);
            
            #[cfg(feature = "cuda")]
            if self.cuda_processor.is_some() {
                println!("🚀 CUDA acceleration: enabled");
            } else {
                println!("🧠 CUDA acceleration: disabled");
            }
        }

        self.checkpoint.phase = ProcessingPhase::FileProcessing;
        self.save_checkpoint().await?;

        if self.config.verbose {
            println!("🔄 Starting parallel file processing...");
        }

        let chunks = self.process_files_to_chunks(input_files).await?;

        if self.shutdown_requested() {
            if self.config.verbose {
                println!("🛑 Shutdown requested after file processing. Saving checkpoint...");
            }
            self.checkpoint.created_chunks = chunks;
            self.save_checkpoint().await?;
            return Ok(self.build_stats());
        }

        self.checkpoint.phase = ProcessingPhase::Merging;
        self.checkpoint.created_chunks = chunks;
        self.save_checkpoint().await?;

        self.merge_chunks_to_output(output_file).await?;

        if self.shutdown_requested() {
            if self.config.verbose {
                println!("🛑 Shutdown requested during merge. Saving checkpoint...");
            }
            self.save_checkpoint().await?;
            return Ok(self.build_stats());
        }

        self.checkpoint.phase = ProcessingPhase::Completed;
        self.save_checkpoint().await?;

        if self.config.verbose {
            println!("✅ Processing completed successfully");
        }

        Ok(self.build_stats())
    }

    async fn process_files_to_chunks(&mut self, input_files: &[PathBuf]) -> Result<Vec<crate::external_sort::checkpoint::ChunkMetadata>> {
        let mut all_chunks = Vec::new();
        let mut tasks = JoinSet::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.processing_threads));

        if self.config.verbose {
            println!("📋 Scheduling {} files for parallel processing with {} threads",
                input_files.len(), self.config.processing_threads);
        }

        for (file_index, file_path) in input_files.iter().enumerate() {
            if self.shutdown_requested() {
                if self.config.verbose {
                    println!("🛑 Shutdown requested during file scheduling. Stopping at file {}/{}", file_index, input_files.len());
                }
                break;
            }

            self.checkpoint.current_file_index = file_index;
            self.checkpoint.current_file_progress = FileProgress {
                file_path: file_path.clone(),
                bytes_processed: 0,
                lines_processed: 0,
                records_processed: 0,
                current_chunk_id: all_chunks.len(),
            };

            let permit = semaphore.clone().acquire_owned().await?;
            let chunk_processor = ChunkProcessor::new(
                self.config.chunk_size_bytes(),
                self.config.io_buffer_size_bytes(),
                self.config.temp_directory.clone(),
                self.config.case_sensitive,
            );
            let file_path_clone = file_path.clone();
            let mut checkpoint_clone = self.checkpoint.clone();
            let shutdown_flag = self.shutdown_flag.clone();
            let chunk_counter = self.chunk_counter.clone();
            let verbose = self.config.verbose;

            tasks.spawn(async move {
                let _permit = permit;

                if verbose {
                    let file_name = file_path_clone
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    println!("🔄 Starting file {}: {}", file_index + 1, file_name);
                }

                let result = chunk_processor.process_file_to_chunks_with_counter(&file_path_clone, &mut checkpoint_clone, shutdown_flag, chunk_counter).await;
                (file_index, result, checkpoint_clone)
            });
        }

        if self.config.verbose {
            println!("⚡ All {} files scheduled for processing, waiting for completion...", input_files.len());
            println!("📊 Progress will be reported as files complete...");
        }

        let mut completed_count = 0;
        let mut last_progress_time = std::time::Instant::now();
        let progress_interval = std::time::Duration::from_secs(5); // Show progress every 5 seconds

        loop {
            // Check for shutdown before waiting for next task
            if self.shutdown_requested() {
                if self.config.verbose {
                    println!("🛑 Shutdown requested. Aborting {} remaining tasks...", tasks.len());
                }
                tasks.abort_all();
                break;
            }

            // Show periodic progress updates
            if self.config.verbose && last_progress_time.elapsed() >= progress_interval {
                let active_tasks = tasks.len();
                if active_tasks > 0 {
                    println!("⏳ Processing... {} files completed, {} files in progress",
                        completed_count, active_tasks);
                    use std::io::{self, Write};
                    let _ = io::stdout().flush();
                }
                last_progress_time = std::time::Instant::now();
            }

            // Use timeout to avoid hanging on join_next
            let task_result = tokio::time::timeout(
                tokio::time::Duration::from_millis(200),
                tasks.join_next()
            ).await;

            match task_result {
                Ok(Some(join_result)) => {
                    match join_result {
                        Ok((file_index, chunks_result, _updated_checkpoint)) => {
                            match chunks_result {
                                Ok(chunks) => {
                                    let chunks_added = chunks.len();
                                    all_chunks.extend(chunks);
                                    self.checkpoint.completed_files.push(input_files[file_index].clone());
                                    self.checkpoint.stats.files_processed += 1;
                                    self.checkpoint.stats.chunks_created = all_chunks.len();
                                    completed_count += 1;

                                    if self.config.verbose {
                                        let file_name = input_files[file_index]
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("unknown");
                                        println!("📂 Processed file {}/{}: {} (added {} chunks, total: {})",
                                            completed_count, input_files.len(), file_name, chunks_added, all_chunks.len());
                                        use std::io::{self, Write};
                                        let _ = io::stdout().flush();
                                    }

                                    if self.checkpoint.stats.files_processed % 5 == 0 {
                                        self.save_checkpoint().await?;
                                        if self.config.verbose {
                                            println!("💾 Checkpoint saved at {} files", self.checkpoint.stats.files_processed);
                                            use std::io::{self, Write};
                                            let _ = io::stdout().flush();
                                        }
                                    }
                                }
                                Err(e) => {
                                    if self.config.verbose {
                                        eprintln!("⚠️ Error processing file {}: {}", input_files[file_index].display(), e);
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Task was aborted or panicked - this is expected during shutdown
                            if self.config.verbose && !self.shutdown_requested() {
                                eprintln!("⚠️ Task was aborted or panicked");
                            }
                        }
                    }
                }
                Ok(None) => {
                    // No more tasks
                    break;
                }
                Err(_) => {
                    // Timeout - check shutdown and continue
                    continue;
                }
            }
        }

        // Save final checkpoint before returning
        if !all_chunks.is_empty() {
            self.save_checkpoint().await?;
        }

        if self.shutdown_requested() && self.config.verbose {
            println!("🛑 File processing stopped due to shutdown signal. Processed {}/{} files.",
                self.checkpoint.stats.files_processed, input_files.len());
        }

        Ok(all_chunks)
    }

    async fn merge_chunks_to_output(&mut self, output_file: &Path) -> Result<()> {
        if self.config.verbose {
            println!("🔗 Starting merge of {} chunks", self.checkpoint.created_chunks.len());
        }

        let chunks = self.checkpoint.created_chunks.clone();
        self.merger.validate_chunks(&chunks)?;
        self.merger.merge_chunks(&chunks, output_file, &mut self.checkpoint, self.shutdown_flag.clone()).await?;

        if self.config.verbose {
            println!("✅ Merge completed: {} unique records", self.checkpoint.stats.unique_records);
        }

        Ok(())
    }

    async fn save_checkpoint(&mut self) -> Result<()> {
        self.checkpoint.update_timestamp();
        self.checkpoint.stats.processing_time_ms = self.start_time.elapsed().as_millis() as u64;
        self.checkpoint.save(&self.config.temp_directory)?;
        Ok(())
    }

    fn shutdown_requested(&self) -> bool {
        self.shutdown_flag.load(Ordering::Relaxed)
    }

    fn build_stats(&self) -> ExternalSortStats {
        ExternalSortStats {
            total_records: self.checkpoint.stats.total_records,
            unique_records: self.checkpoint.stats.unique_records,
            duplicates_removed: self.checkpoint.stats.duplicates_removed,
            chunks_created: self.checkpoint.stats.chunks_created,
            files_processed: self.checkpoint.stats.files_processed,
            processing_time_ms: self.checkpoint.stats.processing_time_ms,
            sort_time_ms: 0,
            merge_time_ms: 0,
            disk_usage_mb: self.checkpoint.stats.disk_usage_mb,
            peak_memory_mb: self.checkpoint.stats.peak_memory_mb,
        }
    }

    pub fn cleanup(&self) -> Result<()> {
        self.chunk_processor.cleanup_all_chunks(&self.checkpoint.created_chunks)?;
        
        let checkpoint_file = self.config.temp_directory.join(CHECKPOINT_FILE_NAME);
        if checkpoint_file.exists() {
            std::fs::remove_file(checkpoint_file)?;
        }

        if self.config.temp_directory.exists() {
            if let Err(_) = std::fs::remove_dir(&self.config.temp_directory) {
            }
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn create_optimized_cuda_processor(config: &ExternalSortConfig) -> Result<MultiStreamCudaProcessor> {
        use crate::cuda::processor::CudaProcessor;
        use crate::config::model::{CudaConfig, BatchSizes};

        // First, get hardware properties
        let temp_processor = CudaProcessor::new(CudaConfig::default(), 0)?;
        let props = temp_processor.get_properties()
            .map_err(|e| anyhow::anyhow!("Failed to get GPU properties: {}", e))?;

        if config.verbose {
            println!("🔍 Detected GPU capabilities:");
            println!("   💾 Total Memory: {:.2} GB", props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("   🆓 Free Memory: {:.2} GB", props.free_memory as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("   🧮 Compute Capability: {}.{}", props.compute_capability_major, props.compute_capability_minor);
            println!("   🧵 Max Threads/Block: {}", props.max_threads_per_block);
            println!("   🚌 Memory Bus Width: {} bits", props.memory_bus_width);
        }

        // Calculate optimal configuration based on hardware
        let available_memory = (props.free_memory as f64 * config.cuda_memory_percent / 100.0) as usize;

        // Optimize batch size based on available memory
        let optimal_batch_size = (available_memory / 1000) // Rough estimate: 1KB per record
            .min(config.cuda_batch_size * 5) // Cap at 5x config value
            .max(config.cuda_batch_size); // At least config value

        // Optimize threads per block based on hardware
        let optimal_threads_per_block = if props.max_threads_per_block >= 1024 {
            1024 // Use maximum for modern GPUs
        } else if props.max_threads_per_block >= 512 {
            512
        } else {
            256 // Fallback for older GPUs
        };

        // Calculate optimal stream count based on compute capability and memory
        let optimal_stream_count = if props.compute_capability_major >= 8 {
            // Modern GPUs (Ampere+) can handle more streams
            ((available_memory / (512 * 1024 * 1024)).min(8).max(4)) as usize // 4-8 streams
        } else if props.compute_capability_major >= 7 {
            // Turing/Volta
            6
        } else {
            // Older architectures
            4
        };

        // Create optimized CUDA configuration
        let optimized_cuda_config = CudaConfig {
            gpu_memory_usage_percent: config.cuda_memory_percent as u8,
            estimated_bytes_per_record: 1000, // Conservative estimate
            min_batch_size: optimal_batch_size / 10,
            max_batch_size: optimal_batch_size,
            max_url_buffer_size: 512, // Increased for better performance
            max_username_buffer_size: 128,
            threads_per_block: optimal_threads_per_block as usize,
            batch_sizes: BatchSizes {
                small: optimal_batch_size / 10,
                medium: optimal_batch_size / 4,
                large: optimal_batch_size / 2,
                xlarge: optimal_batch_size,
            },
        };

        if config.verbose {
            println!("🚀 Optimized CUDA configuration:");
            println!("   📦 Batch size: {} (was {})", optimal_batch_size, config.cuda_batch_size);
            println!("   🧵 Threads/block: {} (detected max: {})", optimal_threads_per_block, props.max_threads_per_block);
            println!("   🌊 Stream count: {} (optimized for CC {}.{})", optimal_stream_count, props.compute_capability_major, props.compute_capability_minor);
            println!("   💾 Memory allocation: {:.2} GB", available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        }

        // Create the multi-stream processor with optimized settings
        MultiStreamCudaProcessor::new(optimized_cuda_config, 0, optimal_stream_count)
    }
}
