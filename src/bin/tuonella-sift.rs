use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tuonella_sift::config::Config;
use tuonella_sift::core::checkpoint::{CheckpointManager, ProcessingState, ProcessingPhase};
use tuonella_sift::core::checkpoint_handler::CheckpointHandler;
use tuonella_sift::core::deduplication::{ProcessingStats, process_single_csv_with_checkpoint};
use tuonella_sift::core::memory_manager::MemoryManager;
// Note: Using fully qualified paths for shutdown-aware functions
use tuonella_sift::utils::system::format_duration;
use tokio::signal;

#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;

/// Comprehensively clean up the entire temp directory on successful completion
/// This ensures all temporary files, including any orphaned files, are removed
/// Only cleans up if processing is truly complete to prevent data loss
fn cleanup_temp_directory(temp_dir: &str, verbose: bool) {
    let temp_path = Path::new(temp_dir);
    
    if !temp_path.exists() {
        return; // Nothing to clean up
    }
    
    // Additional safety check: look for checkpoint file
    let checkpoint_path = temp_path.join("checkpoint.json");
    if checkpoint_path.exists() {
        if verbose {
            println!("⚠️ Checkpoint file still exists - skipping cleanup to preserve resume capability");
        }
        return;
    }
    
    if verbose {
        println!("🧽 Performing comprehensive temp directory cleanup...");
    }
    
    // Remove all contents of the temp directory
    match std::fs::read_dir(temp_path) {
        Ok(entries) => {
            let mut removed_count = 0;
            let mut failed_count = 0;
            
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    
                    let result = if path.is_dir() {
                        std::fs::remove_dir_all(&path)
                    } else {
                        std::fs::remove_file(&path)
                    };
                    
                    match result {
                        Ok(_) => {
                            removed_count += 1;
                            if verbose {
                                println!("   ✅ Removed: {}", path.display());
                            }
                        }
                        Err(e) => {
                            failed_count += 1;
                            if verbose {
                                println!("   ⚠️ Failed to remove {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
            
            if verbose {
                println!("🧽 Cleanup summary: {} items removed, {} failures", removed_count, failed_count);
            }
        }
        Err(e) => {
            if verbose {
                println!("⚠️ Failed to read temp directory {}: {}", temp_dir, e);
            }
        }
    }
}

fn print_completion_stats(stats: ProcessingStats, start_time: Instant, state: &ProcessingState, output_path: &Path) {
    let elapsed = start_time.elapsed();
    let total_elapsed_secs = state.get_elapsed_time() + elapsed.as_secs();
    let processing_rate = stats.total_records as f64 / elapsed.as_secs_f64().max(f64::EPSILON);

    println!("\n🎉 Resume completed successfully! 🎉");
    println!("=======================================");
    println!("📊 Total records: {}", stats.total_records);
    println!("✨ Unique records preserved: {}", stats.unique_records);
    println!("🗑️ Duplicates banished: {} ({:.2}%)",
             stats.duplicates_removed,
             100.0 * stats.duplicates_removed as f64 / stats.total_records.max(1) as f64);
    println!("⚠️ Invalid records: {}", stats.invalid_records);
    println!("⏱️ Resume session time: {}", format_duration(elapsed));
    println!("⏱️ Total processing time: {} seconds", total_elapsed_secs);
    println!("🔄 Processing rate: {:.2} rec/sec", processing_rate);
    println!("📜 Output written to: {}", output_path.display());
}

async fn resume_processing_from_checkpoint(
    state: ProcessingState,
    config: &Config,
    output_path: &Path,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<&tuonella_sift::cuda::processor::CudaProcessor>,
    verbose: bool,
    shutdown_flag: Arc<AtomicBool>,
    mut checkpoint_manager: CheckpointManager,
) -> Result<()> {
    println!("🔄 Resuming processing from checkpoint...");
    
    // Enhanced resume implementation with proper file skipping and temp file reuse
    println!("📊 Checkpoint info:");
    println!("   • Processing phase: {:?}", state.processing_phase);
    println!("   • Files completed: {}/{}", state.completed_files.len(), state.discovered_files.len());
    println!("   • Temp files available: {}", state.temp_files_created.len());
    println!("   • Current file position: {}% in file {}", 
        if state.current_file_total_lines > 0 { 
            (state.current_file_lines_processed * 100) / state.current_file_total_lines 
        } else { 0 },
        state.current_file_index
    );

    let start_time = Instant::now();

    // Use the enhanced logic to get remaining files (excludes completed ones)
    let remaining_files = checkpoint_manager.get_remaining_files(&state);

    // Handle different processing phases
    match state.processing_phase {
        tuonella_sift::core::checkpoint::ProcessingPhase::Completed => {
            println!("✅ Processing was already completed in the previous session!");
            
            // Clean up checkpoint since we're done
            if let Err(e) = checkpoint_manager.cleanup_checkpoint() {
                if verbose {
                    println!("⚠️ Failed to clean up checkpoint: {}", e);
                }
            }
            
            println!("📜 Output should be available at: {}", output_path.display());
            return Ok(());
        }
        tuonella_sift::core::checkpoint::ProcessingPhase::Deduplication => {
            // Verify that file processing is actually complete before proceeding with deduplication
            let remaining_files = checkpoint_manager.get_remaining_files(&state);
            
            if !remaining_files.is_empty() {
                println!("⚠️ Deduplication phase detected, but {} files still need processing", remaining_files.len());
                println!("🔄 Continuing file processing before deduplication...");
                
                // Continue with file processing (fall through to the file processing logic below)
            } else {
                println!("🔗 Resuming from deduplication phase with {} existing temp files", state.temp_files_created.len());
                
                // Create checkpoint handler for resume
                let resume_checkpoint_handler = Arc::new(CheckpointHandler::new(
                    Path::new(&config.io.temp_directory),
                    config.io.checkpoint_auto_save_interval_seconds,
                    state.clone(),
                    Some(shutdown_flag.clone()),
                    verbose,
                ));
                
                // Skip file processing and go straight to deduplication with resumption support
                let stats = tuonella_sift::core::deduplication::deduplicate_records_with_resumption(
                    &state.temp_files_created,
                    output_path,
                    config,
                    #[cfg(feature = "cuda")]
                    cuda_processor,
                    verbose,
                    Some(shutdown_flag.clone()),
                    Some(resume_checkpoint_handler),
                )?;
                
                print_completion_stats(stats, start_time, &state, output_path);
                
                // Comprehensive cleanup of entire temp directory on successful completion
                cleanup_temp_directory(&config.io.temp_directory, verbose);
                
                return Ok(());
            }
        }
        _ => {
            // Continue with file processing
        }
    }

    if remaining_files.is_empty() {
        println!("✅ All files were already processed! Moving to deduplication phase.");
        
        // Update phase and save checkpoint
        let mut updated_state = state.clone();
        updated_state.update_phase(tuonella_sift::core::checkpoint::ProcessingPhase::Deduplication);
        let _ = checkpoint_manager.save_checkpoint(&updated_state);
        
        // Create checkpoint handler for deduplication
        let dedup_checkpoint_handler = Arc::new(CheckpointHandler::new(
            Path::new(&config.io.temp_directory),
            config.io.checkpoint_auto_save_interval_seconds,
            updated_state.clone(),
            Some(shutdown_flag.clone()),
            verbose,
        ));
        
        // Proceed to deduplication with resumption support
        let stats = tuonella_sift::core::deduplication::deduplicate_records_with_resumption(
            &state.temp_files_created,
            output_path,
            config,
            #[cfg(feature = "cuda")]
            cuda_processor,
            verbose,
            Some(shutdown_flag.clone()),
            Some(dedup_checkpoint_handler),
        )?;
        
        print_completion_stats(stats, start_time, &state, output_path);
        
        // Comprehensive cleanup of entire temp directory on successful completion
        cleanup_temp_directory(&config.io.temp_directory, verbose);
        
        return Ok(());
    }

    println!("📁 Resuming with {} remaining files to process", remaining_files.len());
    println!("♻️ Reusing {} existing temp files", state.temp_files_created.len());

    // Resume from last working file    
    let temp_dir = Path::new(&config.io.temp_directory);
    std::fs::create_dir_all(temp_dir)?;

    let mut new_temp_files = Vec::new();
    let updated_state = state.clone(); // We start with the loaded state
    
    // The CheckpointHandler will manage state updates during the resume.
    let resume_checkpoint_handler = Arc::new(CheckpointHandler::new(
        temp_dir,
        config.io.checkpoint_auto_save_interval_seconds,
        updated_state.clone(),
        Some(shutdown_flag.clone()),
        verbose,
    ));

    // Create a MemoryManager and a shared error writer for single file processing.
    let mut memory_manager = MemoryManager::from_config(config)?;
    let error_log_path = temp_dir.join("validation_errors_resume.log");
    let error_file = std::fs::File::create(&error_log_path)?;
    let mut error_writer = std::io::BufWriter::new(error_file);

    for (i, file_path) in remaining_files.iter().enumerate() {
        if shutdown_flag.load(Ordering::Relaxed) {
            println!("🛑 Processing interrupted during resume.");
            // The CheckpointHandler will save the latest state on interrupt.
            return Ok(());
        }

        if verbose {
            println!("📁 Resuming processing for file {}/{}: {}", i + 1, remaining_files.len(), file_path.display());
        }

        // Update the phase in the handler's state to the current file being processed.
        if let Some(original_index) = state.discovered_files.iter().position(|p| p == file_path) {
             resume_checkpoint_handler.set_phase(
                ProcessingPhase::FileProcessing {
                    file_index: original_index
                }
            )?;
        }

        // Define a unique temp file for this pass using consistent naming.
        let temp_file_for_this_pass = temp_dir.join(format!("temp_{}.csv", state.temp_files_created.len() + i));

        // Call the public, single-file processing function.
        if let Err(e) = process_single_csv_with_checkpoint(
            file_path,
            &temp_file_for_this_pass,
            &mut error_writer,
            &mut memory_manager,
            config,
            verbose,
            Some(shutdown_flag.clone()),
            Some(resume_checkpoint_handler.clone()),
        ) {
             eprintln!("❌ Error processing file {} during resume: {}", file_path.display(), e);
             // Decide if we should stop or continue
             continue;
        }

        // If a temp file was created, add it to our list for the final merge.
        if temp_file_for_this_pass.exists() && temp_file_for_this_pass.metadata()?.len() > 0 {
            new_temp_files.push(temp_file_for_this_pass);
        }

        // The handler internally marks the file as complete and saves the checkpoint.
        resume_checkpoint_handler.mark_file_completed(file_path.clone())?;
        
        if verbose {
            println!("✅ Completed processing for remaining file: {}", file_path.display());
        }
    }

    // After the loop, the handler has the updated state.

    // Check if we were interrupted again
    if shutdown_flag.load(Ordering::Relaxed) {
        println!("💾 Processing interrupted after file loop. Checkpoint updated.");
        return Ok(());
    }

    // Merge temp files without duplicates
    let mut all_temp_files = state.temp_files_created.clone();
    for temp_file in new_temp_files {
        if !all_temp_files.contains(&temp_file) {
            all_temp_files.push(temp_file);
        }
    }

    println!("🔗 Merging {} temp files (including {} from previous session)...",
             all_temp_files.len(), state.temp_files_created.len());

    let stats = tuonella_sift::core::deduplication::deduplicate_records_with_shutdown(
        &all_temp_files,
        output_path,
        config,
        #[cfg(feature = "cuda")]
        cuda_processor,
        verbose,
        Some(shutdown_flag.clone()),
    )?;

    if verbose {
        println!("🧹 Cleaning up temporary files...");
    }
    for temp_file in &all_temp_files {
        if let Err(e) = std::fs::remove_file(temp_file) {
            if verbose {
                println!("⚠️ Failed to remove temp file {}: {}", temp_file.display(), e);
            }
        }
    }

    if let Err(e) = checkpoint_manager.cleanup_checkpoint() {
        if verbose {
            println!("⚠️ Failed to clean up checkpoint: {}", e);
        }
    }

    // Comprehensive cleanup of entire temp directory on successful completion
    cleanup_temp_directory(&config.io.temp_directory, verbose);

    let elapsed = start_time.elapsed();
    let total_elapsed_secs = state.get_elapsed_time() + elapsed.as_secs();
    let processing_rate = stats.total_records as f64 / elapsed.as_secs_f64().max(f64::EPSILON);

    println!("\n🎉 Resume completed successfully! 🎉");
    println!("=======================================");
    println!("📊 Total records: {}", stats.total_records);
    println!("✨ Unique records preserved: {}", stats.unique_records);
    println!("🗑️ Duplicates banished: {} ({:.2}%)",
             stats.duplicates_removed,
             100.0 * stats.duplicates_removed as f64 / stats.total_records.max(1) as f64);
    println!("⚠️ Invalid records: {}", stats.invalid_records);
    println!("⏱️ Resume session time: {}", format_duration(elapsed));
    println!("⏱️ Total processing time: {} seconds", total_elapsed_secs);
    println!("🔄 Processing rate: {:.2} rec/sec", processing_rate);
    println!("📜 Output written to: {}", output_path.display());

    Ok(())
}

#[derive(Parser)]
#[command(name = "tuonella-sift")]
#[command(about = "🧹 Tuonella Sift: High-Performance CSV Deduplicator ✨")]
#[command(version)]
struct Args {
    #[arg(short, long, help = "Input directory containing CSV files")]
    input: PathBuf,

    #[arg(short, long, help = "Output file for deduplicated results")]
    output: Option<PathBuf>,

    #[arg(short, long, default_value = "config.json", help = "Configuration file")]
    config: PathBuf,

    #[arg(short, long, help = "Verbose output")]
    verbose: bool,

    #[arg(long, help = "Force CPU processing (disable CUDA)")]
    force_cpu: bool,

    #[arg(long, help = "Resume from previous checkpoint (if available)")]
    resume: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let start_time = Instant::now();

    let config_path = &args.config;
    println!("📚 Loading configuration from {}", config_path.display());
    let config = Config::load(config_path).await?;

    std::fs::create_dir_all(&config.io.temp_directory)?;
    std::fs::create_dir_all(&config.io.output_directory)?;

    let output_path = match &args.output {
        Some(path) => {
            if path.is_dir() {
                path.join("out.csv")
            } else {
                path.to_path_buf()
            }
        }
        None => Path::new(&config.io.output_directory).join("deduplicated_output.csv")
    };

    let checkpoint_manager = CheckpointManager::new(
        Path::new(&config.io.temp_directory),
        config.io.checkpoint_auto_save_interval_seconds,
    );

    // Set up signal handling for graceful shutdown
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    tokio::spawn(async move {
        if let Err(e) = signal::ctrl_c().await {
            eprintln!("Failed to listen for shutdown signal: {}", e);
        } else {
            println!("\n🛑 Shutdown signal received. Saving checkpoint...");
            shutdown_flag_clone.store(true, Ordering::Relaxed);
        }
    });

    #[cfg(feature = "cuda")]
    let cuda_processor = if !args.force_cpu && config.processing.enable_cuda {
        match CudaProcessor::new(config.cuda.clone(), 0) {
            Ok(processor) => {
                if args.verbose {
                    println!("🚀 GPU powers activated!");
                }
                Some(processor)
            }
            Err(e) => {
                if args.verbose {
                    eprintln!("💥 GPU summoning failed: {}", e);
                    println!("🐢 Falling back to CPU processing");
                }
                None
            }
        }
    } else {
        if args.verbose {
            if args.force_cpu {
                println!("🧠 CPU mode forced by user");
            } else if !config.processing.enable_cuda {
                println!("⚙️ CPU mode selected in configuration");
            }
        }
        None
    };
    
    #[cfg(not(feature = "cuda"))]
    let _cuda_processor: Option<()> = None;

    if args.resume {
        if checkpoint_manager.checkpoint_exists() {
            match checkpoint_manager.load_checkpoint() {
                Ok(state) => {
                    if let Err(e) = checkpoint_manager.validate_checkpoint(&state, &args.input, &output_path, &args.config) {
                        eprintln!("❌ Checkpoint validation failed: {}", e);
                        eprintln!("💡 Use --input, --output, and --config with the same values as the previous run");
                        return Err(e);
                    }

                    println!("🔄 Resuming from checkpoint created at {}", state.timestamp);
                    println!("📊 Previous progress: {:.1}% complete", state.calculate_progress());
                    println!("📈 Records processed so far: {}", state.total_records_processed);
                    println!("✨ Unique records found: {}", state.unique_records_count);
                    println!("🗑️ Duplicates removed: {}", state.duplicates_removed);

                    return resume_processing_from_checkpoint(
                        state,
                        &config,
                        &output_path,
                        #[cfg(feature = "cuda")]
                        cuda_processor.as_ref(),
                        args.verbose,
                        shutdown_flag.clone(),
                        checkpoint_manager,
                    ).await;
                }
                Err(e) => {
                    eprintln!("❌ Failed to load checkpoint: {}", e);
                    eprintln!("💡 Starting fresh processing instead");
                }
            }
        } else {
            println!("ℹ️ No checkpoint found. Starting fresh processing.");
        }
    }

    println!("\n🧙 Tuonella Sift is awakening...");
    println!("🔍 Input: {}", args.input.display());
    println!("📝 Output: {}", output_path.display());

    #[cfg(feature = "cuda")]
    if cuda_processor.is_some() {
        println!("🚀 GPU powers activated!");
    } else if args.force_cpu {
        println!("🧠 CPU mode forced by user. GPUs are taking the day off.");
    } else if !config.processing.enable_cuda {
        println!("⚙️ CPU mode selected in configuration. GPUs remain dormant.");
    } else {
        println!("🐢 Falling back to CPU processing (GPU initialization failed)");
    }

    if args.verbose {
        println!("\n🔎 Examining data...");
    }

    // Create initial processing state
    let csv_files = tuonella_sift::core::validation::discover_csv_files(&args.input)?;
    #[cfg(feature = "cuda")]
    let cuda_enabled = cuda_processor.is_some();
    #[cfg(not(feature = "cuda"))]
    let cuda_enabled = false;
    
    let mut processing_state = ProcessingState::new(
        args.input.clone(),
        output_path.clone(),
        args.config.clone(),
        cuda_enabled,
        args.verbose,
        config.memory.memory_usage_percent,
    );
    processing_state.discovered_files = csv_files;

    // Create checkpoint handler
    let checkpoint_handler = Arc::new(CheckpointHandler::new(
        Path::new(&config.io.temp_directory),
        config.io.checkpoint_auto_save_interval_seconds,
        processing_state,
        Some(shutdown_flag.clone()),
        args.verbose,
    ));

    let temp_files = tuonella_sift::core::deduplication::process_csv_files_with_checkpoint(
        &args.input,
        &config,
        args.verbose,
        Some(shutdown_flag.clone()),
        Some(checkpoint_handler.clone()),
    )?;

    if temp_files.is_empty() {
        println!("😱 No valid CSV files found.");
        return Ok(());
    }

    println!("📊 Found and processed {} CSV files. Let the judgment begin!", temp_files.len());

    // Only update checkpoint phase to Deduplication if we actually have processed files
    // and aren't resuming from an interrupted state
    if !temp_files.is_empty() {
        checkpoint_handler.set_phase(ProcessingPhase::Deduplication)?;
        checkpoint_handler.force_save_checkpoint()?;
        
        if args.verbose {
            println!("💾 Checkpoint updated - entering deduplication phase");
        }
    }

    println!("\n⚔️ Commencing the great deduplication cull...");
    
    let stats = tuonella_sift::core::deduplication::deduplicate_records_with_resumption(
        &temp_files,
        &output_path,
        &config,
        #[cfg(feature = "cuda")]
        cuda_processor.as_ref(),
        args.verbose,
        Some(shutdown_flag.clone()),
        Some(checkpoint_handler.clone()),
    )?;

    // Check if we were interrupted
    if shutdown_flag.load(Ordering::Relaxed) {
        // The checkpoint has already been saved by the checkpoint handler during processing
        println!("🔄 You can resume with: ./tuonella-sift --input {} --output {} --config {} --resume",
                 args.input.display(), output_path.display(), args.config.display());
        return Ok(());
    }

    // Update checkpoint handler with deduplication stats
    checkpoint_handler.update_stats(stats.total_records, stats.unique_records, stats.duplicates_removed, stats.invalid_records)?;
    checkpoint_handler.set_phase(ProcessingPhase::Completed)?;

    if args.verbose {
        println!("🧹 Sweeping away temporary files...");
    }
    for temp_file in &temp_files {
        if let Err(e) = std::fs::remove_file(temp_file) {
            if args.verbose {
                println!("⚠️ One escaped the broom: {} - {}", temp_file.display(), e);
            }
        }
    }

    if let Err(e) = checkpoint_manager.cleanup_checkpoint() {
        if args.verbose {
            println!("⚠️ Failed to clean up checkpoint: {}", e);
        }
    }

    // Comprehensive cleanup of entire temp directory on successful completion
    cleanup_temp_directory(&config.io.temp_directory, args.verbose);

    let elapsed = start_time.elapsed();
    let processing_rate = stats.total_records as f64 / elapsed.as_secs_f64().max(f64::EPSILON);

    println!("\n🎉 Deduplication completed successfully! 🎉");
    println!("=======================================");
    println!("📊 Total records: {}", stats.total_records);
    println!("✨ Unique records preserved: {}", stats.unique_records);
    println!("🗑️ Duplicates banished: {} ({:.2}%)",
             stats.duplicates_removed,
             100.0 * stats.duplicates_removed as f64 / stats.total_records.max(1) as f64);
    println!("⚠️ Invalid records (sent to the void): {}", stats.invalid_records);
    println!("⏱️ Processing time: {}", format_duration(elapsed));
    println!("🔄 Processing rate: {:.2} rec/sec", processing_rate);
    println!("📜 Output written to: {}", output_path.display());

    Ok(())
} 