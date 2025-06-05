use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tuonella_sift::config::Config;
use tuonella_sift::core::checkpoint::{CheckpointManager, ProcessingState};
use tuonella_sift::core::deduplication::ProcessingStats;
// Note: Using fully qualified paths for shutdown-aware functions
use tuonella_sift::utils::system::format_duration;
use tokio::signal;

#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;

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
            println!("🔗 Resuming from deduplication phase with {} existing temp files", state.temp_files_created.len());
            
            // Skip file processing and go straight to deduplication
            let stats = tuonella_sift::core::deduplication::deduplicate_records_with_shutdown(
                &state.temp_files_created,
                output_path,
                config,
                #[cfg(feature = "cuda")]
                cuda_processor,
                verbose,
                Some(shutdown_flag.clone()),
            )?;
            
            print_completion_stats(stats, start_time, &state, output_path);
            return Ok(());
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
        
        // Proceed to deduplication
        let stats = tuonella_sift::core::deduplication::deduplicate_records_with_shutdown(
            &state.temp_files_created,
            output_path,
            config,
            #[cfg(feature = "cuda")]
            cuda_processor,
            verbose,
            Some(shutdown_flag.clone()),
        )?;
        
        print_completion_stats(stats, start_time, &state, output_path);
        return Ok(());
    }

    println!("📁 Resuming with {} remaining files to process", remaining_files.len());
    println!("♻️ Reusing {} existing temp files", state.temp_files_created.len());

    // Resume from last working file    
    let temp_dir = Path::new(&config.io.temp_directory);
    std::fs::create_dir_all(temp_dir)?;

    let mut new_temp_files = Vec::new();
    let mut updated_state = state.clone();
    
    for (i, file_path) in remaining_files.iter().enumerate() {
        if shutdown_flag.load(Ordering::Relaxed) {
            println!("🛑 Processing interrupted during resume.");
            
            // Save updated checkpoint with current progress
            updated_state.temp_files_created.extend(new_temp_files.clone());
            let _ = checkpoint_manager.save_checkpoint(&updated_state);
            return Ok(());
        }

        // Update processing phase
        updated_state.update_phase(tuonella_sift::core::checkpoint::ProcessingPhase::FileProcessing { 
            file_index: state.current_file_index + i 
        });

        let _single_file_temp_dir = temp_dir.join(format!("single_file_{}", state.current_file_index + i));
        std::fs::create_dir_all(&_single_file_temp_dir)?;

        let single_file_dir = file_path.parent().unwrap_or(Path::new("."));
        let temp_files_for_this_file = tuonella_sift::core::deduplication::process_csv_files_with_validation_and_shutdown(
            single_file_dir,
            config,
            verbose,
            Some(shutdown_flag.clone()),
        )?;

        new_temp_files.extend(temp_files_for_this_file);
        updated_state.mark_file_completed(file_path.clone());

        // Incremental checkpoint save
        if checkpoint_manager.track_records_processed(10000) { // Estimate records per file
            updated_state.temp_files_created.extend(new_temp_files.iter().cloned());
            if let Ok(saved) = checkpoint_manager.auto_save_if_needed(&updated_state) {
                if saved && verbose {
                    println!("💾 Incremental checkpoint saved");
                }
            }
        }

        if verbose {
            println!("📁 Processed remaining file {}/{}: {}",
                     i + 1, remaining_files.len(), file_path.display());
        }
    }

    // Check if we were interrupted again
    if shutdown_flag.load(Ordering::Relaxed) {
        println!("💾 Processing interrupted again. Checkpoint updated.");
        return Ok(());
    }

    let mut all_temp_files = state.temp_files_created.clone();
    all_temp_files.extend(new_temp_files);

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
        println!("� GPU powers activated!");
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

    let temp_files = tuonella_sift::core::deduplication::process_csv_files_with_validation_and_shutdown(
        &args.input,
        &config,
        args.verbose,
        Some(shutdown_flag.clone()),
    )?;

    if temp_files.is_empty() {
        println!("😱 No valid CSV files found.");
        return Ok(());
    }

    println!("📊 Found and processed {} CSV files. Let the judgment begin!", temp_files.len());

    println!("\n⚔️ Commencing the great deduplication cull...");
    let stats = tuonella_sift::core::deduplication::deduplicate_records_with_shutdown(
        &temp_files,
        &output_path,
        &config,
        #[cfg(feature = "cuda")]
        cuda_processor.as_ref(),
        args.verbose,
        Some(shutdown_flag.clone()),
    )?;

    // Check if we were interrupted
    if shutdown_flag.load(Ordering::Relaxed) {
        println!("💾 Saving checkpoint before shutdown...");

        // Create processing state for checkpoint
        let mut processing_state = ProcessingState::new(
            args.input.clone(),
            output_path.clone(),
            args.config.clone(),
            #[cfg(feature = "cuda")]
            cuda_processor.is_some(),
            #[cfg(not(feature = "cuda"))]
            false,
            args.verbose,
            config.memory.memory_usage_percent,
        );

        // Update state with current progress
        let csv_files = tuonella_sift::core::validation::discover_csv_files(&args.input)?;
        processing_state.discovered_files = csv_files;
        processing_state.temp_files_created = temp_files.clone();
        processing_state.update_stats(stats.total_records, stats.unique_records, stats.duplicates_removed, stats.invalid_records);

        // Save checkpoint
        let mut checkpoint_manager_for_save = CheckpointManager::new(
            Path::new(&config.io.temp_directory),
            config.io.checkpoint_auto_save_interval_seconds,
        );

        if let Err(e) = checkpoint_manager_for_save.save_checkpoint(&processing_state) {
            eprintln!("⚠️ Failed to save checkpoint: {}", e);
        } else {
            println!("✅ Checkpoint saved to: {}", checkpoint_manager_for_save.get_checkpoint_path().display());
        }

        println!("🔄 You can resume with: ./tuonella-sift --input {} --output {} --config {} --resume",
                 args.input.display(), output_path.display(), args.config.display());
        return Ok(());
    }

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