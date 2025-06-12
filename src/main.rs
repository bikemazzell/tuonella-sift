use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tuonella_sift::external_sort::{ExternalSortConfig, ExternalSortProcessor};

#[derive(Parser)]
#[command(name = "tuonella-sift")]
#[command(about = "High-performance CSV deduplication with external sort and CUDA acceleration")]
struct Args {
    #[arg(short, long, help = "Input directory containing CSV files")]
    input: PathBuf,

    #[arg(short, long, help = "Output file for deduplicated results")]
    output: PathBuf,

    #[arg(short, long, default_value = "config.json", help = "Configuration file")]
    config: PathBuf,

    #[arg(short, long, help = "Resume from checkpoint if available")]
    resume: bool,

    #[arg(short, long, help = "Verbose output")]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Load or create config
    let mut config = if args.config.exists() {
        ExternalSortConfig::from_file(&args.config)?
    } else {
        println!("Config file not found, creating default: {}", args.config.display());
        let default_config = ExternalSortConfig::default();
        default_config.to_file(&args.config)?;
        default_config
    };

    if args.verbose {
        config.verbose = true;
    }

    // Set up shutdown handler
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n🛑 Shutdown signal received. Gracefully stopping...");
        println!("💾 Saving checkpoint and terminating current file processing...");
        shutdown_flag_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    // Validate input directory
    if !args.input.exists() {
        return Err(anyhow::anyhow!("Input directory does not exist: {}", args.input.display()));
    }

    // Discover CSV files
    let input_files = discover_csv_files(&args.input)?;
    if input_files.is_empty() {
        return Err(anyhow::anyhow!("No CSV files found in: {}", args.input.display()));
    }

    println!("🔍 Found {} CSV files to process", input_files.len());

    // Ensure output directory exists
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start_time = std::time::Instant::now();

    // Store temp directory for checkpoint operations
    let temp_directory = config.temp_directory.clone();

    // Create processor with shutdown signal
    let mut processor = ExternalSortProcessor::new(config)?
        .with_shutdown_signal(shutdown_flag.clone());

    // Load checkpoint if resuming
    if args.resume && tuonella_sift::external_sort::checkpoint::SortCheckpoint::exists(&temp_directory) {
        println!("📥 Loading checkpoint from previous run...");
        let checkpoint = tuonella_sift::external_sort::checkpoint::SortCheckpoint::load(&temp_directory)?;
        processor = processor.with_checkpoint(checkpoint);
    }

    // Process files
    let stats = processor.process(&input_files, &args.output).await?;
    let total_time = start_time.elapsed();

    // Check if we were interrupted
    if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
        println!("\n⚠️ Processing was interrupted by user request");
        println!("💾 Checkpoint saved - you can resume with --resume flag");
        println!("📊 Partial results:");
    } else {
        println!("\n🎉 Processing completed successfully! 🎉");
        println!("=======================================");
        
        // Clean up on successful completion
        if let Err(e) = processor.cleanup() {
            println!("⚠️ Warning: Failed to clean up temporary files: {}", e);
        }
    }

    // Print statistics
    println!("⏱️ Total time: {:.2}s", total_time.as_secs_f64());
    println!("✨ Unique records: {}", stats.unique_records);
    println!("🗑️ Duplicates removed: {}", stats.duplicates_removed);

    Ok(())
}

/// Discover CSV files in a directory
fn discover_csv_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut csv_files = Vec::new();
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().to_lowercase() == "csv" {
                    csv_files.push(path);
                }
            }
        }
    }
    
    csv_files.sort();
    Ok(csv_files)
}