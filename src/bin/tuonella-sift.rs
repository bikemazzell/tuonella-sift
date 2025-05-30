use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tuonella_sift::config::Config;
use tuonella_sift::core::deduplication::{deduplicate_records, process_csv_files_with_validation};
use tuonella_sift::utils::system::format_duration;

#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;

#[derive(Parser)]
#[command(name = "tuonella-sift")]
#[command(about = "GPU accelerated CSV deduplication tool")]
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let start_time = Instant::now();

    // Load configuration
    let config_path = &args.config;
    println!("Loading configuration from {}", config_path.display());
    let config = Config::load(config_path).await?;

    // Setup directories
    std::fs::create_dir_all(&config.io.temp_directory)?;
    std::fs::create_dir_all(&config.io.output_directory)?;

    let output_path = args.output.unwrap_or_else(|| {
        Path::new(&config.io.output_directory).join("deduplicated_output.csv")
    });

    println!("Starting CSV deduplication...");
    println!("Input directory: {}", args.input.display());
    println!("Output file: {}", output_path.display());

    // Initialize CUDA processor if available and not disabled
    #[cfg(feature = "cuda")]
    let cuda_processor = if !args.force_cpu && config.processing.enable_cuda {
        match CudaProcessor::new(config.cuda.clone(), 0) {
            Ok(processor) => {
                println!("CUDA processor initialized successfully");
                Some(processor)
            }
            Err(e) => {
                eprintln!("Failed to initialize CUDA processor: {}", e);
                println!("Falling back to CPU processing");
                None
            }
        }
    } else {
        if args.force_cpu {
            println!("CUDA processing disabled by --force-cpu flag");
        } else if !config.processing.enable_cuda {
            println!("CUDA processing disabled in configuration");
        }
        None
    };

    // Process CSV files
    if args.verbose {
        println!("Pre-processing CSV files with validation...");
    }

    let temp_files = process_csv_files_with_validation(
        &args.input,
        &config,
        args.verbose,
    )?;

    if temp_files.is_empty() {
        println!("No valid CSV files found in input directory");
        return Ok(());
    }

    println!("Processed {} CSV files", temp_files.len());

    // Deduplicate records
    let stats = deduplicate_records(
        &temp_files,
        &output_path,
        &config,
        #[cfg(feature = "cuda")]
        cuda_processor.as_ref(),
        args.verbose,
    )?;

    // Clean up temporary files
    for temp_file in &temp_files {
        if let Err(e) = std::fs::remove_file(temp_file) {
            if args.verbose {
                println!("Warning: Failed to remove temp file {}: {}", temp_file.display(), e);
            }
        }
    }

    // Print summary
    let elapsed = start_time.elapsed();
    let processing_rate = if elapsed.as_secs() > 0 {
        stats.total_records as f64 / elapsed.as_secs_f64()
    } else {
        stats.total_records as f64
    };

    println!("\nDeduplication completed successfully!");
    println!("---------------------------------------");
    println!("Total records processed: {}", stats.total_records);
    println!("Unique records: {}", stats.unique_records);
    println!("Duplicates removed: {} ({:.2}%)",
             stats.duplicates_removed,
             100.0 * stats.duplicates_removed as f64 / stats.total_records.max(1) as f64);
    println!("Invalid records: {}", stats.invalid_records);
    println!("Processing time: {}", format_duration(elapsed));
    println!("Processing rate: {:.2} records/second", processing_rate);
    println!("Output written to: {}", output_path.display());

    Ok(())
} 