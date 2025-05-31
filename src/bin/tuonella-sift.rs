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

    println!("\n🧙 Tuonella Sift is awakening...");
    println!("🔍 Input: {}", args.input.display());
    println!("📝 Output: {}", output_path.display());

    #[cfg(feature = "cuda")]
    let cuda_processor = if !args.force_cpu && config.processing.enable_cuda {
        match CudaProcessor::new(config.cuda.clone(), 0) {
            Ok(processor) => {
                println!("🚀 GPU powers activated!");
                Some(processor)
            }
            Err(e) => {
                eprintln!("💥 GPU summoning failed: {}", e);
                println!("🐢 Falling back to CPU processing (it's slower, but honest work)");
                None
            }
        }
    } else {
        if args.force_cpu {
            println!("🧠 CPU mode forced by user. GPUs are taking the day off.");
        } else if !config.processing.enable_cuda {
            println!("⚙️ CPU mode selected in configuration. GPUs remain dormant.");
        }
        None
    };

    if args.verbose {
        println!("\n🔎 Examining data...");
    }

    let temp_files = process_csv_files_with_validation(
        &args.input,
        &config,
        args.verbose,
    )?;

    if temp_files.is_empty() {
        println!("😱 No valid CSV files found.");
        return Ok(());
    }

    println!("📊 Found and processed {} CSV files. Let the judgment begin!", temp_files.len());

    println!("\n⚔️ Commencing the great deduplication cull...");
    let stats = deduplicate_records(
        &temp_files,
        &output_path,
        &config,
        #[cfg(feature = "cuda")]
        cuda_processor.as_ref(),
        args.verbose,
    )?;

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