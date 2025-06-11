use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tuonella_sift::external_sort::{ExternalSortConfig, ExternalSortProcessor};

#[derive(Parser)]
#[command(name = "external-sort")]
#[command(about = "High-performance external sort with CUDA acceleration")]
struct Args {
    #[arg(short, long, help = "Input directory containing CSV files")]
    input: PathBuf,

    #[arg(short, long, help = "Output file for deduplicated results")]
    output: PathBuf,

    #[arg(short, long, default_value = "external_sort_config.json", help = "Configuration file")]
    config: PathBuf,

    #[arg(short, long, help = "Resume from checkpoint if available")]
    resume: bool,

    #[arg(short, long, help = "Verbose output")]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

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

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n🛑 Shutdown signal received. Gracefully stopping...");
        println!("💾 Saving checkpoint and terminating current file processing...");
        shutdown_flag_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    if !args.input.exists() {
        return Err(anyhow::anyhow!("Input directory does not exist: {}", args.input.display()));
    }

    let input_files = discover_csv_files(&args.input)?;
    if input_files.is_empty() {
        return Err(anyhow::anyhow!("No CSV files found in: {}", args.input.display()));
    }

    println!("🔍 Found {} CSV files to process", input_files.len());

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start_time = std::time::Instant::now();

    // Create processor with shutdown signal
    let mut processor = ExternalSortProcessor::new(config)?
        .with_shutdown_signal(shutdown_flag.clone());

    let stats = processor.process(&input_files, &args.output).await?;
    let total_time = start_time.elapsed();

    // Check if we were interrupted
    if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
        println!("\n⚠️ Processing was interrupted by user request");
        println!("💾 Checkpoint saved - you can resume with the same command");
        println!("📊 Partial results:");
    } else {
        println!("\n🎉 Processing completed successfully! 🎉");
        println!("=======================================");
    }

    println!("📊 Total records: {}", stats.total_records);
    println!("✨ Unique records: {}", stats.unique_records);
    println!("🗑️ Duplicates removed: {} ({:.2}%)", 
        stats.duplicates_removed,
        100.0 * stats.duplicates_removed as f64 / stats.total_records.max(1) as f64
    );
    println!("📁 Files processed: {}", stats.files_processed);
    println!("📦 Chunks created: {}", stats.chunks_created);
    println!("💾 Peak memory: {:.1} MB", stats.peak_memory_mb);
    println!("💿 Disk usage: {:.1} MB", stats.disk_usage_mb);
    println!("⏱️ Total time: {:.2}s", total_time.as_secs_f64());
    
    let throughput = stats.total_records as f64 / total_time.as_secs_f64();
    println!("🔄 Throughput: {:.0} records/sec", throughput);

    Ok(())
}

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
