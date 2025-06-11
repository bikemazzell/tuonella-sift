use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use tuonella_sift::external_sort::{ExternalSortConfig, ExternalSortProcessor};
use tuonella_sift::core::validation::discover_csv_files;

#[derive(Parser)]
#[command(name = "tuonella-sift-external")]
#[command(about = "🧹 Tuonella Sift: External Sort Edition - High-Performance CSV Deduplicator")]
#[command(version)]
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

    #[arg(long, help = "Force CPU processing (disable CUDA)")]
    force_cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("🧙 Tuonella Sift External Sort Edition");
    println!("🔍 Input: {}", args.input.display());
    println!("📝 Output: {}", args.output.display());

    let mut config = if args.config.exists() {
        ExternalSortConfig::from_file(&args.config)?
    } else {
        println!("📄 Config file not found, creating default: {}", args.config.display());
        let default_config = ExternalSortConfig::default();
        default_config.to_file(&args.config)?;
        default_config
    };

    if args.verbose {
        config.verbose = true;
    }

    if args.force_cpu {
        config.enable_cuda = false;
        println!("🧠 CPU mode forced by user");
    }

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n🛑 Shutdown signal received. Gracefully stopping...");
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

    let total_size: u64 = input_files.iter()
        .filter_map(|f| std::fs::metadata(f).ok())
        .map(|m| m.len())
        .sum();

    println!("📊 Total input size: {:.2} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0));

    let memory_limit = config.memory_limit_bytes();
    println!("🧠 Memory limit: {:.2} GB", memory_limit as f64 / (1024.0 * 1024.0 * 1024.0));

    if config.enable_cuda {
        println!("🚀 CUDA acceleration: enabled");
    } else {
        println!("🧠 CUDA acceleration: disabled");
    }

    let start_time = Instant::now();
    
    let mut processor = ExternalSortProcessor::new(config)?
        .with_shutdown_signal(shutdown_flag);

    let stats = processor.process(&input_files, &args.output).await?;
    let total_time = start_time.elapsed();

    processor.cleanup()?;

    println!("\n🎉 Processing completed successfully! 🎉");
    println!("=======================================");
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

    let compression_ratio = if stats.total_records > 0 {
        stats.unique_records as f64 / stats.total_records as f64
    } else {
        1.0
    };
    println!("📈 Compression ratio: {:.2}x", 1.0 / compression_ratio);

    Ok(())
}
