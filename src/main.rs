use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

mod config;
mod deduplicator;
mod record;
mod utils;

use config::Config;
use deduplicator::Deduplicator;

#[derive(Parser)]
#[command(name = "tuonella-sift")]
#[command(about = "Tuonella Sift - High-performance CSV deduplication tool that sifts through massive datasets")]
struct Args {
    #[arg(short, long, help = "Input directory containing CSV files")]
    input: PathBuf,

    #[arg(short, long, help = "Output directory for deduplicated files")]
    output: Option<PathBuf>,

    #[arg(short, long, default_value = "config.json", help = "Configuration file path")]
    config: PathBuf,

    #[arg(short, long, help = "Verbose output")]
    verbose: bool,

    #[arg(long, help = "Resume from checkpoint if available")]
    resume: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config = Config::load(&args.config).await?;
    
    let verbosity = if args.verbose { "verbose" } else { &config.logging.verbosity };
    utils::setup_logging(verbosity, &config.logging.log_file)?;

    info!("Starting Tuonella Sift deduplication");
    info!("Input directory: {}", args.input.display());
    
    let output_dir = args.output.unwrap_or_else(|| {
        PathBuf::from(&config.io.output_directory)
    });
    
    info!("Output directory: {}", output_dir.display());

    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input.display());
    }

    let mut deduplicator = Deduplicator::new(config).await?;
    
    let stats = if args.resume {
        deduplicator.resume_processing(&args.input, &output_dir).await?
    } else {
        deduplicator.process_directory(&args.input, &output_dir).await?
    };

    info!("Deduplication completed successfully!");
    info!("Files processed: {}", stats.files_processed);
    info!("Total records: {}", stats.total_records);
    info!("Unique records: {}", stats.unique_records);
    info!("Duplicates removed: {}", stats.duplicates_removed);
    info!("Processing time: {:.2}s", stats.processing_time_seconds);

    Ok(())
} 