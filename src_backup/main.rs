use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

mod config;
mod constants;
mod deduplicator;
mod patterns;
mod record;
mod utils;

#[cfg(feature = "cuda")]
mod cuda_processor;

use config::Config;
use deduplicator::Deduplicator;

#[derive(Parser)]
#[command(name = "tuonella-sift")]
#[command(about = "Tuonella Sift - High-performance CSV deduplication tool")]
struct Args {
    #[arg(short, long, help = "Input directory containing CSV files")]
    input: PathBuf,

    #[arg(short, long, help = "Output directory for deduplicated files")]
    output: Option<PathBuf>,

    #[arg(short, long, default_value = "cpu.config.json", help = "Configuration file path")]
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

    let (total_memory_gb, available_memory_gb, max_usable_gb) = config.get_memory_info().await?;

    info!("Starting Tuonella Sift deduplication");
    info!("System memory:");
    info!("  - Total: {:.1} GB", total_memory_gb);
    info!("  - Available: {:.1} GB", available_memory_gb);
    info!("  - Max usable ({}%): {:.1} GB", config.memory.max_ram_usage_percent, max_usable_gb);
    info!("");
    info!("Configuration:");
    info!("  Memory:");
    info!("    - Max RAM usage: {}% ({:.1} GB)", config.memory.max_ram_usage_percent, max_usable_gb);
    info!("    - Batch size: {:.1} GB", config.memory.batch_size_gb);
    info!("    - Auto-detect memory: {}", config.memory.auto_detect_memory);
    info!("  Processing:");
    info!("    - CUDA enabled: {}", config.processing.enable_cuda);
    info!("    - Chunk size: {} MB", config.processing.chunk_size_mb);
    info!("    - Max output file size: {} GB", config.processing.max_output_file_size_gb);
    info!("    - Max threads: {}", if config.processing.max_threads == 0 { "auto".to_string() } else { config.processing.max_threads.to_string() });

    info!("Input directory: {}", args.input.display());

    let output_dir = args.output.unwrap_or_else(|| {
        PathBuf::from(&config.io.output_directory)
    });

    info!("Output directory: {}", output_dir.display());

    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input.display());
    }

    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
        info!("Created output directory: {}", output_dir.display());
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn test_args_parsing_basic() {
        // Test basic argument parsing
        let args = Args::try_parse_from(&[
            "tuonella-sift",
            "-i", "/input/path",
            "-o", "/output/path"
        ]).unwrap();

        assert_eq!(args.input, PathBuf::from("/input/path"));
        assert_eq!(args.output, Some(PathBuf::from("/output/path")));
        assert_eq!(args.config, PathBuf::from("cpu.config.json")); // default
        assert!(!args.verbose);
        assert!(!args.resume);
    }

    #[test]
    fn test_args_parsing_with_config() {
        // Test with custom config file
        let args = Args::try_parse_from(&[
            "tuonella-sift",
            "-i", "/input/path",
            "-c", "custom_config.json"
        ]).unwrap();

        assert_eq!(args.input, PathBuf::from("/input/path"));
        assert_eq!(args.config, PathBuf::from("custom_config.json"));
        assert!(args.output.is_none()); // not specified
    }

    #[test]
    fn test_args_parsing_verbose_and_resume() {
        // Test verbose and resume flags
        let args = Args::try_parse_from(&[
            "tuonella-sift",
            "-i", "/input/path",
            "--verbose",
            "--resume"
        ]).unwrap();

        assert_eq!(args.input, PathBuf::from("/input/path"));
        assert!(args.verbose);
        assert!(args.resume);
    }

    #[test]
    fn test_args_parsing_long_form() {
        // Test long-form arguments
        let args = Args::try_parse_from(&[
            "tuonella-sift",
            "--input", "/input/path",
            "--output", "/output/path",
            "--config", "my_config.json",
            "--verbose",
            "--resume"
        ]).unwrap();

        assert_eq!(args.input, PathBuf::from("/input/path"));
        assert_eq!(args.output, Some(PathBuf::from("/output/path")));
        assert_eq!(args.config, PathBuf::from("my_config.json"));
        assert!(args.verbose);
        assert!(args.resume);
    }

    #[test]
    fn test_args_parsing_missing_required() {
        // Test that missing required argument fails
        let result = Args::try_parse_from(&[
            "tuonella-sift",
            "-o", "/output/path"
        ]);

        assert!(result.is_err(), "Should fail when input argument is missing");
    }

    #[test]
    fn test_args_parsing_help() {
        // Test that help can be generated without error
        let result = Args::try_parse_from(&[
            "tuonella-sift",
            "--help"
        ]);

        // Help should cause an error (exit code), but it's a "good" error
        assert!(result.is_err());
    }

    #[test]
    fn test_verbosity_logic() {
        // Test the verbosity logic from main function
        let verbose_args = Args {
            input: PathBuf::from("/input"),
            output: None,
            config: PathBuf::from("cpu.config.json"),
            verbose: true,
            resume: false,
        };

        let non_verbose_args = Args {
            input: PathBuf::from("/input"),
            output: None,
            config: PathBuf::from("cpu.config.json"),
            verbose: false,
            resume: false,
        };

        // Simulate the verbosity logic from main
        let config_verbosity = "normal";

        let verbosity1 = if verbose_args.verbose { "verbose" } else { config_verbosity };
        let verbosity2 = if non_verbose_args.verbose { "verbose" } else { config_verbosity };

        assert_eq!(verbosity1, "verbose");
        assert_eq!(verbosity2, "normal");
    }

    #[test]
    fn test_output_directory_logic() {
        // Test output directory fallback logic
        let args_with_output = Args {
            input: PathBuf::from("/input"),
            output: Some(PathBuf::from("/custom/output")),
            config: PathBuf::from("cpu.config.json"),
            verbose: false,
            resume: false,
        };

        let args_without_output = Args {
            input: PathBuf::from("/input"),
            output: None,
            config: PathBuf::from("cpu.config.json"),
            verbose: false,
            resume: false,
        };

        // Simulate the output directory logic from main
        let config_output_dir = "./output";

        let output_dir1 = args_with_output.output.unwrap_or_else(|| {
            PathBuf::from(config_output_dir)
        });

        let output_dir2 = args_without_output.output.unwrap_or_else(|| {
            PathBuf::from(config_output_dir)
        });

        assert_eq!(output_dir1, PathBuf::from("/custom/output"));
        assert_eq!(output_dir2, PathBuf::from("./output"));
    }
}
