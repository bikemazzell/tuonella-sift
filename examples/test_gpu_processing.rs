use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use anyhow::Result;
use tuonella_sift::config::model::Config;
use tuonella_sift::core::memory_manager::MemoryManager;
use tuonella_sift::core::deduplication::{process_csv_files_with_algorithm_streaming, process_temp_files_with_gpu};

#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;
#[cfg(feature = "cuda")]
use tuonella_sift::config::model::CudaConfig;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing GPU Processing Pipeline (Algorithm Step 2.2)");

    // Create test data
    let test_dir = Path::new("./test_gpu_processing");
    fs::create_dir_all(test_dir)?;

    let input_dir = test_dir.join("input");
    let output_dir = test_dir.join("output");
    fs::create_dir_all(&input_dir)?;
    fs::create_dir_all(&output_dir)?;

    // Create test CSV file
    let test_csv = input_dir.join("test_data.csv");
    create_test_csv(&test_csv)?;

    // Create a simple test configuration
    let config = Config {
        memory: tuonella_sift::config::model::MemoryConfig {
            max_ram_usage_gb: 2,
            auto_detect_memory: true,
        },
        processing: tuonella_sift::config::model::ProcessingConfig {
            enable_cuda: true,
            chunk_size_mb: 64,
            record_chunk_size: 1000,
            max_memory_records: 10000,
        },
        io: tuonella_sift::config::model::IoConfig {
            temp_directory: test_dir.join("temp").to_string_lossy().to_string(),
            output_directory: test_dir.join("output").to_string_lossy().to_string(),
        },
        deduplication: tuonella_sift::config::model::DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
        },
        logging: tuonella_sift::config::model::LoggingConfig {
            verbosity: "normal".to_string(),
        },
        #[cfg(feature = "cuda")]
        cuda: tuonella_sift::config::model::CudaConfig::default(),
    };
    fs::create_dir_all(&config.io.temp_directory)?;

    // Initialize memory manager
    let mut memory_manager = MemoryManager::new(Some(2))?; // 2GB limit for testing

    println!("📊 Memory Manager initialized:");
    let stats = memory_manager.get_memory_stats()?;
    println!("{}", stats.format_summary());

    // Step 1: Process CSV files with algorithm streaming (Steps 1-3)
    println!("\n🔄 Step 1: Processing CSV files with algorithm streaming...");
    let temp_files = process_csv_files_with_algorithm_streaming(
        &input_dir,
        &config,
        &mut memory_manager,
        true
    )?;

    println!("✅ Generated {} temporary files", temp_files.len());
    for temp_file in &temp_files {
        println!("  📁 {}", temp_file.display());
    }

    #[cfg(feature = "cuda")]
    {
        // Step 2: Initialize CUDA processor
        println!("\n🚀 Step 2: Initializing CUDA processor...");
        let cuda_config = config.cuda.clone();

        match CudaProcessor::new(cuda_config, 0) {
            Ok(cuda_processor) => {
                println!("✅ CUDA processor initialized successfully");
                println!("📊 Optimal batch size: {}", cuda_processor.get_optimal_batch_size());

                // Step 3: Process temporary files with GPU (Algorithm Step 2.2)
                println!("\n🚀 Step 3: Processing temporary files with GPU acceleration...");
                let output_file = output_dir.join("deduplicated_gpu.csv");

                let stats = process_temp_files_with_gpu(
                    &temp_files,
                    &output_file,
                    &config,
                    &memory_manager,
                    &cuda_processor,
                    true
                )?;

                println!("✅ GPU processing complete!");
                println!("📈 Processing Statistics:");
                println!("  📊 Total records processed: {}", stats.total_records);
                println!("  ✨ Unique records: {}", stats.unique_records);
                println!("  🗑️ Duplicates removed: {}", stats.duplicates_removed);
                println!("  ⏱️ Processing time: {:.2}s", stats.processing_time_seconds);
                println!("  📁 Files processed: {}", stats.files_processed);
                println!("  ❌ Invalid records: {}", stats.invalid_records);

                println!("\n📄 Output file: {}", output_file.display());

                // Verify output file exists and has content
                if output_file.exists() {
                    let content = fs::read_to_string(&output_file)?;
                    let line_count = content.lines().count();
                    println!("  📏 Output file has {} lines", line_count);

                    if line_count > 1 {
                        println!("  📝 First few lines:");
                        for (i, line) in content.lines().take(5).enumerate() {
                            println!("    {}: {}", i + 1, line);
                        }
                    }
                } else {
                    println!("  ❌ Output file was not created");
                }
            }
            Err(e) => {
                println!("❌ Failed to initialize CUDA processor: {}", e);
                println!("💡 This is expected if CUDA is not available on this system");
                println!("🔄 Falling back to CPU-only processing would happen in real usage");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n⚠️ CUDA feature not enabled. Compile with --features cuda to test GPU processing.");
        println!("💡 This example demonstrates the algorithm structure without GPU acceleration.");
    }

    // Cleanup
    println!("\n🧹 Cleaning up test files...");
    fs::remove_dir_all(test_dir)?;
    println!("✅ Test complete!");

    Ok(())
}

fn create_test_csv(path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Create test data with various formats
    let test_data = vec![
        "user1@example.com,password123,https://example.com/login",
        "user2@test.org,secret456,http://test.org/app",
        "user3@site.net,pass789,android://com.example.app",
        "user1@example.com,password123,https://example.com/login", // Duplicate
        "user4@demo.com,demo123,https://demo.com/",
        "user5@sample.io,sample456,ftp://sample.io/files",
        "user2@test.org,secret456,http://test.org/app", // Duplicate
        "user6@new.com,new789,https://new.com/portal",
        "invalid line without email",
        "user7@valid.com,valid123,https://valid.com/home",
    ];

    let line_count = test_data.len();
    for line in &test_data {
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    println!("📝 Created test CSV with {} lines", line_count);

    Ok(())
}
