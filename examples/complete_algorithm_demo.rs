use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use anyhow::Result;
use tuonella_sift::config::model::Config;
use tuonella_sift::core::memory_manager::MemoryManager;
use tuonella_sift::core::deduplication::ProcessingStats;

#[cfg(feature = "cuda")]
use tuonella_sift::core::deduplication::process_with_complete_algorithm;
use tuonella_sift::core::deduplication::process_with_complete_algorithm_cpu_fallback;

#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 Complete Algorithm Pipeline Demonstration");
    println!("📋 This demonstrates the full implementation of docs/algorithm.md");
    println!("🔄 Sections 2.1, 2.2, 2.3, and 3 - Complete Pipeline");

    // Create test environment
    let test_dir = Path::new("./complete_algorithm_demo");
    fs::create_dir_all(test_dir)?;

    let input_dir = test_dir.join("input");
    let output_dir = test_dir.join("output");
    fs::create_dir_all(&input_dir)?;
    fs::create_dir_all(&output_dir)?;

    // Create comprehensive test data
    create_comprehensive_test_data(&input_dir)?;

    // Create configuration
    let config = create_test_config(test_dir)?;

    // Initialize memory manager
    let mut memory_manager = MemoryManager::new(Some(4))?; // 4GB limit for testing

    println!("\n📊 System Resources:");
    let stats = memory_manager.get_memory_stats()?;
    println!("{}", stats.format_summary());

    let output_file = output_dir.join("final_deduplicated.csv");

    #[cfg(feature = "cuda")]
    {
        // Try GPU-accelerated complete algorithm first
        println!("\n🚀 Attempting GPU-Accelerated Complete Algorithm Pipeline...");

        match CudaProcessor::new(config.cuda.clone(), 0) {
            Ok(cuda_processor) => {
                println!("✅ CUDA processor initialized successfully");
                println!("📊 GPU Details:");
                println!("  🎯 Optimal batch size: {}", cuda_processor.get_optimal_batch_size());

                let stats = process_with_complete_algorithm(
                    &input_dir,
                    &output_file,
                    &config,
                    &mut memory_manager,
                    &cuda_processor,
                    true
                )?;

                println!("\n🎉 GPU-Accelerated Complete Algorithm Pipeline SUCCESS!");
                print_final_statistics(&stats);
                verify_output(&output_file)?;
            }
            Err(e) => {
                println!("❌ CUDA initialization failed: {}", e);
                println!("🔄 Falling back to CPU-only complete algorithm...");

                let stats = process_with_complete_algorithm_cpu_fallback(
                    &input_dir,
                    &output_file,
                    &config,
                    &mut memory_manager,
                    true
                )?;

                println!("\n🎉 CPU Complete Algorithm Pipeline SUCCESS!");
                print_final_statistics(&stats);
                verify_output(&output_file)?;
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n🔄 Running CPU-only Complete Algorithm Pipeline...");

        let stats = process_with_complete_algorithm_cpu_fallback(
            &input_dir,
            &output_file,
            &config,
            &mut memory_manager,
            true
        )?;

        println!("\n🎉 CPU Complete Algorithm Pipeline SUCCESS!");
        print_final_statistics(&stats);
        verify_output(&output_file)?;
    }

    // Cleanup
    println!("\n🧹 Cleaning up demo files...");
    fs::remove_dir_all(test_dir)?;
    println!("✅ Demo complete!");

    Ok(())
}

fn create_comprehensive_test_data(input_dir: &Path) -> Result<()> {
    println!("\n📝 Creating comprehensive test dataset...");

    // Create multiple CSV files with different formats and edge cases
    let test_files = vec![
        ("users_batch1.csv", vec![
            "email,password,website",
            "john.doe@example.com,password123,https://example.com/login",
            "jane.smith@test.org,secret456,http://test.org/app",
            "bob.wilson@site.net,pass789,android://com.example.app",
            "john.doe@example.com,password123,https://example.com/login", // Duplicate
            "alice.brown@demo.com,demo123,https://demo.com/",
            "charlie.davis@sample.io,sample456,ftp://sample.io/files",
        ]),
        ("users_batch2.csv", vec![
            "username,pwd,url",
            "jane.smith@test.org,secret456,http://test.org/app", // Duplicate from batch1
            "david.lee@new.com,new789,https://new.com/portal",
            "emma.taylor@valid.com,valid123,https://valid.com/home",
            "frank.miller@unique.org,unique456,android://com.unique.app",
            "invalid line without email,password,url", // Invalid
            "grace.wilson@final.net,final789,https://final.net/dashboard",
        ]),
        ("mixed_format.csv", vec![
            "user|pass|site",
            "henry.clark@mixed.com|mixed123|https://mixed.com",
            "ivy.johnson@format.org|format456|android://com.format.app",
            "jack.brown@test.com|test789|http://test.com/login",
            "", // Empty line
            "kate.davis@last.net|last123|https://last.net/home",
        ]),
    ];

    let mut total_lines = 0;
    for (filename, lines) in test_files {
        let file_path = input_dir.join(filename);
        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file);

        for line in &lines {
            writeln!(writer, "{}", line)?;
            total_lines += 1;
        }
        writer.flush()?;

        println!("  📄 Created {}: {} lines", filename, lines.len());
    }

    println!("✅ Test dataset created: {} total lines across {} files", total_lines, 3);
    Ok(())
}

fn create_test_config(test_dir: &Path) -> Result<Config> {
    let config = Config {
        memory: tuonella_sift::config::model::MemoryConfig {
            max_ram_usage_gb: 4,
            auto_detect_memory: true,
        },
        processing: tuonella_sift::config::model::ProcessingConfig {
            enable_cuda: true,
            chunk_size_mb: 32,
            record_chunk_size: 1000,
            max_memory_records: 50000,
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
    Ok(config)
}

fn print_final_statistics(stats: &ProcessingStats) {
    println!("📊 Complete Algorithm Pipeline Statistics:");
    println!("  📁 Files processed: {}", stats.files_processed);
    println!("  📊 Total records processed: {}", stats.total_records);
    println!("  ✨ Unique records retained: {}", stats.unique_records);
    println!("  🗑️ Duplicates removed: {}", stats.duplicates_removed);
    println!("  ❌ Invalid records skipped: {}", stats.invalid_records);
    println!("  ⏱️ Total processing time: {:.3}s", stats.processing_time_seconds);

    if stats.total_records > 0 {
        let dedup_rate = (stats.duplicates_removed as f64 / stats.total_records as f64) * 100.0;
        let validity_rate = ((stats.total_records - stats.invalid_records) as f64 / stats.total_records as f64) * 100.0;
        println!("  📈 Deduplication rate: {:.1}%", dedup_rate);
        println!("  ✅ Data validity rate: {:.1}%", validity_rate);
    }
}

fn verify_output(output_file: &Path) -> Result<()> {
    println!("\n🔍 Verifying output file...");

    if !output_file.exists() {
        println!("❌ Output file does not exist!");
        return Ok(());
    }

    let content = fs::read_to_string(output_file)?;
    let lines: Vec<&str> = content.lines().collect();

    println!("✅ Output file verification:");
    println!("  📄 File: {}", output_file.display());
    println!("  📏 Total lines: {}", lines.len());

    if lines.len() > 1 {
        println!("  📋 Header: {}", lines[0]);
        println!("  📝 Sample records:");
        for (i, line) in lines.iter().skip(1).take(5).enumerate() {
            println!("    {}: {}", i + 1, line);
        }

        if lines.len() > 6 {
            println!("    ... and {} more records", lines.len() - 6);
        }
    }

    // Verify CSV format
    let header_fields = lines[0].split(',').count();
    println!("  🔧 CSV format: {} columns", header_fields);

    if header_fields == 3 {
        println!("  ✅ Correct CSV format (username,password,normalized_url)");
    } else {
        println!("  ⚠️ Unexpected CSV format");
    }

    Ok(())
}
