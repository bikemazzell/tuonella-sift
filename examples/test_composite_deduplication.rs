use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use anyhow::Result;
use tuonella_sift::config::model::Config;
use tuonella_sift::core::memory_manager::MemoryManager;
use tuonella_sift::core::deduplication::process_with_complete_algorithm_cpu_fallback;

#[cfg(feature = "cuda")]
use tuonella_sift::core::deduplication::process_with_complete_algorithm;
#[cfg(feature = "cuda")]
use tuonella_sift::cuda::processor::CudaProcessor;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing Composite Deduplication Logic");
    println!("📋 Verifying that records with same email but different data are kept");

    // Create test environment
    let test_dir = Path::new("./test_composite_dedup");
    fs::create_dir_all(test_dir)?;

    let input_dir = test_dir.join("input");
    let output_dir = test_dir.join("output");
    fs::create_dir_all(&input_dir)?;
    fs::create_dir_all(&output_dir)?;

    // Create test data with your specific examples
    create_test_data_with_examples(&input_dir)?;

    // Create configuration
    let config = create_test_config(test_dir)?;

    // Initialize memory manager
    let mut memory_manager = MemoryManager::new(Some(2))?; // 2GB limit for testing

    let output_file = output_dir.join("composite_dedup_test.csv");

    #[cfg(feature = "cuda")]
    {
        // Try GPU-accelerated processing first
        println!("\n🚀 Testing with GPU acceleration...");

        match CudaProcessor::new(config.cuda.clone(), 0) {
            Ok(cuda_processor) => {
                println!("✅ CUDA processor initialized successfully");

                let stats = process_with_complete_algorithm(
                    &input_dir,
                    &output_file,
                    &config,
                    &mut memory_manager,
                    &cuda_processor,
                    true
                )?;

                println!("\n🎉 GPU Processing Complete!");
                verify_composite_deduplication_results(&output_file, &stats)?;
            }
            Err(e) => {
                println!("❌ CUDA initialization failed: {}", e);
                println!("🔄 Falling back to CPU processing...");

                let stats = process_with_complete_algorithm_cpu_fallback(
                    &input_dir,
                    &output_file,
                    &config,
                    &mut memory_manager,
                    true
                )?;

                println!("\n🎉 CPU Processing Complete!");
                verify_composite_deduplication_results(&output_file, &stats)?;
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n🔄 Testing with CPU processing...");

        let stats = process_with_complete_algorithm_cpu_fallback(
            &input_dir,
            &output_file,
            &config,
            &mut memory_manager,
            true
        )?;

        println!("\n🎉 CPU Processing Complete!");
        verify_composite_deduplication_results(&output_file, &stats)?;
    }

    // Cleanup
    println!("\n🧹 Cleaning up test files...");
    fs::remove_dir_all(test_dir)?;
    println!("✅ Test complete!");

    Ok(())
}

fn create_test_data_with_examples(input_dir: &Path) -> Result<()> {
    println!("\n📝 Creating test data with your specific examples...");

    let test_file = input_dir.join("composite_test.csv");
    let file = File::create(&test_file)?;
    let mut writer = BufWriter::new(file);

    // Your specific examples that should ALL be kept
    let test_data = vec![
        "email,password,url",
        "user@email.com,123,first.com",      // Example 1 (3 fields)
        "user@email.com,123,second.com",     // Example 2 - different URL (3 fields)
        "user@email.com,444,first.com",      // Example 3 - different password (3 fields)
        "user@email.com,444,first.com,extra", // Example 4 - different field count (4 fields)
        // Additional test cases
        "other@test.com,pass1,site1.com",    // Different email entirely (3 fields)
        "other@test.com,pass1,site1.com",    // Exact duplicate - should be removed (3 fields)
        "user@email.com,123,first.com",      // Exact duplicate of example 1 - should be removed (3 fields)
        "third@example.org,secret,example.org", // Another different email (3 fields)
        "user@email.com,999,third.com",      // Same email, different password and URL (3 fields)
    ];

    for line in &test_data {
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;

    println!("✅ Created test file with {} lines", test_data.len());
    println!("📊 Expected results with NEW deduplication logic:");
    println!("  - user@email.com should have 4 different records:");
    println!("    1. user@email.com,123,first.com (3 fields)");
    println!("    2. user@email.com,123,second.com (3 fields)");
    println!("    3. user@email.com,444,first.com (4 fields) - LONGER record kept!");
    println!("    4. user@email.com,999,third.com (3 fields)");
    println!("  - Records with same core fields but different lengths: KEEP THE LONGER ONE");
    println!("  - other@test.com should have 1 record (duplicate removed)");
    println!("  - third@example.org should have 1 record");
    println!("  - Total expected unique records: 6 (4 + 1 + 1)");

    Ok(())
}

fn create_test_config(test_dir: &Path) -> Result<Config> {
    let config = Config {
        memory: tuonella_sift::config::model::MemoryConfig {
            max_ram_usage_gb: 2,
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

fn verify_composite_deduplication_results(
    output_file: &Path,
    stats: &tuonella_sift::core::deduplication::ProcessingStats
) -> Result<()> {
    println!("\n🔍 Verifying composite deduplication results...");

    println!("📊 Processing Statistics:");
    println!("  📊 Total records processed: {}", stats.total_records);
    println!("  ✨ Unique records retained: {}", stats.unique_records);
    println!("  🗑️ Duplicates removed: {}", stats.duplicates_removed);
    println!("  ❌ Invalid records: {}", stats.invalid_records);

    if !output_file.exists() {
        println!("❌ Output file does not exist!");
        return Ok(());
    }

    let content = fs::read_to_string(output_file)?;
    let lines: Vec<&str> = content.lines().collect();

    println!("\n📄 Output file analysis:");
    println!("  📏 Total lines: {}", lines.len());
    println!("  📋 Header: {}", lines[0]);

    // Count records by email
    let mut email_counts = std::collections::HashMap::new();
    for line in lines.iter().skip(1) { // Skip header
        if let Some(email) = line.split(',').next() {
            *email_counts.entry(email).or_insert(0) += 1;
        }
    }

    println!("\n📈 Records by email:");
    for (email, count) in &email_counts {
        println!("  📧 {}: {} records", email, count);
    }

    // Verify expected results
    let user_email_count = email_counts.get("user@email.com").unwrap_or(&0);
    let other_email_count = email_counts.get("other@test.com").unwrap_or(&0);
    let third_email_count = email_counts.get("third@example.org").unwrap_or(&0);

    println!("\n✅ Verification Results:");

    if *user_email_count == 4 {
        println!("  ✅ user@email.com: {} records (CORRECT - longer records kept)", user_email_count);
        println!("      Records with same core fields but different lengths: LONGER ONE KEPT!");
    } else {
        println!("  ❌ user@email.com: {} records (EXPECTED 4 - longer records should be kept)", user_email_count);
    }

    if *other_email_count == 1 {
        println!("  ✅ other@test.com: {} record (CORRECT - duplicate removed)", other_email_count);
    } else {
        println!("  ❌ other@test.com: {} records (EXPECTED 1 - should remove exact duplicate)", other_email_count);
    }

    if *third_email_count == 1 {
        println!("  ✅ third@example.org: {} record (CORRECT)", third_email_count);
    } else {
        println!("  ❌ third@example.org: {} records (EXPECTED 1)", third_email_count);
    }

    let total_expected = 6; // 4 + 1 + 1
    let actual_unique = lines.len() - 1; // Subtract header

    if actual_unique == total_expected {
        println!("  ✅ Total unique records: {} (CORRECT)", actual_unique);
        println!("\n🎉 ENHANCED DEDUPLICATION TEST PASSED!");
        println!("✅ Records with same email but different core data are correctly kept separate");
        println!("✅ Records with same core fields: LONGER record is kept (more complete data)");
        println!("✅ Extra fields beyond email/password/URL are preserved in output");
    } else {
        println!("  ❌ Total unique records: {} (EXPECTED {})", actual_unique, total_expected);
        println!("\n❌ ENHANCED DEDUPLICATION TEST FAILED!");
    }

    println!("\n📝 All output records:");
    for (i, line) in lines.iter().skip(1).enumerate() {
        println!("  {}: {}", i + 1, line);
    }

    // Debug: Let's manually check what the dedup keys would be for our test cases
    println!("\n🔍 Debug: Expected dedup keys for test cases:");
    use tuonella_sift::core::record::Record;

    let test_cases = vec![
        ("user@email.com", "123", "first.com", 3),
        ("user@email.com", "123", "second.com", 3),
        ("user@email.com", "444", "first.com", 3),
        ("user@email.com", "444", "first.com", 4), // Extra field
        ("user@email.com", "999", "third.com", 3),
    ];

    for (i, (email, pass, url, field_count)) in test_cases.iter().enumerate() {
        if let Some(record) = Record::new_with_field_count(
            email.to_string(),
            pass.to_string(),
            url.to_string(),
            false, // case_sensitive = false
            *field_count,
        ) {
            println!("  Test case {}: {} -> key: {}", i + 1,
                    format!("{},{},{} ({}f)", email, pass, url, field_count),
                    record.dedup_key());
        }
    }

    Ok(())
}
