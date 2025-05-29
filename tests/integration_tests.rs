use anyhow::Result;
use std::fs;
use tempfile::TempDir;
use tokio;
use tuonella_sift::{Config, Deduplicator};

/// Helper function to create a temporary directory with test CSV files
fn create_test_data(temp_dir: &TempDir) -> Result<()> {
    let test_data_dir = temp_dir.path().join("input");
    fs::create_dir_all(&test_data_dir)?;

    // Create test CSV file 1
    let csv1_content = r#"user1@example.com,password123,https://www.facebook.com/user1
test@gmail.com,secret456,http://m.twitter.com/test
admin@site.org,admin789,google.com/search?q=test
user1@example.com,password123,facebook.com/user1
different@email.com,pass123,https://www.linkedin.com/in/user"#;

    fs::write(test_data_dir.join("test1.csv"), csv1_content)?;

    // Create test CSV file 2
    let csv2_content = r#"john@doe.com,mypass,https://github.com/johndoe
jane@smith.net,janepass,www.instagram.com/jane
user1@example.com,password123,https://facebook.com/user1/
test@gmail.com,secret456,mobile.twitter.com/test"#;

    fs::write(test_data_dir.join("test2.csv"), csv2_content)?;

    Ok(())
}

/// Helper function to create a test configuration
async fn create_test_config(temp_dir: &TempDir) -> Result<Config> {
    let config_content = format!(r#"{{
        "memory": {{
            "max_ram_usage_percent": 30,
            "batch_size_gb": 1,
            "auto_detect_memory": false
        }},
        "processing": {{
            "max_threads": 2,
            "enable_cuda": false,
            "chunk_size_mb": 16,
            "max_output_file_size_gb": 1
        }},
        "io": {{
            "temp_directory": "{}",
            "output_directory": "{}",
            "enable_memory_mapping": true,
            "parallel_io": true
        }},
        "deduplication": {{
            "case_sensitive_usernames": false,
            "normalize_urls": true,
            "strip_url_params": true,
            "strip_url_prefixes": true,
            "completeness_strategy": "character_count",
            "field_detection_sample_percent": 5.0,
            "min_sample_size": 50,
            "max_sample_size": 1000
        }},
        "logging": {{
            "verbosity": "normal",
            "progress_interval_seconds": 30,
            "log_file": "{}"
        }},
        "recovery": {{
            "enable_checkpointing": false,
            "checkpoint_interval_records": 100000
        }}
    }}"#,
        temp_dir.path().join("temp").display(),
        temp_dir.path().join("output").display(),
        temp_dir.path().join("test.log").display()
    );

    let config_path = temp_dir.path().join("config.json");
    fs::write(&config_path, config_content)?;

    // Load the config using the actual Config::load method
    Config::load(&config_path).await
}

#[tokio::test]
async fn test_end_to_end_deduplication() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Create test data
    create_test_data(&temp_dir)?;

    // Create test configuration
    let config = create_test_config(&temp_dir).await?;

    // Create output directory
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir)?;

    // Initialize deduplicator
    let mut deduplicator = Deduplicator::new(config).await?;

    // Process the test data
    let input_dir = temp_dir.path().join("input");

    // Debug: Check if input files exist
    println!("Input directory: {}", input_dir.display());
    let input_files: Vec<_> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok())
        .collect();
    println!("Found {} files in input directory", input_files.len());
    for file in &input_files {
        println!("  - {}", file.file_name().to_string_lossy());
    }

    let stats = deduplicator.process_directory(&input_dir, &output_dir).await?;

    // Verify results
    assert!(stats.files_processed > 0, "Should have processed at least one file");
    assert!(stats.total_records > 0, "Should have found some records");
    assert!(stats.unique_records > 0, "Should have some unique records");

    // Check that output files were created
    let output_files: Vec<_> = fs::read_dir(&output_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "csv")
                .unwrap_or(false)
        })
        .collect();

    assert!(!output_files.is_empty(), "Should have created output CSV files");

    // Verify that duplicates were actually removed
    // We know from our test data that user1@example.com appears multiple times
    // with the same password and similar URLs (facebook.com variants)
    println!("Duplicates removed: {}", stats.duplicates_removed);
    println!("Total records: {}", stats.total_records);
    println!("Unique records: {}", stats.unique_records);

    // Let's check the output file to see what was actually written
    let output_files: Vec<_> = fs::read_dir(&output_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "csv")
                .unwrap_or(false)
        })
        .collect();

    if let Some(output_file) = output_files.first() {
        let content = fs::read_to_string(output_file.path())?;
        println!("Output file content:\n{}", content);
    }

    assert!(stats.duplicates_removed > 0, "Should have removed some duplicates");

    println!("Integration test results:");
    println!("  Files processed: {}", stats.files_processed);
    println!("  Total records: {}", stats.total_records);
    println!("  Unique records: {}", stats.unique_records);
    println!("  Duplicates removed: {}", stats.duplicates_removed);
    println!("  Processing time: {:.2}s", stats.processing_time_seconds);

    Ok(())
}

#[tokio::test]
async fn test_empty_directory_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Create empty input directory
    let input_dir = temp_dir.path().join("empty_input");
    fs::create_dir_all(&input_dir)?;

    // Create test configuration
    let config = create_test_config(&temp_dir).await?;

    // Create output directory
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir)?;

    // Initialize deduplicator
    let mut deduplicator = Deduplicator::new(config).await?;

    // Process the empty directory
    let stats = deduplicator.process_directory(&input_dir, &output_dir).await?;

    // Verify results for empty directory
    assert_eq!(stats.files_processed, 0, "Should have processed zero files");
    assert_eq!(stats.total_records, 0, "Should have zero records");
    assert_eq!(stats.unique_records, 0, "Should have zero unique records");
    assert_eq!(stats.duplicates_removed, 0, "Should have removed zero duplicates");

    Ok(())
}

#[tokio::test]
async fn test_malformed_csv_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Create input directory with malformed CSV
    let input_dir = temp_dir.path().join("input");
    fs::create_dir_all(&input_dir)?;

    // Create a malformed CSV file
    let malformed_csv = r#"incomplete,line
user@example.com,password123
this,line,has,too,many,fields,and,should,be,ignored
another@email.com,validpass,https://example.com"#;

    fs::write(input_dir.join("malformed.csv"), malformed_csv)?;

    // Create test configuration
    let config = create_test_config(&temp_dir).await?;

    // Create output directory
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir)?;

    // Initialize deduplicator
    let mut deduplicator = Deduplicator::new(config).await?;

    // Process the directory with malformed CSV
    let stats = deduplicator.process_directory(&input_dir, &output_dir).await?;

    // Should handle malformed data gracefully
    assert!(stats.files_processed > 0, "Should have processed the file");
    // The exact number of records depends on how many lines are valid
    // but it should not crash

    println!("Malformed CSV test results:");
    println!("  Files processed: {}", stats.files_processed);
    println!("  Total records: {}", stats.total_records);
    println!("  Unique records: {}", stats.unique_records);

    Ok(())
}

#[tokio::test]
async fn test_config_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Test invalid configuration
    let invalid_config_content = r#"{
        "memory": {
            "max_ram_usage_percent": 150,
            "batch_size_gb": -1,
            "auto_detect_memory": false
        },
        "processing": {
            "max_threads": 0,
            "enable_cuda": false,
            "chunk_size_mb": 0,
            "max_output_file_size_gb": 0
        },
        "io": {
            "temp_directory": "./temp",
            "output_directory": "./output",
            "enable_memory_mapping": true,
            "parallel_io": true
        },
        "deduplication": {
            "case_sensitive_usernames": false,
            "normalize_urls": true,
            "strip_url_params": true,
            "strip_url_prefixes": true,
            "completeness_strategy": "invalid_strategy",
            "field_detection_sample_percent": -5.0,
            "min_sample_size": 1000,
            "max_sample_size": 50
        },
        "logging": {
            "verbosity": "invalid_level",
            "log_file": "./test.log"
        },
        "recovery": {
            "enable_checkpoints": false,
            "checkpoint_interval_seconds": 300
        }
    }"#;

    let config_path = temp_dir.path().join("invalid_config.json");
    fs::write(&config_path, invalid_config_content)?;

    // Try to load the invalid config
    let result = Config::load(&config_path).await;

    // Should fail validation
    assert!(result.is_err(), "Invalid config should fail to load");

    Ok(())
}
