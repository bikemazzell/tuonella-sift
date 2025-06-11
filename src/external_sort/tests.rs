#[cfg(test)]
mod tests {
    use std::fs;
    use tempfile::tempdir;
    use crate::external_sort::{ExternalSortConfig, ExternalSortProcessor};
    use crate::external_sort::record::SortRecord;

    #[test]
    fn test_sort_record_creation() {
        let record = SortRecord::new(
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            vec!["extra1".to_string(), "extra2".to_string()],
        );

        assert_eq!(record.username, "user@example.com");
        assert_eq!(record.password, "password123");
        assert_eq!(record.url, "https://example.com");
        assert_eq!(record.normalized_username, "user@example.com");
        assert_eq!(record.normalized_url, "example.com");
        assert_eq!(record.extra_fields.len(), 2);
    }

    #[test]
    fn test_csv_parsing() {
        let csv_line = "user@test.com,pass123,https://site.com,field1,field2";
        let record = SortRecord::from_csv_line(csv_line).unwrap();

        assert_eq!(record.username, "user@test.com");
        assert_eq!(record.password, "pass123");
        assert_eq!(record.url, "https://site.com");
        assert_eq!(record.extra_fields, vec!["field1", "field2"]);
    }

    #[test]
    fn test_csv_output() {
        let record = SortRecord::new(
            "test@example.com".to_string(),
            "secret".to_string(),
            "example.com".to_string(),
            vec!["extra".to_string()],
        );

        let csv_line = record.to_csv_line();
        assert_eq!(csv_line, "test@example.com,secret,example.com,extra");
    }

    #[test]
    fn test_dedup_key() {
        let record = SortRecord::new(
            "User@Example.com".to_string(),
            "password".to_string(),
            "site.com".to_string(),
            vec![],
        );

        assert_eq!(record.dedup_key(false), "user@example.com:password");
        assert_eq!(record.dedup_key(true), "User@Example.com:password");
    }

    #[test]
    fn test_url_normalization() {
        let test_cases = vec![
            ("https://example.com/", "example.com"),
            ("http://test.com", "test.com"),
            ("android://com.app@data", "com.app"),
            ("ftp://files.com/", "files.com"),
            ("plain.com", "plain.com"),
        ];

        for (input, expected) in test_cases {
            let record = SortRecord::new(
                "user".to_string(),
                "pass".to_string(),
                input.to_string(),
                vec![],
            );
            assert_eq!(record.normalized_url, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = ExternalSortConfig::default();
        assert!(config.validate().is_ok());

        config.memory_usage_percent = 5.0;
        assert!(config.validate().is_err());

        config.memory_usage_percent = 95.0;
        assert!(config.validate().is_err());

        config.memory_usage_percent = 50.0;
        config.chunk_size_mb = 10;
        assert!(config.validate().is_err());

        config.chunk_size_mb = 5000;
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_basic_processing() {
        let temp_dir = tempdir().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_file = temp_dir.path().join("output.csv");
        
        fs::create_dir_all(&input_dir).unwrap();

        let test_data = vec![
            "user1@test.com,pass1,site1.com",
            "user2@test.com,pass2,site2.com",
            "user1@test.com,pass1,site1.com",
            "user3@test.com,pass3,site3.com",
        ];

        let input_file = input_dir.join("test.csv");
        fs::write(&input_file, test_data.join("\n")).unwrap();

        let mut config = ExternalSortConfig::default();
        config.temp_directory = temp_dir.path().join("temp");
        config.chunk_size_mb = 64;
        config.verbose = false;
        config.enable_cuda = false;

        let mut processor = ExternalSortProcessor::new(config).unwrap();
        let stats = processor.process(&[input_file], &output_file).await.unwrap();

        println!("Stats: total={}, unique={}, duplicates={}, files={}",
            stats.total_records, stats.unique_records, stats.duplicates_removed, stats.files_processed);

        assert!(stats.files_processed >= 1);
        assert!(output_file.exists());

        let output_content = fs::read_to_string(&output_file).unwrap();
        let lines: Vec<&str> = output_content.trim().split('\n').collect();
        assert_eq!(lines.len(), 3);

        processor.cleanup().unwrap();
    }

    #[test]
    fn test_checkpoint_serialization() {
        use crate::external_sort::checkpoint::SortCheckpoint;

        let checkpoint = SortCheckpoint::new(
            vec!["/path/to/file1.csv".into(), "/path/to/file2.csv".into()],
            "/path/to/output.csv".into(),
            "/tmp/sort".into(),
        );

        let temp_dir = tempdir().unwrap();
        checkpoint.save(temp_dir.path()).unwrap();

        let loaded = SortCheckpoint::load(temp_dir.path()).unwrap();
        assert_eq!(loaded.input_files.len(), 2);
        assert_eq!(loaded.output_file.to_string_lossy(), "/path/to/output.csv");
    }

    #[test]
    fn test_invalid_utf8_handling() {
        // Test that invalid lines are properly skipped
        let long_string = "a".repeat(5000);
        let invalid_lines = vec![
            "", // Empty line
            "user@test.com", // Too few fields
            &long_string, // Too long
            "user@test.com,pass,site.com", // Valid line
            "user\x00invalid,pass,site", // Contains null byte
            "user2@test.com,pass2,site2.com", // Another valid line
        ];

        let mut valid_count = 0;
        for line in invalid_lines {
            if SortRecord::from_csv_line(line).is_some() {
                valid_count += 1;
            }
        }

        assert_eq!(valid_count, 2); // Only 2 valid lines should be parsed
    }

    // Additional comprehensive tests for external_sort modules

    #[test]
    fn test_config_file_operations() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.json");
        
        let config = ExternalSortConfig::default();
        
        // Test writing config to file
        config.to_file(&config_path).unwrap();
        assert!(config_path.exists());
        
        // Test reading config from file
        let loaded_config = ExternalSortConfig::from_file(&config_path).unwrap();
        assert_eq!(config.memory_usage_percent, loaded_config.memory_usage_percent);
        assert_eq!(config.chunk_size_mb, loaded_config.chunk_size_mb);
        assert_eq!(config.processing_threads, loaded_config.processing_threads);
    }

    #[test]
    fn test_config_size_calculations() {
        let config = ExternalSortConfig {
            memory_usage_percent: 50.0,
            chunk_size_mb: 256,
            io_buffer_size_kb: 128,
            processing_threads: 4,
            enable_cuda: false,
            cuda_batch_size: 100000,
            cuda_memory_percent: 80.0,
            temp_directory: std::path::PathBuf::from("/tmp"),
            enable_compression: false,
            merge_buffer_size_kb: 512,
            case_sensitive: false,
            normalize_urls: true,
            email_only_usernames: false,
            verbose: false,
            merge_progress_interval_seconds: 10,
        };
        
        // Test size calculations
        assert_eq!(config.chunk_size_bytes(), 256 * 1024 * 1024);
        assert_eq!(config.io_buffer_size_bytes(), 128 * 1024);
        assert_eq!(config.merge_buffer_size_bytes(), 512 * 1024);
        
        // Memory limit should be reasonable (depends on system memory)
        let memory_limit = config.memory_limit_bytes();
        assert!(memory_limit > 0);
    }

    #[test]
    fn test_config_invalid_values() {
        use crate::external_sort::config::ExternalSortConfig;
        
        // Test invalid memory usage percent
        let mut config = ExternalSortConfig::default();
        config.memory_usage_percent = 95.0; // Above max (90%)
        assert!(config.validate().is_err());
        
        config.memory_usage_percent = 5.0; // Below min (10%)
        assert!(config.validate().is_err());
        
        // Test invalid chunk size
        config.memory_usage_percent = 60.0; // Valid
        config.chunk_size_mb = 32; // Below min (64)
        assert!(config.validate().is_err());
        
        config.chunk_size_mb = 5000; // Above max (4096)
        assert!(config.validate().is_err());
        
        // Test invalid processing threads
        config.chunk_size_mb = 512; // Valid
        config.processing_threads = 0; // Below min (1)
        assert!(config.validate().is_err());
        
        config.processing_threads = 64; // Above max (32)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_checkpoint_operations() {
        use tempfile::tempdir;
        use crate::external_sort::checkpoint::SortCheckpoint;
        use std::path::PathBuf;
        
        let temp_dir = tempdir().unwrap();
        let input_files = vec![PathBuf::from("input1.csv"), PathBuf::from("input2.csv")];
        let output_file = PathBuf::from("output.csv");
        let temp_directory = temp_dir.path().to_path_buf();
        
        let mut checkpoint = SortCheckpoint::new(input_files.clone(), output_file.clone(), temp_directory.clone());
        
        // Test checkpoint creation
        assert_eq!(checkpoint.input_files, input_files);
        assert_eq!(checkpoint.output_file, output_file);
        assert_eq!(checkpoint.temp_directory, temp_directory);
        
        // Test timestamp update - just check that it changes
        let original_timestamp = checkpoint.timestamp;
        checkpoint.update_timestamp();
        // The timestamp should be updated (may be same or different depending on system resolution)
        assert!(checkpoint.timestamp >= original_timestamp);
        
        // Test progress calculation for FileProcessing phase
        checkpoint.phase = crate::external_sort::checkpoint::ProcessingPhase::FileProcessing;
        checkpoint.completed_files.push(input_files[0].clone()); // One file completed out of two
        let progress = checkpoint.progress_percentage();
        assert!((progress - 40.0).abs() < f64::EPSILON); // 1/2 * 80% = 40%
        
        // Test save and load
        checkpoint.save(temp_dir.path()).unwrap();
        assert!(SortCheckpoint::exists(temp_dir.path()));
        
        let loaded_checkpoint = SortCheckpoint::load(temp_dir.path()).unwrap();
        assert_eq!(checkpoint.input_files, loaded_checkpoint.input_files);
        assert_eq!(checkpoint.output_file, loaded_checkpoint.output_file);
    }

    #[test]
    fn test_sort_record_normalization() {
        let record = SortRecord::new(
            "Test@Example.COM".to_string(),
            "password".to_string(),
            "HTTPS://WWW.Example.COM/Path".to_string(),
            vec!["Extra".to_string()],
        );
        
        // Test that normalization is applied (URL normalization keeps www and path)
        assert_eq!(record.normalized_username, "test@example.com");
        assert_eq!(record.normalized_url, "www.example.com/path");
        
        // Original fields should be unchanged
        assert_eq!(record.username, "Test@Example.COM");
        assert_eq!(record.url, "HTTPS://WWW.Example.COM/Path");
    }

    #[test]
    fn test_sort_record_comparison() {
        let record1 = SortRecord::new(
            "a@example.com".to_string(),
            "pass1".to_string(),
            "example.com".to_string(),
            vec![],
        );
        
        let record2 = SortRecord::new(
            "b@example.com".to_string(),
            "pass2".to_string(),
            "example.com".to_string(),
            vec![],
        );
        
        let record3 = SortRecord::new(
            "a@example.com".to_string(),
            "pass1".to_string(), // Same password for true equality
            "different.com".to_string(),
            vec![],
        );
        
        // Test ordering based on dedup key (username:password)
        assert!(record1 < record2);
        
        // Test equality based on dedup key (same normalized username + password)
        assert_eq!(record1.dedup_key(false), record3.dedup_key(false));
    }

    #[test]
    fn test_record_serialization() {
        let record = SortRecord::new(
            "test@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            vec!["field1".to_string(), "field2".to_string()],
        );
        
        // Test JSON serialization
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("test@example.com"));
        assert!(json.contains("password123"));
        assert!(json.contains("https://example.com"));
        
        // Test deserialization
        let deserialized: SortRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.username, deserialized.username);
        assert_eq!(record.password, deserialized.password);
        assert_eq!(record.url, deserialized.url);
        assert_eq!(record.extra_fields, deserialized.extra_fields);
    }

    #[test]
    fn test_record_edge_cases() {
        // Test record with special characters
        let record = SortRecord::new(
            "user+tag@domain.co.uk".to_string(),
            "pass,with,commas".to_string(),
            "https://sub.domain.com:8080/path?query=value#anchor".to_string(),
            vec!["field,with,comma".to_string()],
        );
        
        let csv_line = record.to_csv_line();
        assert!(csv_line.contains("user+tag@domain.co.uk"));
        
        // Test parsing it back
        let parsed = SortRecord::from_csv_line(&csv_line);
        assert!(parsed.is_some());
        let parsed_record = parsed.unwrap();
        assert_eq!(record.username, parsed_record.username);
    }

    #[test]
    fn test_record_unicode_handling() {
        let record = SortRecord::new(
            "测试@example.com".to_string(),
            "密码123".to_string(),
            "https://日本.example.com".to_string(),
            vec!["中文字段".to_string()],
        );
        
        // Test that unicode is preserved
        assert_eq!(record.username, "测试@example.com");
        assert_eq!(record.password, "密码123");
        assert_eq!(record.url, "https://日本.example.com");
        assert_eq!(record.extra_fields[0], "中文字段");
        
        // Test CSV round-trip
        let csv_line = record.to_csv_line();
        let parsed = SortRecord::from_csv_line(&csv_line);
        assert!(parsed.is_some());
        let parsed_record = parsed.unwrap();
        assert_eq!(record.username, parsed_record.username);
    }

    #[test]
    fn test_invalid_csv_formats() {
        // Test various invalid CSV formats
        let invalid_cases = vec![
            "", // Empty line
            "single_field", // Not enough fields
            "user,pass,url,extra,too,many,fields,for,normal,processing", // Many fields should still work
        ];
        
        for case in invalid_cases {
            let result = SortRecord::from_csv_line(case);
            if case == "" || case == "single_field" {
                assert!(result.is_none(), "Expected None for: {}", case);
            } else {
                // Lines with enough fields should parse
                assert!(result.is_some(), "Expected Some for: {}", case);
            }
        }
    }

    #[test]
    fn test_dedup_key_case_sensitivity() {
        let record = SortRecord::new(
            "Test@Example.COM".to_string(),
            "password".to_string(),
            "HTTPS://Example.COM".to_string(),
            vec![],
        );
        
        // Test case insensitive dedup key (uses normalized_username:password)
        let case_insensitive_key = record.dedup_key(false);
        assert_eq!(case_insensitive_key, "test@example.com:password");
        
        // Test case sensitive dedup key (uses original username:password)
        let case_sensitive_key = record.dedup_key(true);
        assert_eq!(case_sensitive_key, "Test@Example.COM:password");
    }

    #[tokio::test]
    async fn test_processor_initialization() {
        let config = ExternalSortConfig::default();
        let processor = ExternalSortProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_complex_url_normalization() {
        let test_cases = vec![
            ("https://www.Example.COM/Path", "www.example.com/path"),
            ("http://SUB.DOMAIN.ORG:8080/path?query", "sub.domain.org:8080/path?query"),
            ("HTTPS://SITE.NET/", "site.net"),
            ("android://com.app.package@HOST.COM", "com.app.package"),
            ("ftp://files.SERVER.NET/path", "files.server.net/path"),
            ("mailto:user@DOMAIN.COM", "user@domain.com"),
        ];
        
        for (input, expected) in test_cases {
            let record = SortRecord::new(
                "user@test.com".to_string(),
                "pass".to_string(),
                input.to_string(),
                vec![],
            );
            assert_eq!(record.normalized_url, expected, "Failed for input: {}", input);
        }
    }
}
