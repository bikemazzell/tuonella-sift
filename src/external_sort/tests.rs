#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use crate::external_sort::{ExternalSortConfig, ExternalSortProcessor};
    use crate::external_sort::record::SortRecord;
    use crate::external_sort::chunk::ChunkProcessor;
    use crate::external_sort::checkpoint::{ChunkMetadata, SortCheckpoint};
    use crate::external_sort::merger::ChunkMerger;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

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

    // Tests for chunk.rs functionality
    #[test]
    fn test_estimate_chunk_count() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024, // 1MB chunks
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Test various file sizes
        assert_eq!(processor.estimate_chunk_count(0), 1); // Minimum 1 chunk
        assert_eq!(processor.estimate_chunk_count(1024), 1);
        assert_eq!(processor.estimate_chunk_count(1024 * 1024), 1);
        assert_eq!(processor.estimate_chunk_count(1024 * 1024 + 1), 2);
        assert_eq!(processor.estimate_chunk_count(5 * 1024 * 1024), 5);
        assert_eq!(processor.estimate_chunk_count(10 * 1024 * 1024 - 1), 10);
        assert_eq!(processor.estimate_chunk_count(10 * 1024 * 1024), 10);
    }

    #[test]
    fn test_cleanup_chunk() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Create a test chunk file
        let chunk_path = temp_dir.path().join("test_chunk.csv");
        fs::write(&chunk_path, "test,data\n").unwrap();
        
        let chunk_metadata = ChunkMetadata {
            chunk_id: 0,
            file_path: chunk_path.clone(),
            record_count: 1,
            file_size_bytes: 10,
            is_sorted: true,
            source_files: vec![],
        };

        // Verify file exists
        assert!(chunk_path.exists());

        // Cleanup the chunk
        processor.cleanup_chunk(&chunk_metadata).unwrap();

        // Verify file is deleted
        assert!(!chunk_path.exists());
    }

    #[test]
    fn test_cleanup_chunk_missing_file() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        let chunk_metadata = ChunkMetadata {
            chunk_id: 0,
            file_path: temp_dir.path().join("nonexistent.csv"),
            record_count: 0,
            file_size_bytes: 0,
            is_sorted: true,
            source_files: vec![],
        };

        // Should not error on missing file
        let result = processor.cleanup_chunk(&chunk_metadata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cleanup_all_chunks() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Create multiple chunk files
        let mut chunks = vec![];
        for i in 0..3 {
            let chunk_path = temp_dir.path().join(format!("chunk_{}.csv", i));
            fs::write(&chunk_path, format!("test{},data{}\n", i, i)).unwrap();
            
            chunks.push(ChunkMetadata {
                chunk_id: i,
                file_path: chunk_path.clone(),
                record_count: 1,
                file_size_bytes: 10,
                is_sorted: true,
                source_files: vec![],
            });
            
            assert!(chunk_path.exists());
        }

        // Cleanup all chunks
        processor.cleanup_all_chunks(&chunks).unwrap();

        // Verify all files are deleted
        for chunk in &chunks {
            assert!(!chunk.file_path.exists());
        }
    }

    #[test]
    fn test_cleanup_all_chunks_with_errors() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Mix of existing and non-existing files
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: temp_dir.path().join("exists.csv"),
                record_count: 1,
                file_size_bytes: 10,
                is_sorted: true,
                source_files: vec![],
            },
            ChunkMetadata {
                chunk_id: 1,
                file_path: temp_dir.path().join("missing.csv"),
                record_count: 0,
                file_size_bytes: 0,
                is_sorted: true,
                source_files: vec![],
            },
        ];

        // Create only the first file
        fs::write(&chunks[0].file_path, "data\n").unwrap();

        // Should handle mixed cases without panicking
        let result = processor.cleanup_all_chunks(&chunks);
        assert!(result.is_ok());

        // Existing file should be deleted
        assert!(!chunks[0].file_path.exists());
    }

    #[tokio::test]
    async fn test_sort_and_write_chunk() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            false, // case insensitive
        );

        // Create unsorted records
        let records = vec![
            SortRecord::new(
                "zebra@test.com".to_string(),
                "pass3".to_string(),
                "https://site3.com".to_string(),
                vec![],
            ),
            SortRecord::new(
                "alpha@test.com".to_string(),
                "pass1".to_string(),
                "https://site1.com".to_string(),
                vec![],
            ),
            SortRecord::new(
                "beta@test.com".to_string(),
                "pass2".to_string(),
                "https://site2.com".to_string(),
                vec![],
            ),
            SortRecord::new(
                "alpha@test.com".to_string(),
                "pass1_dup".to_string(),
                "https://site1.com".to_string(),
                vec![],
            ),
        ];

        let metadata = processor.sort_and_write_chunk(
            0,
            records,
            vec![PathBuf::from("test.csv")],
        ).await.unwrap();

        // Verify chunk file was created
        assert!(metadata.file_path.exists());
        assert_eq!(metadata.chunk_id, 0);
        // Note: The actual deduplication happens based on normalized_username + normalized_url
        // All 4 records have different dedup keys, so no duplicates are removed
        assert_eq!(metadata.record_count, 4); // All 4 records kept (different passwords)
        assert!(metadata.is_sorted);

        // Read and verify content is sorted
        let content = fs::read_to_string(&metadata.file_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 4);
        
        // Should be sorted by normalized username + url (all have same url)
        // So sorted by username: alpha (x2), beta, zebra
        assert!(lines[0].starts_with("alpha@test.com"));
        assert!(lines[1].starts_with("alpha@test.com")); // Second alpha record
        assert!(lines[2].starts_with("beta@test.com"));
        assert!(lines[3].starts_with("zebra@test.com"));

        // Cleanup
        fs::remove_file(&metadata.file_path).unwrap();
    }

    #[tokio::test]
    async fn test_sort_and_write_chunk_case_sensitive() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true, // case sensitive
        );

        let records = vec![
            SortRecord::new(
                "User@test.com".to_string(),
                "pass1".to_string(),
                "https://site.com".to_string(),
                vec![],
            ),
            SortRecord::new(
                "user@test.com".to_string(),
                "pass2".to_string(),
                "https://site.com".to_string(),
                vec![],
            ),
        ];

        let metadata = processor.sort_and_write_chunk(
            0,
            records,
            vec![],
        ).await.unwrap();

        // Both records should be kept (different case)
        assert_eq!(metadata.record_count, 2);

        // Cleanup
        fs::remove_file(&metadata.file_path).unwrap();
    }

    #[test]
    fn test_read_chunk_records() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Create a chunk file with test data
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        let test_data = "user1@test.com,pass1,https://site1.com\nuser2@test.com,pass2,https://site2.com\n";
        fs::write(&chunk_path, test_data).unwrap();

        let chunk_metadata = ChunkMetadata {
            chunk_id: 0,
            file_path: chunk_path,
            record_count: 2,
            file_size_bytes: test_data.len() as u64,
            is_sorted: true,
            source_files: vec![],
        };

        // Read records
        let records = processor.read_chunk_records(&chunk_metadata).unwrap();
        
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].username, "user1@test.com");
        assert_eq!(records[1].username, "user2@test.com");
    }

    #[test]
    fn test_read_chunk_records_with_invalid_lines() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024 * 1024,
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Create chunk with some invalid lines
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        let test_data = "user1@test.com,pass1,https://site1.com\ninvalid_line\nuser2@test.com,pass2,https://site2.com\n";
        fs::write(&chunk_path, test_data).unwrap();

        let chunk_metadata = ChunkMetadata {
            chunk_id: 0,
            file_path: chunk_path,
            record_count: 2,
            file_size_bytes: test_data.len() as u64,
            is_sorted: true,
            source_files: vec![],
        };

        // Should skip invalid lines
        let records = processor.read_chunk_records(&chunk_metadata).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_process_file_to_chunks() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            100, // Very small chunk size to force multiple chunks
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Create a test CSV file
        let test_file = temp_dir.path().join("test.csv");
        let mut content = String::new();
        for i in 0..10 {
            content.push_str(&format!("user{}@test.com,pass{},https://site{}.com\n", i, i, i));
        }
        fs::write(&test_file, content).unwrap();

        // Create checkpoint
        let mut checkpoint = SortCheckpoint::new(
            vec![test_file.clone()],
            temp_dir.path().join("output.csv"),
            temp_dir.path().to_path_buf(),
        );

        // Process file
        let chunks = processor.process_file_to_chunks(&test_file, &mut checkpoint).await.unwrap();

        // Should create multiple chunks due to small chunk size
        assert!(chunks.len() > 1);

        // Verify all chunks exist
        for chunk in &chunks {
            assert!(chunk.file_path.exists());
        }

        // Cleanup
        processor.cleanup_all_chunks(&chunks).unwrap();
    }

    #[test]
    fn test_estimate_chunk_count_edge_cases() {
        let temp_dir = tempdir().unwrap();
        let processor = ChunkProcessor::new(
            1024, // 1KB chunks
            8192,
            temp_dir.path().to_path_buf(),
            true,
        );

        // Edge cases
        assert_eq!(processor.estimate_chunk_count(0), 1); // Zero size -> 1 chunk
        assert_eq!(processor.estimate_chunk_count(1), 1);
        assert_eq!(processor.estimate_chunk_count(1023), 1);
        assert_eq!(processor.estimate_chunk_count(1024), 1);
        assert_eq!(processor.estimate_chunk_count(1025), 2);
        // For u64::MAX, the calculation might overflow on 32-bit systems
        // Just verify it returns a non-zero value
        assert!(processor.estimate_chunk_count(u64::MAX) > 0);
    }

    // Tests for merger.rs functionality
    #[test]
    fn test_validate_chunks_success() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10);

        // Create valid chunk files
        let mut chunks = vec![];
        for i in 0..3 {
            let chunk_path = temp_dir.path().join(format!("chunk_{}.csv", i));
            fs::write(&chunk_path, "data\n").unwrap();
            
            chunks.push(ChunkMetadata {
                chunk_id: i,
                file_path: chunk_path,
                record_count: 1,
                file_size_bytes: 5,
                is_sorted: true,
                source_files: vec![],
            });
        }

        // Should validate successfully
        let result = merger.validate_chunks(&chunks);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_chunks_missing_file() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10);

        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: temp_dir.path().join("nonexistent.csv"),
                record_count: 1,
                file_size_bytes: 10,
                is_sorted: true,
                source_files: vec![],
            },
        ];

        // Should fail with missing file error
        let result = merger.validate_chunks(&chunks);
        assert!(result.is_err());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("does not exist"));
    }

    #[test]
    fn test_validate_chunks_unsorted() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10);

        // Create a file but mark it as unsorted
        let chunk_path = temp_dir.path().join("chunk.csv");
        fs::write(&chunk_path, "data\n").unwrap();

        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk_path,
                record_count: 1,
                file_size_bytes: 5,
                is_sorted: false, // Not sorted
                source_files: vec![],
            },
        ];

        // Should fail with not sorted error
        let result = merger.validate_chunks(&chunks);
        assert!(result.is_err());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("not sorted"));
    }

    #[test]
    fn test_estimate_merge_time() {
        let merger = ChunkMerger::new(8192, true, 10);

        // Test with no chunks
        let chunks: Vec<ChunkMetadata> = vec![];
        assert_eq!(merger.estimate_merge_time(&chunks), 1000); // Minimum 1 second

        // Test with chunks containing records
        let temp_dir = tempdir().unwrap();
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: temp_dir.path().join("chunk1.csv"),
                record_count: 1000,
                file_size_bytes: 10000,
                is_sorted: true,
                source_files: vec![],
            },
            ChunkMetadata {
                chunk_id: 1,
                file_path: temp_dir.path().join("chunk2.csv"),
                record_count: 2000,
                file_size_bytes: 20000,
                is_sorted: true,
                source_files: vec![],
            },
        ];

        // Should estimate based on total records (3000 * 0.001 = 3ms, but min is 1000ms)
        assert_eq!(merger.estimate_merge_time(&chunks), 1000);

        // Test with many records
        let big_chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: temp_dir.path().join("big.csv"),
                record_count: 10_000_000,
                file_size_bytes: 100_000_000,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        // 10M * 0.001 = 10000ms
        assert_eq!(merger.estimate_merge_time(&big_chunks), 10000);
    }

    #[tokio::test]
    async fn test_merge_chunks_empty() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10);
        let output_file = temp_dir.path().join("output.csv");
        
        let chunks: Vec<ChunkMetadata> = vec![];
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        // Should handle empty chunks gracefully
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Output file should not be created for empty chunks
        assert!(!output_file.exists());
    }

    #[tokio::test]
    async fn test_merge_chunks_single() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, false, 10);
        let output_file = temp_dir.path().join("output.csv");
        
        // Create a single chunk with sorted data
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        let content = "alice@test.com,pass1,https://site1.com\nbob@test.com,pass2,https://site2.com\n";
        fs::write(&chunk_path, content).unwrap();
        
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk_path,
                record_count: 2,
                file_size_bytes: content.len() as u64,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        // Merge single chunk
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Verify output
        assert!(output_file.exists());
        let output_content = fs::read_to_string(&output_file).unwrap();
        assert_eq!(output_content.lines().count(), 2);
        assert!(output_content.contains("alice@test.com"));
        assert!(output_content.contains("bob@test.com"));
    }

    #[tokio::test]
    async fn test_merge_chunks_multiple_with_duplicates() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, false, 10); // case insensitive
        let output_file = temp_dir.path().join("output.csv");
        
        // Create multiple chunks - each chunk must be internally sorted for proper merge
        // Chunk 1: alice, charlie (sorted by dedup key)
        let chunk1_path = temp_dir.path().join("chunk_0.csv");
        fs::write(&chunk1_path, "alice@test.com,pass1,https://site.com\ncharlie@test.com,pass3,https://site.com\n").unwrap();
        
        // Chunk 2: bob, david (sorted by dedup key)  
        let chunk2_path = temp_dir.path().join("chunk_1.csv");
        fs::write(&chunk2_path, "bob@test.com,pass2,https://site.com\ndavid@test.com,pass4,https://site.com\n").unwrap();
        
        // Chunk 3: alice duplicate (exact same record)
        let chunk3_path = temp_dir.path().join("chunk_2.csv");
        fs::write(&chunk3_path, "alice@test.com,pass1,https://site.com\n").unwrap();
        
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk1_path,
                record_count: 2,
                file_size_bytes: 80,
                is_sorted: true,
                source_files: vec![],
            },
            ChunkMetadata {
                chunk_id: 1,
                file_path: chunk2_path,
                record_count: 2,
                file_size_bytes: 80,
                is_sorted: true,
                source_files: vec![],
            },
            ChunkMetadata {
                chunk_id: 2,
                file_path: chunk3_path,
                record_count: 1,
                file_size_bytes: 50,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        // Merge chunks
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Verify output
        assert!(output_file.exists());
        let output_content = fs::read_to_string(&output_file).unwrap();
        let lines: Vec<&str> = output_content.lines().collect();
        
        // Should have 4 unique records after deduplication
        // alice appears in chunk1 and chunk3 (duplicate), so one should be removed
        assert_eq!(lines.len(), 4);
        
        // Verify we have the expected usernames
        assert!(output_content.contains("alice@test.com"));
        assert!(output_content.contains("bob@test.com"));
        assert!(output_content.contains("charlie@test.com"));
        assert!(output_content.contains("david@test.com"));
        
        // Check stats - should have 4 unique and 1 duplicate removed
        assert_eq!(checkpoint.stats.unique_records, 4);
        assert_eq!(checkpoint.stats.duplicates_removed, 1); // One alice duplicate
    }

    #[tokio::test]
    async fn test_merge_chunks_case_sensitive() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10); // case sensitive
        let output_file = temp_dir.path().join("output.csv");
        
        // Create chunks with different cases
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        fs::write(&chunk_path, 
            "Alice@test.com,pass1,https://site.com\nalice@test.com,pass2,https://site.com\n"
        ).unwrap();
        
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk_path,
                record_count: 2,
                file_size_bytes: 80,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        // Merge with case sensitive
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Both records should be kept (different case)
        let output_content = fs::read_to_string(&output_file).unwrap();
        assert_eq!(output_content.lines().count(), 2);
    }

    #[tokio::test]
    async fn test_merge_chunks_with_shutdown() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, false, 10);
        let output_file = temp_dir.path().join("output.csv");
        
        // Create a large chunk
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        let mut content = String::new();
        for i in 0..10000 {
            content.push_str(&format!("user{}@test.com,pass{},https://site.com\n", i, i));
        }
        fs::write(&chunk_path, &content).unwrap();
        
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk_path,
                record_count: 10000,
                file_size_bytes: content.len() as u64,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        // Set shutdown flag immediately
        let shutdown_flag = Arc::new(AtomicBool::new(true));
        
        // Should handle shutdown gracefully
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Output file might be partially written
        if output_file.exists() {
            let output_content = fs::read_to_string(&output_file).unwrap();
            // Should have stopped early due to shutdown
            assert!(output_content.lines().count() < 10000);
        }
    }

    #[test]
    fn test_merge_entry_ordering() {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        // Test the MergeEntry ordering for the heap
        let record1 = SortRecord::new(
            "alice@test.com".to_string(),
            "pass".to_string(),
            "https://site.com".to_string(),
            vec![],
        );
        
        let record2 = SortRecord::new(
            "bob@test.com".to_string(),
            "pass".to_string(),
            "https://site.com".to_string(),
            vec![],
        );
        
        let record3 = SortRecord::new(
            "charlie@test.com".to_string(),
            "pass".to_string(),
            "https://site.com".to_string(),
            vec![],
        );
        
        // Create a min-heap using Reverse
        let mut heap = BinaryHeap::new();
        
        // Add in random order
        heap.push(Reverse((record2.dedup_key(false), 1)));
        heap.push(Reverse((record1.dedup_key(false), 0)));
        heap.push(Reverse((record3.dedup_key(false), 2)));
        
        // Should pop in sorted order
        let first = heap.pop().unwrap();
        assert!(first.0.0.starts_with("alice"));
        
        let second = heap.pop().unwrap();
        assert!(second.0.0.starts_with("bob"));
        
        let third = heap.pop().unwrap();
        assert!(third.0.0.starts_with("charlie"));
    }

    #[tokio::test]
    async fn test_merge_chunks_with_invalid_records() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, false, 10);
        let output_file = temp_dir.path().join("output.csv");
        
        // Create chunk with some invalid lines
        let chunk_path = temp_dir.path().join("chunk_0.csv");
        fs::write(&chunk_path, 
            "valid1@test.com,pass1,https://site.com\n\ninvalid_line_no_commas\nvalid2@test.com,pass2,https://site.com\n"
        ).unwrap();
        
        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: chunk_path,
                record_count: 2, // Only counting valid records
                file_size_bytes: 100,
                is_sorted: true,
                source_files: vec![],
            },
        ];
        
        let mut checkpoint = SortCheckpoint::new(
            vec![],
            output_file.clone(),
            temp_dir.path().to_path_buf(),
        );
        
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        // Should skip invalid lines
        let result = merger.merge_chunks(&chunks, &output_file, &mut checkpoint, shutdown_flag).await;
        assert!(result.is_ok());
        
        // Only valid records in output
        let output_content = fs::read_to_string(&output_file).unwrap();
        assert_eq!(output_content.lines().count(), 2);
        assert!(output_content.contains("valid1@test.com"));
        assert!(output_content.contains("valid2@test.com"));
        assert!(!output_content.contains("invalid_line"));
    }

    #[test]
    fn test_validate_chunks_multiple_errors() {
        let temp_dir = tempdir().unwrap();
        let merger = ChunkMerger::new(8192, true, 10);

        // Create one valid file
        let valid_path = temp_dir.path().join("valid.csv");
        fs::write(&valid_path, "data\n").unwrap();

        let chunks = vec![
            ChunkMetadata {
                chunk_id: 0,
                file_path: valid_path,
                record_count: 1,
                file_size_bytes: 5,
                is_sorted: true,
                source_files: vec![],
            },
            ChunkMetadata {
                chunk_id: 1,
                file_path: temp_dir.path().join("missing.csv"),
                record_count: 1,
                file_size_bytes: 5,
                is_sorted: true,
                source_files: vec![],
            },
        ];

        // Should fail on first error (missing file)
        let result = merger.validate_chunks(&chunks);
        assert!(result.is_err());
    }
}
