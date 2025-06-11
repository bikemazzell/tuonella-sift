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
}
