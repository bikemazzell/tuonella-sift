use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use anyhow::Result;
use crate::config::model::{Config, DeduplicationConfig};
use crate::core::record::Record;
use crate::core::validation::{parse_csv_line, detect_field_positions, EMAIL_REGEX, DELIMITER_REGEX};

#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaProcessor;

/// Represents processing statistics
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub processing_time_seconds: f64,
    pub files_processed: usize,
    pub invalid_records: usize,
}

/// Process CSV files with validation and write to temporary files
///
/// This function:
/// 1. Reads each CSV file line by line
/// 2. Validates each line using various criteria
/// 3. Writes valid lines to temporary files
///
/// Returns a list of temporary file paths
pub fn process_csv_files_with_validation(
    input_dir: &Path,
    config: &Config,
    verbose: bool,
) -> Result<Vec<PathBuf>> {
    let mut temp_files = Vec::new();
    let csv_files = super::validation::discover_csv_files(input_dir)?;

    // Create temporary directory if it doesn't exist
    fs::create_dir_all(&config.io.temp_directory)?;

    // Create error log file for corrupted/invalid lines
    let error_log_path = Path::new(&config.io.temp_directory).join("invalid_lines.log");
    let error_log_file = File::create(&error_log_path)?;
    let mut error_writer = BufWriter::new(error_log_file);

    if verbose {
        println!("Found {} CSV files to process", csv_files.len());
        println!("Invalid lines will be logged to: {}", error_log_path.display());
    }

    for (i, csv_file) in csv_files.iter().enumerate() {
        if verbose {
            println!("Processing file {}/{}: {}", i + 1, csv_files.len(), csv_file.display());
        }

        let temp_file = Path::new(&config.io.temp_directory)
            .join(format!("temp_{}.csv", i));

        process_single_csv_file_with_validation(
            csv_file,
            &temp_file,
            &mut error_writer,
            &config.processing.chunk_size_mb,
            verbose
        )?;
        temp_files.push(temp_file);
    }

    error_writer.flush()?;
    if verbose {
        println!("Error log written to: {}", error_log_path.display());
    }

    Ok(temp_files)
}

/// Process a single CSV file with validation
///
/// This function:
/// 1. Reads the file line by line
/// 2. Validates each line
/// 3. Writes valid lines to a temporary file
fn process_single_csv_file_with_validation(
    input_path: &Path,
    output_path: &Path,
    error_writer: &mut BufWriter<File>,
    chunk_size_mb: &usize,
    verbose: bool
) -> Result<()> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    let mut valid_lines = 0;
    let mut total_lines = 0;
    let mut invalid_lines = 0;
    let ram_buffer_size_bytes = chunk_size_mb * 1024 * 1024;
    let mut ram_buffer = Vec::with_capacity(ram_buffer_size_bytes);

    for line in reader.lines() {
        let line = line?;
        total_lines += 1;

        // Line-by-Line Pre-Validation
        let mut skip_reason = None;

        // Skip the Line If: It contains no printable characters
        if !line.chars().any(|c| c.is_ascii_graphic()) {
            skip_reason = Some("no printable characters");
        }
        // Skip the Line If: It does not contain at least one delimiter
        else if !DELIMITER_REGEX.is_match(&line) {
            skip_reason = Some("no delimiter found");
        }
        // Skip the Line If: It does not contain an email address
        else if !EMAIL_REGEX.is_match(&line) {
            skip_reason = Some("no email address found");
        }
        // Perform more advanced validation on the CSV fields
        else {
            let fields = parse_csv_line(&line);
            
            if fields.len() < 3 {
                skip_reason = Some("fewer than 3 fields");
            } else {
                // Check if the line has valid field positions
                let (user_idx, password_idx, url_idx) = detect_field_positions(&fields);
                
                if user_idx >= fields.len() || password_idx >= fields.len() || url_idx >= fields.len() {
                    skip_reason = Some("invalid field positions detected");
                }
                // Validate URL field - if we detect a URL protocol in a field that's not the URL field, skip it
                else {
                    for (i, field) in fields.iter().enumerate() {
                        if i != url_idx && 
                           (field.starts_with("http://") || 
                            field.starts_with("https://") || 
                            field.starts_with("android://") || 
                            field.starts_with("ftp://") || 
                            field.starts_with("mailto://")) {
                            skip_reason = Some("URL protocol found in non-URL field");
                            break;
                        }
                    }
                }
            }
        }

        if let Some(reason) = skip_reason {
            // Log invalid or skipped lines
            writeln!(error_writer, "{}:{}: {} - {}",
                input_path.display(), total_lines, reason, line)?;
            invalid_lines += 1;
            continue;
        }

        // If the line passes all checks, store it in the RAM buffer
        let line_bytes = line.as_bytes();

        // Check if adding this line would exceed RAM buffer capacity
        if ram_buffer.len() + line_bytes.len() + 1 > ram_buffer_size_bytes {
            // Write Filtered Lines to a Temporary File
            writer.write_all(&ram_buffer)?;
            ram_buffer.clear();
        }

        ram_buffer.extend_from_slice(line_bytes);
        ram_buffer.push(b'\n');
        valid_lines += 1;
    }

    // Write remaining buffer content
    if !ram_buffer.is_empty() {
        writer.write_all(&ram_buffer)?;
    }

    writer.flush()?;

    if verbose {
        println!("  {} valid lines, {} invalid lines out of {} total lines",
                valid_lines, invalid_lines, total_lines);
    }

    Ok(())
}

/// The main deduplication function
///
/// This function:
/// 1. Processes all temporary files
/// 2. Builds a deduplication map
/// 3. Writes unique records to the output file
pub fn deduplicate_records(
    temp_files: &[PathBuf],
    output_path: &Path,
    config: &Config,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<&CudaProcessor>,
    verbose: bool,
) -> Result<ProcessingStats> {
    let start_time = Instant::now();
    let mut dedup_map: HashMap<String, Record> = HashMap::new();
    let mut stats = ProcessingStats::default();
    stats.files_processed = temp_files.len();

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write CSV header
    writeln!(writer, "username,password,normalized_url")?;

    if verbose {
        println!("Memory limit: {} records", config.processing.max_memory_records);
    }

    for (i, temp_file) in temp_files.iter().enumerate() {
        if verbose {
            println!("Deduplicating file {}/{}: {}", i + 1, temp_files.len(), temp_file.display());
        }

        let file = File::open(temp_file)?;
        let reader = BufReader::new(file);

        let mut records_batch = Vec::new();

        for line in reader.lines() {
            let line = line?;
            stats.total_records += 1;

            // Parse the line to extract fields
            if let Some(record) = parse_line_to_record(&line, &config.deduplication) {
                records_batch.push(record);

                // Process batch when it reaches chunk size
                if records_batch.len() >= config.processing.record_chunk_size {
                    process_record_batch(
                        &mut records_batch,
                        &mut dedup_map,
                        &mut stats,
                        #[cfg(feature = "cuda")]
                        cuda_processor,
                        &config.deduplication,
                    )?;

                    // Check memory limit and flush if necessary
                    if dedup_map.len() >= config.processing.max_memory_records {
                        if verbose {
                            println!("  Memory limit reached ({} records), flushing to disk...", dedup_map.len());
                        }
                        flush_records_to_disk(&mut dedup_map, &mut writer)?;
                    }
                }
            } else {
                stats.invalid_records += 1;
            }
        }

        // Process remaining records in batch
        if !records_batch.is_empty() {
            process_record_batch(
                &mut records_batch,
                &mut dedup_map,
                &mut stats,
                #[cfg(feature = "cuda")]
                cuda_processor,
                &config.deduplication,
            )?;
        }
    }

    // Write all unique records to output
    for record in dedup_map.values() {
        // Use normalized URL for output, not the original URL
        writeln!(writer, "{},{},{}", record.user, record.password, record.normalized_url)?;
    }

    writer.flush()?;
    stats.unique_records = dedup_map.len();
    stats.duplicates_removed = stats.total_records - stats.unique_records;
    stats.processing_time_seconds = start_time.elapsed().as_secs_f64();

    Ok(stats)
}

/// Parse a CSV line into a Record
///
/// This function:
/// 1. Parses the line into fields
/// 2. Detects field positions
/// 3. Creates a Record
fn parse_line_to_record(line: &str, config: &DeduplicationConfig) -> Option<Record> {
    // Parse CSV line into fields
    let fields: Vec<String> = parse_csv_line(line);

    if fields.len() < 3 {
        return None;
    }

    // Check if any field contains an email before proceeding
    let has_email = fields.iter().any(|field| EMAIL_REGEX.is_match(field));
    if !has_email {
        return None; // Skip rows without email addresses
    }

    // Detect which fields are username, password, and URL
    let (user_idx, password_idx, url_idx) = detect_field_positions(&fields);

    if user_idx >= fields.len() || password_idx >= fields.len() || url_idx >= fields.len() {
        return None;
    }

    // Ensure the detected user field is actually an email (double-check)
    if !EMAIL_REGEX.is_match(&fields[user_idx]) {
        return None; // Skip if detected user field is not an email
    }

    Record::new(
        fields[user_idx].clone(),
        fields[password_idx].clone(),
        fields[url_idx].clone(),
        config.case_sensitive_usernames,
    )
}

/// Process a batch of records
///
/// This function:
/// 1. Normalizes URLs with CUDA if available
/// 2. Inserts records into the deduplication map
fn process_record_batch(
    records_batch: &mut Vec<Record>,
    dedup_map: &mut HashMap<String, Record>,
    _stats: &mut ProcessingStats,
    #[cfg(feature = "cuda")]
    cuda_processor: Option<&CudaProcessor>,
    config: &DeduplicationConfig,
) -> Result<()> {
    // Process records with CUDA if available
    #[cfg(feature = "cuda")]
    if let Some(processor) = cuda_processor {
        // Convert to the format expected by CUDA processor
        let mut cuda_records: Vec<crate::cuda::processor::CudaRecord> = records_batch
            .iter()
            .map(|r| {
                crate::cuda::processor::CudaRecord {
                    user: r.user.clone(),
                    password: r.password.clone(),
                    url: r.url.clone(),
                    normalized_user: String::new(),
                    normalized_url: String::new(),
                }
            })
            .collect();

        if !cuda_records.is_empty() {
            processor.process_batch(&mut cuda_records, config.case_sensitive_usernames)?;

            // Update our records with CUDA-processed data
            for (i, cuda_record) in cuda_records.iter().enumerate() {
                if i < records_batch.len() {
                    records_batch[i].normalized_user = cuda_record.normalized_user.clone();
                    records_batch[i].normalized_url = cuda_record.normalized_url.clone();
                }
            }
        }
    } else {
        // CPU fallback: normalize records manually
        #[cfg(not(feature = "cuda"))]
        for record in records_batch.iter_mut() {
            record.normalized_url = super::validation::normalize_url(&record.url);
            record.normalized_user = if config.case_sensitive_usernames {
                record.user.clone()
            } else {
                record.user.to_lowercase()
            };
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // CPU processing when CUDA is not available
        for record in records_batch.iter_mut() {
            record.normalized_url = super::validation::normalize_url(&record.url);
            record.normalized_user = if config.case_sensitive_usernames {
                record.user.clone()
            } else {
                record.user.to_lowercase()
            };
        }
    }

    // Insert records into deduplication map
    for record in records_batch.drain(..) {
        let key = record.dedup_key();

        match dedup_map.get(&key) {
            Some(existing) => {
                if record.is_more_complete_than(existing) {
                    dedup_map.insert(key, record);
                }
            }
            None => {
                dedup_map.insert(key, record);
            }
        }
    }

    Ok(())
}

/// Write records to disk and clear the map
fn flush_records_to_disk(
    dedup_map: &mut HashMap<String, Record>,
    writer: &mut BufWriter<File>,
) -> Result<()> {
    // Write all current records to output and clear the map to free memory
    for record in dedup_map.values() {
        writeln!(writer, "{},{},{}", record.user, record.password, record.normalized_url)?;
    }
    writer.flush()?;
    dedup_map.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_csv(path: &Path, lines: &[&str]) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for line in lines {
            writeln!(writer, "{}", line)?;
        }
        writer.flush()?;
        Ok(())
    }

    #[test]
    fn test_parse_line_to_record() {
        let config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
        };

        // Valid line
        let line = "user@example.com,password123,https://example.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "user@example.com");
        assert_eq!(record.password, "password123");
        assert_eq!(record.url, "https://example.com");
        assert_eq!(record.normalized_user, "user@example.com");
        assert_eq!(record.normalized_url, "example.com");

        // Line with fields in different order
        let line = "https://site.com,user@site.com,pass123";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "user@site.com");
        assert_eq!(record.password, "pass123");
        assert_eq!(record.url, "https://site.com");

        // Invalid line - no email
        let line = "username,password,site.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_none());

        // Invalid line - too few fields
        let line = "user@example.com,password";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_none());
    }

    #[test]
    fn test_process_record_batch() -> Result<()> {
        let config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
        };

        // Create test records
        let mut records = vec![
            Record {
                user: "user1@example.com".to_string(),
                password: "pass1".to_string(),
                url: "https://example.com".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                completeness_score: 0.0,
            },
            Record {
                user: "user1@example.com".to_string(), // Same user (duplicate)
                password: "betterpass".to_string(),     // Better password
                url: "http://example.com/login".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                completeness_score: 0.0,
            },
            Record {
                user: "user2@example.com".to_string(),
                password: "pass2".to_string(),
                url: "https://another.com".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                completeness_score: 0.0,
            },
        ];

        let mut dedup_map = HashMap::new();
        let mut stats = ProcessingStats::default();

        process_record_batch(
            &mut records,
            &mut dedup_map,
            &mut stats,
            #[cfg(feature = "cuda")]
            None,
            &config,
        )?;

        // Check results
        assert_eq!(dedup_map.len(), 2); // Should be 2 unique users
        assert!(dedup_map.contains_key("user1@example.com"));
        assert!(dedup_map.contains_key("user2@example.com"));

        // Check that the better record was kept for user1
        let user1_record = dedup_map.get("user1@example.com").unwrap();
        assert_eq!(user1_record.password, "betterpass");

        Ok(())
    }

    #[test]
    fn test_deduplicate_records() -> Result<()> {
        // Create temporary directory
        let temp_dir = tempdir()?;
        
        // Create test files
        let file1_path = temp_dir.path().join("test1.csv");
        let file2_path = temp_dir.path().join("test2.csv");
        let output_path = temp_dir.path().join("output.csv");

        create_test_csv(&file1_path, &[
            "user1@example.com,pass1,https://example.com",
            "user2@example.com,pass2,https://another.com",
        ])?;

        create_test_csv(&file2_path, &[
            "user1@example.com,betterpass,http://example.com/login", // Duplicate with better password
            "user3@example.com,pass3,https://third.com", // New user
        ])?;

        // Create config
        let config = Config {
            memory: crate::config::model::MemoryConfig {
                max_ram_usage_gb: 1,
                auto_detect_memory: true,
            },
            processing: crate::config::model::ProcessingConfig {
                enable_cuda: false,
                chunk_size_mb: 1,
                record_chunk_size: 10,
                max_memory_records: 1000,
            },
            io: crate::config::model::IoConfig {
                temp_directory: temp_dir.path().to_string_lossy().to_string(),
                output_directory: temp_dir.path().to_string_lossy().to_string(),
            },
            deduplication: DeduplicationConfig {
                case_sensitive_usernames: false,
                normalize_urls: true,
            },
            logging: crate::config::model::LoggingConfig {
                verbosity: "normal".to_string(),
            },
            #[cfg(feature = "cuda")]
            cuda: crate::config::model::CudaConfig::default(),
        };

        // Run deduplication
        let stats = deduplicate_records(
            &[file1_path, file2_path],
            &output_path,
            &config,
            #[cfg(feature = "cuda")]
            None,
            false,
        )?;

        // Check stats
        assert_eq!(stats.total_records, 4);
        assert_eq!(stats.unique_records, 3);
        assert_eq!(stats.duplicates_removed, 1);

        // Check output file
        let file = File::open(&output_path)?;
        let reader = BufReader::new(file);
        let lines: Result<Vec<String>, _> = reader.lines().collect();
        let lines = lines?;

        assert_eq!(lines.len(), 4); // Header + 3 unique records
        assert!(lines[0].contains("username") && lines[0].contains("password")); // Header
        assert!(lines.iter().any(|l| l.contains("user1@example.com") && l.contains("betterpass")));
        assert!(lines.iter().any(|l| l.contains("user2@example.com")));
        assert!(lines.iter().any(|l| l.contains("user3@example.com")));

        Ok(())
    }
} 