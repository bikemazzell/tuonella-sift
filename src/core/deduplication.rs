use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use anyhow::Result;
use crate::config::model::{Config, DeduplicationConfig};
use crate::core::record::Record;
use crate::core::validation::{parse_csv_line, detect_field_positions, detect_field_positions_with_config, is_valid_line_with_config, EMAIL_REGEX, DELIMITER_REGEX};
use crate::core::memory_manager::MemoryManager;
use crate::core::performance_monitor::PerformanceMonitor;
use crate::utils::system::{get_process_memory_usage, get_memory_info};
use crate::constants::{
    VALIDATION_ERRORS_FILENAME, TEMP_FILE_PREFIX, FINAL_DEDUPLICATED_FILENAME,
    MIN_FIELD_COUNT, BYTES_PER_MB, PROTOCOL_HTTP, PROTOCOL_HTTPS, PROTOCOL_ANDROID,
    PROTOCOL_FTP, PROTOCOL_MAILTO, CORE_FIELD_COUNT, EXTRA_FIELDS_START_INDEX
};
#[cfg(feature = "cuda")]
use crate::constants::{
    DEFAULT_USERNAME_HEADER, DEFAULT_PASSWORD_HEADER, DEFAULT_URL_HEADER,
    GPU_CHUNK_PROCESSING_BATCH_SIZE, GPU_TEMP_FILE_READ_CHUNK_SIZE_MB
};
#[cfg(feature = "cuda")]
use crate::cuda::processor::{CudaProcessor, CudaRecord};

#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    pub total_records: usize,
    pub unique_records: usize,
    pub duplicates_removed: usize,
    pub processing_time_seconds: f64,
    pub files_processed: usize,
    pub invalid_records: usize,
}

/// Process CSV files with algorithm-compliant streaming and RAM buffering
///
/// This function implements the exact algorithm from docs/algorithm.md:
/// 1. Stream files line by line with pre-validation (CPU)
/// 2. Use RAM buffer to accumulate valid lines
/// 3. Write to temporary files when buffer is full
/// 4. Use MemoryManager for proper buffer allocation and monitoring
///
/// Returns a list of temporary file paths
pub fn process_csv_files_with_algorithm_streaming(
    input_dir: &Path,
    config: &Config,
    memory_manager: &mut MemoryManager,
    verbose: bool,
) -> Result<Vec<PathBuf>> {
    let mut temp_files = Vec::new();
    let csv_files = super::validation::discover_csv_files(input_dir)?;

    // Create temporary directory if it doesn't exist
    fs::create_dir_all(&config.io.temp_directory)?;

    // Create error log file
    let error_log_path = Path::new(&config.io.temp_directory).join(VALIDATION_ERRORS_FILENAME);
    let error_file = File::create(&error_log_path)?;
    let mut error_writer = BufWriter::new(error_file);

    if verbose {
        println!("🔄 Processing {} CSV files with algorithm-compliant streaming...", csv_files.len());
        println!("📊 Memory Manager Status:");
        let stats = memory_manager.get_memory_stats()?;
        println!("{}", stats.format_summary());
    }

    for (i, csv_file) in csv_files.iter().enumerate() {
        if verbose {
            println!("📂 Processing file {}/{}: {}", i + 1, csv_files.len(), csv_file.display());
        }

        let temp_file = Path::new(&config.io.temp_directory)
            .join(format!("{}{}.csv", TEMP_FILE_PREFIX, i));

        process_single_csv_with_algorithm_streaming(
            csv_file,
            &temp_file,
            &mut error_writer,
            memory_manager,
            config,
            verbose
        )?;

        temp_files.push(temp_file);
    }

    error_writer.flush()?;
    if verbose {
        println!("📝 Error log written to: {}", error_log_path.display());
        println!("✅ Algorithm-compliant file processing complete!");
    }

    Ok(temp_files)
}

/// Legacy function for backward compatibility
///
/// This function maintains the old interface but uses config-based memory manager
pub fn process_csv_files_with_validation(
    input_dir: &Path,
    config: &Config,
    verbose: bool,
) -> Result<Vec<PathBuf>> {
    // Create a config-based memory manager
    let mut memory_manager = MemoryManager::from_config(config)?;

    // Delegate to the new algorithm-compliant implementation
    process_csv_files_with_algorithm_streaming(input_dir, config, &mut memory_manager, verbose)
}

/// Process a single CSV file with algorithm-compliant streaming and RAM buffering
///
/// This function implements the exact algorithm from docs/algorithm.md:
/// 1. Stream files line by line (read one line at a time)
/// 2. Line-by-Line Pre-Validation (CPU)
/// 3. Store valid lines in RAM buffer
/// 4. Write to temporary file when buffer is full
/// 5. Use MemoryManager for proper buffer management
fn process_single_csv_with_algorithm_streaming(
    input_path: &Path,
    output_path: &Path,
    error_writer: &mut BufWriter<File>,
    memory_manager: &mut MemoryManager,
    config: &Config,
    verbose: bool
) -> Result<()> {
    if verbose {
        println!("  📖 Streaming file: {}", input_path.display());
    }

    // Open the CSV file in streaming mode (read one line at a time)
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    let mut valid_lines = 0;
    let mut total_lines = 0;
    let mut invalid_lines = 0;
    let mut temp_file_count = 0;

    // Clear RAM buffer before starting
    memory_manager.clear_ram_buffer();

    // Stream files line by line as per algorithm with UTF-8 error handling
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(_e) => {
                // Handle UTF-8 encoding errors gracefully
                // Note: Console output commented out to reduce spam, but error logging preserved
                // if verbose {
                //     println!("    ⚠️ UTF-8 encoding error at line {}: {}", total_lines + 1, _e);
                // }
                // Skip this line and continue processing
                total_lines += 1;
                invalid_lines += 1;
                writeln!(error_writer, "{}:{}: UTF-8 encoding error - {}",
                    input_path.display(), total_lines, _e)?;
                continue;
            }
        };
        total_lines += 1;

        // Line-by-Line Pre-Validation (CPU) as per algorithm
        let mut skip_reason = None;

        // Skip the Line If: It contains no printable characters
        if !line.chars().any(|c| c.is_ascii_graphic()) {
            skip_reason = Some("no printable characters");
        }
        // Skip the Line If: It does not contain at least one delimiter
        else if !DELIMITER_REGEX.is_match(&line) {
            skip_reason = Some("no delimiter found");
        }
        // Skip the Line If: It does not contain a valid username (email or printable based on config)
        else if config.deduplication.email_username_only && !EMAIL_REGEX.is_match(&line) {
            skip_reason = Some("no email address found");
        }
        else if !config.deduplication.email_username_only && !is_valid_line_with_config(&line, false, config.deduplication.allow_two_field_lines) {
            skip_reason = Some("no valid username found");
        }
        // Perform more advanced validation on the CSV fields
        else {
            let fields = parse_csv_line(&line);

            if config.deduplication.allow_two_field_lines && fields.len() == 2 {
                let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, config.deduplication.email_username_only, true);
                if user_idx >= fields.len() || password_idx >= fields.len() {
                    skip_reason = Some("invalid field positions detected");
                }
            } else if fields.len() < MIN_FIELD_COUNT {
                skip_reason = Some("fewer than 3 fields");
            } else {
                let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, config.deduplication.email_username_only, config.deduplication.allow_two_field_lines);
                if user_idx >= fields.len() || password_idx >= fields.len() || url_idx >= fields.len() {
                    skip_reason = Some("invalid field positions detected");
                }
                else {
                    for (i, field) in fields.iter().enumerate() {
                        if i != url_idx &&
                            (field.starts_with(PROTOCOL_HTTP) ||
                            field.starts_with(PROTOCOL_HTTPS) ||
                            field.starts_with(PROTOCOL_ANDROID) ||
                            field.starts_with(PROTOCOL_FTP) ||
                            field.starts_with(PROTOCOL_MAILTO)) {
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

        // If the line passes all checks, store it in the RAM buffer (as per algorithm)
        let line_with_newline = format!("{}\n", line);
        let line_bytes = line_with_newline.as_bytes();

        // Check if adding this line would exceed RAM buffer capacity
        if !memory_manager.can_fit_in_ram_buffer(line_bytes.len()) {
            // Write Filtered Lines to a Temporary File (as per algorithm)
            if verbose && temp_file_count == 0 {
                println!("    💾 RAM buffer full, writing to temporary file...");
            }

            let buffer_contents = memory_manager.get_ram_buffer_contents();
            writer.write_all(buffer_contents)?;
            memory_manager.clear_ram_buffer();
            temp_file_count += 1;

            // Check memory pressure after buffer flush
            if memory_manager.check_memory_pressure()? {
                if verbose {
                    println!("    ⚠️ Memory pressure detected during processing");
                }
            }
        }

        // Add line to RAM buffer
        if memory_manager.add_to_ram_buffer(line_bytes)? {
            valid_lines += 1;
            memory_manager.record_processed(1);
        } else {
            // This shouldn't happen since we checked capacity above, but handle gracefully
            writeln!(error_writer, "{}:{}: buffer overflow - {}",
                input_path.display(), total_lines, line)?;
            invalid_lines += 1;
        }
    }

    // Write remaining buffer content (as per algorithm)
    let remaining_contents = memory_manager.get_ram_buffer_contents();
    if !remaining_contents.is_empty() {
        writer.write_all(remaining_contents)?;
        if verbose && temp_file_count > 0 {
            println!("    💾 Writing final buffer contents to temporary file");
        }
    }

    writer.flush()?;

    if verbose {
        println!("    ✅ Processed {} lines: {} valid, {} invalid",
                total_lines, valid_lines, invalid_lines);
        if temp_file_count > 0 {
            println!("    📁 Buffer flushed {} times during processing", temp_file_count);
        }
    }

    Ok(())
}

/// Process a single CSV file with validation
///
/// This function:
/// 1. Reads the file line by line
/// 2. Validates each line
/// 3. Writes valid lines to a temporary file
#[allow(dead_code)]
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
    let ram_buffer_size_bytes = chunk_size_mb * BYTES_PER_MB;
    let mut ram_buffer = Vec::with_capacity(ram_buffer_size_bytes);

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(_e) => {
                // Handle UTF-8 encoding errors gracefully
                // Note: Console output commented out to reduce spam, but error logging preserved
                // if verbose {
                //     println!("    ⚠️ UTF-8 encoding error at line {}: {}", total_lines + 1, _e);
                // }
                // Skip this line and continue processing
                total_lines += 1;
                invalid_lines += 1;
                writeln!(error_writer, "{}:{}: UTF-8 encoding error - {}",
                    input_path.display(), total_lines, _e)?;
                continue;
            }
        };
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

            if fields.len() < MIN_FIELD_COUNT {
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
                           (field.starts_with(PROTOCOL_HTTP) ||
                            field.starts_with(PROTOCOL_HTTPS) ||
                            field.starts_with(PROTOCOL_ANDROID) ||
                            field.starts_with(PROTOCOL_FTP) ||
                            field.starts_with(PROTOCOL_MAILTO)) {
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

/// The main deduplication function with performance monitoring
///
/// This function:
/// 1. Processes all temporary files
/// 2. Builds a deduplication map
/// 3. Writes unique records to the output file
/// 4. Monitors performance and resource usage
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

    // Initialize performance monitor
    let mut performance_monitor = PerformanceMonitor::new();
    let mut last_performance_report = Instant::now();
    let performance_report_interval = Duration::from_secs(config.performance.report_interval_seconds);

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write CSV header - we'll determine the header from the first record
    let mut header_written = false;

    if verbose {
        println!("Memory limit: {} records", config.processing.max_memory_records);
        if config.performance.enable_monitoring {
            println!("🔍 Performance monitoring enabled - reports every {} seconds", config.performance.report_interval_seconds);
        } else {
            println!("🔍 Performance monitoring disabled");
        }
    }

    for (i, temp_file) in temp_files.iter().enumerate() {
        if verbose {
            println!("Deduplicating file {}/{}: {}", i + 1, temp_files.len(), temp_file.display());
        }

        let _file_start_time = Instant::now();
        let file = File::open(temp_file)?;
        let reader = BufReader::new(file);

        let mut records_batch = Vec::new();
        let mut _file_records_processed = 0;

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(line) => line,
                Err(_e) => {
                    // Handle UTF-8 encoding errors gracefully
                    // Note: Console output commented out to reduce spam, but error handling preserved
                    // if verbose {
                    //     println!("    ⚠️ UTF-8 encoding error in temp file {}: {}", temp_file.display(), _e);
                    // }
                    stats.invalid_records += 1;
                    continue;
                }
            };
            stats.total_records += 1;

            // Parse the line to extract fields
            if let Some(record) = parse_line_to_record(&line, &config.deduplication) {
                records_batch.push(record);
                _file_records_processed += 1;

                // Process batch when it reaches chunk size
                if records_batch.len() >= config.processing.record_chunk_size {
                    let batch_start_time = Instant::now();

                    process_record_batch(
                        &mut records_batch,
                        &mut dedup_map,
                        &mut stats,
                        #[cfg(feature = "cuda")]
                        cuda_processor,
                        &config.deduplication,
                    )?;

                    let batch_processing_time = batch_start_time.elapsed();

                    // Add performance sample (only if monitoring is enabled)
                    if config.performance.enable_monitoring {
                        let memory_usage = get_process_memory_usage() as f64 / (1024.0 * 1024.0 * 1024.0); // Convert to GB
                        let memory_usage_percent = (memory_usage / (get_memory_info().0 * config.memory.memory_usage_percent as f64 / 100.0)) * 100.0;

                        #[cfg(feature = "cuda")]
                        let gpu_utilization = if let Some(processor) = cuda_processor {
                            processor.get_gpu_utilization_percent().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        #[cfg(not(feature = "cuda"))]
                        let gpu_utilization = 0.0;

                        performance_monitor.add_sample(
                            records_batch.capacity(),
                            batch_processing_time,
                            Duration::from_millis(0), // IO time (minimal for in-memory processing)
                            memory_usage_percent,
                            gpu_utilization,
                        )?;
                    }

                    // Check memory limit and flush if necessary
                    if dedup_map.len() >= config.processing.max_memory_records {
                        if verbose {
                            println!("  Memory limit reached ({} records), flushing to disk...", dedup_map.len());
                        }
                        flush_records_to_disk(&mut dedup_map, &mut writer)?;
                    }

                    // Performance reporting
                    if config.performance.enable_monitoring && last_performance_report.elapsed() >= performance_report_interval {
                        if verbose {
                            println!("\n📊 Performance Report:");
                            println!("{}", performance_monitor.format_performance_summary());
                        }
                        last_performance_report = Instant::now();
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
    for (i, record) in dedup_map.values().enumerate() {
        // Write header based on first record
        if i == 0 && !header_written {
            // Create header based on the number of fields in the first record
            let mut header_fields = vec!["username".to_string(), "password".to_string()];

            // Add additional field names for extra fields beyond the core 3
            if record.all_fields.len() > CORE_FIELD_COUNT {
                for j in EXTRA_FIELDS_START_INDEX..record.all_fields.len() {
                    header_fields.push(format!("field_{}", j + 1));
                }
            } else {
                header_fields.push("url".to_string());
            }

            writeln!(writer, "{}", header_fields.join(","))?;
            header_written = true;
        }

        // Write record with all fields preserved, but use normalized URL for the URL field
        let mut output_fields = vec![record.user.clone(), record.password.clone()];

        if record.all_fields.len() > 2 {
            // Replace the URL field (index 2) with normalized URL, keep extra fields as-is
            output_fields.push(record.normalized_url.clone());

            // Add any extra fields beyond the core 3
            for j in EXTRA_FIELDS_START_INDEX..record.all_fields.len() {
                output_fields.push(record.all_fields[j].clone());
            }
        } else {
            // Fallback: just add normalized URL
            output_fields.push(record.normalized_url.clone());
        }

        writeln!(writer, "{}", output_fields.join(","))?;
    }

    writer.flush()?;
    stats.unique_records = dedup_map.len();
    stats.duplicates_removed = stats.total_records - stats.unique_records;
    stats.processing_time_seconds = start_time.elapsed().as_secs_f64();

    // Final performance report
    if verbose && config.performance.enable_monitoring {
        println!("\n📊 Final Performance Report:");
        println!("{}", performance_monitor.format_performance_summary());

        let final_memory_usage = get_process_memory_usage() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("💾 Final memory usage: {:.2} GB ({:.1}% of configured limit)",
                final_memory_usage,
                (final_memory_usage / (get_memory_info().0 * config.memory.memory_usage_percent as f64 / 100.0)) * 100.0);
    }

    Ok(stats)
}

/// Process temporary files with GPU acceleration (Algorithm Step 2.2)
///
/// This function implements the exact algorithm from docs/algorithm.md Section 2.2:
/// 1. Read temporary files in chunks of RAM Buffer Size
/// 2. Transfer chunks to GPU buffer for parallel processing
/// 3. Use CUDA kernels for parallel processing (email and URL normalization)
/// 4. Parse lines into Record structs
/// 5. Pass processed records back to CPU for deduplication
#[cfg(feature = "cuda")]
pub fn process_temp_files_with_gpu(
    temp_files: &[PathBuf],
    output_path: &Path,
    config: &Config,
    memory_manager: &MemoryManager,
    cuda_processor: &CudaProcessor,
    verbose: bool,
) -> Result<ProcessingStats> {
    let start_time = Instant::now();
    let mut dedup_map: HashMap<String, Record> = HashMap::new();
    let mut stats = ProcessingStats::default();
    stats.files_processed = temp_files.len();

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write CSV header - we'll determine the header from the first record
    let mut header_written = false;

    if verbose {
        println!("🚀 Processing {} temporary files with GPU acceleration...", temp_files.len());
        println!("📊 GPU batch size: {}", cuda_processor.get_optimal_batch_size());
    }

    // Process each temporary file generated in step 2.1
    for (i, temp_file) in temp_files.iter().enumerate() {
        if verbose {
            println!("🔄 Processing temp file {}/{}: {}", i + 1, temp_files.len(), temp_file.display());
        }

        process_single_temp_file_with_gpu(
            temp_file,
            &mut dedup_map,
            &mut stats,
            config,
            memory_manager,
            cuda_processor,
            verbose,
        )?;

        // Check memory limit and flush if necessary
        if dedup_map.len() >= config.processing.max_memory_records {
            if verbose {
                println!("  💾 Memory limit reached ({} records), flushing to disk...", dedup_map.len());
            }
            flush_records_to_disk(&mut dedup_map, &mut writer)?;
        }
    }

    // Write all unique records to output
    for (i, record) in dedup_map.values().enumerate() {
        // Write header based on first record
        if i == 0 && !header_written {
            // Create header based on the number of fields in the first record
            let mut header_fields = vec![DEFAULT_USERNAME_HEADER.to_string(), DEFAULT_PASSWORD_HEADER.to_string()];

            // Add additional field names for extra fields beyond the core 3
            if record.all_fields.len() > 3 {
                for j in 3..record.all_fields.len() {
                    header_fields.push(format!("field_{}", j + 1));
                }
            } else {
                header_fields.push(DEFAULT_URL_HEADER.to_string());
            }

            writeln!(writer, "{}", header_fields.join(","))?;
            header_written = true;
        }

        // Write record with all fields preserved, but use normalized URL for the URL field
        let mut output_fields = vec![record.user.clone(), record.password.clone()];

        if record.all_fields.len() > 2 {
            // Replace the URL field (index 2) with normalized URL, keep extra fields as-is
            output_fields.push(record.normalized_url.clone());

            // Add any extra fields beyond the core 3
            for j in 3..record.all_fields.len() {
                output_fields.push(record.all_fields[j].clone());
            }
        } else {
            // Fallback: just add normalized URL
            output_fields.push(record.normalized_url.clone());
        }

        writeln!(writer, "{}", output_fields.join(","))?;
    }

    writer.flush()?;
    stats.unique_records = dedup_map.len();
    stats.duplicates_removed = stats.total_records - stats.unique_records;
    stats.processing_time_seconds = start_time.elapsed().as_secs_f64();

    if verbose {
        println!("✅ GPU processing complete!");
        println!("📈 Final stats: {} unique, {} duplicates removed, {:.2}s processing time",
                stats.unique_records, stats.duplicates_removed, stats.processing_time_seconds);
    }

    Ok(stats)
}

/// Process a single temporary file with GPU acceleration
///
/// This implements the algorithm's chunk-based GPU processing:
/// 1. Read file in chunks of RAM Buffer Size
/// 2. Convert lines to CudaRecord structs
/// 3. Process chunks on GPU for normalization
/// 4. Insert into deduplication map
#[cfg(feature = "cuda")]
fn process_single_temp_file_with_gpu(
    temp_file: &Path,
    dedup_map: &mut HashMap<String, Record>,
    stats: &mut ProcessingStats,
    config: &Config,
    _memory_manager: &MemoryManager,
    cuda_processor: &CudaProcessor,
    verbose: bool,
) -> Result<()> {
    let file = File::open(temp_file)?;
    let reader = BufReader::new(file);

    let chunk_size_bytes = GPU_TEMP_FILE_READ_CHUNK_SIZE_MB * BYTES_PER_MB;
    let mut current_chunk_size = 0;
    let mut cuda_records = Vec::new();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(_e) => {
                // Handle UTF-8 encoding errors gracefully
                // Note: Console output commented out to reduce spam, but error handling preserved
                // if verbose {
                //     println!("    ⚠️ UTF-8 encoding error in GPU temp file {}: {}", temp_file.display(), _e);
                // }
                stats.invalid_records += 1;
                continue;
            }
        };
        stats.total_records += 1;
        current_chunk_size += line.len();

        // Parse line into CudaRecord for GPU processing
        if let Some(cuda_record) = parse_line_to_cuda_record(&line, &config.deduplication) {
            cuda_records.push(cuda_record);

            // Process chunk when it reaches RAM buffer size or GPU batch size
            if current_chunk_size >= chunk_size_bytes ||
               cuda_records.len() >= GPU_CHUNK_PROCESSING_BATCH_SIZE {

                process_cuda_records_chunk(
                    &mut cuda_records,
                    dedup_map,
                    stats,
                    cuda_processor,
                    &config.deduplication,
                    verbose,
                )?;

                current_chunk_size = 0;
            }
        } else {
            stats.invalid_records += 1;
        }
    }

    // Process remaining records in final chunk
    if !cuda_records.is_empty() {
        process_cuda_records_chunk(
            &mut cuda_records,
            dedup_map,
            stats,
            cuda_processor,
            &config.deduplication,
            verbose,
        )?;
    }

    Ok(())
}

/// Parse a CSV line into a CudaRecord for GPU processing
#[cfg(feature = "cuda")]
fn parse_line_to_cuda_record(line: &str, config: &DeduplicationConfig) -> Option<CudaRecord> {
    // Parse CSV line into fields
    let fields: Vec<String> = parse_csv_line(line);

    if fields.len() < MIN_FIELD_COUNT {
        return None;
    }

    // Check for valid username based on configuration
    if config.email_username_only {
        // Check if any field contains an email before proceeding
        let has_email = fields.iter().any(|field| EMAIL_REGEX.is_match(field));
        if !has_email {
            return None; // Skip rows without email addresses
        }
    } else {
        // For non-email mode, check if line has at least one field that could be a username
        let has_valid_username = fields.iter().any(|field| {
            use crate::core::validation::PRINTABLE_USERNAME_REGEX;
            PRINTABLE_USERNAME_REGEX.is_match(field) && !field.starts_with("http")
        });
        if !has_valid_username {
            return None;
        }
    }

    // Detect which fields are username, password, and URL
    let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, config.email_username_only, true);

    if user_idx >= fields.len() || password_idx >= fields.len() || url_idx >= fields.len() {
        return None;
    }

    // Validate the detected user field based on configuration
    if config.email_username_only {
        // Ensure the detected user field is actually an email (double-check)
        if !EMAIL_REGEX.is_match(&fields[user_idx]) {
            return None; // Skip if detected user field is not an email
        }
    } else {
        // For non-email mode, validate that it's a printable username
        use crate::core::validation::PRINTABLE_USERNAME_REGEX;
        if !PRINTABLE_USERNAME_REGEX.is_match(&fields[user_idx]) || fields[user_idx].starts_with("http") {
            return None; // Skip if not a valid printable username
        }
    }

    Some(CudaRecord {
        user: fields[user_idx].clone(),
        password: fields[password_idx].clone(),
        url: fields[url_idx].clone(),
        normalized_user: String::new(), // Will be filled by GPU
        normalized_url: String::new(),  // Will be filled by GPU
        field_count: fields.len(),      // Track original field count from CSV
        all_fields: fields.clone(),     // Store all original fields
    })
}

/// Process a chunk of CudaRecords on GPU and insert into deduplication map
#[cfg(feature = "cuda")]
fn process_cuda_records_chunk(
    cuda_records: &mut Vec<CudaRecord>,
    dedup_map: &mut HashMap<String, Record>,
    _stats: &mut ProcessingStats,
    cuda_processor: &CudaProcessor,
    config: &DeduplicationConfig,
    verbose: bool,
) -> Result<()> {
    if cuda_records.is_empty() {
        return Ok(());
    }

    if verbose && cuda_records.len() > 1000 {
        println!("    🚀 Processing {} records on GPU...", cuda_records.len());
    }

    // Process batch with CUDA (normalizes emails and URLs)
    cuda_processor.process_batch(cuda_records, config.case_sensitive_usernames)?;

    // Convert CudaRecords to Records and insert into deduplication map
    for cuda_record in cuda_records.drain(..) {
        let completeness_score = calculate_completeness_score(&cuda_record);
        let record = Record {
            user: cuda_record.user,
            password: cuda_record.password,
            url: cuda_record.url,
            normalized_user: cuda_record.normalized_user,
            normalized_url: cuda_record.normalized_url,
            completeness_score,
            field_count: cuda_record.field_count,  // Include field count for completeness comparison
            all_fields: cuda_record.all_fields,    // Preserve all original fields
        };

        let key = record.dedup_key();
        if verbose && record.user.contains("user@email.com") && record.password == "444" {
            println!("    🔍 Debug: Processing record: {},{},{} ({}f) -> key: {}",
                    record.user, record.password, record.url, record.field_count, key);
        }
        match dedup_map.get(&key) {
            Some(existing) => {
                if verbose && record.user.contains("user@email.com") && record.password == "444" {
                    println!("      🔄 Found existing record with same key, comparing completeness...");
                }
                if record.is_more_complete_than(existing) {
                    if verbose && record.user.contains("user@email.com") && record.password == "444" {
                        println!("      ✅ New record is more complete, replacing");
                    }
                    dedup_map.insert(key, record);
                } else {
                    if verbose && record.user.contains("user@email.com") && record.password == "444" {
                        println!("      ❌ Existing record is more complete, keeping existing");
                    }
                }
            }
            None => {
                if verbose && record.user.contains("user@email.com") && record.password == "444" {
                    println!("      ✅ New unique record, inserting");
                }
                dedup_map.insert(key, record);
            }
        }
    }

    Ok(())
}

/// Calculate completeness score for a CudaRecord
#[cfg(feature = "cuda")]
fn calculate_completeness_score(record: &CudaRecord) -> f32 {
    let mut score = 0.0;

    // Score for core fields
    if !record.user.is_empty() {
        score += 1.0 + (record.user.len() as f32 * 0.01);
    }
    if !record.password.is_empty() {
        score += 1.0 + (record.password.len() as f32 * 0.01);
    }
    if !record.url.is_empty() {
        score += 1.0 + (record.url.len() as f32 * 0.01);
    }

    // Add bonus for extra fields beyond the core 3
    if record.all_fields.len() > 3 {
        for field in record.all_fields.iter().skip(3) {
            if !field.trim().is_empty() {
                score += 1.0 + (field.len() as f32 * 0.01);
            }
        }
    }

    score
}

/// Complete Algorithm Pipeline (Sections 2.1, 2.2, 2.3, 3)
///
/// This function implements the complete algorithm from docs/algorithm.md:
/// 1. Stream files and pre-validate lines (Section 2.1)
/// 2. Transfer validated data to GPU (Section 2.2)
/// 3. Deduplication and hash map storage (Section 2.3)
/// 4. Final output (Section 3)
#[cfg(feature = "cuda")]
pub fn process_with_complete_algorithm(
    input_dir: &Path,
    output_path: &Path,
    config: &Config,
    memory_manager: &mut MemoryManager,
    cuda_processor: &CudaProcessor,
    verbose: bool,
) -> Result<ProcessingStats> {
    if verbose {
        println!("🚀 Starting Complete Algorithm Pipeline");
        println!("📂 Input directory: {}", input_dir.display());
        println!("📄 Output file: {}", output_path.display());
    }

    // Section 2.1: Stream Files and Pre-Validate Lines
    if verbose {
        println!("\n📋 Section 2.1: Stream Files and Pre-Validate Lines");
    }
    let temp_files = process_csv_files_with_algorithm_streaming(
        input_dir,
        config,
        memory_manager,
        verbose
    )?;

    if verbose {
        println!("✅ Section 2.1 Complete: Generated {} temporary files", temp_files.len());
    }

    // Sections 2.2 & 2.3: Transfer to GPU and Deduplication
    if verbose {
        println!("\n📋 Section 2.2 & 2.3: GPU Processing and Deduplication");
    }

    // Create a temporary file for intermediate results (Section 3 requirement)
    let temp_output_dir = Path::new(&config.io.temp_directory);
    let temp_output_file = temp_output_dir.join(FINAL_DEDUPLICATED_FILENAME);

    let stats = process_temp_files_with_gpu(
        &temp_files,
        &temp_output_file,
        config,
        memory_manager,
        cuda_processor,
        verbose
    )?;

    if verbose {
        println!("✅ Section 2.2 & 2.3 Complete: {} unique records processed", stats.unique_records);
    }

    // Section 3: Final Output
    if verbose {
        println!("\n📋 Section 3: Final Output");
    }

    finalize_output(&temp_output_file, output_path, verbose)?;

    if verbose {
        println!("✅ Section 3 Complete: Final output written to {}", output_path.display());
        println!("\n🎉 Complete Algorithm Pipeline Finished Successfully!");
        println!("📊 Final Statistics:");
        println!("  📁 Files processed: {}", stats.files_processed);
        println!("  📊 Total records: {}", stats.total_records);
        println!("  ✨ Unique records: {}", stats.unique_records);
        println!("  🗑️ Duplicates removed: {}", stats.duplicates_removed);
        println!("  ❌ Invalid records: {}", stats.invalid_records);
        println!("  ⏱️ Processing time: {:.2}s", stats.processing_time_seconds);
    }

    // Cleanup temporary files
    cleanup_temporary_files(&temp_files, &temp_output_file, verbose)?;

    Ok(stats)
}

/// CPU-only fallback for the complete algorithm pipeline
///
/// This function provides the same interface but uses CPU-only processing
/// when CUDA is not available
pub fn process_with_complete_algorithm_cpu_fallback(
    input_dir: &Path,
    output_path: &Path,
    config: &Config,
    memory_manager: &mut MemoryManager,
    verbose: bool,
) -> Result<ProcessingStats> {
    if verbose {
        println!("🔄 Starting Complete Algorithm Pipeline (CPU Fallback)");
        println!("📂 Input directory: {}", input_dir.display());
        println!("📄 Output file: {}", output_path.display());
    }

    // Section 2.1: Stream Files and Pre-Validate Lines
    if verbose {
        println!("\n📋 Section 2.1: Stream Files and Pre-Validate Lines");
    }
    let temp_files = process_csv_files_with_algorithm_streaming(
        input_dir,
        config,
        memory_manager,
        verbose
    )?;

    if verbose {
        println!("✅ Section 2.1 Complete: Generated {} temporary files", temp_files.len());
    }

    // Sections 2.2 & 2.3: CPU Processing and Deduplication
    if verbose {
        println!("\n📋 Section 2.2 & 2.3: CPU Processing and Deduplication");
    }

    // Create a temporary file for intermediate results
    let temp_output_dir = Path::new(&config.io.temp_directory);
    let temp_output_file = temp_output_dir.join(FINAL_DEDUPLICATED_FILENAME);

    let stats = deduplicate_records(
        &temp_files,
        &temp_output_file,
        config,
        #[cfg(feature = "cuda")]
        None, // No CUDA processor for CPU fallback
        verbose
    )?;

    if verbose {
        println!("✅ Section 2.2 & 2.3 Complete: {} unique records processed", stats.unique_records);
    }

    // Section 3: Final Output
    if verbose {
        println!("\n📋 Section 3: Final Output");
    }

    finalize_output(&temp_output_file, output_path, verbose)?;

    if verbose {
        println!("✅ Section 3 Complete: Final output written to {}", output_path.display());
        println!("\n🎉 Complete Algorithm Pipeline Finished Successfully!");
    }

    // Cleanup temporary files
    cleanup_temporary_files(&temp_files, &temp_output_file, verbose)?;

    Ok(stats)
}

/// Section 3: Final Output - Write final temporary file to user-specified output location
///
/// This function implements Section 3 of the algorithm:
/// 1. Takes the final temporary file with deduplicated records
/// 2. Writes it to the user-specified output location
/// 3. Ensures proper formatting and error handling
fn finalize_output(temp_output_file: &Path, final_output_path: &Path, verbose: bool) -> Result<()> {
    if verbose {
        println!("  📄 Moving final results from {} to {}",
                temp_output_file.display(), final_output_path.display());
    }

    // Ensure the output directory exists
    if let Some(parent) = final_output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Copy the temporary file to the final location
    // Using copy instead of move to preserve the temporary file for potential cleanup verification
    fs::copy(temp_output_file, final_output_path)?;

    if verbose {
        // Verify the output file
        let metadata = fs::metadata(final_output_path)?;
        println!("  ✅ Final output file created: {} bytes", metadata.len());

        // Show first few lines for verification
        let content = fs::read_to_string(final_output_path)?;
        let lines: Vec<&str> = content.lines().take(5).collect();
        println!("  📝 First {} lines of output:", lines.len());
        for (i, line) in lines.iter().enumerate() {
            println!("    {}: {}", i + 1, line);
        }
    }

    Ok(())
}

/// Cleanup temporary files after processing
///
/// This function removes temporary files created during processing
/// to free up disk space and maintain cleanliness
fn cleanup_temporary_files(temp_files: &[PathBuf], temp_output_file: &Path, verbose: bool) -> Result<()> {
    if verbose {
        println!("  🧹 Cleaning up {} temporary files...", temp_files.len() + 1);
    }

    // Remove temporary CSV files from Section 2.1
    for temp_file in temp_files {
        if temp_file.exists() {
            fs::remove_file(temp_file)?;
            if verbose {
                println!("    🗑️ Removed: {}", temp_file.display());
            }
        }
    }

    // Remove the final temporary file
    if temp_output_file.exists() {
        fs::remove_file(temp_output_file)?;
        if verbose {
            println!("    🗑️ Removed: {}", temp_output_file.display());
        }
    }

    if verbose {
        println!("  ✅ Cleanup complete");
    }

    Ok(())
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

    // Check for valid username based on configuration
    if config.email_username_only {
        // Check if any field contains an email before proceeding
        let has_email = fields.iter().any(|field| EMAIL_REGEX.is_match(field));
        if !has_email {
            return None; // Skip rows without email addresses
        }
    } else {
        // For non-email mode, check if line has at least one field that could be a username
        let has_valid_username = fields.iter().any(|field| {
            use crate::core::validation::PRINTABLE_USERNAME_REGEX;
            PRINTABLE_USERNAME_REGEX.is_match(field) && !field.starts_with("http")
        });
        if !has_valid_username {
            return None;
        }
    }

    // Detect which fields are username, password, and URL
    let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, config.email_username_only, config.allow_two_field_lines);

    if user_idx >= fields.len() || password_idx >= fields.len() || url_idx >= fields.len() {
        return None;
    }

    // Validate the detected user field based on configuration
    if config.email_username_only {
        // Ensure the detected user field is actually an email (double-check)
        if !EMAIL_REGEX.is_match(&fields[user_idx]) {
            return None; // Skip if detected user field is not an email
        }
    } else {
        // For non-email mode, validate that it's a printable username
        use crate::core::validation::PRINTABLE_USERNAME_REGEX;
        if !PRINTABLE_USERNAME_REGEX.is_match(&fields[user_idx]) || fields[user_idx].starts_with("http") {
            return None; // Skip if not a valid printable username
        }
    }

    Record::new_from_fields(
        fields.clone(),  // Pass all fields to preserve extra data
        user_idx,
        password_idx,
        url_idx,
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
                    field_count: r.field_count,  // Include field count
                    all_fields: r.all_fields.clone(),  // Include all fields
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
        // CPU fallback: normalize records manually when CUDA processor is not provided
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
        // Write record with all fields preserved, but use normalized URL for the URL field
        let mut output_fields = vec![record.user.clone(), record.password.clone()];

        if record.all_fields.len() > 2 {
            // Replace the URL field (index 2) with normalized URL, keep extra fields as-is
            output_fields.push(record.normalized_url.clone());

            // Add any extra fields beyond the core 3
            for j in 3..record.all_fields.len() {
                output_fields.push(record.all_fields[j].clone());
            }
        } else {
            // Fallback: just add normalized URL
            output_fields.push(record.normalized_url.clone());
        }

        writeln!(writer, "{}", output_fields.join(","))?;
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
    use crate::constants::{
        TEST_CHUNK_SIZE_MB, TEST_RECORD_CHUNK_SIZE, TEST_MAX_MEMORY_RECORDS
    };

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
            email_username_only: true,
            allow_two_field_lines: false,
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
            email_username_only: true,
            allow_two_field_lines: false,
        };

        // Create test records using the proper constructor
        let mut records = vec![
            Record::new(
                "user1@example.com".to_string(),
                "pass1".to_string(),
                "https://example.com".to_string(),
                config.case_sensitive_usernames,
            ).unwrap(),
            Record::new(
                "user1@example.com".to_string(), // Same user, same password, same URL (exact duplicate)
                "pass1".to_string(),             // Same password
                "https://example.com".to_string(), // Same URL
                config.case_sensitive_usernames,
            ).unwrap(),
            Record::new(
                "user2@example.com".to_string(),
                "pass2".to_string(),
                "https://another.com".to_string(),
                config.case_sensitive_usernames,
            ).unwrap(),
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
        assert_eq!(dedup_map.len(), 2); // Should be 2 unique records (duplicate removed)

        assert!(dedup_map.contains_key("user1@example.com|pass1|example.com"));
        assert!(dedup_map.contains_key("user2@example.com|pass2|another.com"));

        // Check that one of the duplicate records was kept for user1
        let user1_record = dedup_map.get("user1@example.com|pass1|example.com").unwrap();
        assert_eq!(user1_record.password, "pass1");

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_parse_line_to_cuda_record() {
        use crate::config::model::DeduplicationConfig;

        let config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: true,
            allow_two_field_lines: false,
        };

        // Valid line
        let line = "user@example.com,password123,https://example.com";
        let cuda_record = parse_line_to_cuda_record(line, &config);
        assert!(cuda_record.is_some());
        let cuda_record = cuda_record.unwrap();
        assert_eq!(cuda_record.user, "user@example.com");
        assert_eq!(cuda_record.password, "password123");
        assert_eq!(cuda_record.url, "https://example.com");
        assert_eq!(cuda_record.normalized_user, ""); // Will be filled by GPU
        assert_eq!(cuda_record.normalized_url, "");  // Will be filled by GPU

        // Invalid line (no email)
        let line = "not_an_email,password123,https://example.com";
        let cuda_record = parse_line_to_cuda_record(line, &config);
        assert!(cuda_record.is_none());

        // Invalid line (too few fields)
        let line = "user@example.com,password123";
        let cuda_record = parse_line_to_cuda_record(line, &config);
        assert!(cuda_record.is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_calculate_completeness_score() {
        use crate::cuda::processor::CudaRecord;

        let record = CudaRecord {
            user: "user@example.com".to_string(),
            password: "password123".to_string(),
            url: "https://example.com".to_string(),
            normalized_user: "user@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            field_count: 3,
            all_fields: vec!["user@example.com".to_string(), "password123".to_string(), "https://example.com".to_string()],
        };

        let score = calculate_completeness_score(&record);
        assert!(score > 3.0); // Should have base score of 3.0 plus length bonuses

        let empty_record = CudaRecord {
            user: "".to_string(),
            password: "".to_string(),
            url: "".to_string(),
            normalized_user: "".to_string(),
            normalized_url: "".to_string(),
            field_count: 3,
            all_fields: vec!["".to_string(), "".to_string(), "".to_string()],
        };

        let empty_score = calculate_completeness_score(&empty_record);
        assert_eq!(empty_score, 0.0);
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
            "user1@example.com,pass1,https://example.com", // Exact duplicate - should be removed
            "user3@example.com,pass3,https://third.com", // New user
        ])?;

        // Create config
        let config = Config {
            memory: crate::config::model::MemoryConfig {
                memory_usage_percent: 10, // 10% for testing
                auto_detect_memory: true,
            },
            processing: crate::config::model::ProcessingConfig {
                enable_cuda: false,
                chunk_size_mb: TEST_CHUNK_SIZE_MB,
                record_chunk_size: TEST_RECORD_CHUNK_SIZE,
                max_memory_records: TEST_MAX_MEMORY_RECORDS,
            },
            io: crate::config::model::IoConfig {
                temp_directory: temp_dir.path().to_string_lossy().to_string(),
                output_directory: temp_dir.path().to_string_lossy().to_string(),
            },
            deduplication: DeduplicationConfig {
                case_sensitive_usernames: false,
                normalize_urls: true,
                email_username_only: true,
                allow_two_field_lines: false,
            },
            logging: crate::config::model::LoggingConfig {
                verbosity: "normal".to_string(),
            },
            performance: crate::config::model::PerformanceConfig::default(),
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
        assert!(lines.iter().any(|l| l.contains("user1@example.com") && l.contains("pass1")));
        assert!(lines.iter().any(|l| l.contains("user2@example.com")));
        assert!(lines.iter().any(|l| l.contains("user3@example.com")));

        Ok(())
    }

    #[test]
    fn test_parse_line_to_record_with_printable_usernames() {
        // Test with email_username_only = false
        let config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: false,
            allow_two_field_lines: false,
        };

        // Valid line with printable username (not email)
        let line = "john_doe123,password456,https://example.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "john_doe123");
        assert_eq!(record.password, "password456");
        assert_eq!(record.url, "https://example.com");

        // Valid line with special characters in username
        let line = "user.name+tag,secret123,https://site.org";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "user.name+tag");

        // Valid line with email (should still work)
        let line = "user@example.com,password123,https://example.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "user@example.com");

        // Invalid line - username too short (empty)
        let line = ",password123,https://example.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_none());

        // Invalid line - no valid username (all fields are URLs or empty)
        let line = "https://baduser,https://another.com,https://example.com";
        let record = parse_line_to_record(line, &config);
        assert!(record.is_none());

        // Test with email_username_only = true (original behavior)
        let email_config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: true,
            allow_two_field_lines: false,
        };

        // Should reject non-email usernames
        let line = "john_doe123,password456,https://example.com";
        let record = parse_line_to_record(line, &email_config);
        assert!(record.is_none());

        // Should accept email usernames
        let line = "user@example.com,password123,https://example.com";
        let record = parse_line_to_record(line, &email_config);
        assert!(record.is_some());
    }

    #[test]
    fn test_field_detection_with_printable_usernames() {
        use crate::core::validation::detect_field_positions_with_config;

        // Test with email_username_only = false
        let fields = vec![
            "john_doe".to_string(),
            "secret123".to_string(),
            "https://example.com".to_string(),
        ];

        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
        assert_eq!(user_idx, 0);  // john_doe should be detected as username
        assert_eq!(password_idx, 1);
        assert_eq!(url_idx, 2);

        // Test with mixed fields
        let fields = vec![
            "https://site.com".to_string(),
            "username123".to_string(),
            "password456".to_string(),
        ];

        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
        assert_eq!(url_idx, 0);   // URL should be detected first
        assert_eq!(user_idx, 1);  // username123 should be detected as username
        assert_eq!(password_idx, 2);

        // Test with email_username_only = true (original behavior)
        let fields = vec![
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
        ];

        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true, false);
        assert_eq!(user_idx, 0);  // email should be detected as username
        assert_eq!(password_idx, 1);
        assert_eq!(url_idx, 2);
    }

    #[test]
    fn test_line_validation_with_config() {
        use crate::core::validation::is_valid_line_with_config;

        // Test with email_username_only = false
        let line = "john_doe,password123,https://example.com";
        assert!(is_valid_line_with_config(line, false, false));

        let line = "user@example.com,password123,https://example.com";
        assert!(is_valid_line_with_config(line, false, false));

        // Valid - has a valid username field (password123 is a valid printable username)
        let line = ",password123,https://example.com";
        assert!(is_valid_line_with_config(line, false, false));

        // Invalid - no valid username (all fields are URLs or empty)
        let line = ",https://password.com,https://example.com";
        assert!(!is_valid_line_with_config(line, false, false));

        // Test with email_username_only = true
        let line = "user@example.com,password123,https://example.com";
        assert!(is_valid_line_with_config(line, true, false));

        // Should be invalid for non-email usernames when email_username_only = true
        let line = "john_doe,password123,https://example.com";
        // Note: is_valid_line_with_config with email_username_only=true should check for email presence
        // This test depends on the EMAIL_REGEX matching the line
        assert!(!is_valid_line_with_config(line, true, false));
    }

    #[test]
    fn test_two_field_mode_username_password() {
        let config = DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: false,
            allow_two_field_lines: true,
        };

        // Two lines with same username, different passwords
        let line1 = "user@email.com:123";
        let line2 = "user@email.com:456";
        let fields1 = parse_csv_line(line1);
        let fields2 = parse_csv_line(line2);
        // Should detect username and password
        let (user_idx1, pass_idx1, url_idx1) = detect_field_positions_with_config(&fields1, false, true);
        let (user_idx2, pass_idx2, url_idx2) = detect_field_positions_with_config(&fields2, false, true);
        assert_eq!(fields1.len(), 2);
        assert_eq!(fields2.len(), 2);
        assert!(user_idx1 < 2 && pass_idx1 < 2);
        assert!(user_idx2 < 2 && pass_idx2 < 2);
        assert_ne!(fields1[pass_idx1], fields2[pass_idx2]);
        // Both should be valid lines
        assert!(is_valid_line_with_config(line1, false, true));
        assert!(is_valid_line_with_config(line2, false, true));
        // Simulate deduplication: both should be retained as unique
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        seen.insert(format!("{}|{}", fields1[user_idx1], fields1[pass_idx1]));
        seen.insert(format!("{}|{}", fields2[user_idx2], fields2[pass_idx2]));
        assert_eq!(seen.len(), 2);
    }
}