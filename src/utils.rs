use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter};
use walkdir::WalkDir;
use std::fs::File;
use std::io::{BufRead, BufReader};
use glob::Pattern;
// use crate::constants::{MIN_FIELDS_FOR_PROCESSING};

pub fn setup_logging(verbosity: &str, _log_file: &str) -> Result<()> {
    let level = match verbosity {
        "silent" => Level::ERROR,
        "normal" => Level::INFO,
        "verbose" => Level::DEBUG,
        _ => Level::INFO,
    };

    let filter = EnvFilter::from_default_env()
        .add_directive(format!("dataset_dedup={}", level).parse()?);

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    Ok(())
}

pub fn discover_csv_files<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>> {
    let mut csv_files = Vec::new();
    let dir_path = directory.as_ref();
    let ignore_file_path = dir_path.join(".siftignore");
    let ignore_patterns = load_ignore_patterns(&ignore_file_path).unwrap_or_else(|_| Vec::new());

    for entry in WalkDir::new(dir_path).follow_links(false) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if is_ignored(filename, &ignore_patterns) {
                    continue;
                }
            }

            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().to_lowercase() == "csv" {
                    csv_files.push(path.to_path_buf());
                }
            }
        }
    }
    
    csv_files.sort_by_key(|path| {
        std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0)
    });
    
    csv_files.reverse();
    
    Ok(csv_files)
}

fn load_ignore_patterns(path: &Path) -> Result<Vec<Pattern>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut patterns = Vec::new();
    for line in reader.lines() {
        let line = line?.trim().to_string();
        if !line.is_empty() && !line.starts_with('#') {
            match Pattern::new(&line) {
                Ok(pattern) => patterns.push(pattern),
                Err(e) => eprintln!("Warning: Invalid glob pattern in ignore file: {} - {}", line, e),
            }
        }
    }
    Ok(patterns)
}

fn is_ignored(filename: &str, patterns: &[Pattern]) -> bool {
    for pattern in patterns {
        if pattern.matches(filename) {
            return true;
        }
    }
    false
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{:.1}h", seconds / 3600.0)
    }
}

pub fn estimate_remaining_time(
    processed: usize,
    total: usize,
    elapsed_seconds: f64,
) -> Option<f64> {
    if processed == 0 || elapsed_seconds <= 0.0 {
        return None;
    }
    
    let rate = processed as f64 / elapsed_seconds;
    let remaining = total.saturating_sub(processed) as f64;
    
    Some(remaining / rate)
}

/* /// Enhanced record qualification check
pub fn is_record_qualifies_for_processing(record: &csv::StringRecord) -> bool {
    // Check if record has minimum number of fields
    if record.len() < MIN_FIELDS_FOR_PROCESSING {
        return false;
    }
    
    // Check if record has any non-empty, non-whitespace content
    let has_content = record.iter().any(|field| !field.trim().is_empty());
    if !has_content {
        return false;
    }
    
    // Check if record has at least one field with substantial content (more than just punctuation)
    let has_substantial_content = record.iter().any(|field| {
        let trimmed = field.trim();
        trimmed.len() > 2 && trimmed.chars().any(|c| c.is_alphanumeric())
    });
    
    has_substantial_content
} */

/* /// Check if a line has proper delimiters (for pre-validation)
pub fn line_has_delimiters(line: &str, expected_delimiter_count: usize) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    
    let comma_count = line.matches(',').count();
    let semicolon_count = line.matches(';').count();
    let tab_count = line.matches('\t').count();
    let pipe_count = line.matches('|').count();
    
    let max_delimiter_count = comma_count.max(semicolon_count).max(tab_count).max(pipe_count);
    
    // Should have at least the expected number of delimiters (or close to it)
    max_delimiter_count >= expected_delimiter_count.saturating_sub(1)
} */

/* /// Check if input files are sorted (optimization hint)
pub fn detect_sorted_input(sample_records: &[Vec<String>], user_idx: usize) -> SortOrder {
    if sample_records.len() < 10 {
        return SortOrder::Unknown;
    }
    
    let mut ascending_count = 0;
    let mut descending_count = 0;
    let mut total_comparisons = 0;
    
    for window in sample_records.windows(2) {
        if let (Some(current), Some(next)) = (window[0].get(user_idx), window[1].get(user_idx)) {
            total_comparisons += 1;
            match current.cmp(next) {
                std::cmp::Ordering::Less => ascending_count += 1,
                std::cmp::Ordering::Greater => descending_count += 1,
                std::cmp::Ordering::Equal => {} // Neutral
            }
        }
    }
    
    if total_comparisons == 0 {
        return SortOrder::Unknown;
    }
    
    let ascending_ratio = ascending_count as f64 / total_comparisons as f64;
    let descending_ratio = descending_count as f64 / total_comparisons as f64;
    
    // Consider sorted if 70% or more comparisons follow the same order
    if ascending_ratio >= 0.7 {
        SortOrder::Ascending
    } else if descending_ratio >= 0.7 {
        SortOrder::Descending
    } else {
        SortOrder::Random
    }
} */

/* #[derive(Debug, Clone, PartialEq)]
pub enum SortOrder {
    Ascending,
    Descending,
    Random,
    Unknown,
} */

/* /// Detect the delimiter used in a CSV file
pub fn detect_delimiter(reader: &mut BufReader<File>) -> Result<u8> {
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    
    // Reset the reader position
    reader.seek(std::io::SeekFrom::Start(0))?;
    
    let line = first_line.trim();
    if line.is_empty() {
        return Ok(b','); // Default to comma
    }
    
    let delimiters = [(b',', line.matches(',').count()),
                      (b';', line.matches(';').count()),
                      (b'\t', line.matches('\t').count()),
                      (b'|', line.matches('|').count())];
    
    let (best_delimiter, _) = delimiters
        .iter()
        .max_by_key(|(_, count)| count)
        .unwrap_or(&(b',', 0));
    
    Ok(*best_delimiter)
} */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30.0s");
        assert_eq!(format_duration(90.0), "1.5m");
        assert_eq!(format_duration(3660.0), "1.0h");
    }

    #[test]
    fn test_discover_csv_files_with_ignore() {
        let test_dir = tempfile::tempdir().unwrap();
        let dir_path = test_dir.path();

        File::create(dir_path.join("file1.csv")).unwrap();
        File::create(dir_path.join("file2.txt")).unwrap(); // Should be ignored by extension
        File::create(dir_path.join("ignored_by_pattern.csv")).unwrap();
        File::create(dir_path.join("another.csv")).unwrap();

        let ignore_file_path = dir_path.join(".siftignore");
        std::fs::write(&ignore_file_path, "ignored_*.csv\n#comment\n").unwrap();

        let files = discover_csv_files(dir_path).unwrap();
        
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p.file_name().unwrap_or_default() == "file1.csv"));
        assert!(files.iter().any(|p| p.file_name().unwrap_or_default() == "another.csv"));
        assert!(!files.iter().any(|p| p.file_name().unwrap_or_default() == "ignored_by_pattern.csv"));

        // Cleanup
        std::fs::remove_file(ignore_file_path).unwrap();
    }

    #[test]
    fn test_is_ignored() {
        let patterns = vec![Pattern::new("*.log").unwrap(), Pattern::new("temp_*").unwrap()];
        assert!(is_ignored("debug.log", &patterns));
        assert!(is_ignored("temp_file.txt", &patterns));
        assert!(!is_ignored("data.csv", &patterns));
    }

    /* fn test_is_record_qualifies_for_processing() {
        let mut record = csv::StringRecord::new();
        record.push_field("hello");
        record.push_field("world");
        record.push_field("test");
        assert!(is_record_qualifies_for_processing(&record));

        let mut record_with_empty = csv::StringRecord::new();
        record_with_empty.push_field("");
        record_with_empty.push_field("world");
        record_with_empty.push_field("test");
        assert!(is_record_qualifies_for_processing(&record_with_empty));

        let mut record_all_empty_fields = csv::StringRecord::new();
        record_all_empty_fields.push_field("");
        record_all_empty_fields.push_field("   ");
        record_all_empty_fields.push_field("\t\n");
        assert!(!is_record_qualifies_for_processing(&record_all_empty_fields));

        let mut record_only_whitespace = csv::StringRecord::new();
        record_only_whitespace.push_field("   ");
        record_only_whitespace.push_field("   ");
        record_only_whitespace.push_field("   ");
        assert!(!is_record_qualifies_for_processing(&record_only_whitespace));

        let mut record_single_printable = csv::StringRecord::new();
        record_single_printable.push_field("abc");
        record_single_printable.push_field("def");
        record_single_printable.push_field("ghi");
        assert!(is_record_qualifies_for_processing(&record_single_printable));

        let empty_record = csv::StringRecord::new(); // Completely empty record (e.g. from an empty line)
        assert!(!is_record_qualifies_for_processing(&empty_record));
    } */
}
