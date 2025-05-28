use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter};
use walkdir::WalkDir;
use std::fs::File;
use std::io::{BufRead, BufReader};
use glob::Pattern;

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
}
