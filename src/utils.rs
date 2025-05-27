use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter};
use walkdir::WalkDir;

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
    
    for entry in WalkDir::new(directory).follow_links(false) {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
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
} 