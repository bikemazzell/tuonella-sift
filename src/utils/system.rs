use sysinfo::{System, Pid};
use std::time::{Duration, Instant};
use crate::constants::BYTES_PER_GB;

/// Get information about the system's memory
///
/// Returns (total_memory_gb, available_memory_gb)
pub fn get_memory_info() -> (f64, f64) {
    let mut system = System::new_all();
    system.refresh_all();

    let total_memory = system.total_memory() as f64 / BYTES_PER_GB as f64;
    let available_memory = system.available_memory() as f64 / BYTES_PER_GB as f64;

    (total_memory, available_memory)
}

/// Get the memory usage of the current process
///
/// Returns memory usage in bytes
pub fn get_process_memory_usage() -> usize {
    let mut system = System::new_all();
    system.refresh_all();

    let pid = Pid::from_u32(std::process::id());
    if let Some(process) = system.process(pid) {
        process.memory() as usize * 1024 // Convert KB to bytes
    } else {
        0
    }
}

/// Check if the system has enough memory available
///
/// Ensures there's at least the required amount of memory available
pub fn check_memory_available(required_gb: f64) -> bool {
    let (_, available_gb) = get_memory_info();
    available_gb >= required_gb
}

/// Format a duration as a human-readable string
///
/// Formats the duration in the form "HH:MM:SS"
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format bytes as a human-readable string
///
/// Formats bytes as KB, MB, GB, etc.
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let bytes = bytes as f64;
    if bytes < KB {
        format!("{:.0} B", bytes)
    } else if bytes < MB {
        format!("{:.2} KB", bytes / KB)
    } else if bytes < GB {
        format!("{:.2} MB", bytes / MB)
    } else if bytes < TB {
        format!("{:.2} GB", bytes / GB)
    } else {
        format!("{:.2} TB", bytes / TB)
    }
}

/// Estimate the remaining time for a task
///
/// Based on the amount of work done and the time elapsed
pub fn estimate_remaining_time(
    start_time: Instant,
    total_work: usize,
    work_done: usize,
) -> Option<Duration> {
    if work_done == 0 {
        return None;
    }

    let elapsed = start_time.elapsed();
    let progress = work_done as f64 / total_work as f64;
    if progress <= 0.0 {
        return None;
    }

    let total_estimated = elapsed.as_secs_f64() / progress;
    let remaining_secs = total_estimated - elapsed.as_secs_f64();
    
    if remaining_secs <= 0.0 {
        return Some(Duration::from_secs(0));
    }
    
    Some(Duration::from_secs_f64(remaining_secs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_memory_info() {
        let (total, available) = get_memory_info();
        assert!(total > 0.0, "Total memory should be positive");
        assert!(available > 0.0, "Available memory should be positive");
        assert!(available <= total, "Available memory should not exceed total");
    }

    #[test]
    fn test_process_memory_usage() {
        let usage = get_process_memory_usage();
        assert!(usage > 0, "Process memory usage should be positive");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
        assert_eq!(format_duration(Duration::from_secs(61)), "00:01:01");
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
        assert_eq!(format_bytes(1536 * 1024 * 1024), "1.50 GB");
    }

    #[test]
    fn test_estimate_remaining_time() {
        let start = Instant::now();
        
        // No progress yet
        assert_eq!(estimate_remaining_time(start, 100, 0), None);
        
        // Some progress
        sleep(Duration::from_millis(100));
        let remaining = estimate_remaining_time(start, 100, 10);
        assert!(remaining.is_some());
        
        // Complete
        let remaining = estimate_remaining_time(start, 100, 100);
        assert_eq!(remaining.unwrap(), Duration::from_secs(0));
    }
} 