use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use anyhow::Result;

/// Discover CSV files in a directory
///
/// Returns a sorted list of CSV file paths
pub fn discover_csv_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut csv_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().to_lowercase() == "csv" {
                    csv_files.push(path);
                }
            }
        }
    }

    csv_files.sort();
    Ok(csv_files)
}

/// Write a CSV file
///
/// Takes a list of records and writes them to a CSV file
pub fn write_csv<W: Write>(writer: &mut BufWriter<W>, records: &[Vec<String>]) -> Result<()> {
    for record in records {
        let line = record.join(",");
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}

/// Count lines in a file efficiently
///
/// Uses a buffer to read chunks of the file and count newlines
pub fn count_lines(path: &Path) -> Result<usize> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let count = reader.lines().count();
    Ok(count)
}

/// Sample random lines from a file
///
/// Useful for field detection and analysis
pub fn sample_lines(path: &Path, sample_size: usize) -> Result<Vec<String>> {
    let total_lines = count_lines(path)?;
    if total_lines == 0 {
        return Ok(Vec::new());
    }

    let sample_rate = if total_lines <= sample_size {
        1.0 // Sample all lines if there are fewer than requested
    } else {
        sample_size as f64 / total_lines as f64
    };

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::with_capacity(sample_size);

    for line in reader.lines() {
        if rand::random::<f64>() < sample_rate && samples.len() < sample_size {
            if let Ok(line) = line {
                samples.push(line);
            }
        }
    }

    Ok(samples)
}

/// Create a new temporary file with a unique name
pub fn create_temp_file(temp_dir: &Path, prefix: &str) -> Result<PathBuf> {
    let mut counter = 0;
    loop {
        let temp_path = temp_dir.join(format!("{}_{}.tmp", prefix, counter));
        if !temp_path.exists() {
            // Just create an empty file to reserve the name
            File::create(&temp_path)?;
            return Ok(temp_path);
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_discover_csv_files() -> Result<()> {
        let temp_dir = tempdir()?;

        // Create test files with different extensions
        let csv1 = temp_dir.path().join("file1.csv");
        let csv2 = temp_dir.path().join("file2.CSV");
        let txt = temp_dir.path().join("file3.txt");

        File::create(&csv1)?;
        File::create(&csv2)?;
        File::create(&txt)?;

        let files = discover_csv_files(temp_dir.path())?;

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f == &csv1));
        assert!(files.iter().any(|f| f == &csv2));

        Ok(())
    }

    #[test]
    fn test_count_lines() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.txt");

        {
            let file = File::create(&file_path)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "Line 1")?;
            writeln!(writer, "Line 2")?;
            writeln!(writer, "Line 3")?;
            writer.flush()?;
        }

        let count = count_lines(&file_path)?;
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_sample_lines() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.txt");

        // Create a file with 100 lines
        {
            let file = File::create(&file_path)?;
            let mut writer = BufWriter::new(file);
            for i in 0..100 {
                writeln!(writer, "Line {}", i)?;
            }
            writer.flush()?;
        }

        // Sample 10 lines (random sampling, so we expect approximately 10 but allow some variance)
        let samples = sample_lines(&file_path, 10)?;
        assert!(samples.len() <= 10, "Should not exceed requested sample size");
        assert!(samples.len() >= 5, "Should get at least half the requested samples");

        // Sample more lines than in the file
        let samples = sample_lines(&file_path, 200)?;
        assert_eq!(samples.len(), 100);

        Ok(())
    }

    #[test]
    fn test_create_temp_file() -> Result<()> {
        let temp_dir = tempdir()?;

        let temp_file1 = create_temp_file(temp_dir.path(), "test")?;
        let temp_file2 = create_temp_file(temp_dir.path(), "test")?;

        assert!(temp_file1.exists());
        assert!(temp_file2.exists());
        assert_ne!(temp_file1, temp_file2);

        Ok(())
    }
}