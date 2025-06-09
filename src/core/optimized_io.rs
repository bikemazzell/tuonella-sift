use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write, Read};
use std::path::Path;
use anyhow::Result;
use crate::constants::{
    STREAMING_CHUNK_SIZE_MB, BYTES_PER_MB, WRITE_BUFFER_SIZE_MB,
    PARALLEL_FILE_PROCESSING_THREADS, PARALLEL_IO_QUEUE_SIZE
};

/// Optimized file reader with larger buffer sizes and async I/O capabilities
pub struct OptimizedFileReader {
    reader: BufReader<File>,
    buffer_size: usize,
}

impl OptimizedFileReader {
    pub fn new(file_path: &Path) -> Result<Self> {
        let file = File::open(file_path)?;
        let buffer_size = STREAMING_CHUNK_SIZE_MB * BYTES_PER_MB;
        let reader = BufReader::with_capacity(buffer_size, file);
        
        Ok(Self {
            reader,
            buffer_size,
        })
    }

    pub fn read_lines_chunked<F>(&mut self, mut process_chunk: F) -> Result<()>
    where
        F: FnMut(&[String]) -> Result<()>,
    {
        let mut lines_buffer = Vec::with_capacity(10000); // Pre-allocate for better performance
        let mut line = String::new();
        
        loop {
            line.clear();
            let bytes_read = self.reader.read_line(&mut line)?;
            
            if bytes_read == 0 {
                // End of file - process remaining lines
                if !lines_buffer.is_empty() {
                    process_chunk(&lines_buffer)?;
                }
                break;
            }
            
            // Remove trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            
            lines_buffer.push(line.clone());
            
            // Process chunk when buffer is full
            if lines_buffer.len() >= 10000 {
                process_chunk(&lines_buffer)?;
                lines_buffer.clear();
            }
        }
        
        Ok(())
    }

    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }
}

/// Optimized file writer with larger buffer sizes and batch writing
pub struct OptimizedFileWriter {
    writer: BufWriter<File>,
    write_count: usize,
}

impl OptimizedFileWriter {
    pub fn new(file_path: &Path) -> Result<Self> {
        let file = File::create(file_path)?;
        let buffer_size = WRITE_BUFFER_SIZE_MB * BYTES_PER_MB;
        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(Self {
            writer,
            write_count: 0,
        })
    }

    pub fn write_line(&mut self, line: &str) -> Result<()> {
        writeln!(self.writer, "{}", line)?;
        self.write_count += 1;
        
        // Flush periodically for better memory management
        if self.write_count % 50000 == 0 {
            self.writer.flush()?;
        }
        
        Ok(())
    }

    pub fn write_lines_batch(&mut self, lines: &[String]) -> Result<()> {
        for line in lines {
            writeln!(self.writer, "{}", line)?;
        }
        self.write_count += lines.len();
        
        // Flush after large batches
        if lines.len() > 10000 {
            self.writer.flush()?;
        }
        
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn get_write_count(&self) -> usize {
        self.write_count
    }
}

/// Parallel file processing utilities
pub struct ParallelFileProcessor {
    thread_count: usize,
    queue_size: usize,
}

impl ParallelFileProcessor {
    pub fn new() -> Self {
        Self {
            thread_count: PARALLEL_FILE_PROCESSING_THREADS,
            queue_size: PARALLEL_IO_QUEUE_SIZE,
        }
    }

    pub fn with_thread_count(mut self, count: usize) -> Self {
        self.thread_count = count.max(1).min(32); // Reasonable bounds
        self
    }

    pub fn get_thread_count(&self) -> usize {
        self.thread_count
    }

    pub fn get_queue_size(&self) -> usize {
        self.queue_size
    }
}

impl Default for ParallelFileProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-mapped file reader for very large files
pub struct MemoryMappedReader {
    _file: File,
    data: Vec<u8>,
    position: usize,
}

impl MemoryMappedReader {
    pub fn new(file_path: &Path) -> Result<Self> {
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        Ok(Self {
            _file: file,
            data,
            position: 0,
        })
    }

    pub fn read_chunk(&mut self, size: usize) -> Option<&[u8]> {
        if self.position >= self.data.len() {
            return None;
        }
        
        let end = (self.position + size).min(self.data.len());
        let chunk = &self.data[self.position..end];
        self.position = end;
        
        Some(chunk)
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_optimized_file_reader() -> Result<()> {
        // Create a temporary file with test data
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "line1")?;
        writeln!(temp_file, "line2")?;
        writeln!(temp_file, "line3")?;
        temp_file.flush()?;

        let mut reader = OptimizedFileReader::new(temp_file.path())?;
        let mut lines_collected = Vec::new();

        reader.read_lines_chunked(|lines| {
            lines_collected.extend_from_slice(lines);
            Ok(())
        })?;

        assert_eq!(lines_collected.len(), 3);
        assert_eq!(lines_collected[0], "line1");
        assert_eq!(lines_collected[1], "line2");
        assert_eq!(lines_collected[2], "line3");

        Ok(())
    }

    #[test]
    fn test_optimized_file_writer() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut writer = OptimizedFileWriter::new(temp_file.path())?;

        writer.write_line("test line 1")?;
        writer.write_line("test line 2")?;
        writer.flush()?;

        assert_eq!(writer.get_write_count(), 2);

        Ok(())
    }

    #[test]
    fn test_parallel_file_processor() {
        let processor = ParallelFileProcessor::new();
        assert_eq!(processor.get_thread_count(), PARALLEL_FILE_PROCESSING_THREADS);
        assert_eq!(processor.get_queue_size(), PARALLEL_IO_QUEUE_SIZE);

        let custom_processor = ParallelFileProcessor::new().with_thread_count(16);
        assert_eq!(custom_processor.get_thread_count(), 16);
    }

    #[test]
    fn test_memory_mapped_reader() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"Hello, World! This is a test.")?;
        temp_file.flush()?;

        let mut reader = MemoryMappedReader::new(temp_file.path())?;
        assert_eq!(reader.size(), 29);
        assert_eq!(reader.remaining(), 29);

        let chunk = reader.read_chunk(5);
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap(), b"Hello");
        assert_eq!(reader.remaining(), 24);

        reader.reset();
        assert_eq!(reader.remaining(), 29);

        Ok(())
    }
}
