use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use tuonella_sift::core::{
    batch_writer::BatchWriter,
    performance_monitor::PerformanceMonitor,
    parallel_processor::ParallelProcessor,
    record::Record,
};

#[cfg(feature = "cuda")]
use tuonella_sift::core::double_buffer::DoubleBuffer;

/// Demonstration of Section 6: Performance Optimizations
///
/// This example showcases all the performance optimization features:
/// 1. Double buffering for overlapping I/O and GPU processing
/// 2. Performance monitoring and adaptive optimization
/// 3. Batch write optimizations
/// 4. Streaming and parallel processing optimizations
fn main() -> Result<()> {
    println!("🚀 Performance Optimizations Demo");
    println!("==================================");

    // Create test data directory
    let test_dir = Path::new("./test_performance_data");
    std::fs::create_dir_all(test_dir)?;

    // Demo 1: Batch Writer Optimizations
    println!("\n📝 Demo 1: Batch Write Optimizations");
    demo_batch_writer(test_dir)?;

    // Demo 2: Performance Monitoring
    println!("\n📊 Demo 2: Performance Monitoring and Adaptive Optimization");
    demo_performance_monitoring()?;

    // Demo 3: Parallel Processing
    println!("\n🧵 Demo 3: Streaming and Parallel Processing");
    demo_parallel_processing(test_dir)?;

    // Demo 4: Double Buffering (CUDA only)
    #[cfg(feature = "cuda")]
    {
        println!("\n🔄 Demo 4: Double Buffering for GPU Processing");
        demo_double_buffering()?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n🔄 Demo 4: Double Buffering (CUDA feature not enabled)");
        println!("   To see double buffering demo, run with: cargo run --features cuda --example performance_optimizations_demo");
    }

    println!("\n✅ All performance optimization demos completed!");
    Ok(())
}

/// Demo batch writer optimizations
fn demo_batch_writer(test_dir: &Path) -> Result<()> {
    let output_path = test_dir.join("batch_output.csv");
    let mut writer = BatchWriter::with_batch_size(&output_path, 1000)?;

    println!("   Creating test records and writing with batch optimization...");
    let start_time = Instant::now();

    // Create and write test records
    for i in 0..5000 {
        let record = Record {
            user: format!("user{}@example.com", i),
            password: format!("password{}", i),
            url: format!("https://example{}.com", i),
            normalized_user: format!("user{}@example.com", i),
            normalized_url: format!("example{}.com", i),
            completeness_score: 3.0,
            field_count: 3,
            all_fields: vec![
                format!("user{}@example.com", i),
                format!("password{}", i),
                format!("https://example{}.com", i)
            ],
        };
        writer.add_record(record)?;
    }

    writer.finish()?;
    let write_time = start_time.elapsed();

    let metrics = writer.get_metrics();
    println!("   ⏱️  Write time: {:.2}s", write_time.as_secs_f64());
    println!("   {}", metrics.format_summary());
    println!("   📈 Efficiency Score: {:.1}%", metrics.get_efficiency_score() * 100.0);

    Ok(())
}

/// Demo performance monitoring and adaptive optimization
fn demo_performance_monitoring() -> Result<()> {
    let mut monitor = PerformanceMonitor::new();

    println!("   Simulating processing with performance monitoring...");

    // Simulate processing samples with varying performance
    for i in 0..10 {
        let records_processed = 1000 + (i * 100) as usize;
        let processing_time = std::time::Duration::from_millis(100 + (i * 10));
        let io_time = std::time::Duration::from_millis(50 + (i * 5));
        let memory_usage = 50.0 + (i as f64 * 2.0);
        let gpu_utilization = 70.0 + (i as f64 * 3.0);

        monitor.add_sample(
            records_processed,
            processing_time,
            io_time,
            memory_usage,
            gpu_utilization,
        )?;

        // Check if optimization is needed
        if monitor.should_optimize() {
            let optimized_params = monitor.optimize_parameters()?;
            println!("   🔧 Optimized parameters: chunk_size={}MB, batch_size={}, threads={}",
                    optimized_params.chunk_size_mb,
                    optimized_params.batch_size,
                    optimized_params.parallel_threads);
        }
    }

    println!("   {}", monitor.format_performance_summary());

    Ok(())
}

/// Demo parallel processing and streaming
fn demo_parallel_processing(test_dir: &Path) -> Result<()> {
    // Create test files
    let test_files = create_test_files(test_dir, 3)?;

    let mut processor = ParallelProcessor::with_thread_count(4);
    let output_path = test_dir.join("parallel_output.csv");
    let mut writer = BatchWriter::new(&output_path)?;

    println!("   Processing {} files in parallel...", test_files.len());
    let start_time = Instant::now();

    // Simple processor function that extracts records from CSV data
    let processor_fn = |data: &[u8], _file_path: &Path| -> Result<Vec<Record>> {
        let content = String::from_utf8_lossy(data);
        let mut records = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            if line_num == 0 || line.trim().is_empty() {
                continue; // Skip header and empty lines
            }

            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 3 {
                let record = Record {
                    user: fields[0].to_string(),
                    password: fields[1].to_string(),
                    url: fields[2].to_string(),
                    normalized_user: fields[0].to_lowercase(),
                    normalized_url: fields[2].to_lowercase(),
                    completeness_score: 3.0,
                    field_count: fields.len(),
                    all_fields: fields.iter().map(|s| s.to_string()).collect(),
                };
                records.push(record);
            }
        }

        Ok(records)
    };

    processor.process_files_parallel(&test_files, processor_fn, &mut writer)?;
    writer.finish()?;

    let processing_time = start_time.elapsed();
    let metrics = processor.get_metrics();

    println!("   ⏱️  Processing time: {:.2}s", processing_time.as_secs_f64());
    println!("   {}", metrics.format_summary());

    // Optimize thread count based on performance
    let optimized_threads = processor.optimize_thread_count();
    println!("   🔧 Optimized thread count: {}", optimized_threads);

    Ok(())
}

/// Demo double buffering (CUDA only)
#[cfg(feature = "cuda")]
fn demo_double_buffering() -> Result<()> {
    use tuonella_sift::cuda::processor::CudaRecord;

    let double_buffer = DoubleBuffer::new(512)?; // 512MB total capacity

    println!("   Simulating double buffering with overlapping I/O and GPU processing...");

    // Simulate adding records to buffer
    for batch in 0..5 {
        let mut records = Vec::new();
        for i in 0..1000 {
            records.push(CudaRecord {
                user: format!("user{}@batch{}.com", i, batch),
                password: format!("password{}", i),
                url: format!("https://batch{}.example{}.com", batch, i),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec![
                    format!("user{}@batch{}.com", i, batch),
                    format!("password{}", i),
                    format!("https://batch{}.example{}.com", batch, i)
                ],
            });
        }

        // Add records and check if buffer swap is needed
        let needs_swap = double_buffer.add_records(records)?;
        if needs_swap {
            println!("   🔄 Buffer swap triggered for batch {}", batch);
            double_buffer.swap_buffers()?;
        }

        // Simulate processing
        if double_buffer.is_io_buffer_available() {
            println!("   📥 I/O buffer available for batch {}", batch + 1);
        }
    }

    // Get final metrics
    let metrics = double_buffer.get_metrics();
    println!("   {}", metrics.format_summary());

    // Flush any remaining records
    let remaining = double_buffer.flush_remaining()?;
    println!("   🔄 Flushed {} remaining records", remaining.len());

    Ok(())
}

/// Create test CSV files for parallel processing demo
fn create_test_files(test_dir: &Path, file_count: usize) -> Result<Vec<std::path::PathBuf>> {
    let mut file_paths = Vec::new();

    for file_num in 0..file_count {
        let file_path = test_dir.join(format!("test_file_{}.csv", file_num));
        let mut content = String::new();
        content.push_str("user,password,url\n");

        // Add test records
        for i in 0..1000 {
            content.push_str(&format!(
                "user{}@file{}.com,password{},https://file{}.example{}.com\n",
                i, file_num, i, file_num, i
            ));
        }

        std::fs::write(&file_path, content)?;
        file_paths.push(file_path);
    }

    Ok(file_paths)
}
