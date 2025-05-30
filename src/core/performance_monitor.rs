use std::time::{Duration, Instant};
use std::collections::VecDeque;
use anyhow::Result;
use crate::constants::{
    PERFORMANCE_SAMPLE_WINDOW_RECORDS, ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS,
    MIN_THROUGHPUT_RECORDS_PER_SECOND, THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT
};

/// Performance monitoring and adaptive optimization system
///
/// This implements Section 6: "Dynamically adjust chunk sizes based on observed processing speed and resource availability"
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Sample window for performance measurements
    sample_window: VecDeque<PerformanceSample>,
    /// Current performance metrics
    current_metrics: PerformanceMetrics,
    /// Optimization parameters
    optimization_params: OptimizationParams,
    /// Records processed since last optimization
    records_since_optimization: usize,
    /// Start time for current session
    session_start: Instant,
}

/// Individual performance sample
#[derive(Debug, Clone)]
struct PerformanceSample {
    /// Timestamp of the sample
    #[allow(dead_code)]
    timestamp: Instant,
    /// Number of records processed
    records_processed: usize,
    /// Processing time for this sample
    processing_time: Duration,
    /// I/O time for this sample
    io_time: Duration,
    /// Memory usage at time of sample
    memory_usage_percent: f64,
    /// GPU utilization (if available)
    gpu_utilization_percent: f64,
}

/// Current performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Current throughput (records/second)
    pub current_throughput: f64,
    /// Average throughput over sample window
    pub average_throughput: f64,
    /// Peak throughput observed
    pub peak_throughput: f64,
    /// Current processing efficiency (processing_time / total_time)
    pub processing_efficiency: f64,
    /// Current I/O efficiency (io_time / total_time)
    pub io_efficiency: f64,
    /// Memory pressure indicator (0.0 to 1.0)
    pub memory_pressure: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Recommended chunk size based on performance
    pub recommended_chunk_size_mb: usize,
    /// Performance trend (improving, stable, degrading)
    pub performance_trend: PerformanceTrend,
}

/// Performance trend indicator
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Optimization parameters that can be adjusted
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Current chunk size in MB
    pub chunk_size_mb: usize,
    /// Current batch size for processing
    pub batch_size: usize,
    /// Current number of parallel threads
    pub parallel_threads: usize,
    /// Current buffer size in MB
    pub buffer_size_mb: usize,
    /// Whether to use aggressive optimization
    pub aggressive_optimization: bool,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            chunk_size_mb: 256,
            batch_size: 10000,
            parallel_threads: 4,
            buffer_size_mb: 512,
            aggressive_optimization: false,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            current_throughput: 0.0,
            average_throughput: 0.0,
            peak_throughput: 0.0,
            processing_efficiency: 0.0,
            io_efficiency: 0.0,
            memory_pressure: 0.0,
            gpu_utilization: 0.0,
            recommended_chunk_size_mb: 256,
            performance_trend: PerformanceTrend::Stable,
        }
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            sample_window: VecDeque::with_capacity(PERFORMANCE_SAMPLE_WINDOW_RECORDS / 1000), // Store samples per 1000 records
            current_metrics: PerformanceMetrics::default(),
            optimization_params: OptimizationParams::default(),
            records_since_optimization: 0,
            session_start: Instant::now(),
        }
    }

    /// Add a performance sample
    pub fn add_sample(
        &mut self,
        records_processed: usize,
        processing_time: Duration,
        io_time: Duration,
        memory_usage_percent: f64,
        gpu_utilization_percent: f64,
    ) -> Result<()> {
        let sample = PerformanceSample {
            timestamp: Instant::now(),
            records_processed,
            processing_time,
            io_time,
            memory_usage_percent,
            gpu_utilization_percent,
        };

        // Add sample to window
        self.sample_window.push_back(sample);

        // Keep window size manageable
        while self.sample_window.len() > (PERFORMANCE_SAMPLE_WINDOW_RECORDS / 1000) {
            self.sample_window.pop_front();
        }

        // Update current metrics
        self.update_metrics()?;

        // Track records for optimization interval
        self.records_since_optimization += records_processed;

        Ok(())
    }

    /// Update current performance metrics based on sample window
    fn update_metrics(&mut self) -> Result<()> {
        if self.sample_window.is_empty() {
            return Ok(());
        }

        let total_records: usize = self.sample_window.iter().map(|s| s.records_processed).sum();
        let total_processing_time: Duration = self.sample_window.iter().map(|s| s.processing_time).sum();
        let total_io_time: Duration = self.sample_window.iter().map(|s| s.io_time).sum();
        let total_time = total_processing_time + total_io_time;

        // Calculate throughput
        if total_time.as_secs_f64() > 0.0 {
            self.current_metrics.current_throughput = total_records as f64 / total_time.as_secs_f64();
        }

        // Calculate average throughput over session
        let session_duration = self.session_start.elapsed();
        if session_duration.as_secs_f64() > 0.0 {
            let session_records: usize = self.sample_window.iter().map(|s| s.records_processed).sum();
            self.current_metrics.average_throughput = session_records as f64 / session_duration.as_secs_f64();
        }

        // Update peak throughput
        if self.current_metrics.current_throughput > self.current_metrics.peak_throughput {
            self.current_metrics.peak_throughput = self.current_metrics.current_throughput;
        }

        // Calculate efficiency metrics
        if total_time.as_secs_f64() > 0.0 {
            self.current_metrics.processing_efficiency =
                total_processing_time.as_secs_f64() / total_time.as_secs_f64();
            self.current_metrics.io_efficiency =
                total_io_time.as_secs_f64() / total_time.as_secs_f64();
        }

        // Calculate memory pressure (average over window)
        self.current_metrics.memory_pressure =
            self.sample_window.iter().map(|s| s.memory_usage_percent).sum::<f64>() /
            self.sample_window.len() as f64 / 100.0;

        // Calculate GPU utilization (average over window)
        self.current_metrics.gpu_utilization =
            self.sample_window.iter().map(|s| s.gpu_utilization_percent).sum::<f64>() /
            self.sample_window.len() as f64;

        // Determine performance trend
        self.update_performance_trend()?;

        // Calculate recommended chunk size
        self.calculate_recommended_chunk_size()?;

        Ok(())
    }

    /// Update performance trend based on recent samples
    fn update_performance_trend(&mut self) -> Result<()> {
        if self.sample_window.len() < 3 {
            self.current_metrics.performance_trend = PerformanceTrend::Stable;
            return Ok(());
        }

        // Compare recent throughput with earlier throughput
        let recent_samples = &self.sample_window.iter().rev().take(3).collect::<Vec<_>>();
        let earlier_samples = &self.sample_window.iter().take(3).collect::<Vec<_>>();

        let recent_throughput = self.calculate_throughput_for_samples(recent_samples);
        let earlier_throughput = self.calculate_throughput_for_samples(earlier_samples);

        if earlier_throughput > 0.0 {
            let improvement_percent = ((recent_throughput - earlier_throughput) / earlier_throughput) * 100.0;

            if improvement_percent > THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT {
                self.current_metrics.performance_trend = PerformanceTrend::Improving;
            } else if improvement_percent < -THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT {
                self.current_metrics.performance_trend = PerformanceTrend::Degrading;
            } else {
                self.current_metrics.performance_trend = PerformanceTrend::Stable;
            }
        }

        Ok(())
    }

    /// Calculate throughput for a set of samples
    fn calculate_throughput_for_samples(&self, samples: &[&PerformanceSample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let total_records: usize = samples.iter().map(|s| s.records_processed).sum();
        let total_time: Duration = samples.iter().map(|s| s.processing_time + s.io_time).sum();

        if total_time.as_secs_f64() > 0.0 {
            total_records as f64 / total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate recommended chunk size based on performance metrics
    fn calculate_recommended_chunk_size(&mut self) -> Result<()> {
        let current_chunk_size = self.optimization_params.chunk_size_mb;
        let mut recommended_size = current_chunk_size;

        // Adjust based on memory pressure
        if self.current_metrics.memory_pressure > 0.8 {
            // High memory pressure - reduce chunk size
            recommended_size = (current_chunk_size as f64 * 0.8) as usize;
        } else if self.current_metrics.memory_pressure < 0.5 {
            // Low memory pressure - can increase chunk size
            recommended_size = (current_chunk_size as f64 * 1.2) as usize;
        }

        // Adjust based on throughput
        if self.current_metrics.current_throughput < MIN_THROUGHPUT_RECORDS_PER_SECOND {
            // Low throughput - try smaller chunks for better parallelism
            recommended_size = (recommended_size as f64 * 0.7) as usize;
        }

        // Adjust based on I/O efficiency
        if self.current_metrics.io_efficiency > 0.6 {
            // High I/O wait - increase chunk size to reduce I/O overhead
            recommended_size = (recommended_size as f64 * 1.3) as usize;
        }

        // Apply bounds
        recommended_size = recommended_size.max(64).min(2048); // 64MB to 2GB

        self.current_metrics.recommended_chunk_size_mb = recommended_size;
        Ok(())
    }

    /// Check if optimization should be performed
    pub fn should_optimize(&self) -> bool {
        self.records_since_optimization >= ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS
    }

    /// Perform adaptive optimization
    pub fn optimize_parameters(&mut self) -> Result<OptimizationParams> {
        if !self.should_optimize() {
            return Ok(self.optimization_params.clone());
        }

        let mut new_params = self.optimization_params.clone();

        // Adjust chunk size based on recommendation
        new_params.chunk_size_mb = self.current_metrics.recommended_chunk_size_mb;

        // Adjust batch size based on GPU utilization
        if self.current_metrics.gpu_utilization < 70.0 {
            // Low GPU utilization - increase batch size
            new_params.batch_size = (new_params.batch_size as f64 * 1.2) as usize;
        } else if self.current_metrics.gpu_utilization > 95.0 {
            // Very high GPU utilization - might be memory bound
            new_params.batch_size = (new_params.batch_size as f64 * 0.9) as usize;
        }

        // Adjust parallel threads based on performance trend
        match self.current_metrics.performance_trend {
            PerformanceTrend::Degrading => {
                // Try reducing parallelism to reduce contention
                new_params.parallel_threads = (new_params.parallel_threads.saturating_sub(1)).max(1);
            },
            PerformanceTrend::Improving => {
                // Try increasing parallelism
                new_params.parallel_threads = (new_params.parallel_threads + 1).min(16);
            },
            PerformanceTrend::Stable => {
                // Keep current settings
            }
        }

        // Apply bounds
        new_params.batch_size = new_params.batch_size.max(1000).min(100000);
        new_params.buffer_size_mb = (new_params.chunk_size_mb * 2).max(256).min(4096);

        self.optimization_params = new_params.clone();
        self.records_since_optimization = 0;

        Ok(new_params)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.current_metrics
    }

    /// Get current optimization parameters
    pub fn get_optimization_params(&self) -> &OptimizationParams {
        &self.optimization_params
    }

    /// Format performance summary for display
    pub fn format_performance_summary(&self) -> String {
        format!(
            "Performance Monitor:\n\
             ⚡ Current Throughput: {:.1} records/sec\n\
             📊 Average Throughput: {:.1} records/sec\n\
             🏆 Peak Throughput: {:.1} records/sec\n\
             🔄 Processing Efficiency: {:.1}%\n\
             💾 I/O Efficiency: {:.1}%\n\
             🧠 Memory Pressure: {:.1}%\n\
             🖥️  GPU Utilization: {:.1}%\n\
             📈 Trend: {:?}\n\
             💡 Recommended Chunk Size: {} MB",
            self.current_metrics.current_throughput,
            self.current_metrics.average_throughput,
            self.current_metrics.peak_throughput,
            self.current_metrics.processing_efficiency * 100.0,
            self.current_metrics.io_efficiency * 100.0,
            self.current_metrics.memory_pressure * 100.0,
            self.current_metrics.gpu_utilization,
            self.current_metrics.performance_trend,
            self.current_metrics.recommended_chunk_size_mb
        )
    }
}
