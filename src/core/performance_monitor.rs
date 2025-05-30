use std::time::{Duration, Instant};
use std::collections::VecDeque;
use anyhow::Result;
use crate::constants::{
    PERFORMANCE_SAMPLE_WINDOW_RECORDS, ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS,
    MIN_THROUGHPUT_RECORDS_PER_SECOND, THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT
};

#[derive(Debug)]
pub struct PerformanceMonitor {
    sample_window: VecDeque<PerformanceSample>,
    current_metrics: PerformanceMetrics,
    optimization_params: OptimizationParams,
    records_since_optimization: usize,
    session_start: Instant,
}
#[derive(Debug, Clone)]
struct PerformanceSample {
    #[allow(dead_code)]
    timestamp: Instant,
    records_processed: usize,
    processing_time: Duration,
    io_time: Duration,
    memory_usage_percent: f64,
    gpu_utilization_percent: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub current_throughput: f64,
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub processing_efficiency: f64,
    pub io_efficiency: f64,
    pub memory_pressure: f64,
    pub gpu_utilization: f64,
    pub recommended_chunk_size_mb: usize,
    pub performance_trend: PerformanceTrend,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

#[derive(Debug, Clone)]
pub struct OptimizationParams {
    pub chunk_size_mb: usize,
    pub batch_size: usize,
    pub parallel_threads: usize,
    pub buffer_size_mb: usize,
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
    pub fn new() -> Self {
        Self {
            sample_window: VecDeque::with_capacity(PERFORMANCE_SAMPLE_WINDOW_RECORDS / 1000),
            current_metrics: PerformanceMetrics::default(),
            optimization_params: OptimizationParams::default(),
            records_since_optimization: 0,
            session_start: Instant::now(),
        }
    }

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

        self.sample_window.push_back(sample);

        while self.sample_window.len() > (PERFORMANCE_SAMPLE_WINDOW_RECORDS / 1000) {
            self.sample_window.pop_front();
        }

        self.update_metrics()?;

        self.records_since_optimization += records_processed;

        Ok(())
    }

    fn update_metrics(&mut self) -> Result<()> {
        if self.sample_window.is_empty() {
            return Ok(());
        }

        let total_records: usize = self.sample_window.iter().map(|s| s.records_processed).sum();
        let total_processing_time: Duration = self.sample_window.iter().map(|s| s.processing_time).sum();
        let total_io_time: Duration = self.sample_window.iter().map(|s| s.io_time).sum();
        let total_time = total_processing_time + total_io_time;

        if total_time.as_secs_f64() > 0.0 {
            self.current_metrics.current_throughput = total_records as f64 / total_time.as_secs_f64();
        }

        let session_duration = self.session_start.elapsed();
        if session_duration.as_secs_f64() > 0.0 {
            let session_records: usize = self.sample_window.iter().map(|s| s.records_processed).sum();
            self.current_metrics.average_throughput = session_records as f64 / session_duration.as_secs_f64();
        }

        if self.current_metrics.current_throughput > self.current_metrics.peak_throughput {
            self.current_metrics.peak_throughput = self.current_metrics.current_throughput;
        }

        if total_time.as_secs_f64() > 0.0 {
            self.current_metrics.processing_efficiency =
                total_processing_time.as_secs_f64() / total_time.as_secs_f64();
            self.current_metrics.io_efficiency =
                total_io_time.as_secs_f64() / total_time.as_secs_f64();
        }

        self.current_metrics.memory_pressure =
            self.sample_window.iter().map(|s| s.memory_usage_percent).sum::<f64>() /
            self.sample_window.len() as f64 / 100.0;

        self.current_metrics.gpu_utilization =
            self.sample_window.iter().map(|s| s.gpu_utilization_percent).sum::<f64>() /
            self.sample_window.len() as f64;

        self.update_performance_trend()?;

        self.calculate_recommended_chunk_size()?;

        Ok(())
    }

    fn update_performance_trend(&mut self) -> Result<()> {
        if self.sample_window.len() < 3 {
            self.current_metrics.performance_trend = PerformanceTrend::Stable;
            return Ok(());
        }

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

    fn calculate_recommended_chunk_size(&mut self) -> Result<()> {
        let current_chunk_size = self.optimization_params.chunk_size_mb;
        let mut recommended_size = current_chunk_size;

        if self.current_metrics.memory_pressure > 0.8 {
            recommended_size = (current_chunk_size as f64 * 0.8) as usize;
        } else if self.current_metrics.memory_pressure < 0.5 {
            recommended_size = (current_chunk_size as f64 * 1.2) as usize;
        }

        if self.current_metrics.current_throughput < MIN_THROUGHPUT_RECORDS_PER_SECOND {
            recommended_size = (recommended_size as f64 * 0.7) as usize;
        }

        if self.current_metrics.io_efficiency > 0.6 {
            recommended_size = (recommended_size as f64 * 1.3) as usize;
        }

        recommended_size = recommended_size.max(64).min(2048);

        self.current_metrics.recommended_chunk_size_mb = recommended_size;
        Ok(())
    }

    pub fn should_optimize(&self) -> bool {
        self.records_since_optimization >= ADAPTIVE_OPTIMIZATION_INTERVAL_RECORDS
    }

    pub fn optimize_parameters(&mut self) -> Result<OptimizationParams> {
        if !self.should_optimize() {
            return Ok(self.optimization_params.clone());
        }

        let mut new_params = self.optimization_params.clone();

        new_params.chunk_size_mb = self.current_metrics.recommended_chunk_size_mb;

        if self.current_metrics.gpu_utilization < 70.0 {
            new_params.batch_size = (new_params.batch_size as f64 * 1.2) as usize;
        } else if self.current_metrics.gpu_utilization > 95.0 {
            new_params.batch_size = (new_params.batch_size as f64 * 0.9) as usize;
        }

        match self.current_metrics.performance_trend {
            PerformanceTrend::Degrading => {
                new_params.parallel_threads = (new_params.parallel_threads.saturating_sub(1)).max(1);
            },
            PerformanceTrend::Improving => {
                new_params.parallel_threads = (new_params.parallel_threads + 1).min(16);
            },
            PerformanceTrend::Stable => {
            }
        }

        new_params.batch_size = new_params.batch_size.max(1000).min(100000);
        new_params.buffer_size_mb = (new_params.chunk_size_mb * 2).max(256).min(4096);

        self.optimization_params = new_params.clone();
        self.records_since_optimization = 0;

        Ok(new_params)
    }

    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.current_metrics
    }

    pub fn get_optimization_params(&self) -> &OptimizationParams {
        &self.optimization_params
    }

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
