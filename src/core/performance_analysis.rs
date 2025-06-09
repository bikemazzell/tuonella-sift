use std::time::{Duration, Instant};
use std::collections::VecDeque;
use anyhow::Result;
use crate::constants::{
    MIN_THROUGHPUT_RECORDS_PER_SECOND, THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT
};

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub records_per_second: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: f64,
    pub io_throughput_mbps: f64,
    pub cpu_utilization_percent: f64,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub description: String,
    pub expected_improvement_percent: f64,
    pub priority: Priority,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationCategory {
    Memory,
    GPU,
    IO,
    CPU,
    Configuration,
}

#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

pub struct PerformanceAnalyzer {
    metrics_history: VecDeque<PerformanceMetrics>,
    max_history_size: usize,
    last_analysis: Option<Instant>,
    analysis_interval: Duration,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            metrics_history: VecDeque::new(),
            max_history_size: 100,
            last_analysis: None,
            analysis_interval: Duration::from_secs(30),
        }
    }

    pub fn add_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics_history.push_back(metrics);
        
        // Keep history size manageable
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }
    }

    pub fn analyze_performance(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        let now = Instant::now();
        
        // Check if enough time has passed since last analysis
        if let Some(last) = self.last_analysis {
            if now.duration_since(last) < self.analysis_interval {
                return Ok(Vec::new());
            }
        }
        
        self.last_analysis = Some(now);
        
        if self.metrics_history.len() < 5 {
            return Ok(Vec::new()); // Need more data points
        }
        
        let mut recommendations = Vec::new();
        
        // Analyze throughput trends
        recommendations.extend(self.analyze_throughput()?);
        
        // Analyze memory usage patterns
        recommendations.extend(self.analyze_memory_usage()?);
        
        // Analyze GPU utilization
        recommendations.extend(self.analyze_gpu_utilization()?);
        
        // Analyze I/O performance
        recommendations.extend(self.analyze_io_performance()?);
        
        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        Ok(recommendations)
    }

    fn analyze_throughput(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        let recent_metrics: Vec<_> = self.metrics_history.iter().rev().take(10).collect();
        let avg_throughput = recent_metrics.iter()
            .map(|m| m.records_per_second)
            .sum::<f64>() / recent_metrics.len() as f64;
        
        if avg_throughput < MIN_THROUGHPUT_RECORDS_PER_SECOND {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Configuration,
                description: format!(
                    "Low throughput detected ({:.0} rec/sec). Consider increasing batch sizes or enabling GPU processing.",
                    avg_throughput
                ),
                expected_improvement_percent: 50.0,
                priority: Priority::High,
            });
        }
        
        // Check for declining throughput trend
        if recent_metrics.len() >= 5 {
            let first_half_avg = recent_metrics[5..].iter()
                .map(|m| m.records_per_second)
                .sum::<f64>() / (recent_metrics.len() - 5) as f64;
            let second_half_avg = recent_metrics[..5].iter()
                .map(|m| m.records_per_second)
                .sum::<f64>() / 5.0;
            
            let decline_percent = ((first_half_avg - second_half_avg) / first_half_avg) * 100.0;
            
            if decline_percent > THROUGHPUT_IMPROVEMENT_THRESHOLD_PERCENT {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Memory,
                    description: format!(
                        "Throughput declining by {:.1}%. Possible memory pressure or resource contention.",
                        decline_percent
                    ),
                    expected_improvement_percent: decline_percent,
                    priority: Priority::Medium,
                });
            }
        }
        
        Ok(recommendations)
    }

    fn analyze_memory_usage(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        let recent_metrics: Vec<_> = self.metrics_history.iter().rev().take(10).collect();
        let avg_memory = recent_metrics.iter()
            .map(|m| m.memory_usage_mb)
            .sum::<f64>() / recent_metrics.len() as f64;
        
        if avg_memory > 16000.0 { // > 16GB
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                description: format!(
                    "High memory usage detected ({:.1} GB). Consider reducing batch sizes or enabling external sorting.",
                    avg_memory / 1024.0
                ),
                expected_improvement_percent: 30.0,
                priority: Priority::High,
            });
        }
        
        // Check for memory growth trend
        if recent_metrics.len() >= 5 {
            let first_memory = recent_metrics.last().unwrap().memory_usage_mb;
            let last_memory = recent_metrics.first().unwrap().memory_usage_mb;
            let growth_percent = ((last_memory - first_memory) / first_memory) * 100.0;
            
            if growth_percent > 20.0 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Memory,
                    description: format!(
                        "Memory usage growing by {:.1}%. Possible memory leak or inefficient allocation patterns.",
                        growth_percent
                    ),
                    expected_improvement_percent: 25.0,
                    priority: Priority::Medium,
                });
            }
        }
        
        Ok(recommendations)
    }

    fn analyze_gpu_utilization(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        let recent_metrics: Vec<_> = self.metrics_history.iter().rev().take(10).collect();
        let avg_gpu_util = recent_metrics.iter()
            .map(|m| m.gpu_utilization_percent)
            .sum::<f64>() / recent_metrics.len() as f64;
        
        if avg_gpu_util < 30.0 && avg_gpu_util > 0.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::GPU,
                description: format!(
                    "Low GPU utilization ({:.1}%). Consider increasing batch sizes or optimizing CUDA kernels.",
                    avg_gpu_util
                ),
                expected_improvement_percent: 100.0,
                priority: Priority::High,
            });
        }
        
        if avg_gpu_util == 0.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::GPU,
                description: "GPU not being utilized. Enable CUDA processing for significant performance gains.".to_string(),
                expected_improvement_percent: 200.0,
                priority: Priority::Critical,
            });
        }
        
        Ok(recommendations)
    }

    fn analyze_io_performance(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        let recent_metrics: Vec<_> = self.metrics_history.iter().rev().take(10).collect();
        let avg_io_throughput = recent_metrics.iter()
            .map(|m| m.io_throughput_mbps)
            .sum::<f64>() / recent_metrics.len() as f64;
        
        if avg_io_throughput < 100.0 { // Less than 100 MB/s
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::IO,
                description: format!(
                    "Low I/O throughput ({:.1} MB/s). Consider using larger buffer sizes or SSD storage.",
                    avg_io_throughput
                ),
                expected_improvement_percent: 50.0,
                priority: Priority::Medium,
            });
        }
        
        Ok(recommendations)
    }

    pub fn get_performance_summary(&self) -> String {
        if self.metrics_history.is_empty() {
            return "No performance data available".to_string();
        }
        
        let recent = self.metrics_history.back().unwrap();
        
        format!(
            "Performance Summary:\n\
             • Throughput: {:.0} records/sec\n\
             • Memory Usage: {:.1} GB\n\
             • GPU Utilization: {:.1}%\n\
             • I/O Throughput: {:.1} MB/s\n\
             • CPU Utilization: {:.1}%",
            recent.records_per_second,
            recent.memory_usage_mb / 1024.0,
            recent.gpu_utilization_percent,
            recent.io_throughput_mbps,
            recent.cpu_utilization_percent
        )
    }

    pub fn get_metrics_count(&self) -> usize {
        self.metrics_history.len()
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.get_metrics_count(), 0);
    }

    #[test]
    fn test_add_metrics() {
        let mut analyzer = PerformanceAnalyzer::new();
        
        let metrics = PerformanceMetrics {
            records_per_second: 1000.0,
            memory_usage_mb: 2048.0,
            gpu_utilization_percent: 75.0,
            io_throughput_mbps: 150.0,
            cpu_utilization_percent: 60.0,
            timestamp: Instant::now(),
        };
        
        analyzer.add_metrics(metrics);
        assert_eq!(analyzer.get_metrics_count(), 1);
    }

    #[test]
    fn test_performance_summary() {
        let mut analyzer = PerformanceAnalyzer::new();
        
        let metrics = PerformanceMetrics {
            records_per_second: 5000.0,
            memory_usage_mb: 4096.0,
            gpu_utilization_percent: 85.0,
            io_throughput_mbps: 200.0,
            cpu_utilization_percent: 70.0,
            timestamp: Instant::now(),
        };
        
        analyzer.add_metrics(metrics);
        let summary = analyzer.get_performance_summary();
        
        assert!(summary.contains("5000 records/sec"));
        assert!(summary.contains("4.0 GB"));
        assert!(summary.contains("85.0%"));
    }
}
