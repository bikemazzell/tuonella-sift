#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::{Instant, Duration};
#[cfg(feature = "cuda")]
use parking_lot::Mutex;
#[cfg(feature = "cuda")]
use crate::cuda::processor::CudaDeviceProperties;
#[cfg(feature = "cuda")]
use crate::constants::MEMORY_PRESSURE_THRESHOLD_PERCENT;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct MemoryUsageSnapshot {
    pub total_memory: usize,
    pub free_memory: usize,
    pub used_memory: usize,
    pub usage_percentage: f64,
    pub timestamp: Instant,
    pub other_processes_memory: usize,
}

#[cfg(feature = "cuda")]
impl MemoryUsageSnapshot {
    pub fn new(total_memory: usize, free_memory: usize) -> Self {
        let used_memory = total_memory - free_memory;
        let usage_percentage = (used_memory as f64 / total_memory as f64) * 100.0;
        
        Self {
            total_memory,
            free_memory,
            used_memory,
            usage_percentage,
            timestamp: Instant::now(),
            other_processes_memory: 0, // Will be calculated externally
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct MemoryPressurePredictor {
    historical_snapshots: Vec<MemoryUsageSnapshot>,
    max_history_size: usize,
    memory_growth_rate: f64, // MB per second
    prediction_window: Duration, // How far ahead to predict
}

#[cfg(feature = "cuda")]
impl MemoryPressurePredictor {
    pub fn new(max_history_size: usize, prediction_window: Duration) -> Self {
        Self {
            historical_snapshots: Vec::with_capacity(max_history_size),
            max_history_size,
            memory_growth_rate: 0.0,
            prediction_window,
        }
    }

    pub fn add_snapshot(&mut self, snapshot: MemoryUsageSnapshot) {
        self.historical_snapshots.push(snapshot);
        
        // Keep only recent history
        if self.historical_snapshots.len() > self.max_history_size {
            self.historical_snapshots.remove(0);
        }
        
        // Recalculate growth rate
        self.calculate_memory_growth_rate();
    }

    fn calculate_memory_growth_rate(&mut self) {
        if self.historical_snapshots.len() < 2 {
            self.memory_growth_rate = 0.0;
            return;
        }

        let recent_count = self.historical_snapshots.len().min(10); // Use last 10 snapshots
        let recent_snapshots = &self.historical_snapshots[self.historical_snapshots.len() - recent_count..];
        
        if recent_snapshots.len() < 2 {
            return;
        }

        let first = &recent_snapshots[0];
        let last = &recent_snapshots[recent_snapshots.len() - 1];
        
        let time_diff = last.timestamp.duration_since(first.timestamp).as_secs_f64();
        if time_diff > 0.0 {
            let memory_diff = last.used_memory as f64 - first.used_memory as f64;
            self.memory_growth_rate = (memory_diff / (1024.0 * 1024.0)) / time_diff; // MB per second
        }
    }

    pub fn predict_memory_pressure(&self, current_usage: f64) -> MemoryPressurePrediction {
        let prediction_secs = self.prediction_window.as_secs_f64();
        let predicted_growth_mb = self.memory_growth_rate * prediction_secs;
        let predicted_growth_bytes = predicted_growth_mb * 1024.0 * 1024.0;
        
        let current_snapshot = self.historical_snapshots.last();
        let predicted_usage_bytes = if let Some(snapshot) = current_snapshot {
            snapshot.used_memory as f64 + predicted_growth_bytes
        } else {
            predicted_growth_bytes
        };

        let total_memory = current_snapshot.map(|s| s.total_memory).unwrap_or(0) as f64;
        let predicted_usage_percentage = if total_memory > 0.0 {
            (predicted_usage_bytes / total_memory) * 100.0
        } else {
            0.0
        };

        MemoryPressurePrediction {
            current_usage_percentage: current_usage,
            predicted_usage_percentage,
            growth_rate_mb_per_sec: self.memory_growth_rate,
            time_to_pressure: self.calculate_time_to_pressure(),
            recommendation: self.get_recommendation(predicted_usage_percentage),
        }
    }

    fn calculate_time_to_pressure(&self) -> Option<Duration> {
        if self.memory_growth_rate <= 0.0 {
            return None; // No growth or shrinking
        }

        let current_snapshot = self.historical_snapshots.last()?;
        let available_mb = current_snapshot.free_memory as f64 / (1024.0 * 1024.0);
        let pressure_threshold_mb = (current_snapshot.total_memory as f64 * (MEMORY_PRESSURE_THRESHOLD_PERCENT as f64 / 100.0)) / (1024.0 * 1024.0);
        let current_usage_mb = current_snapshot.used_memory as f64 / (1024.0 * 1024.0);
        
        if current_usage_mb >= pressure_threshold_mb {
            return Some(Duration::from_secs(0)); // Already at pressure
        }

        let mb_until_pressure = pressure_threshold_mb - current_usage_mb;
        let seconds_until_pressure = mb_until_pressure / self.memory_growth_rate;
        
        if seconds_until_pressure > 0.0 {
            Some(Duration::from_secs_f64(seconds_until_pressure))
        } else {
            None
        }
    }

    fn get_recommendation(&self, predicted_usage: f64) -> MemoryManagementRecommendation {
        match predicted_usage {
            p if p < 70.0 => MemoryManagementRecommendation::Increase,
            p if p < 85.0 => MemoryManagementRecommendation::Maintain,
            p if p < 95.0 => MemoryManagementRecommendation::Reduce,
            _ => MemoryManagementRecommendation::Emergency,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct MemoryPressurePrediction {
    pub current_usage_percentage: f64,
    pub predicted_usage_percentage: f64,
    pub growth_rate_mb_per_sec: f64,
    pub time_to_pressure: Option<Duration>,
    pub recommendation: MemoryManagementRecommendation,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryManagementRecommendation {
    Increase,   // Can increase memory usage
    Maintain,   // Keep current usage
    Reduce,     // Should reduce memory usage
    Emergency,  // Must reduce immediately
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct AdaptiveMemoryManager {
    context: Arc<CudaContext>,
    predictor: Arc<Mutex<MemoryPressurePredictor>>,
    current_pool_size: Arc<Mutex<usize>>,
    min_pool_size: usize,
    max_pool_size: usize,
    base_pool_size: usize,
    last_adjustment: Arc<Mutex<Instant>>,
    adjustment_cooldown: Duration,
    monitoring_enabled: bool,
}

#[cfg(feature = "cuda")]
impl AdaptiveMemoryManager {
    pub fn new(
        context: Arc<CudaContext>, 
        base_pool_size: usize,
        min_pool_size: usize,
        max_pool_size: usize
    ) -> Result<Self> {
        let predictor = MemoryPressurePredictor::new(50, Duration::from_secs(30));
        
        println!("Adaptive memory manager initialized: base={} MB, range={}-{} MB", 
                 base_pool_size / (1024 * 1024), 
                 min_pool_size / (1024 * 1024), 
                 max_pool_size / (1024 * 1024));
        
        Ok(Self {
            context,
            predictor: Arc::new(Mutex::new(predictor)),
            current_pool_size: Arc::new(Mutex::new(base_pool_size)),
            min_pool_size,
            max_pool_size,
            base_pool_size,
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
            adjustment_cooldown: Duration::from_secs(10), // Don't adjust more than once per 10 seconds
            monitoring_enabled: true,
        })
    }

    pub fn update_memory_usage(&self) -> Result<MemoryUsageSnapshot> {
        if !self.monitoring_enabled {
            return Ok(MemoryUsageSnapshot::new(0, 0));
        }

        // Get current GPU memory info
        let (free_memory, total_memory) = cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("Failed to get GPU memory info: {}", e))?;

        let snapshot = MemoryUsageSnapshot::new(total_memory, free_memory);
        
        // Add to predictor
        {
            let mut predictor = self.predictor.lock();
            predictor.add_snapshot(snapshot.clone());
        }

        Ok(snapshot)
    }

    pub fn should_adjust_pool_size(&self) -> Result<bool> {
        let last_adjustment = *self.last_adjustment.lock();
        if last_adjustment.elapsed() < self.adjustment_cooldown {
            return Ok(false); // Still in cooldown period
        }

        let snapshot = self.update_memory_usage()?;
        let prediction = {
            let predictor = self.predictor.lock();
            predictor.predict_memory_pressure(snapshot.usage_percentage)
        };

        Ok(matches!(prediction.recommendation, 
                   MemoryManagementRecommendation::Reduce | 
                   MemoryManagementRecommendation::Emergency |
                   MemoryManagementRecommendation::Increase))
    }

    pub fn adjust_pool_size(&self) -> Result<PoolSizeAdjustment> {
        let snapshot = self.update_memory_usage()?;
        let prediction = {
            let predictor = self.predictor.lock();
            predictor.predict_memory_pressure(snapshot.usage_percentage)
        };

        let current_size = *self.current_pool_size.lock();
        let new_size = self.calculate_new_pool_size(current_size, &prediction)?;
        
        if new_size != current_size {
            *self.current_pool_size.lock() = new_size;
            *self.last_adjustment.lock() = Instant::now();
            
            return Ok(PoolSizeAdjustment {
                old_size: current_size,
                new_size,
                reason: format!("Prediction: {:.1}% -> {:.1}%, recommendation: {:?}", 
                               prediction.current_usage_percentage,
                               prediction.predicted_usage_percentage,
                               prediction.recommendation),
                adjustment_type: if new_size > current_size { 
                    AdjustmentType::Increase 
                } else { 
                    AdjustmentType::Decrease 
                },
            });
        }

        Ok(PoolSizeAdjustment {
            old_size: current_size,
            new_size: current_size,
            reason: "No adjustment needed".to_string(),
            adjustment_type: AdjustmentType::NoChange,
        })
    }

    fn calculate_new_pool_size(&self, current_size: usize, prediction: &MemoryPressurePrediction) -> Result<usize> {
        let adjustment_factor = match prediction.recommendation {
            MemoryManagementRecommendation::Increase => 1.2, // Increase by 20%
            MemoryManagementRecommendation::Maintain => 1.0, // No change
            MemoryManagementRecommendation::Reduce => 0.8,   // Reduce by 20%
            MemoryManagementRecommendation::Emergency => 0.6, // Reduce by 40%
        };

        let new_size = (current_size as f64 * adjustment_factor) as usize;
        Ok(new_size.clamp(self.min_pool_size, self.max_pool_size))
    }

    pub fn get_current_pool_size(&self) -> usize {
        *self.current_pool_size.lock()
    }

    pub fn get_memory_statistics(&self) -> Result<AdaptiveMemoryStats> {
        let snapshot = self.update_memory_usage()?;
        let prediction = {
            let predictor = self.predictor.lock();
            predictor.predict_memory_pressure(snapshot.usage_percentage)
        };

        Ok(AdaptiveMemoryStats {
            current_snapshot: snapshot,
            prediction,
            current_pool_size: self.get_current_pool_size(),
            base_pool_size: self.base_pool_size,
            min_pool_size: self.min_pool_size,
            max_pool_size: self.max_pool_size,
            monitoring_enabled: self.monitoring_enabled,
        })
    }

    pub fn enable_monitoring(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct PoolSizeAdjustment {
    pub old_size: usize,
    pub new_size: usize,
    pub reason: String,
    pub adjustment_type: AdjustmentType,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub enum AdjustmentType {
    Increase,
    Decrease,
    NoChange,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct AdaptiveMemoryStats {
    pub current_snapshot: MemoryUsageSnapshot,
    pub prediction: MemoryPressurePrediction,
    pub current_pool_size: usize,
    pub base_pool_size: usize,
    pub min_pool_size: usize,
    pub max_pool_size: usize,
    pub monitoring_enabled: bool,
}

#[cfg(feature = "cuda")]
impl AdaptiveMemoryStats {
    pub fn format_summary(&self) -> String {
        let time_to_pressure = if let Some(duration) = self.prediction.time_to_pressure {
            format!("{:.1}s", duration.as_secs_f64())
        } else {
            "N/A".to_string()
        };

        format!(
            "Adaptive Memory Management:\n\
             💾 GPU Memory: {:.1}% used ({:.1} GB / {:.1} GB)\n\
             📊 Pool Size: {:.1} MB (range: {:.1}-{:.1} MB)\n\
             📈 Growth Rate: {:.2} MB/s\n\
             ⏱️  Time to Pressure: {}\n\
             🎯 Recommendation: {:?}\n\
             🔮 Predicted Usage: {:.1}%\n\
             📡 Monitoring: {}",
            self.current_snapshot.usage_percentage,
            self.current_snapshot.used_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.current_snapshot.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.current_pool_size as f64 / (1024.0 * 1024.0),
            self.min_pool_size as f64 / (1024.0 * 1024.0),
            self.max_pool_size as f64 / (1024.0 * 1024.0),
            self.prediction.growth_rate_mb_per_sec,
            time_to_pressure,
            self.prediction.recommendation,
            self.prediction.predicted_usage_percentage,
            if self.monitoring_enabled { "Enabled" } else { "Disabled" }
        )
    }
}

// Stub implementations for when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct AdaptiveMemoryManager;

#[cfg(not(feature = "cuda"))]
impl AdaptiveMemoryManager {
    pub fn new(_: (), _: usize, _: usize, _: usize) -> Result<Self, ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_usage_snapshot() {
        let snapshot = MemoryUsageSnapshot::new(8000000000, 2000000000); // 8GB total, 2GB free
        
        assert_eq!(snapshot.total_memory, 8000000000);
        assert_eq!(snapshot.free_memory, 2000000000);
        assert_eq!(snapshot.used_memory, 6000000000);
        assert_eq!(snapshot.usage_percentage, 75.0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_pressure_predictor() {
        let mut predictor = MemoryPressurePredictor::new(10, Duration::from_secs(30));
        
        // Add some snapshots with increasing memory usage
        let base_time = Instant::now();
        for i in 0..5 {
            let used_memory = 4000000000 + (i * 500000000); // Increasing usage
            let snapshot = MemoryUsageSnapshot {
                total_memory: 8000000000,
                free_memory: 8000000000 - used_memory,
                used_memory,
                usage_percentage: (used_memory as f64 / 8000000000.0) * 100.0,
                timestamp: base_time + Duration::from_secs(i as u64 * 10),
                other_processes_memory: 0,
            };
            predictor.add_snapshot(snapshot);
        }
        
        let prediction = predictor.predict_memory_pressure(75.0);
        assert!(prediction.growth_rate_mb_per_sec > 0.0);
        assert!(prediction.predicted_usage_percentage > 75.0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_management_recommendations() {
        let predictor = MemoryPressurePredictor::new(10, Duration::from_secs(30));
        
        assert_eq!(predictor.get_recommendation(60.0), MemoryManagementRecommendation::Increase);
        assert_eq!(predictor.get_recommendation(80.0), MemoryManagementRecommendation::Maintain);
        assert_eq!(predictor.get_recommendation(90.0), MemoryManagementRecommendation::Reduce);
        assert_eq!(predictor.get_recommendation(98.0), MemoryManagementRecommendation::Emergency);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adaptive_memory_manager_creation() {
        // This test requires actual CUDA context
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return; // Skip if no CUDA
        }

        match cudarc::driver::safe::CudaContext::new(0) {
            Ok(context) => {
                let base_size = 64 * 1024 * 1024; // 64MB
                let min_size = 16 * 1024 * 1024;  // 16MB
                let max_size = 256 * 1024 * 1024; // 256MB
                
                match AdaptiveMemoryManager::new(context, base_size, min_size, max_size) {
                    Ok(manager) => {
                        assert_eq!(manager.get_current_pool_size(), base_size);
                        println!("Adaptive memory manager test passed");
                    },
                    Err(e) => {
                        println!("Failed to create adaptive memory manager: {}", e);
                    }
                }
            },
            Err(e) => {
                println!("CUDA context creation failed: {}", e);
            }
        }
    }
}