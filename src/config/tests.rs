#[cfg(test)]
mod tests {
    use crate::config::model::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_batch_sizes_default() {
        let batch_sizes = BatchSizes::default();
        
        assert_eq!(batch_sizes.small, 10000);
        assert_eq!(batch_sizes.medium, 50000);
        assert_eq!(batch_sizes.large, 100000);
        assert_eq!(batch_sizes.xlarge, 500000);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_config_default() {
        let config = CudaConfig::default();
        
        assert_eq!(config.gpu_memory_usage_percent, 50);
        assert_eq!(config.estimated_bytes_per_record, 500);
        assert_eq!(config.min_batch_size, 10000);
        assert_eq!(config.max_batch_size, 1000000);
        assert_eq!(config.max_url_buffer_size, 256);
        assert_eq!(config.max_username_buffer_size, 64);
        assert_eq!(config.threads_per_block, 256);
        
        // Test nested BatchSizes
        assert_eq!(config.batch_sizes.small, 10000);
        assert_eq!(config.batch_sizes.medium, 50000);
        assert_eq!(config.batch_sizes.large, 100000);
        assert_eq!(config.batch_sizes.xlarge, 500000);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_batch_sizes_custom_creation() {
        let batch_sizes = BatchSizes {
            small: 5000,
            medium: 25000,
            large: 75000,
            xlarge: 250000,
        };
        
        assert_eq!(batch_sizes.small, 5000);
        assert_eq!(batch_sizes.medium, 25000);
        assert_eq!(batch_sizes.large, 75000);
        assert_eq!(batch_sizes.xlarge, 250000);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_config_custom_creation() {
        let custom_batch_sizes = BatchSizes {
            small: 1000,
            medium: 5000,
            large: 10000,
            xlarge: 50000,
        };
        
        let config = CudaConfig {
            gpu_memory_usage_percent: 80,
            estimated_bytes_per_record: 1024,
            min_batch_size: 1000,
            max_batch_size: 100000,
            max_url_buffer_size: 512,
            max_username_buffer_size: 128,
            threads_per_block: 512,
            batch_sizes: custom_batch_sizes,
        };
        
        assert_eq!(config.gpu_memory_usage_percent, 80);
        assert_eq!(config.estimated_bytes_per_record, 1024);
        assert_eq!(config.min_batch_size, 1000);
        assert_eq!(config.max_batch_size, 100000);
        assert_eq!(config.max_url_buffer_size, 512);
        assert_eq!(config.max_username_buffer_size, 128);
        assert_eq!(config.threads_per_block, 512);
        
        assert_eq!(config.batch_sizes.small, 1000);
        assert_eq!(config.batch_sizes.medium, 5000);
        assert_eq!(config.batch_sizes.large, 10000);
        assert_eq!(config.batch_sizes.xlarge, 50000);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_config_clone() {
        let config = CudaConfig::default();
        let cloned_config = config.clone();
        
        assert_eq!(config.gpu_memory_usage_percent, cloned_config.gpu_memory_usage_percent);
        assert_eq!(config.estimated_bytes_per_record, cloned_config.estimated_bytes_per_record);
        assert_eq!(config.min_batch_size, cloned_config.min_batch_size);
        assert_eq!(config.max_batch_size, cloned_config.max_batch_size);
        assert_eq!(config.max_url_buffer_size, cloned_config.max_url_buffer_size);
        assert_eq!(config.max_username_buffer_size, cloned_config.max_username_buffer_size);
        assert_eq!(config.threads_per_block, cloned_config.threads_per_block);
        
        assert_eq!(config.batch_sizes.small, cloned_config.batch_sizes.small);
        assert_eq!(config.batch_sizes.medium, cloned_config.batch_sizes.medium);
        assert_eq!(config.batch_sizes.large, cloned_config.batch_sizes.large);
        assert_eq!(config.batch_sizes.xlarge, cloned_config.batch_sizes.xlarge);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_config_serialization() {
        let config = CudaConfig::default();
        
        // Test JSON serialization
        let json_result = serde_json::to_string(&config);
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("gpu_memory_usage_percent"));
        assert!(json_str.contains("estimated_bytes_per_record"));
        assert!(json_str.contains("batch_sizes"));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_config_deserialization() {
        let json_config = r#"
        {
            "gpu_memory_usage_percent": 75,
            "estimated_bytes_per_record": 256,
            "min_batch_size": 5000,
            "max_batch_size": 500000,
            "max_url_buffer_size": 128,
            "max_username_buffer_size": 32,
            "threads_per_block": 128,
            "batch_sizes": {
                "small": 5000,
                "medium": 25000,
                "large": 50000,
                "xlarge": 100000
            }
        }
        "#;
        
        let config_result: Result<CudaConfig, _> = serde_json::from_str(json_config);
        assert!(config_result.is_ok());
        
        let config = config_result.unwrap();
        assert_eq!(config.gpu_memory_usage_percent, 75);
        assert_eq!(config.estimated_bytes_per_record, 256);
        assert_eq!(config.min_batch_size, 5000);
        assert_eq!(config.max_batch_size, 500000);
        assert_eq!(config.max_url_buffer_size, 128);
        assert_eq!(config.max_username_buffer_size, 32);
        assert_eq!(config.threads_per_block, 128);
        
        assert_eq!(config.batch_sizes.small, 5000);
        assert_eq!(config.batch_sizes.medium, 25000);
        assert_eq!(config.batch_sizes.large, 50000);
        assert_eq!(config.batch_sizes.xlarge, 100000);
    }
}