#[cfg(test)]
mod tests {
    use crate::cuda::processor::{CudaRecord, CudaDeviceProperties};
    use crate::constants::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_creation() {
        let record = CudaRecord {
            user: "test@example.com".to_string(),
            password: "password123".to_string(),
            url: "https://example.com".to_string(),
            normalized_user: "".to_string(),
            normalized_url: "".to_string(),
            field_count: 3,
            all_fields: vec!["test@example.com".to_string(), "password123".to_string(), "https://example.com".to_string()],
        };

        assert_eq!(record.user, "test@example.com");
        assert_eq!(record.password, "password123");
        assert_eq!(record.url, "https://example.com");
        assert_eq!(record.field_count, 3);
        assert_eq!(record.all_fields.len(), 3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_properties_creation() {
        let props = CudaDeviceProperties {
            total_memory: TEST_MEMORY_8GB,
            free_memory: TEST_MEMORY_2GB,
            compute_capability_major: TEST_COMPUTE_CAPABILITY_MAJOR as i32,
            compute_capability_minor: TEST_COMPUTE_CAPABILITY_MINOR as i32,
            max_threads_per_block: TEST_MAX_THREADS_PER_BLOCK as i32,
            max_shared_memory_per_block: TEST_MAX_SHARED_MEMORY_PER_BLOCK as i32,
            memory_bus_width: 256,
            l2_cache_size: 4194304, // 4MB
        };

        assert_eq!(props.total_memory, TEST_MEMORY_8GB);
        assert_eq!(props.free_memory, TEST_MEMORY_2GB);
        assert_eq!(props.compute_capability_major, 7);
        assert_eq!(props.compute_capability_minor, 5);
        assert_eq!(props.max_threads_per_block, 1024);
        assert_eq!(props.max_shared_memory_per_block, 49152);
        assert_eq!(props.memory_bus_width, 256);
        assert_eq!(props.l2_cache_size, 4194304);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_clone() {
        let record = CudaRecord {
            user: "user@test.com".to_string(),
            password: "pass".to_string(),
            url: "http://test.com".to_string(),
            normalized_user: "user@test.com".to_string(),
            normalized_url: "test.com".to_string(),
            field_count: 3,
            all_fields: vec!["user@test.com".to_string(), "pass".to_string(), "http://test.com".to_string()],
        };

        let cloned_record = record.clone();
        
        assert_eq!(record.user, cloned_record.user);
        assert_eq!(record.password, cloned_record.password);
        assert_eq!(record.url, cloned_record.url);
        assert_eq!(record.normalized_user, cloned_record.normalized_user);
        assert_eq!(record.normalized_url, cloned_record.normalized_url);
        assert_eq!(record.field_count, cloned_record.field_count);
        assert_eq!(record.all_fields, cloned_record.all_fields);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_properties_clone() {
        let props = CudaDeviceProperties {
            total_memory: TEST_MEMORY_4GB,
            free_memory: TEST_MEMORY_500MB,
            compute_capability_major: 8,
            compute_capability_minor: 6,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 65536,
            memory_bus_width: 384,
            l2_cache_size: 6291456, // 6MB
        };

        let cloned_props = props.clone();
        
        assert_eq!(props.total_memory, cloned_props.total_memory);
        assert_eq!(props.free_memory, cloned_props.free_memory);
        assert_eq!(props.compute_capability_major, cloned_props.compute_capability_major);
        assert_eq!(props.compute_capability_minor, cloned_props.compute_capability_minor);
        assert_eq!(props.max_threads_per_block, cloned_props.max_threads_per_block);
        assert_eq!(props.max_shared_memory_per_block, cloned_props.max_shared_memory_per_block);
        assert_eq!(props.memory_bus_width, cloned_props.memory_bus_width);
        assert_eq!(props.l2_cache_size, cloned_props.l2_cache_size);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_field_manipulation() {
        let mut record = CudaRecord {
            user: "original@email.com".to_string(),
            password: "original_pass".to_string(),
            url: "https://original.com".to_string(),
            normalized_user: "".to_string(),
            normalized_url: "".to_string(),
            field_count: 3,
            all_fields: vec!["original@email.com".to_string(), "original_pass".to_string(), "https://original.com".to_string()],
        };

        // Test field modification
        record.user = "modified@email.com".to_string();
        record.normalized_user = "modified@email.com".to_string();
        record.normalized_url = "original.com".to_string();

        assert_eq!(record.user, "modified@email.com");
        assert_eq!(record.normalized_user, "modified@email.com");
        assert_eq!(record.normalized_url, "original.com");
        assert_eq!(record.password, "original_pass"); // Unchanged
        assert_eq!(record.url, "https://original.com"); // Unchanged
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_with_extra_fields() {
        let record = CudaRecord {
            user: "user@example.com".to_string(),
            password: "password".to_string(),
            url: "https://example.com".to_string(),
            normalized_user: "user@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            field_count: 5,
            all_fields: vec![
                "user@example.com".to_string(),
                "password".to_string(),
                "https://example.com".to_string(),
                "extra_field_1".to_string(),
                "extra_field_2".to_string(),
            ],
        };

        assert_eq!(record.field_count, 5);
        assert_eq!(record.all_fields.len(), 5);
        assert_eq!(record.all_fields[3], "extra_field_1");
        assert_eq!(record.all_fields[4], "extra_field_2");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_empty_fields() {
        let record = CudaRecord {
            user: "".to_string(),
            password: "".to_string(),
            url: "".to_string(),
            normalized_user: "".to_string(),
            normalized_url: "".to_string(),
            field_count: 3,
            all_fields: vec!["".to_string(), "".to_string(), "".to_string()],
        };

        assert_eq!(record.user, "");
        assert_eq!(record.password, "");
        assert_eq!(record.url, "");
        assert_eq!(record.normalized_user, "");
        assert_eq!(record.normalized_url, "");
        assert_eq!(record.field_count, 3);
        assert_eq!(record.all_fields.len(), 3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_properties_memory_calculations() {
        let props = CudaDeviceProperties {
            total_memory: BYTES_PER_GB,
            free_memory: BYTES_PER_GB / 2,
            compute_capability_major: 7,
            compute_capability_minor: 5,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            memory_bus_width: 256,
            l2_cache_size: 4194304,
        };

        let used_memory = props.total_memory - props.free_memory;
        let usage_percent = (used_memory as f64 / props.total_memory as f64) * 100.0;
        
        assert_eq!(used_memory, BYTES_PER_GB / 2);
        assert!((usage_percent - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_record_debug_format() {
        let record = CudaRecord {
            user: "debug@test.com".to_string(),
            password: "debug_pass".to_string(),
            url: "https://debug.com".to_string(),
            normalized_user: "debug@test.com".to_string(),
            normalized_url: "debug.com".to_string(),
            field_count: 3,
            all_fields: vec!["debug@test.com".to_string(), "debug_pass".to_string(), "https://debug.com".to_string()],
        };

        let debug_str = format!("{:?}", record);
        assert!(debug_str.contains("CudaRecord"));
        assert!(debug_str.contains("debug@test.com"));
        assert!(debug_str.contains("debug_pass"));
        assert!(debug_str.contains("https://debug.com"));
    }

    // Test that compiles when CUDA feature is disabled
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_non_cuda_compilation() {
        // This test ensures the module compiles when CUDA is disabled
        // Since most structs are behind #[cfg(feature = "cuda")], 
        // there's nothing to test here, but this verifies compilation works
        assert!(true);
    }
}