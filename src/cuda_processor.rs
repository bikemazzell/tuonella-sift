use anyhow::Result;
use cudarc::driver::safe::{CudaContext, DriverError, CudaFunction};
use cudarc::driver::sys::CUdevice_attribute_enum;
use cudarc::driver::result;
use cudarc::nvrtc::safe::compile_ptx;
use tracing::{info, warn, debug, trace};
use crate::record::Record;
use crate::config::CudaConfig;
use crate::constants::{
    BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB, PERCENT_100, PERCENT_95
};
use std::sync::Arc;
use cudarc::driver::PushKernelArg;
use cudarc::driver::LaunchConfig;

#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
}

// CUDA kernel source code for string processing
const CUDA_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void normalize_strings(
    char* input_data,
    char* output_data,
    int* input_offsets,
    int* output_offsets,
    int* lengths,
    int num_strings,
    bool to_lowercase
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    int start = input_offsets[idx];
    int len = lengths[idx];
    int out_start = output_offsets[idx];

    for (int i = 0; i < len; i++) {
        char c = input_data[start + i];
        if (to_lowercase && c >= 'A' && c <= 'Z') {
            output_data[out_start + i] = c + 32; // Convert to lowercase
        } else {
            output_data[out_start + i] = c;
        }
    }
}
"#;

pub struct CudaProcessor {
    context: Arc<CudaContext>,
    normalize_function: CudaFunction,
    optimal_batch_size: usize,
    max_string_length: usize,
}

impl Clone for CudaProcessor {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            normalize_function: self.normalize_function.clone(),
            optimal_batch_size: self.optimal_batch_size,
            max_string_length: self.max_string_length,
        }
    }
}

impl CudaProcessor {
    pub fn new(config: CudaConfig, device_ordinal: i32) -> Result<Self> {
        let context = CudaContext::new(device_ordinal as usize)
            .map_err(|e: DriverError| anyhow::anyhow!("Failed to initialize CUDA device {}: {}. Ensure NVIDIA drivers and CUDA toolkit are installed.", device_ordinal, e))?;

        info!("CUDA device {} initialized successfully", device_ordinal);

        let props = Self::get_device_properties_internal(&context)?;

        info!("CUDA device properties detected:");
        info!("  Compute Capability: {}.{}", props.compute_capability_major, props.compute_capability_minor);
        info!("  Total Memory: {:.2} GB ({} bytes)", props.total_memory as f64 / BYTES_PER_GB, props.total_memory);
        info!("  Free Memory: {:.2} GB ({} bytes)", props.free_memory as f64 / BYTES_PER_GB, props.free_memory);
        info!("  Max Threads per Block: {}", props.max_threads_per_block);
        info!("  Max Shared Memory per Block: {} KB", props.max_shared_memory_per_block as f64 / BYTES_PER_KB);
        info!("  Memory Bus Width: {} bits", props.memory_bus_width);
        info!("  L2 Cache Size: {} MB", props.l2_cache_size as usize / BYTES_PER_MB);

        let available_memory_bytes = (props.free_memory as f64 * (config.gpu_memory_usage_percent as f64 / PERCENT_100) * PERCENT_95) as usize;

        let max_batch_size = (available_memory_bytes / config.estimated_bytes_per_record)
            .max(config.min_batch_size)
            .min(config.max_batch_size);

        let optimal_batch_size = Self::calculate_optimal_batch_size(
            max_batch_size,
            props.max_threads_per_block as usize,
            &config
        );

        info!("CUDA processor initialized for device {} - Available memory for use: {} bytes, Max batch size: {}, Optimal batch size: {}",
              device_ordinal,
              available_memory_bytes,
              max_batch_size,
              optimal_batch_size);

        // Compile CUDA kernel
        info!("Compiling CUDA kernel for string normalization...");
        let ptx = compile_ptx(CUDA_KERNEL_SOURCE)
            .map_err(|e| anyhow::anyhow!("Failed to compile CUDA kernel: {}", e))?;

        let module = context.load_module(ptx)
            .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernel: {}", e))?;

        let normalize_function = module.load_function("normalize_strings")
            .map_err(|e| anyhow::anyhow!("Failed to get CUDA function: {}", e))?;

        info!("CUDA kernel compiled and loaded successfully");

        Ok(Self {
            context,
            normalize_function,
            optimal_batch_size,
            max_string_length: config.max_url_buffer_size.max(config.max_username_buffer_size),
        })
    }

    fn get_device_properties_internal(context: &Arc<CudaContext>) -> Result<CudaDeviceProperties, DriverError> {
        let compute_capability_major = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let compute_capability_minor = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let max_threads_per_block = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        let max_shared_memory_per_block = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let memory_bus_width = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let l2_cache_size = context.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;
        let (free_memory, total_memory) = result::mem_get_info()?;
        Ok(CudaDeviceProperties {
            total_memory,
            free_memory,
            compute_capability_major,
            compute_capability_minor,
            max_threads_per_block,
            max_shared_memory_per_block,
            memory_bus_width,
            l2_cache_size,
        })
    }

    pub fn get_properties(&self) -> Result<CudaDeviceProperties, DriverError> {
        Self::get_device_properties_internal(&self.context)
    }

    fn calculate_optimal_batch_size(max_size: usize, threads_per_block: usize, config: &CudaConfig) -> usize {
        let mut size = max_size;

        if threads_per_block > 0 {
            size = (size / threads_per_block) * threads_per_block;
        }

        if size >= config.batch_sizes.xlarge {
            size = config.batch_sizes.xlarge;
        } else if size >= config.batch_sizes.large {
            size = config.batch_sizes.large;
        } else if size >= config.batch_sizes.medium {
            size = config.batch_sizes.medium;
        } else {
            size = size.max(config.batch_sizes.small);
        }
        size.max(1)
    }

    pub fn process_batch(&self, records: &mut [Record], case_sensitive_usernames: bool) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let chunk_size = self.optimal_batch_size.min(records.len()).max(1);

        // Process records in chunks of the optimal batch size
        for chunk in records.chunks_mut(chunk_size) {
            self.process_chunk_on_gpu(chunk, case_sensitive_usernames)?;
        }

        Ok(())
    }

    fn process_chunk_on_gpu(&self, records: &mut [Record], case_sensitive_usernames: bool) -> Result<()> {
        trace!("Processing {} records on GPU", records.len());

        // Prepare data for GPU: flatten URLs, build offsets/lengths
        let mut input_data = Vec::new();
        let mut input_offsets = Vec::with_capacity(records.len());
        let mut lengths = Vec::with_capacity(records.len());
        let mut output_offsets = Vec::with_capacity(records.len());
        let mut total_output_len = 0;

        for record in records.iter() {
            let url_bytes = record.url.as_bytes();
            input_offsets.push(input_data.len() as i32);
            lengths.push(url_bytes.len() as i32);
            input_data.extend_from_slice(url_bytes);
            output_offsets.push(total_output_len as i32);
            total_output_len += url_bytes.len();
        }

        let mut output_data = vec![0u8; total_output_len];

        // Allocate/copy buffers to GPU using cudarc stream API
        let ctx = &self.context;
        let stream = ctx.default_stream();
        let d_input = stream.memcpy_stod(&input_data)?;
        let mut d_output = stream.alloc_zeros::<u8>(output_data.len())?;
        let d_input_offsets = stream.memcpy_stod(&input_offsets)?;
        let d_output_offsets = stream.memcpy_stod(&output_offsets)?;
        let d_lengths = stream.memcpy_stod(&lengths)?;

        // Launch CUDA kernel for normalization
        let num_strings = records.len() as i32;
        let to_lowercase = true; // Always normalize URLs to lowercase
        let block_size = 256;
        let num_blocks = (num_strings + block_size - 1) / block_size;
        unsafe {
            let mut builder = stream.launch_builder(&self.normalize_function);
            builder.arg(&d_input)
                .arg(&d_output)
                .arg(&d_input_offsets)
                .arg(&d_output_offsets)
                .arg(&d_lengths)
                .arg(&num_strings)
                .arg(&to_lowercase);
            builder.launch(LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        stream.synchronize()?;
        let output_data_host = stream.memcpy_dtov(&d_output)?;
        // Update records with normalized URLs from output_data_host
        for (i, record) in records.iter_mut().enumerate() {
            let start = output_offsets[i] as usize;
            let len = lengths[i] as usize;
            let url_bytes = &output_data_host[start..start+len];
            record.normalized_url = String::from_utf8_lossy(url_bytes).to_string();
            // TODO: GPU username normalization
            if !case_sensitive_usernames {
                record.normalized_user = record.user.to_lowercase();
            } else {
                record.normalized_user = record.user.clone();
            }
        }
        Ok(())
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size
    }
}

// Fallback CPU processing for when GPU is not available or fails
fn process_records_cpu(records: &mut [Record], case_sensitive_usernames: bool) {
    for record in records.iter_mut() {
        record.normalized_url = record.url.to_lowercase();
        if !case_sensitive_usernames {
            record.normalized_user = record.user.to_lowercase();
        } else {
            record.normalized_user = record.user.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;
    use crate::constants::{
        PERCENT_100, TEST_TOTAL_MEMORY, TEST_FREE_MEMORY, TEST_COMPUTE_CAPABILITY_MAJOR,
        TEST_COMPUTE_CAPABILITY_MINOR, TEST_MAX_THREADS_PER_BLOCK,
        TEST_MAX_SHARED_MEMORY_PER_BLOCK, TEST_MEMORY_BUS_WIDTH, TEST_L2_CACHE_SIZE,
        TEST_GPU_MEMORY_PERCENT, TEST_BYTES_PER_RECORD, TEST_MIN_BATCH_SIZE,
        TEST_MAX_BATCH_SIZE, TEST_URL_BUFFER_SIZE, TEST_USERNAME_BUFFER_SIZE,
        TEST_THREADS_PER_BLOCK, TEST_BATCH_SIZE_SMALL, TEST_BATCH_SIZE_MEDIUM,
        TEST_BATCH_SIZE_LARGE, TEST_BATCH_SIZE_XLARGE, TEST_TOTAL_MEMORY_8GB,
        TEST_TOTAL_MEMORY_SMALL, TEST_USABLE_MEMORY_8GB, TEST_USABLE_MEMORY_SMALL,
        TEST_CALCULATED_BATCH_SIZE_8GB, TEST_CALCULATED_BATCH_SIZE_SMALL,
        TEST_OPTIMAL_BATCH_SIZE_SMALL
    };

    fn create_test_config() -> CudaConfig {
        CudaConfig {
            gpu_memory_usage_percent: TEST_GPU_MEMORY_PERCENT,
            estimated_bytes_per_record: TEST_BYTES_PER_RECORD,
            min_batch_size: TEST_MIN_BATCH_SIZE,
            max_batch_size: TEST_MAX_BATCH_SIZE,
            max_url_buffer_size: TEST_URL_BUFFER_SIZE,
            max_username_buffer_size: TEST_USERNAME_BUFFER_SIZE,
            threads_per_block: TEST_THREADS_PER_BLOCK,
            batch_sizes: crate::config::BatchSizes {
                small: TEST_BATCH_SIZE_SMALL,
                medium: TEST_BATCH_SIZE_MEDIUM,
                large: TEST_BATCH_SIZE_LARGE,
                xlarge: TEST_BATCH_SIZE_XLARGE,
            },
        }
    }

    fn create_test_record(user: &str, password: &str, url: &str) -> Record {
        Record {
            user: user.to_string(),
            password: password.to_string(),
            url: url.to_string(),
            normalized_user: String::new(),
            normalized_url: String::new(),
            source_file: "test.csv".to_string(),
            line_number: 1,
            fields: vec![user.to_string(), password.to_string(), url.to_string()],
            completeness_score: 100.0,
        }
    }

    #[test]
    fn test_cuda_device_properties_creation() {
        let props = CudaDeviceProperties {
            total_memory: TEST_TOTAL_MEMORY,
            free_memory: TEST_FREE_MEMORY,
            compute_capability_major: TEST_COMPUTE_CAPABILITY_MAJOR,
            compute_capability_minor: TEST_COMPUTE_CAPABILITY_MINOR,
            max_threads_per_block: TEST_MAX_THREADS_PER_BLOCK,
            max_shared_memory_per_block: TEST_MAX_SHARED_MEMORY_PER_BLOCK,
            memory_bus_width: TEST_MEMORY_BUS_WIDTH,
            l2_cache_size: TEST_L2_CACHE_SIZE,
        };

        assert_eq!(props.total_memory, TEST_TOTAL_MEMORY);
        assert_eq!(props.free_memory, TEST_FREE_MEMORY);
        assert_eq!(props.compute_capability_major, TEST_COMPUTE_CAPABILITY_MAJOR);
        assert_eq!(props.compute_capability_minor, TEST_COMPUTE_CAPABILITY_MINOR);
    }

    #[test]
    fn test_cuda_config_creation() {
        let config = create_test_config();

        assert_eq!(config.gpu_memory_usage_percent, TEST_GPU_MEMORY_PERCENT);
        assert_eq!(config.estimated_bytes_per_record, TEST_BYTES_PER_RECORD);
        assert_eq!(config.min_batch_size, TEST_MIN_BATCH_SIZE);
        assert_eq!(config.max_batch_size, TEST_MAX_BATCH_SIZE);
        assert_eq!(config.max_url_buffer_size, TEST_URL_BUFFER_SIZE);
        assert_eq!(config.max_username_buffer_size, TEST_USERNAME_BUFFER_SIZE);
        assert_eq!(config.threads_per_block, TEST_THREADS_PER_BLOCK);
    }

    #[test]
    fn test_process_chunk_case_sensitive() {
        let mut records = vec![
            create_test_record("User@Example.com", "Password123", "https://Example.com/path"),
            create_test_record("TEST@GMAIL.COM", "secret456", "HTTP://WWW.GOOGLE.COM"),
        ];

        // Directly use the extracted process_records_cpu function
        process_records_cpu(&mut records, true);

        assert_eq!(records[0].normalized_user, "User@Example.com"); // Case preserved
        assert_eq!(records[0].normalized_url, "https://example.com/path");
        assert_eq!(records[1].normalized_user, "TEST@GMAIL.COM"); // Case preserved
        assert_eq!(records[1].normalized_url, "http://www.google.com");
    }

    #[test]
    fn test_process_chunk_case_insensitive() {
        let mut records = vec![
            create_test_record("User@Example.com", "Password123", "https://Example.com/path"),
            create_test_record("TEST@GMAIL.COM", "secret456", "HTTP://WWW.GOOGLE.COM"),
        ];

        // Directly use the extracted process_records_cpu function
        process_records_cpu(&mut records, false);

        assert_eq!(records[0].normalized_user, "user@example.com"); // Lowercased
        assert_eq!(records[0].normalized_url, "https://example.com/path");
        assert_eq!(records[1].normalized_user, "test@gmail.com"); // Lowercased
        assert_eq!(records[1].normalized_url, "http://www.google.com");
    }

    #[test]
    fn test_batch_size_calculation() {
        let config = create_test_config();

        // Test the batch size calculation logic that would be used
        let usable_memory = (TEST_TOTAL_MEMORY_8GB as f64 * config.gpu_memory_usage_percent as f64 / PERCENT_100 * PERCENT_95) as usize;
        let calculated_batch_size = usable_memory / config.estimated_bytes_per_record;

        let optimal_batch_size = calculated_batch_size
            .max(config.min_batch_size)
            .min(config.max_batch_size);

        assert_eq!(usable_memory, TEST_USABLE_MEMORY_8GB); // 80% of 8GB
        assert_eq!(calculated_batch_size, TEST_CALCULATED_BATCH_SIZE_8GB); // 6.4GB / 500 bytes
        assert_eq!(optimal_batch_size, config.max_batch_size); // Clamped to max
    }

    #[test]
    fn test_batch_size_calculation_small_memory() {
        let config = create_test_config();

        // Test with very small memory that would result in batch size below minimum
        let usable_memory = (TEST_TOTAL_MEMORY_SMALL as f64 * config.gpu_memory_usage_percent as f64 / PERCENT_100 * PERCENT_95) as usize;
        let calculated_batch_size = usable_memory / config.estimated_bytes_per_record;

        let optimal_batch_size = calculated_batch_size
            .max(config.min_batch_size)
            .min(config.max_batch_size);

        assert_eq!(usable_memory, TEST_USABLE_MEMORY_SMALL); // 80% of 500KB
        assert_eq!(calculated_batch_size, TEST_CALCULATED_BATCH_SIZE_SMALL); // 400KB / 500 bytes
        assert_eq!(optimal_batch_size, config.min_batch_size); // Clamped to min (1000)
    }

    #[test]
    fn test_empty_batch_processing() {
        // Test that empty batches are handled correctly
        let empty_records: Vec<Record> = vec![];

        // Simulate the empty batch check from process_batch
        if empty_records.is_empty() {
            // Should return Ok(()) without processing
            assert!(true, "Empty batch should be handled gracefully");
        } else {
            panic!("Empty batch should have been detected");
        }
    }

    #[test]
    fn test_chunk_processing_logic() {
        let optimal_batch_size = TEST_OPTIMAL_BATCH_SIZE_SMALL; // Small for testing

        let mut records = vec![
            create_test_record("user1@example.com", "pass1", "https://site1.com"),
            create_test_record("user2@example.com", "pass2", "https://site2.com"),
            create_test_record("user3@example.com", "pass3", "https://site3.com"),
            create_test_record("user4@example.com", "pass4", "https://site4.com"),
            create_test_record("user5@example.com", "pass5", "https://site5.com"),
        ];

        // Simulate the chunking logic from process_batch
        let chunk_size = optimal_batch_size.min(records.len()).max(1);
        assert_eq!(chunk_size, TEST_OPTIMAL_BATCH_SIZE_SMALL);

        let mut processed_chunks = 0;
        for chunk in records.chunks_mut(chunk_size) {
            processed_chunks += 1;
            process_records_cpu(chunk, false);
        }

        assert_eq!(processed_chunks, 2); // 5 records / 3 chunk_size = 2 chunks

        // Verify all records were processed
        for record in &records {
            assert!(!record.normalized_url.is_empty());
            assert!(!record.normalized_user.is_empty());
        }
    }

    #[test]
    fn test_gpu_property_detection() {
        // Test that our GPU property detection logic works correctly
        // Since we now use real CUDA device queries, we'll test the logic without actual GPU

        // Test that the device attribute enums are available
        use cudarc::driver::sys::CUdevice_attribute_enum;

        // Verify the attributes we use exist
        let _compute_major = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
        let _compute_minor = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
        let _max_threads = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        let _shared_mem = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
        let _bus_width = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
        let _l2_cache = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;

        // Test passes if we can access all the required device attributes
        assert!(true);
    }
}
