#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaContext, DriverError, CudaFunction};
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUdevice_attribute_enum;
#[cfg(feature = "cuda")]
use cudarc::driver::result;
#[cfg(feature = "cuda")]
use cudarc::nvrtc::safe::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::{PushKernelArg, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::config::model::CudaConfig;
#[cfg(feature = "cuda")]
use crate::constants::{
    PERCENT_95, PERCENT_100, DEFAULT_CUDA_GRID_DIM, DEFAULT_CUDA_SHARED_MEM_BYTES,
    DEFAULT_MIN_BUFFER_SIZE, BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB,
    CUDA_MEMORY_POOL_SIZE_MB, CUDA_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK
};

#[cfg(feature = "cuda")]
/// CUDA kernel source code for URL normalization
const CUDA_KERNEL_SOURCE: &str = r#"
// Constant buffer sizes and limits
#define MAX_URL_LENGTH 255
#define MAX_PROTOCOL_LENGTH 10

// Common URL protocols as device constants
__device__ const char* PROTOCOLS[] = {"http://", "https://", "android://", "ftp://", "mailto://"};
__device__ const int PROTOCOL_LENGTHS[] = {7, 8, 10, 6, 8};
__device__ const int NUM_PROTOCOLS = 5;

__device__ bool starts_with(const char* str, int str_len, const char* prefix, int prefix_len) {
    if (str_len < prefix_len) return false;
    for (int i = 0; i < prefix_len; i++) {
        if (str[i] != prefix[i]) return false;
    }
    return true;
}

__device__ int find_char(const char* str, int str_len, char target) {
    for (int i = 0; i < str_len; i++) {
        if (str[i] == target) return i;
    }
    return -1;
}

__device__ int find_last_char(const char* str, int str_len, char target) {
    for (int i = str_len - 1; i >= 0; i--) {
        if (str[i] == target) return i;
    }
    return -1;
}

__device__ inline char to_lower(char c) {
    // Branchless lowercase conversion
    return c | ((c >= 'A' && c <= 'Z') << 5);
}

__device__ int detect_protocol(const char* input, int len) {
    for (int i = 0; i < NUM_PROTOCOLS; i++) {
        if (starts_with(input, len, PROTOCOLS[i], PROTOCOL_LENGTHS[i])) {
            return PROTOCOL_LENGTHS[i];
        }
    }
    return 0;
}

extern "C" __global__ void normalize_emails(
    const char* __restrict__ input_data,
    char* __restrict__ output_data,
    const int* __restrict__ input_offsets,
    const int* __restrict__ output_offsets,
    const int* __restrict__ input_lengths,
    int* __restrict__ output_lengths,
    int num_strings
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    const int start = input_offsets[idx];
    const int len = input_lengths[idx];
    const int out_start = output_offsets[idx];

    if (len == 0) {
        output_lengths[idx] = 0;
        return;
    }

    const char* input = input_data + start;
    char* output = output_data + out_start;

    // Process characters one by one for simplicity and compatibility
    int i = 0;

    // Handle remaining characters
    for (; i < len; i++) {
        output[i] = to_lower(input[i]);
    }

    output_lengths[idx] = len;
}

extern "C" __global__ void normalize_urls(
    const char* __restrict__ input_data,
    char* __restrict__ output_data,
    const int* __restrict__ input_offsets,
    const int* __restrict__ output_offsets,
    const int* __restrict__ input_lengths,
    int* __restrict__ output_lengths,
    int num_strings
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    const int start = input_offsets[idx];
    const int len = input_lengths[idx];
    const int out_start = output_offsets[idx];

    if (len == 0) {
        output_lengths[idx] = 0;
        return;
    }

    const char* input = input_data + start;
    char* output = output_data + out_start;
    int out_len = 0;

    // Detect and skip protocol
    int domain_start = detect_protocol(input, len);

    // Handle android:// special case
    if (domain_start == PROTOCOL_LENGTHS[2]) { // android:// length
        int at_pos = find_last_char(input + domain_start, len - domain_start, '@');
        if (at_pos >= 0) {
            domain_start += at_pos + 1;
        }
    }

    // Skip www. prefix if present
    const int www_len = 4;
    if (len - domain_start >= www_len && starts_with(input + domain_start, len - domain_start, "www.", www_len)) {
        domain_start += www_len;
    }

    // Find end of domain (first occurrence of /, ?, or #)
    int domain_end = len;
    for (int i = domain_start; i < len; i++) {
        char c = input[i];
        if (c == '/' || c == '?' || c == '#') {
            domain_end = i;
            break;
        }
    }

    // Remove trailing slash
    if (domain_end > domain_start && input[domain_end - 1] == '/') {
        domain_end--;
    }

    // Copy and normalize domain
    for (int i = domain_start; i < domain_end && out_len < MAX_URL_LENGTH; i++) {
        output[out_len++] = to_lower(input[i]);
    }

    output_lengths[idx] = out_len;
}
"#;

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaRecord {
    pub user: String,
    pub password: String,
    pub url: String,
    pub normalized_user: String,
    pub normalized_url: String,
    pub field_count: usize,
    pub all_fields: Vec<String>,
}

#[cfg(feature = "cuda")]
pub struct CudaProcessor {
    context: Arc<CudaContext>,
    normalize_urls_function: CudaFunction,
    normalize_emails_function: CudaFunction,
    optimal_batch_size: usize,
    max_string_length: usize,
    optimal_block_size: usize,
    optimal_grid_size: usize,
    memory_pool_size: usize,
}

#[cfg(feature = "cuda")]
impl Clone for CudaProcessor {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            normalize_urls_function: self.normalize_urls_function.clone(),
            normalize_emails_function: self.normalize_emails_function.clone(),
            optimal_batch_size: self.optimal_batch_size,
            max_string_length: self.max_string_length,
            optimal_block_size: self.optimal_block_size,
            optimal_grid_size: self.optimal_grid_size,
            memory_pool_size: self.memory_pool_size,
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaProcessor {
    pub fn new(config: CudaConfig, device_ordinal: i32) -> Result<Self> {
        let context = CudaContext::new(device_ordinal as usize)
            .map_err(|e: DriverError| anyhow::anyhow!("Failed to initialize CUDA device {}: {}. Ensure NVIDIA drivers and CUDA toolkit are installed.", device_ordinal, e))?;

        println!("CUDA device {} initialized successfully", device_ordinal);

        let props = Self::get_device_properties_internal(&context)?;

        if config.gpu_memory_usage_percent > 100 {
            return Err(anyhow::anyhow!("Invalid GPU memory usage percent: {}. Must be <= 100.", config.gpu_memory_usage_percent));
        }

        println!("CUDA device properties detected:");
        println!("  Compute Capability: {}.{}", props.compute_capability_major, props.compute_capability_minor);
        println!("  Total Memory: {:.2} GB ({} bytes)", props.total_memory as f64 / BYTES_PER_GB as f64, props.total_memory);
        println!("  Free Memory: {:.2} GB ({} bytes)", props.free_memory as f64 / BYTES_PER_GB as f64, props.free_memory);
        println!("  Max Threads per Block: {}", props.max_threads_per_block);
        println!("  Max Shared Memory per Block: {} KB", props.max_shared_memory_per_block as f64 / BYTES_PER_KB as f64);
        println!("  Memory Bus Width: {} bits", props.memory_bus_width);
        println!("  L2 Cache Size: {} MB", props.l2_cache_size as usize / (BYTES_PER_MB as usize));

        let available_memory_bytes = (props.free_memory as f64 * (config.gpu_memory_usage_percent as f64 / PERCENT_100) * PERCENT_95) as usize;

        let max_batch_size = (available_memory_bytes / config.estimated_bytes_per_record)
            .max(config.min_batch_size)
            .min(config.max_batch_size);

        let optimal_batch_size = Self::calculate_optimal_batch_size(
            max_batch_size,
            props.max_threads_per_block as usize,
            &config
        );

        // Calculate optimal block and grid sizes based on device properties
        let optimal_block_size = Self::calculate_optimal_block_size(&props, &config);
        let optimal_grid_size = Self::calculate_optimal_grid_size(optimal_batch_size, optimal_block_size);

        // Calculate memory pool size
        let memory_pool_size = (CUDA_MEMORY_POOL_SIZE_MB * BYTES_PER_MB).min(available_memory_bytes / 4);

        println!("CUDA processor initialized for device {} - Available memory for use: {} bytes, Max batch size: {}, Optimal batch size: {}, Block size: {}, Grid size: {}, Memory pool: {} MB",
              device_ordinal,
              available_memory_bytes,
              max_batch_size,
              optimal_batch_size,
              optimal_block_size,
              optimal_grid_size,
              memory_pool_size / BYTES_PER_MB);

        println!("Compiling CUDA kernel for string normalization...");
        let ptx = compile_ptx(CUDA_KERNEL_SOURCE)
            .map_err(|e| anyhow::anyhow!("Failed to compile CUDA kernel: {}", e))?;

        let module = context.load_module(ptx)
            .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernel: {}", e))?;

        let normalize_urls_function = module.load_function("normalize_urls")
            .map_err(|e| anyhow::anyhow!("Failed to get CUDA normalize_urls function: {}", e))?;

        let normalize_emails_function = module.load_function("normalize_emails")
            .map_err(|e| anyhow::anyhow!("Failed to get CUDA normalize_emails function: {}", e))?;

        println!("CUDA kernels compiled and loaded successfully");

        Ok(Self {
            context,
            normalize_urls_function,
            normalize_emails_function,
            optimal_batch_size,
            max_string_length: config.max_url_buffer_size.max(config.max_username_buffer_size),
            optimal_block_size,
            optimal_grid_size,
            memory_pool_size,
        })
    }

    pub fn get_device_properties_internal(context: &Arc<CudaContext>) -> Result<CudaDeviceProperties, DriverError> {
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

    fn calculate_optimal_block_size(props: &CudaDeviceProperties, config: &CudaConfig) -> usize {
        // Start with configured threads per block
        let mut block_size = config.threads_per_block;

        // Ensure it's a multiple of warp size for optimal performance
        block_size = (block_size / CUDA_WARP_SIZE) * CUDA_WARP_SIZE;

        // Clamp to device limits
        block_size = block_size.min(props.max_threads_per_block as usize);
        block_size = block_size.min(CUDA_MAX_THREADS_PER_BLOCK);

        // Ensure minimum warp size
        block_size.max(CUDA_WARP_SIZE)
    }

    fn calculate_optimal_grid_size(batch_size: usize, block_size: usize) -> usize {
        if block_size == 0 {
            return 1;
        }
        (batch_size + block_size - 1) / block_size
    }

    pub fn process_batch(&self, records: &mut [CudaRecord], case_sensitive_usernames: bool) -> Result<()> {
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

    fn process_chunk_on_gpu(&self, records: &mut [CudaRecord], case_sensitive_usernames: bool) -> Result<()> {
        // Process URLs and emails separately for better GPU utilization
        self.process_urls_on_gpu(records)?;
        if !case_sensitive_usernames {
            self.process_emails_on_gpu(records)?;
        } else {
            // For case-sensitive usernames, just copy the original
            for record in records.iter_mut() {
                record.normalized_user = record.user.clone();
            }
        }
        Ok(())
    }

    fn process_urls_on_gpu(&self, records: &mut [CudaRecord]) -> Result<()> {
        // Prepare data for GPU: flatten URLs, build offsets/lengths
        let mut input_data = Vec::new();
        let mut input_offsets = Vec::with_capacity(records.len());
        let mut input_lengths = Vec::with_capacity(records.len());
        let mut output_offsets = Vec::with_capacity(records.len());
        let mut total_output_len = 0;

        for record in records.iter() {
            let url_bytes = record.url.as_bytes();
            input_offsets.push(input_data.len() as i32);
            input_lengths.push(url_bytes.len() as i32);
            input_data.extend_from_slice(url_bytes);
            output_offsets.push(total_output_len as i32);
            total_output_len += url_bytes.len().max(DEFAULT_MIN_BUFFER_SIZE); // Ensure minimum buffer size
        }

        // Use the default stream for now (we'll optimize with multiple streams later)
        let ctx = &self.context;
        let stream = ctx.default_stream();

        let d_input = stream.memcpy_stod(&input_data)?;
        let d_output = stream.alloc_zeros::<u8>(total_output_len)?;
        let d_input_offsets = stream.memcpy_stod(&input_offsets)?;
        let d_output_offsets = stream.memcpy_stod(&output_offsets)?;
        let d_input_lengths = stream.memcpy_stod(&input_lengths)?;
        let d_output_lengths = stream.alloc_zeros::<i32>(records.len())?;

        let num_strings = records.len() as i32;
        let block_size = self.optimal_block_size;
        let num_blocks = self.optimal_grid_size.min((num_strings as usize + block_size - 1) / block_size);

        unsafe {
            let mut builder = stream.launch_builder(&self.normalize_urls_function);
            builder.arg(&d_input)
                .arg(&d_output)
                .arg(&d_input_offsets)
                .arg(&d_output_offsets)
                .arg(&d_input_lengths)
                .arg(&d_output_lengths)
                .arg(&num_strings);
            builder.launch(LaunchConfig {
                grid_dim: (num_blocks as u32, DEFAULT_CUDA_GRID_DIM as u32, DEFAULT_CUDA_GRID_DIM as u32),
                block_dim: (block_size as u32, DEFAULT_CUDA_GRID_DIM as u32, DEFAULT_CUDA_GRID_DIM as u32),
                shared_mem_bytes: DEFAULT_CUDA_SHARED_MEM_BYTES,
            })?;
        }
        stream.synchronize()?;

        let output_data_host = stream.memcpy_dtov(&d_output)?;
        let output_lengths_host = stream.memcpy_dtov(&d_output_lengths)?;

        // Update records with normalized URLs from output_data_host
        for (i, record) in records.iter_mut().enumerate() {
            let start = output_offsets[i] as usize;
            let len = output_lengths_host[i] as usize;
            let url_bytes = &output_data_host[start..start+len];
            record.normalized_url = String::from_utf8_lossy(url_bytes).to_string();
        }
        Ok(())
    }

    fn process_emails_on_gpu(&self, records: &mut [CudaRecord]) -> Result<()> {
        let mut input_data = Vec::new();
        let mut input_offsets = Vec::with_capacity(records.len());
        let mut input_lengths = Vec::with_capacity(records.len());
        let mut output_offsets = Vec::with_capacity(records.len());
        let mut total_output_len = 0;

        for record in records.iter() {
            let email_bytes = record.user.as_bytes();
            input_offsets.push(input_data.len() as i32);
            input_lengths.push(email_bytes.len() as i32);
            input_data.extend_from_slice(email_bytes);
            output_offsets.push(total_output_len as i32);
            total_output_len += email_bytes.len();
        }

        // Use the default stream for now
        let ctx = &self.context;
        let stream = ctx.default_stream();

        let d_input = stream.memcpy_stod(&input_data)?;
        let d_output = stream.alloc_zeros::<u8>(total_output_len)?;
        let d_input_offsets = stream.memcpy_stod(&input_offsets)?;
        let d_output_offsets = stream.memcpy_stod(&output_offsets)?;
        let d_input_lengths = stream.memcpy_stod(&input_lengths)?;
        let d_output_lengths = stream.alloc_zeros::<i32>(records.len())?;

        let num_strings = records.len() as i32;
        let block_size = self.optimal_block_size;
        let num_blocks = self.optimal_grid_size.min((num_strings as usize + block_size - 1) / block_size);

        unsafe {
            let mut builder = stream.launch_builder(&self.normalize_emails_function);
            builder.arg(&d_input)
                .arg(&d_output)
                .arg(&d_input_offsets)
                .arg(&d_output_offsets)
                .arg(&d_input_lengths)
                .arg(&d_output_lengths)
                .arg(&num_strings);
            builder.launch(LaunchConfig {
                grid_dim: (num_blocks as u32, DEFAULT_CUDA_GRID_DIM as u32, DEFAULT_CUDA_GRID_DIM as u32),
                block_dim: (block_size as u32, DEFAULT_CUDA_GRID_DIM as u32, DEFAULT_CUDA_GRID_DIM as u32),
                shared_mem_bytes: DEFAULT_CUDA_SHARED_MEM_BYTES,
            })?;
        }
        stream.synchronize()?;

        let output_data_host = stream.memcpy_dtov(&d_output)?;
        let output_lengths_host = stream.memcpy_dtov(&d_output_lengths)?;

        for (i, record) in records.iter_mut().enumerate() {
            let start = output_offsets[i] as usize;
            let len = output_lengths_host[i] as usize;
            let email_bytes = &output_data_host[start..start+len];
            record.normalized_user = String::from_utf8_lossy(email_bytes).to_string();
        }
        Ok(())
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size
    }



    pub fn release_gpu_resources(&self) -> Result<()> {
        let stream = self.context.default_stream();
        stream.synchronize()?;

        // When GPU buffers go out of scope, they are automatically freed
        // This method ensures synchronization and can be extended for manual cleanup if needed

        Ok(())
    }

    pub fn get_gpu_memory_usage(&self) -> Result<(usize, usize), DriverError> {
        use cudarc::driver::result;
        result::mem_get_info()
    }

    pub fn check_gpu_memory_pressure(&self) -> Result<bool> {
        let (free_memory, total_memory) = self.get_gpu_memory_usage()
            .map_err(|e| anyhow::anyhow!("Failed to get GPU memory info: {}", e))?;

        let used_memory = total_memory - free_memory;
        let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;

        Ok(usage_percent > crate::constants::MEMORY_PRESSURE_THRESHOLD_PERCENT)
    }
}

// When CUDA is not available, provide empty stubs
#[cfg(not(feature = "cuda"))]
pub struct CudaProcessor;

#[cfg(not(feature = "cuda"))]
pub struct CudaDeviceProperties;

#[cfg(not(feature = "cuda"))]
pub struct CudaRecord;

#[cfg(not(feature = "cuda"))]
impl CudaProcessor {
    pub fn new(_: (), _: i32) -> Result<Self, ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_string_processing() -> Result<()> {
        // Initialize CUDA processor with test configuration
        let config = CudaConfig {
            gpu_memory_usage_percent: 80,
            max_url_buffer_size: 256,
            max_username_buffer_size: 128,
            estimated_bytes_per_record: 512,
            min_batch_size: 1,
            max_batch_size: 1000,
            threads_per_block: 256,
            batch_sizes: crate::config::model::BatchSizes {
                small: 64,
                medium: 128,
                large: 256,
                xlarge: 512,
            },
        };

        let processor = CudaProcessor::new(config, 0)?;

        // Test case 1: Basic URL normalization
        let mut records = vec![
            CudaRecord {
                user: "Test@Example.com".to_string(),
                password: "pass123".to_string(),
                url: "https://www.Example.com/path?query#fragment".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["Test@Example.com".to_string(), "pass123".to_string(), "https://www.Example.com/path?query#fragment".to_string()],
            },
            CudaRecord {
                user: "USER@DOMAIN.COM".to_string(),
                password: "pass456".to_string(),
                url: "http://Sub.Domain.com/".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["USER@DOMAIN.COM".to_string(), "pass456".to_string(), "http://Sub.Domain.com/".to_string()],
            },
        ];

        // Process records
        processor.process_batch(&mut records, false)?;

        // Verify results
        assert_eq!(records[0].normalized_url, "example.com");
        assert_eq!(records[0].normalized_user, "test@example.com");
        assert_eq!(records[1].normalized_url, "sub.domain.com");
        assert_eq!(records[1].normalized_user, "user@domain.com");

        // Test case 2: Android URL handling
        let mut android_records = vec![
            CudaRecord {
                user: "User1@test.com".to_string(),
                password: "pass789".to_string(),
                url: "android://AbC123@com.example.app/".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["User1@test.com".to_string(), "pass789".to_string(), "android://AbC123@com.example.app/".to_string()],
            },
        ];

        processor.process_batch(&mut android_records, false)?;
        assert_eq!(android_records[0].normalized_url, "com.example.app");
        assert_eq!(android_records[0].normalized_user, "user1@test.com");

        // Test case 3: Case-sensitive username mode
        let mut case_sensitive_records = vec![
            CudaRecord {
                user: "MixedCase@Domain.com".to_string(),
                password: "pass123".to_string(),
                url: "https://test.com".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["MixedCase@Domain.com".to_string(), "pass123".to_string(), "https://test.com".to_string()],
            },
        ];

        processor.process_batch(&mut case_sensitive_records, true)?;
        assert_eq!(case_sensitive_records[0].normalized_url, "test.com");
        assert_eq!(case_sensitive_records[0].normalized_user, "MixedCase@Domain.com"); // Should preserve case

        // Test case 4: Empty strings
        let mut empty_records = vec![
            CudaRecord {
                user: "".to_string(),
                password: "pass123".to_string(),
                url: "".to_string(),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["".to_string(), "pass123".to_string(), "".to_string()],
            },
        ];

        processor.process_batch(&mut empty_records, false)?;
        assert_eq!(empty_records[0].normalized_url, "");
        assert_eq!(empty_records[0].normalized_user, "");

        // Test case 5: Maximum length handling
        let long_url = "https://".to_string() + &"a".repeat(300) + ".com/path";
        let mut long_records = vec![
            CudaRecord {
                user: "test@example.com".to_string(),
                password: "pass123".to_string(),
                url: long_url,
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec!["test@example.com".to_string(), "pass123".to_string(), "long_url".to_string()],
            },
        ];

        processor.process_batch(&mut long_records, false)?;
        assert!(long_records[0].normalized_url.len() <= 255); // Should truncate to MAX_URL_LENGTH

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_batch_processing() -> Result<()> {
        let config = CudaConfig {
            gpu_memory_usage_percent: 80,
            max_url_buffer_size: 256,
            max_username_buffer_size: 128,
            estimated_bytes_per_record: 512,
            min_batch_size: 1,
            max_batch_size: 1000,
            threads_per_block: 256,
            batch_sizes: crate::config::model::BatchSizes {
                small: 64,
                medium: 128,
                large: 256,
                xlarge: 512,
            },
        };

        let processor = CudaProcessor::new(config, 0)?;

        // Create a large batch of records
        let mut records: Vec<CudaRecord> = (0..1000).map(|i| {
            CudaRecord {
                user: format!("User{}@example.com", i),
                password: format!("pass{}", i),
                url: format!("https://site{}.example.com/path", i),
                normalized_user: String::new(),
                normalized_url: String::new(),
                field_count: 3,
                all_fields: vec![
                    format!("User{}@example.com", i),
                    format!("pass{}", i),
                    format!("https://site{}.example.com/path", i),
                ],
            }
        }).collect();

        // Process the large batch
        processor.process_batch(&mut records, false)?;

        // Verify results
        for (i, record) in records.iter().enumerate() {
            assert_eq!(record.normalized_url, format!("site{}.example.com", i));
            assert_eq!(record.normalized_user, format!("user{}@example.com", i));
        }

        Ok(())
    }
}