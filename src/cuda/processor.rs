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
use crate::constants::{PERCENT_95, PERCENT_100};

#[cfg(feature = "cuda")]
/// CUDA kernel source code for URL normalization
const CUDA_KERNEL_SOURCE: &str = r#"
__device__ bool starts_with(const char* str, int str_len, const char* prefix, int prefix_len) {
    if (str_len < prefix_len) return false;
    for (int i = 0; i < prefix_len; i++) {
        if (str[i] != prefix[i]) return false;
    }
    return true;
}

__device__ int find_char(const char* str, int str_len, char c) {
    for (int i = 0; i < str_len; i++) {
        if (str[i] == c) return i;
    }
    return -1;
}

__device__ int find_last_char(const char* str, int str_len, char c) {
    for (int i = str_len - 1; i >= 0; i--) {
        if (str[i] == c) return i;
    }
    return -1;
}

__device__ char to_lower(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c + 32;
    }
    return c;
}

extern "C" __global__ void normalize_emails(
    char* input_data,
    char* output_data,
    int* input_offsets,
    int* output_offsets,
    int* input_lengths,
    int* output_lengths,
    int num_strings
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    int start = input_offsets[idx];
    int len = input_lengths[idx];
    int out_start = output_offsets[idx];

    if (len == 0) {
        output_lengths[idx] = 0;
        return;
    }

    const char* input = input_data + start;
    char* output = output_data + out_start;
    int out_len = 0;

    // Convert email to lowercase
    for (int i = 0; i < len; i++) {
        output[out_len++] = to_lower(input[i]);
    }

    output_lengths[idx] = out_len;
}

extern "C" __global__ void normalize_urls(
    char* input_data,
    char* output_data,
    int* input_offsets,
    int* output_offsets,
    int* input_lengths,
    int* output_lengths,
    int num_strings
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    int start = input_offsets[idx];
    int len = input_lengths[idx];
    int out_start = output_offsets[idx];

    if (len == 0) {
        output_lengths[idx] = 0;
        return;
    }

    const char* input = input_data + start;
    char* output = output_data + out_start;
    int out_len = 0;

    // Step 1: Find and skip protocol (anything before "://")
    int protocol_end = 0;
    for (int i = 0; i < len - 2; i++) {
        if (input[i] == ':' && input[i+1] == '/' && input[i+2] == '/') {
            protocol_end = i + 3;
            break;
        }
    }

    // Step 2: Handle android URLs with @ symbol - extract domain after @
    int domain_start = protocol_end;
    if (protocol_end > 0) {
        // Look for android in the protocol part
        bool is_android = false;
        if (protocol_end >= 10) { // "android://" is 10 chars
            if (starts_with(input, protocol_end, "android://", 10)) {
                is_android = true;
            }
        }

        if (is_android) {
            // Find the last @ symbol
            int at_pos = find_last_char(input + protocol_end, len - protocol_end, '@');
            if (at_pos >= 0) {
                domain_start = protocol_end + at_pos + 1;
            }
        }
    }

    // Step 3: Skip www. prefix
    int content_start = domain_start;
    if (len - domain_start >= 4 && starts_with(input + domain_start, len - domain_start, "www.", 4)) {
        content_start += 4;
    }

    // Step 4: Find end of domain (before /, ?, #)
    int content_end = len;
    for (int i = content_start; i < len; i++) {
        if (input[i] == '/' || input[i] == '?' || input[i] == '#') {
            content_end = i;
            break;
        }
    }

    // Step 5: Remove trailing slash if it's the only character after domain
    if (content_end < len && input[content_end] == '/' && content_end == len - 1) {
        content_end = len - 1;
    }

    // Step 6: For Android URLs, keep full reverse domain notation
    // For HTTP/HTTPS URLs, keep full domain including subdomains
    // Copy and convert to lowercase
    for (int i = content_start; i < content_end && out_len < 255; i++) {
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
    pub field_count: usize,  // Track original field count for completeness comparison
    pub all_fields: Vec<String>,  // Store all original fields to preserve extra data
}

#[cfg(feature = "cuda")]
pub struct CudaProcessor {
    context: Arc<CudaContext>,
    normalize_urls_function: CudaFunction,
    normalize_emails_function: CudaFunction,
    optimal_batch_size: usize,
    max_string_length: usize,
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
        println!("  Total Memory: {:.2} GB ({} bytes)", props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0), props.total_memory);
        println!("  Free Memory: {:.2} GB ({} bytes)", props.free_memory as f64 / (1024.0 * 1024.0 * 1024.0), props.free_memory);
        println!("  Max Threads per Block: {}", props.max_threads_per_block);
        println!("  Max Shared Memory per Block: {} KB", props.max_shared_memory_per_block as f64 / 1024.0);
        println!("  Memory Bus Width: {} bits", props.memory_bus_width);
        println!("  L2 Cache Size: {} MB", props.l2_cache_size as usize / (1024 * 1024));

        let available_memory_bytes = (props.free_memory as f64 * (config.gpu_memory_usage_percent as f64 / PERCENT_100) * PERCENT_95) as usize;

        let max_batch_size = (available_memory_bytes / config.estimated_bytes_per_record)
            .max(config.min_batch_size)
            .min(config.max_batch_size);

        let optimal_batch_size = Self::calculate_optimal_batch_size(
            max_batch_size,
            props.max_threads_per_block as usize,
            &config
        );

        println!("CUDA processor initialized for device {} - Available memory for use: {} bytes, Max batch size: {}, Optimal batch size: {}",
              device_ordinal,
              available_memory_bytes,
              max_batch_size,
              optimal_batch_size);

        // Compile CUDA kernel
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
            // Allocate maximum possible output size (could be same as input)
            total_output_len += url_bytes.len().max(256); // Ensure minimum buffer size
        }

        // Allocate/copy buffers to GPU using cudarc stream API
        let ctx = &self.context;
        let stream = ctx.default_stream();
        let d_input = stream.memcpy_stod(&input_data)?;
        let d_output = stream.alloc_zeros::<u8>(total_output_len)?;
        let d_input_offsets = stream.memcpy_stod(&input_offsets)?;
        let d_output_offsets = stream.memcpy_stod(&output_offsets)?;
        let d_input_lengths = stream.memcpy_stod(&input_lengths)?;
        let d_output_lengths = stream.alloc_zeros::<i32>(records.len())?;

        // Launch CUDA kernel for URL normalization
        let num_strings = records.len() as i32;
        let block_size = 256;
        let num_blocks = (num_strings + block_size - 1) / block_size;
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
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
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
        // Prepare data for GPU: flatten emails, build offsets/lengths
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
            // Allocate same size as input for email (lowercase conversion)
            total_output_len += email_bytes.len();
        }

        // Allocate/copy buffers to GPU using cudarc stream API
        let ctx = &self.context;
        let stream = ctx.default_stream();
        let d_input = stream.memcpy_stod(&input_data)?;
        let d_output = stream.alloc_zeros::<u8>(total_output_len)?;
        let d_input_offsets = stream.memcpy_stod(&input_offsets)?;
        let d_output_offsets = stream.memcpy_stod(&output_offsets)?;
        let d_input_lengths = stream.memcpy_stod(&input_lengths)?;
        let d_output_lengths = stream.alloc_zeros::<i32>(records.len())?;

        // Launch CUDA kernel for email normalization
        let num_strings = records.len() as i32;
        let block_size = 256;
        let num_blocks = (num_strings + block_size - 1) / block_size;
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
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        stream.synchronize()?;

        let output_data_host = stream.memcpy_dtov(&d_output)?;
        let output_lengths_host = stream.memcpy_dtov(&d_output_lengths)?;

        // Update records with normalized emails from output_data_host
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