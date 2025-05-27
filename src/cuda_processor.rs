use anyhow::Result;
use tracing::{debug, info};
use std::sync::Arc;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::record::Record;

pub struct CudaProcessor {
    device: Arc<CudaDevice>,
    max_batch_size: usize,
    available_memory_bytes: usize,
    url_normalize_kernel: cudarc::driver::CudaFunction,
    username_normalize_kernel: cudarc::driver::CudaFunction,
}

impl CudaProcessor {
    pub fn new() -> Result<Self> {
        // Initialize CUDA device
        let device = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device 0: {}. Ensure NVIDIA drivers and CUDA toolkit are installed.", e))?;
        
        info!("CUDA device initialized successfully");
        
        // Get GPU memory information
        let (free_memory, total_memory) = Self::get_gpu_memory_info()?;
        
        info!("GPU Memory - Total: {:.2} GB, Free: {:.2} GB", 
              total_memory as f64 / 1_073_741_824.0,
              free_memory as f64 / 1_073_741_824.0);
        
        // Use 80% of free memory for processing
        let available_memory_bytes = (free_memory as f64 * 0.8) as usize;
        
        // Calculate optimal batch size based on memory
        // Estimate: each record needs ~500 bytes on GPU (URLs, usernames, normalized versions)
        let estimated_bytes_per_record = 500;
        let max_batch_size = (available_memory_bytes / estimated_bytes_per_record).max(1000).min(100_000);
        
        info!("CUDA processor initialized - Available memory: {:.2} GB, Max batch size: {}", 
              available_memory_bytes as f64 / 1_073_741_824.0,
              max_batch_size);
        
        // Compile and load CUDA kernels
        let url_normalize_kernel = Self::compile_url_normalize_kernel(&device)?;
        let username_normalize_kernel = Self::compile_username_normalize_kernel(&device)?;
        
        Ok(Self {
            device,
            max_batch_size,
            available_memory_bytes,
            url_normalize_kernel,
            username_normalize_kernel,
        })
    }
    
    fn compile_url_normalize_kernel(device: &Arc<CudaDevice>) -> Result<cudarc::driver::CudaFunction> {
        let kernel_src = r#"
extern "C" __global__ void normalize_urls(
    char* input_urls,
    char* output_urls,
    int* input_lengths,
    int* output_lengths,
    int num_urls,
    int max_url_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_urls) return;
    
    char* input = input_urls + idx * max_url_length;
    char* output = output_urls + idx * max_url_length;
    int input_len = input_lengths[idx];
    
    int out_pos = 0;
    int in_pos = 0;
    
    // Convert to lowercase and remove protocol
    bool skip_protocol = false;
    if (input_len >= 8 && 
        (input[0] == 'h' || input[0] == 'H') &&
        (input[1] == 't' || input[1] == 'T') &&
        (input[2] == 't' || input[2] == 'T') &&
        (input[3] == 'p' || input[3] == 'P')) {
        
        if (input_len >= 8 && input[4] == 's' && input[5] == ':' && input[6] == '/' && input[7] == '/') {
            in_pos = 8; // Skip "https://"
        } else if (input_len >= 7 && input[4] == ':' && input[5] == '/' && input[6] == '/') {
            in_pos = 7; // Skip "http://"
        }
    }
    
    // Skip www. prefix
    if (input_len - in_pos >= 4 &&
        (input[in_pos] == 'w' || input[in_pos] == 'W') &&
        (input[in_pos + 1] == 'w' || input[in_pos + 1] == 'W') &&
        (input[in_pos + 2] == 'w' || input[in_pos + 2] == 'W') &&
        input[in_pos + 3] == '.') {
        in_pos += 4;
    }
    
    // Skip mobile prefixes
    if (input_len - in_pos >= 2 &&
        (input[in_pos] == 'm' || input[in_pos] == 'M') &&
        input[in_pos + 1] == '.') {
        in_pos += 2;
    } else if (input_len - in_pos >= 7 &&
               (input[in_pos] == 'm' || input[in_pos] == 'M') &&
               (input[in_pos + 1] == 'o' || input[in_pos + 1] == 'O') &&
               (input[in_pos + 2] == 'b' || input[in_pos + 2] == 'B') &&
               (input[in_pos + 3] == 'i' || input[in_pos + 3] == 'I') &&
               (input[in_pos + 4] == 'l' || input[in_pos + 4] == 'L') &&
               (input[in_pos + 5] == 'e' || input[in_pos + 5] == 'E') &&
               input[in_pos + 6] == '.') {
        in_pos += 7;
    }
    
    // Copy remaining characters, converting to lowercase and stopping at ? or #
    while (in_pos < input_len && input[in_pos] != '?' && input[in_pos] != '#') {
        char c = input[in_pos];
        if (c >= 'A' && c <= 'Z') {
            c = c + ('a' - 'A'); // Convert to lowercase
        }
        output[out_pos] = c;
        out_pos++;
        in_pos++;
    }
    
    // Remove trailing slash
    if (out_pos > 0 && output[out_pos - 1] == '/') {
        out_pos--;
    }
    
    output_lengths[idx] = out_pos;
    
    // Null terminate
    if (out_pos < max_url_length) {
        output[out_pos] = '\0';
    }
}
"#;

        info!("Compiling URL normalization CUDA kernel...");
        let ptx = compile_ptx(kernel_src)?;
        device.load_ptx(ptx, "normalize_urls", &["normalize_urls"])?;
        device.get_func("normalize_urls", "normalize_urls")
            .ok_or_else(|| anyhow::anyhow!("Failed to get CUDA kernel function"))
    }
    
    fn compile_username_normalize_kernel(device: &Arc<CudaDevice>) -> Result<cudarc::driver::CudaFunction> {
        let kernel_src = r#"
extern "C" __global__ void normalize_usernames(
    char* input_usernames,
    char* output_usernames,
    int* input_lengths,
    int* output_lengths,
    int num_usernames,
    int max_username_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_usernames) return;
    
    char* input = input_usernames + idx * max_username_length;
    char* output = output_usernames + idx * max_username_length;
    int input_len = input_lengths[idx];
    
    int out_pos = 0;
    int in_pos = 0;
    
    // Convert to lowercase
    while (in_pos < input_len && input[in_pos] != '\0') {
        char c = input[in_pos];
        if (c >= 'A' && c <= 'Z') {
            c = c + ('a' - 'A'); // Convert to lowercase
        }
        output[out_pos] = c;
        out_pos++;
        in_pos++;
    }
    
    output_lengths[idx] = out_pos;
    
    // Null terminate
    if (out_pos < max_username_length) {
        output[out_pos] = '\0';
    }
}
"#;

        info!("Compiling username normalization CUDA kernel...");
        let ptx = compile_ptx(kernel_src)?;
        device.load_ptx(ptx, "normalize_usernames", &["normalize_usernames"])?;
        device.get_func("normalize_usernames", "normalize_usernames")
            .ok_or_else(|| anyhow::anyhow!("Failed to get CUDA kernel function"))
    }
    
    fn get_gpu_memory_info() -> Result<(usize, usize)> {
        // Use CUDA driver API to get memory info
        use cudarc::driver::sys;
        
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;
        
        unsafe {
            let result = sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow::anyhow!("Failed to get GPU memory info: {:?}", result));
            }
        }
        
        Ok((free_bytes, total_bytes))
    }
    
    pub fn is_available() -> bool {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
    
    pub fn get_memory_info(&self) -> (usize, usize) {
        // Return (available_memory, max_batch_size)
        (self.available_memory_bytes, self.max_batch_size)
    }
    
    pub fn process_batch(&self, records: &mut [Record], _case_sensitive_usernames: bool) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        
        let batch_size = records.len().min(self.max_batch_size);
        
        for chunk in records.chunks_mut(batch_size) {
            self.process_chunk_gpu(chunk)?;
        }
        
        Ok(())
    }
    
    fn process_chunk_gpu(&self, records: &mut [Record]) -> Result<()> {
        debug!("Processing chunk of {} records with GPU acceleration", records.len());
        
        // Try combined GPU processing for better performance
        match self.normalize_combined_gpu(records) {
            Ok(()) => {
                debug!("Combined GPU normalization completed successfully");
            }
            Err(e) => {
                debug!("Combined GPU processing failed, trying individual kernels: {}", e);
                // Fallback to individual kernels
                match self.normalize_urls_gpu(records) {
                    Ok(()) => {
                        debug!("GPU URL normalization completed successfully");
                        // Now do username normalization on GPU as well
                        match self.normalize_usernames_gpu(records) {
                            Ok(()) => {
                                debug!("GPU username normalization completed successfully");
                            }
                            Err(e) => {
                                debug!("GPU username normalization failed, using CPU fallback: {}", e);
                                // Fallback to CPU for usernames only
                                for record in records.iter_mut() {
                                    record.normalized_user = record.user.to_lowercase();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        debug!("GPU processing failed, falling back to CPU: {}", e);
                        self.process_chunk_cpu_fallback(records)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn normalize_combined_gpu(&self, records: &mut [Record]) -> Result<()> {
        let num_records = records.len();
        if num_records == 0 {
            return Ok(());
        }
        
        // Find maximum lengths for buffer allocation
        let max_url_length = records.iter()
            .map(|r| r.url.len())
            .max()
            .unwrap_or(0)
            .max(256);
            
        let max_username_length = records.iter()
            .map(|r| r.user.len())
            .max()
            .unwrap_or(0)
            .max(64);
        
        // Prepare input data for both URLs and usernames
        let mut input_urls = vec![0u8; num_records * max_url_length];
        let mut input_usernames = vec![0u8; num_records * max_username_length];
        let mut url_lengths = vec![0i32; num_records];
        let mut username_lengths = vec![0i32; num_records];
        
        for (i, record) in records.iter().enumerate() {
            // URLs
            let url_bytes = record.url.as_bytes();
            let url_start_idx = i * max_url_length;
            let url_copy_len = url_bytes.len().min(max_url_length - 1);
            input_urls[url_start_idx..url_start_idx + url_copy_len].copy_from_slice(&url_bytes[..url_copy_len]);
            url_lengths[i] = url_copy_len as i32;
            
            // Usernames
            let username_bytes = record.user.as_bytes();
            let username_start_idx = i * max_username_length;
            let username_copy_len = username_bytes.len().min(max_username_length - 1);
            input_usernames[username_start_idx..username_start_idx + username_copy_len].copy_from_slice(&username_bytes[..username_copy_len]);
            username_lengths[i] = username_copy_len as i32;
        }
        
        // Allocate GPU memory for both operations
        let input_urls_gpu = self.device.htod_copy(input_urls)?;
        let input_usernames_gpu = self.device.htod_copy(input_usernames)?;
        let url_lengths_gpu = self.device.htod_copy(url_lengths)?;
        let username_lengths_gpu = self.device.htod_copy(username_lengths)?;
        
        let output_urls_gpu = self.device.alloc_zeros::<u8>(num_records * max_url_length)?;
        let output_usernames_gpu = self.device.alloc_zeros::<u8>(num_records * max_username_length)?;
        let output_url_lengths_gpu = self.device.alloc_zeros::<i32>(num_records)?;
        let output_username_lengths_gpu = self.device.alloc_zeros::<i32>(num_records)?;
        
        // Launch both kernels
        let threads_per_block = 256;
        let blocks = (num_records + threads_per_block - 1) / threads_per_block;
        
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch URL normalization kernel
        unsafe {
            self.url_normalize_kernel.clone().launch(
                config,
                (
                    &input_urls_gpu,
                    &output_urls_gpu,
                    &url_lengths_gpu,
                    &output_url_lengths_gpu,
                    num_records as i32,
                    max_url_length as i32,
                ),
            )?;
        }
        
        // Launch username normalization kernel
        unsafe {
            self.username_normalize_kernel.clone().launch(
                config,
                (
                    &input_usernames_gpu,
                    &output_usernames_gpu,
                    &username_lengths_gpu,
                    &output_username_lengths_gpu,
                    num_records as i32,
                    max_username_length as i32,
                ),
            )?;
        }
        
        // Copy results back
        let output_urls: Vec<u8> = self.device.dtoh_sync_copy(&output_urls_gpu)?;
        let output_usernames: Vec<u8> = self.device.dtoh_sync_copy(&output_usernames_gpu)?;
        let output_url_lengths: Vec<i32> = self.device.dtoh_sync_copy(&output_url_lengths_gpu)?;
        let output_username_lengths: Vec<i32> = self.device.dtoh_sync_copy(&output_username_lengths_gpu)?;
        
        // Update records with normalized data
        for (i, record) in records.iter_mut().enumerate() {
            // Update URLs
            let url_start_idx = i * max_url_length;
            let url_length = output_url_lengths[i] as usize;
            if url_length > 0 && url_length <= max_url_length {
                let url_bytes = &output_urls[url_start_idx..url_start_idx + url_length];
                record.normalized_url = String::from_utf8_lossy(url_bytes).to_string();
            } else {
                record.normalized_url = normalize_url_simple(&record.url);
            }
            
            // Update usernames
            let username_start_idx = i * max_username_length;
            let username_length = output_username_lengths[i] as usize;
            if username_length > 0 && username_length <= max_username_length {
                let username_bytes = &output_usernames[username_start_idx..username_start_idx + username_length];
                record.normalized_user = String::from_utf8_lossy(username_bytes).to_string();
            } else {
                record.normalized_user = record.user.to_lowercase();
            }
        }
        
        Ok(())
    }
    
    fn normalize_urls_gpu(&self, records: &mut [Record]) -> Result<()> {
        let num_records = records.len();
        if num_records == 0 {
            return Ok(());
        }
        
        // Find maximum URL length for buffer allocation
        let max_url_length = records.iter()
            .map(|r| r.url.len())
            .max()
            .unwrap_or(0)
            .max(256); // Minimum buffer size
        
        // Prepare input data
        let mut input_urls = vec![0u8; num_records * max_url_length];
        let mut input_lengths = vec![0i32; num_records];
        
        for (i, record) in records.iter().enumerate() {
            let url_bytes = record.url.as_bytes();
            let start_idx = i * max_url_length;
            let copy_len = url_bytes.len().min(max_url_length - 1);
            
            input_urls[start_idx..start_idx + copy_len].copy_from_slice(&url_bytes[..copy_len]);
            input_lengths[i] = copy_len as i32;
        }
        
        // Allocate GPU memory
        let input_urls_gpu = self.device.htod_copy(input_urls)?;
        let input_lengths_gpu = self.device.htod_copy(input_lengths)?;
        let output_urls_gpu = self.device.alloc_zeros::<u8>(num_records * max_url_length)?;
        let output_lengths_gpu = self.device.alloc_zeros::<i32>(num_records)?;
        
        // Launch kernel
        let threads_per_block = 256;
        let blocks = (num_records + threads_per_block - 1) / threads_per_block;
        
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.url_normalize_kernel.clone().launch(
                config,
                (
                    &input_urls_gpu,
                    &output_urls_gpu,
                    &input_lengths_gpu,
                    &output_lengths_gpu,
                    num_records as i32,
                    max_url_length as i32,
                ),
            )?;
        }
        
        // Copy results back
        let output_urls: Vec<u8> = self.device.dtoh_sync_copy(&output_urls_gpu)?;
        let output_lengths: Vec<i32> = self.device.dtoh_sync_copy(&output_lengths_gpu)?;
        
        // Update records with normalized URLs
        for (i, record) in records.iter_mut().enumerate() {
            let start_idx = i * max_url_length;
            let length = output_lengths[i] as usize;
            
            if length > 0 && length <= max_url_length {
                let url_bytes = &output_urls[start_idx..start_idx + length];
                record.normalized_url = String::from_utf8_lossy(url_bytes).to_string();
            } else {
                // Fallback to CPU normalization for this record
                record.normalized_url = normalize_url_simple(&record.url);
            }
        }
        
        Ok(())
    }
    
    fn normalize_usernames_gpu(&self, records: &mut [Record]) -> Result<()> {
        let num_records = records.len();
        if num_records == 0 {
            return Ok(());
        }
        
        // Find maximum username length for buffer allocation
        let max_username_length = records.iter()
            .map(|r| r.user.len())
            .max()
            .unwrap_or(0)
            .max(64); // Minimum buffer size for usernames
        
        // Prepare input data
        let mut input_usernames = vec![0u8; num_records * max_username_length];
        let mut input_lengths = vec![0i32; num_records];
        
        for (i, record) in records.iter().enumerate() {
            let username_bytes = record.user.as_bytes();
            let start_idx = i * max_username_length;
            let copy_len = username_bytes.len().min(max_username_length - 1);
            
            input_usernames[start_idx..start_idx + copy_len].copy_from_slice(&username_bytes[..copy_len]);
            input_lengths[i] = copy_len as i32;
        }
        
        // Allocate GPU memory
        let input_usernames_gpu = self.device.htod_copy(input_usernames)?;
        let input_lengths_gpu = self.device.htod_copy(input_lengths)?;
        let output_usernames_gpu = self.device.alloc_zeros::<u8>(num_records * max_username_length)?;
        let output_lengths_gpu = self.device.alloc_zeros::<i32>(num_records)?;
        
        // Launch kernel
        let threads_per_block = 256;
        let blocks = (num_records + threads_per_block - 1) / threads_per_block;
        
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.username_normalize_kernel.clone().launch(
                config,
                (
                    &input_usernames_gpu,
                    &output_usernames_gpu,
                    &input_lengths_gpu,
                    &output_lengths_gpu,
                    num_records as i32,
                    max_username_length as i32,
                ),
            )?;
        }
        
        // Copy results back
        let output_usernames: Vec<u8> = self.device.dtoh_sync_copy(&output_usernames_gpu)?;
        let output_lengths: Vec<i32> = self.device.dtoh_sync_copy(&output_lengths_gpu)?;
        
        // Update records with normalized usernames
        for (i, record) in records.iter_mut().enumerate() {
            let start_idx = i * max_username_length;
            let length = output_lengths[i] as usize;
            
            if length > 0 && length <= max_username_length {
                let username_bytes = &output_usernames[start_idx..start_idx + length];
                record.normalized_user = String::from_utf8_lossy(username_bytes).to_string();
            } else {
                // Fallback to CPU normalization for this record
                record.normalized_user = record.user.to_lowercase();
            }
        }
        
        Ok(())
    }
    
    // Keep CPU fallback for error cases
    fn process_chunk_cpu_fallback(&self, records: &mut [Record]) -> Result<()> {
        debug!("Processing chunk of {} records (CPU fallback)", records.len());
        
        for record in records.iter_mut() {
            record.normalized_url = normalize_url_simple(&record.url);
            record.normalized_user = record.user.to_lowercase();
        }
        
        Ok(())
    }
    
    pub fn get_max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

fn normalize_url_simple(url: &str) -> String {
    let mut result = url.to_lowercase();
    
    // Remove protocol
    if result.starts_with("https://") {
        result = result[8..].to_string();
    } else if result.starts_with("http://") {
        result = result[7..].to_string();
    }
    
    // Remove www prefix
    if result.starts_with("www.") {
        result = result[4..].to_string();
    }
    
    // Remove mobile prefixes
    if result.starts_with("m.") {
        result = result[2..].to_string();
    } else if result.starts_with("mobile.") {
        result = result[7..].to_string();
    }
    
    // Remove query parameters
    if let Some(pos) = result.find('?') {
        result = result[..pos].to_string();
    }
    
    // Remove fragment
    if let Some(pos) = result.find('#') {
        result = result[..pos].to_string();
    }
    
    // Remove trailing slash
    if result.ends_with('/') {
        result = result[..result.len()-1].to_string();
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_processor_creation() {
        if CudaProcessor::is_available() {
            let processor = CudaProcessor::new();
            assert!(processor.is_ok());
            
            if let Ok(processor) = processor {
                let (available_memory, max_batch_size) = processor.get_memory_info();
                println!("Available GPU memory: {:.2} GB", available_memory as f64 / 1_073_741_824.0);
                println!("Max batch size: {}", max_batch_size);
                
                // Basic sanity checks
                assert!(available_memory > 0, "Available memory should be greater than 0");
                assert!(max_batch_size >= 1000, "Batch size should be at least 1000");
                assert!(max_batch_size <= 100_000, "Batch size should not exceed 100,000");
            }
        } else {
            println!("CUDA not available, skipping test");
        }
    }
    
    #[test]
    fn test_gpu_memory_detection() {
        if CudaProcessor::is_available() {
            let device = cudarc::driver::CudaDevice::new(0).unwrap();
            let memory_info = CudaProcessor::get_gpu_memory_info();
            
            assert!(memory_info.is_ok(), "Memory detection should succeed");
            
            if let Ok((free, total)) = memory_info {
                println!("GPU Memory - Free: {} bytes, Total: {} bytes", free, total);
                assert!(free > 0, "Free memory should be greater than 0");
                assert!(total > 0, "Total memory should be greater than 0");
                assert!(free <= total, "Free memory should not exceed total memory");
            }
        } else {
            println!("CUDA not available, skipping memory detection test");
        }
    }
    
    #[test]
    fn test_url_normalization() {
        assert_eq!(normalize_url_simple("https://www.facebook.com/user123/"), "facebook.com/user123");
        assert_eq!(normalize_url_simple("http://m.facebook.com/user123"), "facebook.com/user123");
        assert_eq!(normalize_url_simple("facebook.com/user123?ref=123"), "facebook.com/user123");
    }
    
    #[test]
    fn test_gpu_url_normalization() {
        if CudaProcessor::is_available() {
            let processor = CudaProcessor::new();
            if let Ok(processor) = processor {
                // Create test records
                let mut records = vec![
                    Record {
                        user: "user1".to_string(),
                        password: "pass1".to_string(),
                        url: "https://www.facebook.com/user123/".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["user1".to_string(), "pass1".to_string(), "https://www.facebook.com/user123/".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 1,
                    },
                    Record {
                        user: "user2".to_string(),
                        password: "pass2".to_string(),
                        url: "http://m.facebook.com/user123".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["user2".to_string(), "pass2".to_string(), "http://m.facebook.com/user123".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 2,
                    },
                    Record {
                        user: "user3".to_string(),
                        password: "pass3".to_string(),
                        url: "facebook.com/user123?ref=123".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["user3".to_string(), "pass3".to_string(), "facebook.com/user123?ref=123".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 3,
                    },
                ];
                
                // Process with GPU
                let result = processor.process_batch(&mut records, false);
                assert!(result.is_ok(), "GPU processing should succeed");
                
                // Verify normalization results
                assert_eq!(records[0].normalized_url, "facebook.com/user123");
                assert_eq!(records[1].normalized_url, "facebook.com/user123");
                assert_eq!(records[2].normalized_url, "facebook.com/user123");
                
                // Verify usernames are also normalized
                assert_eq!(records[0].normalized_user, "user1");
                assert_eq!(records[1].normalized_user, "user2");
                assert_eq!(records[2].normalized_user, "user3");
                
                println!("GPU URL normalization test passed!");
                println!("Original URLs:");
                println!("  https://www.facebook.com/user123/ -> {}", records[0].normalized_url);
                println!("  http://m.facebook.com/user123 -> {}", records[1].normalized_url);
                println!("  facebook.com/user123?ref=123 -> {}", records[2].normalized_url);
            } else {
                println!("Failed to create CUDA processor, skipping GPU test");
            }
        } else {
            println!("CUDA not available, skipping GPU URL normalization test");
        }
    }
    
    #[test]
    fn test_combined_gpu_normalization() {
        if CudaProcessor::is_available() {
            let processor = CudaProcessor::new();
            if let Ok(processor) = processor {
                // Create test records with mixed case usernames
                let mut records = vec![
                    Record {
                        user: "User1@Example.COM".to_string(),
                        password: "pass1".to_string(),
                        url: "https://www.facebook.com/user123/".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["User1@Example.COM".to_string(), "pass1".to_string(), "https://www.facebook.com/user123/".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 1,
                    },
                    Record {
                        user: "USER2@DOMAIN.ORG".to_string(),
                        password: "pass2".to_string(),
                        url: "http://m.facebook.com/user123".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["USER2@DOMAIN.ORG".to_string(), "pass2".to_string(), "http://m.facebook.com/user123".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 2,
                    },
                ];
                
                // Process with GPU
                let result = processor.process_batch(&mut records, false);
                assert!(result.is_ok(), "GPU processing should succeed");
                
                // Verify URL normalization results
                assert_eq!(records[0].normalized_url, "facebook.com/user123");
                assert_eq!(records[1].normalized_url, "facebook.com/user123");
                
                // Verify username normalization results
                assert_eq!(records[0].normalized_user, "user1@example.com");
                assert_eq!(records[1].normalized_user, "user2@domain.org");
                
                println!("Combined GPU normalization test passed!");
                println!("URL normalization:");
                println!("  https://www.facebook.com/user123/ -> {}", records[0].normalized_url);
                println!("  http://m.facebook.com/user123 -> {}", records[1].normalized_url);
                println!("Username normalization:");
                println!("  User1@Example.COM -> {}", records[0].normalized_user);
                println!("  USER2@DOMAIN.ORG -> {}", records[1].normalized_user);
            } else {
                println!("Failed to create CUDA processor, skipping combined GPU test");
            }
        } else {
            println!("CUDA not available, skipping combined GPU normalization test");
        }
    }
    
    #[test]
    fn test_optimized_combined_gpu_processing() {
        if CudaProcessor::is_available() {
            let processor = CudaProcessor::new();
            if let Ok(processor) = processor {
                // Create a larger batch to test performance optimization
                let mut records = vec![
                    Record {
                        user: "User1@Example.COM".to_string(),
                        password: "pass1".to_string(),
                        url: "https://www.facebook.com/user123/".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["User1@Example.COM".to_string(), "pass1".to_string(), "https://www.facebook.com/user123/".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 1,
                    },
                    Record {
                        user: "USER2@DOMAIN.ORG".to_string(),
                        password: "pass2".to_string(),
                        url: "http://m.facebook.com/user123".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["USER2@DOMAIN.ORG".to_string(), "pass2".to_string(), "http://m.facebook.com/user123".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 2,
                    },
                    Record {
                        user: "TestUser@SITE.NET".to_string(),
                        password: "pass3".to_string(),
                        url: "https://mobile.twitter.com/test?param=value".to_string(),
                        normalized_user: String::new(),
                        normalized_url: String::new(),
                        fields: vec!["TestUser@SITE.NET".to_string(), "pass3".to_string(), "https://mobile.twitter.com/test?param=value".to_string()],
                        completeness_score: 3,
                        source_file: "test".to_string(),
                        line_number: 3,
                    },
                ];
                
                // Process with optimized GPU
                let result = processor.process_batch(&mut records, false);
                assert!(result.is_ok(), "Optimized GPU processing should succeed");
                
                // Verify URL normalization results
                assert_eq!(records[0].normalized_url, "facebook.com/user123");
                assert_eq!(records[1].normalized_url, "facebook.com/user123");
                assert_eq!(records[2].normalized_url, "twitter.com/test");
                
                // Verify username normalization results
                assert_eq!(records[0].normalized_user, "user1@example.com");
                assert_eq!(records[1].normalized_user, "user2@domain.org");
                assert_eq!(records[2].normalized_user, "testuser@site.net");
                
                println!("Optimized combined GPU processing test passed!");
                println!("URL normalization:");
                println!("  https://www.facebook.com/user123/ -> {}", records[0].normalized_url);
                println!("  http://m.facebook.com/user123 -> {}", records[1].normalized_url);
                println!("  https://mobile.twitter.com/test?param=value -> {}", records[2].normalized_url);
                println!("Username normalization:");
                println!("  User1@Example.COM -> {}", records[0].normalized_user);
                println!("  USER2@DOMAIN.ORG -> {}", records[1].normalized_user);
                println!("  TestUser@SITE.NET -> {}", records[2].normalized_user);
            } else {
                println!("Failed to create CUDA processor, skipping optimized GPU test");
            }
        } else {
            println!("CUDA not available, skipping optimized GPU processing test");
        }
    }
} 