use anyhow::Result;
use crate::constants::*;

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_sse42: bool,
    pub has_neon: bool,
}

#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub cpu_features: CpuFeatures,
    pub max_vector_width: usize,
    pub theoretical_speedup: f32,
    pub optimal_chunk_size: usize,
}

pub struct SimdStringProcessor {
    capabilities: SimdCapabilities,
}

impl SimdStringProcessor {
    pub fn new() -> Result<Self> {
        let cpu_features = detect_cpu_features();
        let capabilities = SimdCapabilities {
            cpu_features,
            max_vector_width: calculate_max_vector_width(&cpu_features),
            theoretical_speedup: calculate_theoretical_speedup(&cpu_features),
            optimal_chunk_size: calculate_optimal_chunk_size(&cpu_features),
        };

        Ok(Self {
            capabilities,
        })
    }

    pub fn normalize_urls_to_lowercase(&self, urls: &[String]) -> Result<Vec<String>> {
        let chunk_size = self.capabilities.optimal_chunk_size;
        let mut results = Vec::with_capacity(urls.len());

        for url_chunk in urls.chunks(chunk_size) {
            let mut chunk_results = if self.capabilities.cpu_features.has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    avx2_optimizations::normalize_urls_avx2(url_chunk)?
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.normalize_urls_fallback(url_chunk)?
                }
            } else if self.capabilities.cpu_features.has_neon {
                #[cfg(target_arch = "aarch64")]
                {
                    neon_optimizations::normalize_urls_neon(url_chunk)?
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.normalize_urls_fallback(url_chunk)?
                }
            } else {
                self.normalize_urls_fallback(url_chunk)?
            };

            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    pub fn validate_emails_simd(&self, emails: &[String]) -> Result<Vec<bool>> {
        let chunk_size = self.capabilities.optimal_chunk_size;
        let mut results = Vec::with_capacity(emails.len());

        for email_chunk in emails.chunks(chunk_size) {
            let mut chunk_results = if self.capabilities.cpu_features.has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    avx2_optimizations::validate_emails_avx2(email_chunk)?
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.validate_emails_fallback(email_chunk)?
                }
            } else if self.capabilities.cpu_features.has_neon {
                #[cfg(target_arch = "aarch64")]
                {
                    neon_optimizations::validate_emails_neon(email_chunk)?
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.validate_emails_fallback(email_chunk)?
                }
            } else {
                self.validate_emails_fallback(email_chunk)?
            };

            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    pub fn hash_strings_simd(&self, strings: &[String]) -> Result<Vec<u64>> {
        let chunk_size = self.capabilities.optimal_chunk_size;
        let mut results = Vec::with_capacity(strings.len());

        for string_chunk in strings.chunks(chunk_size) {
            let mut chunk_results = if self.capabilities.cpu_features.has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    avx2_optimizations::hash_strings_avx2(string_chunk)?
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.hash_strings_fallback(string_chunk)?
                }
            } else if self.capabilities.cpu_features.has_neon {
                #[cfg(target_arch = "aarch64")]
                {
                    neon_optimizations::hash_strings_neon(string_chunk)?
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.hash_strings_fallback(string_chunk)?
                }
            } else {
                self.hash_strings_fallback(string_chunk)?
            };

            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    pub fn parse_csv_fields_simd(&self, lines: &[String], delimiter: char) -> Result<Vec<Vec<String>>> {
        let chunk_size = self.capabilities.optimal_chunk_size;
        let mut results = Vec::with_capacity(lines.len());

        for line_chunk in lines.chunks(chunk_size) {
            let mut chunk_results = if self.capabilities.cpu_features.has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    avx2_optimizations::parse_csv_fields_avx2(line_chunk, delimiter)?
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.parse_csv_fields_fallback(line_chunk, delimiter)?
                }
            } else if self.capabilities.cpu_features.has_neon {
                #[cfg(target_arch = "aarch64")]
                {
                    neon_optimizations::parse_csv_fields_neon(line_chunk, delimiter)?
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.parse_csv_fields_fallback(line_chunk, delimiter)?
                }
            } else {
                self.parse_csv_fields_fallback(line_chunk, delimiter)?
            };

            results.append(&mut chunk_results);
        }

        Ok(results)
    }

    pub fn get_capabilities(&self) -> &SimdCapabilities {
        &self.capabilities
    }

    fn normalize_urls_fallback(&self, urls: &[String]) -> Result<Vec<String>> {
        Ok(urls.iter().map(|url| url.to_lowercase()).collect())
    }

    fn validate_emails_fallback(&self, emails: &[String]) -> Result<Vec<bool>> {
        Ok(emails.iter().map(|email| {
            email.contains('@') && email.contains('.') && email.len() > 5
        }).collect())
    }

    fn hash_strings_fallback(&self, strings: &[String]) -> Result<Vec<u64>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        Ok(strings.iter().map(|s| {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        }).collect())
    }

    fn parse_csv_fields_fallback(&self, lines: &[String], delimiter: char) -> Result<Vec<Vec<String>>> {
        Ok(lines.iter().map(|line| {
            line.split(delimiter).map(|s| s.to_string()).collect()
        }).collect())
    }
}

fn detect_cpu_features() -> CpuFeatures {
    CpuFeatures {
        has_avx2: is_x86_feature_detected!("avx2"),
        has_avx512: is_x86_feature_detected!("avx512f"),
        has_sse42: is_x86_feature_detected!("sse4.2"),
        has_neon: cfg!(target_arch = "aarch64"),
    }
}

fn calculate_max_vector_width(features: &CpuFeatures) -> usize {
    if features.has_avx512 {
        SIMD_AVX512_WIDTH_BYTES
    } else if features.has_avx2 {
        SIMD_AVX2_WIDTH_BYTES
    } else if features.has_sse42 {
        SIMD_SSE_NEON_WIDTH_BYTES
    } else if features.has_neon {
        SIMD_SSE_NEON_WIDTH_BYTES
    } else {
        SIMD_SCALAR_WIDTH_BYTES
    }
}

fn calculate_theoretical_speedup(features: &CpuFeatures) -> f32 {
    if features.has_avx512 {
        SIMD_AVX512_WIDTH_BYTES as f32
    } else if features.has_avx2 {
        SIMD_AVX2_WIDTH_BYTES as f32
    } else if features.has_sse42 {
        SIMD_SSE_NEON_WIDTH_BYTES as f32
    } else if features.has_neon {
        SIMD_SSE_NEON_WIDTH_BYTES as f32
    } else {
        SIMD_SCALAR_WIDTH_BYTES as f32
    }
}

fn calculate_optimal_chunk_size(features: &CpuFeatures) -> usize {
    let base_chunk_size = if features.has_avx512 {
        SIMD_AVX512_CHUNK_SIZE
    } else if features.has_avx2 {
        SIMD_AVX2_CHUNK_SIZE
    } else if features.has_sse42 || features.has_neon {
        SIMD_SSE_NEON_CHUNK_SIZE
    } else {
        SIMD_SCALAR_CHUNK_SIZE
    };

    base_chunk_size.min(SIMD_CACHE_SIZE_LIMIT)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2_optimizations {
    use super::*;
    use std::arch::x86_64::*;

    pub fn normalize_urls_avx2(urls: &[String]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(urls.len());

        for url in urls {
            if url.len() < SIMD_AVX2_PROCESS_SIZE {
                results.push(url.to_lowercase());
                continue;
            }

            let mut normalized = String::with_capacity(url.len());
            let bytes = url.as_bytes();
            let mut processed = 0;

            while processed + SIMD_AVX2_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = _mm256_loadu_si256(bytes.as_ptr().add(processed) as *const __m256i);
                    let lowercase_chunk = simd_to_lowercase_avx2(chunk);
                    
                    let mut temp_buffer = [0u8; SIMD_AVX2_PROCESS_SIZE];
                    _mm256_storeu_si256(temp_buffer.as_mut_ptr() as *mut __m256i, lowercase_chunk);
                    
                    if let Ok(chunk_str) = std::str::from_utf8(&temp_buffer) {
                        normalized.push_str(chunk_str);
                    } else {
                        for &byte in &temp_buffer {
                            normalized.push(byte as char);
                        }
                    }
                }
                processed += SIMD_AVX2_PROCESS_SIZE;
            }

            if processed < bytes.len() {
                let remaining = std::str::from_utf8(&bytes[processed..])
                    .unwrap_or("")
                    .to_lowercase();
                normalized.push_str(&remaining);
            }

            results.push(normalized);
        }

        Ok(results)
    }

    pub fn validate_emails_avx2(emails: &[String]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(emails.len());

        for email in emails {
            if email.len() < SIMD_AVX2_PROCESS_SIZE {
                results.push(email.contains('@') && email.contains('.') && email.len() > 5);
                continue;
            }

            let bytes = email.as_bytes();
            let mut has_at = false;
            let mut has_dot = false;
            let mut processed = 0;

            while processed + SIMD_AVX2_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = _mm256_loadu_si256(bytes.as_ptr().add(processed) as *const __m256i);
                    
                    let at_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'@' as i8));
                    if _mm256_movemask_epi8(at_mask) != 0 {
                        has_at = true;
                    }
                    
                    let dot_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'.' as i8));
                    if _mm256_movemask_epi8(dot_mask) != 0 {
                        has_dot = true;
                    }
                }
                processed += SIMD_AVX2_PROCESS_SIZE;
            }

            for &byte in &bytes[processed..] {
                if byte == b'@' { has_at = true; }
                if byte == b'.' { has_dot = true; }
            }

            results.push(has_at && has_dot && email.len() > 5);
        }

        Ok(results)
    }

    pub fn hash_strings_avx2(strings: &[String]) -> Result<Vec<u64>> {
        let mut results = Vec::with_capacity(strings.len());

        for string in strings {
            let mut hash = FNV1A_OFFSET_BASIS;
            let prime = FNV1A_PRIME;

            let bytes = string.as_bytes();
            let mut processed = 0;

            while processed + SIMD_AVX2_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = _mm256_loadu_si256(bytes.as_ptr().add(processed) as *const __m256i);
                    
                    let mut temp_buffer = [0u8; SIMD_AVX2_PROCESS_SIZE];
                    _mm256_storeu_si256(temp_buffer.as_mut_ptr() as *mut __m256i, chunk);
                    
                    for &byte in &temp_buffer {
                        hash ^= byte as u64;
                        hash = hash.wrapping_mul(prime);
                    }
                }
                processed += SIMD_AVX2_PROCESS_SIZE;
            }

            for &byte in &bytes[processed..] {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(prime);
            }

            results.push(hash);
        }

        Ok(results)
    }

    pub fn parse_csv_fields_avx2(lines: &[String], delimiter: char) -> Result<Vec<Vec<String>>> {
        let mut results = Vec::with_capacity(lines.len());
        let delimiter_byte = delimiter as u8;

        for line in lines {
            if line.len() < SIMD_AVX2_PROCESS_SIZE {
                results.push(line.split(delimiter).map(|s| s.to_string()).collect());
                continue;
            }

            let bytes = line.as_bytes();
            let mut field_positions = Vec::new();
            let mut processed = 0;

            while processed + SIMD_AVX2_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = _mm256_loadu_si256(bytes.as_ptr().add(processed) as *const __m256i);
                    let delim_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(delimiter_byte as i8));
                    let mask_bits = _mm256_movemask_epi8(delim_mask);
                    
                    if mask_bits != 0 {
                        for bit_pos in 0..SIMD_AVX2_PROCESS_SIZE {
                            if (mask_bits & (1 << bit_pos)) != 0 {
                                field_positions.push(processed + bit_pos);
                            }
                        }
                    }
                }
                processed += SIMD_AVX2_PROCESS_SIZE;
            }

            for (i, &byte) in bytes[processed..].iter().enumerate() {
                if byte == delimiter_byte {
                    field_positions.push(processed + i);
                }
            }

            let mut fields = Vec::new();
            let mut start = 0;
            
            for &pos in &field_positions {
                if let Ok(field) = std::str::from_utf8(&bytes[start..pos]) {
                    fields.push(field.to_string());
                }
                start = pos + 1;
            }
            
            if start < bytes.len() {
                if let Ok(field) = std::str::from_utf8(&bytes[start..]) {
                    fields.push(field.to_string());
                }
            }

            results.push(fields);
        }

        Ok(results)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_to_lowercase_avx2(input: __m256i) -> __m256i {
        let a_minus_1 = _mm256_set1_epi8((b'A' - 1) as i8);
        let z_plus_1 = _mm256_set1_epi8((b'Z' + 1) as i8);
        let case_diff = _mm256_set1_epi8(SIMD_AVX2_PROCESS_SIZE as i8);

        let gt_a = _mm256_cmpgt_epi8(input, a_minus_1);
        let lt_z = _mm256_cmpgt_epi8(z_plus_1, input);
        let is_upper = _mm256_and_si256(gt_a, lt_z);

        let to_add = _mm256_and_si256(is_upper, case_diff);
        _mm256_add_epi8(input, to_add)
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon_optimizations {
    use super::*;
    use std::arch::aarch64::*;

    pub fn normalize_urls_neon(urls: &[String]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(urls.len());

        for url in urls {
            if url.len() < SIMD_NEON_PROCESS_SIZE {
                results.push(url.to_lowercase());
                continue;
            }

            let mut normalized = String::with_capacity(url.len());
            let bytes = url.as_bytes();
            let mut processed = 0;

            while processed + SIMD_NEON_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = vld1q_u8(bytes.as_ptr().add(processed));
                    let lowercase_chunk = simd_to_lowercase_neon(chunk);
                    
                    let mut temp_buffer = [0u8; SIMD_NEON_PROCESS_SIZE];
                    vst1q_u8(temp_buffer.as_mut_ptr(), lowercase_chunk);
                    
                    if let Ok(chunk_str) = std::str::from_utf8(&temp_buffer) {
                        normalized.push_str(chunk_str);
                    } else {
                        for &byte in &temp_buffer {
                            normalized.push(byte as char);
                        }
                    }
                }
                processed += SIMD_NEON_PROCESS_SIZE;
            }

            if processed < bytes.len() {
                let remaining = std::str::from_utf8(&bytes[processed..])
                    .unwrap_or("")
                    .to_lowercase();
                normalized.push_str(&remaining);
            }

            results.push(normalized);
        }

        Ok(results)
    }

    pub fn validate_emails_neon(emails: &[String]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(emails.len());

        for email in emails {
            if email.len() < SIMD_NEON_PROCESS_SIZE {
                results.push(email.contains('@') && email.contains('.') && email.len() > 5);
                continue;
            }

            let bytes = email.as_bytes();
            let mut has_at = false;
            let mut has_dot = false;
            let mut processed = 0;

            while processed + SIMD_NEON_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = vld1q_u8(bytes.as_ptr().add(processed));
                    
                    let at_vec = vdupq_n_u8(b'@');
                    let at_mask = vceqq_u8(chunk, at_vec);
                    if vmaxvq_u8(at_mask) != 0 {
                        has_at = true;
                    }
                    
                    let dot_vec = vdupq_n_u8(b'.');
                    let dot_mask = vceqq_u8(chunk, dot_vec);
                    if vmaxvq_u8(dot_mask) != 0 {
                        has_dot = true;
                    }
                }
                processed += SIMD_NEON_PROCESS_SIZE;
            }

            for &byte in &bytes[processed..] {
                if byte == b'@' { has_at = true; }
                if byte == b'.' { has_dot = true; }
            }

            results.push(has_at && has_dot && email.len() > 5);
        }

        Ok(results)
    }

    pub fn hash_strings_neon(strings: &[String]) -> Result<Vec<u64>> {
        let mut results = Vec::with_capacity(strings.len());

        for string in strings {
            let mut hash = FNV1A_OFFSET_BASIS;
            let prime = FNV1A_PRIME;

            let bytes = string.as_bytes();
            let mut processed = 0;

            while processed + SIMD_NEON_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = vld1q_u8(bytes.as_ptr().add(processed));
                    let mut temp_buffer = [0u8; SIMD_NEON_PROCESS_SIZE];
                    vst1q_u8(temp_buffer.as_mut_ptr(), chunk);
                    
                    for &byte in &temp_buffer {
                        hash ^= byte as u64;
                        hash = hash.wrapping_mul(prime);
                    }
                }
                processed += SIMD_NEON_PROCESS_SIZE;
            }

            for &byte in &bytes[processed..] {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(prime);
            }

            results.push(hash);
        }

        Ok(results)
    }

    pub fn parse_csv_fields_neon(lines: &[String], delimiter: char) -> Result<Vec<Vec<String>>> {
        let mut results = Vec::with_capacity(lines.len());
        let delimiter_byte = delimiter as u8;

        for line in lines {
            if line.len() < SIMD_NEON_PROCESS_SIZE {
                results.push(line.split(delimiter).map(|s| s.to_string()).collect());
                continue;
            }

            let bytes = line.as_bytes();
            let mut field_positions = Vec::new();
            let mut processed = 0;

            while processed + SIMD_NEON_PROCESS_SIZE <= bytes.len() {
                unsafe {
                    let chunk = vld1q_u8(bytes.as_ptr().add(processed));
                    let delim_vec = vdupq_n_u8(delimiter_byte);
                    let delim_mask = vceqq_u8(chunk, delim_vec);
                    
                    let mut temp_mask = [0u8; SIMD_NEON_PROCESS_SIZE];
                    vst1q_u8(temp_mask.as_mut_ptr(), delim_mask);
                    
                    for (i, &mask_byte) in temp_mask.iter().enumerate() {
                        if mask_byte != 0 {
                            field_positions.push(processed + i);
                        }
                    }
                }
                processed += SIMD_NEON_PROCESS_SIZE;
            }

            for (i, &byte) in bytes[processed..].iter().enumerate() {
                if byte == delimiter_byte {
                    field_positions.push(processed + i);
                }
            }

            let mut fields = Vec::new();
            let mut start = 0;
            
            for &pos in &field_positions {
                if let Ok(field) = std::str::from_utf8(&bytes[start..pos]) {
                    fields.push(field.to_string());
                }
                start = pos + 1;
            }
            
            if start < bytes.len() {
                if let Ok(field) = std::str::from_utf8(&bytes[start..]) {
                    fields.push(field.to_string());
                }
            }

            results.push(fields);
        }

        Ok(results)
    }

    #[target_feature(enable = "neon")]
    unsafe fn simd_to_lowercase_neon(input: uint8x16_t) -> uint8x16_t {
        let a_minus_1 = vdupq_n_u8(b'A' - 1);
        let z_plus_1 = vdupq_n_u8(b'Z' + 1);
        let case_diff = vdupq_n_u8(SIMD_NEON_PROCESS_SIZE as u8);

        let gt_a = vcgtq_u8(input, a_minus_1);
        let lt_z = vcltq_u8(input, z_plus_1);
        let is_upper = vandq_u8(gt_a, lt_z);

        let to_add = vandq_u8(is_upper, case_diff);
        vaddq_u8(input, to_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = detect_cpu_features();
        println!("Detected features: {:?}", features);
    }

    #[test]
    fn test_simd_processor_creation() {
        let processor = SimdStringProcessor::new();
        assert!(processor.is_ok());
        
        let processor = processor.unwrap();
        let capabilities = processor.get_capabilities();
        assert!(capabilities.max_vector_width >= 1);
        assert!(capabilities.theoretical_speedup >= 1.0);
        assert!(capabilities.optimal_chunk_size >= 1024);
    }

    #[test]
    fn test_url_normalization_fallback() {
        let processor = SimdStringProcessor::new().unwrap();
        let urls = vec![
            "HTTP://EXAMPLE.COM".to_string(),
            "HTTPS://TEST.ORG/PATH".to_string(),
            "FTP://FILES.COM".to_string(),
        ];
        
        let normalized = processor.normalize_urls_to_lowercase(&urls).unwrap();
        assert_eq!(normalized[0], "http://example.com");
        assert_eq!(normalized[1], "https://test.org/path");
        assert_eq!(normalized[2], "ftp://files.com");
    }

    #[test]
    fn test_email_validation_fallback() {
        let processor = SimdStringProcessor::new().unwrap();
        let emails = vec![
            "test@example.com".to_string(),
            "invalid-email".to_string(),
            "user@domain.org".to_string(),
            "short".to_string(),
        ];
        
        let results = processor.validate_emails_simd(&emails).unwrap();
        assert_eq!(results[0], true);
        assert_eq!(results[1], false);
        assert_eq!(results[2], true);
        assert_eq!(results[3], false);
    }

    #[test]
    fn test_string_hashing_fallback() {
        let processor = SimdStringProcessor::new().unwrap();
        let strings = vec![
            "test1".to_string(),
            "test2".to_string(),
            "test1".to_string(), // Duplicate
        ];
        
        let hashes = processor.hash_strings_simd(&strings).unwrap();
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], hashes[2]);
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn test_csv_parsing_fallback() {
        let processor = SimdStringProcessor::new().unwrap();
        let lines = vec![
            "field1,field2,field3".to_string(),
            "a,b,c,d".to_string(),
            "single".to_string(),
        ];
        
        let results = processor.parse_csv_fields_simd(&lines, ',').unwrap();
        assert_eq!(results[0], vec!["field1", "field2", "field3"]);
        assert_eq!(results[1], vec!["a", "b", "c", "d"]);
        assert_eq!(results[2], vec!["single"]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_url_normalization() {
        if is_x86_feature_detected!("avx2") {
            let urls = vec![
                "HTTP://EXAMPLE.COM/VERY/LONG/PATH/WITH/MANY/CHARACTERS".to_string(),
                "HTTPS://ANOTHER.VERY.LONG.DOMAIN.NAME.COM/PATH".to_string(),
            ];
            
            let result = avx2_optimizations::normalize_urls_avx2(&urls);
            assert!(result.is_ok());
            
            let normalized = result.unwrap();
            assert_eq!(normalized[0], "http://example.com/very/long/path/with/many/characters");
            assert_eq!(normalized[1], "https://another.very.long.domain.name.com/path");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_email_validation() {
        if is_x86_feature_detected!("avx2") {
            let emails = vec![
                "this.is.a.very.long.email.address@example.com".to_string(),
                "another.long.email.without.domain".to_string(),
            ];
            
            let result = avx2_optimizations::validate_emails_avx2(&emails);
            assert!(result.is_ok());
            
            let validation = result.unwrap();
            assert_eq!(validation[0], true);
            assert_eq!(validation[1], false);
        }
    }
}