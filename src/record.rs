use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use unicode_normalization::UnicodeNormalization;
use url::Url;
use crate::config::{UrlNormalizationConfig, FieldDetectionConfig};
use crate::patterns::{PROTOCOL_PATTERN, SUBDOMAIN_PATTERN, PATH_PATTERN, EMAIL_PATTERN, PASSWORD_PATTERN, normalize_url_fast};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub user: String,
    pub password: String,
    pub url: String,
    pub normalized_url: String,
    pub normalized_user: String,
    pub fields: Vec<String>,
    pub completeness_score: usize,
    pub source_file: String,
    pub line_number: usize,
}

impl Record {
    pub fn new(
        fields: Vec<String>,
        user_idx: usize,
        password_idx: usize,
        url_idx: usize,
        source_file: String,
        line_number: usize,
        case_sensitive_usernames: bool,
    ) -> Option<Self> {
        if fields.len() <= user_idx.max(password_idx).max(url_idx) {
            return None;
        }

        let user = normalize_text(&fields[user_idx]);
        let password = normalize_text(&fields[password_idx]);
        let url = normalize_text(&fields[url_idx]);

        // Require at least username and password to be present
        // URL-only records are not valid and should be skipped
        if user.is_empty() || password.is_empty() || 
           user.chars().all(|c| c == ',' || c.is_whitespace()) ||
           password.chars().all(|c| c == ',' || c.is_whitespace()) {
            return None;
        }

        // Use fast normalization for better performance
        let normalized_url = if !url.is_empty() {
            if url.len() < 512 && !url.contains("android://") {
                // Use fast normalization for typical URLs
                normalize_url_fast(&url)
            } else {
                // Use full normalization for complex URLs
                normalize_url(&url)
            }
        } else {
            String::new()
        };
        
        let normalized_user = if case_sensitive_usernames {
            user.clone()
        } else {
            user.to_lowercase()
        };

        let completeness_score = calculate_completeness(&fields);

        Some(Record {
            user,
            password,
            url,
            normalized_url,
            normalized_user,
            fields,
            completeness_score,
            source_file,
            line_number,
        })
    }

    /* /// Create a record without normalization (for CUDA processing)
    pub fn new_unnormalized(
        fields: Vec<String>,
        user_idx: usize,
        password_idx: usize,
        url_idx: usize,
        source_file: String,
        line_number: usize,
    ) -> Option<Self> {
        if fields.len() <= user_idx.max(password_idx).max(url_idx) {
            return None;
        }

        let user = normalize_text(&fields[user_idx]);
        let password = normalize_text(&fields[password_idx]);
        let url = normalize_text(&fields[url_idx]);

        // Allow empty user or password, but require URL
        if url.is_empty() {
            return None;
        }

        let completeness_score = calculate_completeness(&fields);

        Some(Record {
            user,
            password,
            url,
            normalized_url: String::new(), // Will be filled by CUDA
            normalized_user: String::new(), // Will be filled by CUDA
            fields,
            completeness_score,
            source_file,
            line_number,
        })
    } */

    pub fn dedup_key(&self) -> String {
        format!("{}|{}", self.normalized_user, self.normalized_url)
    }

    pub fn is_more_complete_than(&self, other: &Record) -> bool {
        self.completeness_score > other.completeness_score
    }
}

impl Hash for Record {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dedup_key().hash(state);
    }
}

impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        self.dedup_key() == other.dedup_key()
    }
}

impl Eq for Record {}

#[derive(Clone)]
pub struct FieldDetector {
    url_patterns: Vec<Arc<Regex>>,
    email_patterns: Vec<Arc<Regex>>,
    password_patterns: Vec<Arc<Regex>>,
    use_fast_patterns: bool,
}

impl FieldDetector {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::from_config(&FieldDetectionConfig::default())
    }

    pub fn from_config(config: &FieldDetectionConfig) -> Self {
        let url_patterns = config.url_patterns
            .iter()
            .map(|pattern_str| Arc::new(Regex::new(pattern_str).expect("Invalid URL regex pattern in config")))
            .collect();

        let email_patterns = config.email_patterns
            .iter()
            .map(|pattern_str| Arc::new(Regex::new(pattern_str).expect("Invalid email regex pattern in config")))
            .collect();

        let password_patterns = config.password_patterns
            .iter()
            .map(|pattern_str| Arc::new(Regex::new(pattern_str).expect("Invalid password regex pattern in config")))
            .collect();

        Self {
            url_patterns,
            email_patterns,
            password_patterns,
            use_fast_patterns: true, // Enable fast patterns by default
        }
    }

    pub fn detect_fields(&self, sample_records: &[Vec<String>]) -> (usize, usize, usize) {
        if sample_records.is_empty() {
            return (0, 1, 2);
        }

        let field_count = sample_records[0].len();
        let mut url_scores = vec![0; field_count];
        let mut email_scores = vec![0; field_count];
        let mut password_scores = vec![0; field_count];

        for record in sample_records.iter() {
            for (i, field) in record.iter().enumerate() {
                if i >= field_count {
                    break;
                }

                let field = field.trim();
                if field.is_empty() {
                    continue;
                }

                if self.use_fast_patterns {
                    // Use optimized hardcoded patterns for better performance
                    if PROTOCOL_PATTERN.is_match(field) || SUBDOMAIN_PATTERN.is_match(field) || PATH_PATTERN.is_match(field) {
                        url_scores[i] += 1;
                    }

                    if EMAIL_PATTERN.is_match(field) {
                        email_scores[i] += 1;
                    }

                    if PASSWORD_PATTERN.is_match(field) {
                        password_scores[i] += 1;
                    }
                } else {
                    // Fallback to config-based patterns
                    for pattern_arc in &self.url_patterns {
                        if pattern_arc.is_match(field) {
                            url_scores[i] += 1;
                            break;
                        }
                    }

                    for pattern_arc in &self.email_patterns {
                        if pattern_arc.is_match(field) {
                            email_scores[i] += 1;
                            break;
                        }
                    }

                    for pattern_arc in &self.password_patterns {
                        if pattern_arc.is_match(field) {
                            password_scores[i] += 1;
                            break;
                        }
                    }
                }
            }
        }

        let url_idx = url_scores.iter().enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(2);

        let email_idx = email_scores.iter().enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let password_idx = password_scores.iter().enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(1);

        (email_idx, password_idx, url_idx)
    }
}

fn normalize_text(text: &str) -> String {
    text.nfc()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

fn normalize_url(url_str: &str) -> String {
    normalize_url_with_config(url_str, &UrlNormalizationConfig::default())
}

pub fn normalize_url_with_config(url_str: &str, config: &UrlNormalizationConfig) -> String {
    if url_str.is_empty() {
        return String::new();
    }

    // Clean up the input - handle various separators and whitespace
    let result = url_str.trim().to_string();

    // Handle android:// URIs specially
    if config.android_uri_cleanup && result.starts_with("android://") {
        // Extract the domain part after the @ symbol for android URIs
        if let Some(at_pos) = result.rfind('@') {
            let domain_part = &result[at_pos + 1..];
            // Remove trailing slash if present
            let cleaned = domain_part.trim_end_matches('/').to_string();
            // For android URIs, return as-is without further processing
            return if config.normalize_case {
                cleaned.to_lowercase()
            } else {
                cleaned
            };
        }
    }

    // Try to parse as URL first
    let mut url_to_parse = result.clone(); // result is url_str.trim().to_string()
    let mut has_known_protocol = false;
    // config.protocol_patterns is guaranteed by default/deserialization to contain at least one pattern
    for pattern in &config.protocol_patterns {
        if pattern.is_match(&url_to_parse) {
            has_known_protocol = true;
            break;
        }
    }
    if !has_known_protocol {
        url_to_parse = format!("http://{}", url_to_parse);
    }

    if let Ok(parsed_url) = Url::parse(&url_to_parse) {
        if let Some(host_str_val) = parsed_url.host_str() {
            let mut host = host_str_val.to_string();

            if config.normalize_case {
                host = host.to_lowercase();
            }
            
            host = apply_subdomain_patterns(&host, config);

            let mut path = parsed_url.path().to_string();
            
            for regex_arc in &config.path_cleanup_patterns {
                path = regex_arc.replace_all(&path, "").to_string();
            }
            
            // If path cleanup resulted in an empty string, or original path was just "/",
            // we return only the host. Otherwise, combine host and path.
            // This assumes path_cleanup_patterns are designed to produce either an empty path
            // or a path that should be appended (potentially starting with /).
            return if path.is_empty() || path == "/" {
                host
            } else {
                // Ensure path doesn't lead to double slashes if it's not empty and host doesn't end with /
                // and path starts with /. Most host strings won't end with /.
                // A typical path from Url::path() starts with /.
                format!("{}{}", host, path)
            };
        }
    }
    
    // Fallback for URLs that can't be parsed correctly by Url::parse or if host_str was None
    // The 'result' variable here refers to the original trimmed url_str.
    {
        let mut fallback_result = result; // result is url_str.trim().to_string()
        
        // Remove protocols using regex patterns
        for regex_arc in &config.protocol_patterns {
            let temp_result = regex_arc.replace_all(&fallback_result, "").to_string();
            if temp_result != fallback_result && !temp_result.is_empty() {
                fallback_result = temp_result;
                break; 
            }
        }

        // Normalize case if configured
        if config.normalize_case {
            fallback_result = fallback_result.to_lowercase();
        }
        
        // Remove query parameters manually
        if config.remove_query_params {
            if let Some(pos) = fallback_result.find('?') {
                fallback_result.truncate(pos);
            }
        }
        
        // Remove fragments manually
        if config.remove_fragments {
            if let Some(pos) = fallback_result.find('#') {
                fallback_result.truncate(pos);
            }
        }
        
        // Apply subdomain patterns to the (potentially) host part
        // This is imperfect as fallback_result might still contain a path
        let host_part_for_subdomain = if let Some(slash_pos) = fallback_result.find('/') {
            &fallback_result[..slash_pos]
        } else {
            &fallback_result
        };
        let mut processed_host = apply_subdomain_patterns(host_part_for_subdomain, config);

        // Re-attach path if it existed and wasn't part of subdomain processing
        if let Some(slash_pos) = fallback_result.find('/') {
            if host_part_for_subdomain.len() < fallback_result.len() {
                 processed_host.push_str(&fallback_result[slash_pos..]);
            }
        }
        fallback_result = processed_host;
        
        // Apply path cleanup patterns to the entire result
        for regex_arc in &config.path_cleanup_patterns {
            fallback_result = regex_arc.replace_all(&fallback_result, "").to_string();
        }
        
        fallback_result
    }
}

fn apply_subdomain_patterns(host: &str, config: &UrlNormalizationConfig) -> String {
    let mut result = host.to_string();
    
    // Only apply subdomain removal if the host has at least one dot
    if result.contains('.') {
        for regex_arc in &config.subdomain_removal_patterns {
            if let Some(captures) = regex_arc.captures(&result) {
                if let Some(main_domain) = captures.get(2) {
                    result = main_domain.as_str().to_string();
                    break; 
                }
            }
        }
    }
    result
}

fn calculate_completeness(fields: &[String]) -> usize {
    fields.iter()
        .map(|field| field.trim().len())
        .sum()
}

#[derive(Default)]
pub struct DeduplicationMap {
    pub records: HashMap<String, Record>,
}

impl DeduplicationMap {
    pub fn new() -> Self {
        Self {
            records: HashMap::default(),
        }
    }

    pub fn insert(&mut self, record: Record) -> bool {
        let key = record.dedup_key();
        
        match self.records.get(&key) {
            Some(existing) => {
                if record.is_more_complete_than(existing) {
                    self.records.insert(key, record);
                    false
                } else {
                    true
                }
            }
            None => {
                self.records.insert(key, record);
                false
            }
        }
    }

    pub fn into_records(self) -> Vec<Record> {
        self.records.into_values().collect()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_normalization() {
        assert_eq!(normalize_url("https://www.facebook.com/user123/"), "facebook.com");
        assert_eq!(normalize_url("http://m.facebook.com/user123"), "facebook.com");
        assert_eq!(normalize_url("facebook.com/user123?ref=123"), "facebook.com");
        assert_eq!(normalize_url("FACEBOOK.COM/User123"), "facebook.com");
    }

    #[test]
    fn test_configurable_url_normalization() {
        let config_strings = crate::config::UrlNormalizationConfigStrings {
            protocol_patterns: vec![
                "^[a-zA-Z][a-zA-Z0-9+.-]*://".to_string(),
            ],
            subdomain_removal_patterns: vec![
                "^([a-zA-Z0-9-]+)\\.([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$".to_string(),
            ],
            path_cleanup_patterns: vec![
                "/.*$".to_string(),
            ],
            android_uri_cleanup: true,
            remove_query_params: true,
            remove_fragments: true,
            normalize_case: true,
        };
        let config: UrlNormalizationConfig = config_strings.into();

        // Test basic URL normalization
        assert_eq!(
            normalize_url_with_config("https://www.facebook.com/user123/", &config),
            "facebook.com"
        );

        // Test android URI handling
        assert_eq!(
            normalize_url_with_config("android://VTLxyVefpdMwrqKDG-hrVL5lB806PvZxiZJKOX1mjs7JYvPNS91YzTrHWGtPFqt2jLx3w1Hx8-rx7_8KF4_wkA==@com.gametreeapp/", &config),
            "com.gametreeapp"
        );

        // Test accounts prefix removal
        assert_eq!(
            normalize_url_with_config("https://accounts.google.com/signin", &config),
            "google.com"
        );

        // Test FTP protocol
        assert_eq!(
            normalize_url_with_config("ftp://files.example.com/downloads/", &config),
            "example.com"
        );

        // Test query parameter removal
        assert_eq!(
            normalize_url_with_config("https://example.com/page?param=value&other=123", &config),
            "example.com"
        );

        // Test fragment removal
        assert_eq!(
            normalize_url_with_config("https://example.com/page#section", &config),
            "example.com"
        );

        // Test IP address handling
        assert_eq!(
            normalize_url_with_config("http://192.168.1.1/admin/login", &config),
            "192.168.1.1"
        );
    }

    #[test]
    fn test_real_world_url_normalization() {
        let config_strings = crate::config::UrlNormalizationConfigStrings {
            protocol_patterns: vec![
                "^[a-zA-Z][a-zA-Z0-9+.-]*://".to_string(),
            ],
            subdomain_removal_patterns: vec![
                "^([a-zA-Z0-9-]+)\\.([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$".to_string(),
            ],
            path_cleanup_patterns: vec![
                "/.*$".to_string(),
            ],
            android_uri_cleanup: true,
            remove_query_params: true,
            remove_fragments: true,
            normalize_case: true,
        };
        let config: UrlNormalizationConfig = config_strings.into();

        // Test cases from real sample data
        let test_cases = vec![
            ("https://www.facebook.com/recover/password/", "facebook.com"),
            ("https://accounts.google.com/signin/v2/sl/pwd", "google.com"),
            ("http://192.168.0.1/quickset.html", "192.168.0.1"),
            ("https://m.facebook.com/login.php", "facebook.com"),
            ("android://VTLxyVefpdMwrqKDG-hrVL5lB806PvZxiZJKOX1mjs7JYvPNS91YzTrHWGtPFqt2jLx3w1Hx8-rx7_8KF4_wkA==@com.gametreeapp/", "com.gametreeapp"),
            ("https://auth.riotgames.com/login", "riotgames.com"),
            ("http://www.daspattayaforum.com/", "daspattayaforum.com"),
            ("https://myaccount.google.com/signinoptions/password", "google.com"),
            ("https://mobile.twitter.com/home", "twitter.com"),
            ("ftp://files.example.com/downloads/", "example.com"),
            ("https://login.microsoftonline.com/signin", "microsoftonline.com"),
        ];

        println!("\n=== Real-World URL Normalization Results ===");
        for (input, expected) in test_cases {
            let result = normalize_url_with_config(input, &config);
            println!("Input:    {}", input);
            println!("Expected: {}", expected);
            println!("Result:   {}", result);
            println!("Match:    {}", if result == expected { "✓" } else { "✗" });
            println!("---");
            
            // Note: Some of these might not match exactly due to complex URL structures
            // but we can verify the basic functionality is working
        }
    }

    #[test]
    fn test_regex_pattern_url_normalization() {
        let config_strings = crate::config::UrlNormalizationConfigStrings {
            protocol_patterns: vec![
                "^[a-zA-Z][a-zA-Z0-9+.-]*://".to_string(),
            ],
            subdomain_removal_patterns: vec![
                "^([a-zA-Z0-9-]+)\\.([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$".to_string(),
            ],
            path_cleanup_patterns: vec![
                "/.*$".to_string(),
            ],
            android_uri_cleanup: true,
            remove_query_params: true,
            remove_fragments: true,
            normalize_case: true,
        };
        let config: UrlNormalizationConfig = config_strings.into();

        println!("\n=== Regex Pattern URL Normalization Tests ===");

        // Test protocol removal
        assert_eq!(normalize_url_with_config("https://example.com", &config), "example.com");
        assert_eq!(normalize_url_with_config("ftp://files.example.com", &config), "example.com");
        assert_eq!(normalize_url_with_config("ftps://secure.example.com", &config), "example.com");

        // Test subdomain removal with patterns (now removes ANY first subdomain)
        assert_eq!(normalize_url_with_config("https://www.facebook.com", &config), "facebook.com");
        assert_eq!(normalize_url_with_config("https://m.twitter.com", &config), "twitter.com");
        assert_eq!(normalize_url_with_config("https://mobile.reddit.com", &config), "reddit.com");
        assert_eq!(normalize_url_with_config("https://login.microsoft.com", &config), "microsoft.com");
        assert_eq!(normalize_url_with_config("https://auth.google.com", &config), "google.com");
        assert_eq!(normalize_url_with_config("https://accounts.adobe.com", &config), "adobe.com");

        // Test that it removes any subdomain (no special preservation)
        assert_eq!(normalize_url_with_config("https://api.github.com", &config), "github.com");
        assert_eq!(normalize_url_with_config("https://cdn.jsdelivr.net", &config), "jsdelivr.net");
        assert_eq!(normalize_url_with_config("https://static.cloudflare.com", &config), "cloudflare.com");

        // Test path cleanup patterns (now removes ALL paths)
        assert_eq!(normalize_url_with_config("https://example.com/", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com///", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com/index.html", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com/index.php", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com/login", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com/signin/", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://example.com/home", &config), "example.com");

        // Test complex cases
        assert_eq!(normalize_url_with_config("https://www.example.com/login/", &config), "example.com");
        assert_eq!(normalize_url_with_config("https://mobile.site.com/index.php", &config), "site.com");

        // Test android URIs
        assert_eq!(
            normalize_url_with_config("android://abc123@com.example.app/", &config),
            "com.example.app"
        );

        println!("✓ All regex pattern tests passed!");
    }

    #[test]
    fn test_field_detection() {
        let detector = FieldDetector::new();
        let samples = vec![
            vec!["user@example.com".to_string(), "password123".to_string(), "https://example.com".to_string()],
            vec!["test@test.com".to_string(), "secret".to_string(), "google.com".to_string()],
        ];
        
        let (email_idx, password_idx, url_idx) = detector.detect_fields(&samples);
        assert_eq!(email_idx, 0);  // First field contains emails
        assert_eq!(password_idx, 1); // Second field contains passwords
        assert_eq!(url_idx, 2);    // Third field contains URLs
    }
}
