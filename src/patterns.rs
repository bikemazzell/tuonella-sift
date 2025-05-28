use once_cell::sync::Lazy;
use regex::Regex;

/// Hardcoded, optimized regex patterns for maximum performance
/// These are compiled once and reused throughout the application

// URL detection patterns (optimized for speed)
pub static PROTOCOL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z][a-zA-Z0-9+.-]*://").unwrap()
});

pub static SUBDOMAIN_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^([a-zA-Z0-9-]+)\.([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$").unwrap()
});

pub static PATH_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"/.*$").unwrap()
});

// Email detection patterns
pub static EMAIL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
});

// Password detection patterns
pub static PASSWORD_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Match passwords: 4+ chars, alphanumeric + some special chars, but exclude @ and ://
    // This should match typical passwords but not emails or URLs
    Regex::new(r"^[a-zA-Z0-9!#$%^&*()_+=\-\[\]{};':,<>/?]{4,}$").unwrap()
});

// URL normalization patterns (optimized)
/* pub static SUBDOMAIN_REMOVAL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(www\.|m\.|mobile\.)").unwrap()
});

pub static QUERY_PARAMS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[?#].*$").unwrap()
});

pub static PATH_CLEANUP_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"/+$").unwrap()
}); */

/// Fast URL normalization using hardcoded patterns
pub fn normalize_url_fast(url: &str) -> String {
    // Handle empty URLs
    if url.is_empty() {
        return String::new();
    }

    // Clean up the input - handle various separators and whitespace
    let result = url.trim().to_lowercase();

    // Handle android URIs specially
    if result.starts_with("android://") {
        if let Some(at_pos) = result.rfind('@') {
            let domain_part = &result[at_pos + 1..];
            // Remove trailing slash if present
            return domain_part.trim_end_matches('/').to_string();
        }
    }

    // Try to parse URL directly first
    if let Ok(parsed_url) = url::Url::parse(&result) {
        if let Some(host) = parsed_url.host_str() {
            let mut host = host.to_string();
            
            // Apply subdomain removal pattern
            if let Some(captures) = SUBDOMAIN_PATTERN.captures(&host) {
                if let Some(main_domain) = captures.get(2) {
                    host = main_domain.as_str().to_string();
                }
            }
            return host;
        }
    }

    // If direct parsing failed, try with http:// prefix
    let mut result = result;
    if !result.starts_with("http://") && 
       !result.starts_with("https://") && 
       !result.starts_with("ftp://") {
        result = format!("http://{}", result);
    }

    // Try parsing with the current protocol
    if let Ok(parsed_url) = url::Url::parse(&result) {
        if let Some(host) = parsed_url.host_str() {
            let mut host = host.to_string();
            
            // Apply subdomain removal pattern
            if let Some(captures) = SUBDOMAIN_PATTERN.captures(&host) {
                if let Some(main_domain) = captures.get(2) {
                    host = main_domain.as_str().to_string();
                }
            }
            return host;
        }
    }

    // Fallback: manual protocol and path removal
    let mut result = result;

    // Extract domain from FTP URL if present
    if result.starts_with("ftp://") {
        let without_protocol = result.strip_prefix("ftp://").unwrap();
        let domain_end = without_protocol.find('/').unwrap_or(without_protocol.len());
        let domain = &without_protocol[..domain_end];
        
        // Try parsing with http:// prefix to use URL parser's host extraction
        let with_http = format!("http://{}", domain);
        if let Ok(parsed_url) = url::Url::parse(&with_http) {
            if let Some(host) = parsed_url.host_str() {
                let mut host = host.to_string();
                
                // Apply subdomain removal pattern
                if let Some(captures) = SUBDOMAIN_PATTERN.captures(&host) {
                    if let Some(main_domain) = captures.get(2) {
                        host = main_domain.as_str().to_string();
                    }
                }
                return host;
            }
        }
        
        // If parsing failed, return the domain as is
        return domain.to_string();
    }

    // Remove other protocols using string manipulation
    if let Some(stripped) = result.strip_prefix("http://") {
        result = stripped.to_string();
    } else if let Some(stripped) = result.strip_prefix("https://") {
        result = stripped.to_string();
    }

    // Remove path using regex pattern
    result = PATH_PATTERN.replace(&result, "").to_string();

    // Remove query parameters
    if let Some(pos) = result.find('?') {
        result = result[..pos].to_string();
    }

    // Remove fragments
    if let Some(pos) = result.find('#') {
        result = result[..pos].to_string();
    }

    // Remove www. prefix if present
    if result.starts_with("www.") {
        result = result[4..].to_string();
    }

    // Apply subdomain removal pattern
    if let Some(captures) = SUBDOMAIN_PATTERN.captures(&result) {
        if let Some(main_domain) = captures.get(2) {
            result = main_domain.as_str().to_string();
        }
    }

    result
}

/* // Helper function for fast domain extraction
fn extract_common_domain(url: &str) -> Option<String> {
    // List of common domains that don't need complex normalization
    const COMMON_DOMAINS: &[&str] = &[
        "facebook.com",
        "google.com",
        "twitter.com",
        "github.com",
        "linkedin.com",
        "instagram.com",
    ];

    for &domain in COMMON_DOMAINS {
        if url.contains(domain) {
            return Some(domain.to_string());
        }
    }
    None
} */

/* /// Fast field detection using hardcoded patterns and caching
pub fn detect_field_type_fast(field: &str) -> FieldType {
    // Fast path for empty fields
    if field.is_empty() {
        return FieldType::Unknown;
    }

    // Fast checks for common patterns without regex
    if field.contains('@') {
        if field.ends_with(".com") || field.ends_with(".net") || field.ends_with(".org") {
            return FieldType::Email;
        }
    }

    if field.starts_with("http://") || field.starts_with("https://") || 
       field.ends_with(".com") || field.ends_with(".net") || field.ends_with(".org") {
        return FieldType::Url;
    }

    // Password heuristics without regex
    if field.len() >= 8 && field.len() <= 64 &&
       field.chars().any(|c| c.is_ascii_digit()) &&
       field.chars().any(|c| c.is_ascii_uppercase()) {
        return FieldType::Password;
    }

    // Fallback to regex patterns for complex cases
    if EMAIL_PATTERN.is_match(field) {
        FieldType::Email
    } else if PROTOCOL_PATTERN.is_match(field) || 
              SUBDOMAIN_PATTERN.is_match(field) {
        FieldType::Url
    } else if PASSWORD_PATTERN.is_match(field) {
        FieldType::Password
    } else {
        FieldType::Unknown
    }
} */

/* pub fn detect_field_type(field: &str) -> FieldType {
    // Use proper URL and email validation instead of hardcoded patterns
    if field.contains('@') {
        // Use proper email validation
        if EMAIL_PATTERN.is_match(field) {
            return FieldType::Email;
        }
    }

    // Use proper URL validation
    if PROTOCOL_PATTERN.is_match(field) || SUBDOMAIN_PATTERN.is_match(field) {
        return FieldType::Url;
    }

    // ... existing code ...

    // Password heuristics without regex
    if field.len() >= 8 && field.len() <= 64 &&
       field.chars().any(|c| c.is_ascii_digit()) &&
       field.chars().any(|c| c.is_ascii_uppercase()) {
        return FieldType::Password;
    }

    // Fallback to regex patterns for complex cases
    if EMAIL_PATTERN.is_match(field) {
        FieldType::Email
    } else if PROTOCOL_PATTERN.is_match(field) || 
              SUBDOMAIN_PATTERN.is_match(field) {
        FieldType::Url
    } else if PASSWORD_PATTERN.is_match(field) {
        FieldType::Password
    } else {
        FieldType::Unknown
    }
} */

/* #[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Email,
    Url,
    Password,
    Unknown,
} */
