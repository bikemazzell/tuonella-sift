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
    let trimmed_url = url.trim();
    if trimmed_url.is_empty() {
        return String::new();
    }

    // Always lowercase for consistent processing
    let lowercased_url = trimmed_url.to_lowercase();

    // Attempt to strip any protocol.
    // PROTOCOL_PATTERN is ^[a-zA-Z][a-zA-Z0-9+.-]*://
    let protocol_stripped_url = PROTOCOL_PATTERN.replace(&lowercased_url, "").to_string();
    
    // Prepend "http://" to the protocol-stripped URL to help Url::parse identify the host.
    // This is useful for inputs like "example.com/path" which become "http://example.com/path".
    let parse_candidate = format!("http://{}", protocol_stripped_url);

    if let Ok(parsed_url) = url::Url::parse(&parse_candidate) {
        if let Some(host_str) = parsed_url.host_str() {
            let mut host = host_str.to_string();

            // Remove www. prefix specifically, as it's very common
            if host.starts_with("www.") {
                host = host[4..].to_string();
            }

            // Apply general subdomain removal pattern.
            // SUBDOMAIN_PATTERN captures `sub.domain.tld` as `sub` and `domain.tld`.
            // We want `domain.tld` (group 2).
            // If it doesn't match (e.g., "example.com", "localhost"), use host as is.
            if let Some(captures) = SUBDOMAIN_PATTERN.captures(&host) {
                if let Some(main_domain) = captures.get(2) {
                    return main_domain.as_str().to_string();
                }
            }
            return host; // Return (potentially www-stripped) host
        }
    }

    // Fallback: If Url::parse failed or didn't yield a host, manually process.
    // Start with the already protocol-stripped and lowercased string.
    let mut fallback_result = protocol_stripped_url;

    // Remove path using regex pattern (e.g., /.*$)
    fallback_result = PATH_PATTERN.replace(&fallback_result, "").to_string();

    // Remove query parameters
    if let Some(pos) = fallback_result.find('?') {
        fallback_result.truncate(pos);
    }

    // Remove fragments
    if let Some(pos) = fallback_result.find('#') {
        fallback_result.truncate(pos);
    }

    // Remove www. prefix if present (might be missed if parsing failed early)
    if fallback_result.starts_with("www.") {
        fallback_result = fallback_result[4..].to_string();
    }

    // Apply subdomain removal pattern again for the fallback processed string
    if let Some(captures) = SUBDOMAIN_PATTERN.captures(&fallback_result) {
        if let Some(main_domain) = captures.get(2) {
            fallback_result = main_domain.as_str().to_string();
        }
    }

    fallback_result
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
