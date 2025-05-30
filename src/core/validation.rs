use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use anyhow::Result;
use crate::constants::{PRINTABLE_USERNAME_MIN_LENGTH, PRINTABLE_USERNAME_MAX_LENGTH};

// Regular expressions for validation
pub static EMAIL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap()
});

pub static DELIMITER_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[,;\t|]").unwrap()
});

// Regex for validating printable usernames (any printable ASCII characters)
pub static PRINTABLE_USERNAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    let pattern = format!(r"^[\x21-\x7E]{{{},{}}}$", PRINTABLE_USERNAME_MIN_LENGTH, PRINTABLE_USERNAME_MAX_LENGTH);
    Regex::new(&pattern).unwrap()
});

/// Normalizes a URL by removing protocol prefixes, www, and paths
///
/// Algorithm:
/// 1. Convert to lowercase
/// 2. Remove protocol (http://, https://, etc.)
/// 3. Handle Android URLs specially (extract domain after @)
/// 4. Remove www prefix
/// 5. Extract domain (before paths, queries, etc.)
/// 6. Remove trailing slashes
pub fn normalize_url(url: &str) -> String {
    if url.is_empty() {
        return String::new();
    }

    let mut normalized = url.to_lowercase();

    // Handle Android URLs specially
    if normalized.starts_with("android://") {
        // Extract domain after @ symbol for Android URLs
        if let Some(at_pos) = normalized.rfind('@') {
            normalized = normalized[at_pos + 1..].to_string();

            // Remove trailing slash and anything after it
            if let Some(slash_pos) = normalized.find('/') {
                normalized = normalized[..slash_pos].to_string();
            }

            // Remove trailing slash if still present
            if normalized.ends_with('/') {
                normalized.pop();
            }

            return normalized;
        }
    }

    // Remove all protocol prefixes (https://, http://, ftp://, mailto://, etc.)
    for prefix in &["https://", "http://", "android://", "ftp://", "mailto://"] {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].to_string();
            break;
        }
    }

    // Remove www. prefix
    if normalized.starts_with("www.") {
        normalized = normalized[4..].to_string();
    }

    // Extract domain (and port if present) before any path, query parameters, or fragments
    if let Some(slash_pos) = normalized.find('/') {
        normalized = normalized[..slash_pos].to_string();
    }

    if let Some(query_pos) = normalized.find('?') {
        normalized = normalized[..query_pos].to_string();
    }

    if let Some(fragment_pos) = normalized.find('#') {
        normalized = normalized[..fragment_pos].to_string();
    }

    // Remove trailing slash if present
    if normalized.ends_with('/') {
        normalized.pop();
    }

    normalized
}

/// Parses a CSV line into fields
///
/// Handles:
/// - Quoted fields (removes outer quotes)
/// - Escaped quotes (converts "" to ")
/// - Different delimiters (comma, semicolon, tab, pipe)
pub fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    // Escaped quote - add single quote to field
                    current_field.push('"');
                    chars.next();
                } else {
                    // Toggle quote state but don't add quote to field
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                fields.push(current_field.trim().to_string());
                current_field.clear();
            }
            _ => {
                current_field.push(ch);
            }
        }
    }

    fields.push(current_field.trim().to_string());
    fields
}

/// Detects field positions in a parsed CSV line
///
/// Tries to intelligently identify which fields contain:
/// - Email/username
/// - Password
/// - URL
///
/// Returns a tuple of (user_idx, password_idx, url_idx)
pub fn detect_field_positions(fields: &[String]) -> (usize, usize, usize) {
    let mut user_idx = 0;
    let mut password_idx = 1;
    let mut url_idx = 2;

    // First pass: identify URL fields (most distinctive)
    for (i, field) in fields.iter().enumerate() {
        if is_url_field(field) {
            url_idx = i;
            break; // Take the first URL field found
        }
    }

    // Second pass: identify email fields for username
    let mut found_email = false;
    for (i, field) in fields.iter().enumerate() {
        if i != url_idx && EMAIL_REGEX.is_match(field) {
            user_idx = i;
            found_email = true;
            break; // Take the first email field found
        }
    }

    // If no email found, this should not happen but handle gracefully
    if !found_email {
        return (fields.len(), fields.len(), fields.len()); // Invalid indices
    }

    // Third pass: find password field (the remaining field)
    for i in 0..fields.len() {
        if i != user_idx && i != url_idx {
            password_idx = i;
            break;
        }
    }

    // Ensure all indices are different and within bounds
    if user_idx >= fields.len() { user_idx = 0; }
    if password_idx >= fields.len() { password_idx = (user_idx + 1) % fields.len(); }
    if url_idx >= fields.len() { url_idx = (password_idx + 1) % fields.len(); }

    // Final validation: ensure all indices are unique
    if user_idx == url_idx || user_idx == password_idx || password_idx == url_idx {
        // Fallback to simple positional assignment
        url_idx = url_idx.min(fields.len() - 1);
        user_idx = (url_idx + 1) % fields.len();
        password_idx = (url_idx + 2) % fields.len();
    }

    (user_idx, password_idx, url_idx)
}

/// Detects field positions in a parsed CSV line with configuration
///
/// Tries to intelligently identify which fields contain:
/// - Email/username (based on email_username_only flag)
/// - Password
/// - URL
///
/// Returns a tuple of (user_idx, password_idx, url_idx)
pub fn detect_field_positions_with_config(fields: &[String], email_username_only: bool) -> (usize, usize, usize) {
    let mut user_idx = 0;
    let mut password_idx = 1;
    let mut url_idx = 2;

    // First pass: identify URL fields (most distinctive)
    for (i, field) in fields.iter().enumerate() {
        if is_url_field(field) {
            url_idx = i;
            break; // Take the first URL field found
        }
    }

    // Second pass: identify username fields
    let mut found_user = false;
    for (i, field) in fields.iter().enumerate() {
        if i != url_idx {
            let is_valid_user = if email_username_only {
                EMAIL_REGEX.is_match(field)
            } else {
                // For non-email usernames, check if it's a valid printable string
                // and not obviously a URL or password-like field
                is_valid_username_field(field)
            };

            if is_valid_user {
                user_idx = i;
                found_user = true;
                break; // Take the first valid username field found
            }
        }
    }

    // If no valid username found, return invalid indices
    if !found_user {
        return (fields.len(), fields.len(), fields.len()); // Invalid indices
    }

    // Third pass: find password field (the remaining field)
    for i in 0..fields.len() {
        if i != user_idx && i != url_idx {
            password_idx = i;
            break;
        }
    }

    // Ensure all indices are different and within bounds
    if user_idx >= fields.len() { user_idx = 0; }
    if password_idx >= fields.len() { password_idx = (user_idx + 1) % fields.len(); }
    if url_idx >= fields.len() { url_idx = (password_idx + 1) % fields.len(); }

    // Final validation: ensure all indices are unique
    if user_idx == url_idx || user_idx == password_idx || password_idx == url_idx {
        // Fallback to simple positional assignment
        url_idx = url_idx.min(fields.len() - 1);
        user_idx = (url_idx + 1) % fields.len();
        password_idx = (url_idx + 2) % fields.len();
    }

    (user_idx, password_idx, url_idx)
}

/// Checks if a field is a valid username (printable characters, not URL-like)
pub fn is_valid_username_field(field: &str) -> bool {
    // Check for empty or whitespace-only fields
    if field.trim().is_empty() {
        return false;
    }

    // Check basic printable character requirements
    if !PRINTABLE_USERNAME_REGEX.is_match(field) {
        return false;
    }

    // Exclude fields that look like URLs
    if is_url_field(field) {
        return false;
    }

    // Exclude fields that contain URL-like patterns
    if field.starts_with("http://") || field.starts_with("https://") ||
       field.starts_with("android://") || field.starts_with("ftp://") {
        return false;
    }

    // Exclude fields that look like long random strings (likely passwords)
    // This is a heuristic - very long strings with mixed case and numbers are likely passwords
    if field.len() > 50 && field.chars().any(|c| c.is_uppercase()) &&
       field.chars().any(|c| c.is_lowercase()) && field.chars().any(|c| c.is_numeric()) {
        return false;
    }

    true
}

/// Validates a CSV line with configuration
///
/// Checks for:
/// - Printable characters
/// - Delimiter presence
/// - Username presence (email or printable based on config)
///
/// Returns true if line is valid, false otherwise
pub fn is_valid_line_with_config(line: &str, email_username_only: bool) -> bool {
    // Skip if no printable characters
    if !line.chars().any(|c| c.is_ascii_graphic()) {
        return false;
    }

    // Skip if no delimiter
    if !DELIMITER_REGEX.is_match(line) {
        return false;
    }

    // Check for username presence based on configuration
    if email_username_only {
        // Skip if no email address
        if !EMAIL_REGEX.is_match(line) {
            return false;
        }
    } else {
        // For non-email mode, check if line has at least one field that could be a username
        let fields = parse_csv_line(line);
        if fields.len() < 3 {
            return false;
        }

        let has_valid_username = fields.iter().any(|field| is_valid_username_field(field));
        if !has_valid_username {
            return false;
        }
    }

    true
}

/// Checks if a field appears to be a URL
fn is_url_field(field: &str) -> bool {
    // Check for various URL patterns with protocols (highest priority)
    if field.starts_with("http://") ||
       field.starts_with("https://") ||
       field.starts_with("android://") ||
       field.starts_with("ftp://") ||
       field.starts_with("mailto://") {
        return true;
    }

    // Check for domain-like patterns with paths/queries/fragments
    if field.contains('.') && (field.contains('/') || field.contains('?') || field.contains('#')) {
        return true;
    }

    // Check for reverse domain notation (Android apps)
    if field.contains('.') && field.split('.').count() >= 4 && !field.contains('@') && field.len() > 30 {
        return true;
    }

    false
}

/// Validates a CSV line
///
/// Checks for:
/// - Printable characters
/// - Delimiter presence
/// - Email presence
///
/// Returns true if line is valid, false otherwise
pub fn is_valid_line(line: &str) -> bool {
    // Skip if no printable characters
    if !line.chars().any(|c| c.is_ascii_graphic()) {
        return false;
    }

    // Skip if no delimiter
    if !DELIMITER_REGEX.is_match(line) {
        return false;
    }

    // Skip if no email address
    if !EMAIL_REGEX.is_match(line) {
        return false;
    }

    true
}

/// Discovers CSV files in a directory
pub fn discover_csv_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut csv_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().to_lowercase() == "csv" {
                    csv_files.push(path);
                }
            }
        }
    }

    csv_files.sort();
    Ok(csv_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_normalization() {
        // Test basic HTTPS URLs
        assert_eq!(normalize_url("https://ais.usvisa-info.com/es-co/niv/users/sign_in"), "ais.usvisa-info.com");
        assert_eq!(normalize_url("https://www.netflix.com/it/Login"), "netflix.com");
        assert_eq!(normalize_url("http://www.example.com/path?query=1#fragment"), "example.com");

        // Test Android URLs
        assert_eq!(
            normalize_url("android://QbSh06Ymwz0AAlEe1U4nVUM0baM8f4MrbJ7I5B9aCy8fjqjh2PiCa__sDoO5djRyh8Or2msG7cJu5koaF8eAGA==@com.smule.singandroid/"),
            "com.smule.singandroid"
        );

        // Test various protocols
        assert_eq!(normalize_url("ftp://files.example.com/download"), "files.example.com");
        assert_eq!(normalize_url("mailto://user@example.com"), "user@example.com");

        // Test edge cases
        assert_eq!(normalize_url(""), "");
        assert_eq!(normalize_url("example.com"), "example.com");
        assert_eq!(normalize_url("www.example.com"), "example.com");
        assert_eq!(normalize_url("example.com/"), "example.com");
    }

    #[test]
    fn test_csv_line_parsing() {
        let line = "user@example.com,password123,\"https://example.com/login\"";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "user@example.com");
        assert_eq!(fields[1], "password123");
        assert_eq!(fields[2], "https://example.com/login");

        // Test with quotes and escaped quotes
        let line = "\"user,with,commas\"@example.com,\"pass\"\"word\",http://site.com";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "user,with,commas@example.com");
        assert_eq!(fields[1], "pass\"word");
        assert_eq!(fields[2], "http://site.com");
    }

    #[test]
    fn test_field_detection() {
        // Test with email, URL, and password
        let fields = vec![
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string()
        ];
        let (user_idx, password_idx, url_idx) = detect_field_positions(&fields);
        assert_eq!(user_idx, 0);
        assert_eq!(password_idx, 1);
        assert_eq!(url_idx, 2);

        // Test with fields in different order
        let fields = vec![
            "https://site.com".to_string(),
            "user@site.com".to_string(),
            "pass123".to_string()
        ];
        let (user_idx, password_idx, url_idx) = detect_field_positions(&fields);
        assert_eq!(user_idx, 1);
        assert_eq!(password_idx, 2);
        assert_eq!(url_idx, 0);
    }

    #[test]
    fn test_line_validation() {
        // Valid line
        assert!(is_valid_line("user@example.com,password123,https://example.com"));

        // Invalid lines
        assert!(!is_valid_line(""));  // Empty
        assert!(!is_valid_line("   "));  // Whitespace only
        assert!(!is_valid_line("not an email,password,site.com"));  // No email
        assert!(!is_valid_line("user@example.com password site.com"));  // No delimiter
    }
}