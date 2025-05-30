use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use anyhow::Result;
use crate::constants::{
    PRINTABLE_USERNAME_MIN_LENGTH, PRINTABLE_USERNAME_MAX_LENGTH,
    PROTOCOL_HTTP, PROTOCOL_HTTPS, PROTOCOL_ANDROID, PROTOCOL_FTP, PROTOCOL_MAILTO,
    URL_WWW_PREFIX, CSV_EXTENSION, LONG_PASSWORD_HEURISTIC_LENGTH,
    REVERSE_DOMAIN_MIN_LENGTH, REVERSE_DOMAIN_MIN_PARTS, MIN_FIELD_COUNT
};

pub static EMAIL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap()
});

pub static DELIMITER_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[,;\t|]").unwrap()
});

pub static PRINTABLE_USERNAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    let pattern = format!(r"^[\x21-\x7E]{{{},{}}}$", PRINTABLE_USERNAME_MIN_LENGTH, PRINTABLE_USERNAME_MAX_LENGTH);
    Regex::new(&pattern).unwrap()
});

pub fn normalize_url(url: &str) -> String {
    if url.is_empty() {
        return String::new();
    }

    let mut normalized = url.to_lowercase();

    if normalized.starts_with(PROTOCOL_ANDROID) {
        if let Some(at_pos) = normalized.rfind('@') {
            normalized = normalized[at_pos + 1..].to_string();

            if let Some(slash_pos) = normalized.find('/') {
                normalized = normalized[..slash_pos].to_string();
            }

            if normalized.ends_with('/') {
                normalized.pop();
            }

            return normalized;
        }
    }

    for prefix in &[PROTOCOL_HTTPS, PROTOCOL_HTTP, PROTOCOL_ANDROID, PROTOCOL_FTP, PROTOCOL_MAILTO] {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].to_string();
            break;
        }
    }

    if normalized.starts_with(URL_WWW_PREFIX) {
        normalized = normalized[URL_WWW_PREFIX.len()..].to_string();
    }

    if let Some(slash_pos) = normalized.find('/') {
        normalized = normalized[..slash_pos].to_string();
    }

    if let Some(query_pos) = normalized.find('?') {
        normalized = normalized[..query_pos].to_string();
    }

    if let Some(fragment_pos) = normalized.find('#') {
        normalized = normalized[..fragment_pos].to_string();
    }

    if normalized.ends_with('/') {
        normalized.pop();
    }

    normalized
}

pub fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    current_field.push('"');
                    chars.next();
                } else {
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

pub fn detect_field_positions(fields: &[String]) -> (usize, usize, usize) {
    let mut user_idx = 0;
    let mut password_idx = 1;
    let mut url_idx = 2;

    for (i, field) in fields.iter().enumerate() {
        if is_url_field(field) {
            url_idx = i;
            break;
        }
    }

    let mut found_email = false;
    for (i, field) in fields.iter().enumerate() {
        if i != url_idx && EMAIL_REGEX.is_match(field) {
            user_idx = i;
            found_email = true;
            break;
        }
    }

    if !found_email {
        return (fields.len(), fields.len(), fields.len());
    }

    for i in 0..fields.len() {
        if i != user_idx && i != url_idx {
            password_idx = i;
            break;
        }
    }

    if user_idx >= fields.len() { user_idx = 0; }
    if password_idx >= fields.len() { password_idx = (user_idx + 1) % fields.len(); }
    if url_idx >= fields.len() { url_idx = (password_idx + 1) % fields.len(); }

    if user_idx == url_idx || user_idx == password_idx || password_idx == url_idx {
        url_idx = url_idx.min(fields.len() - 1);
        user_idx = (url_idx + 1) % fields.len();
        password_idx = (url_idx + 2) % fields.len();
    }

    (user_idx, password_idx, url_idx)
}

pub fn detect_field_positions_with_config(fields: &[String], email_username_only: bool) -> (usize, usize, usize) {
    let mut user_idx = 0;
    let mut password_idx = 1;
    let mut url_idx = 2;

    for (i, field) in fields.iter().enumerate() {
        if is_url_field(field) {
            url_idx = i;
            break;
        }
    }

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
                break;
            }
        }
    }

    if !found_user {
        return (fields.len(), fields.len(), fields.len());
    }

    for i in 0..fields.len() {
        if i != user_idx && i != url_idx {
            password_idx = i;
            break;
        }
    }

    if user_idx >= fields.len() { user_idx = 0; }
    if password_idx >= fields.len() { password_idx = (user_idx + 1) % fields.len(); }
    if url_idx >= fields.len() { url_idx = (password_idx + 1) % fields.len(); }

    if user_idx == url_idx || user_idx == password_idx || password_idx == url_idx {
        url_idx = url_idx.min(fields.len() - 1);
        user_idx = (url_idx + 1) % fields.len();
        password_idx = (url_idx + 2) % fields.len();
    }

    (user_idx, password_idx, url_idx)
}

pub fn is_valid_username_field(field: &str) -> bool {
    if field.trim().is_empty() {
        return false;
    }

    if !PRINTABLE_USERNAME_REGEX.is_match(field) {
        return false;
    }

    if is_url_field(field) {
        return false;
    }

    if field.starts_with(PROTOCOL_HTTP) || field.starts_with(PROTOCOL_HTTPS) ||
       field.starts_with(PROTOCOL_ANDROID) || field.starts_with(PROTOCOL_FTP) {
        return false;
    }

    // Exclude fields that look like long random strings (likely passwords)
    // This is a heuristic - very long strings with mixed case and numbers are likely passwords
    if field.len() > LONG_PASSWORD_HEURISTIC_LENGTH && field.chars().any(|c| c.is_uppercase()) &&
       field.chars().any(|c| c.is_lowercase()) && field.chars().any(|c| c.is_numeric()) {
        return false;
    }

    true
}

pub fn is_valid_line_with_config(line: &str, email_username_only: bool) -> bool {
    if !line.chars().any(|c| c.is_ascii_graphic()) {
        return false;
    }

    if !DELIMITER_REGEX.is_match(line) {
        return false;
    }

    if email_username_only {
        if !EMAIL_REGEX.is_match(line) {
            return false;
        }
    } else {
        let fields = parse_csv_line(line);
        if fields.len() < MIN_FIELD_COUNT {
            return false;
        }

        let has_valid_username = fields.iter().any(|field| is_valid_username_field(field));
        if !has_valid_username {
            return false;
        }
    }

    true
}

fn is_url_field(field: &str) -> bool {
    if field.starts_with(PROTOCOL_HTTP) ||
       field.starts_with(PROTOCOL_HTTPS) ||
       field.starts_with(PROTOCOL_ANDROID) ||
       field.starts_with(PROTOCOL_FTP) ||
       field.starts_with(PROTOCOL_MAILTO) {
        return true;
    }

    if field.contains('.') && (field.contains('/') || field.contains('?') || field.contains('#')) {
        return true;
    }

    if field.contains('.') && field.split('.').count() >= REVERSE_DOMAIN_MIN_PARTS && !field.contains('@') && field.len() > REVERSE_DOMAIN_MIN_LENGTH {
        return true;
    }

    false
}

pub fn is_valid_line(line: &str) -> bool {
    if !line.chars().any(|c| c.is_ascii_graphic()) {
        return false;
    }

    if !DELIMITER_REGEX.is_match(line) {
        return false;
    }

    if !EMAIL_REGEX.is_match(line) {
        return false;
    }

    true
}

pub fn discover_csv_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut csv_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().to_lowercase() == CSV_EXTENSION {
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
        assert_eq!(normalize_url("https://ais.usvisa-info.com/es-co/niv/users/sign_in"), "ais.usvisa-info.com");
        assert_eq!(normalize_url("https://www.netflix.com/it/Login"), "netflix.com");
        assert_eq!(normalize_url("http://www.example.com/path?query=1#fragment"), "example.com");

        assert_eq!(
            normalize_url("android://QbSh06Ymwz0AAlEe1U4nVUM0baM8f4MrbJ7I5B9aCy8fjqjh2PiCa__sDoO5djRyh8Or2msG7cJu5koaF8eAGA==@com.smule.singandroid/"),
            "com.smule.singandroid"
        );

        assert_eq!(normalize_url("ftp://files.example.com/download"), "files.example.com");
        assert_eq!(normalize_url("mailto://user@example.com"), "user@example.com");

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

        let line = "\"user,with,commas\"@example.com,\"pass\"\"word\",http://site.com";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "user,with,commas@example.com");
        assert_eq!(fields[1], "pass\"word");
        assert_eq!(fields[2], "http://site.com");
    }

    #[test]
    fn test_field_detection() {
        let fields = vec![
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string()
        ];
        let (user_idx, password_idx, url_idx) = detect_field_positions(&fields);
        assert_eq!(user_idx, 0);
        assert_eq!(password_idx, 1);
        assert_eq!(url_idx, 2);

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
        assert!(is_valid_line("user@example.com,password123,https://example.com"));

        assert!(!is_valid_line(""));
        assert!(!is_valid_line("   "));
        assert!(!is_valid_line("not an email,password,site.com"));
        assert!(!is_valid_line("user@example.com password site.com"));
    }
}