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

// Common URL protocols
const URL_PROTOCOLS: &[&str] = &[
    PROTOCOL_HTTP,
    PROTOCOL_HTTPS,
    PROTOCOL_ANDROID,
    PROTOCOL_FTP,
    PROTOCOL_MAILTO,
];

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
                match (in_quotes, chars.peek()) {
                    (false, _) if current_field.is_empty() => in_quotes = true,
                    (true, Some('"')) => {
                        current_field.push('"');
                        chars.next();
                    },
                    (true, _) => {
                        in_quotes = false;
                        if chars.peek().map_or(false, |&c| !c.is_whitespace() && c != ',') {
                            continue;
                        }
                    },
                    (false, Some('"')) => {
                        current_field.push('"');
                        chars.next();
                    },
                    _ => current_field.push('"'),
                }
            },
            ',' if !in_quotes => {
                fields.push(current_field.trim().to_string());
                current_field.clear();
            },
            _ => current_field.push(ch),
        }
    }

    if in_quotes {
        current_field.push('"');
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
    // Initialize default positions
    let mut positions = (0, 1, 2);
    
    // Early return if we don't have enough fields
    if fields.len() < MIN_FIELD_COUNT {
        return (fields.len(), fields.len(), fields.len());
    }

    // First find URL field as it's the most distinctive
    positions.2 = fields.iter()
        .position(|field| is_url_field(field))
        .unwrap_or(2);

    // Find username field, excluding the URL field
    let username_pos = fields.iter()
        .enumerate()
        .find(|(i, field)| {
            *i != positions.2 && (
                if email_username_only {
                    EMAIL_REGEX.is_match(field)
                } else {
                    is_valid_username_field(field)
                }
            )
        })
        .map(|(i, _)| i);

    // If no valid username found, return invalid positions
    if username_pos.is_none() {
        return (fields.len(), fields.len(), fields.len());
    }
    positions.0 = username_pos.unwrap();

    // Find password field - first available field that's not URL or username
    positions.1 = fields.iter()
        .enumerate()
        .find(|(i, _)| *i != positions.0 && *i != positions.2)
        .map(|(i, _)| i)
        .unwrap_or((positions.0 + 1) % fields.len());

    // Ensure no field position collisions
    if positions.0 == positions.1 || positions.0 == positions.2 || positions.1 == positions.2 {
        positions = (
            positions.2.saturating_add(1) % fields.len(),
            positions.2.saturating_add(2) % fields.len(),
            positions.2
        );
    }

    positions
}

fn is_url_field(field: &str) -> bool {
    // Check for standard URL protocols
    if URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
        return true;
    }

    // Check for domain-like structure with path/query/fragment
    if field.contains('.') && 
       (field.contains('/') || field.contains('?') || field.contains('#')) {
        let domain_part = field.split('.').next().unwrap_or("");
        let after_dot = field.split('.').nth(1).unwrap_or("");

        // Validate domain-like structure
        let is_valid_domain = domain_part.len() >= 2 && 
            !domain_part.starts_with(&['/', '@', '#', '?'] as &[char]) &&
            domain_part.chars().all(|c| c.is_alphanumeric() || c == '-') &&
            after_dot.chars().all(|c| c.is_alphanumeric());

        if is_valid_domain {
            return true;
        }
    }

    // Check for reverse domain notation
    if field.contains('.') && 
       field.split('.').count() >= REVERSE_DOMAIN_MIN_PARTS && 
       !field.contains('@') && 
       field.len() > REVERSE_DOMAIN_MIN_LENGTH {
        return field.split('.')
            .all(|part| part.len() >= 2 && 
                !part.starts_with('-') && 
                !part.ends_with('-') && 
                part.chars().all(|c| c.is_alphanumeric() || c == '-'));
    }

    false
}

fn is_valid_username_field(field: &str) -> bool {
    let field = field.trim();
    
    // Basic validation - must be non-empty and match printable pattern
    if field.is_empty() || !PRINTABLE_USERNAME_REGEX.is_match(field) {
        return false;
    }

    // Must not be a URL or URL-like pattern
    if is_url_field(field) || URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
        return false;
    }

    // Check if it looks like a password (long string with mixed case and numbers)
    !(field.len() > LONG_PASSWORD_HEURISTIC_LENGTH && {
        let chars: Vec<_> = field.chars().collect();
        chars.iter().any(|c| c.is_uppercase()) &&
        chars.iter().any(|c| c.is_lowercase()) &&
        chars.iter().any(|c| c.is_numeric())
    })
}

pub fn is_valid_line_with_config(line: &str, email_username_only: bool) -> bool {
    // Basic line validation
    if !line.chars().any(|c| c.is_ascii_graphic()) || !DELIMITER_REGEX.is_match(line) {
        return false;
    }

    // Email-only mode validation
    if email_username_only {
        return EMAIL_REGEX.is_match(line);
    }

    // General validation
    let fields = parse_csv_line(line);
    fields.len() >= MIN_FIELD_COUNT && fields.iter().any(|field| is_valid_username_field(field))
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

    #[test]
    fn test_special_character_passwords() {
        // Test the specific case that was failing
        let fields = vec![
            "alzaidabdulelah@gmail.com".to_string(),
            "/g$WyY_.*@8RwC#".to_string(),
            "https://us.battle.net/login/en-gb/".to_string()
        ];

        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true);
        assert_eq!(user_idx, 0, "Email should be detected as username");
        assert_eq!(password_idx, 1, "Special character string should be detected as password");
        assert_eq!(url_idx, 2, "URL should be detected as URL field");

        // Test that the password field is not mistakenly identified as a URL
        assert!(!is_url_field(&fields[password_idx]), "Password with special chars should not be detected as URL");
        
        // Test that the URL field is correctly identified
        assert!(is_url_field(&fields[url_idx]), "URL should be detected as URL");

        // Additional test cases for passwords with URL-like patterns
        let test_passwords = vec![
            "pass@word123",
            "http.pass",
            "my.password/123",
            "pass/word@123.com",
            "user.name@domain",
            "/path/like/password",
            "password#with@special.chars"
        ];

        for password in test_passwords {
            let message = format!("Password '{}' should not be detected as URL", password);
            assert!(!is_url_field(password), "{}", message);
        }

        // Test complete line parsing
        let line = "alzaidabdulelah@gmail.com,/g$WyY_.*@8RwC#,https://us.battle.net/login/en-gb/";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3, "Line should be parsed into three fields");
        
        // Verify the line is considered valid
        assert!(is_valid_line_with_config(line, true), 
               "Line with special character password should be considered valid");
    }

    #[test]
    fn test_email_with_quotes() {
        // Test the specific failing case
        let line = "root@her0in.de,As26013069!\",https://wannafake.com/signup";
        
        // Test that the line is considered valid
        assert!(is_valid_line_with_config(line, true), 
               "Line with quoted password should be considered valid");

        // Test field parsing
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3, "Line should be parsed into three fields");
        assert_eq!(fields[0], "root@her0in.de", "Email should be parsed correctly");
        
        // Test that the email is recognized as valid
        assert!(EMAIL_REGEX.is_match(&fields[0]), "Email should be recognized as valid");

        // Test field detection
        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true);
        assert_eq!(user_idx, 0, "Email should be detected as username");
        assert_eq!(password_idx, 1, "Password field should be detected");
        assert_eq!(url_idx, 2, "URL field should be detected");

        // Test with other similar email patterns
        let test_emails = vec![
            "user@domain.com",
            "root@her0in.de",
            "user@sub.domain.co.uk",
            "user.name@domain.com",
            "user+tag@domain.com",
            "user123@domain.com",
            "user@domain123.com"
        ];

        for email in test_emails {
            let message = format!("Email '{}' should be recognized as valid", email);
            assert!(EMAIL_REGEX.is_match(email), "{}", message);
        }
    }
}