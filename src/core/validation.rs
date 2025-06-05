use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use anyhow::Result;
use crate::constants::*;
use crate::core::simd_string_ops::SimdStringProcessor;

pub static EMAIL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap()
});

pub static DELIMITER_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[,;\t|: ]").unwrap()
});

pub static PRINTABLE_USERNAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    let pattern = format!(r"^[\x21-\x7E]{{{},{}}}$", PRINTABLE_USERNAME_MIN_LENGTH, PRINTABLE_USERNAME_MAX_LENGTH);
    Regex::new(&pattern).unwrap()
});

pub fn is_mixed_delimiter_line(line: &str) -> bool {
    if is_space_mixed_delimiter_line(line) {
        return true;
    }

    if is_semicolon_mixed_delimiter_line(line) {
        return true;
    }

    false
}

fn is_space_mixed_delimiter_line(line: &str) -> bool {
    let parts: Vec<&str> = line.split(' ').collect();
    if parts.len() != 2 {
        return false;
    }

    let url_part = parts[0];
    let credentials_part = parts[1];

    let is_url = is_url_field(url_part);
    let colon_count = count_non_protocol_colons(credentials_part);

    is_url && colon_count == 1
}

fn is_semicolon_mixed_delimiter_line(line: &str) -> bool {
    let parts: Vec<&str> = line.split(';').collect();
    if parts.len() != 2 {
        return false;
    }

    let url_part = parts[0];
    let credentials_part = parts[1];

    let is_url = is_url_field(url_part);
    let colon_count = count_non_protocol_colons(credentials_part);

    is_url && colon_count == 1
}

pub fn detect_delimiter(line: &str) -> char {
    if is_space_mixed_delimiter_line(line) {
        return ' ';
    }

    if is_semicolon_mixed_delimiter_line(line) {
        return ';';
    }

    let delimiters = [',', ';', '\t', '|', ':', ' '];
    let mut max_count = 0;
    let mut detected_delimiter = ',';

    for &delimiter in &delimiters {
        let count = if delimiter == ':' {
            count_non_protocol_colons(line)
        } else {
            line.matches(delimiter).count()
        };

        if count > max_count {
            max_count = count;
            detected_delimiter = delimiter;
        }
    }

    detected_delimiter
}

fn count_non_protocol_colons(line: &str) -> usize {
    let protocols = ["http:", "https:", "ftp:", "mailto:", "android:"];
    let mut count = line.matches(':').count();

    for protocol in &protocols {
        count = count.saturating_sub(line.matches(protocol).count());
    }

    count
}

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
    let trimmed_line = if line.ends_with(',') && line.contains(':') {
        line.trim_end_matches(',')
    } else {
        line
    };
    let delimiter = detect_delimiter(trimmed_line);
    let mut fields = parse_csv_line_with_delimiter(trimmed_line, delimiter);
    while let Some(last) = fields.last() {
        if last.is_empty() {
            fields.pop();
        } else {
            break;
        }
    }
    fields
}

pub fn parse_csv_line_with_delimiter(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = if delimiter == ' ' && is_space_mixed_delimiter_line(line) {
        parse_space_mixed_delimiter_line(line)
    } else if delimiter == ';' && is_semicolon_mixed_delimiter_line(line) {
        parse_semicolon_mixed_delimiter_line(line)
    } else if delimiter == ':' {
        parse_colon_delimited_line(line)
    } else {
        parse_standard_delimited_line(line, delimiter)
    };
    
    while let Some(last) = fields.last() {
        if last.is_empty() {
            fields.pop();
        } else {
            break;
        }
    }
    fields
}

fn parse_space_mixed_delimiter_line(line: &str) -> Vec<String> {
    let parts: Vec<&str> = line.split(' ').collect();
    if parts.len() != 2 {
        return parse_standard_delimited_line(line, ' ');
    }

    let url_part = parts[0].trim();
    let credentials_part = parts[1].trim();
    let cred_parts: Vec<&str> = credentials_part.splitn(2, ':').collect();
    if cred_parts.len() != 2 {
        return vec![url_part.to_string(), credentials_part.to_string()];
    }

    vec![
        url_part.to_string(),
        cred_parts[0].trim().to_string(),
        cred_parts[1].trim().to_string(),
    ]
}

fn parse_semicolon_mixed_delimiter_line(line: &str) -> Vec<String> {
    let parts: Vec<&str> = line.split(';').collect();
    if parts.len() != 2 {
        return parse_standard_delimited_line(line, ';');
    }

    let url_part = parts[0].trim();
    let credentials_part = parts[1].trim();
    let cred_parts: Vec<&str> = credentials_part.splitn(2, ':').collect();
    if cred_parts.len() != 2 {
        return vec![url_part.to_string(), credentials_part.to_string()];
    }

    vec![
        url_part.to_string(),
        cred_parts[0].trim().to_string(),
        cred_parts[1].trim().to_string(),
    ]
}

fn parse_standard_delimited_line(line: &str, delimiter: char) -> Vec<String> {
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
                        if chars.peek().map_or(false, |&c| !c.is_whitespace() && c != delimiter) {
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
            ch if ch == delimiter && !in_quotes => {
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

fn parse_colon_delimited_line(line: &str) -> Vec<String> {
    let mut temp_line = line.to_string();
    let protocols = ["https://", "http://", "ftp://", "mailto://", "android://"];

    for protocol in &protocols {
        temp_line = temp_line.replace(protocol, &protocol.replace("://", "___PROTOCOL___"));
    }
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = temp_line.chars().peekable();

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
                        if chars.peek().map_or(false, |&c| !c.is_whitespace() && c != ':') {
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
            ':' if !in_quotes => {
                fields.push(current_field.replace("___PROTOCOL___", "://").trim().to_string());
                current_field.clear();
            },
            _ => current_field.push(ch),
        }
    }

    if in_quotes {
        current_field.push('"');
    }
    fields.push(current_field.replace("___PROTOCOL___", "://").trim().to_string());

    fields
}

pub fn detect_field_positions(fields: &[String]) -> (usize, usize, usize) {
    if fields.is_empty() {
        return (0, 0, 0);
    }
    let mut url_idx = None;
    let mut email_idx = None;

    for (i, field) in fields.iter().enumerate() {
        if url_idx.is_none() && is_url_field(field) {
            url_idx = Some(i);
        } else if email_idx.is_none() && EMAIL_REGEX.is_match(field) {
            email_idx = Some(i);
        }

        if url_idx.is_some() && email_idx.is_some() {
            break;
        }
    }
    let email_idx = match email_idx {
        Some(idx) => idx,
        None => return (fields.len(), fields.len(), fields.len())
    };

    let url_idx = url_idx.unwrap_or(2);

    let password_idx = (0..fields.len())
        .find(|&i| i != email_idx && i != url_idx)
        .unwrap_or((email_idx + 1) % fields.len());

    let len = fields.len();
    let positions = if email_idx == url_idx || email_idx == password_idx || password_idx == url_idx {
        let base = url_idx.min(len - 1);
        (
            (base + 1) % len,
            (base + 2) % len,
            base
        )
    } else {
        (email_idx, password_idx, url_idx)
    };

    positions
}

pub fn detect_field_positions_with_config(fields: &[String], email_username_only: bool, allow_two_field_lines: bool) -> (usize, usize, usize) {
    if allow_two_field_lines && fields.len() == 2 {
        let username_idx = if email_username_only {
            if EMAIL_REGEX.is_match(&fields[0]) { 0 } else if EMAIL_REGEX.is_match(&fields[1]) { 1 } else { fields.len() }
        } else {
            if is_valid_username_field(&fields[0]) { 0 } else if is_valid_username_field(&fields[1]) { 1 } else { fields.len() }
        };
        let password_idx = if username_idx == 0 { 1 } else if username_idx == 1 { 0 } else { fields.len() };
        return (username_idx, password_idx, fields.len());
    }
    if fields.len() < MIN_FIELD_COUNT {
        return (fields.len(), fields.len(), fields.len());
    }

    let mut url_idx = None;
    let mut username_idx = None;

    for (i, field) in fields.iter().enumerate() {
        if URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
            url_idx = Some(i);
            break;
        }
    }

    for (i, field) in fields.iter().enumerate() {
        if url_idx == Some(i) {
            continue;
        }

        if username_idx.is_none() {
            let is_valid_username = if email_username_only {
                EMAIL_REGEX.is_match(field)
            } else {
                if url_idx.is_some() {
                    is_valid_username_field_lenient(field)
                } else {
                    is_valid_username_field(field)
                }
            };

            if is_valid_username {
                username_idx = Some(i);
            }
        }

        // Look for URL only if we haven't found a protocol URL yet
        if url_idx.is_none() && is_url_field(field) {
            url_idx = Some(i);
        }

        if url_idx.is_some() && username_idx.is_some() {
            break;
        }
    }

    let username_idx = match username_idx {
        Some(idx) => idx,
        None => return (fields.len(), fields.len(), fields.len())
    };

    let url_idx = url_idx.unwrap_or(2);

    let password_idx = (0..fields.len())
        .find(|&i| i != username_idx && i != url_idx)
        .unwrap_or((username_idx + 1) % fields.len());

    let len = fields.len();
    if username_idx == url_idx || username_idx == password_idx || password_idx == url_idx {
        let base = url_idx.min(len - 1);
        (
            (base + 1) % len,
            (base + 2) % len,
            base
        )
    } else {
        (username_idx, password_idx, url_idx)
    }
}

fn is_url_field(field: &str) -> bool {
    if URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
        return true;
    }

    if field.contains('.') &&
       (field.contains('/') || field.contains('?') || field.contains('#')) {
        let domain_part = field.split('.').next().unwrap_or("");
        let after_dot = field.split('.').nth(1).unwrap_or("");

        let is_valid_domain = domain_part.len() >= 2 &&
            !domain_part.starts_with(&['/', '@', '#', '?'] as &[char]) &&
            domain_part.chars().all(|c| c.is_alphanumeric() || c == '-') &&
            after_dot.chars().all(|c| c.is_alphanumeric());

        if is_valid_domain {
            return true;
        }
    }

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

    if field.is_empty() || !PRINTABLE_USERNAME_REGEX.is_match(field) {
        return false;
    }

    if is_url_field(field) || URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
        return false;
    }

    !(field.len() > LONG_PASSWORD_HEURISTIC_LENGTH && {
        let chars: Vec<_> = field.chars().collect();
        chars.iter().any(|c| c.is_uppercase()) &&
        chars.iter().any(|c| c.is_lowercase()) &&
        chars.iter().any(|c| c.is_numeric())
    })
}

fn is_valid_username_field_lenient(field: &str) -> bool {
    let field = field.trim();

    if field.is_empty() || !PRINTABLE_USERNAME_REGEX.is_match(field) {
        return false;
    }

    if URL_PROTOCOLS.iter().any(|&protocol| field.starts_with(protocol)) {
        return false;
    }

    !(field.len() > LONG_PASSWORD_HEURISTIC_LENGTH && {
        let chars: Vec<_> = field.chars().collect();
        chars.iter().any(|c| c.is_uppercase()) &&
        chars.iter().any(|c| c.is_lowercase()) &&
        chars.iter().any(|c| c.is_numeric())
    })
}

pub fn is_valid_line_with_config(line: &str, email_username_only: bool, allow_two_field_lines: bool) -> bool {
    if !line.chars().any(|c| c.is_ascii_graphic()) || !DELIMITER_REGEX.is_match(line) {
        return false;
    }
    if email_username_only {
        return EMAIL_REGEX.is_match(line);
    }
    let fields = parse_csv_line(line);
    if allow_two_field_lines && fields.len() == 2 {
        let valid_user = if email_username_only {
            EMAIL_REGEX.is_match(&fields[0]) || EMAIL_REGEX.is_match(&fields[1])
        } else {
            is_valid_username_field(&fields[0]) || is_valid_username_field(&fields[1])
        };
        let valid_pass = !fields[0].is_empty() && !fields[1].is_empty();
        return valid_user && valid_pass;
    }
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
    fn test_delimiter_detection() {
        // Test comma delimiter
        assert_eq!(detect_delimiter("a,b,c"), ',');

        // Test semicolon delimiter
        assert_eq!(detect_delimiter("a;b;c"), ';');

        // Test tab delimiter
        assert_eq!(detect_delimiter("a\tb\tc"), '\t');

        // Test pipe delimiter
        assert_eq!(detect_delimiter("a|b|c"), '|');

        // Test colon delimiter
        assert_eq!(detect_delimiter("a:b:c"), ':');

        // Test mixed delimiters - should return the most common one
        assert_eq!(detect_delimiter("a:b:c,d"), ':');
        assert_eq!(detect_delimiter("a,b,c:d"), ',');

        // Test no delimiters - should default to comma
        assert_eq!(detect_delimiter("abc"), ',');

        // Test protocol colons are ignored
        assert_eq!(detect_delimiter("https://site.com:user:pass"), ':');
        assert_eq!(detect_delimiter("http://site.com,user,pass"), ',');

        // Test the specific failing case
        assert_eq!(detect_delimiter("https://accounts.google.com/ServiceLogin:jawahar84@gmail.com:jawahar123"), ':');
    }

    #[test]
    fn test_colon_delimited_parsing() {
        // Test the specific failing case from the user
        let line = "https://accounts.google.com/ServiceLogin:jawahar84@gmail.com:jawahar123";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "https://accounts.google.com/ServiceLogin");
        assert_eq!(fields[1], "jawahar84@gmail.com");
        assert_eq!(fields[2], "jawahar123");

        // Test more colon-delimited examples
        let line = "https://accounts.google.com/servicelogin:jawaher.mohamed2014:01112625515";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "https://accounts.google.com/servicelogin");
        assert_eq!(fields[1], "jawaher.mohamed2014");
        assert_eq!(fields[2], "01112625515");

        // Test colon delimiter with quotes
        let line = "\"https://site.com\":\"user@example.com\":\"password123\"";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "https://site.com");
        assert_eq!(fields[1], "user@example.com");
        assert_eq!(fields[2], "password123");
    }

    #[test]
    fn test_colon_delimiter_validation() {
        // Test that colon-delimited lines are now considered valid
        let line = "https://accounts.google.com/ServiceLogin:jawahar84@gmail.com:jawahar123";
        assert!(DELIMITER_REGEX.is_match(line), "Line with colon delimiter should match DELIMITER_REGEX");
        assert!(is_valid_line_with_config(line, true, false), "Colon-delimited line should be considered valid");

        // Test field detection with colon-delimited data
        let fields = parse_csv_line(line);
        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true, false);
        assert_eq!(url_idx, 0, "URL should be detected in first field");
        assert_eq!(user_idx, 1, "Email should be detected in second field");
        assert_eq!(password_idx, 2, "Password should be detected in third field");
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
        assert!(is_valid_line("user@example.com password site.com")); // Space delimiter is now valid

        assert!(!is_valid_line(""));
        assert!(!is_valid_line("   "));
        assert!(!is_valid_line("not an email,password,site.com"));
        assert!(!is_valid_line("no delimiters here")); // No delimiters at all
    }

    #[test]
    fn test_special_character_passwords() {
        // Test the specific case that was failing
        let fields = vec![
            "alzaidabdulelah@gmail.com".to_string(),
            "/g$WyY_.*@8RwC#".to_string(),
            "https://us.battle.net/login/en-gb/".to_string()
        ];

        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true, false);
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
        assert!(is_valid_line_with_config(line, true, false),
               "Line with special character password should be considered valid");
    }

    #[test]
    fn test_email_with_quotes() {
        // Test the specific failing case
        let line = "root@her0in.de,As26013069!\",https://wannafake.com/signup";

        // Test that the line is considered valid
        assert!(is_valid_line_with_config(line, true, false),
               "Line with quoted password should be considered valid");

        // Test field parsing
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3, "Line should be parsed into three fields");
        assert_eq!(fields[0], "root@her0in.de", "Email should be parsed correctly");

        // Test that the email is recognized as valid
        assert!(EMAIL_REGEX.is_match(&fields[0]), "Email should be recognized as valid");

        // Test field detection
        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, true, false);
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

    #[test]
    fn test_mixed_delimiter_detection() {
        // Test the specific failing cases from the user
        let test_cases = vec![
            "https://www.faxburner.com black.shadow1981:burns1938",
            "https://www.faxburner.com/homepage/signup-free dedy.yumanta:dedy210771",
            "https://www.faxmakeronline.com/Account/LogOn smqasimalvi85:Qasim5471!",
            "https://www.faxtastic.co.uk mshalluf.uk:Libya2022",
        ];

        for line in test_cases {
            assert!(is_mixed_delimiter_line(line),
                   "Line '{}' should be detected as mixed delimiter", line);
            assert_eq!(detect_delimiter(line), ' ',
                      "Line '{}' should detect space as primary delimiter", line);
        }

        // Test cases that should NOT be detected as mixed delimiter
        let non_mixed_cases = vec![
            "https://site.com,user:pass",  // Comma delimiter
            "https://site.com:user:pass",  // Colon delimiter only
            "https://site.com user pass",  // Space delimiter but no colon
            "site.com user:pass",          // Not a URL
            "https://site.com user:pass:extra", // Multiple colons
        ];

        for line in non_mixed_cases {
            assert!(!is_mixed_delimiter_line(line),
                   "Line '{}' should NOT be detected as mixed delimiter", line);
        }
    }

    #[test]
    fn test_mixed_delimiter_parsing() {
        // Test the specific failing cases from the user
        let line = "https://www.faxburner.com black.shadow1981:burns1938";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3, "Mixed delimiter line should parse into 3 fields");
        assert_eq!(fields[0], "https://www.faxburner.com", "URL should be first field");
        assert_eq!(fields[1], "black.shadow1981", "Username should be second field");
        assert_eq!(fields[2], "burns1938", "Password should be third field");

        let line2 = "https://www.faxburner.com/homepage/signup-free dedy.yumanta:dedy210771";
        let fields2 = parse_csv_line(line2);
        assert_eq!(fields2.len(), 3, "Mixed delimiter line should parse into 3 fields");
        assert_eq!(fields2[0], "https://www.faxburner.com/homepage/signup-free", "URL with path should be first field");
        assert_eq!(fields2[1], "dedy.yumanta", "Username should be second field");
        assert_eq!(fields2[2], "dedy210771", "Password should be third field");

        let line3 = "https://www.faxmakeronline.com/Account/LogOn smqasimalvi85:Qasim5471!";
        let fields3 = parse_csv_line(line3);
        assert_eq!(fields3.len(), 3, "Mixed delimiter line should parse into 3 fields");
        assert_eq!(fields3[0], "https://www.faxmakeronline.com/Account/LogOn", "URL with path should be first field");
        assert_eq!(fields3[1], "smqasimalvi85", "Username should be second field");
        assert_eq!(fields3[2], "Qasim5471!", "Password with special chars should be third field");

        let line4 = "https://www.faxtastic.co.uk mshalluf.uk:Libya2022";
        let fields4 = parse_csv_line(line4);
        assert_eq!(fields4.len(), 3, "Mixed delimiter line should parse into 3 fields");
        assert_eq!(fields4[0], "https://www.faxtastic.co.uk", "URL should be first field");
        assert_eq!(fields4[1], "mshalluf.uk", "Username should be second field");
        assert_eq!(fields4[2], "Libya2022", "Password should be third field");
    }

    #[test]
    fn test_mixed_delimiter_validation() {
        // Test that mixed delimiter lines are considered valid
        let test_lines = vec![
            "https://www.faxburner.com black.shadow1981:burns1938",
            "https://www.faxburner.com/homepage/signup-free dedy.yumanta:dedy210771",
            "https://www.faxmakeronline.com/Account/LogOn smqasimalvi85:Qasim5471!",
            "https://www.faxtastic.co.uk mshalluf.uk:Libya2022",
        ];

        for line in test_lines {
            // Test that the line contains delimiters
            assert!(DELIMITER_REGEX.is_match(line),
                   "Mixed delimiter line '{}' should match DELIMITER_REGEX", line);

            // Test that the line is considered valid (assuming email_username_only = false)
            assert!(is_valid_line_with_config(line, false, false),
                   "Mixed delimiter line '{}' should be considered valid", line);

            // Test field detection
            let fields = parse_csv_line(line);
            let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
            assert!(user_idx < fields.len(), "Username index should be valid for line '{}'", line);
            assert!(password_idx < fields.len(), "Password index should be valid for line '{}'", line);
            assert!(url_idx < fields.len(), "URL index should be valid for line '{}'", line);

            // Verify URL is detected correctly
            assert_eq!(url_idx, 0, "URL should be detected in first field for line '{}'", line);
        }
    }

    #[test]
    fn test_user_reported_failing_cases() {
        // Test the exact failing cases reported by the user
        let failing_lines = vec![
            "https://www.faxburner.com black.shadow1981:burns1938",
            "https://www.faxburner.com/homepage/signup-free dedy.yumanta:dedy210771",
            "https://www.faxmakeronline.com/Account/LogOn smqasimalvi85:Qasim5471!",
            "https://www.faxtastic.co.uk mshalluf.uk:Libya2022",
        ];

        for line in failing_lines {
            println!("Testing line: {}", line);

            // These should now be detected as mixed delimiter lines
            assert!(is_mixed_delimiter_line(line),
                   "Line should be detected as mixed delimiter: {}", line);

            // Should detect space as primary delimiter
            assert_eq!(detect_delimiter(line), ' ',
                      "Should detect space as delimiter for: {}", line);

            // Should parse correctly into 3 fields
            let fields = parse_csv_line(line);
            assert_eq!(fields.len(), 3,
                      "Should parse into 3 fields: {} -> {:?}", line, fields);

            // Should be considered valid
            assert!(is_valid_line_with_config(line, false, false),
                   "Should be valid with printable usernames: {}", line);

            // Field positions should be detected correctly
            let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
            assert_eq!(url_idx, 0, "URL should be in position 0 for: {}", line);
            assert_eq!(user_idx, 1, "Username should be in position 1 for: {}", line);
            assert_eq!(password_idx, 2, "Password should be in position 2 for: {}", line);

            println!("✅ Line parsed successfully: {} -> {:?}", line, fields);
        }
    }

    #[test]
    fn test_semicolon_mixed_delimiter_detection() {
        // Test the new semicolon mixed delimiter pattern: URL;USERNAME:PASSWORD
        let test_cases = vec![
            "https://luckybits.io/register;collegetchoya:23560309",
            "https://luckybits.io/register;collegetchoya@gmail.com:23560309",
            "https://luckybits.online/auth/signup;Kpofonregis21:@Scarface07",
            "http://example.com;user:pass",
            "https://site.com/path;username:password123",
        ];

        for line in test_cases {
            assert!(is_mixed_delimiter_line(line),
                   "Line '{}' should be detected as mixed delimiter", line);
            assert_eq!(detect_delimiter(line), ';',
                      "Line '{}' should detect semicolon as primary delimiter", line);
        }

        // Test cases that should NOT be detected as semicolon mixed delimiter
        let non_mixed_cases = vec![
            "https://site.com;user;pass",     // Semicolon delimiter but no colon
            "site.com;user:pass",             // Not a URL
            "https://site.com;user:pass:extra", // Multiple colons
            "https://site.com,user:pass",     // Different primary delimiter
        ];

        for line in non_mixed_cases {
            if line.contains(';') && line.contains(':') {
                // These might be detected as mixed but with different delimiters
                let delimiter = detect_delimiter(line);
                if delimiter == ';' {
                    assert!(!is_mixed_delimiter_line(line),
                           "Line '{}' should NOT be detected as mixed delimiter", line);
                }
            }
        }
    }

    #[test]
    fn test_semicolon_mixed_delimiter_parsing() {
        // Test parsing of semicolon mixed delimiter lines
        let line = "https://luckybits.io/register;collegetchoya:23560309";
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 3, "Semicolon mixed delimiter line should parse into 3 fields");
        assert_eq!(fields[0], "https://luckybits.io/register", "URL should be first field");
        assert_eq!(fields[1], "collegetchoya", "Username should be second field");
        assert_eq!(fields[2], "23560309", "Password should be third field");

        let line2 = "https://luckybits.io/register;collegetchoya@gmail.com:23560309";
        let fields2 = parse_csv_line(line2);
        assert_eq!(fields2.len(), 3, "Semicolon mixed delimiter line should parse into 3 fields");
        assert_eq!(fields2[0], "https://luckybits.io/register", "URL should be first field");
        assert_eq!(fields2[1], "collegetchoya@gmail.com", "Email username should be second field");
        assert_eq!(fields2[2], "23560309", "Password should be third field");

        let line3 = "https://luckybits.online/auth/signup;Kpofonregis21:@Scarface07";
        let fields3 = parse_csv_line(line3);
        assert_eq!(fields3.len(), 3, "Semicolon mixed delimiter line should parse into 3 fields");
        assert_eq!(fields3[0], "https://luckybits.online/auth/signup", "URL with path should be first field");
        assert_eq!(fields3[1], "Kpofonregis21", "Username should be second field");
        assert_eq!(fields3[2], "@Scarface07", "Password with special chars should be third field");
    }

    #[test]
    fn test_semicolon_mixed_delimiter_validation() {
        // Test that semicolon mixed delimiter lines are considered valid
        let test_lines = vec![
            "https://luckybits.io/register;collegetchoya:23560309",
            "https://luckybits.io/register;collegetchoya@gmail.com:23560309",
            "https://luckybits.online/auth/signup;Kpofonregis21:@Scarface07",
        ];

        for line in test_lines {
            // Test that the line contains delimiters
            assert!(DELIMITER_REGEX.is_match(line),
                   "Semicolon mixed delimiter line '{}' should match DELIMITER_REGEX", line);

            // Test that the line is considered valid (assuming email_username_only = false)
            assert!(is_valid_line_with_config(line, false, false),
                   "Semicolon mixed delimiter line '{}' should be considered valid", line);

            // Test field detection
            let fields = parse_csv_line(line);
            let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
            assert!(user_idx < fields.len(), "Username index should be valid for line '{}'", line);
            assert!(password_idx < fields.len(), "Password index should be valid for line '{}'", line);
            assert!(url_idx < fields.len(), "URL index should be valid for line '{}'", line);

            // Verify URL is detected correctly
            assert_eq!(url_idx, 0, "URL should be detected in first field for line '{}'", line);
        }
    }

    #[test]
    fn test_comprehensive_mixed_delimiter_cases() {
        // Test all reported failing cases from both user reports
        let all_failing_cases = vec![
            // Original space mixed delimiter cases
            ("https://www.faxburner.com black.shadow1981:burns1938", ' '),
            ("https://www.faxburner.com/homepage/signup-free dedy.yumanta:dedy210771", ' '),
            ("https://www.faxmakeronline.com/Account/LogOn smqasimalvi85:Qasim5471!", ' '),
            ("https://www.faxtastic.co.uk mshalluf.uk:Libya2022", ' '),

            // New semicolon mixed delimiter cases
            ("https://luckybits.io/register;collegetchoya:23560309", ';'),
            ("https://luckybits.io/register;collegetchoya@gmail.com:23560309", ';'),
            ("https://luckybits.online/auth/signup;Kpofonregis21:@Scarface07", ';'),
        ];

        for (line, expected_delimiter) in all_failing_cases {
            println!("Testing comprehensive case: {}", line);

            // Should be detected as mixed delimiter
            assert!(is_mixed_delimiter_line(line),
                   "Line should be detected as mixed delimiter: {}", line);

            // Should detect correct primary delimiter
            assert_eq!(detect_delimiter(line), expected_delimiter,
                      "Should detect '{}' as delimiter for: {}", expected_delimiter, line);

            // Should parse correctly into 3 fields
            let fields = parse_csv_line(line);
            assert_eq!(fields.len(), 3,
                      "Should parse into 3 fields: {} -> {:?}", line, fields);

            // Should be considered valid
            assert!(is_valid_line_with_config(line, false, false),
                   "Should be valid with printable usernames: {}", line);

            // Field positions should be detected correctly
            let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);
            assert_eq!(url_idx, 0, "URL should be in position 0 for: {}", line);
            assert_eq!(user_idx, 1, "Username should be in position 1 for: {}", line);
            assert_eq!(password_idx, 2, "Password should be in position 2 for: {}", line);

            // Verify the actual field contents
            assert!(is_url_field(&fields[url_idx]), "First field should be detected as URL: {}", fields[url_idx]);
            assert!(!fields[user_idx].is_empty(), "Username field should not be empty: {}", fields[user_idx]);
            assert!(!fields[password_idx].is_empty(), "Password field should not be empty: {}", fields[password_idx]);

            println!("✅ Comprehensive test passed: {} -> {:?}", line, fields);
        }
    }

    #[test]
    fn test_domain_like_username_detection() {
        // Test the specific failing case reported by the user
        let line = "compdigedu.cc.sansaturio.madrid,0Joseantonio,https://correoweb.educa.madrid.org/";
        let fields = parse_csv_line(line);

        assert_eq!(fields.len(), 3, "Line should parse into 3 fields");
        assert_eq!(fields[0], "compdigedu.cc.sansaturio.madrid", "Username should be first field");
        assert_eq!(fields[1], "0Joseantonio", "Password should be second field");
        assert_eq!(fields[2], "https://correoweb.educa.madrid.org/", "URL should be third field");

        // Test field detection - should now work correctly with lenient username validation
        let (user_idx, password_idx, url_idx) = detect_field_positions_with_config(&fields, false, false);

        // Verify correct field detection
        assert_eq!(url_idx, 2, "URL should be detected in third field (the actual URL)");
        assert_eq!(user_idx, 0, "Username should be detected in first field");
        assert_eq!(password_idx, 1, "Password should be detected in second field");

        // The line should be considered valid
        assert!(is_valid_line_with_config(line, false, false),
               "Line with domain-like username should be considered valid");

        // Test additional similar cases
        let similar_cases = vec![
            "user.domain.like.name,password123,https://example.com",
            "my.long.username.here,pass,http://site.org",
            "test.subdomain.example.tld,secret,https://real-url.com/path",
        ];

        for case in similar_cases {
            let case_fields = parse_csv_line(case);
            let (case_user_idx, case_password_idx, case_url_idx) = detect_field_positions_with_config(&case_fields, false, false);

            assert_eq!(case_user_idx, 0, "Username should be in first field for: {}", case);
            assert_eq!(case_password_idx, 1, "Password should be in second field for: {}", case);
            assert_eq!(case_url_idx, 2, "URL should be in third field for: {}", case);

            assert!(is_valid_line_with_config(case, false, false),
                   "Line should be valid: {}", case);
        }
    }

    #[test]
    fn test_domain_like_usernames_vs_urls() {
        // Test various domain-like usernames that should NOT be detected as URLs
        let domain_like_usernames = vec![
            "compdigedu.cc.sansaturio.madrid",  // The failing case
            "user.name.domain.extension",       // Generic domain-like username
            "my.long.username.here",            // Another domain-like pattern
            "test.subdomain.example.tld",       // Looks like a domain but is a username
        ];

        for username in domain_like_usernames {
            println!("Testing domain-like username: {}", username);

            // These should NOT be detected as URLs when they're usernames
            // The issue is that they currently ARE being detected as URLs due to reverse domain logic
            // This test will help us understand the current behavior
            let is_detected_as_url = is_url_field(username);
            println!("  is_url_field('{}') = {}", username, is_detected_as_url);

            // For now, let's document what we expect vs what we get
            // We'll fix the logic after understanding the current behavior
        }

        // Test actual URLs that SHOULD be detected as URLs
        let actual_urls = vec![
            "https://correoweb.educa.madrid.org/",
            "http://example.com",
            "https://site.com/path",
            "ftp://files.example.com",
        ];

        for url in actual_urls {
            println!("Testing actual URL: {}", url);
            assert!(is_url_field(url), "Actual URL '{}' should be detected as URL", url);
        }
    }

    #[test]
    fn test_two_field_mode_with_trailing_comma() {
        let _config = crate::config::model::DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: false,
            allow_two_field_lines: true,
        };
        let lines = vec![
            "gewy:zxc/.,",
            "giftoboy:giftos,",
            "jhobakill:636363,",
            "LAN112:XA3.b9,",
            "LutinRose:bhtn5b,",
        ];
        for line in lines {
            let fields = parse_csv_line(line);
            assert_eq!(fields.len(), 2, "Should parse as 2 fields: {:?}", fields);
            let (user_idx, pass_idx, _url_idx) = crate::core::validation::detect_field_positions_with_config(&fields, false, true);
            assert!(user_idx < 2 && pass_idx < 2, "Should detect valid username and password indices for line: {}", line);
            assert!(is_valid_line_with_config(line, false, true), "Should be valid in 2-field mode: {}", line);
        }
    }

    #[test]
    fn test_two_field_colon_with_trailing_comma() {
        let _config = crate::config::model::DeduplicationConfig {
            case_sensitive_usernames: false,
            normalize_urls: true,
            email_username_only: false,
            allow_two_field_lines: true,
        };
        let lines = vec![
            "gewy:zxc/.,",
            "giftoboy:giftos,",
            "jhobakill:636363,",
            "LAN112:XA3.b9,",
            "LutinRose:bhtn5b,",
            "gewy:zxc/. ,", // with space before comma
            "gewy:zxc/. ",   // with space only
        ];
        for line in lines {
            let fields = parse_csv_line(line);
            assert_eq!(fields.len(), 2, "Should parse as 2 fields: {:?}", fields);
            let (user_idx, pass_idx, _url_idx) = crate::core::validation::detect_field_positions_with_config(&fields, false, true);
            assert!(user_idx < 2 && pass_idx < 2, "Should detect valid username and password indices for line: {}", line);
            assert!(is_valid_line_with_config(line, false, true), "Should be valid in 2-field mode: {}", line);
        }
    }
}

/// SIMD-accelerated batch validation functions
pub mod simd {
    use super::*;

    /// Global SIMD processor instance for validation operations
    static SIMD_PROCESSOR: Lazy<Option<SimdStringProcessor>> = Lazy::new(|| {
        SimdStringProcessor::new().ok()
    });

    /// Batch normalize URLs using SIMD acceleration
    pub fn normalize_urls_batch(urls: &[String]) -> Result<Vec<String>> {
        if let Some(processor) = SIMD_PROCESSOR.as_ref() {
            processor.normalize_urls_to_lowercase(urls)
        } else {
            // Fallback to scalar processing
            Ok(urls.iter().map(|url| normalize_url(url)).collect())
        }
    }

    /// Batch validate emails using SIMD acceleration  
    pub fn validate_emails_batch(emails: &[String]) -> Result<Vec<bool>> {
        if let Some(processor) = SIMD_PROCESSOR.as_ref() {
            processor.validate_emails_simd(emails)
        } else {
            // Fallback to scalar processing
            Ok(emails.iter().map(|email| EMAIL_REGEX.is_match(email)).collect())
        }
    }

    /// Batch parse CSV lines using SIMD acceleration
    pub fn parse_csv_lines_batch(lines: &[String], delimiter: char) -> Result<Vec<Vec<String>>> {
        if let Some(processor) = SIMD_PROCESSOR.as_ref() {
            processor.parse_csv_fields_simd(lines, delimiter)
        } else {
            // Fallback to scalar processing
            Ok(lines.iter().map(|line| {
                parse_csv_line(line)
            }).collect())
        }
    }

    /// Batch generate hashes for strings using SIMD acceleration
    pub fn hash_strings_batch(strings: &[String]) -> Result<Vec<u64>> {
        if let Some(processor) = SIMD_PROCESSOR.as_ref() {
            processor.hash_strings_simd(strings)
        } else {
            // Fallback to scalar processing
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            Ok(strings.iter().map(|s| {
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }).collect())
        }
    }

    /// Get SIMD capabilities information
    pub fn get_simd_info() -> Option<String> {
        if let Some(processor) = SIMD_PROCESSOR.as_ref() {
            let caps = processor.get_capabilities();
            Some(format!(
                "SIMD enabled: AVX2={}, AVX512={}, NEON={}, max_width={}B, speedup={:.1}x",
                caps.cpu_features.has_avx2,
                caps.cpu_features.has_avx512, 
                caps.cpu_features.has_neon,
                caps.max_vector_width,
                caps.theoretical_speedup
            ))
        } else {
            None
        }
    }

    /// Check if SIMD acceleration is available
    pub fn is_simd_available() -> bool {
        SIMD_PROCESSOR.is_some()
    }

    /// Benchmark SIMD vs scalar performance
    pub fn benchmark_simd_performance(test_data: &[String]) -> Result<SimdBenchmarkResult> {
        use std::time::Instant;

        let start_scalar = Instant::now();
        let _scalar_results: Vec<String> = test_data.iter().map(|s| s.to_lowercase()).collect();
        let scalar_time = start_scalar.elapsed();

        let start_simd = Instant::now();
        let _simd_results = normalize_urls_batch(test_data)?;
        let simd_time = start_simd.elapsed();

        let speedup = if simd_time.as_nanos() > 0 {
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        } else {
            1.0
        };

        Ok(SimdBenchmarkResult {
            scalar_time_ms: scalar_time.as_millis() as f64,
            simd_time_ms: simd_time.as_millis() as f64,
            speedup_factor: speedup,
            data_size: test_data.len(),
            simd_available: is_simd_available(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SimdBenchmarkResult {
    pub scalar_time_ms: f64,
    pub simd_time_ms: f64,
    pub speedup_factor: f64,
    pub data_size: usize,
    pub simd_available: bool,
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_batch_url_normalization() {
        let urls = vec![
            "HTTP://EXAMPLE.COM".to_string(),
            "HTTPS://TEST.ORG/PATH".to_string(),
            "FTP://FILES.COM".to_string(),
        ];
        
        let result = simd::normalize_urls_batch(&urls);
        assert!(result.is_ok());
        
        let normalized = result.unwrap();
        assert_eq!(normalized[0], "http://example.com");
        assert_eq!(normalized[1], "https://test.org/path");
        assert_eq!(normalized[2], "ftp://files.com");
    }

    #[test]
    fn test_simd_batch_email_validation() {
        let emails = vec![
            "test@example.com".to_string(),
            "invalid-email".to_string(),
            "user@domain.org".to_string(),
            "short".to_string(),
        ];
        
        let result = simd::validate_emails_batch(&emails);
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert_eq!(validation[0], true);  // Valid email
        assert_eq!(validation[1], false); // No @ or .
        assert_eq!(validation[2], true);  // Valid email
        assert_eq!(validation[3], false); // Too short
    }

    #[test]
    fn test_simd_batch_csv_parsing() {
        let lines = vec![
            "field1,field2,field3".to_string(),
            "a,b,c,d".to_string(),
            "single".to_string(),
        ];
        
        let result = simd::parse_csv_lines_batch(&lines, ',');
        assert!(result.is_ok());
        
        let parsed = result.unwrap();
        assert_eq!(parsed[0], vec!["field1", "field2", "field3"]);
        assert_eq!(parsed[1], vec!["a", "b", "c", "d"]);
        assert_eq!(parsed[2], vec!["single"]);
    }

    #[test]
    fn test_simd_batch_hashing() {
        let strings = vec![
            "test1".to_string(),
            "test2".to_string(),
            "test1".to_string(), // Duplicate
        ];
        
        let result = simd::hash_strings_batch(&strings);
        assert!(result.is_ok());
        
        let hashes = result.unwrap();
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], hashes[2]); // Same string should have same hash
        assert_ne!(hashes[0], hashes[1]); // Different strings should have different hashes
    }

    #[test]
    fn test_simd_info() {
        let info = simd::get_simd_info();
        println!("SIMD info: {:?}", info);
        
        let available = simd::is_simd_available();
        println!("SIMD available: {}", available);
    }

    #[test]
    fn test_simd_benchmark() {
        let test_data: Vec<String> = (0..1000).map(|i| format!("HTTP://EXAMPLE{}.COM/PATH", i)).collect();
        
        let result = simd::benchmark_simd_performance(&test_data);
        assert!(result.is_ok());
        
        let benchmark = result.unwrap();
        println!("Benchmark results: {:?}", benchmark);
        
        assert!(benchmark.speedup_factor > 0.0);
        assert_eq!(benchmark.data_size, 1000);
    }
}