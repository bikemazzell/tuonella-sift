use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use crate::constants::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortRecord {
    pub username: String,
    pub password: String,
    pub url: String,
    pub extra_fields: Vec<String>,
    pub normalized_username: String,
    pub normalized_url: String,
}

impl SortRecord {
    pub fn new(username: String, password: String, url: String, extra_fields: Vec<String>) -> Self {
        Self {
            normalized_username: username.to_lowercase(),
            normalized_url: normalize_url(&url),
            username,
            password,
            url,
            extra_fields,
        }
    }

    pub fn from_csv_line(line: &str) -> Option<Self> {
        // Skip lines that are clearly not valid CSV data
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.len() > MAX_FIELD_LENGTH * 10 {
            return None;
        }

        // Check for obvious binary data patterns
        if trimmed.chars().any(|c| c.is_control() && c != '\t' && c != '\r' && c != '\n') {
            return None;
        }

        let fields = parse_csv_fields(trimmed);
        if fields.len() < MIN_RECORD_FIELDS {
            return None;
        }

        let username = fields[0].trim().to_string();
        let password = fields.get(1).map(|s| s.trim().to_string()).unwrap_or_default();
        let url = fields.get(2).map(|s| s.trim().to_string()).unwrap_or_default();
        let extra_fields = fields.iter().skip(3).map(|s| s.trim().to_string()).collect();

        if username.is_empty() || username.len() > MAX_FIELD_LENGTH {
            return None;
        }

        if password.len() > MAX_FIELD_LENGTH || url.len() > MAX_FIELD_LENGTH {
            return None;
        }

        Some(Self::new(username, password, url, extra_fields))
    }

    pub fn dedup_key(&self, case_sensitive: bool) -> String {
        if case_sensitive {
            format!("{}:{}", self.username, self.password)
        } else {
            format!("{}:{}", self.normalized_username, self.password)
        }
    }

    pub fn to_csv_line(&self) -> String {
        let mut fields = vec![self.username.clone(), self.password.clone()];
        if !self.url.is_empty() {
            fields.push(self.url.clone());
        }
        for field in &self.extra_fields {
            fields.push(field.clone());
        }
        fields.join(&CSV_FIELD_SEPARATOR.to_string())
    }

    pub fn field_count(&self) -> usize {
        2 + if self.url.is_empty() { 0 } else { 1 } + self.extra_fields.len()
    }

    pub fn estimated_size(&self) -> usize {
        self.username.len() + self.password.len() + self.url.len() 
            + self.extra_fields.iter().map(|f| f.len()).sum::<usize>()
            + ESTIMATED_RECORD_SIZE_BYTES
    }
}

impl PartialEq for SortRecord {
    fn eq(&self, other: &Self) -> bool {
        self.dedup_key(false) == other.dedup_key(false)
    }
}

impl Eq for SortRecord {}

impl PartialOrd for SortRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dedup_key(false).cmp(&other.dedup_key(false))
    }
}

fn parse_csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        // Skip invalid control characters except tabs
        if ch.is_control() && ch != '\t' && ch != '\r' && ch != '\n' {
            continue;
        }

        match ch {
            CSV_QUOTE_CHAR if !in_quotes => {
                in_quotes = true;
            }
            CSV_QUOTE_CHAR if in_quotes => {
                if chars.peek() == Some(&CSV_QUOTE_CHAR) {
                    chars.next();
                    current_field.push(CSV_QUOTE_CHAR);
                } else {
                    in_quotes = false;
                }
            }
            CSV_FIELD_SEPARATOR if !in_quotes => {
                fields.push(current_field.trim().to_string());
                current_field.clear();
            }
            ':' if !in_quotes && current_field.contains('@') => {
                // Handle email:password format
                fields.push(current_field.trim().to_string());
                current_field.clear();
            }
            ' ' if !in_quotes && fields.len() >= 2 => {
                // Handle space-separated format after email and password
                fields.push(current_field.trim().to_string());
                current_field.clear();
            }
            _ => {
                current_field.push(ch);
            }
        }
    }

    if !current_field.trim().is_empty() {
        fields.push(current_field.trim().to_string());
    }

    fields
}

fn normalize_url(url: &str) -> String {
    let mut normalized = url.to_lowercase();

    if normalized.starts_with("https://") {
        normalized = normalized[8..].to_string();
    } else if normalized.starts_with("http://") {
        normalized = normalized[7..].to_string();
    } else if normalized.starts_with("android://") {
        normalized = normalized[10..].to_string();
        if let Some(at_pos) = normalized.find('@') {
            normalized = normalized[..at_pos].to_string();
        }
    } else if normalized.starts_with("ftp://") {
        normalized = normalized[6..].to_string();
    } else if normalized.starts_with("mailto:") {
        normalized = normalized[7..].to_string();
    }

    if normalized.ends_with('/') {
        normalized.pop();
    }

    normalized
}
