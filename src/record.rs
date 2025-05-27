
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use unicode_normalization::UnicodeNormalization;
use url::Url;

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

        if user.is_empty() || url.is_empty() {
            return None;
        }

        let normalized_url = normalize_url(&url);
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

    /// Create a record without normalization (for CUDA processing)
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

        if user.is_empty() || url.is_empty() {
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
    }

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

pub struct FieldDetector {
    url_patterns: Vec<Regex>,
    email_patterns: Vec<Regex>,
    password_patterns: Vec<Regex>,
}

impl FieldDetector {
    pub fn new() -> Self {
        let url_patterns = vec![
            Regex::new(r"^https?://").unwrap(),
            Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}").unwrap(),
            Regex::new(r"\.(com|org|net|edu|gov|mil|int|co\.|ac\.|org\.)").unwrap(),
        ];

        let email_patterns = vec![
            Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap(),
        ];

        let password_patterns = vec![
            Regex::new(r"^[a-zA-Z0-9!@#$%^&*()_+=\-\[\]{};':,.<>/?]{4,}$").unwrap(),
        ];

        Self {
            url_patterns,
            email_patterns,
            password_patterns,
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

                for pattern in &self.url_patterns {
                    if pattern.is_match(field) {
                        url_scores[i] += 1;
                        break;
                    }
                }

                for pattern in &self.email_patterns {
                    if pattern.is_match(field) {
                        email_scores[i] += 1;
                        break;
                    }
                }

                for pattern in &self.password_patterns {
                    if pattern.is_match(field) {
                        password_scores[i] += 1;
                        break;
                    }
                }
            }
        }

        let url_idx = url_scores
            .iter()
            .enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(2);

        let user_idx = email_scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != url_idx)
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let password_idx = password_scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != url_idx && *idx != user_idx)
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(1);

        (user_idx, password_idx, url_idx)
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
    if url_str.is_empty() {
        return String::new();
    }

    let mut url_str = url_str.split(&[',', ';', ' ', '\t'][..]).next()
        .unwrap_or(url_str)
        .trim()
        .to_string();

    if !url_str.starts_with("http://") && !url_str.starts_with("https://") {
        url_str = format!("http://{}", url_str);
    }

    if let Ok(parsed_url) = Url::parse(&url_str) {
        let mut host = parsed_url.host_str().unwrap_or("").to_lowercase();
        
        if host.starts_with("www.") {
            host = host[4..].to_string();
        }
        if host.starts_with("m.") {
            host = host[2..].to_string();
        }
        if host.starts_with("mobile.") {
            host = host[7..].to_string();
        }

        let path = parsed_url.path().trim_end_matches('/');
        
        format!("{}{}", host, path)
    } else {
        url_str.to_lowercase().replace("http://", "").replace("https://", "")
    }
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
        assert_eq!(normalize_url("https://www.facebook.com/user123/"), "facebook.com/user123");
        assert_eq!(normalize_url("http://m.facebook.com/user123"), "facebook.com/user123");
        assert_eq!(normalize_url("facebook.com/user123?ref=123"), "facebook.com/user123");
        assert_eq!(normalize_url("FACEBOOK.COM/User123"), "facebook.com/User123");
    }

    #[test]
    fn test_field_detection() {
        let detector = FieldDetector::new();
        let samples = vec![
            vec!["user@example.com".to_string(), "password123".to_string(), "https://example.com".to_string()],
            vec!["test@test.com".to_string(), "secret".to_string(), "google.com".to_string()],
        ];
        
        let (user_idx, password_idx, url_idx) = detector.detect_fields(&samples);
        assert_eq!(user_idx, 0);
        assert_eq!(password_idx, 1);
        assert_eq!(url_idx, 2);
    }
} 