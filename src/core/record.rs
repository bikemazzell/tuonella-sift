use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub user: String,
    pub password: String,
    pub url: String,
    pub normalized_user: String,
    pub normalized_url: String,
    pub completeness_score: f32,
}

impl Record {
    pub fn new(user: String, password: String, url: String, case_sensitive: bool) -> Option<Self> {
        if user.trim().is_empty() || password.trim().is_empty() {
            return None;
        }

        let normalized_user = if case_sensitive {
            user.clone()
        } else {
            user.to_lowercase()
        };

        let normalized_url = super::validation::normalize_url(&url);
        let completeness_score = calculate_completeness(&user, &password, &url);

        Some(Record {
            user,
            password,
            url,
            normalized_user,
            normalized_url,
            completeness_score,
        })
    }

    pub fn dedup_key(&self) -> String {
        // Use the email field as the key for deduplication
        self.normalized_user.clone()
    }

    pub fn is_more_complete_than(&self, other: &Record) -> bool {
        self.completeness_score > other.completeness_score
    }

    pub fn compare_completeness(&self, other: &Record) -> Ordering {
        self.completeness_score.partial_cmp(&other.completeness_score).unwrap_or(Ordering::Equal)
    }
}

/// Calculates a completeness score for a record based on its fields
///
/// The score increases with:
/// - The presence of non-empty fields
/// - The length of each field
pub fn calculate_completeness(user: &str, password: &str, url: &str) -> f32 {
    let mut score = 0.0;

    if !user.trim().is_empty() {
        score += 1.0 + (user.len() as f32 * 0.01);
    }

    if !password.trim().is_empty() {
        score += 1.0 + (password.len() as f32 * 0.01);
    }

    if !url.trim().is_empty() {
        score += 1.0 + (url.len() as f32 * 0.01);
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_creation() {
        // Valid record
        let record = Record::new(
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            false,
        );
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.user, "user@example.com");
        assert_eq!(record.password, "password123");
        assert_eq!(record.normalized_user, "user@example.com");

        // Invalid record - empty user
        let record = Record::new(
            "".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            false,
        );
        assert!(record.is_none());

        // Invalid record - empty password
        let record = Record::new(
            "user@example.com".to_string(),
            "".to_string(),
            "https://example.com".to_string(),
            false,
        );
        assert!(record.is_none());
    }

    #[test]
    fn test_completeness_calculation() {
        let score1 = calculate_completeness("user@example.com", "pass", "https://example.com");
        let score2 = calculate_completeness("user@example.com", "pass", "");
        let score3 = calculate_completeness("user@example.com", "password123", "https://example.com");

        assert!(score1 > score2, "Record with URL should have higher score");
        assert!(score3 > score1, "Record with longer password should have higher score");
    }

    #[test]
    fn test_record_comparison() {
        let record1 = Record {
            user: "user1@example.com".to_string(),
            password: "password".to_string(),
            url: "https://example.com".to_string(),
            normalized_user: "user1@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            completeness_score: 3.5,
        };

        let record2 = Record {
            user: "user1@example.com".to_string(),
            password: "password123".to_string(),
            url: "https://example.com/login".to_string(),
            normalized_user: "user1@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            completeness_score: 4.2,
        };

        assert!(record2.is_more_complete_than(&record1), 
                "Record with higher completeness score should be more complete");
        assert_eq!(record2.compare_completeness(&record1), Ordering::Greater);
        assert_eq!(record1.dedup_key(), record2.dedup_key(), 
                   "Records with same normalized email should have same dedup key");
    }
} 