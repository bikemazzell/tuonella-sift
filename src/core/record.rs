use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use crate::constants::{
    CORE_FIELD_COUNT, USERNAME_FIELD_INDEX, PASSWORD_FIELD_INDEX, URL_FIELD_INDEX,
    BASE_FIELD_SCORE, FIELD_LENGTH_WEIGHT
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub user: String,
    pub password: String,
    pub url: String,
    pub normalized_user: String,
    pub normalized_url: String,
    pub completeness_score: f32,
    pub field_count: usize,
    pub all_fields: Vec<String>,
}

impl Record {
    pub fn new(user: String, password: String, url: String, case_sensitive: bool) -> Option<Self> {
        let all_fields = vec![user.clone(), password.clone(), url.clone()];
        Self::from_fields_with_config(all_fields, case_sensitive)
    }

    pub fn from_fields_with_config(fields: Vec<String>, case_sensitive: bool) -> Option<Self> {
        if fields.len() < CORE_FIELD_COUNT {
            return None;
        }

        let user = fields[USERNAME_FIELD_INDEX].clone();
        let password = fields[PASSWORD_FIELD_INDEX].clone();
        let url = fields[URL_FIELD_INDEX].clone();

        if user.is_empty() || password.is_empty() || url.is_empty() {
            return None;
        }

        let normalized_url = super::validation::normalize_url(&url);
        let completeness_score = calculate_completeness_score(&user, &password, &normalized_url);
        let normalized_user = if case_sensitive { user.clone() } else { user.to_lowercase() };

        Some(Record {
            user,
            password,
            url: normalized_url,
            normalized_user,
            normalized_url: String::new(),
            field_count: fields.len(),
            all_fields: fields,
            completeness_score,
        })
    }

    pub fn new_with_field_count(user: String, password: String, url: String, case_sensitive: bool, field_count: usize) -> Option<Self> {
        let all_fields = vec![user.clone(), password.clone(), url.clone()];
        Self::new_from_fields_with_count(all_fields, 0, 1, 2, case_sensitive, field_count)
    }

    pub fn new_from_fields(
        all_fields: Vec<String>,
        user_idx: usize,
        password_idx: usize,
        url_idx: usize,
        case_sensitive: bool
    ) -> Option<Self> {
        if user_idx >= all_fields.len() || password_idx >= all_fields.len() || url_idx >= all_fields.len() {
            return None;
        }

        let user = all_fields[user_idx].clone();
        let password = all_fields[password_idx].clone();
        let url = all_fields[url_idx].clone();

        if user.trim().is_empty() || password.trim().is_empty() {
            return None;
        }

        let normalized_user = if case_sensitive {
            user.clone()
        } else {
            user.to_lowercase()
        };

        let normalized_url = super::validation::normalize_url(&url);
        let completeness_score = calculate_completeness_score(&user, &password, &normalized_url);
        let field_count = all_fields.len();

        Some(Record {
            user,
            password,
            url,
            normalized_user,
            normalized_url,
            completeness_score,
            field_count,
            all_fields,
        })
    }

    pub fn new_from_fields_with_count(
        all_fields: Vec<String>,
        user_idx: usize,
        password_idx: usize,
        url_idx: usize,
        case_sensitive: bool,
        field_count: usize
    ) -> Option<Self> {
        if user_idx >= all_fields.len() || password_idx >= all_fields.len() || url_idx >= all_fields.len() {
            return None;
        }

        let user = all_fields[user_idx].clone();
        let password = all_fields[password_idx].clone();
        let url = all_fields[url_idx].clone();

        if user.trim().is_empty() || password.trim().is_empty() {
            return None;
        }

        let normalized_user = if case_sensitive {
            user.clone()
        } else {
            user.to_lowercase()
        };

        let normalized_url = super::validation::normalize_url(&url);
        let completeness_score = calculate_completeness_score(&user, &password, &normalized_url);

        Some(Record {
            user,
            password,
            url,
            normalized_user,
            normalized_url,
            completeness_score,
            field_count,
            all_fields,
        })
    }

    pub fn dedup_key(&self) -> String {
        format!("{}|{}|{}",
                self.normalized_user,
                self.password,
                self.normalized_url)
    }

    pub fn is_more_complete_than(&self, other: &Record) -> bool {
        self.completeness_score > other.completeness_score
    }

    pub fn compare_completeness(&self, other: &Record) -> Ordering {
        self.completeness_score.partial_cmp(&other.completeness_score).unwrap_or(Ordering::Equal)
    }
}

fn calculate_completeness_score(user: &str, password: &str, url: &str) -> f32 {
    let mut score: f32 = 0.0;

    // Score for username
    if !user.is_empty() {
        score += BASE_FIELD_SCORE as f32 + (user.len() as f32 * FIELD_LENGTH_WEIGHT as f32);
    }

    // Score for password
    if !password.is_empty() {
        score += BASE_FIELD_SCORE as f32 + (password.len() as f32 * FIELD_LENGTH_WEIGHT as f32);
    }

    // Score for URL
    if !url.is_empty() {
        score += BASE_FIELD_SCORE as f32 + (url.len() as f32 * FIELD_LENGTH_WEIGHT as f32);
    }

    score
}



pub fn calculate_completeness_with_all_fields(all_fields: &[String]) -> f32 {
    let mut score = 0.0;

    for field in all_fields {
        if !field.trim().is_empty() {
            score += 1.0 + (field.len() as f32 * 0.01);
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_creation() {
        let record = Record::new(
            "user@example.com".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            true,
        );
        assert!(record.is_some());

        let record = Record::new(
            "".to_string(),
            "password123".to_string(),
            "https://example.com".to_string(),
            true,
        );
        assert!(record.is_none());

        let record = Record::new(
            "user@example.com".to_string(),
            "".to_string(),
            "https://example.com".to_string(),
            true,
        );
        assert!(record.is_none());
    }

    #[test]
    fn test_completeness_score() {
        let user = "test@example.com";
        let password = "password123";
        let url = "https://example.com";

        let score = calculate_completeness_score(user, password, url);
        assert!(score > 0.0);

        let empty_score = calculate_completeness_score("", "", "");
        assert_eq!(empty_score, 0.0);
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
            field_count: 3,
            all_fields: vec!["user1@example.com".to_string(), "password".to_string(), "https://example.com".to_string()],
        };

        let record2 = Record {
            user: "user1@example.com".to_string(),
            password: "password123".to_string(),
            url: "https://example.com/login".to_string(),
            normalized_user: "user1@example.com".to_string(),
            normalized_url: "example.com".to_string(),
            completeness_score: 4.2,
            field_count: 3,
            all_fields: vec!["user1@example.com".to_string(), "password123".to_string(), "https://example.com/login".to_string()],
        };

        assert!(record2.is_more_complete_than(&record1),
                "Record with higher completeness score should be more complete");
        assert_eq!(record2.compare_completeness(&record1), Ordering::Greater);

        assert_ne!(record1.dedup_key(), record2.dedup_key(),
                   "Records with different passwords should have different dedup keys");
    }

    #[test]
    fn test_composite_dedup_key() {
        let record1 = Record {
            user: "user@email.com".to_string(),
            password: "123".to_string(),
            url: "first.com".to_string(),
            normalized_user: "user@email.com".to_string(),
            normalized_url: "first.com".to_string(),
            completeness_score: 3.0,
            field_count: 3,
            all_fields: vec!["user@email.com".to_string(), "123".to_string(), "first.com".to_string()],
        };

        let record2 = Record {
            user: "user@email.com".to_string(),
            password: "123".to_string(),
            url: "second.com".to_string(),
            normalized_user: "user@email.com".to_string(),
            normalized_url: "second.com".to_string(),
            completeness_score: 3.0,
            field_count: 3,
            all_fields: vec!["user@email.com".to_string(), "123".to_string(), "second.com".to_string()],
        };

        assert_ne!(record1.dedup_key(), record2.dedup_key(),
                   "Records with same email but different URLs should have different keys");

        let record3 = Record {
            user: "user@email.com".to_string(),
            password: "444".to_string(),
            url: "first.com".to_string(),
            normalized_user: "user@email.com".to_string(),
            normalized_url: "first.com".to_string(),
            completeness_score: 3.0,
            field_count: 3,
            all_fields: vec!["user@email.com".to_string(), "444".to_string(), "first.com".to_string()],
        };

        assert_ne!(record1.dedup_key(), record3.dedup_key(),
                   "Records with same email but different passwords should have different keys");

        let record4 = Record {
            user: "user@email.com".to_string(),
            password: "444".to_string(),
            url: "first.com".to_string(),
            normalized_user: "user@email.com".to_string(),
            normalized_url: "first.com".to_string(),
            completeness_score: 4.0,
            field_count: 4,
            all_fields: vec!["user@email.com".to_string(), "444".to_string(), "first.com".to_string(), "extra".to_string()],
        };

        assert_eq!(record3.dedup_key(), record4.dedup_key(),
                   "Records with same core fields should have same key (completeness decides winner)");
        assert!(record4.is_more_complete_than(&record3),
                "Record with more fields should be more complete");

        let record5 = Record {
            user: "user@email.com".to_string(),
            password: "444".to_string(),
            url: "first.com".to_string(),
            normalized_user: "user@email.com".to_string(),
            normalized_url: "first.com".to_string(),
            completeness_score: 3.0,
            field_count: 3,
            all_fields: vec!["user@email.com".to_string(), "444".to_string(), "first.com".to_string()],
        };

        assert_eq!(record3.dedup_key(), record5.dedup_key(),
                   "Exact duplicate records should have same dedup key");
    }
}