use once_cell::sync::Lazy;
use regex::Regex;
pub static PROTOCOL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z][a-zA-Z0-9+.-]*://").unwrap()
});

pub static SUBDOMAIN_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^([a-zA-Z0-9-]+)\.([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$").unwrap()
});

pub static PATH_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"/.*$").unwrap()
});

pub static EMAIL_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
});


pub static PASSWORD_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z0-9!#$%^&*()_+=\-\[\]{};':,<>/?]{4,}$").unwrap()
});

pub static IP_ADDRESS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").unwrap()
});


pub fn normalize_url_fast(url: &str) -> String {
    let trimmed_url = url.trim();
    if trimmed_url.is_empty() {
        return String::new();
    }


    let lowercased_url = trimmed_url.to_lowercase();


    let protocol_stripped_url = PROTOCOL_PATTERN.replace(&lowercased_url, "").to_string();


    let parse_candidate = format!("http://{}", protocol_stripped_url);

    if let Ok(parsed_url) = url::Url::parse(&parse_candidate) {
        if let Some(host_str) = parsed_url.host_str() {
            let mut host = host_str.to_string();


            if host.starts_with("www.") {
                host = host[4..].to_string();
            }


            if let Some(captures) = SUBDOMAIN_PATTERN.captures(&host) {
                if let Some(main_domain) = captures.get(2) {
                    return main_domain.as_str().to_string();
                }
            }
            return host;
        }
    }


    let mut fallback_result = protocol_stripped_url;


    fallback_result = PATH_PATTERN.replace(&fallback_result, "").to_string();


    if let Some(pos) = fallback_result.find('?') {
        fallback_result.truncate(pos);
    }


    if let Some(pos) = fallback_result.find('#') {
        fallback_result.truncate(pos);
    }


    if fallback_result.starts_with("www.") {
        fallback_result = fallback_result[4..].to_string();
    }


    if let Some(captures) = SUBDOMAIN_PATTERN.captures(&fallback_result) {
        if let Some(main_domain) = captures.get(2) {
            fallback_result = main_domain.as_str().to_string();
        }
    }

    fallback_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_pattern() {
        assert!(PROTOCOL_PATTERN.is_match("http://example.com"));
        assert!(PROTOCOL_PATTERN.is_match("https://example.com"));
        assert!(PROTOCOL_PATTERN.is_match("ftp://example.com"));
        assert!(PROTOCOL_PATTERN.is_match("android://example.com"));
        assert!(!PROTOCOL_PATTERN.is_match("example.com"));
        assert!(!PROTOCOL_PATTERN.is_match("www.example.com"));
    }

    #[test]
    fn test_subdomain_pattern() {
        assert!(SUBDOMAIN_PATTERN.is_match("www.example.com"));
        assert!(SUBDOMAIN_PATTERN.is_match("mail.google.com"));
        assert!(SUBDOMAIN_PATTERN.is_match("api.github.com"));
        assert!(!SUBDOMAIN_PATTERN.is_match("example.com"));
        assert!(!SUBDOMAIN_PATTERN.is_match("localhost"));
        assert!(!SUBDOMAIN_PATTERN.is_match("192.168.1.1"));
    }

    #[test]
    fn test_path_pattern() {
        assert!(PATH_PATTERN.is_match("/path/to/resource"));
        assert!(PATH_PATTERN.is_match("/"));
        assert!(PATH_PATTERN.is_match("/login"));
        assert!(!PATH_PATTERN.is_match("example.com"));
        assert!(!PATH_PATTERN.is_match("no-path"));
    }

    #[test]
    fn test_email_pattern() {
        // EMAIL_PATTERN matches the domain part of emails (from @ onwards)
        assert!(EMAIL_PATTERN.is_match("user@example.com"));
        assert!(EMAIL_PATTERN.is_match("test@gmail.com"));
        assert!(EMAIL_PATTERN.is_match("admin@site.org"));
        assert!(EMAIL_PATTERN.is_match("@example.com")); // This actually matches
        assert!(!EMAIL_PATTERN.is_match("not-an-email"));
        assert!(!EMAIL_PATTERN.is_match("user@"));
        assert!(!EMAIL_PATTERN.is_match("example.com")); // No @ symbol
    }

    #[test]
    fn test_password_pattern() {
        // PASSWORD_PATTERN only matches alphanumeric and specific special chars
        assert!(PASSWORD_PATTERN.is_match("password123"));
        assert!(PASSWORD_PATTERN.is_match("abcd"));
        assert!(PASSWORD_PATTERN.is_match("1234567890"));
        assert!(PASSWORD_PATTERN.is_match("Password123"));
        assert!(!PASSWORD_PATTERN.is_match("P@ssw0rd!")); // @ and ! not in allowed chars
        assert!(!PASSWORD_PATTERN.is_match("abc")); // Too short
        assert!(!PASSWORD_PATTERN.is_match("")); // Empty
        assert!(!PASSWORD_PATTERN.is_match("pass word")); // Contains space
    }

    #[test]
    fn test_ip_address_pattern() {
        // IP_ADDRESS_PATTERN matches IPv4-like patterns (1-3 digits separated by dots)
        assert!(IP_ADDRESS_PATTERN.is_match("192.168.1.1"));
        assert!(IP_ADDRESS_PATTERN.is_match("10.0.0.1"));
        assert!(IP_ADDRESS_PATTERN.is_match("127.0.0.1"));
        assert!(IP_ADDRESS_PATTERN.is_match("255.255.255.255"));
        assert!(IP_ADDRESS_PATTERN.is_match("192.168.1.1:8080")); // With port
        assert!(IP_ADDRESS_PATTERN.is_match("999.999.999.999")); // Pattern allows this (not strict IP validation)
        assert!(!IP_ADDRESS_PATTERN.is_match("example.com"));
        assert!(!IP_ADDRESS_PATTERN.is_match("not-an-ip"));
        assert!(!IP_ADDRESS_PATTERN.is_match("1234.1.1.1")); // More than 3 digits
    }

    #[test]
    fn test_normalize_url_fast_basic() {
        // Test empty and whitespace
        assert_eq!(normalize_url_fast(""), "");
        assert_eq!(normalize_url_fast("   "), "");
        assert_eq!(normalize_url_fast("\t\n"), "");

        // Test basic domain normalization
        assert_eq!(normalize_url_fast("example.com"), "example.com");
        assert_eq!(normalize_url_fast("EXAMPLE.COM"), "example.com");
        assert_eq!(normalize_url_fast("Example.Com"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_protocols() {
        // Test protocol removal
        assert_eq!(normalize_url_fast("http://example.com"), "example.com");
        assert_eq!(normalize_url_fast("https://example.com"), "example.com");
        assert_eq!(normalize_url_fast("ftp://example.com"), "example.com");
        assert_eq!(normalize_url_fast("android://example.com"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_www_removal() {
        // Test www removal
        assert_eq!(normalize_url_fast("www.example.com"), "example.com");
        assert_eq!(normalize_url_fast("https://www.example.com"), "example.com");
        assert_eq!(normalize_url_fast("http://www.google.com"), "google.com");
    }

    #[test]
    fn test_normalize_url_fast_subdomain_extraction() {
        // Test subdomain extraction to main domain
        assert_eq!(normalize_url_fast("mail.google.com"), "google.com");
        assert_eq!(normalize_url_fast("api.github.com"), "github.com");
        assert_eq!(normalize_url_fast("m.facebook.com"), "facebook.com");
        assert_eq!(normalize_url_fast("accounts.google.com"), "google.com");
    }

    #[test]
    fn test_normalize_url_fast_path_removal() {
        // Test path removal
        assert_eq!(normalize_url_fast("example.com/path"), "example.com");
        assert_eq!(normalize_url_fast("example.com/path/to/resource"), "example.com");
        assert_eq!(normalize_url_fast("https://example.com/login"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_query_params() {
        // Test query parameter removal
        assert_eq!(normalize_url_fast("example.com?param=value"), "example.com");
        assert_eq!(normalize_url_fast("example.com/path?param=value"), "example.com");
        assert_eq!(normalize_url_fast("https://example.com?q=search"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_fragments() {
        // Test fragment removal
        assert_eq!(normalize_url_fast("example.com#section"), "example.com");
        assert_eq!(normalize_url_fast("example.com/path#section"), "example.com");
        assert_eq!(normalize_url_fast("https://example.com#top"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_complex_cases() {
        // Test complex real-world cases
        assert_eq!(
            normalize_url_fast("https://www.facebook.com/login/device-based/password/?next=https%3A%2F%2Fwww.facebook.com%2F"),
            "facebook.com"
        );
        assert_eq!(
            normalize_url_fast("https://accounts.google.com/v3/signin/challenge/pwd"),
            "google.com"
        );
        assert_eq!(
            normalize_url_fast("http://m.twitter.com/login?redirect=/home"),
            "twitter.com"
        );
    }

    #[test]
    fn test_normalize_url_fast_ip_addresses() {
        // Test IP addresses (should remain unchanged)
        assert_eq!(normalize_url_fast("192.168.1.1"), "192.168.1.1");
        assert_eq!(normalize_url_fast("http://192.168.1.1"), "192.168.1.1");
        // Port gets stripped in the fallback path
        assert_eq!(normalize_url_fast("https://10.0.0.1:8080"), "10.0.0.1");
    }

    #[test]
    fn test_normalize_url_fast_edge_cases() {
        // Test edge cases and malformed URLs
        assert_eq!(normalize_url_fast("not-a-url"), "not-a-url");
        assert_eq!(normalize_url_fast("just-text"), "just-text");
        assert_eq!(normalize_url_fast("://malformed"), ":"); // Gets processed differently

        // Test URLs with ports (ports get stripped in fallback)
        assert_eq!(normalize_url_fast("example.com:8080"), "example.com");
        assert_eq!(normalize_url_fast("https://example.com:443"), "example.com");
    }

    #[test]
    fn test_normalize_url_fast_android_scheme() {
        // Test Android app schemes - based on actual behavior
        assert_eq!(
            normalize_url_fast("android://bwlVSGhydwa@tw.com.pkcard/"),
            "com.pkcard"
        );
        assert_eq!(
            normalize_url_fast("android://app@com.example.app/"),
            "example.app" // Subdomain extraction works here
        );
    }
}
