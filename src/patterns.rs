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
