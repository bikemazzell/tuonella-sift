pub mod model;

use anyhow::Result;
use std::path::Path;
use tokio::fs;

// Re-export main types
pub use self::model::Config;

impl Config {
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).await?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
} 