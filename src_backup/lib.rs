pub mod config;
pub mod constants;
pub mod deduplicator;
pub mod patterns;
pub mod record;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda_processor;

pub use config::Config;
pub use deduplicator::Deduplicator;
pub use record::{Record, FieldDetector, DeduplicationMap};
