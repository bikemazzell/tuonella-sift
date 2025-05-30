pub mod constants;
pub mod core;
pub mod config;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda;

// Re-export main types for easier access
pub use config::model::Config;
pub use core::record::Record;
pub use core::deduplication::{deduplicate_records, ProcessingStats};

#[cfg(feature = "cuda")]
pub use cuda::processor::CudaProcessor; 