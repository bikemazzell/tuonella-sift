// External Sort module - the main implementation
pub mod external_sort;

// Constants used by external_sort
pub mod constants;

// CUDA support (optional)
#[cfg(feature = "cuda")]
pub mod cuda;

// Config module - only needed for CUDA types
#[cfg(feature = "cuda")]
pub mod config;

// Re-export main types for convenience
pub use external_sort::{ExternalSortConfig, ExternalSortProcessor, ExternalSortStats};