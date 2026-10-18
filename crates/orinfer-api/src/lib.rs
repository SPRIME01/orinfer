//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orinfer-engine.
mod config;
mod server;
pub use config::{BatchConfig, PreprocessingLimits, ServerConfig};
pub use server::batch::{BatchReport, run as run_batch};
pub use server::run;
