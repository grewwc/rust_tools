//! # JSON Tools
//!
//! This module provides JSON data processing utilities, including parsing,
//! formatting, sorting, and diffing.
//!
//! ## Feature Overview
//!
//! - [`diff_json`] - Compares the differences between two JSON values
//! - [`sanitize_json_input`] - Cleans and normalizes JSON input
//! - [`Json`] - JSON value wrapper type
//! - [`DiffEntry`] - JSON diff entry type
//! - [`ParseOptions`] - JSON parsing options
//!
//! ## Usage Examples
//!
//! ### JSON Diffing
//!
//! ```rust
//! use rust_tools::jsonw::diff_json;
//! use serde_json::json;
//!
//! let old = json!({"name": "Alice", "age": 25});
//! let new = json!({"name": "Alice", "age": 26, "city": "Beijing"});
//!
//! let diffs = diff_json(&old, &new, false);
//! for diff in diffs {
//!     println!("路径：{}, 旧值：{:?}, 新值：{:?}", diff.key, diff.old, diff.new);
//! }
//! ```
//!
//! ### Sanitizing JSON Input
//!
//! ```rust
//! use rust_tools::jsonw::sanitize_json_input;
//!
//! // Sanitize JSON that contains comments
//! let input = r#"{
//!     // this is a comment
//!     "name": "Alice"
//! }"#;
//!
//! use rust_tools::jsonw::ParseOptions;
//! let options = ParseOptions::default();
//! let cleaned = sanitize_json_input(input, options);
//! // cleaned is now in a form accepted by standard JSON parsers
//! ```
//!
//! ## Types
//!
//! - [`DiffEntry`] - Represents a JSON diff entry
//! - [`Json`] - JSON value wrapper
//! - [`ParseOptions`] - Options controlling JSON parsing behavior

pub mod diff;
pub mod json;
pub mod sanitize;
pub mod sort;
pub mod types;

// Re-export commonly used types and functions
pub use diff::diff_json;
pub use sanitize::sanitize_json_input;
pub use types::{DiffEntry, Json, ParseOptions};
