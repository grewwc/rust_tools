//! # String Tools
//!
//! This module provides a variety of string-processing utility functions
//! covering common string manipulation scenarios.
//!
//! ## Module overview
//!
//! - [`calc`] - string calculation (length, byte count, etc.)
//! - [`check`] - string checking (empty, whitespace, etc.)
//! - [`find`] - string lookup (substrings, characters, etc.)
//! - [`mod@format`] - string formatting (line wrapping, indentation, etc.)
//! - [`indices`] - string index operations
//! - [`move`] - string move/copy operations
//! - [`search`] - string search
//! - [`split`] - string splitting
//! - [`trim`] - string trimming (whitespace, custom characters, etc.)
//!
//! ## Examples
//!
//! ### Trimming
//!
//! ```rust
//! use rust_tools::strw::trim_cutset;
//!
//! let text = "xxxhello worldxxx";
//! let trimmed = trim_cutset(text, "x");
//! assert_eq!(trimmed, "hello world");
//! ```
//!
//! ### Splitting
//!
//! ```rust
//! use rust_tools::strw::split_no_empty;
//!
//! let parts: Vec<&str> = split_no_empty("a,,b,,,c", ",");
//! assert_eq!(parts, vec!["a", "b", "c"]);
//! ```
//!
//! ### Text formatting
//!
//! ```rust
//! use rust_tools::strw::wrap;
//!
//! let text = "hello world this is a long line";
//! let wrapped = wrap(text, 20, 0, "-");
//! println!("{}", wrapped);
//! ```
//!
//! ### String checking
//!
//! ```rust
//! use rust_tools::strw::is_blank;
//!
//! assert!(is_blank("   "));
//! assert!(is_blank(""));
//! assert!(!is_blank("hello"));
//! ```

pub mod calc;
pub mod check;
pub mod find;
pub mod format;
pub mod indices;
pub mod r#move;
pub mod search;
pub mod split;
pub mod trim;

// Re-export the public contents of all submodules
pub use calc::*;
pub use check::*;
pub use find::*;
pub use format::*;
pub use indices::*;
pub use r#move::*;
pub use search::*;
pub use split::*;
pub use trim::*;
