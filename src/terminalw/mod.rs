//! # Terminal Tools
//!
//! This module provides terminal-related utilities, including file lookup,
//! path handling, and command parsing.
//!
//! ## Module overview
//!
//! - [`filepath`] - file path handling (glob matching, etc.)
//! - [`find()`] - file lookup
//! - [`parser`] - command parser
//! - [`utils`] - general-purpose helper functions
//!
//! ## Main features
//!
//! ### File lookup
//!
//! Provides a powerful file lookup that supports:
//! - concurrent lookup across multiple directories
//! - file-type filtering
//! - exclude patterns
//! - depth limits
//!
//! ```rust,ignore
//! use rust_tools::terminalw::find;
//!
//! // The find function requires Arc<Task> and Arc<WaitGroup> arguments
//! // See the terminalw::find module docs for full usage
//! ```
//!
//! ### Glob matching
//!
//! Supports case-insensitive glob pattern matching:
//!
//! ```rust,ignore
//! use rust_tools::terminalw::{glob_paths, glob_case_insensitive};
//!
//! // Find all files matching the pattern
//! let paths = glob_paths("*.rs", ".");
//!
//! // Case-insensitive matching
//! let paths = glob_case_insensitive("README.md", ".");
//! ```
//!
//! ### Command parsing
//!
//! Parses terminal commands and arguments:
//!
//! ```rust
//! use rust_tools::terminalw::{Parser, ParserOption};
//!
//! let parser = Parser::new();
//! // Parse a command...
//! ```
//!
//! ## Configuration options
//!
//! The file lookup module offers several configurable global options:
//!
//! - [`MAX_LEVEL`] - maximum lookup depth
//! - [`COUNT`] - whether to show counts
//! - [`VERBOSE`] - verbose mode
//! - [`EXTENSIONS`] - file-extension filtering
//! - [`EXCLUDE`] - exclude patterns

pub mod filepath;
pub mod find;
mod internal;
pub mod parser;
pub mod utils;

// Re-export commonly used types and functions
pub use filepath::{glob_case_insensitive, glob_paths};
pub use find::{
    CHECK_EXTENSION, COUNT, EXCLUDE, EXTENSIONS, FILE_NAMES_NOT_CHECK, FILE_NAMES_TO_CHECK,
    MAX_LEVEL, NUM_PRINT, SyncSet, VERBOSE, WaitGroup, change_threads, find,
};
pub use internal::actiontype::ActionList;
pub use parser::{Parser, ParserOption, disable_parser_number, new_parser};
pub use utils::{add_quote, format_file_extensions, map_to_string};
