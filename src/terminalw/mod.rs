//! # 终端工具 (Terminal Tools)
//!
//! 本模块提供终端相关的工具功能，包括文件查找、路径处理、命令解析等。
//!
//! ## 子模块概览
//!
//! - [`filepath`] - 文件路径处理（glob 匹配等）
//! - [`find()`] - 文件查找功能
//! - [`parser`] - 命令解析器
//! - [`utils`] - 通用工具函数
//!
//! ## 主要功能
//!
//! ### 文件查找
//!
//! 提供强大的文件查找功能，支持：
//! - 多目录并发查找
//! - 文件类型过滤
//! - 排除模式
//! - 深度限制
//!
//! ```rust,ignore
//! use rust_tools::terminalw::find;
//!
//! // find 函数需要 Arc<Task> 和 Arc<WaitGroup> 参数
//! // 详细用法请参考 terminalw::find 模块文档
//! ```
//!
//! ### Glob 匹配
//!
//! 支持大小写不敏感的 glob 模式匹配：
//!
//! ```rust,ignore
//! use rust_tools::terminalw::{glob_paths, glob_case_insensitive};
//!
//! // 查找所有匹配模式的文件
//! let paths = glob_paths("*.rs", ".");
//!
//! // 大小写不敏感匹配
//! let paths = glob_case_insensitive("README.md", ".");
//! ```
//!
//! ### 命令解析
//!
//! 解析终端命令和参数：
//!
//! ```rust
//! use rust_tools::terminalw::{Parser, ParserOption};
//!
//! let parser = Parser::new();
//! // 解析命令...
//! ```
//!
//! ## 配置选项
//!
//! 文件查找模块提供多个可配置的全局选项：
//!
//! - [`MAX_LEVEL`] - 最大查找深度
//! - [`COUNT`] - 是否显示计数
//! - [`VERBOSE`] - 详细模式
//! - [`EXTENSIONS`] - 文件扩展名过滤
//! - [`EXCLUDE`] - 排除模式

pub mod filepath;
pub mod find;
mod internal;
pub mod parser;
pub mod utils;

// 重新导出常用类型和函数
pub use filepath::{glob_case_insensitive, glob_paths};
pub use find::{
    CHECK_EXTENSION, COUNT, EXCLUDE, EXTENSIONS, FILE_NAMES_NOT_CHECK, FILE_NAMES_TO_CHECK,
    MAX_LEVEL, NUM_PRINT, SyncSet, VERBOSE, WaitGroup, change_threads, find,
};
pub use internal::actiontype::ActionList;
pub use parser::{Parser, ParserOption, disable_parser_number, new_parser};
pub use utils::{add_quote, format_file_extensions, map_to_string};
