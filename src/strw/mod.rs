//! # 字符串处理工具 (String Tools)
//!
//! 本模块提供多种字符串处理工具函数，涵盖常见的字符串操作场景。
//!
//! ## 子模块概览
//!
//! - [`calc`] - 字符串计算（长度、字节数等）
//! - [`check`] - 字符串检查（是否为空、空白字符等）
//! - [`find`] - 字符串查找（查找子串、字符等）
//! - [`mod@format`] - 字符串格式化（文本换行、缩进等）
//! - [`indices`] - 字符串索引操作
//! - [`move`] - 字符串移动/复制操作
//! - [`search`] - 字符串搜索
//! - [`split`] - 字符串分割
//! - [`trim`] - 字符串修剪（去除空白字符、指定字符等）
//!
//! ## 使用示例
//!
//! ### 字符串修剪
//!
//! ```rust
//! use rust_tools::strw::trim_cutset;
//!
//! let text = "xxxhello worldxxx";
//! let trimmed = trim_cutset(text, "x");
//! assert_eq!(trimmed, "hello world");
//! ```
//!
//! ### 字符串分割
//!
//! ```rust
//! use rust_tools::strw::split_no_empty;
//!
//! let parts: Vec<&str> = split_no_empty("a,,b,,,c", ",");
//! assert_eq!(parts, vec!["a", "b", "c"]);
//! ```
//!
//! ### 文本格式化
//!
//! ```rust
//! use rust_tools::strw::wrap;
//!
//! let text = "hello world this is a long line";
//! let wrapped = wrap(text, 20, 0, "-");
//! println!("{}", wrapped);
//! ```
//!
//! ### 字符串检查
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

// 重新导出所有子模块的公共内容
pub use calc::*;
pub use check::*;
pub use find::*;
pub use format::*;
pub use indices::*;
pub use r#move::*;
pub use search::*;
pub use split::*;
pub use trim::*;
