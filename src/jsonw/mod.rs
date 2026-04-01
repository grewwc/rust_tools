//! # JSON 处理工具 (JSON Tools)
//!
//! 本模块提供 JSON 数据的处理工具，包括解析、格式化、排序和差异比较等功能。
//!
//! ## 功能概览
//!
//! - [`diff_json`] - 比较两个 JSON 值的差异
//! - [`sanitize_json_input`] - 清理和规范化 JSON 输入
//! - [`Json`] - JSON 值包装器类型
//! - [`DiffEntry`] - JSON 差异条目类型
//! - [`ParseOptions`] - JSON 解析选项
//!
//! ## 使用示例
//!
//! ### JSON 差异比较
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
//! ### 清理 JSON 输入
//!
//! ```rust
//! use rust_tools::jsonw::sanitize_json_input;
//!
//! // 清理带有注释的 JSON
//! let input = r#"{
//!     // 这是一个注释
//!     "name": "Alice"
//! }"#;
//!
//! use rust_tools::jsonw::ParseOptions;
//! let options = ParseOptions::default();
//! let cleaned = sanitize_json_input(input, options);
//! // cleaned 现在是可以被标准 JSON 解析器接受的格式
//! ```
//!
//! ## 类型
//!
//! - [`DiffEntry`] - 表示 JSON 差异的条目
//! - [`Json`] - JSON 值包装器
//! - [`ParseOptions`] - 控制 JSON 解析行为的选项

pub mod diff;
pub mod json;
pub mod sanitize;
pub mod sort;
pub mod types;

// 重新导出常用类型和函数
pub use diff::diff_json;
pub use sanitize::sanitize_json_input;
pub use types::{DiffEntry, Json, ParseOptions};
