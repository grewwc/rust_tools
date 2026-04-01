//! # Rust Tools
//!
//! 一个多功能的 Rust 工具库，提供常用的数据结构、算法和实用工具。
//!
//! ## 模块概览
//!
//! - [`algow`] - 算法工具（二分查找等）
//! - [`clipboard`] - 剪贴板操作（文本、图片、二进制数据）
//! - [`cmd`] - 命令执行工具
//! - [`common`] - 通用类型和工具函数
//! - [`cw`] - 容器和数据结构（各种集合、树、图等）
//! - [`jsonw`] - JSON 处理工具（解析、格式化、差异比较）
//! - [`pdfw`] - PDF 文件处理
//! - [`sortw`] - 排序工具
//! - [`strw`] - 字符串处理工具
//! - [`terminalw`] - 终端相关工具（文件查找、路径处理等）
//!
//! ## 快速开始
//!
//! ```rust
//! use rust_tools::strw::trim::trim_cutset;
//! use rust_tools::cw::queue::Queue;
//! use rust_tools::algow::bisect_left;
//!
//! // 字符串处理
//! let trimmed = trim_cutset("  hello  ", " ");
//! assert_eq!(trimmed, "hello");
//!
//! // 使用队列
//! let mut queue: Queue<i32> = Queue::new();
//! queue.enqueue(1);
//! queue.enqueue(2);
//!
//! // 二分查找
//! let arr = [1, 3, 5, 7, 9];
//! let pos = bisect_left(&arr, &5);
//! assert_eq!(pos, 2);
//! ```
//!
//! ## 特性
//!
//! - 🚀 高性能：使用 `rustc-hash` 提供快速的哈希表实现
//! - 📦 模块化：按需使用各个模块
//! - 🔧 实用：提供日常开发中常用的工具函数
//! - 📚 文档完善：每个模块都有详细的使用示例

pub mod algow;
pub mod clipboard;
pub mod cmd;
pub mod common;
pub mod cw;
pub mod jsonw;
pub mod pdfw;
pub mod sortw;
pub mod strw;
pub mod terminalw;
