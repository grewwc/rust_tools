//! # 通用工具 (Common Tools)
//!
//! 本模块提供项目中多个模块共享的通用类型和工具函数。
//!
//! ## 子模块概览
//!
//! - [`configw`] - 配置相关工具
//! - [`editor`] - 编辑器集成工具
//! - [`filename`] - 文件名处理工具
//! - [`prompt`] - 提示和交互工具
//! - [`types`] - 通用类型定义
//! - [`utils`] - 通用工具函数
//!
//! ## 核心类型
//!
//! 本模块最重要的类型是 [`types`] 中定义的高性能集合类型：
//!
//! - [`FastMap<K, V>`](types::FastMap) - 基于 FxHasher 的高性能 HashMap
//! - [`FastSet<T>`](types::FastSet) - 基于 FxHasher 的高性能 HashSet
//!
//! 这些类型使用 `rustc-hash` crate 提供的 FxHasher，在大多数场景下
//! 比标准的 HashMap/HashSet 更快。
//!
//! ## 使用示例
//!
//! ### 使用高性能集合
//!
//! ```rust
//! use rust_tools::common::types::{FastMap, FastSet};
//!
//! // 快速 HashMap
//! let mut map: FastMap<&str, i32> = FastMap::default();
//! map.insert("a", 1);
//! map.insert("b", 2);
//!
//! // 快速 HashSet
//! let mut set: FastSet<i32> = FastSet::default();
//! set.insert(1);
//! set.insert(2);
//! set.insert(3);
//! ```

pub mod configw;
pub mod editor;
pub mod filename;
pub mod prompt;
pub mod types;
pub mod utils;
