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
//! use rust_tools::commonw::{FastMap, FastSet};
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

// 重新导出常用类型

/// 测量并打印函数执行时间（使用 `eprintln!` 输出）。
/// 
/// 每次调用函数时，会自动计算执行耗时并打印类似 `[timing] my_function took 1.23 ms`。
/// 可以接受一个可选的字符串参数作为标签，例如：`#[measure_time("custom_label")]`。
///
/// # Example
/// ```rust
/// use rust_tools::commonw::measure_time;
/// 
/// #[measure_time]
/// fn compute() {
///     // ... 耗时操作
/// }
/// ```
pub use rust_tools_macros::measure_time;

/// 与 `measure_time` 类似，但只在 Debug 模式（`cfg!(debug_assertions)`）下生效。
/// Release 模式下不会产生任何性能开销或输出。
pub use rust_tools_macros::debug_measure_time;

/// 为函数添加 LRU（最近最少使用）缓存机制，自动缓存给定参数的返回结果。
/// 
/// 需要指定 `cap`（容量），还可以可选地指定 `ttl_ms`（缓存过期时间，毫秒）。
/// 函数的参数必须实现 `Clone`, `Hash`, 和 `Eq`，返回值必须实现 `Clone`。
///
/// # Example
/// ```rust
/// use rust_tools::commonw::lru_cache;
/// 
/// // 容量为 100，缓存过期时间为 5000 毫秒
/// #[lru_cache(cap = 100, ttl_ms = 5000)]
/// fn heavy_computation(a: i32, b: i32) -> i32 {
///     // ... 耗时计算
///     a + b
/// }
/// ```
pub use rust_tools_macros::lru_cache;

pub use types::{FastMap, FastSet};

/// 返回 `available_parallelism / 2`，至少为 1。
///
/// 项目里多处沿用此模式来配置并行度（避免 100% 占满 CPU）。统一封装后
/// 不再依赖 `num_cpus` crate；`available_parallelism` 失败时退回到 1。
#[inline]
pub fn half_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(1)
        .max(1)
}

/// 文件遍历时应当整体跳过的目录名（构建产物 / 依赖缓存 / VCS 元信息）。
///
/// 这些目录通常体积巨大且不含用户源码，扫描进去会让 search/glob 类工具
/// 单线程吃满 CPU（典型如 `target/`、`node_modules/`）。各处 walker /
/// glob 后处理都应统一调用 [`is_skip_dir`] 过滤。
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".cargo",
];

/// 判断目录名是否在 [`SKIP_DIRS`] 黑名单中。
#[inline]
pub fn is_skip_dir(name: &str) -> bool {
    SKIP_DIRS.iter().any(|d| *d == name)
}

/// 判断给定路径中是否含有任何 [`SKIP_DIRS`] 段；用于 glob 结果后置过滤。
pub fn path_contains_skip_dir(path: &std::path::Path) -> bool {
    use std::path::Component;
    path.components().any(|c| match c {
        Component::Normal(s) => s.to_str().map(is_skip_dir).unwrap_or(false),
        _ => false,
    })
}
