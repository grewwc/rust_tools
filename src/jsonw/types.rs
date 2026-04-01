//! JSON 处理工具的类型定义

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON 值包装器
///
/// 提供对 `serde_json::Value` 的封装，便于进行 JSON 操作。
///
/// # 示例
///
/// ```rust
/// use rust_tools::jsonw::Json;
///
/// let json = Json::default();
/// // 可以进一步扩展功能
/// ```
#[derive(Debug, Clone)]
pub struct Json {
    pub(crate) value: Value,
}

impl Default for Json {
    /// 创建默认的空 JSON 对象
    fn default() -> Self {
        Self {
            value: Value::Object(serde_json::Map::new()),
        }
    }
}

/// JSON 解析选项
///
/// 控制 JSON 解析时的行为。
///
/// # 示例
///
/// ```rust
/// use rust_tools::jsonw::ParseOptions;
///
/// // 使用默认选项
/// let options = ParseOptions::default();
/// assert!(options.allow_comment);
/// assert!(options.remove_special_chars);
///
/// // 自定义选项
/// let options = ParseOptions {
///     allow_comment: false,
///     remove_special_chars: false,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ParseOptions {
    /// 是否允许注释
    ///
    /// 如果为 `true`，解析器会尝试处理 JSON 中的注释（非标准 JSON 特性）
    pub allow_comment: bool,
    /// 是否移除特殊字符
    ///
    /// 如果为 `true`，解析器会尝试清理输入中的特殊字符
    pub remove_special_chars: bool,
}

impl Default for ParseOptions {
    /// 创建默认解析选项
    ///
    /// 默认配置：
    /// - `allow_comment`: true
    /// - `remove_special_chars`: true
    fn default() -> Self {
        Self {
            allow_comment: true,
            remove_special_chars: true,
        }
    }
}

/// JSON 差异条目
///
/// 表示两个 JSON 值之间的一个差异。
///
/// # 示例
///
/// ```rust
/// use rust_tools::jsonw::DiffEntry;
/// use serde_json::json;
///
/// let entry = DiffEntry {
///     key: "name".to_string(),
///     old: json!(null),
///     new: json!("Alice"),
/// };
///
/// assert_eq!(entry.key, "name");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffEntry {
    /// 差异所在的路径/键名
    pub key: String,
    /// 旧值
    pub old: Value,
    /// 新值
    pub new: Value,
}
