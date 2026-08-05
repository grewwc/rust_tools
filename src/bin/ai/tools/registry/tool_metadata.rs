//! 内置工具元数据加载。
//!
//! 每个工具对应一个独立 JSON 文件：`src/bin/ai/tool_descriptions/<tool>.json`，
//! 内容形如：
//!
//! ```json
//! {
//!   "name": "read_file",
//!   "description": "...",
//!   "parameters": { "type": "object", "properties": { ... } }
//! }
//! ```
//!
//! 其中 `parameters` 是发给模型的 JSON Schema。`execute` 是 Rust 函数，
//! 必须留在代码里；描述与参数 schema 都属于声明性元数据，统一外置。
//!
//! build.rs 编译期扫描该目录，把所有文件以 `include_str!` 嵌入并生成
//! `BUILTIN_TOOL_DESCRIPTION_FILES` 常量；运行时首次访问时解析并缓存。
//!
//! 用户可在以下位置放同名文件覆盖（不需要复制全部，只覆盖想改的工具），
//! 优先级从高到低：
//! 1. `AIO_TOOL_DESCRIPTIONS_DIR` 环境变量指定的目录
//! 2. 配置键 `ai.tool_descriptions.dir` 指定的目录（在 `~/.configW` 中配置）
//! 3. `~/.config/rust_tools/tool_descriptions/`
//! 4. 可执行文件同目录的 `tool_descriptions/`
//!
//! 注册处 `ToolSpec.description` 保持空占位，描述与参数 schema 全部来自元数据。
//! 某工具若缺少可用元数据（漏建文件 / 拼错文件名 / 内部 `name` 与注册名不一致 /
//! 描述为空），会静默退化为空描述 + 空 schema；`every_registered_tool_has_
//! builtin_metadata` 测试在编译期登记表与内置元数据之间钉死全覆盖契约。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::commonw::{configw, utils::expanduser};

// build.rs 生成的清单。
include!(concat!(env!("OUT_DIR"), "/tool_description_files.rs"));

/// 单个工具元数据文件的 JSON 结构。
#[derive(Debug, Deserialize)]
struct ToolMetadataFile {
    name: String,
    description: String,
    #[serde(default = "empty_parameters")]
    parameters: Value,
}

fn empty_parameters() -> Value {
    serde_json::json!({ "type": "object" })
}

#[derive(Debug, Clone)]
pub(crate) struct ToolMetadata {
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

fn parse_metadata_entry(content: &str, source: &str) -> Option<(String, ToolMetadata)> {
    match serde_json::from_str::<ToolMetadataFile>(content) {
        Ok(file) => Some((
            file.name,
            ToolMetadata {
                description: file.description,
                parameters: file.parameters,
            },
        )),
        Err(err) => {
            eprintln!("[tools] failed to parse tool metadata from {source}: {err}");
            None
        }
    }
}

/// 用户自定义工具元数据目录，按优先级从低到高排列（`load_metadata` 依次应用，
/// 后应用者覆盖前者，因此数组末尾优先级最高）。见模块文档。
fn user_tool_description_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // 4. 可执行文件同目录（最低）。
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let dir = exe_dir.join("tool_descriptions");
            if dir.is_dir() {
                dirs.push(dir);
            }
        }
    }

    // 3. ~/.config/rust_tools/tool_descriptions。
    if let Some(home) = std::env::var_os("HOME") {
        let dir = PathBuf::from(home).join(".config/rust_tools/tool_descriptions");
        if dir.is_dir() {
            dirs.push(dir);
        }
    }

    // 2. 配置键 ai.tool_descriptions.dir（~/.configW）。
    if let Some(raw) = configw::get_all_config().get_opt(AiConfig::TOOL_DESCRIPTIONS_DIR) {
        let raw = raw.trim().to_string();
        if !raw.is_empty() {
            let dir = PathBuf::from(expanduser(&raw).as_ref());
            if dir.is_dir() {
                dirs.push(dir);
            }
        }
    }

    // 1. AIO_TOOL_DESCRIPTIONS_DIR 环境变量（最高）。
    if let Ok(dir) = std::env::var("AIO_TOOL_DESCRIPTIONS_DIR") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            dirs.push(dir);
        }
    }

    dirs
}

fn load_user_dir(dir: &Path, into: &mut HashMap<String, ToolMetadata>) {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let source = path.display().to_string();
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                if let Some((name, metadata)) = parse_metadata_entry(&content, &source) {
                    into.insert(name, metadata);
                }
            }
            Err(err) => {
                eprintln!("[tools] failed to read {source}: {err}");
            }
        }
    }
}

/// 内置（编译期嵌入）元数据。独立成函数便于测试做「注册工具全覆盖」的
/// 干净校验，不掺入用户目录覆盖。
fn load_builtin_metadata() -> HashMap<String, ToolMetadata> {
    let mut metadata = HashMap::new();
    for (name, content) in BUILTIN_TOOL_DESCRIPTION_FILES {
        if let Some((parsed_name, entry)) =
            parse_metadata_entry(content, &format!("builtin:{name}"))
        {
            metadata.insert(parsed_name, entry);
        }
    }
    metadata
}

fn load_metadata() -> &'static HashMap<String, ToolMetadata> {
    static CACHE: OnceLock<HashMap<String, ToolMetadata>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut metadata = load_builtin_metadata();
        // 用户目录覆盖（优先级从低到高，后应用者覆盖前者）。
        for dir in user_tool_description_dirs() {
            load_user_dir(&dir, &mut metadata);
        }
        metadata
    })
}

/// 返回内置工具的有效描述。
///
/// - 元数据（内置 JSON 或用户覆盖）中存在 `name` 且描述非空时，使用配置值。
/// - 缺失或为空时返回 `fallback`。注册处 `ToolSpec.description` 现为占位空串，
///   正常路径不会用到 `fallback`；全覆盖契约由
///   `every_registered_tool_has_builtin_metadata` 测试钉死，空描述不再视为
///   可接受的兜底。
pub(crate) fn tool_description(name: &str, fallback: &str) -> String {
    let metadata = load_metadata();
    metadata
        .get(name)
        .map(|m| m.description.clone())
        .filter(|description| !description.trim().is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

/// 返回内置工具的有效参数 JSON Schema。
///
/// 配置文件中存在 `parameters` 时使用配置值；缺失时返回 `{"type": "object"}`。
pub(crate) fn tool_parameters(name: &str) -> Value {
    let metadata = load_metadata();
    metadata
        .get(name)
        .map(|m| m.parameters.clone())
        .unwrap_or_else(empty_parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::registry::common::ToolRegistration;

    #[test]
    fn builtin_metadata_files_parse() {
        assert!(!BUILTIN_TOOL_DESCRIPTION_FILES.is_empty());
        let metadata = load_metadata();
        assert!(metadata.contains_key("read_file"));
        assert!(metadata["read_file"].description.contains("line-numbered"));
        assert!(metadata["read_file"]
            .parameters
            .get("properties")
            .is_some());
        // 所有内置文件都应成功解析。
        assert_eq!(
            metadata.len(),
            BUILTIN_TOOL_DESCRIPTION_FILES.len(),
            "all builtin tool metadata files should parse"
        );
    }

    #[test]
    fn resolve_description_falls_back_when_absent() {
        let resolved = tool_description("__definitely_missing_tool__", "fallback text");
        assert_eq!(resolved, "fallback text");
    }

    #[test]
    fn resolve_parameters_defaults_to_object_when_absent() {
        let params = tool_parameters("__definitely_missing_tool__");
        assert_eq!(params, serde_json::json!({ "type": "object" }));
    }

    #[test]
    fn every_registered_tool_has_builtin_metadata() {
        // 钉死「注册登记表 ↔ 内置 JSON 文件」的双向全覆盖与一致性契约：
        // 1. 每个已注册工具都必须有对应 JSON，且描述非空、schema 结构合法；
        // 2. 每个内置 JSON 都必须对应一个已注册工具（无孤儿文件，防止删工具后
        //    遗留的陈旧元数据继续被嵌入）；
        // 3. JSON 内部的 `name` 字段必须与文件名 stem 一致，否则 build.rs 用
        //    stem 登记、运行时用内部 name 查表会发生静默错位。
        //
        // 注意：这里用 load_builtin_metadata()（不掺用户目录覆盖），并把内置
        // 清单按「文件名 stem」和「内部 name」分别建表，以暴露二者不一致。
        let metadata = load_builtin_metadata();
        let mut missing = Vec::new();      // 注册了但无 JSON（或内部 name 不匹配）
        let mut empty_desc = Vec::new();
        let mut bad_schema = Vec::new();
        for reg in inventory::iter::<ToolRegistration> {
            let name = &*reg.spec.name;
            match metadata.get(name) {
                None => missing.push(name.to_string()),
                Some(entry) => {
                    if entry.description.trim().is_empty() {
                        empty_desc.push(name.to_string());
                    }
                    // schema 合法：type=="object" 且 properties 是对象（可能为空对象）。
                    let props = entry.parameters.get("properties");
                    let is_object_schema = entry.parameters.get("type").and_then(|v| v.as_str())
                        == Some("object");
                    if !(is_object_schema && props.map(|v| v.is_object()).unwrap_or(false)) {
                        bad_schema.push(name.to_string());
                    }
                }
            }
        }

        // 反向：内置 JSON 中存在但未注册的工具（孤儿文件）。metadata 的 key
        // 是 JSON 内部 name；它必须落在注册集内。
        let registered: std::collections::BTreeSet<&str> = inventory::iter::<ToolRegistration>
            .into_iter()
            .map(|r| r.spec.name)
            .collect();
        let mut orphan = Vec::new();
        for (stem, content) in BUILTIN_TOOL_DESCRIPTION_FILES {
            let parsed: ToolMetadataFile =
                serde_json::from_str(content).expect("builtin metadata must parse");
            if parsed.name != *stem {
                orphan.push(format!("{}.json has inner name {:?}", stem, parsed.name));
            }
            if !registered.contains(parsed.name.as_str()) {
                orphan.push(format!(
                    "{}.json (name {:?}) has no registered tool",
                    stem, parsed.name
                ));
            }
        }

        assert!(
            missing.is_empty(),
            "registered tools missing builtin metadata file (or inner `name` mismatch): {missing:?}"
        );
        assert!(
            empty_desc.is_empty(),
            "registered tools with empty metadata description: {empty_desc:?}"
        );
        assert!(
            bad_schema.is_empty(),
            "registered tools whose `parameters` is not an object schema with `properties`: {bad_schema:?}"
        );
        assert!(
            orphan.is_empty(),
            "builtin metadata files with no matching registered tool, or filename/inner-name mismatch: {orphan:?}"
        );
    }
}
