//! search_overflow — 会话归档（overflow）专用搜索工具
//!
//! 上下文压缩后，被移出的原文零压缩归档到会话 assets 目录：
//! - `overflow-history.md`：被折叠的原始消息（用户/助手/工具结果）
//! - `tool-overflow-compressed/`：单条工具结果的完整快照
//! - `folded-tool-groups/`：整组折叠工具调用的原始消息
//! - `internal-note-overflow/`：被预算裁剪的内部上下文注记
//!
//! 模型需要找回被压缩的内容时，read_file 只能按已知路径分页读取；而压缩后
//! 模型往往只知道"大致有哪些内容"，不知道精确路径/行号。search_overflow
//! 复用共享内容搜索引擎（text_grep_tools::run_content_search），把搜索根
//! 固定为**当前会话**的归档目录，按查询返回带行号与 context 的 snippet，
//! 模型据此再 read_file 精读。
//!
//! 安全设计：搜索根不接受任意路径，只由 `current_session_assets_dir()` 计算，
//! 无活动 driver context 时直接报错，杜绝跨会话读取其它会话的归档。

use std::path::Path;

use serde_json::Value;

use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
use crate::ai::tools::storage::file_store::current_session_assets_dir;
use crate::ai::tools::text_grep_tools::{ContentSearchOptions, run_content_search};

/// 归档单文件大小上限：会话可能积累远超普通源文件（2 MiB）的 overflow 文件，
/// 读入内存逐行匹配即可，不必与普通搜索共用同一上限。
const OVERFLOW_MAX_FILE_SIZE: u64 = u64::MAX;
/// 每个文件最多保留多少条 snippet（与共享引擎的 MAX_MATCHES 一致）。
const MAX_MATCHES: usize = 200;
/// 归档内无匹配时引擎的返回串（与 run_content_search 的实现保持一致）。
const NO_MATCHES_MARKER: &str = "No matches found.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SearchScope {
    /// 全部会话归档内容。
    All,
    /// 仅 overflow-history.md（被折叠的原始消息）
    History,
    /// 仅 tool-overflow-compressed/（单条工具结果快照）
    ToolOutputs,
}

impl SearchScope {
    fn parse(raw: &str) -> SearchScope {
        match raw.trim() {
            "history" => SearchScope::History,
            "tool_outputs" => SearchScope::ToolOutputs,
            _ => SearchScope::All, // "all" 与未知值都回退到全量
        }
    }
}

struct OverflowSearchParams<'a> {
    query: &'a str,
    is_regex: bool,
    case_sensitive: bool,
    context_lines: usize,
    max_results: usize,
    file_pattern: Option<&'a str>,
    scope: SearchScope,
}

fn execute_search_overflow(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing 'query' parameter")?;
    if query.trim().is_empty() {
        return Err("query must not be empty".to_string());
    }
    let assets_dir = current_session_assets_dir().ok_or(
        "No active session archive: cannot resolve the current session's overflow directory.",
    )?;

    let params = OverflowSearchParams {
        query,
        is_regex: args["is_regex"].as_bool().unwrap_or(false),
        case_sensitive: args["case_sensitive"].as_bool().unwrap_or(true),
        context_lines: args["context_lines"].as_u64().unwrap_or(2).min(5) as usize,
        max_results: args["max_results"]
            .as_u64()
            .unwrap_or(50)
            .min(MAX_MATCHES as u64) as usize,
        file_pattern: args["file_pattern"].as_str(),
        scope: args["scope"]
            .as_str()
            .map(SearchScope::parse)
            .unwrap_or(SearchScope::All),
    };
    run_overflow_search(&assets_dir, &params)
}

/// 在给定会话归档目录内执行搜索。与 `execute_search_overflow` 分离以便单测
/// 直接构造临时归档目录调用（无需 driver context）。
fn run_overflow_search(
    assets_dir: &Path,
    params: &OverflowSearchParams<'_>,
) -> Result<String, String> {
    let roots: Vec<std::path::PathBuf> = match params.scope {
        SearchScope::History => vec![assets_dir.join("overflow-history.md")],
        SearchScope::ToolOutputs => vec![assets_dir.join("tool-overflow-compressed")],
        SearchScope::All => vec![
            assets_dir.join("overflow-history.md"),
            assets_dir.join("tool-overflow-compressed"),
            assets_dir.join("folded-tool-groups"),
            assets_dir.join("internal-note-overflow"),
            assets_dir.join("user-overflow-preserved"),
            assets_dir.join("image-overflow-preserved"),
        ],
    };

    let roots: Vec<_> = roots.into_iter().filter(|root| root.exists()).collect();
    if roots.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'",
            params.query
        ));
    }
    let mut sections: Vec<String> = Vec::new();
    // 为每个归档根预留份额，避免 overflow-history 的早期高频命中耗尽共享额度，
    // 令后面的工具组、内部注记或用户/图片保全归档永久不可见。
    let base_quota = params.max_results / roots.len();
    let extra = params.max_results % roots.len();
    for (root_index, root) in roots.iter().enumerate() {
        let root_quota = base_quota + usize::from(root_index < extra);
        if root_quota == 0 {
            continue;
        }
        let options = ContentSearchOptions {
            query: params.query,
            is_regex: params.is_regex,
            case_sensitive: params.case_sensitive,
            context_lines: params.context_lines,
            max_results: root_quota,
            file_pattern: params.file_pattern,
            extensions: None,
            // 展示绝对路径：read_file 按 effective_cwd() 解析相对路径，相对路径
            // 会让模型复制到项目目录下的错误位置；绝对路径可直接喂给 read_file。
            display_root: None,
            max_file_size: OVERFLOW_MAX_FILE_SIZE,
        };
        match run_content_search(root, &options) {
            Ok(out) if out == NO_MATCHES_MARKER => {}
            Ok(out) => sections.push(out),
            Err(e) => return Err(e),
        }
    }

    if sections.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'",
            params.query
        ));
    }
    Ok(sections.join("\n"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "search_overflow",
        description: "",

        execute: execute_search_overflow,
        groups: &["builtin", "core"],
    }
});

// search_overflow 的结果是"找回被压缩内容"的定位指针：内容复现代价高（再跑
// 一次相同搜索同样昂贵），禁止有损压缩，只能零压缩外溢留指针 stub；但旧结果
// 一旦被模型判定过时，允许裁剪释放上下文（与 read_file 一致）。
// 回答"搜索结果会不会立刻被压缩"：不会——命中结果保持原样，只在上下文预算
// 耗尽时整体外溢到磁盘并留指针，无行裁剪/摘要。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "search_overflow",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn make_temp_dir() -> PathBuf {
        // 必须唯一：并行测试若撞名，`create_dir_all` 幂等不报错，两个测试会共享
        // 同一目录，先结束的 `remove_dir_all` 会删掉另一个测试正在 seed/search
        // 的目录，引擎吞掉读错误后表现为偶发的 "No matches found" 断言失败。
        let dir = std::env::temp_dir().join(format!(
            "search_overflow_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seed_archive(dir: &Path) {
        fs::write(
            dir.join("overflow-history.md"),
            "## 用户\n原始问题：帮我实现一个工具\n## 助手\n回答：好的，foo 相关决策已记录。\n## 工具结果\n- 某条被压缩的命令输出\n",
        )
        .unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(
            tool_dir.join("20260804T140000Z-execute_command-deadbeef.txt"),
            "original_command: grep -n foo\nfoo line 1\nfoo line 2\n",
        )
        .unwrap();
        fs::write(
            tool_dir.join("20260804T140000Z-read_file-deadbeef.txt"),
            "read_file content\nbar line\n",
        )
        .unwrap();
        let folded_dir = dir.join("folded-tool-groups");
        fs::create_dir_all(&folded_dir).unwrap();
        fs::write(folded_dir.join("group.md"), "folded foo evidence\n").unwrap();
        let note_dir = dir.join("internal-note-overflow");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("note.md"), "internal foo state\n").unwrap();
        let user_dir = dir.join("user-overflow-preserved");
        fs::create_dir_all(&user_dir).unwrap();
        fs::write(user_dir.join("user.md"), "preserved foo request\n").unwrap();
        let image_dir = dir.join("image-overflow-preserved");
        fs::create_dir_all(&image_dir).unwrap();
        fs::write(image_dir.join("image.md"), "preserved foo image context\n").unwrap();
    }

    fn params(query: &str) -> OverflowSearchParams<'_> {
        OverflowSearchParams {
            query,
            is_regex: false,
            case_sensitive: true,
            context_lines: 1,
            max_results: 50,
            file_pattern: None,
            scope: SearchScope::All,
        }
    }

    #[test]
    fn search_all_scopes_both_locations() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("foo")).unwrap();
        assert!(
            out.contains("overflow-history.md"),
            "history file in results: {out}"
        );
        assert!(
            out.contains("tool-overflow-compressed/20260804T140000Z-execute_command-deadbeef.txt"),
            "tool output in results: {out}"
        );
        assert!(out.contains("foo line 1"));
        assert!(out.contains("folded-tool-groups/group.md"), "{out}");
        assert!(out.contains("internal-note-overflow/note.md"), "{out}");
        assert!(out.contains("user-overflow-preserved/user.md"), "{out}");
        assert!(out.contains("image-overflow-preserved/image.md"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_history_scope_only() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("原始问题");
        p.scope = SearchScope::History;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("overflow-history.md"));
        assert!(!out.contains("tool-overflow-compressed"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_tool_outputs_scope_with_pattern() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("foo");
        p.scope = SearchScope::ToolOutputs;
        p.file_pattern = Some("*execute_command*");
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("execute_command"),
            "command snapshot matched: {out}"
        );
        assert!(
            !out.contains("read_file"),
            "read_file snapshot excluded: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_case_insensitive() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("FOO");
        p.case_sensitive = false;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("foo line 1"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_no_matches_reports_cleanly() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("zzz_absent")).unwrap();
        assert!(out.contains("No matches found"), "clean miss: {out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_missing_archive_roots_are_skipped() {
        let dir = make_temp_dir(); // 空目录：两个根都不存在
        let out = run_overflow_search(&dir, &params("foo")).unwrap();
        assert!(out.contains("No matches found"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_all_scope_shares_max_results_across_roots() {
        let dir = make_temp_dir();
        // 每个根都放入大量匹配行，确保每个根都能独立命中 max_results 次
        fs::write(dir.join("overflow-history.md"), &"alpha\n".repeat(100)).unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("a.txt"), &"alpha\n".repeat(100)).unwrap();
        let folded_dir = dir.join("folded-tool-groups");
        fs::create_dir_all(&folded_dir).unwrap();
        fs::write(folded_dir.join("a.md"), &"alpha\n".repeat(100)).unwrap();
        let note_dir = dir.join("internal-note-overflow");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("a.md"), &"alpha\n".repeat(100)).unwrap();

        let mut p = params("alpha");
        p.max_results = 5;
        p.context_lines = 0;
        let out = run_overflow_search(&dir, &p).unwrap();

        // max_results 是跨所有根的共享上限，4 个根不能各自返回 5 条
        let section_count = out.matches("match(es) in").count();
        assert_eq!(
            section_count, 1,
            "max_results 应作为共享额度：期望仅 1 个根返回结果，实际 {section_count} 个: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
