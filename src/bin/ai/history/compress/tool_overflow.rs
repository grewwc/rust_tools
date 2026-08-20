//! 工具溢出处理与持久化摘要构建。
//!
//! - `prepare_tool_messages_structured`：结构化裁剪 tool 消息
//! - `build_persisted_summary_text` / `build_persisted_summary_text_with_app`：构建持久化摘要
//! - `write_preserved_tool_overflow_file` 等：将溢出内容写入归档文件
//! - `structured_tool_output_summary`：工具结果结构化摘要
//! - `is_non_compressible_tool` / `is_preserved_user_or_image_stub`：工具分类判断

use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::files::extract_key_lines;
use crate::ai::{
    history::HistoryMessageSummarizer,
    tools::{storage::file_store::FileStore, tool_history_policy},
    types::App,
};

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
use super::tool_groups::{recent_tool_group_message_indices, recent_tool_result_groups};
use super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, IMAGE_OVERFLOW_SPILL_MIN_CHARS, KEEP_RECENT_TOOL_GROUPS,
    KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX, PRESERVED_CONTENT_STUB_PREFIX,
    PRESERVED_IMAGE_OVERFLOW_DIR, PRESERVED_TOOL_OVERFLOW_DIR, PRESERVED_USER_OVERFLOW_DIR,
    PlannedArchiveWrite, USER_OVERFLOW_SPILL_MIN_CHARS, automatic_summary_body, content_sha256_hex,
    dedup_adjacent, keep_recent_user_turns_when_trimming, message_contains_image,
    normalize_whitespace, redact_images_except_last, strip_nested_prior_summary_prefixes,
    tool_message_indices, value_to_string,
};

const PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str = "[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]";
const LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str =
    "Output preserved for non-compressible tool `";

pub(super) async fn build_persisted_summary_text_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> String {
    let mut prepared = messages.to_vec();
    prepare_tool_messages_structured(
        &mut prepared,
        360,
        KEEP_RECENT_TOOL_GROUPS,
        None,
        None,
        &FxHashSet::default(),
    );
    redact_images_except_last(&mut prepared, 0);
    dedup_adjacent(&mut prepared);
    normalize_internal_notes_for_summary_model(&mut prepared);

    if let Some(summary) = app.summarize_history_messages(&prepared, max_chars).await {
        let summary = normalize_whitespace(&summary);
        if !summary.is_empty() {
            return summary;
        }
    }

    build_persisted_summary_text(messages, max_chars)
}

pub(super) fn normalize_internal_notes_for_summary_model(messages: &mut Vec<Message>) {
    let mut out = Vec::with_capacity(messages.len());
    let mut seen_auto_summary = false;

    for mut message in messages.drain(..) {
        if message.role == ROLE_INTERNAL_NOTE {
            let text = value_to_string(&message.content);
            if let Some(body) = automatic_summary_body(&text) {
                if seen_auto_summary {
                    continue;
                }
                let body = strip_nested_prior_summary_prefixes(body);
                if !body.is_empty() {
                    message.content = Value::String(format!(
                        "Existing history summary (for this compression to absorb; do not copy verbatim):\n{}",
                        summarize_text(&body, 2_000)
                    ));
                    out.push(message);
                    seen_auto_summary = true;
                }
                continue;
            }

            if text.trim_start().contains(COMPRESSED_TOOL_EVIDENCE_MARKER) {
                out.push(message);
                continue;
            }

            // 普通 internal_note 多为过程性提示、cache/loop 状态或 self_note 的
            // inline 副本。它们不应被当成长期历史事实交给摘要模型反复吸收。
            continue;
        }
        out.push(message);
    }

    *messages = out;
}

pub(super) fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let indices = tool_message_indices(messages);
    let protected_indices = recent_tool_group_message_indices(messages, keep_recent_groups);
    for &idx in &indices {
        let message = &mut messages[idx];
        let text = value_to_string(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        // 已外溢的 precision 工具结果是稳定指针，不能再把 stub 当成原始结果
        // 外溢一次。否则每轮压缩都会写出 `stub -> stub` 新文件，既泄漏磁盘，
        // 也让模型必须沿多层指针才能回到原始证据。
        if is_preserved_tool_overflow_stub(&text) {
            continue;
        }

        let tool_name = message
            .tool_call_id
            .as_deref()
            .and_then(|id| id_to_tool_name.get(id))
            .map(|s| s.as_str());
        if let Some(name) = tool_name
            && is_non_compressible_tool(name)
        {
            // 最近完整工具组不外溢：刚读到的文件/检索结果必须在下一轮请求里
            // 完整可见，否则模型看到的是「已卸载，请重读」stub，会立刻再发一次
            // 同样的 read_file——在会话超软阈值、每轮都跑压缩时表现为无限重读。
            // 只有保护尾窗之外的旧 precision 结果才零压缩外溢到磁盘。
            let is_explicitly_protected = message
                .tool_call_id
                .as_ref()
                .is_some_and(|id| protected_tool_call_ids.contains(id));
            if !is_explicitly_protected
                && !protected_indices.contains(&idx)
                && text.chars().count() > max_chars_per_msg
            {
                // 回读本 session 已归档 asset 时复用既有文件，避免「外溢→回读→
                // 再外溢」每次铸造新文件（与 Path C 复用逻辑一致）。
                let existing_asset_path = overflow_dir.and_then(|dir| {
                    message
                        .tool_call_id
                        .as_deref()
                        .and_then(|id| id_to_tool_args.get(id))
                        .and_then(|args| {
                            preserved_tool_overflow_path_in_arguments(name, args, dir, cwd)
                        })
                });
                let reused_existing = existing_asset_path.is_some();
                if let Some(path) = existing_asset_path.or_else(|| {
                    overflow_dir.and_then(|dir| {
                        write_preserved_tool_overflow_file(
                            dir,
                            message.tool_call_id.as_deref(),
                            name,
                            &text,
                        )
                    })
                }) {
                    let recall_lines = message
                        .tool_call_id
                        .as_deref()
                        .and_then(|id| id_to_tool_args.get(id))
                        .map(|args| build_tool_overflow_recall_lines(name, args))
                        .unwrap_or_default();
                    let stub =
                        build_preserved_tool_overflow_stub(&path, name, &text, &recall_lines);
                    // 防膨胀：外溢必须真正腾出空间。小结果（如几百字节的 grep 输出）
                    // 换成带完整预览的 stub 往往比原文更大，只会虚增占用、还逼模型
                    // 去回读归档——正是「读结果一直被归档成 stub」的成因。膨胀时保留
                    // 原文；仅删除本次新建的文件（复用的 asset 归其它消息的外溢所有，
                    // 删除会让那条消息的 stub 悬空）。与 enforce_protected_precision_
                    // group_budget / spill_protected_precision_to_fit 的守卫一致。
                    if stub.chars().count() >= text.chars().count() {
                        if !reused_existing {
                            let _ = std::fs::remove_file(&path);
                        }
                    } else {
                        message.content = Value::String(stub);
                    }
                }
            }
            continue;
        }

        if protected_indices.contains(&idx) {
            // 最近完整工具组的普通工具结果仍保留全文，避免误伤近端上下文。
            continue;
        }

        let summary = structured_tool_output_summary(&text, max_chars_per_msg);
        if !summary.is_empty() && summary != text {
            message.content = Value::String(summary);
        }
    }
}

/// 对请求上下文中的工具结果施加不可绕过的物理上限。
///
/// canonical history 始终保留原文；这里仅修改请求侧副本。普通近端结果继续保持
/// raw，只有超过绝对上限时才外溢，避免 SQLite snapshot 水位之后的 canonical
/// tail 绕过 current-turn 投影，再次把超大输出送进模型。
pub(super) fn cap_oversized_tool_results_for_context(
    messages: &mut [Message],
    hard_cap_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> usize {
    if hard_cap_chars == 0 {
        return 0;
    }

    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let mut capped = 0;
    for idx in tool_message_indices(messages) {
        let text = value_to_string(&messages[idx].content);
        if is_preserved_tool_overflow_stub(&text) || text.chars().nth(hard_cap_chars).is_none() {
            continue;
        }

        let tool_call_id = messages[idx].tool_call_id.as_deref();
        let tool_name = tool_call_id
            .and_then(|id| id_to_tool_name.get(id))
            .map(String::as_str)
            .unwrap_or("unknown_tool");
        let recall_lines = tool_call_id
            .and_then(|id| id_to_tool_args.get(id))
            .map(|args| build_tool_overflow_recall_lines(tool_name, args))
            .unwrap_or_default();
        // 回读本 session 已归档 asset（read_file / execute_command cat 指向
        // tool-overflow-compressed/ 直接子文件）时复用既有文件，而不是再铸造一个
        // 随机名新文件——否则「外溢→回读→再外溢」会在每次回读时生成新 archive，
        // 模型永远沿新指针重读，形成无界链。与 Path C 的复用逻辑保持一致。
        let existing_asset_path = overflow_dir.and_then(|dir| {
            tool_call_id
                .and_then(|id| id_to_tool_args.get(id))
                .and_then(|args| {
                    preserved_tool_overflow_path_in_arguments(tool_name, args, dir, cwd)
                })
        });
        let replacement = existing_asset_path
            .or_else(|| {
                overflow_dir.and_then(|dir| {
                    write_preserved_tool_overflow_file(dir, tool_call_id, tool_name, &text)
                })
            })
            .map(|path| {
                build_preserved_tool_overflow_stub(&path, tool_name, &text, &recall_lines)
            })
            .unwrap_or_else(|| {
                let mut stub = format!(
                    "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
                     Output for tool `{tool_name}` exceeded the absolute request-context cap, but the session asset could not be written. Canonical history still retains the raw result."
                );
                for line in &recall_lines {
                    stub.push('\n');
                    stub.push_str(line);
                }
                stub.push('\n');
                stub.push_str(&build_overflow_content_preview(&text));
                stub
            });
        messages[idx].content = Value::String(replacement);
        capped += 1;
    }
    capped
}

/// 最新并行批次可能单独超过上下文窗口。此时仍按完整组判定，但对注册为高精度
/// grounding 的结果设置 inline 上限：超过预算的结果零压缩外溢并保留可召回 stub。
/// `task` / `task_wait` 等聚合结果没有注册该标志，不会挤占 read_file 等证据的预算。
pub(super) fn enforce_protected_precision_group_budget(
    messages: &mut [Message],
    keep_recent_groups: usize,
    inline_budget: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
    allow_overflow_protected: bool,
) {
    let Some(overflow_dir) = overflow_dir else {
        return;
    };
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);

    for group in recent_tool_result_groups(messages, keep_recent_groups) {
        let mut precision_results: Vec<(usize, String)> = group
            .into_iter()
            .filter_map(|idx| {
                let tool_name = messages[idx]
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_tool_name.get(id))?;
                tool_history_policy(tool_name)
                    .counts_toward_precision_inline_budget()
                    .then(|| (idx, tool_name.clone()))
            })
            .collect();

        // 已外溢成 stub / 空文本的结果不再计入可用预算：它们已被固定协议文本占用，
        // 记入会让 total_chars 虚高，导致同组其它结果被多外溢。
        let mut total_chars = precision_results
            .iter()
            .map(|(idx, _)| value_to_string(&messages[*idx].content))
            .filter(|text| !text.trim().is_empty() && !is_preserved_tool_overflow_stub(text))
            .map(|text| text.chars().count())
            .sum::<usize>();
        precision_results.sort_unstable_by_key(|(idx, _)| {
            std::cmp::Reverse(value_to_string(&messages[*idx].content).chars().count())
        });

        // 优先外溢最大的结果，以最少的 stub 腾出足够空间；其余同组证据仍完整可见。
        for (idx, tool_name) in precision_results {
            if total_chars <= inline_budget {
                break;
            }
            let text = value_to_string(&messages[idx].content);
            if text.trim().is_empty() || is_preserved_tool_overflow_stub(&text) {
                continue;
            }
            // protected（当前 turn）结果默认保持原样以留在上下文内；仅在 Path C 兜底时
            // 允许零压缩外溢到 asset，避免后续被有损截断后原文不可恢复。
            if !allow_overflow_protected
                && messages[idx]
                    .tool_call_id
                    .as_ref()
                    .is_some_and(|id| protected_tool_call_ids.contains(id))
            {
                continue;
            }
            let text_len = text.chars().count();
            // 回读本 session 已归档的 asset（read_file 指向 tool-overflow-compressed/
            // 直接子文件）时复用既有文件，而不是再铸造一个随机名新文件——否则
            // 「外溢→回读→再外溢」会在每次回读时生成新 archive，形成无界链，
            // 模型永远拿不到稳定的内容。与 Path C 的复用逻辑保持一致。
            let tool_call_id = messages[idx].tool_call_id.as_deref();
            let existing_asset_path = tool_call_id
                .and_then(|id| id_to_tool_args.get(id))
                .and_then(|args| {
                    preserved_tool_overflow_path_in_arguments(&tool_name, args, overflow_dir, cwd)
                });
            let (path, wrote_new) = if let Some(path) = existing_asset_path {
                (path, false)
            } else {
                let Some(path) = write_preserved_tool_overflow_file(
                    overflow_dir,
                    tool_call_id,
                    &tool_name,
                    &text,
                ) else {
                    continue;
                };
                (path, true)
            };
            let recall_lines = messages[idx]
                .tool_call_id
                .as_deref()
                .and_then(|id| id_to_tool_args.get(id))
                .map(|args| build_tool_overflow_recall_lines(&tool_name, args))
                .unwrap_or_default();
            let stub = build_preserved_tool_overflow_stub(&path, &tool_name, &text, &recall_lines);
            // 外溢必须严格腾出空间：小结果换成更大的 stub 是膨胀而非压缩，
            // 会虚增预算占用且让模型看不到真实结果（死循环放大因素）。
            // 膨胀时删除刚写的文件、保留原文，与 Path C 的防膨胀守卫一致。
            // 只删除本次新建的文件：复用的 asset 归其它消息的外溢所有，
            // 删除会让那条消息的 stub 悬空（历史会话出现过归档文件被删的故障）。
            let stub_chars = stub.chars().count();
            if stub_chars >= text_len {
                if wrote_new {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
            messages[idx].content = Value::String(stub);
            // 记账必须计入 stub 自身占用：只减原文会低估剩余占用，让预算判定失真。
            total_chars = total_chars
                .saturating_sub(text_len)
                .saturating_add(stub_chars);
        }
    }
}

/// Path C 的全局兜底：跨工具组收集所有受保护且禁止有损压缩的结果，并优先外溢
/// 最大的原文，直到整个请求回到 hard target 或没有可外溢候选。候选放宽到
/// `!allows_lossy_compress()`（而非仅高精度 inline 预算工具），使 `task_wait` 等
/// 聚合但禁压缩的大结果也走无损外溢+file 指针，避免落到后续有损截断被静默丢真相。
pub(super) fn spill_protected_precision_to_fit(
    messages: &mut [Message],
    hard_target_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) -> usize {
    let Some(overflow_dir) = overflow_dir else {
        return 0;
    };
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let mut candidates = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, message)| {
            let id = message.tool_call_id.as_ref()?;
            if !protected_tool_call_ids.contains(id) {
                return None;
            }
            let tool_name = id_to_tool_name.get(id)?;
            let text = value_to_string(&message.content);
            // 当前轮若直接回读本 session 的 archive，Path C 不能再复制一份原文。
            // 复用原 asset 的指针既保住 hard target，又避免「外溢→回读→再外溢」循环。
            let existing_asset_path = id_to_tool_args.get(id).and_then(|args| {
                preserved_tool_overflow_path_in_arguments(&tool_name, args, overflow_dir, cwd)
            });
            (!text.trim().is_empty()
                && !is_preserved_tool_overflow_stub(&text)
                && !tool_history_policy(tool_name).allows_lossy_compress())
            .then(|| {
                (
                    idx,
                    tool_name.clone(),
                    existing_asset_path,
                    text.chars().count(),
                )
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(_, _, _, chars)| std::cmp::Reverse(*chars));

    let mut spilled = 0usize;
    for (idx, tool_name, existing_asset_path, _) in candidates {
        if super::messages_total_chars(messages) <= hard_target_chars {
            break;
        }
        let text = value_to_string(&messages[idx].content);
        let (path, wrote_new_archive) = if let Some(path) = existing_asset_path {
            (path, false)
        } else {
            let Some(path) = write_preserved_tool_overflow_file(
                overflow_dir,
                messages[idx].tool_call_id.as_deref(),
                &tool_name,
                &text,
            ) else {
                continue;
            };
            (path, true)
        };
        let recall_lines = messages[idx]
            .tool_call_id
            .as_deref()
            .and_then(|id| id_to_tool_args.get(id))
            .map(|args| build_tool_overflow_recall_lines(&tool_name, args))
            .unwrap_or_default();
        let full_stub = build_preserved_tool_overflow_stub(&path, &tool_name, &text, &recall_lines);
        let replacement = if full_stub.chars().count() < text.chars().count() {
            full_stub
        } else if let Some(pointer_stub) = minimize_overflow_stub_to_pointer(&full_stub) {
            if pointer_stub.chars().count() < text.chars().count() {
                pointer_stub
            } else {
                if wrote_new_archive {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
        } else {
            if wrote_new_archive {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        };
        messages[idx].content = Value::String(replacement);
        spilled += 1;
    }
    spilled
}

pub(super) fn build_tool_call_name_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.name.clone());
        }
    }
    out
}

pub(super) fn build_tool_call_arguments_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.arguments.clone());
        }
    }
    out
}

/// 返回 `read_file` 直接读取的本 session 工具溢出归档路径。
///
/// 对一般工具结果必须落盘，以保住 hard target；但 archive 本身已经是
/// runtime 写入的稳定资产，再复制一次只会让模型在回读大文件时持续产生新 archive。
/// 仅复用 `tool-overflow-compressed` 的直接子文件：session asset 根中的 `tmp`、
/// checkpoint 等文件可在同一 session 后续被写入；若直接保留其指针，历史结果会
/// 随源文件变化。两端 canonicalize 后再判断目录，避免 `..` 或 symlink 越过边界。
fn preserved_tool_overflow_read_file_path(arguments: &str, overflow_dir: &Path) -> Option<PathBuf> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    let raw_path = value_string_from_keys(&args, &["file_path", "path", "filePath"])?;
    archived_asset_path_from_raw(&raw_path, overflow_dir)
}

/// 统一识别「读取本 session 归档 asset」的直接文件路径，供两条主路径与 Path C 复用
/// 既有 archive，避免「外溢→回读→再外溢」每次回读都铸造新 uuid 文件、让模型沿
/// 新指针无限重读。
///
/// - `read_file`：`file_path` / `path` / `filePath` 字段；
/// - `execute_command`：`command` 字符串内嵌的归档路径（`cat`/`head`/`grep` 等）。
/// 其余工具不返回路径（复用语义不适用，行为与旧 `read_file` 守卫一致）。
fn preserved_tool_overflow_path_in_arguments(
    tool_name: &str,
    arguments: &str,
    overflow_dir: &Path,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    match tool_name {
        "read_file" => preserved_tool_overflow_read_file_path(arguments, overflow_dir),
        "execute_command" => {
            let command = value_string_from_keys(&args, &["command"])?;
            archived_asset_path_in_command(&command, overflow_dir, cwd)
        }
        _ => None,
    }
}

/// 校验 raw 路径为 `tool-overflow-compressed` 的直接子文件；两端 canonicalize。
fn archived_asset_path_from_raw(raw_path: &str, overflow_dir: &Path) -> Option<PathBuf> {
    let preserved_dir = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .canonicalize()
        .ok()?;
    // 必须与 read_file 共用相同的 relative-path 解析规则；直接 canonicalize 会错误地
    // 以进程 cwd 为基准，忽略 subagent 的 effective_cwd。
    let source_path = FileStore::new(PathBuf::from(raw_path))
        .path()
        .canonicalize()
        .ok()?;
    if !source_path.is_file() || source_path.parent() != Some(preserved_dir.as_path()) {
        return None;
    }
    Some(source_path)
}

/// 从 execute_command 的 `command` 字符串中识别「读取本 session 归档 asset」的路径。
/// 匹配 archive 直接子文件的绝对路径、相对 effective_cwd 的路径或裸文件名
/// （文件名含 uuid，基本唯一）。archive 数量有限，每轮压缩扫描成本可接受。
fn archived_asset_path_in_command(
    command: &str,
    overflow_dir: &Path,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let preserved_dir = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .canonicalize()
        .ok()?;
    for entry in std::fs::read_dir(&preserved_dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        // 与 read_file 路径（archived_asset_path_from_raw）一致：拒绝指向
        // `tool-overflow-compressed` 目录外的 symlink 目标。
        if !canonical.is_file() || canonical.parent() != Some(preserved_dir.as_path()) {
            continue;
        }
        let abs = canonical.to_string_lossy().into_owned();
        if command.contains(&abs) {
            return Some(canonical);
        }
        if let Some(cwd) = &cwd
            && let Ok(rel) = canonical.strip_prefix(cwd)
            && let Some(rel_str) = rel.to_str()
            && command.contains(rel_str)
        {
            return Some(canonical);
        }
        if let Some(name) = canonical.file_name().and_then(|n| n.to_str())
            && command.contains(name)
        {
            return Some(canonical);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn assistant_call(id: &str, name: &str) -> Message {
        assistant_call_args(id, name, "{}")
    }

    fn assistant_call_args(id: &str, name: &str, arguments: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_result(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    #[test]
    fn preserved_tool_overflow_stub_is_not_spilled_again() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-tool-overflow-stub-{}", uuid::Uuid::new_v4()));
        let mut messages = vec![
            assistant_call("old", "read_file"),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let first_stub = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&first_stub));
        let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        assert_eq!(value_to_string(&messages[1].content), first_stub);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_read_file_overflow_stub_keeps_original_target_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-read-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "read_file",
                r#"{"file_path":"src/lib.rs","offset":120,"limit":40}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_file_path: src/lib.rs"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("- original_range: lines=120..159"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("Archived snapshot of an earlier read"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_file_path: src/lib.rs"),
            "anchor: {anchor}"
        );
        assert!(
            anchor.contains("Archived snapshot of an earlier read"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_stub_preview_includes_line_numbered_key_lines() {
        // read_file 输出带 `{:>6}\t` 行号前缀：key_lines 应解析前缀、用真实行号
        // 做 L 标签（而不是被前缀挡住匹配而全部落空），让长文件外溢后仍能按行号定位。
        let content = "\
     1\tuse std::fmt;\n\
     2\t\n\
     3\tpub fn main() {\n\
     4\t    let x = 1;\n\
     5\t}\n\
     6\tfn helper() {}\n\
     7\t//! crate docs\n\
     8\tstruct Foo;\n";
        let preview = build_overflow_content_preview(content);
        assert!(preview.contains("- key_lines (5):"), "preview: {preview}");
        assert!(preview.contains("L1: use std::fmt;"), "preview: {preview}");
        assert!(preview.contains("L3: pub fn main()"), "preview: {preview}");
        assert!(preview.contains("L6: fn helper()"), "preview: {preview}");
        assert!(preview.contains("L8: struct Foo;"), "preview: {preview}");
    }

    #[test]
    fn preserved_execute_command_overflow_stub_keeps_original_command_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-command-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "execute_command",
                r#"{"command":"git log --stat","cwd":"/repo"}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_command: git log --stat"),
            "stub: {stub}"
        );
        assert!(stub.contains("- original_cwd: /repo"), "stub: {stub}");
        assert!(
            stub.contains("Continue from `original_command` / `original_cwd`"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_command: git log --stat"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn legacy_tool_overflow_stub_is_recognized() {
        let legacy = "Output preserved for non-compressible tool `read_file`.\n\
            - file_path: /tmp/result.txt\n\
            - use read_file to inspect exact content.\n\
            Preview (for recall; not exhaustive):";
        assert!(is_preserved_tool_overflow_stub(legacy));
    }

    #[test]
    fn protected_precision_budget_excludes_aggregated_task_results() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-precision-group-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut call = assistant_call("read", "read_file");
        call.tool_calls.as_mut().unwrap().push(ToolCall {
            id: "task".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "task_wait".to_string(),
                arguments: "{}".to_string(),
            },
        });
        let mut messages = vec![
            call,
            tool_result("read", &"r".repeat(1_000)),
            tool_result("task", &"t".repeat(10_000)),
        ];

        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            200,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        assert!(is_preserved_tool_overflow_stub(&value_to_string(
            &messages[1].content
        )));
        assert_eq!(value_to_string(&messages[2].content).len(), 10_000);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn precision_budget_never_expands_small_results_into_larger_stubs() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-precision-group-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut call = assistant_call("small", "read_file");
        call.tool_calls.as_mut().unwrap().push(ToolCall {
            id: "big".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        });
        let mut messages = vec![
            call,
            tool_result("small", &"s".repeat(100)),
            tool_result("big", &"b".repeat(10_000)),
        ];

        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            200,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        // 小结果（100 字符）换成带长路径的 stub 反而膨胀：必须保留原文内联。
        assert_eq!(value_to_string(&messages[1].content), "s".repeat(100));
        // 大结果被外溢成 stub，且 stub 严格短于原文。
        let stub = value_to_string(&messages[2].content);
        assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
        assert!(stub.chars().count() < 10_000, "{stub}");
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn enforce_group_budget_reuses_reread_archive_asset_instead_of_rearchiving() {
        // 回读已归档 asset 的 read_file 结果（跨 turn 后不再是 protected）再次进入
        // group 外溢时，必须复用既有归档文件（stub 指向同一文件），而不是铸造
        // 随机名新文件——否则「外溢→回读→再外溢」每次回读都会生成新 archive，
        // 形成无界链，模型永远拿不到稳定的内容。
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-precision-group-reuse-{}", uuid::Uuid::new_v4()));
        // 1) 用 Path C 生成一个归档 asset
        let mut messages = vec![
            assistant_call("spill", "read_file"),
            tool_result("spill", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];
        let mut protected = FxHashSet::default();
        protected.insert("spill".to_string());
        let stub1 = spill_protected_precision_to_fit(
            &mut messages,
            80,
            Some(&overflow_dir),
            None,
            &protected,
        );
        assert!(stub1 > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) 模型回读该归档：结果(1000 字符) 超过 group inline 预算 → 触发 enforce 外溢
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            120,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        // 3) 复用既有 asset：目录仍只有 1 个文件，且 stub 指向同一个 archive_path
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn cap_oversized_reuses_reread_archive_asset_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-cap-reread-reuse-{}", uuid::Uuid::new_v4()));
        // 1) 先由 cap 自身写一个归档 asset
        let mut messages = vec![
            assistant_call("first", "read_file"),
            tool_result("first", &"y".repeat(70_000)),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert!(capped > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) 模型回读归档（正文 70k > 64k hard cap）→ 复用既有文件，不写新文件
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert_eq!(capped, 1);
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_reuses_reread_archive_asset_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-prepare-reread-reuse-{}", uuid::Uuid::new_v4()));
        // 1) 先由 prepare 写一个归档 asset（旧 read_file 结果，超出 480 阈值、非尾窗）
        let mut messages = vec![
            assistant_call("first", "read_file"),
            tool_result("first", &"z".repeat(2_000)),
        ];
        prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) 模型回读归档，结果再进 prepare（非保护、非尾窗）→ 复用既有文件
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_spill_is_deterministic_across_reprojections() {
        // P1 回归：同一条 canonical tool 结果在两次独立的投影里被外溢时，必须映射到
        // 同一个确定性归档文件，而不是每轮铸造一个随机名新副本（旧行为在单会话里
        // 造成 368 个文件仅 211 份唯一内容的无界膨胀）。两次投影用**不同**的
        // overflow_dir 无法体现幂等，因此复用同一个 dir 模拟同一 session 的逐轮压缩。
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-prepare-deterministic-spill-{}",
            uuid::Uuid::new_v4()
        ));
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        // 单行长结果：预览经行截断后 stub 显著小于原文 → 确实外溢（不触发防膨胀守卫）。
        let big = "b".repeat(4_000);
        let build = || {
            vec![
                assistant_call("spill", "read_file"),
                tool_result("spill", &big),
                assistant_call("recent", "read_file"),
                tool_result("recent", "recent result"),
            ]
        };

        let mut first = build();
        prepare_tool_messages_structured(
            &mut first,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let first_stub = value_to_string(&first[1].content);
        assert!(is_preserved_tool_overflow_stub(&first_stub), "{first_stub}");
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);

        // 第二次投影：同一条 canonical 结果（同 tool_call_id + 同正文）再次被压缩。
        // 确定性命名 → 命中既有文件、不新增副本，stub 文本逐轮稳定。
        let mut second = build();
        prepare_tool_messages_structured(
            &mut second,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let second_stub = value_to_string(&second[1].content);
        assert_eq!(
            second_stub, first_stub,
            "重投影后 stub 文本必须稳定（prompt cache 不断裂）"
        );
        assert_eq!(
            std::fs::read_dir(&archive_dir).unwrap().count(),
            1,
            "同一结果重复外溢不得铸造新归档文件"
        );

        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_keeps_small_multiline_result_inline() {
        // P2 回归：一个几百字节的多行 grep 结果（> max_chars_per_msg 但很小），
        // 换成带完整 head/tail 预览的 stub 反而更大。防膨胀守卫必须保留原文内联，
        // 不写归档文件——否则模型看到「已卸载，请重读」而反复回读（会话 9f4d0fae 的
        // 「读结果一直被归档成 stub」正是此路径：673 字符 grep 被换成更大的 stub）。
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-prepare-small-inline-{}", uuid::Uuid::new_v4()));
        // 20 行、每行 ~30 字符 ≈ 600 字符：超过 max_chars_per_msg=480，但整段会被
        // 预览逐字包含，stub 必然不小于原文。
        let grep_like = (0..20)
            .map(|i| format!("src/bin/ai/mod.rs:{i}: use crate::ai::x;"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(grep_like.chars().count() > 480);
        let mut messages = vec![
            assistant_call("grep", "execute_command"),
            tool_result("grep", &grep_like),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );

        let content = value_to_string(&messages[1].content);
        assert_eq!(content, grep_like, "小的多行精确结果必须保留原文内联");
        assert!(
            !is_preserved_tool_overflow_stub(&content),
            "不应被换成 stub"
        );
        // 没有归档文件被写出（膨胀时删除新建文件；此处应压根没写成功的净收益）。
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archived = std::fs::read_dir(&archive_dir)
            .map(|d| d.count())
            .unwrap_or(0);
        assert_eq!(archived, 0, "膨胀结果不得留下归档文件");

        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn cap_reuses_execute_command_cat_archive_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-cap-cat-reuse-{}", uuid::Uuid::new_v4()));
        // 1) 先由 cap 写一个 execute_command 归档 asset
        let mut messages = vec![
            assistant_call("run", "execute_command"),
            tool_result("run", &"log line\n".repeat(30_000)),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert!(capped > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) 模型用 `cat <archive>` 回读归档正文（> hard cap）→ 复用既有文件
        let run_args = serde_json::json!({
            "command": format!("cat {}", archive_path.to_string_lossy()),
            "pty": false,
        });
        let mut messages = vec![
            assistant_call_args("re-cat", "execute_command", &run_args.to_string()),
            tool_result("re-cat", &raw),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert_eq!(capped, 1);
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn path_c_spills_all_protected_precision_groups_without_recent_group_cap() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = Vec::new();
        let mut protected = FxHashSet::default();
        for index in 0..8 {
            let id = format!("read-{index}");
            protected.insert(id.clone());
            messages.push(assistant_call(&id, "read_file"));
            messages.push(tool_result(&id, &"line of exact evidence\n".repeat(600)));
        }

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        // 覆盖 Path C 的后半段：spill 后仍超预算时会进入 emergency cap。所有
        // preserved stub 必须先缩成不可截断的最小指针，不能再被通用 head/tail 截断。
        assert!(super::super::messages_total_chars(&messages) > 4_000);
        super::super::emergency_cap_messages_to_fit(
            &mut messages,
            4_000,
            160,
            Some(&overflow_dir),
            &protected,
        );

        assert_eq!(spilled, 8);
        let stubs = messages
            .iter()
            .filter_map(|message| {
                let content = value_to_string(&message.content);
                is_preserved_tool_overflow_stub(&content).then_some(content)
            })
            .collect::<Vec<_>>();
        assert_eq!(stubs.len(), 8);
        for stub in stubs {
            let file_path = stub
                .lines()
                .find_map(|line| line.strip_prefix("- file_path: "))
                .expect("minimal overflow stub must retain file_path");
            assert!(Path::new(file_path).is_file());
            assert!(!stub.contains("Preview ("));
        }
        assert!(super::super::messages_total_chars(&messages) <= 4_000);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn path_c_reuses_reread_session_asset_instead_of_rearchiving_it() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let test_root =
            std::env::temp_dir().join(format!("ai-reread-session-asset-{}", uuid::Uuid::new_v4()));
        let effective_cwd = test_root.join("workspace");
        let overflow_dir = effective_cwd.join("session-assets");
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        std::fs::create_dir_all(&archive_dir).unwrap();
        let archive_path = archive_dir.join("prior-read.txt");
        let content = "previously preserved evidence\n".repeat(800);
        std::fs::write(&archive_path, &content).unwrap();
        let relative_archive_path = archive_path
            .strip_prefix(&effective_cwd)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let arguments = serde_json::json!({
            "file_path": relative_archive_path,
            "offset": 1,
            "limit": 10_000,
        })
        .to_string();
        let mut protected = FxHashSet::default();
        protected.insert("reread".to_string());
        let mut messages = vec![
            assistant_call_args("reread", "read_file", &arguments),
            tool_result("reread", &content),
        ];

        let spilled =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
                spill_protected_precision_to_fit(
                    &mut messages,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected,
                )
            });

        assert_eq!(spilled, 1);
        let stub = value_to_string(&messages[1].content);
        let file_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .expect("reused stub must retain the existing archive pointer");
        assert_eq!(Path::new(file_path), archive_path.canonicalize().unwrap());
        assert!(stub.contains("- original_range: lines=1..10000"));
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        assert_eq!(std::fs::read_to_string(&archive_path).unwrap(), content);
        let _ = std::fs::remove_dir_all(test_root);
    }

    #[test]
    fn path_c_snapshots_mutable_session_temp_asset_instead_of_reusing_it() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let test_root =
            std::env::temp_dir().join(format!("ai-reread-session-temp-{}", uuid::Uuid::new_v4()));
        let effective_cwd = test_root.join("workspace");
        let overflow_dir = effective_cwd.join("session-assets");
        let temp_dir = overflow_dir.join("tmp");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let temp_path = temp_dir.join("mutable.txt");
        let content = "temporary evidence before mutation\n".repeat(800);
        std::fs::write(&temp_path, &content).unwrap();
        let relative_temp_path = temp_path
            .strip_prefix(&effective_cwd)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let arguments = serde_json::json!({
            "file_path": relative_temp_path,
            "offset": 1,
            "limit": 10_000,
        })
        .to_string();
        let mut protected = FxHashSet::default();
        protected.insert("reread".to_string());
        let mut messages = vec![
            assistant_call_args("reread", "read_file", &arguments),
            tool_result("reread", &content),
        ];

        let spilled =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
                spill_protected_precision_to_fit(
                    &mut messages,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected,
                )
            });

        assert_eq!(spilled, 1);
        let stub = value_to_string(&messages[1].content);
        let snapshot_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .map(PathBuf::from)
            .expect("mutable session file must be snapshotted into an overflow archive");
        assert_ne!(snapshot_path, temp_path.canonicalize().unwrap());
        assert!(snapshot_path.starts_with(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR)));
        assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);

        std::fs::write(&temp_path, "temporary evidence after mutation\n").unwrap();
        assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);
        let _ = std::fs::remove_dir_all(test_root);
    }

    #[test]
    fn path_c_spills_aggregated_task_wait_result_losslessly() {
        // task_wait 禁止有损压缩但不占 inline 预算；Path C 全局兜底必须把它无损外溢
        // 并留下 file 指针，而不是排除在候选之外、任由后续有损截断丢失聚合真相。
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-taskwait-{}",
            uuid::Uuid::new_v4()
        ));
        let mut protected = FxHashSet::default();
        protected.insert("wait".to_string());
        let mut messages = vec![
            assistant_call("wait", "task_wait"),
            tool_result("wait", &"aggregated subagent conclusion\n".repeat(600)),
        ];

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        assert_eq!(spilled, 1, "task_wait 大结果应被 Path C 无损外溢");
        let stub = value_to_string(&messages[1].content);
        assert!(
            is_preserved_tool_overflow_stub(&stub),
            "外溢后应替换为 overflow stub"
        );
        let file_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .expect("overflow stub 必须保留可召回的 file_path 指针");
        assert!(Path::new(file_path.trim()).is_file(), "外溢原文必须落盘");
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn path_c_does_not_expand_short_protected_results_into_stubs() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-short-{}",
            uuid::Uuid::new_v4()
        ));
        let mut protected = FxHashSet::default();
        protected.insert("read-short".to_string());
        let mut messages = vec![
            assistant_call("read-short", "read_file"),
            tool_result("read-short", "ok"),
        ];
        let before = super::super::messages_total_chars(&messages);

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        assert_eq!(spilled, 0);
        assert_eq!(value_to_string(&messages[1].content), "ok");
        assert_eq!(super::super::messages_total_chars(&messages), before);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// 构造一条「首次外溢」形态的 stub（含多行 Preview 正文），用于折叠测试。
    fn overflow_stub_with_preview(file_path: &str, tool_name: &str) -> String {
        let full = (0..40)
            .map(|i| format!("line {i}: some content"))
            .collect::<Vec<_>>()
            .join("\n");
        build_preserved_tool_overflow_stub(Path::new(file_path), tool_name, &full, &[])
    }

    #[test]
    fn collapse_overflow_stub_to_anchor_drops_preview_keeps_file_path() {
        let stub = overflow_stub_with_preview("/tmp/session/read-abc.txt", "read_file");
        // 前置条件：首次 stub 确实带 Preview 正文。
        assert!(stub.contains("Preview ("));

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("should collapse");
        // 预览正文被丢弃。
        assert!(!anchor.contains("Preview ("));
        // file_path 保留。
        assert!(anchor.contains("- file_path: /tmp/session/read-abc.txt"));
        // 工具名保留（新格式用 "Output preserved for tool"）。
        assert!(anchor.contains("Output preserved for tool `read_file`"));
        // read_file 类型的归档有"通常无需重新读取"提示（而非旧的诱导式 "use read_file"）。
        assert!(anchor.contains("Archived snapshot of an earlier read"));
        // 仍是合法 stub（前缀不变），后续压缩链继续按 stub 豁免识别。
        assert!(is_preserved_tool_overflow_stub(&anchor));
        // 体量骤降。
        assert!(anchor.len() < stub.len());
    }

    #[test]
    fn age_out_overflow_stub_previews_is_idempotent() {
        let stub = overflow_stub_with_preview("/tmp/session/read-xyz.txt", "read_file");
        // 两条 user 轮，让 stub 落在保护尾窗之外（retained_turn_start 之前）。
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("s", "read_file"),
            tool_result("s", "placeholder"),
            user_msg("q2"),
            user_msg("q3"),
        ];
        messages[2].content = Value::String(stub);

        age_out_overflow_stub_previews(&mut messages, 1);
        let after_first = value_to_string(&messages[2].content);
        assert!(!after_first.contains("Preview ("));

        // 再跑一次：已是锚点形态，内容不得再变（防 stub->stub 抖动）。
        age_out_overflow_stub_previews(&mut messages, 1);
        assert_eq!(value_to_string(&messages[2].content), after_first);
    }

    #[test]
    fn age_out_overflow_stub_previews_respects_protected_tail() {
        // 早期 stub（尾窗外）与近端 stub（尾窗内）各一条。
        let early = overflow_stub_with_preview("/tmp/session/early.txt", "read_file");
        let recent = overflow_stub_with_preview("/tmp/session/recent.txt", "read_file");
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("early", "read_file"),
            tool_result("early", "placeholder"),
            user_msg("q2"),
            assistant_call("recent", "read_file"),
            tool_result("recent", "placeholder"),
        ];
        messages[2].content = Value::String(early);
        messages[5].content = Value::String(recent.clone());

        // 保护最近 1 个 user 轮（q2 起）：早期 stub 折叠，尾窗内 recent 保留完整预览。
        age_out_overflow_stub_previews(&mut messages, 1);
        assert!(!value_to_string(&messages[2].content).contains("Preview ("));
        assert_eq!(value_to_string(&messages[5].content), recent);
        assert!(value_to_string(&messages[5].content).contains("Preview ("));
    }
}

/// 「读取/检索」类工具的输出零压缩（不行裁剪、不去重折叠、不整组删除），
/// 超阈值时只做"零压缩外溢到会话文件 + 留指针 stub"。这类输出复现代价高，
/// 一旦被压掉模型就会反复重跑同一次检索（典型失忆/原地打转症状）。
///
/// 现在改为查询工具自身声明的历史保留策略
/// （`ToolHistoryPolicyRegistration`，见各工具注册文件），而非在此硬编码
/// 工具名列表。默认未注册的工具允许有损压缩；只有显式声明
/// `lossy_compress: Never` 的工具（`read_file` / 检索类 / `execute_command`）
/// 返回 true。`plan` 不再禁止有损压缩：最新一版由最近工具组保护窗口完整保留，
/// 旧版可摘要压缩以释放上下文。注意：这与「是否允许 LLM 裁剪」是正交维度——见 `llm_prune.rs`。
pub(super) fn is_non_compressible_tool(tool_name: &str) -> bool {
    !crate::ai::tools::registry::common::tool_history_policy(tool_name).allows_lossy_compress()
}

/// 为尚未外溢的高精度工具结果规划确定性 asset 和带 `file_path` 的稳定 stub。
///
/// 本函数只生成 [`PlannedArchiveWrite`]，不触碰文件系统。调用方确认采用整个 fold
/// 方案后再统一 commit，避免被拒绝的 speculative fold 留下磁盘副作用。已有 stub
/// 直接复用，不产生新写入。
pub(super) fn plan_noncompressible_tool_result_for_fold(
    overflow_dir: Option<&Path>,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
    recall_lines: &[String],
) -> Option<(String, Option<PlannedArchiveWrite>)> {
    if is_preserved_tool_overflow_stub(content) {
        return Some((content.to_string(), None));
    }
    let overflow_dir = overflow_dir?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let identity = format!("{tool_call_id}\0{content}");
    let digest = content_sha256_hex(identity.as_bytes());
    let path = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .join(format!("folded-{safe_tool}-{}.txt", &digest[..24]));
    let stub = build_preserved_tool_overflow_stub(&path, tool_name, content, recall_lines);
    Some((
        stub,
        Some(PlannedArchiveWrite::new(path, content.to_string())),
    ))
}

/// LLM prune 路径使用的即时稳定归档；与 fold 的两阶段计划相互独立。文件名由
/// `tool_call_id` 确定性派生（而非随机 uuid + 时间戳）。
///
/// LLM 引导裁剪（`llm_prune::apply_pruning`）作用于每次模型请求前的临时 `messages`
/// 投影：同一条 canonical tool 消息在后续 turn 重建投影时会被再次裁剪。若沿用随机
/// 文件名，则每轮都会写出新副本、生成不同 stub 文本 → prompt
/// cache 从该点断裂 + 磁盘副本单调膨胀。用确定性文件名后，同一 `tool_call_id`
/// 的归档幂等：文件已存在则跳过写入，stub 文本逐轮稳定。
pub(super) fn preserve_pruned_tool_result_stable(
    overflow_dir: Option<&Path>,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
    recall_lines: &[String],
) -> Option<String> {
    if is_preserved_tool_overflow_stub(content) {
        return Some(content.to_string());
    }
    let path = overflow_dir.and_then(|dir| {
        write_preserved_tool_overflow_file_stable(dir, tool_call_id, tool_name, content)
    })?;
    Some(build_preserved_tool_overflow_stub(
        &path,
        tool_name,
        content,
        recall_lines,
    ))
}

/// 以 `tool_call_id` 派生确定性文件名写出归档；文件已存在则直接复用，不重复写。
fn write_preserved_tool_overflow_file_stable(
    overflow_dir: &Path,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let safe_id = sanitize_overflow_name_component(tool_call_id);
    let file_name = format!("pruned-{safe_tool}-{safe_id}.txt");
    let path = dir.join(file_name);
    // 幂等：内容不随轮次变化，已存在则不重复写盘（也保住 prompt cache 稳定）。
    if !path.exists() {
        std::fs::write(&path, content).ok()?;
    }
    Some(path)
}

/// 把工具名 / id 归一成仅含字母数字与 `-`/`_` 的安全文件名片段。
fn sanitize_overflow_name_component(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
}

/// 非 speculative 的即时外溢路径。文件名由 `(tool_call_id, content)` 确定性派生
/// （而非随机 uuid + 时间戳）：同一条 canonical tool 结果在后续 turn 重建投影时会被
/// 反复外溢，随机命名会让「每轮压缩铸造一份新副本」无界膨胀（实测单会话 368 个归档
/// 文件仅 211 份唯一内容），且模型沿 stub 回读时指针每轮漂移、永远拿不到稳定内容。
/// 确定性命名后同一结果幂等映射到同一文件：已存在则复用、不重复写盘，stub 文本逐轮
/// 稳定（与 fold 的 `folded-`、prune 的 `pruned-` 归档命名策略一致）。
/// fold 不得调用本函数；fold 的写入统一由 `PlannedArchiveWrite` 在候选被采纳后提交。
fn write_preserved_tool_overflow_file(
    overflow_dir: &Path,
    tool_call_id: Option<&str>,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let identity = format!("{}\0{content}", tool_call_id.unwrap_or(""));
    let digest = content_sha256_hex(identity.as_bytes());
    let path = dir.join(format!("spilled-{safe_tool}-{}.txt", &digest[..24]));
    // 幂等：内容不随轮次变化，已存在则不重复写盘（也保住 prompt cache 稳定）。
    if !path.exists() {
        std::fs::write(&path, content).ok()?;
    }
    Some(path)
}

fn build_preserved_tool_overflow_stub(
    path: &Path,
    tool_name: &str,
    full_content: &str,
    recall_lines: &[String],
) -> String {
    // 仍把全文外溢到磁盘以控制上下文体积，但在 stub 内保留 head+tail 预览，
    // 让后续 turn 拥有"召回锚点"——模型据此判断是否真的需要重新 read_file，
    // 避免早期读到的代码被搬走后出现"失忆/反复重读"。
    // 提示文案保持中性：明确告知"仅在需要完整内容时才读取"，防止 LLM 看到
    // file_path 就无条件重读导致外溢→重读→再外溢的无限循环。
    let preview = build_overflow_content_preview(full_content);
    let tool_hint = preserved_tool_overflow_hint(tool_name, recall_lines);
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {}",
        path.display(),
    );
    for line in recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(tool_hint);
    out.push('\n');
    out.push_str(&preview);
    out
}

fn preserved_tool_overflow_hint(tool_name: &str, recall_lines: &[String]) -> &'static str {
    let has_original_file_path = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_file_path: "));
    let has_original_command = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_command: "));
    match tool_name {
        "read_file" if has_original_file_path => {
            "Archived snapshot of an earlier read. `original_range` marks lines already covered - for current content, read `original_file_path` past that range (identical re-reads are deduped); read `file_path` only for the exact historical output."
        }
        "read_file" => {
            "Archived snapshot of an earlier read. Read `file_path` only if the preview is insufficient and you need the exact output; identical re-reads are deduped."
        }
        "execute_command" if has_original_command => {
            "Archived command output. Continue from `original_command` / `original_cwd`; `file_path` is a text archive, not a source file - read it only for the full log."
        }
        _ => "Archived output; `file_path` holds the full text. Read it only if the preview is insufficient.",
    }
}

pub(super) fn build_tool_overflow_recall_lines(tool_name: &str, arguments: &str) -> Vec<String> {
    let Ok(args) = serde_json::from_str::<Value>(arguments) else {
        return Vec::new();
    };

    match tool_name {
        "read_file" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(path) = value_string_from_keys(&args, &["file_path", "path", "filePath"]) {
                lines.push(format!(
                    "- original_file_path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }

            if let Some((label, range)) = read_file_range_summary(&args) {
                lines.push(format!("- original_range: {label}={range}"));
            }
            lines
        }
        "tree" => value_string_from_keys(&args, &["path"])
            .map(|path| {
                vec![format!(
                    "- original_path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                )]
            })
            .unwrap_or_default(),
        "execute_command" | "run_command" | "shell" | "bash" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(command) = value_string_from_keys(&args, &["command"]) {
                lines.push(format!(
                    "- original_command: {}",
                    truncate_to_chars(&normalize_whitespace(&command), 720)
                ));
            }
            if let Some(cwd) = value_string_from_keys(&args, &["cwd"]) {
                let cwd = normalize_whitespace(&cwd);
                if !cwd.is_empty() {
                    lines.push(format!("- original_cwd: {}", truncate_to_chars(&cwd, 240)));
                }
            }
            lines
        }
        _ => Vec::new(),
    }
}

fn value_string_from_keys(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(|value| value.to_string())
}

fn read_file_range_summary(args: &Value) -> Option<(&'static str, String)> {
    let start_line = args.get("startLine").and_then(Value::as_u64);
    let end_line = args.get("endLine").and_then(Value::as_u64);
    if let (Some(start_line), Some(end_line)) = (start_line, end_line) {
        return Some(("lines", format!("{start_line}..{end_line}")));
    }

    let offset = args.get("offset").and_then(Value::as_u64);
    let limit = args.get("limit").and_then(Value::as_u64);
    match (offset, limit) {
        (Some(offset), Some(limit)) if limit > 0 => Some((
            "lines",
            format!("{offset}..{}", offset + limit.saturating_sub(1)),
        )),
        (Some(offset), _) => Some(("offset", offset.to_string())),
        _ => None,
    }
}

pub(super) fn is_preserved_tool_overflow_stub(text: &str) -> bool {
    let text = text.trim_start();
    if text.starts_with(PRESERVED_TOOL_OVERFLOW_STUB_PREFIX) {
        return text.contains("\n- file_path: ");
    }
    // Legacy formats (older sessions):
    // - "Output preserved for non-compressible tool `..."  (pre-refactor)
    // - "Output preserved for tool `..."                   (new format)
    if (text.starts_with(LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX)
        || text.starts_with("Output preserved for tool `"))
        && text.contains("\n- file_path: ")
    {
        return true;
    }
    false
}

/// 把一条已外溢的 tool overflow stub 的 head+tail 预览体收敛为「单行召回锚点」
/// （仅保留 `file_path:` 指针 + 回读提示，丢弃 `Preview (...)` 及其后所有行）。
///
/// 老 stub 的预览在长会话里单调累积（真实案例：800 条 × ~1KB ≈ 849KB），而
/// `file_path` 才是模型精确回读的唯一必要信息——预览只是「首次召回锚点」，一旦
/// 该 stub 已远离当前工作焦点，预览正文的边际价值趋近于 0。收敛后每条从 ~1KB
/// 降到 ~200 字符，召回能力零损失（仍可 read_file 回读原文）。
///
/// 解析失败（无法定位 file_path 或工具名）返回 `None`，保持原文不动，绝不破坏。
/// 对「已经是锚点形态」（不含 `Preview (` 段）的 stub 亦返回 `None`，保证幂等、
/// 不产生 stub→stub 抖动。
fn collapse_overflow_stub_to_anchor(text: &str) -> Option<String> {
    if !is_preserved_tool_overflow_stub(text) {
        return None;
    }
    // 已是锚点形态（无预览段）：幂等，返回 None 表示无需改写。
    if !text.contains("Preview (") {
        return None;
    }
    // 支持新旧两种格式解析 tool_name
    let tool_name = text
        .split_once("non-compressible tool `")
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(name, _)| name.to_string())
        .or_else(|| {
            text.split_once("Output preserved for tool `")
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(name, _)| name.to_string())
        })?;
    let file_path = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(str::trim)
        .filter(|p| !p.is_empty())?;
    let recall_lines = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("- original_"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let tool_hint = if tool_name == "read_file" {
        if recall_lines
            .iter()
            .any(|line| line.starts_with("- original_file_path: "))
        {
            "Archived snapshot of an earlier read. Read `original_file_path` for current content, `file_path` only for the exact historical output."
        } else {
            "Archived snapshot of an earlier read; read `file_path` only for the exact historical output."
        }
    } else if tool_name == "execute_command"
        && recall_lines
            .iter()
            .any(|line| line.starts_with("- original_command: "))
    {
        "Archived command output; usually no re-read needed - continue from `original_command` / `original_cwd`."
    } else {
        "Full output at `file_path`; read it only if needed."
    };
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {file_path}"
    );
    for line in &recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(tool_hint);
    Some(out)
}

/// Path C 的最终硬预算阶段只允许去掉 overflow stub 的预览与召回附注，不能把
/// 唯一的 asset 指针交给通用 head+tail 截断。返回的最小 stub 仍保留协议 marker、
/// 工具名和 `file_path`，因此后续可以精确回读原始证据。
fn minimize_overflow_stub_to_pointer(text: &str) -> Option<String> {
    if !is_preserved_tool_overflow_stub(text) {
        return None;
    }
    let tool_name = text
        .split_once("non-compressible tool `")
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(name, _)| name)
        .or_else(|| {
            text.split_once("Output preserved for tool `")
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(name, _)| name)
        })?;
    let file_path = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(str::trim)
        .filter(|path| !path.is_empty())?;
    Some(format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`.\n\
         - file_path: {file_path}"
    ))
}

pub(super) fn minimize_overflow_stubs_for_hard_budget(messages: &mut [Message]) {
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        let Value::String(text) = &message.content else {
            continue;
        };
        let Some(minimal) = minimize_overflow_stub_to_pointer(text) else {
            continue;
        };
        message.content = Value::String(minimal);
    }
}

pub(super) fn is_preserved_tool_overflow_content(content: &Value) -> bool {
    content
        .as_str()
        .is_some_and(is_preserved_tool_overflow_stub)
}

/// 将「保护尾窗之外」的 overflow stub 预览体老化折叠为单行锚点。仅作用于已外溢
/// 的 tool stub（`is_preserved_tool_overflow_stub`），不碰原始 tool 结果；尾窗内
/// stub 保留完整 head+tail 预览（当前工作焦点仍需要它的召回上下文）。
///
/// 与预算驱动的组折叠互补：即便某条 stub 所在的组因近端保护未被
/// `fold_early_tool_groups` 折叠，其预览正文也会随对话推进老化收敛，防止历史里
/// 上百条早期 read_file 预览单调累积。
pub(super) fn age_out_overflow_stub_previews(
    messages: &mut [Message],
    keep_recent_user_turns: usize,
) {
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for message in messages.iter_mut().take(protected_tail_start) {
        if message.role != "tool" {
            continue;
        }
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some(anchor) = collapse_overflow_stub_to_anchor(text) {
            message.content = Value::String(anchor);
        }
    }
}

/// 为外溢内容生成 head+tail 预览。短内容直接全量保留；长内容保留前后各若干行，
/// 中间用占位行折叠，并标注省略的行数。
fn build_overflow_content_preview(content: &str) -> String {
    const HEAD_LINES: usize = 8;
    const TAIL_LINES: usize = 4;
    const MAX_LINE_CHARS: usize = 200;
    const MAX_KEY_LINES: usize = 20;

    let truncate_line = |line: &str| -> String {
        if line.chars().count() > MAX_LINE_CHARS {
            let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
            format!("{kept} …")
        } else {
            line.to_string()
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let mut out = String::from("Preview (for recall; not exhaustive):\n");
    // 源码/文本类大结果：附带头带行号的结构索引（fn/struct/impl/use/错误等关键行），
    // 与捕获期 overflow stub 的 key_lines 对齐。压缩外溢后模型仍能按行号定位目标
    // 区域，只重读需要的 range，避免大文件（数千行）中段只能盲目重读。
    let key_lines = extract_key_lines(content, MAX_KEY_LINES);
    if !key_lines.is_empty() {
        out.push_str(&format!("- key_lines ({}):\n", key_lines.len()));
        for line in &key_lines {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }
    if total <= HEAD_LINES + TAIL_LINES {
        for line in &lines {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    } else {
        for line in &lines[..HEAD_LINES] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
        out.push_str(&format!(
            "... [{} line(s) omitted; read the file above for full content] ...\n",
            total - HEAD_LINES - TAIL_LINES
        ));
        for line in &lines[total - TAIL_LINES..] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

const MERGED_PRESERVED_USER_STUB_PREFIX: &str = "较早的用户内容已归档（共 ";
const MERGED_PRESERVED_ARCHIVE_DIR_PREFIX: &str = "归档目录: ";

fn parse_merged_preserved_message_stub(text: &str) -> Option<(usize, Vec<String>)> {
    let count = text
        .strip_prefix(MERGED_PRESERVED_USER_STUB_PREFIX)?
        .split_once(" 条")?
        .0
        .parse::<usize>()
        .ok()?;
    if count == 0 {
        return None;
    }

    let mut dirs = Vec::new();
    for line in text.lines() {
        // 兼容首版合并 stub 使用的「归档文件」字段；该字段实际存放目录。
        let Some(dir) = line
            .strip_prefix(MERGED_PRESERVED_ARCHIVE_DIR_PREFIX)
            .or_else(|| line.strip_prefix("归档文件: "))
            .map(str::trim)
            .filter(|dir| !dir.is_empty())
        else {
            continue;
        };
        if !dirs.iter().any(|existing| existing == dir) {
            dirs.push(dir.to_string());
        }
    }
    (!dirs.is_empty()).then_some((count, dirs))
}

fn build_merged_preserved_message_stub(count: usize, dirs: &[String]) -> String {
    let mut merged =
        format!("较早的用户内容已归档（共 {count} 条，原文零压缩保存在会话归档目录）。\n");
    for dir in dirs {
        merged.push_str(MERGED_PRESERVED_ARCHIVE_DIR_PREFIX);
        merged.push_str(dir);
        merged.push('\n');
    }
    merged.push_str(
        "这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖较早用户原文且现有摘要不足时，逐个使用 tree 列出上述归档目录，再按时间戳和类型定位 JSON 文件，最后使用 read_file 读取具体文件；不要对目录直接调用 read_file。",
    );
    merged
}

/// 将保护尾窗之外的 user/image 外溢 stub 合并为一条带归档目录的指针。
///
/// user/image stub 是 role=user 的占位消息：`first_trim_candidate` / truncate /
/// emergency cap / tool-only 老化折叠都不会再触碰它们，长会话（尤其图片消息按
/// 名义成本计费后）会让 stub 单调累积且没有任何收敛路径。把旧 stub 折叠成
/// 单条合并指针后，占位开销从 O(N) 收敛到 O(1)；原文仍在磁盘零压缩保存，
/// 目录 + 时间戳命名可通过 tree + read_file 回读。只合并保护尾窗之外的 stub，最近几轮保持
/// 逐条指针以便精确召回。
pub(super) fn merge_old_user_overflow_stubs(
    messages: &mut Vec<Message>,
    keep_recent_user_turns: usize,
) {
    const MERGE_MIN_STUB_COUNT: usize = 4;

    // 后续 mid-turn 摘要仍会按最近 2/3 个真实 user 边界切分。即使当前预算已把
    // keep_recent_user_turns 降到 1，也不能把这些结构边界一并折叠进 internal_note，
    // 否则 retained_turn_start 会误判为“历史不足”，整段旧历史将无法进入摘要。
    let structural_tail_turns =
        keep_recent_user_turns.max(KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX);
    let protected_tail_start = retained_turn_start(messages, structural_tail_turns);
    let mut stub_indices = Vec::new();
    let mut merged_stub_count = 0usize;
    let mut single_stub_count = 0usize;
    let mut archived_message_count = 0usize;
    let mut dirs: Vec<String> = Vec::new();
    for (idx, message) in messages.iter().take(protected_tail_start).enumerate() {
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some((count, merged_dirs)) = parse_merged_preserved_message_stub(text) {
            stub_indices.push(idx);
            merged_stub_count += 1;
            archived_message_count = archived_message_count.saturating_add(count);
            for dir in merged_dirs {
                if !dirs.iter().any(|existing| existing == &dir) {
                    dirs.push(dir);
                }
            }
            continue;
        }
        if message.role != "user" {
            continue;
        }
        let Some((_kind, file_path)) = parse_preserved_message_stub(text) else {
            continue;
        };
        stub_indices.push(idx);
        single_stub_count += 1;
        archived_message_count = archived_message_count.saturating_add(1);
        if let Some(parent) = Path::new(&file_path).parent() {
            let dir = parent.to_string_lossy().into_owned();
            if !dirs.iter().any(|d| d == &dir) {
                dirs.push(dir);
            }
        }
    }
    // 尚无合并指针时至少积累 4 条才折叠；已有合并指针后，新老 stub 都并回同一条，
    // 避免每新增 4 条就永久多出一个合并指针，重新退化为 O(N)。
    // 旧版快照可能已含 role=user 的合并指针（修复前生成）。即使本轮无需重新折叠
    // （只有一条合并指针且无新增单条 stub），也必须把已有合并指针的角色迁移为
    // internal_note，否则 retained_turn_start 会继续把它当成真实 user 边界，
    // 污染后续摘要切点。
    for &idx in &stub_indices {
        let is_merged = match &messages[idx].content {
            Value::String(text) => parse_merged_preserved_message_stub(text).is_some(),
            _ => false,
        };
        if is_merged && messages[idx].role != ROLE_INTERNAL_NOTE {
            messages[idx].role = ROLE_INTERNAL_NOTE.to_string();
        }
    }
    if dirs.is_empty()
        || (merged_stub_count == 0 && single_stub_count < MERGE_MIN_STUB_COUNT)
        || (merged_stub_count == 1 && single_stub_count == 0)
    {
        return;
    }

    let merged = build_merged_preserved_message_stub(archived_message_count, &dirs);

    // 从后往前删除，避免下标失效；合并指针写到第一条 stub 的 Message 上。
    // 它描述的是运行时归档元数据，不是新的用户请求；若继续保留 `user` 角色，
    // 删除其余 stub 后会伪造轮次边界，使后续 tail/摘要切分把多轮旧消息误判
    // 成一个近期用户轮次。
    for &idx in stub_indices.iter().skip(1).rev() {
        messages.remove(idx);
    }
    messages[stub_indices[0]].role = ROLE_INTERNAL_NOTE.to_string();
    messages[stub_indices[0]].content = Value::String(merged);
}

pub(super) fn is_preserved_user_or_image_stub(text: &str) -> bool {
    parse_merged_preserved_message_stub(text).is_some()
        || parse_preserved_message_stub(text).is_some()
}

fn parse_preserved_message_stub(text: &str) -> Option<(String, String)> {
    if let Some(payload) = text.strip_prefix(PRESERVED_CONTENT_STUB_PREFIX) {
        let value = serde_json::from_str::<Value>(payload).ok()?;
        let kind = value.get("kind")?.as_str()?.to_string();
        let file_path = value.get("file_path")?.as_str()?.to_string();
        return ((kind == "user" || kind == "image") && !file_path.is_empty())
            .then_some((kind, file_path));
    }

    let kind = if text.starts_with("较早的用户图片内容已归档") {
        "image"
    } else if text.starts_with("较早的用户") {
        "user"
    } else {
        return None;
    };
    let file_path = text
        .lines()
        .find_map(|line| line.strip_prefix("归档文件: "))?
        .trim();
    (!file_path.is_empty()).then(|| (kind.to_string(), file_path.to_string()))
}

/// 将内部归档协议转换成模型可理解的上下文说明，同时兼容已经落盘的旧 JSON stub。
pub(in crate::ai) fn normalize_preserved_message_stubs_for_model(messages: &mut [Message]) {
    for message in messages {
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some((count, dirs)) = parse_merged_preserved_message_stub(text) {
            message.content = Value::String(build_merged_preserved_message_stub(count, &dirs));
            // 合并指针是运行时归档元数据，不是用户请求；旧版快照可能仍以
            // role=user 落盘，在此兜底迁移，避免污染 user/assistant 配对。
            message.role = ROLE_INTERNAL_NOTE.to_string();
            continue;
        }
        let Some((kind, file_path)) = parse_preserved_message_stub(text) else {
            continue;
        };
        message.content = Value::String(build_preserved_message_overflow_stub(
            Path::new(&file_path),
            &kind,
        ));
    }
}

fn first_preserved_content_spill_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages, budget);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for (idx, message) in messages.iter().enumerate() {
        if idx >= protected_tail_start {
            break;
        }
        if is_system_like_role(&message.role) || message.role == "tool" {
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            continue;
        }

        let text = value_to_string(&message.content);
        if is_preserved_user_or_image_stub(&text) {
            continue;
        }

        // value_to_string 会把图片 base64 折叠成 "[图片]"，无法反映真实体量。
        // 对图片消息改用原始 content 的序列化长度判断是否需要外溢，与「把大图
        // 搬到会话临时文件」的意图一致；普通文本消息仍按 value_to_string 计费。
        let char_count = if message_contains_image(&message.content) {
            message.content.to_string().chars().count()
        } else {
            text.chars().count()
        };
        if message_contains_image(&message.content) && char_count >= IMAGE_OVERFLOW_SPILL_MIN_CHARS
        {
            return Some(idx);
        }
        if message.role == "user" && char_count >= USER_OVERFLOW_SPILL_MIN_CHARS {
            return Some(idx);
        }
    }
    None
}

fn write_preserved_message_overflow_file(
    overflow_dir: &Path,
    message: &Message,
    kind: &str,
) -> Option<PathBuf> {
    let subdir = if kind == "image" {
        PRESERVED_IMAGE_OVERFLOW_DIR
    } else {
        PRESERVED_USER_OVERFLOW_DIR
    };
    let dir = overflow_dir.join(subdir);
    std::fs::create_dir_all(&dir).ok()?;
    let file_name = format!(
        "{}-{}-{}.json",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        kind,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(file_name);

    let mut payload = serde_json::Map::new();
    payload.insert("role".to_string(), Value::String(message.role.clone()));
    payload.insert("kind".to_string(), Value::String(kind.to_string()));
    payload.insert("content".to_string(), message.content.clone());
    if let Some(tool_calls) = &message.tool_calls {
        payload.insert(
            "tool_calls".to_string(),
            serde_json::to_value(tool_calls).ok()?,
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        payload.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }

    let serialized = serde_json::to_string_pretty(&Value::Object(payload)).ok()?;
    std::fs::write(&path, serialized).ok()?;
    Some(path)
}

fn build_preserved_message_overflow_stub(path: &Path, kind: &str) -> String {
    let content_kind = if kind == "image" { "图片" } else { "文本" };
    format!(
        "较早的用户{content_kind}内容已归档，原文未丢失。\n归档文件: {}\n这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖原文时，才使用 read_file 读取该文件。",
        path.display()
    )
}

pub(super) fn try_spill_preserved_message_to_stub(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) -> bool {
    let Some(idx) = first_preserved_content_spill_candidate(messages, budget) else {
        return false;
    };
    let kind = if message_contains_image(&messages[idx].content) {
        "image"
    } else {
        "user"
    };
    let Some(path) = write_preserved_message_overflow_file(overflow_dir, &messages[idx], kind)
    else {
        return false;
    };
    messages[idx].content = Value::String(build_preserved_message_overflow_stub(&path, kind));
    true
}

/// 主动把体量过大的旧 user / 图片消息（保护尾窗之前的）搬到会话临时文件，
/// 原地替换为紧凑 stub。原文零压缩保存在磁盘上，但不再占用每轮请求的 payload。
///
/// 这与预算驱动的循环内 spill 互补：自从图片在预算里只按 [`IMAGE_BUDGET_CHARS`]
/// 名义计费后，单张大图本身不再触发 `messages_total_chars > max_chars`，于是
/// 循环内的 spill 永远不会被调用。这里改为「无论是否超预算，只要旧消息原始
/// 体量超过阈值就外溢」，既保证大图/大段用户原文被零压缩归档，又避免它们污染
/// 后续每一轮请求。最新一轮（保护尾窗内）的 user/图片永不外溢。
pub(super) fn spill_oversized_preserved_messages(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) {
    while try_spill_preserved_message_to_stub(messages, overflow_dir, budget) {}
}

fn structured_tool_output_summary(text: &str, max_chars: usize) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return String::new();
    }
    if lines.len() <= 8 {
        let mut out = Vec::new();
        let mut used = 0usize;
        for line in lines
            .into_iter()
            .map(tool_line_signature)
            .filter(|line| !line.is_empty())
        {
            let extra = if out.is_empty() { 0 } else { 1 };
            if used + extra + line.chars().count() > max_chars {
                break;
            }
            used += extra + line.chars().count();
            out.push(line);
        }
        return out.join("\n");
    }

    let mut sections = Vec::new();
    push_section_with_budget(
        &mut sections,
        format!("tool_output_lines: {}", lines.len()),
        max_chars,
    );

    let key_signals = lines
        .iter()
        .filter(|line| is_important_tool_line(line))
        .map(|line| tool_line_signature(line))
        .filter(|line| !line.is_empty())
        .fold(Vec::new(), |mut acc: Vec<String>, line| {
            push_unique_limited_global(&mut acc, line, 4);
            acc
        });
    if !key_signals.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("key_signals: {}", key_signals.join(" || ")),
            max_chars,
        );
    }

    let path_hints = lines
        .iter()
        .flat_map(|line| extract_path_like_tokens(line))
        .fold(Vec::new(), |mut acc: Vec<String>, token| {
            push_unique_limited_global(&mut acc, token, 4);
            acc
        });
    if !path_hints.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("paths: {}", path_hints.join(", ")),
            max_chars,
        );
    }

    let chunk_size = (lines.len() / 3).max(1);
    let mut chunk_summaries = Vec::new();
    for (chunk_index, chunk) in lines.chunks(chunk_size).take(3).enumerate() {
        let chunk_summary = summarize_tool_chunk(chunk_index + 1, chunk);
        if !chunk_summary.is_empty() {
            chunk_summaries.push(chunk_summary);
        }
    }
    if !chunk_summaries.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("chunks:\n- {}", chunk_summaries.join("\n- ")),
            max_chars,
        );
    }

    sections.join("\n")
}

fn push_section_with_budget(target: &mut Vec<String>, section: String, max_chars: usize) {
    if section.is_empty() {
        return;
    }
    let current = if target.is_empty() {
        0
    } else {
        target.join("\n").chars().count() + 1
    };
    if current + section.chars().count() <= max_chars {
        target.push(section);
        return;
    }
    if target.is_empty() {
        target.push(summarize_text(&section, max_chars));
    }
}

fn summarize_tool_chunk(chunk_index: usize, chunk: &[&str]) -> String {
    if chunk.is_empty() {
        return String::new();
    }
    let mut picks: Vec<String> = Vec::new();
    let first = tool_line_signature(chunk[0]);
    if !first.is_empty() {
        push_unique_limited_global(&mut picks, first, 4);
    }
    for line in chunk
        .iter()
        .filter(|line| is_important_tool_line(line))
        .take(2)
    {
        let sig = tool_line_signature(line);
        if !sig.is_empty() {
            push_unique_limited_global(&mut picks, sig, 4);
        }
    }
    if let Some(last) = chunk.last() {
        let last = tool_line_signature(last);
        if !last.is_empty() {
            push_unique_limited_global(&mut picks, last, 4);
        }
    }
    if picks.is_empty() {
        return String::new();
    }
    format!("chunk_{chunk_index}: {}", picks.join(" | "))
}

pub(super) fn tool_line_signature(line: &str) -> String {
    let normalized = normalize_whitespace(line);
    if normalized.is_empty() {
        return String::new();
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.len() <= 18 {
        return normalized;
    }

    let head = words.iter().take(12).copied().collect::<Vec<_>>().join(" ");
    let mut notable_tail = Vec::new();
    for word in words.iter().rev() {
        let token = word.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        });
        if token.is_empty() {
            continue;
        }
        let looks_notable = token.contains('/')
            || token.contains('.')
            || token.chars().any(|ch| ch.is_ascii_digit())
            || looks_like_error_code(token);
        if looks_notable {
            push_unique_limited_global(&mut notable_tail, token.to_string(), 4);
        }
    }
    notable_tail.reverse();
    if notable_tail.is_empty() {
        return head;
    }
    format!("{head} | {}", notable_tail.join(" "))
}

fn is_important_tool_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("exception")
        || lower.contains("timeout")
        || lower.contains("not found")
        || lower.contains("traceback")
        || lower.contains("exit code")
        || lower.contains("warning")
        || lower.contains("completed")
        || lower.contains("success")
}

fn extract_path_like_tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in line.split_whitespace() {
        let token = raw.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        });
        if token.len() > 160 || token.is_empty() {
            continue;
        }
        if token.starts_with("http://") || token.starts_with("https://") {
            continue;
        }
        let looks_like_path = token.contains('/')
            || [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ]
            .iter()
            .any(|suffix| token.ends_with(suffix));
        if looks_like_path {
            push_unique_limited_global(&mut out, token.to_string(), 8);
        }
    }
    out
}

fn looks_like_error_code(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 5 && bytes[0] == b'E' && bytes[1..].iter().all(|byte| byte.is_ascii_digit())
}

fn push_unique_limited_global(target: &mut Vec<String>, value: String, max_items: usize) {
    if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items {
        return;
    }
    target.push(value);
}

pub(super) fn build_persisted_summary_text(messages: &[Message], max_chars: usize) -> String {
    #[derive(Default, Clone)]
    struct TurnSummary {
        topic_key: String,
        topic_label: String,
        user: String,
        user_key: String,
        assistant_final: String,
        tool_names: Vec<String>,
        tool_highlights: Vec<String>,
        count: usize,
    }

    fn normalize_semantic_key(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            let is_cjk = ('\u{4E00}'..='\u{9FFF}').contains(&ch);
            if is_cjk || ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
                continue;
            }
            if ch.is_whitespace() {
                out.push(' ');
            }
        }
        normalize_whitespace(&out)
    }

    fn extract_topic_from_text(text: &str) -> Option<(String, String)> {
        fn trim_punct(s: &str) -> &str {
            s.trim_matches(|ch: char| {
                ch.is_whitespace()
                    || matches!(
                        ch,
                        ',' | '.'
                            | ';'
                            | ':'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '<'
                            | '>'
                            | '"'
                            | '\''
                            | '`'
                    )
            })
        }

        fn candidate_file_token(token: &str) -> Option<&str> {
            let token = trim_punct(token);
            if token.is_empty() || token.len() > 96 {
                return None;
            }
            if token.starts_with("http://") || token.starts_with("https://") {
                return None;
            }
            let token = token.split('#').next().unwrap_or(token);
            let token = token.split('?').next().unwrap_or(token);
            let token = token.split_once(':').map(|(a, _)| a).unwrap_or(token);
            let suffixes = [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ];
            if suffixes.iter().any(|suf| token.ends_with(suf)) {
                return Some(token);
            }
            None
        }

        fn basename(path: &str) -> &str {
            path.rsplit('/').next().unwrap_or(path)
        }

        fn find_error_code(text: &str) -> Option<String> {
            let bytes = text.as_bytes();
            let mut i = 0usize;
            while i + 5 <= bytes.len() {
                if bytes[i] == b'E'
                    && bytes[i + 1].is_ascii_digit()
                    && bytes[i + 2].is_ascii_digit()
                    && bytes[i + 3].is_ascii_digit()
                    && bytes[i + 4].is_ascii_digit()
                {
                    let code = &text[i..i + 5];
                    return Some(code.to_string());
                }
                i += 1;
            }
            None
        }

        if let Some(code) = find_error_code(text) {
            return Some((code.to_ascii_lowercase(), code));
        }

        for raw in text.split_whitespace() {
            if let Some(token) = candidate_file_token(raw) {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
            let token = trim_punct(raw);
            if token.contains('/')
                && token.len() <= 96
                && token.chars().any(|c| c == '.')
                && !token.starts_with("http://")
                && !token.starts_with("https://")
            {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
        }

        None
    }

    fn push_unique_limited(target: &mut Vec<String>, value: String, max_items: usize) {
        if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items
        {
            return;
        }
        target.push(value);
    }

    fn tool_highlight(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let lowered = text.to_ascii_lowercase();
        let important = lowered.contains("error")
            || lowered.contains("failed")
            || lowered.contains("panic")
            || lowered.contains("exception")
            || lowered.contains("[error]");
        if important {
            return extract_important_lines(text, 120);
        }
        summarize_text(&normalize_whitespace(text), 80)
    }

    fn extract_important_lines(text: &str, target_chars: usize) -> String {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return String::new();
        }
        let mut selected: Vec<&str> = Vec::new();
        let mut chars = 0usize;
        for line in &lines {
            let lowered = line.to_ascii_lowercase();
            let is_key = lowered.contains("error")
                || lowered.contains("failed")
                || lowered.contains("panic")
                || lowered.contains("exception")
                || lowered.contains("not found")
                || lowered.contains("timeout");
            if is_key || selected.is_empty() {
                if chars + line.trim().chars().count() + 2 > target_chars {
                    if selected.is_empty() {
                        let trimmed = line.trim();
                        selected.push(trimmed);
                    }
                    break;
                }
                selected.push(line.trim());
                chars += line.trim().chars().count() + 2;
            }
        }
        let result = selected.join("; ");
        if result.chars().count() <= target_chars {
            return result;
        }
        keep_ends_by_chars(&result, target_chars)
    }

    fn finalize_turn(turns: &mut Vec<TurnSummary>, current: &mut TurnSummary) {
        if current.user.trim().is_empty()
            && current.assistant_final.trim().is_empty()
            && current.tool_names.is_empty()
            && current.tool_highlights.is_empty()
        {
            *current = TurnSummary::default();
            return;
        }
        if current.count == 0 {
            current.count = 1;
        }
        turns.push(current.clone());
        *current = TurnSummary::default();
    }

    fn merge_turns(mut turns: Vec<TurnSummary>) -> Vec<TurnSummary> {
        let mut out: Vec<TurnSummary> = Vec::with_capacity(turns.len());
        for turn in turns.drain(..) {
            if let Some(last) = out.last_mut()
                && !turn.user_key.is_empty()
                && last.user_key == turn.user_key
            {
                last.count = last.count.saturating_add(turn.count.max(1));
                if last.topic_label.is_empty() && !turn.topic_label.is_empty() {
                    last.topic_label = turn.topic_label;
                    last.topic_key = turn.topic_key;
                }
                if !turn.assistant_final.is_empty()
                    && turn.assistant_final != last.assistant_final
                    && last.assistant_final.chars().count() < 200
                {
                    if last.assistant_final.is_empty() {
                        last.assistant_final = turn.assistant_final;
                    } else {
                        last.assistant_final = summarize_text(
                            &format!("{} / {}", last.assistant_final, turn.assistant_final),
                            250,
                        );
                    }
                }
                for name in turn.tool_names {
                    push_unique_limited(&mut last.tool_names, name, 6);
                }
                for h in turn.tool_highlights {
                    push_unique_limited(&mut last.tool_highlights, h, 3);
                }
                continue;
            }
            out.push(turn);
        }
        out
    }

    fn render_line(turn: &TurnSummary) -> String {
        let mut line = String::new();
        if turn.count > 1 {
            line.push_str(&format!("repeated ×{} ", turn.count));
        }
        if !turn.topic_label.is_empty() {
            line.push_str("Topic: ");
            line.push_str(&turn.topic_label);
            line.push_str(" | ");
        }
        if !turn.user.is_empty() {
            line.push_str("User: ");
            line.push_str(&turn.user);
        }
        if !turn.assistant_final.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Assistant's previous answer (not independently verified): ");
            line.push_str(&turn.assistant_final);
        }
        if !turn.tool_names.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Tools: ");
            line.push_str(&turn.tool_names.join(", "));
        }
        if !turn.tool_highlights.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Key: ");
            line.push_str(&turn.tool_highlights.join(", "));
        }
        line
    }

    fn render_known_tool_line(turn: &TurnSummary) -> Option<String> {
        if turn.tool_names.is_empty() {
            return None;
        }
        let mut line = String::new();
        line.push_str("- ");
        line.push_str(&turn.tool_names.join(", "));
        if !turn.topic_label.is_empty() {
            line.push_str(" @ ");
            line.push_str(&turn.topic_label);
        }
        if !turn.tool_highlights.is_empty() {
            line.push_str(" => ");
            line.push_str(&turn.tool_highlights.join(", "));
        }
        Some(line)
    }

    fn push_line_with_budget(lines: &mut Vec<String>, mut line: String, max_chars: usize) -> bool {
        let line_chars = line.chars().count();
        if lines.is_empty() {
            if line_chars > max_chars {
                lines.push(summarize_text(&line, max_chars));
                return true;
            }
            lines.push(line);
            return true;
        }
        let current_len = lines.join("\n").chars().count();
        let remaining = max_chars.saturating_sub(current_len + 1);
        if remaining < 30 {
            return false;
        }
        if line_chars > remaining {
            line = summarize_text(&line, remaining);
        }
        if line.chars().count() <= remaining {
            lines.push(line);
            true
        } else {
            false
        }
    }

    let mut initial_goal = String::new();
    let mut pre_summary_lines: Vec<String> = Vec::new();
    let mut turns: Vec<TurnSummary> = Vec::new();
    let mut current = TurnSummary::default();

    for message in messages {
        let text = normalize_whitespace(&value_to_string(&message.content));
        match message.role.as_str() {
            role if role == ROLE_INTERNAL_NOTE => {
                if let Some(body) = automatic_summary_body(&text) {
                    let normalized =
                        summarize_text(&strip_nested_prior_summary_prefixes(body), 400);
                    if !normalized.is_empty() {
                        push_unique_limited(
                            &mut pre_summary_lines,
                            format!("- Earlier summary: {normalized}"),
                            3,
                        );
                    }
                }
            }
            role if is_system_like_role(role) => {}
            "user" => {
                finalize_turn(&mut turns, &mut current);
                if initial_goal.is_empty() {
                    initial_goal = summarize_text(&text, 240);
                }
                current.user = summarize_text(&text, 200);
                current.user_key = truncate_to_chars(&normalize_semantic_key(&text), 160);
                if let Some((k, label)) = extract_topic_from_text(&text) {
                    current.topic_key = k;
                    current.topic_label = label;
                }
                if current.count == 0 {
                    current.count = 1;
                }
            }
            "assistant" => {
                if !text.is_empty() {
                    current.assistant_final = summarize_text(&text, 250);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&text) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        push_unique_limited(
                            &mut current.tool_names,
                            tool_call.function.name.clone(),
                            6,
                        );
                        if current.topic_key.is_empty() {
                            current.topic_key = tool_call.function.name.to_ascii_lowercase();
                            current.topic_label = tool_call.function.name.clone();
                        }
                    }
                }
            }
            "tool" => {
                let h = tool_highlight(&text);
                if !h.is_empty() {
                    push_unique_limited(&mut current.tool_highlights, h.clone(), 3);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&h) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    finalize_turn(&mut turns, &mut current);

    let recent_count = turns.len().min(3);
    let recent_turns: Vec<TurnSummary> = turns
        .iter()
        .rev()
        .take(recent_count)
        .rev()
        .cloned()
        .collect();

    let pending_tasks: Vec<String> = turns
        .iter()
        .rev()
        .take(2)
        .filter(|t| !t.user.is_empty() && t.assistant_final.is_empty())
        .map(|t| t.user.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let merged = merge_turns(turns);
    let mut known_tool_lines: Vec<String> = Vec::new();
    for t in &merged {
        if let Some(line) = render_known_tool_line(t)
            && !known_tool_lines.iter().any(|existing| existing == &line)
            && known_tool_lines.len() < 10
        {
            known_tool_lines.push(line);
        }
    }
    let reserved_tool_chars = if known_tool_lines.is_empty() {
        0
    } else {
        let tool_blob = format!(
            "Verified facts and sources:\n{}",
            known_tool_lines.join("\n")
        );
        tool_blob.chars().count().min(max_chars / 3)
    };
    let body_budget = max_chars
        .saturating_sub(reserved_tool_chars)
        .max(max_chars / 2);
    let mut lines: Vec<String> = Vec::new();
    if !initial_goal.is_empty()
        && !push_line_with_budget(
            &mut lines,
            format!("Main request: {initial_goal}"),
            body_budget,
        )
    {
        return summarize_text(&lines.join("\n"), max_chars);
    }
    for s in pre_summary_lines.into_iter().take(3) {
        if !push_line_with_budget(&mut lines, s, body_budget) {
            return summarize_text(&lines.join("\n"), max_chars);
        }
    }
    for t in &merged {
        if !push_line_with_budget(&mut lines, format!("- {}", render_line(t)), body_budget) {
            break;
        }
    }

    if !known_tool_lines.is_empty() {
        let _ = push_line_with_budget(
            &mut lines,
            "Verified facts and sources:".to_string(),
            max_chars,
        );
        for line in known_tool_lines {
            if !push_line_with_budget(&mut lines, line, max_chars) {
                break;
            }
        }
    }

    if !recent_turns.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "Current work:".to_string(), max_chars);
        for t in &recent_turns {
            let mut parts = Vec::new();
            if !t.topic_label.is_empty() {
                parts.push(format!("Topic: {}", t.topic_label));
            }
            if !t.user.is_empty() {
                parts.push(format!("User: {}", t.user));
            }
            if !t.assistant_final.is_empty() {
                parts.push(format!(
                    "Assistant's previous answer (not independently verified): {}",
                    t.assistant_final
                ));
            }
            if !t.tool_names.is_empty() {
                parts.push(format!("Tools: {}", t.tool_names.join(", ")));
            }
            if !t.tool_highlights.is_empty() {
                parts.push(format!("Key: {}", t.tool_highlights.join(", ")));
            }
            let line = format!("- {}", parts.join(" | "));
            if !push_line_with_budget(&mut lines, summarize_text(&line, 600), max_chars) {
                break;
            }
        }
    }

    if !pending_tasks.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "Pending tasks:".to_string(), max_chars);
        for task in &pending_tasks {
            if !push_line_with_budget(
                &mut lines,
                format!("- {}", summarize_text(task, 300)),
                max_chars,
            ) {
                break;
            }
        }
    }

    summarize_text(&lines.join("\n"), max_chars)
}
