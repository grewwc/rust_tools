//! Per-tool-result execution for the turn loop.
//!
//! Feature clusters (each a child module): result preparation
//! (`prepare`), pre-execution gating (`rejection`), final-response
//! quality gates (`audit_evidence`, `completion_gate`, `final_citations`,
//! `final_recovery`, `validated_claims`), patch retries (`patch_retry`), terminal
//! presentation (`output_format`, `observer`), round orchestration
//! (`round`), post-round followups (`followup`), and the iteration
//! entry points (`iteration`).
pub(super) use crate::ai::{
    driver::tools::{self, ExecuteToolCallsResult},
    history::{
        Message, ROLE_INTERNAL_NOTE, is_runtime_synthetic_user_message,
        runtime_synthetic_user_message,
    },
    mcp::{McpClient, SharedMcpClient},
    middleware::tool::build_tool_executor_chain,
    ports::tool::{ToolExecOutput, ToolExecutor},
    stream::clamp_line_to_terminal_row_with_reserve,
    tools::{storage::file_store::FileStore, task_tools},
    types::{App, ToolCall},
};
pub(super) use regex::Regex;
pub(super) use rust_tools::commonw::FastSet;
pub(super) use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::LazyLock,
};

pub(super) use super::super::persistence::persist_pending_turn_messages_for_model;
pub(super) use super::super::{
    MAX_TOOL_RESULT_LINE_TRIM_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS,
    iteration::no_tool_handoff_note,
    max_tool_result_inline_chars,
    orchestrator::record_force_final_reason,
    types::{IterationExecution, PreparedToolResult, ToolCallExecution, TurnLoopStep},
};
pub(super) use super::{
    messaging::{
        append_cached_tool_results_note, append_message_pair,
        append_tool_result_messages_for_model, parse_prune_meta_and_update_marks,
        record_final_stream_response, record_hidden_self_note, record_tool_inspection_artifacts,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
pub(super) use crate::ai::driver::print::{
    format_tool_output_line, format_tool_output_prefix, print_tool_command_line,
    print_tool_note_line, sanitize_for_terminal,
};
pub(super) use crate::ai::theme::{ACCENT_MUTED, ACCENT_RULE, RESET};


mod prepare;
mod rejection;
mod audit_evidence;
mod completion_gate;
mod final_citations;
mod final_recovery;
mod validated_claims;
mod patch_retry;
mod output_format;
mod observer;
mod round;
mod followup;
mod iteration;

pub(in crate::ai::driver) use patch_retry::stale_patch_targets_from_messages;
pub(in crate::ai::driver::turn_runtime) use completion_gate::{
    completion_evidence_state, completion_tool_result_succeeded,
    tool_call_is_successful_mutation_candidate,
};
pub(in crate::ai::driver::turn_runtime) use iteration::handle_iteration_execution_for_model;
pub(in crate::ai::driver::turn_runtime) use prepare::prepare_recent_tool_result;

// Flat-namespace re-exports over the cluster modules: child clusters resolve
// sibling items through `use super::*` and callers outside `execution/` keep
// the pre-split paths working. `iteration`/`prepare` items are consumed
// non-test via the explicit re-exports above (or only within their own
// cluster), so the glob segments are unused outside test builds.
#[cfg_attr(not(test), allow(unused_imports))]
pub(in crate::ai::driver::turn_runtime) use {
    audit_evidence::*, completion_gate::*, final_citations::*, final_recovery::*, followup::*,
    iteration::*, observer::*, output_format::*, patch_retry::*, prepare::*, rejection::*,
    round::*, validated_claims::*,
};
#[cfg(test)]
mod tests;
