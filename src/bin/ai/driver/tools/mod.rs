use colored::Colorize;
use serde_json::{Value, json};
use std::error::Error;
use rust_tools::commonw::FastMap;
use std::sync::{LazyLock, Mutex};

use crate::ai::{
    mcp::McpClient,
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

mod barrier;
mod oauth;

static TOOL_FAILURES: LazyLock<Mutex<FastMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone)]
enum ToolRoute {
    Builtin,
    Mcp {
        server_name: String,
        tool_name: String,
    },
}

#[derive(Debug, Clone)]
struct PreparedToolCall {
    route: ToolRoute,
    args: Value,
}

pub(super) struct ExecuteToolCallsResult {
    pub(super) executed_tool_calls: Vec<ToolCall>,
    pub(super) tool_results: Vec<ToolResult>,
}

pub(super) struct RunOneResult {
    pub(super) tool_result: ToolResult,
    pub(super) ok: bool,
    pub(super) executed: bool,
}

fn route_tool_call(mcp_client: &McpClient, tool_name: &str) -> ToolRoute {
    if let Some((server_name, tool_name)) = mcp_client.parse_tool_name_for_known_server(tool_name) {
        ToolRoute::Mcp {
            server_name,
            tool_name,
        }
    } else {
        ToolRoute::Builtin
    }
}

fn parse_tool_args(tool_call: &ToolCall) -> Result<Value, ToolResult> {
    let raw_args = tool_call.function.arguments.trim();
    if raw_args.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(raw_args).map_err(|err| ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!("Error: failed to parse arguments: {}", err),
    })
}

fn prepare_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
) -> Result<PreparedToolCall, ToolResult> {
    Ok(PreparedToolCall {
        route: route_tool_call(mcp_client, &tool_call.function.name),
        args: parse_tool_args(tool_call)?,
    })
}

fn requires_user_confirmation_for_tool(_tool_name: &str) -> bool {
    false
}

fn confirm_tool_execution(tool_call: &ToolCall, args: &Value) -> Result<(), RunOneResult> {
    if !requires_user_confirmation_for_tool(&tool_call.function.name) {
        return Ok(());
    }

    let confirm =
        prompt_yes_or_no_interruptible(&format!("Confirm tool execution:{} (y/n): ", args));
    if confirm == Some(true) {
        return Ok(());
    }

    println!("canceled by user.");
    Err(RunOneResult {
        tool_result: ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: if confirm.is_none() {
                format!(
                    "Error: {} canceled by user (Ctrl+C)",
                    tool_call.function.name
                )
            } else {
                format!("Error: {} canceled by user", tool_call.function.name)
            },
        },
        ok: false,
        executed: false,
    })
}

fn remediation_hint(tool_name: &str, err: &str) -> Option<String> {
    let err_lower = err.to_lowercase();

    if tool_name == "mcp_feishu_docs_get_text_by_url" && err_lower.contains("unsupported url") {
        return Some(
            "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL. Use mcp_feishu_docs_search to find the document first, or ask the user for a direct Feishu docs/wiki/sheet URL.".to_string(),
        );
    }

    if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
        return Some(
            "Suggestion: fix the tool arguments to match the declared JSON schema before retrying."
                .to_string(),
        );
    }

    if err_lower.contains("no such file") || err_lower.contains("not found") {
        return Some(
            "Suggestion: verify the path or identifier first, or use a search/list tool to discover the correct target before retrying.".to_string(),
        );
    }

    if err_lower.contains("timeout") || err_lower.contains("timed out") {
        return Some(
            "Suggestion: retry once with a narrower query or a smaller scope. If it still fails, switch to another tool or ask the user.".to_string(),
        );
    }

    None
}

fn format_tool_error(tool_call: &ToolCall, err: &str) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: if let Some(hint) = remediation_hint(&tool_call.function.name, err) {
            format!(
                "Error: {} failed: {}\n{}",
                tool_call.function.name, err, hint
            )
        } else {
            format!("Error: {} failed: {}", tool_call.function.name, err)
        },
    }
}

fn execute_prepared_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => builtin_tools::execute_tool_call_with_args(
            &tool_call.id,
            &tool_call.function.name,
            &prepared.args,
        )
        .map_err(|e| e.to_string()),
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => oauth::execute_mcp_tool_call(
            mcp_client,
            tool_call,
            server_name,
            tool_name,
            &prepared.args,
        ),
    }
}

fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> (ToolRoute, RunOneResult) {
    let prepared = match prepare_tool_call(mcp_client, tool_call) {
        Ok(prepared) => prepared,
        Err(tool_result) => {
            return (
                route_tool_call(mcp_client, &tool_call.function.name),
                RunOneResult {
                    tool_result,
                    ok: false,
                    executed: true,
                },
            );
        }
    };

    if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
        return (prepared.route, result);
    }

    let result = execute_prepared_tool_call(mcp_client, tool_call, &prepared);
    let run_result = match result {
        Ok(tool_result) => RunOneResult {
            tool_result,
            ok: true,
            executed: true,
        },
        Err(err) => RunOneResult {
            tool_result: format_tool_error(tool_call, &err),
            ok: false,
            executed: true,
        },
    };
    if run_result.executed {
        if !run_result.ok {
            if let Ok(mut map) = TOOL_FAILURES.lock() {
                let name = tool_call.function.name.clone();
                let counter = map.entry(name).or_insert(0);
                *counter = counter.saturating_add(1).min(100);
            }
        }
    }

    (prepared.route, run_result)
}

pub(super) fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return tokio::task::block_in_place(|| execute_tool_calls_inner(mcp_client, tool_calls));
    }
    execute_tool_calls_inner(mcp_client, tool_calls)
}

fn execute_tool_calls_inner(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        let is_last = idx + 1 >= tool_calls.len();
        let (route, run_result) = run_one(mcp_client, tool_call);
        let should_barrier = barrier::should_barrier_after(
            &route,
            tool_call,
            run_result.ok,
            &run_result.tool_result.content,
        );

        executed_tool_calls.push(tool_call.clone());
        tool_results.push(run_result.tool_result);
        if run_result.executed {
            println!("\n[Executed] {}", tool_call.function.name.green());
        } else {
            println!("\n[Skipped] {}", tool_call.function.name.yellow());
        }

        if should_barrier && !is_last {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
    })
}

pub(super) fn penalty_for_skill_tools(skill: &crate::ai::skills::SkillManifest) -> f64 {
    if skill.tools.is_empty() {
        return 0.0;
    }
    let tools = &skill.tools;
    let Ok(map) = TOOL_FAILURES.lock() else {
        return 0.0;
    };
    let mut score = 0.0f64;
    for t in tools {
        if let Some(c) = map.get(t) {
            score += (*c as f64).min(10.0);
        }
    }
    score
}
