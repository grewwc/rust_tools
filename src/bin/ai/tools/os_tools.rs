use serde_json::Value;
use std::sync::LazyLock;
use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};
use std::sync::Mutex;
use aios_kernel::kernel::{ProcessCapabilities, SharedKernel};
use aios_kernel::primitives::{ChannelMetaSnapshot, ChannelOwnerTag};
pub static GLOBAL_OS: LazyLock<Mutex<Option<SharedKernel>>> = LazyLock::new(|| Mutex::new(None));

pub fn init_os_tools_globals(os: SharedKernel) {
    if let Ok(mut g) = GLOBAL_OS.lock() {
        *g = Some(os);
    }
}

fn parse_capabilities(args: &Value) -> Option<ProcessCapabilities> {
    let caps = args.get("capabilities")?;
    Some(ProcessCapabilities {
        spawn: caps.get("spawn").and_then(Value::as_bool).unwrap_or(false),
        wait: caps.get("wait").and_then(Value::as_bool).unwrap_or(false),
        ipc_send: caps
            .get("ipc_send")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        ipc_receive: caps
            .get("ipc_receive")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        env_write: caps
            .get("env_write")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        manage_children: caps
            .get("manage_children")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        sleep: caps.get("sleep").and_then(Value::as_bool).unwrap_or(false),
        reap: caps.get("reap").and_then(Value::as_bool).unwrap_or(false),
        signal: caps.get("signal").and_then(Value::as_bool).unwrap_or(false),
    })
}

// 1. spawn_process
fn params_spawn_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name for the process."
            },
            "goal": {
                "type": "string",
                "description": "The specific goal or task for the sub-process to accomplish."
            },
            "priority": {
                "type": "integer",
                "description": "Process priority. 0 is highest, 255 is lowest. Default is 10."
            },
            "quota_turns": {
                "type": "integer",
                "description": "Max number of tool calling turns allowed for this process before it is forcefully yielded. Default is 10."
            },
            "capabilities": {
                "type": "object",
                "description": "Optional exact capability set for the child process. Omitted means inherit the parent capabilities.",
                "properties": {
                    "spawn": { "type": "boolean" },
                    "wait": { "type": "boolean" },
                    "ipc_send": { "type": "boolean" },
                    "ipc_receive": { "type": "boolean" },
                    "env_write": { "type": "boolean" },
                    "manage_children": { "type": "boolean" },
                    "sleep": { "type": "boolean" },
                    "reap": { "type": "boolean" }
                }
            },
            "allowed_tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional whitelist of tool names this process is allowed to call. Omitted means inherit parent's whitelist. Empty array means no restriction."
            }
        },
        "required": ["name", "goal"]
    })
}

fn execute_spawn_process(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing 'name' string parameter.")?;
    let goal = args["goal"].as_str().ok_or("Missing 'goal' string parameter.")?;
    let priority = args["priority"].as_u64().unwrap_or(10) as u8;
    let quota_turns = args["quota_turns"].as_u64().unwrap_or(10) as usize;
    let capabilities = parse_capabilities(args);

    let allowed_tools = if let Some(tools_array) = args.get("allowed_tools").and_then(Value::as_array) {
        let mut set = aios_kernel::FastSet::default();
        for tool in tools_array {
            if let Some(tool_name) = tool.as_str() {
                set.insert(tool_name.to_string());
            }
        }
        Some(set)
    } else {
        None
    };

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let current_pid = os.current_process_id();
            let pid = os.spawn(
                current_pid,
                name.to_string(),
                goal.to_string(),
                priority,
                quota_turns,
                capabilities,
                allowed_tools,
            )?;
            return Ok(format!("Sub-process spawned successfully. PID: {}, Name: {}. The scheduler will execute it autonomously.", pid, name));
        }
    }
    
    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "spawn_process",
        description: "Spawn a new background process in the Agent OS to handle a specific sub-goal or parallel task. The scheduler will execute this autonomously. Returns the PID.",
        parameters: params_spawn_process,
        execute: execute_spawn_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_sleep_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "turns": {
                "type": "integer",
                "description": "How many scheduler ticks to sleep. Minimum is 1."
            }
        }
    })
}

fn execute_sleep_process(args: &Value) -> Result<String, String> {
    let turns = args["turns"].as_u64().unwrap_or(1);
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let until_tick = os.sleep_current(turns)?;
            return Ok(format!(
                "Current process suspended until scheduler tick {}. Yield control now.",
                until_tick
            ));
        }
    }

    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "sleep_process",
        description: "Suspend the current process for a number of scheduler ticks, then resume it later via the ready queue.",
        parameters: params_sleep_process,
        execute: execute_sleep_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// 2. wait_process
fn params_wait_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "The Process ID (PID) to wait for."
            }
        },
        "required": ["pid"]
    })
}

fn execute_wait_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing or invalid 'pid' parameter.")?;

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.wait_on(pid)?;
            return Ok(format!("Current process suspended. Will be awakened when PID {} terminates. Note: Do not emit further output in this turn, just yield control.", pid));
        }
    }
    
    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "wait_process",
        description: "Suspend the current process until the specified child process (PID) terminates. You will be awakened via your mailbox with the child's result.",
        parameters: params_wait_process,
        execute: execute_wait_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_kill_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Child or descendant PID to terminate."
            },
            "reason": {
                "type": "string",
                "description": "Reason recorded in the target process result."
            }
        },
        "required": ["pid"]
    })
}

fn execute_kill_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing or invalid 'pid' parameter.")?;
    let reason = args["reason"]
        .as_str()
        .unwrap_or("terminated by parent process")
        .to_string();

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.kill_process(pid, reason)?;
            return Ok(format!("Process {} terminated successfully.", pid));
        }
    }

    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "kill_process",
        description: "Terminate a child or descendant process when the current process has management capability.",
        parameters: params_kill_process,
        execute: execute_kill_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// 3. send_ipc_message
fn params_send_ipc() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "The target Process ID (PID) to send the message to."
            },
            "message": {
                "type": "string",
                "description": "The message content to send."
            }
        },
        "required": ["pid", "message"]
    })
}

fn execute_send_ipc(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing or invalid 'pid' parameter.")?;
    let message = args["message"].as_str().ok_or("Missing 'message' string parameter.")?;

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.send_ipc(pid, message.to_string())?;
            return Ok(format!("Message sent successfully to PID {}.", pid));
        }
    }
    
    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "send_ipc_message",
        description: "Send an Inter-Process Communication (IPC) message to another running process's mailbox.",
        parameters: params_send_ipc,
        execute: execute_send_ipc,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_reap_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Terminated child or descendant PID to collect and remove from the process table."
            }
        },
        "required": ["pid"]
    })
}

fn execute_reap_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing or invalid 'pid' parameter.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let result = os.reap_process(pid)?;
            return Ok(format!("Reaped process {}. Final result: {}", pid, result));
        }
    }

    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "reap_process",
        description: "Collect a terminated child or descendant process and remove it from the process table.",
        parameters: params_reap_process,
        execute: execute_reap_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// 4. read_mailbox
fn params_read_mailbox() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

fn execute_read_mailbox(_args: &Value) -> Result<String, String> {
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let messages = os.read_mailbox()?;
            if messages.is_empty() {
                return Ok("Mailbox is empty.".to_string());
            } else {
                return Ok(format!("Mailbox messages:\n{}", messages.join("\n---\n")));
            }
        }
    }
    
    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_mailbox",
        description: "Read all pending IPC messages, wake-up notifications, and child process termination results from your mailbox. Calling this empties the mailbox. When the mailbox contains async tool wake-up messages, use them to decide whether to call tool_status, tool_wait, tool_cancel, or continue reasoning with already available results.",
        parameters: params_read_mailbox,
        execute: execute_read_mailbox,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// 5. env tools
fn params_set_env() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "key": { "type": "string" },
            "value": { "type": "string" }
        },
        "required": ["key", "value"]
    })
}

fn execute_set_env(args: &Value) -> Result<String, String> {
    let key = args["key"].as_str().ok_or("Missing 'key'")?;
    let value = args["value"].as_str().ok_or("Missing 'value'")?;

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.set_env(key.to_string(), value.to_string())?;
            return Ok(format!("Environment variable {} set.", key));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "set_env",
        description: "Set an environment variable in the current process's Context Manager. Child processes will inherit this context.",
        parameters: params_set_env,
        execute: execute_set_env,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_ps_processes() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

fn execute_ps_processes(_args: &Value) -> Result<String, String> {
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let os = os.lock().unwrap();
            let procs = os.list_processes();
            if procs.is_empty() {
                return Ok("No processes in the system.".to_string());
            }
            let mut lines = vec!["PID   PPID   PGID  State       Pri  Quota  Used  Tools  Ticks  Daemon  Name".to_string()];
            for p in &procs {
                let ppid = p.parent_pid.map(|id| id.to_string()).unwrap_or("-".to_string());
                let pgid = p.process_group.map(|id| id.to_string()).unwrap_or("-".to_string());
                let state = match &p.state {
                    aios_kernel::kernel::ProcessState::Ready => "Ready",
                    aios_kernel::kernel::ProcessState::Running => "Running",
                    aios_kernel::kernel::ProcessState::Waiting { .. } => "Waiting",
                    aios_kernel::kernel::ProcessState::Sleeping { .. } => "Sleeping",
                    aios_kernel::kernel::ProcessState::Stopped => "Stopped",
                    aios_kernel::kernel::ProcessState::Terminated => "Term",
                };
                let daemon = if p.is_daemon { format!("{}({}/{})", "Y", p.restart_count, p.max_restarts) } else { "N".to_string() };
                lines.push(format!("{:<5} {:<6} {:<5} {:<12} {:<4} {:<6} {:<5} {:<6} {:<6} {:<8} {}", 
                    p.pid, ppid, pgid, state, p.priority, p.quota_turns, p.turns_used, p.tool_calls_used, p.created_at_tick, daemon, p.name));
            }
            return Ok(lines.join("\n"));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "ps_processes",
        description: "List all processes in the Agent OS with their PID, parent PID, state, priority, quota, and name. Use this to inspect the process tree before deciding to kill, wait, or reap.",
        parameters: params_ps_processes,
        execute: execute_ps_processes,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_ps_ipc() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["result_pipes", "all"],
                "description": "Which channels to show. result_pipes focuses on task/async tool result pipes. all shows every channel."
            },
            "only_hanging": {
                "type": "boolean",
                "description": "When true, only show channels that are still live or not yet reclaimable. Default is true."
            }
        }
    })
}

fn is_hanging_channel(snapshot: &ChannelMetaSnapshot) -> bool {
    snapshot.ref_count > 0 || snapshot.queued_len > 0 || !snapshot.closed
}

fn is_result_pipe(snapshot: &ChannelMetaSnapshot) -> bool {
    !matches!(snapshot.owner_tag, ChannelOwnerTag::General)
}

fn execute_ps_ipc(args: &Value) -> Result<String, String> {
    let scope = args["scope"].as_str().unwrap_or("result_pipes");
    let only_hanging = args["only_hanging"].as_bool().unwrap_or(true);

    if scope != "result_pipes" && scope != "all" {
        return Err(format!(
            "Invalid scope {:?}. Valid values: result_pipes, all.",
            scope
        ));
    }

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let os = os.lock().unwrap();
            let mut channels = os.list_channels();
            if scope == "result_pipes" {
                channels.retain(is_result_pipe);
            }
            if only_hanging {
                channels.retain(is_hanging_channel);
            }

            if channels.is_empty() {
                return Ok(match (scope, only_hanging) {
                    ("result_pipes", true) => {
                        "No hanging result pipes in the system.".to_string()
                    }
                    ("result_pipes", false) => "No result pipes in the system.".to_string(),
                    ("all", true) => "No hanging IPC channels in the system.".to_string(),
                    ("all", false) => "No IPC channels in the system.".to_string(),
                    _ => unreachable!(),
                });
            }

            let mut lines = vec![
                "Chan   Tag               Owner  Refs  Queue  Closed  Label                     Holders"
                    .to_string(),
            ];
            for ch in channels {
                let owner = ch
                    .owner_pid
                    .map(|pid| pid.to_string())
                    .unwrap_or_else(|| "-".to_string());
                let holders = if ch.ref_holders.is_empty() {
                    "-".to_string()
                } else {
                    ch.ref_holders.join(", ")
                };
                lines.push(format!(
                    "{:<6} {:<17} {:<6} {:<5} {:<6} {:<7} {:<25} {}",
                    ch.channel.raw(),
                    ch.owner_tag.as_str(),
                    owner,
                    ch.ref_count,
                    ch.queued_len,
                    if ch.closed { "Y" } else { "N" },
                    ch.label,
                    holders
                ));
            }
            return Ok(lines.join("\n"));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "ps_ipc",
        description: "List AIOS IPC channels and result pipes with owner tags, queued messages, and named ref holders. By default this shows only hanging result pipes that still have refs, buffered messages, or are not closed yet.",
        parameters: params_ps_ipc,
        execute: execute_ps_ipc,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_signal_process() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Target child or descendant PID to signal."
            },
            "signal": {
                "type": "string",
                "enum": ["SIGCANCEL", "SIGTERM", "SIGSTOP", "SIGCONT", "SIGKILL"],
                "description": "Signal to send: SIGCANCEL=cooperative cancel current turn/tool work, SIGTERM=graceful termination, SIGSTOP=pause, SIGCONT=resume, SIGKILL=immediate termination with cascade."
            }
        },
        "required": ["pid", "signal"]
    })
}

fn execute_signal_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing or invalid 'pid' parameter.")?;
    let signal_str = args["signal"].as_str().ok_or("Missing 'signal' parameter.")?;
    let signal = match signal_str.to_uppercase().as_str() {
        "SIGCANCEL" => aios_kernel::kernel::Signal::SigCancel,
        "SIGTERM" => aios_kernel::kernel::Signal::SigTerm,
        "SIGSTOP" => aios_kernel::kernel::Signal::SigStop,
        "SIGCONT" => aios_kernel::kernel::Signal::SigCont,
        "SIGKILL" => aios_kernel::kernel::Signal::SigKill,
        other => return Err(format!("Unknown signal: {}. Valid signals: SIGCANCEL, SIGTERM, SIGSTOP, SIGCONT, SIGKILL", other)),
    };

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.signal_process(pid, signal)?;
            return Ok(format!("Signal {} sent to process {}.", signal_str.to_uppercase(), pid));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "signal_process",
        description: "Send a POSIX-like signal to a child or descendant process. SIGCANCEL=request cooperative cancellation of current turn/tool work, SIGTERM=request graceful termination, SIGSTOP=pause execution, SIGCONT=resume paused process, SIGKILL=immediate forced termination (cascades to grandchildren).",
        parameters: params_signal_process,
        execute: execute_signal_process,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// --- Process Group ---

fn params_set_process_group() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pid": { "type": "integer", "description": "PID of the process to assign to a group." },
            "pgid": { "type": "integer", "description": "Process Group ID to assign. Use a new ID to create a group." }
        },
        "required": ["pid", "pgid"]
    })
}

fn execute_set_process_group(args: &Value) -> Result<String, String> {
    let pid = args["pid"].as_u64().ok_or("Missing 'pid'.")?;
    let pgid = args["pgid"].as_u64().ok_or("Missing 'pgid'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.set_process_group(pid, pgid)?;
            return Ok(format!("Process {} assigned to group {}.", pid, pgid));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "set_process_group",
        description: "Assign a process to a process group. Processes in the same group can be signaled together with signal_process_group.",
        parameters: params_set_process_group,
        execute: execute_set_process_group,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_signal_process_group() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pgid": { "type": "integer", "description": "Process Group ID to signal." },
            "signal": {
                "type": "string",
                "enum": ["SIGTERM", "SIGSTOP", "SIGCONT", "SIGKILL"],
                "description": "Signal to send to all processes in the group."
            }
        },
        "required": ["pgid", "signal"]
    })
}

fn execute_signal_process_group(args: &Value) -> Result<String, String> {
    let pgid = args["pgid"].as_u64().ok_or("Missing 'pgid'.")?;
    let signal_str = args["signal"].as_str().ok_or("Missing 'signal'.")?;
    let signal = match signal_str.to_uppercase().as_str() {
        "SIGTERM" => aios_kernel::kernel::Signal::SigTerm,
        "SIGSTOP" => aios_kernel::kernel::Signal::SigStop,
        "SIGCONT" => aios_kernel::kernel::Signal::SigCont,
        "SIGKILL" => aios_kernel::kernel::Signal::SigKill,
        other => return Err(format!("Unknown signal: {}", other)),
    };
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let count = os.signal_process_group(pgid, signal)?;
            return Ok(format!("Signal {} sent to {} processes in group {}.", signal_str.to_uppercase(), count, pgid));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "signal_process_group",
        description: "Send a signal to all processes in a process group. Useful for batch operations like stopping or terminating a group of related processes.",
        parameters: params_signal_process_group,
        execute: execute_signal_process_group,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// --- Shared Memory IPC ---

fn params_shm_create() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "key": { "type": "string", "description": "Unique key for the shared memory region." },
            "value": { "type": "string", "description": "Initial value to store." }
        },
        "required": ["key", "value"]
    })
}

fn execute_shm_create(args: &Value) -> Result<String, String> {
    let key = args["key"].as_str().ok_or("Missing 'key'.")?;
    let value = args["value"].as_str().ok_or("Missing 'value'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.shm_create(key.to_string(), value.to_string())?;
            return Ok(format!("Shared memory '{}' created.", key));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "shm_create",
        description: "Create a new shared memory region with a key-value pair. Other processes can read and write this data. Fails if the key already exists.",
        parameters: params_shm_create,
        execute: execute_shm_create,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_shm_read() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "key": { "type": "string", "description": "Key of the shared memory region to read." }
        },
        "required": ["key"]
    })
}

fn execute_shm_read(args: &Value) -> Result<String, String> {
    let key = args["key"].as_str().ok_or("Missing 'key'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let os = os.lock().unwrap();
            match os.shm_read(key) {
                Ok(value) => Ok(value),
                Err(aios_kernel::kernel::ShmReadError::NotFound) => {
                    Err(format!("Shared memory key '{}' not found.", key))
                }
                Err(aios_kernel::kernel::ShmReadError::PermissionDenied { owner_pid }) => {
                    Err(format!(
                        "Permission denied: cannot read shared memory key '{}' (owner: {}).",
                        key, owner_pid
                    ))
                }
                Err(aios_kernel::kernel::ShmReadError::Corrupted { expected_checksum, actual_checksum }) => {
                    match os.shm_read_degraded(key) {
                        Some(degraded) => Ok(degraded),
                        None => Err(format!(
                            "Data corrupted in shared memory key '{}' (expected: {:#x}, actual: {:#x}).",
                            key, expected_checksum, actual_checksum
                        )),
                    }
                }
                Err(aios_kernel::kernel::ShmReadError::OwnerTerminated { owner_pid }) => {
                    match os.shm_read_degraded(key) {
                        Some(degraded) => Ok(degraded),
                        None => Err(format!(
                            "Owner process {} of shared memory key '{}' has terminated.",
                            owner_pid, key
                        )),
                    }
                }
            }
        } else {
            Err("OS not initialized.".to_string())
        }
    } else {
        Err("OS not initialized.".to_string())
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "shm_read",
        description: "Read the value of a shared memory region by key. Returns degraded data with warning if owner has terminated or data is corrupted. Fails only if key not found or permission denied with no fallback.",
        parameters: params_shm_read,
        execute: execute_shm_read,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_shm_write() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "key": { "type": "string", "description": "Key of the shared memory region to update." },
            "value": { "type": "string", "description": "New value to write." }
        },
        "required": ["key", "value"]
    })
}

fn execute_shm_write(args: &Value) -> Result<String, String> {
    let key = args["key"].as_str().ok_or("Missing 'key'.")?;
    let value = args["value"].as_str().ok_or("Missing 'value'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.shm_write(key.to_string(), value.to_string())?;
            return Ok(format!("Shared memory '{}' updated.", key));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "shm_write",
        description: "Update the value of an existing shared memory region. Fails if the key does not exist (use shm_create first).",
        parameters: params_shm_write,
        execute: execute_shm_write,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

fn params_shm_delete() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "key": { "type": "string", "description": "Key of the shared memory region to delete." }
        },
        "required": ["key"]
    })
}

fn execute_shm_delete(args: &Value) -> Result<String, String> {
    let key = args["key"].as_str().ok_or("Missing 'key'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.shm_delete(key)?;
            return Ok(format!("Shared memory '{}' deleted.", key));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "shm_delete",
        description: "Delete a shared memory region by key.",
        parameters: params_shm_delete,
        execute: execute_shm_delete,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// --- Working Directory ---

fn params_set_working_dir() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "dir": { "type": "string", "description": "Absolute path to set as the working directory for the current process." }
        },
        "required": ["dir"]
    })
}

fn execute_set_working_dir(args: &Value) -> Result<String, String> {
    let dir = args["dir"].as_str().ok_or("Missing 'dir'.")?;
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.set_working_dir(std::path::PathBuf::from(dir))?;
            return Ok(format!("Working directory set to '{}'.", dir));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "set_working_dir",
        description: "Set the working directory for the current process. Child processes will inherit this directory.",
        parameters: params_set_working_dir,
        execute: execute_set_working_dir,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

// --- Daemon Process ---

fn params_spawn_daemon() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "Short name for the daemon process." },
            "goal": { "type": "string", "description": "The goal or task for the daemon to accomplish on each run." },
            "priority": { "type": "integer", "description": "Process priority. Default is 10." },
            "quota_turns": { "type": "integer", "description": "Max turns per run. Default is 10." },
            "max_restarts": { "type": "integer", "description": "Maximum number of automatic restarts when the daemon terminates. Default is 3." }
        },
        "required": ["name", "goal"]
    })
}

fn execute_spawn_daemon(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing 'name'.")?;
    let goal = args["goal"].as_str().ok_or("Missing 'goal'.")?;
    let priority = args["priority"].as_u64().unwrap_or(10) as u8;
    let quota_turns = args["quota_turns"].as_u64().unwrap_or(10) as usize;
    let max_restarts = args["max_restarts"].as_u64().unwrap_or(3) as usize;

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            let current_pid = os.current_process_id();
            let pid = os.spawn_daemon(
                current_pid,
                name.to_string(),
                goal.to_string(),
                priority,
                quota_turns,
                max_restarts,
            )?;
            return Ok(format!("Daemon process spawned. PID: {}, Name: {}, Max restarts: {}. Will auto-restart on termination.", pid, name, max_restarts));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "spawn_daemon",
        description: "Spawn a daemon process that automatically restarts when it terminates (up to max_restarts times). Useful for long-running background services like file watchers or knowledge indexers.",
        parameters: params_spawn_daemon,
        execute: execute_spawn_daemon,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core", "executor"],
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use aios_kernel::kernel::new_shared_kernel;
    use aios_kernel::local::LocalOS;
    use serde_json::json;

    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn with_test_kernel<T>(f: impl FnOnce(SharedKernel) -> T) -> T {
        let _guard = TEST_LOCK.lock().unwrap();
        let kernel = new_shared_kernel(LocalOS::new());
        init_os_tools_globals(kernel.clone());
        let result = f(kernel);
        if let Ok(mut global) = GLOBAL_OS.lock() {
            *global = None;
        }
        result
    }

    #[test]
    fn ps_ipc_defaults_to_hanging_result_pipes() {
        with_test_kernel(|kernel| {
            {
                let mut os = kernel.lock().unwrap();
                os.channel_create(Some(7), 1, "general:mailbox".to_string());
                os.channel_create_tagged_with_holders(
                    Some(42),
                    1,
                    "task_result:task_1".to_string(),
                    ChannelOwnerTag::TaskResult,
                    vec![
                        "task_result.producer".to_string(),
                        "task_result.consumer".to_string(),
                    ],
                );
                os.channel_create_tagged_with_holders(
                    Some(42),
                    4,
                    "async_tool_result:tool_1".to_string(),
                    ChannelOwnerTag::AsyncToolResult,
                    vec!["async_tool.consumer".to_string()],
                );
                let done = os.channel_create_tagged_with_holders(
                    Some(42),
                    1,
                    "task_result:done".to_string(),
                    ChannelOwnerTag::TaskResult,
                    Vec::new(),
                );
                os.channel_close(None, done).unwrap();
            }

            let output = execute_ps_ipc(&json!({})).unwrap();
            assert!(output.contains("task_result:task_1"));
            assert!(output.contains("task_result.producer, task_result.consumer"));
            assert!(output.contains("async_tool_result:tool_1"));
            assert!(output.contains("async_tool.consumer"));
            assert!(!output.contains("general:mailbox"));
            assert!(!output.contains("task_result:done"));
        });
    }

    #[test]
    fn ps_ipc_all_scope_includes_general_channels() {
        with_test_kernel(|kernel| {
            {
                let mut os = kernel.lock().unwrap();
                os.channel_create(Some(9), 2, "general:mailbox".to_string());
            }

            let output = execute_ps_ipc(&json!({
                "scope": "all",
                "only_hanging": false
            }))
            .unwrap();
            assert!(output.contains("general:mailbox"));
            assert!(output.contains("general"));
        });
    }
}
