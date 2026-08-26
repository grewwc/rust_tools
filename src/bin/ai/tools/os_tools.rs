use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};
use aios_kernel::kernel::{ProcessCapabilities, SharedKernel};
use aios_kernel::primitives::{ChannelMetaSnapshot, ChannelOwnerTag};
use serde_json::Value;
use std::sync::LazyLock;
use std::sync::Mutex;

/// Global AIOS kernel handle, injected by the driver at startup via [`init_os_tools_globals`].
///
/// **Synchronous semantics**: `GLOBAL_OS` and [`crate::ai::types::App::os`] hold
/// **the same `Arc<Mutex<Box<dyn Kernel>>>`** (i.e. `SharedKernel`); they are just two ways
/// to obtain the same kernel handle:
/// - `App.os`: used by driver flows (foreground / background loop, turn_runtime)
/// - `GLOBAL_OS`: used by tool implementations (in `os_tools.rs` etc.) because tools have no
///   `App` reference
///
/// Since both wrap the same `Arc<Mutex<...>>`, the two paths take the same lock **exclusively**; this means
/// there is no true deadlock from "two different locks", but it **can trigger `std::sync::Mutex` re-entrant deadlock**
/// (calling into a tool that does `GLOBAL_OS.lock()` from the same thread while holding
/// `app.os.lock()` will block or panic immediately). All synchronous callers must release
/// the `app.os` lock guard before invoking a tool.
///
/// Hot paths should go through [`task_tools::with_os_kernel`](crate::ai::tools::task_tools),
/// which prefers the reference in `DRIVER_CTX` over this static, reducing
/// indirect lookups.
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

fn execute_spawn_process(args: &Value) -> Result<String, String> {
    let name = args["name"]
        .as_str()
        .ok_or("Missing 'name' string parameter.")?;
    let goal = args["goal"]
        .as_str()
        .ok_or("Missing 'goal' string parameter.")?;
    let priority = args["priority"].as_u64().unwrap_or(10) as u8;
    let quota_turns = args["quota_turns"].as_u64().unwrap_or(10) as usize;
    let capabilities = parse_capabilities(args);

    let allowed_tools =
        if let Some(tools_array) = args.get("allowed_tools").and_then(Value::as_array) {
            let mut set = aios_kernel::FastSet::default();
            for tool in tools_array {
                if let Some(tool_name) = tool.as_str() {
                    set.insert(tool_name.to_string());
                }
            }
            if !set.is_empty() {
                for tool_name in crate::ai::tools::baseline_tool_names() {
                    set.insert((*tool_name).to_string());
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
            return Ok(format!(
                "Sub-process spawned successfully. PID: {}, Name: {}. The scheduler will execute it autonomously.",
                pid, name
            ));
        }
    }

    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "spawn_process",
        description: "",

        execute: execute_spawn_process,
        groups: &["builtin", "executor"],
    }
});

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
        description: "",

        execute: execute_sleep_process,
        groups: &["builtin", "executor"],
    }
});

// 2. wait_process

fn execute_wait_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"]
        .as_u64()
        .ok_or("Missing or invalid 'pid' parameter.")?;

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.wait_on(pid)?;
            return Ok(format!(
                "Current process suspended. Will be awakened when PID {} terminates. Note: Do not emit further output in this turn, just yield control.",
                pid
            ));
        }
    }

    Err("OS Scheduler not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "wait_process",
        description: "",

        execute: execute_wait_process,
        groups: &["builtin", "executor"],
    }
});

fn execute_kill_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"]
        .as_u64()
        .ok_or("Missing or invalid 'pid' parameter.")?;
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
        description: "",

        execute: execute_kill_process,
        groups: &["builtin", "executor"],
    }
});

// 3. send_ipc_message

fn execute_send_ipc(args: &Value) -> Result<String, String> {
    let pid = args["pid"]
        .as_u64()
        .ok_or("Missing or invalid 'pid' parameter.")?;
    let message = args["message"]
        .as_str()
        .ok_or("Missing 'message' string parameter.")?;

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
        description: "",

        execute: execute_send_ipc,
        groups: &["builtin", "executor"],
    }
});

fn execute_reap_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"]
        .as_u64()
        .ok_or("Missing or invalid 'pid' parameter.")?;
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
        description: "",

        execute: execute_reap_process,
        groups: &["builtin", "executor"],
    }
});

// 4. read_mailbox

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
        description: "",

        execute: execute_read_mailbox,
        groups: &["builtin", "executor"],
    }
});

// 5. env tools

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
        description: "",

        execute: execute_set_env,
        groups: &["builtin", "executor"],
    }
});

fn execute_ps_processes(_args: &Value) -> Result<String, String> {
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let os = os.lock().unwrap();
            let procs = os.list_processes();
            if procs.is_empty() {
                return Ok("No processes in the system.".to_string());
            }
            let mut lines = vec![
                "PID   PPID   PGID  State       Pri  Quota  Used  Tools  Ticks  Daemon  Name"
                    .to_string(),
            ];
            for p in &procs {
                let ppid = p
                    .parent_pid
                    .map(|id| id.to_string())
                    .unwrap_or("-".to_string());
                let pgid = p
                    .process_group
                    .map(|id| id.to_string())
                    .unwrap_or("-".to_string());
                let state = match &p.state {
                    aios_kernel::kernel::ProcessState::Ready => "Ready",
                    aios_kernel::kernel::ProcessState::Running => "Running",
                    aios_kernel::kernel::ProcessState::Waiting { .. } => "Waiting",
                    aios_kernel::kernel::ProcessState::Sleeping { .. } => "Sleeping",
                    aios_kernel::kernel::ProcessState::Stopped => "Stopped",
                    aios_kernel::kernel::ProcessState::Terminated => "Term",
                };
                let daemon = if p.is_daemon {
                    format!("{}({}/{})", "Y", p.restart_count, p.max_restarts)
                } else {
                    "N".to_string()
                };
                lines.push(format!(
                    "{:<5} {:<6} {:<5} {:<12} {:<4} {:<6} {:<5} {:<6} {:<6} {:<8} {}",
                    p.pid,
                    ppid,
                    pgid,
                    state,
                    p.priority,
                    p.quota_turns,
                    p.turns_used,
                    p.tool_calls_used,
                    p.created_at_tick,
                    daemon,
                    p.name
                ));
            }
            return Ok(lines.join("\n"));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "ps_processes",
        description: "",

        execute: execute_ps_processes,
        groups: &["builtin", "executor"],
    }
});

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
                    ("result_pipes", true) => "No hanging result pipes in the system.".to_string(),
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
        description: "",

        execute: execute_ps_ipc,
        groups: &["builtin", "executor"],
    }
});

fn execute_signal_process(args: &Value) -> Result<String, String> {
    let pid = args["pid"]
        .as_u64()
        .ok_or("Missing or invalid 'pid' parameter.")?;
    let signal_str = args["signal"]
        .as_str()
        .ok_or("Missing 'signal' parameter.")?;
    let signal = match signal_str.to_uppercase().as_str() {
        "SIGCANCEL" => aios_kernel::kernel::Signal::SigCancel,
        "SIGTERM" => aios_kernel::kernel::Signal::SigTerm,
        "SIGSTOP" => aios_kernel::kernel::Signal::SigStop,
        "SIGCONT" => aios_kernel::kernel::Signal::SigCont,
        "SIGKILL" => aios_kernel::kernel::Signal::SigKill,
        other => {
            return Err(format!(
                "Unknown signal: {}. Valid signals: SIGCANCEL, SIGTERM, SIGSTOP, SIGCONT, SIGKILL",
                other
            ));
        }
    };

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os) = guard.as_ref() {
            let mut os = os.lock().unwrap();
            os.signal_process(pid, signal)?;
            return Ok(format!(
                "Signal {} sent to process {}.",
                signal_str.to_uppercase(),
                pid
            ));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "signal_process",
        description: "",

        execute: execute_signal_process,
        groups: &["builtin", "executor"],
    }
});

// --- Process Group ---

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
        description: "",

        execute: execute_set_process_group,
        groups: &["builtin", "executor"],
    }
});

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
            return Ok(format!(
                "Signal {} sent to {} processes in group {}.",
                signal_str.to_uppercase(),
                count,
                pgid
            ));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "signal_process_group",
        description: "",

        execute: execute_signal_process_group,
        groups: &["builtin", "executor"],
    }
});

// --- Shared Memory IPC ---

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
        description: "",

        execute: execute_shm_create,
        groups: &["builtin", "executor"],
    }
});

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
                Err(aios_kernel::kernel::ShmReadError::Corrupted {
                    expected_checksum,
                    actual_checksum,
                }) => match os.shm_read_degraded(key) {
                    Some(degraded) => Ok(degraded),
                    None => Err(format!(
                        "Data corrupted in shared memory key '{}' (expected: {:#x}, actual: {:#x}).",
                        key, expected_checksum, actual_checksum
                    )),
                },
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
        description: "",

        execute: execute_shm_read,
        groups: &["builtin", "executor"],
    }
});

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
        description: "",

        execute: execute_shm_write,
        groups: &["builtin", "executor"],
    }
});

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
        description: "",

        execute: execute_shm_delete,
        groups: &["builtin", "executor"],
    }
});

// --- Working Directory ---

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
        description: "",

        execute: execute_set_working_dir,
        groups: &["builtin", "executor"],
    }
});

// --- Daemon Process ---

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
            return Ok(format!(
                "Daemon process spawned. PID: {}, Name: {}, Max restarts: {}. Will auto-restart on termination.",
                pid, name, max_restarts
            ));
        }
    }
    Err("OS not initialized.".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "spawn_daemon",
        description: "",

        execute: execute_spawn_daemon,
        groups: &["builtin", "executor"],
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

    #[test]
    fn spawn_process_schema_exposes_signal_capability() {
        assert_eq!(
            crate::ai::tools::registry::tool_metadata::tool_parameters("spawn_process")["properties"]
                ["capabilities"]["properties"]["signal"],
            serde_json::json!({
                "type": "boolean",
                "description": "Allow signaling child or descendant processes."
            })
        );
    }

    #[test]
    fn spawn_process_whitelist_keeps_baseline_tools_available() {
        with_test_kernel(|kernel| {
            let root_pid = {
                let mut os = kernel.lock().unwrap();
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None)
            };

            let output = execute_spawn_process(&json!({
                "name": "child",
                "goal": "inspect code",
                "allowed_tools": ["execute_command"]
            }))
            .expect("spawn_process should succeed");
            assert!(output.contains("PID: 2"));

            let os = kernel.lock().unwrap();
            let child = os.get_process(2).expect("child process should exist");
            assert_eq!(child.parent_pid, Some(root_pid));
            assert!(child.allowed_tools.contains("execute_command"));
            for tool_name in crate::ai::tools::baseline_tool_names() {
                assert!(
                    child.allowed_tools.contains(*tool_name),
                    "baseline tool '{tool_name}' should be auto-allowed"
                );
            }
        });
    }

    #[test]
    fn spawn_process_empty_whitelist_stays_unrestricted() {
        with_test_kernel(|kernel| {
            {
                let mut os = kernel.lock().unwrap();
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
            }

            execute_spawn_process(&json!({
                "name": "child",
                "goal": "inspect code",
                "allowed_tools": []
            }))
            .expect("spawn_process should succeed");

            let os = kernel.lock().unwrap();
            let child = os.get_process(2).expect("child process should exist");
            assert!(
                child.allowed_tools.is_empty(),
                "empty whitelist should remain unrestricted"
            );
        });
    }
}
