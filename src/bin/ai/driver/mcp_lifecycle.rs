//! MCP lifecycle: preload task management, initialization, and status display.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::io::Write;
use std::sync::atomic::Ordering;

use super::{
    McpConfigProbe, McpInitReport, PreparedMcpInit, apply_prepared_mcp_init,
    prepare_mcp_initialization_from_path, prepare_mcp_initialization_from_path_interruptible,
    print, signal,
};
use crate::ai::{mcp::SharedMcpClient, types::App};

pub(super) fn apply_prepared_mcp_with_shared_client(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    prepared: PreparedMcpInit,
) -> McpInitReport {
    let mut guard = mcp_client.lock().unwrap_or_else(|err| err.into_inner());
    apply_prepared_mcp_init(app, &mut guard, prepared)
}

pub(super) fn announce_mcp_loading_if_needed(
    mcp_probe: &McpConfigProbe,
    mcp_initialized: bool,
    mcp_loading_announced: &mut bool,
) {
    if *mcp_loading_announced || mcp_initialized || !mcp_probe.exists {
        return;
    }
    print!(
        "{}",
        print::format_section_header(
            "mcp",
            Some(&format!(
                "{} configured servers · loading...",
                mcp_probe.server_count
            ))
        )
    );
    std::io::stdout().flush().ok();
    *mcp_loading_announced = true;
}

pub(super) fn emit_mcp_loaded_header(report: &McpInitReport, mcp_loading_announced: &mut bool) {
    if !report.loaded {
        return;
    }

    let header = print::format_section_header(
        "mcp",
        Some(&format!(
            "{} servers, {} tools",
            report.server_count, report.tool_count
        )),
    );
    if *mcp_loading_announced {
        print!("\r\x1b[2K{}\n", header);
    } else {
        println!("{header}");
    }
}

pub(super) async fn finalize_mcp_preload_task(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    task: tokio::task::JoinHandle<Option<PreparedMcpInit>>,
) -> Option<McpInitReport> {
    match task.await {
        Ok(Some(prepared)) => Some(apply_prepared_mcp_with_shared_client(
            app, mcp_client, prepared,
        )),
        Ok(None) => None,
        Err(err) => {
            if app.shutdown.load(Ordering::Relaxed) || signal::request_interrupt_ready() {
                return None;
            }
            eprintln!("[mcp] background preload task failed: {}", err);
            let fallback =
                prepare_mcp_initialization_from_path(mcp_probe.config_path.clone()).await;
            Some(apply_prepared_mcp_with_shared_client(
                app, mcp_client, fallback,
            ))
        }
    }
}

pub(super) async fn try_finalize_mcp_preload(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    mcp_initialized: &mut bool,
    mcp_loading_announced: &mut bool,
    mcp_preload_task: &mut Option<tokio::task::JoinHandle<Option<PreparedMcpInit>>>,
) {
    if *mcp_initialized || !mcp_probe.exists {
        return;
    }

    let Some(task) = mcp_preload_task.as_ref() else {
        return;
    };
    if !task.is_finished() {
        return;
    }

    let task = mcp_preload_task.take().unwrap();
    let Some(report) = finalize_mcp_preload_task(app, mcp_client, mcp_probe, task).await else {
        return;
    };

    *mcp_initialized = true;
    emit_mcp_loaded_header(&report, mcp_loading_announced);
}

pub(super) async fn ensure_mcp_initialized_for_turn(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    mcp_initialized: &mut bool,
    mcp_loading_announced: &mut bool,
    mcp_preload_task: &mut Option<tokio::task::JoinHandle<Option<PreparedMcpInit>>>,
    show_status: bool,
) {
    if *mcp_initialized || !mcp_probe.exists {
        return;
    }

    if show_status {
        announce_mcp_loading_if_needed(mcp_probe, *mcp_initialized, mcp_loading_announced);
    }

    let report = if let Some(task) = mcp_preload_task.take() {
        finalize_mcp_preload_task(app, mcp_client, mcp_probe, task).await
    } else {
        let prepared = prepare_mcp_initialization_from_path(mcp_probe.config_path.clone()).await;
        Some(apply_prepared_mcp_with_shared_client(
            app, mcp_client, prepared,
        ))
    };

    let Some(report) = report else {
        return;
    };

    *mcp_initialized = true;
    if show_status {
        emit_mcp_loaded_header(&report, mcp_loading_announced);
    }
}

pub(super) fn spawn_mcp_preload_task(
    config_path: String,
) -> tokio::task::JoinHandle<Option<PreparedMcpInit>> {
    tokio::spawn(async move {
        let interrupt_futex = signal::alloc_interrupt_futex("mcp_preload_interrupt");
        let prepared =
            prepare_mcp_initialization_from_path_interruptible(config_path, interrupt_futex).await;
        if let Some(addr) = interrupt_futex {
            signal::destroy_interrupt_futex(addr);
        }
        prepared
    })
}

pub(super) fn should_preload_mcp(_one_shot_mode: bool, mcp_probe: &McpConfigProbe) -> bool {
    mcp_probe.exists
}
