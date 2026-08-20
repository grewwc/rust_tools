#![allow(dead_code)]
mod agents;
mod background;
mod cli;
mod config;
pub mod config_schema;
mod driver;
mod errors;
mod files;
mod history;
mod knowledge;
mod mcp;
mod model_names;
mod models;
mod persona;
mod pipeline;
mod ports;
mod middleware;
mod prompt;
mod provider;
mod request;
mod request_protocol;
mod skills;
mod stream;
mod theme;
pub(crate) mod tools;
mod types;

pub(in crate::ai) use rust_tools_macros::{agent_hang_debug, agent_hang_span};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_support {
    use std::sync::{LazyLock, Mutex};

    pub(super) static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
}

/// 同步入口：在创建 tokio runtime 之前判断是否进入后台模式。
/// 后台模式不 fork：父进程用 posix_spawn 重新 exec 一个全新的
/// `a --daemon-child <session>` 作为 daemon（见 `background::spawn_daemon_child`），
/// 避免 fork 把父进程半初始化的 CF/os_log/objc 状态带进子进程。
pub fn entry() -> Result<(), Box<dyn std::error::Error>> {
    // 内部 daemon 标记：后台模式父进程在参数最前面加上
    // `--daemon-child <session_id>`，这里剥离后按正常参数解析，再路由到
    // daemon 子进程入口（`background::run_background_child`）。
    let mut args: Vec<String> = std::env::args().collect();
    let daemon_session = if args.get(1).map(String::as_str) == Some("--daemon-child") {
        let sid = args.get(2).cloned();
        args.drain(1..3);
        sid
    } else {
        None
    };

    if daemon_session.is_some() {
        // 新 exec 的子进程必须在解析 CLI / 创建 runtime 前建立独立 session。
        // 不能用 `process_group(0)` 代替：它只设置 PGID，且会使 `setsid` 失败。
        background::detach_daemon_session()?;
    }

    let cli = cli::parse_cli_args(args.into_iter());

    if let Some(session_id) = daemon_session {
        return background::run_background_child(cli, session_id);
    }
    if cli.background {
        return background::run_background(cli);
    }
    if let Some(ref session_id) = cli.stop_session {
        if session_id.is_empty() {
            return Err("--stop 需要指定 session id，例如：a --stop <session-id>".into());
        }
        eprintln!("[stop] 正在停止 session {session_id}...");
        background::stop_background(session_id)?;
        return Ok(());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(driver::run_with_cli(cli))
}

mod ff_embed {
    pub mod cli {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/ff/cli.rs"));
    }
    pub mod exclude {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/bin/ff/exclude.rs"
        ));
    }
    pub mod output {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/ff/output.rs"));
    }
    pub mod search {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/ff/search.rs"));
    }
}
