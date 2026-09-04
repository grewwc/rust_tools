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
mod middleware;
mod model_names;
mod models;
mod persona;
mod pipeline;
mod ports;
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

/// Synchronous entry point: decides whether to enter background mode before creating
/// the tokio runtime. Background mode does not fork: the parent re-execs a fresh
/// `a --daemon-child <session>` as the daemon via posix_spawn (see
/// `background::spawn_daemon_child`), avoiding fork carrying the parent's
/// half-initialized CF/os_log/objc state into the child.
pub fn entry() -> Result<(), Box<dyn std::error::Error>> {
    // Internal daemon marker: the background-mode parent prepends
    // `--daemon-child <session_id>` to the arguments; strip it here, parse the rest
    // normally, then route to the daemon child entry point
    // (`background::run_background_child`).
    let mut args: Vec<String> = std::env::args().collect();
    let daemon_session = if args.get(1).map(String::as_str) == Some("--daemon-child") {
        let sid = args.get(2).cloned();
        args.drain(1..3);
        sid
    } else {
        None
    };

    if daemon_session.is_some() {
        // The freshly exec'd child must establish its own session before parsing the
        // CLI / creating the runtime. `process_group(0)` is not a substitute: it only
        // sets the PGID and would make `setsid` fail.
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
