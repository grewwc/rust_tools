#![allow(dead_code)]
mod agents;
mod background;
mod cli;
mod code_discovery_policy;
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
mod prompt;
mod provider;
mod request;
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
/// 后台模式需要先 daemonize（fork）再构建 runtime——在多线程 runtime 启动后
/// fork 会丢失 worker 线程导致死锁，因此必须在 runtime 之前完成 detach。
pub fn entry() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli::parse_cli_args(std::env::args());
    if cli.background {
        return background::run_background(cli);
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
