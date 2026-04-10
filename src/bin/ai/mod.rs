#![allow(dead_code)]
mod agents;
mod cli;
mod code_discovery_policy;
mod config;
pub mod config_schema;
mod errors;
mod driver;
mod files;
mod history;
mod knowledge;
mod mcp;
mod model_names;
mod models;
mod provider;
mod prompt;
mod request;
mod skills;
mod stream;
mod theme;
mod tools;
mod types;

pub(in crate::ai) use rust_tools_macros::{agent_hang_debug, agent_hang_span};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_support {
    use std::sync::{LazyLock, Mutex};

    pub(super) static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
}

pub use driver::run;

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
