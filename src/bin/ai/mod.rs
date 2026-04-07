#![allow(dead_code)]
mod cli;
mod config;
mod driver;
mod files;
mod history;
mod knowledge;
mod mcp;
mod model_names;
mod models;
mod prompt;
mod request;
mod skills;
mod stream;
mod tools;
mod types;

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
