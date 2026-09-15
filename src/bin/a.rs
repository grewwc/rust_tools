mod ai;
pub use rust_tools::*;

fn main() {
    // rustls has no default crypto provider (reqwest uses rustls-no-provider
    // to keep aws-lc-rs out of the binary); install ring before any TLS use.
    rust_tools::ensure_rustls_provider();
    // Use the synchronous entry point so that background mode (-bg) can finish
    // daemonizing before the tokio runtime is created.
    if let Err(err) = ai::entry() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
