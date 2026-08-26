mod ai;
pub use rust_tools::*;

fn main() {
    // Use the synchronous entry point so that background mode (-bg) can finish
    // daemonizing before the tokio runtime is created.
    if let Err(err) = ai::entry() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
