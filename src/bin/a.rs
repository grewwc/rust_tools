mod ai;
pub use rust_tools::*;
#[tokio::main]
async fn main() {
    if let Err(err) = ai::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
