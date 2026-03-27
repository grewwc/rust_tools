mod ai;
pub use rust_tools::*;

fn main() {
    if let Err(err) = ai::run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
