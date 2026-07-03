mod ai;
pub use rust_tools::*;

fn main() {
    // 使用同步入口，以便后台模式 (-bg) 在创建 tokio runtime 之前完成 daemonize。
    if let Err(err) = ai::entry() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
