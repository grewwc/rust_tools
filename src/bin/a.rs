mod ai;
pub use rust_tools::*;

fn main() {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = rt.block_on(ai::run()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
