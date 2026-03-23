fn main() {
    if let Err(err) = rust_tools::ai::run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
