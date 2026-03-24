#[path = "ff/cli.rs"]
mod cli;
#[path = "ff/exclude.rs"]
mod exclude;
#[path = "ff/output.rs"]
mod output;
#[path = "ff/search.rs"]
mod search;

fn main() {
    let Some(opts) = cli::parse_from_env() else {
        return;
    };

    let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
        .worker_threads((num_cpus::get()/2).max(1))
        .enable_io()
        .enable_time()
        .build()
    else {
        return;
    };

    let found = rt.block_on(search::run_async(&opts));
    if found > 1 && opts.verbose {
        let summary = format!("{found} matches found");
        println!("{}", "-".repeat(summary.len()));
        println!("{} matches found", found.min(opts.num_print));
    }
}
