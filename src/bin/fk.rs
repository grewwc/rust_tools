#[path = "fk/cli.rs"]
mod cli;
#[path = "fk/matcher.rs"]
mod matcher;
#[path = "fk/output.rs"]
mod output;
#[path = "fk/search.rs"]
mod search;

fn main() {
    let Some(opts) = cli::parse_from_env() else {
        return;
    };

    let _allowed = search::configure_terminalw(&opts);
    let found = match search::run(&opts) {
        Ok(v) => v,
        Err(e) => {
            if opts.verbose {
                eprintln!("{e}");
            }
            return;
        }
    };
    search::print_summary(found);
}

