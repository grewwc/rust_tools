use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local, LocalResult, NaiveDateTime, TimeZone};
use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(
    about = "Convert unix timestamps and datetimes (go_tools tt compatible)",
    after_help = "usage: tt 1603372219690"
)]
struct Cli {
    #[arg(short = 'd', action = ArgAction::SetTrue, help = "show current date in date format")]
    d: bool,

    #[arg(long = "dt", action = ArgAction::SetTrue, help = "show current date in datetime format")]
    dt: bool,

    #[arg(value_name = "INPUT")]
    input: Option<String>,
}

fn normalize_args(args: impl Iterator<Item = String>) -> Vec<String> {
    args.map(|arg| {
        let bytes = arg.as_bytes();
        if bytes.len() > 2 && bytes[0] == b'-' && bytes[1] != b'-' && bytes[1].is_ascii_alphabetic()
        {
            format!("-{arg}")
        } else {
            arg
        }
    })
    .collect()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn format_go_like(dt: DateTime<Local>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S %z %Z").to_string()
}

fn parse_datetime_local(input: &str) -> Result<DateTime<Local>, String> {
    let naive =
        NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S").map_err(|e| e.to_string())?;
    match Local.from_local_datetime(&naive) {
        LocalResult::Single(dt) => Ok(dt),
        LocalResult::Ambiguous(a, _) => Ok(a),
        LocalResult::None => Err("invalid local datetime".to_string()),
    }
}

fn parse_unix_like(input: i64) -> DateTime<Local> {
    let threshold = Local
        .with_ymd_and_hms(2500, 1, 1, 0, 0, 0)
        .single()
        .unwrap_or_else(|| Local.timestamp_opt(0, 0).single().unwrap());

    let as_secs = Local.timestamp_opt(input, 0).single();
    if let Some(dt) = as_secs
        && dt <= threshold
    {
        return dt;
    }

    let secs = input / 1000;
    Local
        .timestamp_opt(secs, 0)
        .single()
        .unwrap_or_else(|| Local.timestamp_opt(0, 0).single().unwrap())
}

fn main() {
    let cli = Cli::parse_from(normalize_args(std::env::args()));

    if cli.d {
        println!("{}", Local::now().format("%Y-%m-%d"));
        return;
    }

    if cli.dt {
        println!("{}", Local::now().format("%Y-%m-%d %H:%M:%S"));
        return;
    }

    let Some(input) = cli
        .input
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        println!("{} ms", now_millis());
        return;
    };

    match input.parse::<i64>() {
        Ok(unix) => {
            let dt = parse_unix_like(unix);
            println!("{}", format_go_like(dt));
        }
        Err(_) => match parse_datetime_local(input) {
            Ok(dt) => println!("{} s", dt.timestamp_millis() / 1000),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_args_upgrades_single_dash_multi_letter_flag() {
        let argv =
            normalize_args(["tt".to_string(), "-dt".to_string(), "-d".to_string()].into_iter());
        assert!(argv.iter().any(|a| a == "--dt"));
        assert!(argv.iter().any(|a| a == "-d"));
    }

    #[test]
    fn parse_unix_like_treats_large_value_as_millis() {
        let dt = parse_unix_like(1_603_372_219_690);
        assert_eq!(dt.timestamp(), 1_603_372_219);
    }

    #[test]
    fn parse_datetime_local_parses_seconds_precision() {
        let dt = parse_datetime_local("2020-01-02 03:04:05").unwrap();
        assert_eq!(dt.timestamp_millis() / 1000, dt.timestamp());
    }
}
