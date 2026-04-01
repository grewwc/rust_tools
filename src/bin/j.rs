use std::path::Path;

use clap::Parser;
use rust_tools::{clipboardw::string_content, jsonw};
use serde_json::Value;

#[derive(Parser)]
#[command(about = "JSON diff/format utilities (go_tools jsondiff compatible subset)")]
struct Cli {
    #[arg(short = 'f', value_name = "FILE", num_args = 0..=1, default_missing_value = "")]
    format: Option<String>,

    #[arg(short = 'o', default_value = "", value_name = "FILE")]
    output: String,

    #[arg(long, default_value_t = false, help = "sort arrays before diff")]
    sort: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "multi-thread (accepted, currently ignored)"
    )]
    mt: bool,

    #[arg(short = 'p', default_value_t = false, help = "print result to stdout")]
    print: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "escape clipboard/stdin as JSON string"
    )]
    quote: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "remove \\n and \\r when quoting/printing"
    )]
    oneline: bool,

    #[arg(long, default_value_t = false, help = "print JSON length in clipboard")]
    len: bool,

    #[arg(value_name = "OLD NEW", num_args = 0..=2)]
    files: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cli.quote {
        let mut content = string_content::get_clipboard_content();
        if cli.oneline {
            content = content.replace(['\n', '\r'], "");
        }
        if content.is_empty() {
            content = read_stdin_all();
        }
        let quoted = serde_json::to_string(&content).unwrap_or_default();
        println!("{quoted}");
        return;
    }

    if cli.oneline && cli.format.is_none() && !cli.len && cli.files.is_empty() {
        let content = string_content::get_clipboard_content().replace(['\n', '\r'], "");
        println!("{content}");
        return;
    }

    if cli.len {
        let content = string_content::get_clipboard_content();
        match jsonw::Json::from_str(&content, jsonw::ParseOptions::default()) {
            Ok(j) => {
                let msg = if j.is_array() { "Array" } else { "Object" };
                println!("{msg}: {}", j.len());
            }
            Err(_) => {
                println!("clipboard content is not a valid json");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(fname) = cli.format.as_deref() {
        let options = jsonw::ParseOptions::default();
        let j = if fname.is_empty() {
            jsonw::Json::from_clipboard(options).unwrap()
        } else {
            jsonw::Json::from_file(fname, options).unwrap()
        };

        let mut formatted = j.to_pretty_string();
        if cli.oneline {
            formatted = formatted.split_whitespace().collect::<String>();
        }
        println!("{}", formatted.chars().take(1024).collect::<String>());

        let output_fname = if fname.is_empty() {
            "_f.json".to_string()
        } else {
            format!("{}_f.json", base_no_ext(fname))
        };
        println!("write file to {output_fname}");
        j.to_file(output_fname, true).unwrap();
        return;
    }

    if cli.files.len() == 2 {
        let options = jsonw::ParseOptions::default();
        let old = jsonw::Json::from_file(&cli.files[0], options).unwrap();
        let new = jsonw::Json::from_file(&cli.files[1], options).unwrap();

        let diff = jsonw::diff_json(old.value(), new.value(), cli.sort);
        let diff_value = serde_json::to_value(diff).unwrap_or(Value::Null);
        let diff_json = jsonw::Json::new(diff_value);

        if cli.print {
            println!("{}", diff_json.to_pretty_string());
        }

        let fname = if cli.output.is_empty() {
            format!(
                "{}_{}_diff.json",
                base_no_ext(&cli.files[0]),
                base_no_ext(&cli.files[1])
            )
        } else {
            cli.output.clone()
        };
        diff_json.to_file(&fname, true).unwrap();
        println!("write to {fname}");
        return;
    }

    eprintln!("usage: j old.json new.json  |  j -f [file]  |  j --quote [--oneline]  |  j --len");
}

fn base_no_ext(path: &str) -> String {
    let p = Path::new(path);
    let file = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();
    let mut name = file;
    if let Some(idx) = name.rfind('.') {
        name.truncate(idx);
    }
    name.replace(' ', "")
}

fn read_stdin_all() -> String {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).ok();
    buf
}
