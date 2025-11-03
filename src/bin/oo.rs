use rust_tools::{clipboard, strw::find::find_first_non_blank};

use clap::{CommandFactory, Parser};

#[derive(Parser)]
#[command(about = "Command-line interface for clipboard operations. copy/paste text or images.")]
struct Cli {
    #[arg(short, long, num_args = 0..=1, default_missing_value = "", value_name = "FILE", help = "paste from clipboard to file (default: 'output')")]
    paste: Option<String>,

    #[arg(short, long, num_args = 0..=1, default_missing_value = "", value_name = "FILE", help = "copy from file to clipboard (default: 'output')")]
    copy: Option<String>,
}

fn handle_paste_to_file(fname: &str) -> bool {
    if let Ok(_) = clipboard::string_content::save_to_file(fname) {
        return true;
    }
    if let Ok(_) = clipboard::image_content::save_to_file(fname) {
        return true;
    }
    false
}

fn handle_copy_from_file(fname: &str) -> bool {
    if let Ok(_) = clipboard::string_content::copy_from_file(fname) {
        return true;
    }

    if let Ok(_) = clipboard::image_content::copy_from_file(fname) {
        return true;
    }
    false
}

const DEFAULT_FILE_NAME: &'static str = "output";

fn main() {
    let cli = Cli::parse();

    let copy_str = cli.copy.as_deref().unwrap_or("");
    let paste_str = cli.paste.as_deref().unwrap_or("");

    let fname: Option<&str> = find_first_non_blank(&[copy_str, paste_str]);
    let fname = match fname {
        None => DEFAULT_FILE_NAME,
        Some(val) => val,
    };

    let mut result = false;
    if cli.copy.is_some() {
        result = result || handle_copy_from_file(&fname);
    } else if cli.paste.is_some() {
        // paste
        result = result || handle_paste_to_file(&fname);
    } else {
        Cli::command().print_help().unwrap();
        return;
    }

    if !result {
        eprintln!("oo failed");
    }
}

