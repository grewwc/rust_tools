use std::env;

use rust_tools::{clipboard, strw::find::find_first_non_blank};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Output file path (optional)
    file: Option<String>,

    #[arg(short, long, default_value = "", help = "paste from clipboard")]
    paste: String,

    #[arg(short, long, default_value = "", help = "copy to clipboard")]
    copy: String,
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

fn main() {
    let cli = Cli::parse();
    let file = cli.file.unwrap_or("output".to_string());
    let fname: Option<&str> = find_first_non_blank(&[cli.copy.as_str(), cli.paste.as_str()]);

    let fname = match fname {
        None => file.as_str(),
        Some(val) => val,
    };
    let mut result = false;
    if cli.copy != "" {
        result = result || handle_copy_from_file(&fname);
    } else {
        // paste
        result = result || handle_paste_to_file(&fname);
    }

    if !result {
        eprintln!("oo failed");
    }
}
