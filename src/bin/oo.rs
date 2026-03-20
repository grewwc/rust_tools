use std::io::{self, BufRead, Read};

use rust_tools::{clipboard, strw::find::find_first_non_blank};

use clap::{CommandFactory, Parser};

#[derive(Parser)]
#[command(about = "Command-line interface for clipboard operations. copy/paste text or images.")]
struct Cli {
    #[arg(short, long, num_args = 0..=1, default_missing_value = "", value_name = "FILE", help = "paste from clipboard (or stdin) to file (default: 'output')")]
    paste: Option<String>,

    #[arg(short, long, num_args = 0..=1, default_missing_value = "", value_name = "FILE", help = "copy from file to clipboard (default: 'output'). Image copy uses OSC52 bridge by default for SSH paste; set OO_PREFER_NATIVE_IMAGE=1 to disable")]
    copy: Option<String>,

    #[arg(
        short = 'B',
        long,
        help = "bridge: encode image clipboard as base64 text clipboard (run on LOCAL machine so remote `oo -p` can retrieve it via OSC52)"
    )]
    bridge: bool,
}

fn stdin_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

fn is_ssh() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

fn try_paste_image_via_osc52(fname: &str) -> Result<(), String> {
    clipboard::image_content::save_to_file(fname).map_err(|e| e.to_string())
}

fn handle_paste_to_file(fname: &str) -> Result<(), String> {
    // When stdin is piped (not a TTY), read raw bytes from stdin and write to file.
    // This allows: cat image.png | ssh host "oo -p file.png"
    if !stdin_is_tty() {
        let mut bytes = Vec::new();
        io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|e| format!("stdin read error: {e}"))?;
        if !bytes.is_empty() {
            std::fs::write(fname, &bytes).map_err(|e| format!("write error: {e}"))?;
            println!("save to file: {fname}");
            return Ok(());
        }
    }

    if clipboard::binary_content::save_to_file(fname).is_ok() {
        return Ok(());
    }
    if clipboard::string_content::save_to_file(fname).is_ok() {
        return Ok(());
    }

    // First attempt at image via OSC52.
    if try_paste_image_via_osc52(fname).is_ok() {
        return Ok(());
    }

    // In SSH sessions: OSC52 only carries text, so a natively-copied image won't arrive.
    // Prompt the user to run `oo -B` on their local machine to re-encode the image as text,
    // then retry once.
    if is_ssh() && stdin_is_tty() {
        eprintln!("No image data found in clipboard via OSC52.");
        eprintln!("On your LOCAL machine, run:  oo -B");
        eprintln!("Then press Enter here to retry...");
        let stdin = io::stdin();
        let mut line = String::new();
        let _ = stdin.lock().read_line(&mut line);

        if try_paste_image_via_osc52(fname).is_ok() {
            return Ok(());
        }
    }

    Err(format!(
        "no image found in clipboard\n\
        hint: on your LOCAL machine run `oo -B` to encode the image as text, then retry `oo -p {fname}` here"
    ))
}

fn handle_copy_from_file(fname: &str) -> bool {
    if clipboard::string_content::copy_from_file(fname).is_ok() {
        return true;
    }

    if clipboard::image_content::copy_from_file(fname).is_ok() {
        return true;
    }
    if clipboard::binary_content::copy_from_file(fname).is_ok() {
        return true;
    }
    false
}

const DEFAULT_FILE_NAME: &str = "output";

fn main() {
    let cli = Cli::parse();

    if cli.bridge {
        match clipboard::image_content::bridge_image_to_text_clipboard() {
            Ok(()) => {}
            Err(e) => eprintln!("oo -B failed: {e}"),
        }
        return;
    }

    let copy_str = cli.copy.as_deref().unwrap_or("");
    let paste_str = cli.paste.as_deref().unwrap_or("");

    let fname: Option<&str> = find_first_non_blank(&[copy_str, paste_str]);
    let fname = match fname {
        None => DEFAULT_FILE_NAME,
        Some(val) => val,
    };

    if cli.copy.is_some() {
        if !handle_copy_from_file(fname) {
            eprintln!("oo failed");
        }
    } else if cli.paste.is_some() {
        if let Err(e) = handle_paste_to_file(fname) {
            eprintln!("oo failed: {e}");
        }
    } else {
        Cli::command().print_help().unwrap();
    }
}
