use std::io::{self, BufRead, Read};

use rust_tools::{clipboardw, strw::find::find_first_non_blank};

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

    #[arg(
        short = 'w',
        long,
        help = "watch: run on LOCAL machine; auto-bridge any copied image (keeps original image + adds base64 text) so remote SSH paste just works. Ctrl+C to stop"
    )]
    watch: bool,
}

fn stdin_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

fn is_ssh() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

fn saved_file_message(path: &str) -> String {
    format!("save to file: {path}")
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

    // 统一粘贴入口：只读取一次剪贴板，按内容类型（图片/文本/二进制）
    // 自动选择扩展名保存，避免旧实现 binary→string→image 三级联各自重新
    // 查询一次 OSC52（大图每查一次都要重新传输整份剪贴板）。
    let saved_path = match clipboardw::paste_to_file(fname) {
        Ok(path) => path,
        Err(e) => {
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

                if let Ok(path) = clipboardw::paste_to_file(fname) {
                    println!("{}", saved_file_message(&path));
                    return Ok(());
                }
            }

            return Err(format!(
                "no image found in clipboard ({e})\n\
                hint: on your LOCAL machine run `oo -B` to encode the image as text, then retry `oo -p {fname}` here"
            ));
        }
    };

    println!("{}", saved_file_message(&saved_path));
    Ok(())
}

fn handle_copy_from_file(fname: &str) -> bool {
    if clipboardw::string_content::copy_from_file(fname).is_ok() {
        return true;
    }

    if clipboardw::image_content::copy_from_file(fname).is_ok() {
        return true;
    }
    if clipboardw::binary_content::copy_from_file(fname).is_ok() {
        return true;
    }
    false
}

const DEFAULT_FILE_NAME: &str = "output";

fn main() {
    let cli = Cli::parse();

    if cli.watch {
        if let Err(e) = clipboardw::image_content::watch_clipboard_bridge() {
            eprintln!("oo --watch failed: {e}");
        }
        return;
    }

    if cli.bridge {
        match clipboardw::image_content::bridge_image_to_text_clipboard() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_image_save_message_uses_resolved_jpg_path() {
        let path = clipboardw::image_content::resolved_save_path(DEFAULT_FILE_NAME);
        assert_eq!(saved_file_message(&path), "save to file: output.jpg");
    }
}
