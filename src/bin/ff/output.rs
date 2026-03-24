use colored::Colorize;
use std::{
    fs,
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub static PRINT_DISABLED: AtomicBool = AtomicBool::new(false);

fn parse_file_size(size: u64) -> String {
    const K: f64 = 1024.0;
    let s = size as f64;
    if size < 1024 {
        return format!("{size}");
    }
    if size < 1024 * 1024 {
        return format!("{:.2}K", s / K);
    }
    if size < 1024 * 1024 * 1024 {
        return format!("{:.2}M", s / K / K);
    }
    format!("{:.2}G", s / K / K / K)
}

fn highlight_match(path_str: &str, match_base: &str) -> String {
    let normalized = path_str.replace('\\', "/");
    normalized.replace(match_base, &match_base.green().to_string())
}

fn maybe_add_dir_trailing_sep(path: &str, is_dir: bool) -> String {
    if !is_dir {
        return path.to_string();
    }
    if path.ends_with(std::path::MAIN_SEPARATOR) || path.ends_with('/') {
        return path.to_string();
    }
    if std::path::MAIN_SEPARATOR == '/' {
        format!("{path}/")
    } else {
        format!("{path}{}", std::path::MAIN_SEPARATOR)
    }
}

fn strip_workdir(abs: &Path, wd: &Path) -> String {
    if let Ok(stripped) = abs.strip_prefix(wd) {
        let mut s = stripped.to_string_lossy().to_string();
        while s.starts_with(std::path::MAIN_SEPARATOR) || s.starts_with('/') {
            s.remove(0);
        }
        return s;
    }
    abs.to_string_lossy().to_string()
}

fn format_mtime(t: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y.%m.%d/%H:%M:%S").to_string()
}

pub fn print_match(
    abs: &Path,
    wd: &Path,
    match_base: &str,
    relative: bool,
    verbose: bool,
    print_md5: bool,
) -> Result<(), String> {
    if PRINT_DISABLED.load(Ordering::Relaxed) {
        return Ok(());
    }

    let mut display = if relative {
        strip_workdir(abs, wd)
    } else {
        abs.to_string_lossy().to_string()
    };

    let meta = fs::metadata(abs).ok();
    if let Some(m) = &meta {
        display = maybe_add_dir_trailing_sep(&display, m.is_dir());
    }

    let mut out = highlight_match(&display, match_base);

    if verbose {
        if let Some(m) = meta {
            out.push_str("  ");
            out.push_str(&parse_file_size(m.len()));
            if let Ok(t) = m.modified() {
                out.push_str("  ");
                out.push_str(&format_mtime(t));
            }
        } else {
            return Err(format!("failed to stat {}", abs.to_string_lossy()));
        }
    }

    if print_md5 {
        let bytes = fs::read(abs).map_err(|e| e.to_string())?;
        let digest = md5::compute(bytes);
        out.push('\t');
        out.push_str(&format!("{:x}", digest));
    }

    let mut stdout = io::stdout().lock();
    if let Err(e) = writeln!(stdout, "{out}") {
        if e.kind() == io::ErrorKind::BrokenPipe {
            PRINT_DISABLED.store(true, Ordering::Relaxed);
            return Ok(());
        }
        return Err(e.to_string());
    }
    Ok(())
}
