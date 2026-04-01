use crate::{matcher, output};
use rust_tools::{strw::indices::substring_quiet, terminalw};
use rust_tools::cw::SkipSet;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
};

const DEFAULT_EXTENSIONS: &[&str] = &[
    ".py", ".cpp", ".js", ".txt", ".h", ".hpp", ".c", ".tex", ".html", ".css", ".java", ".go",
    ".cc", ".htm", ".ts", ".xml", ".php", ".sc", "",
];

pub fn configure_terminalw(opts: &crate::cli::Options) -> SkipSet<String> {
    terminalw::COUNT.store(0, Ordering::Relaxed);
    terminalw::NUM_PRINT.store(opts.num_print, Ordering::Relaxed);
    terminalw::VERBOSE.store(opts.verbose, Ordering::Relaxed);
    terminalw::MAX_LEVEL.store(opts.max_level, Ordering::Relaxed);
    terminalw::change_threads(opts.thread_count);

    if !opts.file_names.is_empty() {
        for f in &opts.file_names {
            terminalw::FILE_NAMES_TO_CHECK.add(f);
        }
    }
    if !opts.not_file_names.is_empty() {
        for f in &opts.not_file_names {
            terminalw::FILE_NAMES_NOT_CHECK.add(f);
        }
    }

    let mut allowed: SkipSet<String> = SkipSet::new(16);
    if !opts.ext.trim().is_empty() {
        for e in terminalw::format_file_extensions(&opts.ext) {
            terminalw::EXTENSIONS.add(&e);
            allowed.insert(e);
        }
        terminalw::CHECK_EXTENSION.store(true, Ordering::Relaxed);
        return allowed;
    }

    if !opts.ext_exclude.trim().is_empty() {
        for e in DEFAULT_EXTENSIONS {
            let ee = if e.is_empty() {
                "".to_string()
            } else {
                e.to_string()
            };
            if !ee.is_empty() {
                allowed.insert(ee);
            }
        }
        for ex in terminalw::format_file_extensions(&opts.ext_exclude) {
            allowed.remove(&ex);
        }
        for e in allowed.iter() {
            terminalw::EXTENSIONS.add(e);
        }
        terminalw::CHECK_EXTENSION.store(true, Ordering::Relaxed);
        return allowed;
    }

    terminalw::CHECK_EXTENSION.store(false, Ordering::Relaxed);
    allowed
}

fn read_line_trim_newline(buf: &mut String, reader: &mut BufReader<File>) -> Option<String> {
    buf.clear();
    let n = reader.read_line(buf).ok()?;
    if n == 0 {
        return None;
    }
    while buf.ends_with('\n') || buf.ends_with('\r') {
        buf.pop();
    }
    Some(buf.clone())
}

fn check_file(
    filename: String,
    match_mode: Arc<matcher::MatchMode>,
    overlap_hint_len: usize,
    num_lines: usize,
    max_len: usize,
) {
    type Hit = (String, Vec<(usize, usize)>, usize);

    if terminalw::COUNT.load(Ordering::Relaxed) >= terminalw::NUM_PRINT.load(Ordering::Relaxed) {
        return;
    }

    let Ok(file) = File::open(&filename) else {
        if terminalw::VERBOSE.load(Ordering::Relaxed) {
            eprintln!("failed to open {filename}");
        }
        return;
    };
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    let mut lineno: usize = 0;
    while let Some(mut line) = read_line_trim_newline(&mut buf, &mut reader) {
        lineno += 1;

        let mut matched: Option<Hit> = None;
        if let Some((src, ranges)) = matcher::match_line(&line, &match_mode) {
            matched = Some((src.to_string(), ranges, lineno));
        } else if num_lines > 1 {
            let mut cnt = 1usize;
            while cnt < num_lines {
                let Some(next) = read_line_trim_newline(&mut buf, &mut reader) else {
                    break;
                };
                lineno += 1;
                cnt += 1;
                line.push_str(&next);
                if let Some((src, ranges)) = matcher::match_line(&line, &match_mode) {
                    matched = Some((src.to_string(), ranges, lineno));
                    break;
                }
                line = matcher::crop_for_overlap(&line, overlap_hint_len);
            }
        }

        let Some((src, ranges, hit_line)) = matched else {
            continue;
        };

        let cur = terminalw::COUNT.fetch_add(1, Ordering::SeqCst) + 1;
        if cur > terminalw::NUM_PRINT.load(Ordering::Relaxed) {
            terminalw::COUNT.fetch_sub(1, Ordering::SeqCst);
            return;
        }

        let Ok(abs) = std::fs::canonicalize(PathBuf::from(&filename)) else {
            if terminalw::VERBOSE.load(Ordering::Relaxed) {
                eprintln!("failed to abs {filename}");
            }
            return;
        };

        let preview = substring_quiet(&src, 0, max_len as isize);
        let preview = preview.trim().to_string();
        let preview = output::highlight_ranges(&preview, ranges);
        output::print_hit(&abs, hit_line, &preview);
    }
}

pub fn run(opts: &crate::cli::Options) -> Result<i64, String> {
    let (target_after_word, match_mode) = matcher::build_matcher(
        opts.target.clone(),
        opts.is_regex,
        opts.ignore_case,
        opts.strict,
        opts.word,
    )?;

    let overlap_hint_len = match_mode.overlap_hint_len().max(target_after_word.len());
    let task = {
        let match_mode = Arc::new(match_mode);
        let num_lines = opts.num_lines;
        let max_len = opts.max_len;
        Arc::new(move |filename: String| {
            check_file(
                filename,
                Arc::clone(&match_mode),
                overlap_hint_len,
                num_lines,
                max_len,
            );
        })
    };

    let wg = Arc::new(terminalw::WaitGroup::new());
    terminalw::find(&opts.root_dir, task, Arc::clone(&wg), 0);
    wg.wait();

    Ok(terminalw::COUNT.load(Ordering::Relaxed))
}

pub fn print_summary(found: i64) {
    let summary = format!("{found} matches found\n");
    print!("{}", "-".repeat(summary.len()));
    let shown = found.min(terminalw::NUM_PRINT.load(Ordering::Relaxed));
    println!("\n{} matches found", shown);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_extensions_non_empty() {
        assert!(!DEFAULT_EXTENSIONS.contains(&".rs"));
    }

    #[test]
    fn test_read_line_trim_newline() {
        let dir = std::env::temp_dir();
        let file = dir.join("rust_tools_fk_read_line_test.txt");
        std::fs::write(&file, "a\r\nb\n").unwrap();
        let f = File::open(&file).unwrap();
        let mut reader = BufReader::new(f);
        let mut buf = String::new();
        assert_eq!(read_line_trim_newline(&mut buf, &mut reader).unwrap(), "a");
        assert_eq!(read_line_trim_newline(&mut buf, &mut reader).unwrap(), "b");
    }
}
