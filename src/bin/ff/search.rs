use crate::{exclude, output};
use colored::Colorize;
use rust_tools::cw::concurrent_hash_map::ConcurrentHashMap;
use rust_tools::terminalw;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
};
use tokio::sync::{Mutex, mpsc};

fn file_ends_with(abs: &str, filename: &str) -> bool {
    let abs = abs.replace('\\', "/");
    if let Some(idx) = abs.rfind(filename) {
        if idx == 0 || idx + filename.len() != abs.len() {
            return false;
        }
        return abs.as_bytes().get(idx.saturating_sub(1)).copied() == Some(b'/');
    }
    false
}

fn collect_matches_in_dir(
    dir: &Path,
    targets: &[String],
    glob_mode: bool,
    case_insensitive: bool,
    verbose: bool,
) -> Vec<PathBuf> {
    if targets.is_empty() {
        return Vec::new();
    }

    let dir_str = dir.to_string_lossy().to_string();
    if glob_mode {
        let mut out = Vec::new();
        for t in targets {
            let m = if case_insensitive {
                terminalw::glob_case_insensitive(t, &dir_str)
            } else {
                terminalw::glob_paths(t, &dir_str)
            };
            match m {
                Ok(paths) => out.extend(paths.into_iter().map(PathBuf::from)),
                Err(e) => {
                    if verbose {
                        eprintln!("{}", e.red());
                    }
                }
            }
        }
        return out;
    }

    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let abs = entry.path();
        let name = abs
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let abs_str = abs.to_string_lossy().to_string();
        for t in targets {
            if abs_str == *t || name == *t || file_ends_with(&abs_str, t) {
                out.push(abs.clone());
                break;
            }
        }
    }
    out
}

fn root_dirs_from_pattern(root_pat: &str) -> std::collections::VecDeque<PathBuf> {
    let mut root_dirs: std::collections::VecDeque<PathBuf> = std::collections::VecDeque::new();
    if let Ok(paths) = glob::glob(root_pat) {
        for entry in paths.flatten() {
            root_dirs.push_back(entry);
        }
    }
    if root_dirs.is_empty() {
        root_dirs.push_back(PathBuf::from(root_pat));
    }
    root_dirs
}

enum Msg {
    Dir(PathBuf),
    Stop,
}

fn enqueue_dir(tx: &mpsc::UnboundedSender<Msg>, inflight: &AtomicUsize, dir: PathBuf) {
    inflight.fetch_add(1, Ordering::SeqCst);
    let _ = tx.send(Msg::Dir(dir));
}

fn maybe_send_stop(tx: &mpsc::UnboundedSender<Msg>, stop_sent: &AtomicBool, worker_count: usize) {
    if stop_sent
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        for _ in 0..worker_count {
            let _ = tx.send(Msg::Stop);
        }
    }
}

pub async fn run_async(opts: &crate::cli::Options) -> i64 {
    if opts.targets.iter().any(|t| Path::new(t).is_absolute()) {
        return run_absolute_targets(opts);
    }
    run_walk_async(opts).await
}

fn run_absolute_targets(opts: &crate::cli::Options) -> i64 {
    let mut printed = 0_i64;
    for t in opts.targets.iter().filter(|t| Path::new(t).is_absolute()) {
        let abs = PathBuf::from(t);
        if opts.only_dir && !abs.is_dir() {
            continue;
        }
        if exclude::should_exclude(&abs.to_string_lossy(), &opts.excludes) {
            continue;
        }
        printed += 1;
        let match_base = abs
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let _ = output::print_match(
            &abs,
            &opts.wd,
            &match_base,
            opts.relative,
            opts.verbose,
            opts.print_md5,
        );
        return printed;
    }
    printed
}

fn scan_dir_blocking(
    dir: &PathBuf,
    targets: &[String],
    glob_mode: bool,
    case_insensitive: bool,
    verbose: bool,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let matches = collect_matches_in_dir(dir, targets, glob_mode, case_insensitive, verbose);

    let Ok(entries) = fs::read_dir(dir) else {
        return (matches, Vec::new());
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            subdirs.push(p);
        }
    }
    (matches, subdirs)
}

fn process_match_blocking(
    m: PathBuf,
    opts: &crate::cli::Options,
    count: &Arc<AtomicI64>,
    stop: &Arc<AtomicBool>,
    printed: &Arc<ConcurrentHashMap<PathBuf, ()>>,
) {
    if stop.load(Ordering::Relaxed) || output::PRINT_DISABLED.load(Ordering::Relaxed) {
        stop.store(true, Ordering::Relaxed);
        return;
    }

    let abs = fs::canonicalize(&m).unwrap_or(m);
    if opts.only_dir && !abs.is_dir() {
        return;
    }
    if exclude::should_exclude(&abs.to_string_lossy(), &opts.excludes) {
        return;
    }

    if printed.put_if_absent(abs.clone(), ()).is_some() {
        return;
    }

    let match_base = abs
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let cur = count.fetch_add(1, Ordering::SeqCst) + 1;
    if cur > opts.num_print {
        count.fetch_sub(1, Ordering::SeqCst);
        stop.store(true, Ordering::Relaxed);
        return;
    }

    if let Err(e) = output::print_match(
        &abs,
        &opts.wd,
        &match_base,
        opts.relative,
        opts.verbose,
        opts.print_md5,
    ) && opts.verbose
    {
        eprintln!("{}", e.red());
    }
}

async fn run_walk_async(opts: &crate::cli::Options) -> i64 {
    let (tx, rx) = mpsc::unbounded_channel::<Msg>();
    let rx = Arc::new(Mutex::new(rx));

    let count = Arc::new(AtomicI64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let inflight = Arc::new(AtomicUsize::new(0));
    let stop_sent = Arc::new(AtomicBool::new(false));
    let printed = Arc::new(ConcurrentHashMap::<PathBuf, ()>::default());

    let worker_count = opts.thread_count.max(1);

    for root in root_dirs_from_pattern(&opts.root_pat) {
        enqueue_dir(&tx, &inflight, root);
    }

    let mut handles = Vec::new();
    for _ in 0..worker_count {
        let rx = Arc::clone(&rx);
        let tx = tx.clone();
        let opts = opts.clone();
        let count = Arc::clone(&count);
        let stop = Arc::clone(&stop);
        let inflight = Arc::clone(&inflight);
        let stop_sent = Arc::clone(&stop_sent);
        let printed = Arc::clone(&printed);

        handles.push(tokio::spawn(async move {
            loop {
                if output::PRINT_DISABLED.load(Ordering::Relaxed) {
                    stop.store(true, Ordering::Relaxed);
                }
                if stop.load(Ordering::Relaxed) {
                    maybe_send_stop(&tx, &stop_sent, worker_count);
                }

                let msg = {
                    let mut guard = rx.lock().await;
                    guard.recv().await
                };

                let Some(msg) = msg else {
                    return;
                };

                match msg {
                    Msg::Stop => return,
                    Msg::Dir(dir) => {
                        if count.load(Ordering::Relaxed) >= opts.num_print {
                            stop.store(true, Ordering::Relaxed);
                        }

                        if !stop.load(Ordering::Relaxed) {
                            let dir2 = dir.clone();
                            let opts2 = opts.clone();
                            let count2 = Arc::clone(&count);
                            let stop2 = Arc::clone(&stop);
                            let printed2 = Arc::clone(&printed);
                            let res = tokio::task::spawn_blocking(move || {
                                if stop2.load(Ordering::Relaxed) {
                                    return Vec::new();
                                }
                                let (matches, subdirs) = scan_dir_blocking(
                                    &dir2,
                                    &opts2.targets,
                                    opts2.glob_mode,
                                    opts2.case_insensitive,
                                    opts2.verbose,
                                );
                                for m in matches {
                                    process_match_blocking(m, &opts2, &count2, &stop2, &printed2);
                                    if stop2.load(Ordering::Relaxed) {
                                        return Vec::new();
                                    }
                                }
                                if stop2.load(Ordering::Relaxed) {
                                    Vec::new()
                                } else {
                                    subdirs
                                }
                            })
                            .await;

                            if let Ok(subdirs) = res
                                && !stop.load(Ordering::Relaxed)
                            {
                                for d in subdirs {
                                    if stop.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    enqueue_dir(&tx, &inflight, d);
                                }
                            }
                        }

                        let remaining = inflight.fetch_sub(1, Ordering::SeqCst).saturating_sub(1);
                        if remaining == 0 {
                            maybe_send_stop(&tx, &stop_sent, worker_count);
                            return;
                        }
                    }
                }
            }
        }));
    }

    drop(tx);

    for h in handles {
        let _ = h.await;
    }

    count.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_ends_with() {
        assert!(file_ends_with("/a/b/c.txt", "c.txt"));
        assert!(!file_ends_with("/a/b/c.txt", "b"));
        assert!(!file_ends_with("/a/b/c.txt", "txt"));
    }

    #[test]
    fn test_collect_matches_in_dir_returns_existing_paths() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rust_tools_ff_test_{}", stamp));
        let dir = root.join("src").join("bin");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("ff.rs");
        std::fs::write(&file, "x").unwrap();

        let matches = collect_matches_in_dir(&dir, &[String::from("ff.rs")], false, false, false);
        assert!(!matches.is_empty());
        for p in matches {
            assert!(std::fs::metadata(&p).is_ok(), "path should exist: {:?}", p);
        }
    }

    #[test]
    fn test_dedup_by_canonical_path() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rust_tools_ff_test_dedup_{}", stamp));
        let real_dir = root.join("real");
        let link_dir = root.join("link");
        std::fs::create_dir_all(&real_dir).unwrap();
        let file = real_dir.join("a.pdf");
        std::fs::write(&file, "x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real_dir, &link_dir).unwrap();

        let opts = crate::cli::Options {
            verbose: false,
            only_dir: false,
            print_md5: false,
            glob_mode: false,
            case_insensitive: false,
            relative: true,
            num_print: i64::MAX,
            thread_count: 1,
            wd: root.clone(),
            root_pat: root.to_string_lossy().to_string(),
            targets: vec!["a.pdf".to_string()],
            excludes: Vec::new(),
        };

        let count = Arc::new(AtomicI64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let printed = Arc::new(ConcurrentHashMap::<PathBuf, ()>::default());

        let m1 = real_dir.join("a.pdf");
        let m2 = link_dir.join("a.pdf");
        process_match_blocking(m1, &opts, &count, &stop, &printed);
        process_match_blocking(m2, &opts, &count, &stop, &printed);

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
