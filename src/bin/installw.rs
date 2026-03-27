use std::{
    collections::{BTreeSet, HashMap},
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::SystemTime,
};

use rust_tools::cw::graph::DirectedGraph;

type ModuleGraph = DirectedGraph<Rc<PathBuf>>;
type ModuleInterner = HashMap<PathBuf, Rc<PathBuf>>;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Build,
    Install,
}

fn run() -> Result<(), String> {
    let (mode, bins) = parse_args();
    let cwd = env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    let repo_root = find_repo_root(&cwd).ok_or("cannot find repo root")?;
    let src_dir = repo_root.join("src");
    let bin_dir = src_dir.join("bin");
    let install_dir = env::var("INSTALL_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.join("bin"));
    let release_dir = repo_root.join("target").join("release");
    let bins = if bins.is_empty() {
        list_bin_stems(&bin_dir)?
    } else {
        bins
    };

    let all_rs_files = list_rs_files(&src_dir)?;
    let (graph, interner) = build_module_graph(&all_rs_files)?;

    let mut out = Vec::new();
    for bin in bins {
        let bin_rs = bin_dir.join(format!("{bin}.rs"));
        if !bin_rs.exists() {
            continue;
        }
        let installed_bin = install_dir.join(&bin);
        let built_bin = release_dir.join(&bin);

        let deps = deps_for_bin(&bin, &bin_rs, &repo_root, &graph, &interner)?;
        let newest_src = newest_mtime(&deps)?;
        match mode {
            Mode::Build => {
                if !built_bin.exists() {
                    out.push(bin);
                    continue;
                }
                let newest_installed = newest_existing_mtime(&[&installed_bin])?;
                if newest_installed.is_none_or(|t| newest_src > t) {
                    out.push(bin);
                }
            }
            Mode::Install => {
                if !built_bin.exists() {
                    continue;
                }
                let built_time = file_mtime(&built_bin)?;
                let install_time = newest_existing_mtime(&[&installed_bin])?;
                if install_time.is_none_or(|t| built_time > t) {
                    out.push(bin);
                }
            }
        }
    }

    print!("{}", out.join(" "));
    Ok(())
}

fn parse_args() -> (Mode, Vec<String>) {
    let mut mode = Mode::Build;
    let mut bins = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--" {
            bins.extend(args);
            break;
        }
        if arg == "--mode"
            && let Some(v) = args.next()
        {
            mode = parse_mode_value(&v);
            continue;
        }
        if let Some(v) = arg.strip_prefix("--mode=") {
            mode = parse_mode_value(v);
            continue;
        }
        bins.push(arg);
    }
    (mode, bins)
}

fn parse_mode_value(v: &str) -> Mode {
    match v.trim().to_ascii_lowercase().as_str() {
        "install" => Mode::Install,
        _ => Mode::Build,
    }
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start);
    while let Some(dir) = cur {
        if dir.join("Cargo.toml").is_file() && dir.join("src").is_dir() {
            return Some(dir.to_path_buf());
        }
        cur = dir.parent();
    }
    None
}

fn list_bin_stems(bin_dir: &Path) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    for entry in fs::read_dir(bin_dir).map_err(|e| format!("read_dir {bin_dir:?}: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("rs")) {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            result.push(stem.to_string());
        }
    }
    result.sort();
    Ok(result)
}

fn list_rs_files(src_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![src_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("read_dir {dir:?}: {e}"))? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension() == Some(OsStr::new("rs")) {
                out.push(path);
            }
        }
    }
    Ok(out)
}

fn build_module_graph(files: &[PathBuf]) -> Result<(ModuleGraph, ModuleInterner), String> {
    let file_set = files.iter().cloned().collect::<BTreeSet<_>>();
    let mut interner: ModuleInterner = HashMap::new();
    let mut g = DirectedGraph::new();
    for file in files {
        let node = interner
            .entry(file.clone())
            .or_insert_with(|| Rc::new(file.clone()))
            .clone();
        g.add_node(node);
    }
    for file in files {
        let content = fs::read_to_string(file).map_err(|e| format!("read {file:?}: {e}"))?;
        let dir = file.parent().unwrap_or_else(|| Path::new("."));
        let edges = parse_mod_edges(&content, dir);
        for dep in edges {
            if file_set.contains(&dep) {
                let u = interner
                    .get(file)
                    .cloned()
                    .unwrap_or_else(|| Rc::new(file.clone()));
                let v = interner
                    .get(&dep)
                    .cloned()
                    .unwrap_or_else(|| Rc::new(dep.clone()));
                g.add_edge(u, v);
            }
        }
    }
    Ok((g, interner))
}

fn parse_mod_edges(content: &str, current_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut pending_path: Option<String> = None;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with("#[path") {
            if let Some(p) = extract_path_attr(line) {
                pending_path = Some(p);
            }
            continue;
        }

        let mod_name = extract_mod_decl(line);
        let Some(mod_name) = mod_name else {
            continue;
        };

        if let Some(p) = pending_path.take() {
            out.push(current_dir.join(p));
            continue;
        }

        out.push(current_dir.join(format!("{mod_name}.rs")));
        out.push(current_dir.join(mod_name).join("mod.rs"));
    }
    out
}

fn extract_path_attr(line: &str) -> Option<String> {
    let key = "path";
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    let quote = rest.find('"')?;
    let rest = &rest[quote + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_mod_decl(line: &str) -> Option<String> {
    let line = line.trim_start_matches("pub ");
    let line = line.trim_start_matches("pub(crate) ");
    let line = line.trim_start_matches("pub(super) ");
    let line = line.trim_start_matches("pub(in ");
    let line = if let Some(idx) = line.find(") ") {
        &line[idx + 2..]
    } else {
        line
    };
    let line = line.trim();
    if !line.starts_with("mod ") {
        return None;
    }
    let rest = line.strip_prefix("mod ")?;
    let name = rest
        .split(|c: char| c == ';' || c.is_whitespace() || c == '{')
        .next()?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn deps_for_bin(
    bin_name: &str,
    bin_rs: &Path,
    repo_root: &Path,
    graph: &DirectedGraph<Rc<PathBuf>>,
    interner: &HashMap<PathBuf, Rc<PathBuf>>,
) -> Result<Vec<Rc<PathBuf>>, String> {
    let content = fs::read_to_string(bin_rs).map_err(|e| format!("read {bin_rs:?}: {e}"))?;
    let roots = extract_lib_roots_from_bin(&content);

    let mut deps: BTreeSet<Rc<PathBuf>> = BTreeSet::new();
    let bin_rs = bin_rs.to_path_buf();
    collect_reachable(graph, interner, &bin_rs, &mut deps);

    for root in roots {
        if let Some(root_file) = resolve_lib_root_file(&root, repo_root) {
            collect_reachable(graph, interner, &root_file, &mut deps);
        }
    }

    if deps.len() == 1 {
        let default_root = default_lib_root_for_bin(bin_name);
        if let Some(root_file) = resolve_lib_root_file(&default_root, repo_root) {
            collect_reachable(graph, interner, &root_file, &mut deps);
        }
    }

    for extra in extra_build_inputs(repo_root) {
        deps.insert(Rc::new(extra));
    }

    Ok(deps.into_iter().collect())
}

fn extra_build_inputs(repo_root: &Path) -> Vec<PathBuf> {
    let mut out = vec![repo_root.join("Cargo.toml")];
    let cargo_lock = repo_root.join("Cargo.lock");
    if cargo_lock.is_file() {
        out.push(cargo_lock);
    }
    out
}

fn extract_lib_roots_from_bin(content: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (prefix, slice) in [
        ("rust_tools::", content),
        ("use rust_tools::", content),
        ("use crate::", content),
    ] {
        let mut start = 0usize;
        while let Some(idx) = slice[start..].find(prefix) {
            let idx = start + idx + prefix.len();
            let rest = &slice[idx..];
            let seg = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<String>();
            if !seg.is_empty() {
                out.insert(seg);
            }
            start = idx;
        }
    }
    out
}

fn default_lib_root_for_bin(bin_name: &str) -> String {
    match bin_name {
        "a" => "ai".to_string(),
        _ => "common".to_string(),
    }
}

fn resolve_lib_root_file(root: &str, repo_root: &Path) -> Option<PathBuf> {
    let src = repo_root.join("src");
    let direct = src.join(format!("{root}.rs"));
    if direct.is_file() {
        return Some(direct);
    }
    let mod_rs = src.join(root).join("mod.rs");
    if mod_rs.is_file() {
        return Some(mod_rs);
    }
    None
}

fn collect_reachable(
    graph: &DirectedGraph<Rc<PathBuf>>,
    interner: &HashMap<PathBuf, Rc<PathBuf>>,
    start: &PathBuf,
    out: &mut BTreeSet<Rc<PathBuf>>,
) {
    let Some(start) = interner.get(start).cloned() else {
        return;
    };
    if !out.insert(start.clone()) {
        return;
    }
    let mut q = vec![start];
    while let Some(u) = q.pop() {
        for v in graph.adj(&u) {
            if out.insert(v.clone()) {
                q.push(v);
            }
        }
    }
}

fn newest_mtime(files: &[Rc<PathBuf>]) -> Result<SystemTime, String> {
    let mut newest = SystemTime::UNIX_EPOCH;
    for f in files {
        let t = file_mtime(f.as_ref())?;
        if t > newest {
            newest = t;
        }
    }
    Ok(newest)
}

fn file_mtime(path: &Path) -> Result<SystemTime, String> {
    let meta = fs::metadata(path).map_err(|e| format!("metadata {path:?}: {e}"))?;
    meta.modified().map_err(|e| format!("mtime {path:?}: {e}"))
}

fn newest_existing_mtime(paths: &[&Path]) -> Result<Option<SystemTime>, String> {
    let mut newest: Option<SystemTime> = None;
    for p in paths {
        let Ok(meta) = fs::metadata(p) else {
            continue;
        };
        let t = meta.modified().map_err(|e| format!("mtime {p:?}: {e}"))?;
        newest = Some(match newest {
            Some(cur) if cur >= t => cur,
            _ => t,
        });
    }
    Ok(newest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deps_for_re_includes_bin_modules() {
        let cwd = env::current_dir().unwrap();
        let repo_root = find_repo_root(&cwd).unwrap();
        let src_dir = repo_root.join("src");
        let bin_dir = src_dir.join("bin");

        let all_rs_files = list_rs_files(&src_dir).unwrap();
        let (graph, interner) = build_module_graph(&all_rs_files).unwrap();

        let bin_rs = bin_dir.join("re.rs");
        let deps = deps_for_bin("re", &bin_rs, &repo_root, &graph, &interner).unwrap();
        let deps = deps
            .into_iter()
            .map(|p| p.as_ref().clone())
            .collect::<BTreeSet<_>>();

        assert!(deps.contains(&bin_dir.join("re").join("features").join("mod.rs")));
        assert!(deps.contains(&bin_dir.join("re").join("features").join("search.rs")));
    }
}
