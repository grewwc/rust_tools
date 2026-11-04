use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::SystemTime,
};

use rust_tools::cw::{SkipSet, graph::DirectedGraph};

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
        let installed_bin = install_dir.join(&bin);
        let built_bin = release_dir.join(&bin);
        match mode {
            Mode::Build => {
                let bin_rs = bin_dir.join(format!("{bin}.rs"));
                if !bin_rs.exists() {
                    continue;
                }
                let deps = deps_for_bin(&bin, &bin_rs, &repo_root, &graph, &interner)?;
                let newest_src = newest_mtime(&deps)?;
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
                // Install mode only compares the release artifact against the
                // installed copy, so it also covers bins without a
                // src/bin/<bin>.rs (e.g. re in crates/re and the mcp_* crates),
                // which lets the Makefile skip re-copying up-to-date binaries
                // instead of forcing an unconditional copy.
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
    let mut file_set = SkipSet::new(16);
    for file in files {
        file_set.insert(file.clone());
    }
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
        let edges = parse_mod_edges(&content, file);
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

fn parse_mod_edges(content: &str, current_file: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut pending_path: Option<String> = None;
    let current_dir = current_file.parent().unwrap_or_else(|| Path::new("."));
    let implicit_mod_dir = implicit_module_dir(current_file);
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

        out.push(implicit_mod_dir.join(format!("{mod_name}.rs")));
        out.push(implicit_mod_dir.join(mod_name).join("mod.rs"));
    }
    out
}

fn implicit_module_dir(current_file: &Path) -> PathBuf {
    let current_dir = current_file.parent().unwrap_or_else(|| Path::new("."));
    let file_name = current_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if file_name == "mod.rs" || is_crate_root_file(current_file) {
        return current_dir.to_path_buf();
    }
    let stem = current_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    current_dir.join(stem)
}

fn is_crate_root_file(current_file: &Path) -> bool {
    let Some(parent) = current_file.parent() else {
        return false;
    };
    let file_name = current_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if matches!(file_name, "lib.rs" | "main.rs") {
        return true;
    }
    matches!(
        parent.file_name().and_then(|s| s.to_str()),
        Some("bin" | "tests" | "examples" | "benches")
    )
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

fn parse_include_paths(content: &str, current_file: &Path) -> Vec<PathBuf> {
    let current_dir = current_file.parent().unwrap_or_else(|| Path::new("."));
    let bytes = content.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < content.len() {
        pos = skip_include_trivia(content, pos);
        let rest = &content[pos..];
        let Some(ch) = rest.chars().next() else {
            break;
        };
        let literal = if matches!(ch, 'b' | 'c') {
            &rest[1..]
        } else {
            rest
        };
        if let Some(len) = string_literal_len(literal) {
            pos += rest.len() - literal.len() + len;
            continue;
        }
        // Skip character literals without consuming lifetime annotations.
        if ch == '\'' {
            let mut end = pos + 1;
            if bytes.get(end) == Some(&b'\\') {
                end += 2;
            } else if let Some(ch) = content[end..].chars().next() {
                end += ch.len_utf8();
            }
            if bytes.get(end) == Some(&b'\'') {
                pos = end + 1;
                continue;
            }
        }
        if !ch.is_alphabetic() && ch != '_' {
            pos += ch.len_utf8();
            continue;
        }
        let start = pos;
        pos += ch.len_utf8();
        while let Some(ch) = content[pos..].chars().next() {
            if !ch.is_alphanumeric() && ch != '_' {
                break;
            }
            pos += ch.len_utf8();
        }
        if !matches!(
            &content[start..pos],
            "include_str" | "include_bytes" | "include"
        ) {
            continue;
        }
        let mut arg = skip_include_trivia(content, pos);
        if bytes.get(arg) != Some(&b'!') {
            continue;
        }
        arg = skip_include_trivia(content, arg + 1);
        let closing = match bytes.get(arg) {
            Some(b'(') => b')',
            Some(b'[') => b']',
            Some(b'{') => b'}',
            _ => continue,
        };
        arg = skip_include_trivia(content, arg + 1);
        let Some((path, len)) = parse_include_literal(&content[arg..]) else {
            continue;
        };
        let mut end = skip_include_trivia(content, arg + len);
        if bytes.get(end) == Some(&b',') {
            end = skip_include_trivia(content, end + 1);
        }
        if bytes.get(end) == Some(&closing) {
            out.push(current_dir.join(path));
            pos = end + 1;
        }
    }
    out
}

fn skip_include_trivia(content: &str, mut pos: usize) -> usize {
    let bytes = content.as_bytes();
    loop {
        while bytes.get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        if bytes.get(pos..pos + 2) == Some(b"//") {
            pos = content[pos..]
                .find('\n')
                .map_or(content.len(), |len| pos + len + 1);
        } else if bytes.get(pos..pos + 2) == Some(b"/*") {
            pos += 2;
            let mut depth = 1;
            while pos < bytes.len() && depth > 0 {
                match bytes.get(pos..pos + 2) {
                    Some(b"/*") => {
                        depth += 1;
                        pos += 2;
                    }
                    Some(b"*/") => {
                        depth -= 1;
                        pos += 2;
                    }
                    _ => pos += 1,
                }
            }
        } else {
            return pos;
        }
    }
}

// Find string boundaries without decoding or validating byte string escapes.
fn string_literal_len(source: &str) -> Option<usize> {
    if let Some(raw) = source.strip_prefix('r') {
        let hashes = raw.bytes().take_while(|b| *b == b'#').count();
        let body = raw[hashes..].strip_prefix('"')?;
        let closing = format!("\"{}", "#".repeat(hashes));
        return Some(
            body.find(&closing)
                .map_or(source.len(), |end| hashes + 2 + end + closing.len()),
        );
    }
    let mut chars = source.strip_prefix('"')?.char_indices();
    while let Some((index, ch)) = chars.next() {
        match ch {
            '"' => return Some(index + 2),
            '\\' => {
                chars.next();
            }
            _ => {}
        }
    }
    Some(source.len())
}

fn parse_include_literal(source: &str) -> Option<(String, usize)> {
    if let Some(raw) = source.strip_prefix('r') {
        let hashes = raw.bytes().take_while(|b| *b == b'#').count();
        let body = raw[hashes..].strip_prefix('"')?;
        let closing = format!("\"{}", "#".repeat(hashes));
        let end = body.find(&closing)?;
        return Some((body[..end].to_string(), hashes + 2 + end + closing.len()));
    }
    let mut chars = source.strip_prefix('"')?.char_indices();
    let mut path = String::new();
    while let Some((index, ch)) = chars.next() {
        match ch {
            '"' => return Some((path, index + 2)),
            '\\' => match chars.next()?.1 {
                '"' => path.push('"'),
                '\'' => path.push('\''),
                '\\' => path.push('\\'),
                'n' => path.push('\n'),
                'r' => path.push('\r'),
                't' => path.push('\t'),
                '0' => path.push('\0'),
                'x' => {
                    let hi = chars.next()?.1.to_digit(16)?;
                    let lo = chars.next()?.1.to_digit(16)?;
                    let value = hi * 16 + lo;
                    if value > 0x7f {
                        return None;
                    }
                    path.push(char::from_u32(value)?);
                }
                'u' => {
                    if chars.next()?.1 != '{' {
                        return None;
                    }
                    let mut digits = String::new();
                    loop {
                        match chars.next()?.1 {
                            '}' => break,
                            '_' => {}
                            digit => digits.push(digit),
                        }
                    }
                    path.push(char::from_u32(u32::from_str_radix(&digits, 16).ok()?)?);
                }
                '\n' => {
                    while chars
                        .clone()
                        .next()
                        .is_some_and(|(_, ch)| ch.is_whitespace())
                    {
                        chars.next();
                    }
                }
                _ => return None,
            },
            _ => path.push(ch),
        }
    }
    None
}

fn deps_for_bin(
    bin_name: &str,
    bin_rs: &Path,
    repo_root: &Path,
    graph: &DirectedGraph<Rc<PathBuf>>,
    interner: &HashMap<PathBuf, Rc<PathBuf>>,
) -> Result<Vec<PathBuf>, String> {
    let content = fs::read_to_string(bin_rs).map_err(|e| format!("read {bin_rs:?}: {e}"))?;
    let roots = extract_lib_roots_from_bin(&content);

    let mut deps: SkipSet<PathBuf> = SkipSet::new(16);
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

    for file in deps.to_vec() {
        let content = fs::read_to_string(&file).map_err(|e| format!("read {file:?}: {e}"))?;
        for included in parse_include_paths(&content, &file) {
            deps.insert(included);
        }
    }

    for extra in extra_build_inputs(repo_root) {
        deps.insert(extra);
    }
    Ok(deps.to_vec())
}

fn extra_build_inputs(repo_root: &Path) -> Vec<PathBuf> {
    let mut out = vec![repo_root.join("Cargo.toml")];
    let cargo_lock = repo_root.join("Cargo.lock");
    if cargo_lock.is_file() {
        out.push(cargo_lock);
    }
    out
}

fn extract_lib_roots_from_bin(content: &str) -> Vec<String> {
    let mut out = SkipSet::new(16);
    for prefix in ["rust_tools::", "use rust_tools::", "use crate::"] {
        let mut start = 0usize;
        while let Some(idx) = content[start..].find(prefix) {
            let offset = start + idx + prefix.len();
            let rest = &content[offset..];

            // Extract until the next semicolon to bound the search
            let end_stmt = rest.find(';').unwrap_or(rest.len());
            let stmt = &rest[..end_stmt];

            // Extract all alphanumeric segments within the statement
            let mut word = String::new();
            for c in stmt.chars() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    word.push(c);
                } else {
                    if !word.is_empty() {
                        out.insert(word.clone());
                        word.clear();
                    }
                }
            }
            if !word.is_empty() {
                out.insert(word);
            }

            start = offset + end_stmt;
        }
    }
    out.to_vec()
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
    out: &mut SkipSet<PathBuf>,
) {
    let Some(start) = interner.get(start).cloned() else {
        return;
    };
    if !out.insert(start.as_ref().clone()) {
        return;
    }
    let mut q = vec![start.as_ref().clone()];
    while let Some(u) = q.pop() {
        let u_rc = interner
            .get(&u)
            .cloned()
            .unwrap_or_else(|| Rc::new(u.clone()));
        for v in graph.adj(&u_rc) {
            let v_path = v.as_ref().clone();
            if out.insert(v_path.clone()) {
                q.push(v_path);
            }
        }
    }
}

fn newest_mtime(files: &[PathBuf]) -> Result<SystemTime, String> {
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
    fn parse_include_paths_resolves_literal_inputs() {
        let source = r#"
            const PROMPT: &str = include_str!("system_prompts/correctness_guardrails.md");
            const DATA: &[u8] = include_bytes!("data/blob.bin");
            include!("generated.rs");
        "#;
        let current_file = Path::new("src/bin/ai/driver/skill_runtime.rs");
        let current_dir = current_file.parent().unwrap();
        assert_eq!(
            parse_include_paths(source, current_file),
            vec![
                current_dir.join("system_prompts/correctness_guardrails.md"),
                current_dir.join("data/blob.bin"),
                current_dir.join("generated.rs"),
            ]
        );
    }

    #[test]
    fn parse_include_paths_handles_raw_strings() {
        let source = r####"
            include_str!(r"plain.md");
            include_bytes!(r#"one.bin"#);
            include!(r###"dir/a"b#c.rs"###);
        "####;
        assert_eq!(
            parse_include_paths(source, Path::new("src/main.rs")),
            vec![
                PathBuf::from("src/plain.md"),
                PathBuf::from("src/one.bin"),
                PathBuf::from("src/dir/a\"b#c.rs"),
            ]
        );
    }

    #[test]
    fn parse_include_paths_decodes_escaped_strings() {
        let source = r#"include_str!("dir\\quoted\"file\u{2e}\x6d\x64");
            include_str!("line\
                joined.md");"#;
        assert_eq!(
            parse_include_paths(source, Path::new("src/main.rs")),
            vec![
                PathBuf::from("src/dir\\quoted\"file.md"),
                PathBuf::from("src/linejoined.md")
            ]
        );
    }

    #[test]
    fn parse_include_paths_skips_comments_and_string_contents() {
        let source = r###"
            // include_str!("comment.md");
            /* outer /* include!("nested.rs"); */ include_bytes!("comment.bin"); */
            let text = "include_str!(\"quoted.md\")";
            let raw = r#"" include!("raw.rs"); "#;
            let bytes = br#"" include_bytes!("raw.bin"); "#;
            let quote = '"';
            let lifetime: &'static str = "text";
            other_include_str!("other.md");
            include_dir!("directory");
            include_str /* name */ ! // delimiter
                ( /* argument */ "real.md", /* trailing */ );
            include_bytes!["real.bin"];
            include! { "real.rs" }
        "###;
        assert_eq!(
            parse_include_paths(source, Path::new("src/main.rs")),
            vec![
                PathBuf::from("src/real.md"),
                PathBuf::from("src/real.bin"),
                PathBuf::from("src/real.rs")
            ]
        );
    }

    #[test]
    fn parse_include_paths_skips_non_ascii_byte_strings() {
        let source = r#"let _ = b"\xff";
include_str!("real.md");"#;
        assert_eq!(
            parse_include_paths(source, Path::new("src/main.rs")),
            vec![PathBuf::from("src/real.md")]
        );
    }

    #[test]
    fn parse_include_paths_handles_unclosed_literals_and_comments() {
        for source in [
            r#"let _ = "unterminated;"#,
            r#"let _ = b"\xff;"#,
            r###"let _ = br##"include_str!("fake.md");"#"###,
            "/* outer /* inner */",
        ] {
            assert!(parse_include_paths(source, Path::new("src/main.rs")).is_empty());
        }
    }

    #[test]
    fn parse_include_paths_skips_non_literal_arguments() {
        let source = r#"
            include_str!(concat!("prompts/", "dynamic.md"));
            include_bytes!(env!("ASSET_PATH"));
            include!(GENERATED_PATH);
            include_str!("prefix.md" + SUFFIX);
            include_str!("literal.md");
        "#;
        assert_eq!(
            parse_include_paths(source, Path::new("src/main.rs")),
            vec![PathBuf::from("src/literal.md")]
        );
    }

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
        let mut deps_set = SkipSet::new(16);
        for p in deps {
            deps_set.insert(p.clone());
        }

        assert!(deps_set.contains(&bin_dir.join("re").join("features").join("mod.rs")));
        assert!(deps_set.contains(&bin_dir.join("re").join("features").join("search.rs")));
    }

    #[test]
    fn deps_for_a_include_nested_file_module_children() {
        let cwd = env::current_dir().unwrap();
        let repo_root = find_repo_root(&cwd).unwrap();
        let src_dir = repo_root.join("src");
        let bin_dir = src_dir.join("bin");

        let all_rs_files = list_rs_files(&src_dir).unwrap();
        let (graph, interner) = build_module_graph(&all_rs_files).unwrap();

        let bin_rs = bin_dir.join("a.rs");
        let deps = deps_for_bin("a", &bin_rs, &repo_root, &graph, &interner).unwrap();
        let mut deps_set = SkipSet::new(16);
        for p in deps {
            deps_set.insert(p.clone());
        }

        assert!(deps_set.contains(&bin_dir.join("ai").join("prompt.rs")));
        assert!(deps_set.contains(&bin_dir.join("ai").join("prompt").join("multiline.rs")));
        assert!(
            deps_set.contains(
                &bin_dir
                    .join("ai")
                    .join("prompt")
                    .join("multiline")
                    .join("render.rs")
            )
        );
        assert!(
            deps_set.contains(&bin_dir.join("ai/driver/system_prompts/correctness_guardrails.md"))
        );
    }
}
