use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

const MAX_STRUCTURAL_FILES: usize = 5_000;
const MAX_STRUCTURAL_MATCHES: usize = 200;
const MAX_STRUCTURAL_FILE_SIZE: u64 = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StructuralSearch<'a> {
    RawQuery(&'a str),
    Intent(&'a str),
}

impl StructuralSearch<'_> {
    fn is_find_calls(self) -> bool {
        matches!(self, StructuralSearch::Intent("find_calls"))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StructuralFilters<'a> {
    pub(crate) name: Option<&'a str>,
    pub(crate) contains_text: Option<&'a str>,
    pub(crate) call_kind: Option<&'a str>,
    pub(crate) receiver: Option<&'a str>,
    pub(crate) qualified_name: Option<&'a str>,
}

#[derive(Debug, Clone)]
struct StructuralMatch {
    file_path: String,
    line: usize,
    captures: Vec<(String, String)>,
}

pub(crate) fn execute_structural_search(
    request: StructuralSearch<'_>,
    target: &str,
    file_pattern: Option<&str>,
    filters: StructuralFilters<'_>,
) -> Result<String, String> {
    let target_path = Path::new(target);
    let files = collect_target_files(target_path, file_pattern)?;
    if files.is_empty() {
        return Ok(format!(
            "No supported source files found under '{}' for AST structural search.",
            target
        ));
    }

    // 并行 AST 搜索，但结果必须全局限流；否则宽泛查询会把依赖/生成代码
    // 的所有 capture 都堆进内存，最后在格式化输出前就可能把进程打爆。
    let max_threads = rust_tools::commonw::half_parallelism();
    let chunk_size = (files.len() / max_threads).max(1);
    let chunks: Vec<Vec<PathBuf>> = files.chunks(chunk_size).map(|c| c.to_vec()).collect();

    let all_matches: Arc<Mutex<Vec<(String, Vec<StructuralMatch>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let error_occurred: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let unsupported_count: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let match_count: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let limit_reached: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        for chunk in chunks {
            let matches_ref = Arc::clone(&all_matches);
            let error_ref = Arc::clone(&error_occurred);
            let unsupported_ref = Arc::clone(&unsupported_count);
            let match_count_ref = Arc::clone(&match_count);
            let limit_ref = Arc::clone(&limit_reached);
            scope.spawn(move || {
                let mut local_results: Vec<(String, Vec<StructuralMatch>)> = Vec::new();
                for file in chunk {
                    if limit_ref.load(Ordering::Relaxed)
                        || match_count_ref.load(Ordering::Relaxed) >= MAX_STRUCTURAL_MATCHES
                    {
                        limit_ref.store(true, Ordering::Relaxed);
                        break;
                    }
                    if error_ref.lock().unwrap().is_some() {
                        break;
                    }
                    let Some((language_name, language)) = language_for_path(&file) else {
                        unsupported_ref.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let query = match resolve_query_for_language(request, language_name) {
                        Ok(Some(query)) => query,
                        Ok(None) => {
                            unsupported_ref.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        Err(err) => {
                            *error_ref.lock().unwrap() = Some(err);
                            return;
                        }
                    };
                    let Ok(metadata) = fs::metadata(&file) else {
                        continue;
                    };
                    if metadata.len() > MAX_STRUCTURAL_FILE_SIZE {
                        continue;
                    }
                    let content = match fs::read_to_string(&file) {
                        Ok(content) => content,
                        Err(_) => continue,
                    };
                    let filtered = match run_query_on_file(
                        language_name,
                        request,
                        filters,
                        language,
                        &query,
                        &file,
                        &content,
                        &match_count_ref,
                    ) {
                        Ok(file_matches) => file_matches,
                        Err(err) => {
                            *error_ref.lock().unwrap() = Some(format!(
                                "Failed to run AST structural search for {} file '{}': {}",
                                language_name,
                                file.display(),
                                err
                            ));
                            return;
                        }
                    };
                    if !filtered.is_empty() {
                        local_results.push((language_name.to_string(), filtered));
                    }
                    if match_count_ref.load(Ordering::Relaxed) >= MAX_STRUCTURAL_MATCHES {
                        limit_ref.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                if !local_results.is_empty() {
                    matches_ref.lock().unwrap().extend(local_results);
                }
            });
        }
    });

    // Check for errors
    if let Some(err) = error_occurred.lock().unwrap().as_ref() {
        return Err(err.clone());
    }

    let all_results = all_matches.lock().unwrap();
    let mut matches: Vec<StructuralMatch> = Vec::new();
    for (_, file_matches) in all_results.iter() {
        matches.extend(file_matches.iter().cloned());
    }
    let unsupported_files = unsupported_count.load(Ordering::Relaxed);

    if matches.is_empty() {
        return Ok(format!(
            "No AST structural matches for {} under '{}'{}{}.{}",
            describe_request(request),
            target,
            file_pattern
                .map(|p| format!(" with file_pattern '{}'", p))
                .unwrap_or_default(),
            describe_filters(filters),
            if unsupported_files > 0 {
                format!(" Skipped {} unsupported files.", unsupported_files)
            } else {
                String::new()
            }
        ));
    }

    matches.sort_by(|a, b| {
        a.file_path
            .cmp(&b.file_path)
            .then_with(|| a.line.cmp(&b.line))
    });
    matches.truncate(MAX_STRUCTURAL_MATCHES);

    Ok(format_structural_matches(
        &matches,
        limit_reached.load(Ordering::Relaxed)
            || match_count.load(Ordering::Relaxed) >= MAX_STRUCTURAL_MATCHES,
    ))
}

fn describe_filters(filters: StructuralFilters<'_>) -> String {
    let mut parts = Vec::new();
    if let Some(name) = filters.name.filter(|s| !s.trim().is_empty()) {
        parts.push(format!(" name contains '{}'", name));
    }
    if let Some(text) = filters.contains_text.filter(|s| !s.trim().is_empty()) {
        parts.push(format!(" capture contains '{}'", text));
    }
    if let Some(call_kind) = filters.call_kind.filter(|s| !s.trim().is_empty()) {
        parts.push(format!(" call_kind='{}'", call_kind));
    }
    if let Some(receiver) = filters.receiver.filter(|s| !s.trim().is_empty()) {
        parts.push(format!(" receiver contains '{}'", receiver));
    }
    if let Some(qualified_name) = filters.qualified_name.filter(|s| !s.trim().is_empty()) {
        parts.push(format!(" qualified_name contains '{}'", qualified_name));
    }
    parts.join("")
}

fn match_filters(item: &StructuralMatch, filters: StructuralFilters<'_>) -> bool {
    if let Some(name_filter) = filters.name.filter(|s| !s.trim().is_empty()) {
        let matches_name = item
            .captures
            .iter()
            .filter(|(capture_name, _)| capture_name == "name")
            .any(|(_, text)| text.contains(name_filter));
        if !matches_name {
            return false;
        }
    }

    if let Some(text_filter) = filters.contains_text.filter(|s| !s.trim().is_empty()) {
        let matches_text = item
            .captures
            .iter()
            .any(|(_, text)| text.contains(text_filter));
        if !matches_text {
            return false;
        }
    }

    if let Some(call_kind_filter) = filters.call_kind.filter(|s| !s.trim().is_empty()) {
        let matches_call_kind = item
            .captures
            .iter()
            .filter(|(capture_name, _)| capture_name == "call_kind")
            .any(|(_, text)| text == call_kind_filter);
        if !matches_call_kind {
            return false;
        }
    }

    if let Some(receiver_filter) = filters.receiver.filter(|s| !s.trim().is_empty()) {
        let matches_receiver = item
            .captures
            .iter()
            .filter(|(capture_name, _)| capture_name == "receiver")
            .any(|(_, text)| text.contains(receiver_filter));
        if !matches_receiver {
            return false;
        }
    }

    if let Some(qualified_name_filter) = filters.qualified_name.filter(|s| !s.trim().is_empty()) {
        let matches_qualified_name = item
            .captures
            .iter()
            .filter(|(capture_name, _)| capture_name == "qualified_name")
            .any(|(_, text)| text.contains(qualified_name_filter));
        if !matches_qualified_name {
            return false;
        }
    }

    true
}

fn normalize_match(
    language_name: &str,
    request: StructuralSearch<'_>,
    item: StructuralMatch,
) -> StructuralMatch {
    if !request.is_find_calls() {
        return item;
    }

    let mut primary_capture: Option<(String, String)> = None;
    let mut normalized = Vec::with_capacity(item.captures.len() + 5);
    for (capture_name, text) in item.captures {
        if primary_capture.is_none() && matches!(capture_name.as_str(), "name" | "constructor_name")
        {
            primary_capture = Some((capture_name, text));
            continue;
        }
        normalized.push((capture_name, text));
    }

    let Some((primary_capture_name, raw_name)) = primary_capture else {
        return StructuralMatch {
            file_path: item.file_path,
            line: item.line,
            captures: normalized,
        };
    };

    let normalized_name = normalize_call_name(language_name, &raw_name);
    let qualified_name = normalize_qualified_name(language_name, &raw_name);
    let receiver = extract_receiver(&qualified_name);
    let call_kind = classify_call_kind(&primary_capture_name, &receiver);

    normalized.push((
        "name".to_string(),
        if normalized_name.is_empty() {
            raw_name.clone()
        } else {
            normalized_name.clone()
        },
    ));
    normalized.push(("call_kind".to_string(), call_kind.to_string()));

    if !qualified_name.is_empty() {
        normalized.push(("qualified_name".to_string(), qualified_name.clone()));
    }
    if let Some(receiver) = receiver.filter(|s| !s.is_empty()) {
        normalized.push(("receiver".to_string(), receiver));
    }
    if primary_capture_name == "name" {
        normalized.push(("raw_name".to_string(), raw_name));
    } else if raw_name != normalized_name && raw_name != qualified_name {
        normalized.push(("raw_name".to_string(), raw_name));
    } else if primary_capture_name == "constructor_name" {
        normalized.push(("raw_name".to_string(), raw_name));
    }

    StructuralMatch {
        file_path: item.file_path,
        line: item.line,
        captures: normalized,
    }
}

fn normalize_call_name(_language_name: &str, raw: &str) -> String {
    let s = raw.trim();
    if s.is_empty() {
        return String::new();
    }

    let candidate = s;

    if let Some(last) = last_identifier(candidate) {
        return last;
    }

    candidate.to_string()
}

fn normalize_qualified_name(_language_name: &str, raw: &str) -> String {
    let s = raw.trim();
    if s.is_empty() {
        return String::new();
    }
    s.to_string()
}

fn extract_receiver(qualified_name: &str) -> Option<String> {
    let separators = ["?.", "->", "::", "."];
    let mut best: Option<(usize, usize)> = None;
    for sep in separators {
        if let Some(idx) = qualified_name.rfind(sep) {
            let end = idx;
            if best.is_none_or(|(best_idx, _)| idx > best_idx) {
                best = Some((idx, end));
            }
        }
    }
    let (_, end) = best?;
    let receiver = qualified_name[..end].trim();
    (!receiver.is_empty()).then(|| receiver.to_string())
}

fn classify_call_kind(primary_capture_name: &str, receiver: &Option<String>) -> &'static str {
    if primary_capture_name == "constructor_name" {
        "constructor_call"
    } else if receiver.is_some() {
        "method_call"
    } else {
        "function_call"
    }
}

fn last_identifier(s: &str) -> Option<String> {
    let mut end = None;
    for (idx, ch) in s.char_indices().rev() {
        if is_ident_char(ch) {
            end = Some(idx + ch.len_utf8());
            break;
        }
    }
    let end = end?;

    let mut start = 0usize;
    for (idx, ch) in s[..end].char_indices().rev() {
        if !is_ident_char(ch) {
            start = idx + ch.len_utf8();
            break;
        }
    }

    let ident = &s[start..end];
    (!ident.is_empty()).then(|| ident.to_string())
}

fn is_ident_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
}

fn describe_request(request: StructuralSearch<'_>) -> String {
    match request {
        StructuralSearch::RawQuery(query) => format!("query '{}'", query),
        StructuralSearch::Intent(intent) => format!("intent '{}'", intent),
    }
}

fn resolve_query_for_language(
    request: StructuralSearch<'_>,
    language_name: &str,
) -> Result<Option<String>, String> {
    match request {
        StructuralSearch::RawQuery(query) => Ok(Some(query.to_string())),
        StructuralSearch::Intent(intent) => structural_query_for_intent(language_name, intent),
    }
}

fn structural_query_for_intent(
    language_name: &str,
    intent: &str,
) -> Result<Option<String>, String> {
    let query = match (language_name, intent) {
        ("rust", "find_functions") => {
            "(function_item name: (identifier) @name)".to_string()
        }
        ("rust", "find_classes") => [
            "(struct_item name: (type_identifier) @name)",
            "(enum_item name: (type_identifier) @name)",
            "(trait_item name: (type_identifier) @name)",
        ]
        .join("\n"),
        ("rust", "find_methods") => [
            "(impl_item body: (declaration_list (function_item name: (identifier) @name)))",
            "(trait_item body: (declaration_list (function_signature_item name: (identifier) @name)))",
        ]
        .join("\n"),
        ("rust", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        ("python", "find_functions") => {
            "(function_definition name: (identifier) @name)".to_string()
        }
        ("python", "find_classes") => {
            "(class_definition name: (identifier) @name)".to_string()
        }
        ("python", "find_methods") => {
            "(class_definition body: (block (function_definition name: (identifier) @name)))"
                .to_string()
        }
        ("python", "find_calls") => {
            "(call function: (_) @name)".to_string()
        }

        ("java", "find_functions") | ("java", "find_methods") => [
            "(method_declaration name: (identifier) @name)",
            "(constructor_declaration name: (identifier) @name)",
        ]
        .join("\n"),
        ("java", "find_classes") => [
            "(class_declaration name: (identifier) @name)",
            "(interface_declaration name: (identifier) @name)",
            "(enum_declaration name: (identifier) @name)",
            "(record_declaration name: (identifier) @name)",
            "(annotation_type_declaration name: (identifier) @name)",
        ]
        .join("\n"),
        ("java", "find_calls") => [
            "(method_invocation name: (identifier) @name)",
            "(object_creation_expression type: (_) @constructor_name)",
        ]
        .join("\n"),

        ("go", "find_functions") => [
            "(function_declaration name: (identifier) @name)",
            "(method_declaration name: (field_identifier) @name)",
        ]
        .join("\n"),
        ("go", "find_classes") => [
            "(type_spec name: (type_identifier) @name type: (struct_type))",
            "(type_spec name: (type_identifier) @name type: (interface_type))",
        ]
        .join("\n"),
        ("go", "find_methods") => {
            "(method_declaration name: (field_identifier) @name)".to_string()
        }
        ("go", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        ("javascript", "find_functions") => [
            "(function_declaration name: (identifier) @name)",
            "(generator_function_declaration name: (identifier) @name)",
        ]
        .join("\n"),
        ("javascript", "find_classes") => {
            "(class_declaration name: (identifier) @name)".to_string()
        }
        ("javascript", "find_methods") => {
            "(method_definition name: (property_identifier) @name)".to_string()
        }
        ("javascript", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        ("typescript", "find_functions") => [
            "(function_declaration name: (identifier) @name)",
            "(generator_function_declaration name: (identifier) @name)",
        ]
        .join("\n"),
        ("typescript", "find_classes") => [
            "(class_declaration name: (type_identifier) @name)",
            "(interface_declaration name: (type_identifier) @name)",
            "(enum_declaration name: (identifier) @name)",
            "(type_alias_declaration name: (type_identifier) @name)",
        ]
        .join("\n"),
        ("typescript", "find_methods") => [
            "(method_definition name: (property_identifier) @name)",
            "(method_signature name: (property_identifier) @name)",
        ]
        .join("\n"),
        ("typescript", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        ("c", "find_functions") => {
            "(function_definition declarator: (_) @name)".to_string()
        }
        ("c", "find_classes") => [
            "(struct_specifier name: (type_identifier) @name)",
            "(union_specifier name: (type_identifier) @name)",
            "(enum_specifier name: (type_identifier) @name)",
        ]
        .join("\n"),
        ("c", "find_methods") => {
            "(function_definition declarator: (_) @name)".to_string()
        }
        ("c", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        ("cpp", "find_functions") => {
            "(function_definition declarator: (_) @name)".to_string()
        }
        ("cpp", "find_classes") => [
            "(class_specifier name: (type_identifier) @name)",
            "(struct_specifier name: (type_identifier) @name)",
            "(enum_specifier name: (type_identifier) @name)",
        ]
        .join("\n"),
        ("cpp", "find_methods") => [
            "(class_specifier body: (field_declaration_list (function_definition declarator: (_) @name)))",
            "(class_specifier body: (field_declaration_list (declaration declarator: (_) @name)))",
            "(struct_specifier body: (field_declaration_list (function_definition declarator: (_) @name)))",
            "(struct_specifier body: (field_declaration_list (declaration declarator: (_) @name)))",
        ]
        .join("\n"),
        ("cpp", "find_calls") => {
            "(call_expression function: (_) @name)".to_string()
        }

        (
            "rust"
            | "python"
            | "java"
            | "cpp"
            | "go"
            | "javascript"
            | "typescript"
            | "c",
            other,
        ) => {
            return Err(format!("Unsupported structural intent: {}", other));
        }
        _ => return Ok(None),
    };
    Ok(Some(query))
}

fn run_query_on_file(
    language_name: &str,
    request: StructuralSearch<'_>,
    filters: StructuralFilters<'_>,
    language: Language,
    query_src: &str,
    file_path: &Path,
    content: &str,
    match_count: &AtomicUsize,
) -> Result<Vec<StructuralMatch>, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| format!("failed to set parser language: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("failed to parse source".to_string())?;
    let query = Query::new(&language, query_src)
        .map_err(|e| format!("invalid tree-sitter query: {}", e))?;

    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut out = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());
    while {
        matches.advance();
        matches.get().is_some()
    } {
        if match_count.load(Ordering::Relaxed) >= MAX_STRUCTURAL_MATCHES {
            break;
        }
        let m = matches.get().expect("checked is_some above");
        let mut captures = Vec::new();
        let mut line = None;
        for capture in m.captures.iter().copied() {
            let name = capture_names
                .get(capture.index as usize)
                .map(|s| (*s).to_string())
                .unwrap_or_default();
            let text = capture
                .node
                .utf8_text(content.as_bytes())
                .map(|s| truncate_chars(s.trim(), 160))
                .unwrap_or_default();
            if line.is_none() {
                line = Some(capture.node.start_position().row + 1);
            }
            captures.push((name, text));
        }
        if !captures.is_empty() {
            let item = normalize_match(
                language_name,
                request,
                StructuralMatch {
                    file_path: file_path.to_string_lossy().to_string(),
                    line: line.unwrap_or(1),
                    captures,
                },
            );
            if !match_filters(&item, filters) {
                continue;
            }
            if !try_reserve_match_slot(match_count) {
                break;
            }
            out.push(item);
        }
    }
    Ok(out)
}

fn try_reserve_match_slot(match_count: &AtomicUsize) -> bool {
    match_count
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
            (count < MAX_STRUCTURAL_MATCHES).then_some(count + 1)
        })
        .is_ok()
}

fn format_structural_matches(matches: &[StructuralMatch], limit_reached: bool) -> String {
    let mut out = String::new();
    out.push_str("AST structural matches:\n");
    for item in matches {
        out.push_str(&format!("{}:{}\n", item.file_path, item.line));
        for (name, text) in &item.captures {
            out.push_str(&format!("  @{} = {}\n", name, text));
        }
    }
    if limit_reached {
        out.push_str(&format!(
            "\n... (match limit reached; showing first {} matches, narrow path/file_pattern/name/contains_text to continue)",
            MAX_STRUCTURAL_MATCHES
        ));
    }
    out.trim_end().to_string()
}

fn collect_target_files(target: &Path, file_pattern: Option<&str>) -> Result<Vec<PathBuf>, String> {
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }
    if !target.exists() {
        return Err(format!("Path not found: {}", target.display()));
    }
    if !target.is_dir() {
        return Err(format!(
            "Target is not a file or directory: {}",
            target.display()
        ));
    }

    if let Some(pattern) = file_pattern.filter(|p| !p.trim().is_empty()) {
        let root = target.to_string_lossy().to_string();
        let matches = crate::terminalw::glob_paths(pattern, &root)
            .map_err(|e| format!("file_pattern glob failed: {}", e))?;
        let files = matches
            .into_iter()
            .map(PathBuf::from)
            .filter(|p| p.is_file() && !path_contains_skipped_dir(p))
            .take(MAX_STRUCTURAL_FILES)
            .collect::<Vec<_>>();
        return Ok(files);
    }

    let mut out = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(target.to_path_buf());
    let mut scanned_dirs = 0usize;
    let max_dirs = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        if out.len() >= MAX_STRUCTURAL_FILES {
            break;
        }
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            break;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            if out.len() >= MAX_STRUCTURAL_FILES {
                break;
            }
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() && language_for_path(&path).is_some() {
                out.push(path);
            } else if ft.is_dir() && !ft.is_symlink() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !should_skip_dir(name) {
                    queue.push_back(path);
                }
            }
        }
    }
    Ok(out)
}

fn should_skip_dir(name: &str) -> bool {
    name.starts_with('.') || rust_tools::commonw::is_skip_dir(name)
}

fn path_contains_skipped_dir(path: &Path) -> bool {
    use std::path::Component;
    path.components().any(|component| match component {
        Component::Normal(name) => name.to_str().map(should_skip_dir).unwrap_or(false),
        _ => false,
    })
}

fn language_for_path(path: &Path) -> Option<(&'static str, Language)> {
    let ext = path.extension().and_then(|s| s.to_str())?;
    match ext {
        "c" | "h" => Some(("c", Language::from(tree_sitter_c::LANGUAGE))),
        "rs" => Some(("rust", Language::from(tree_sitter_rust::LANGUAGE))),
        "py" => Some(("python", Language::from(tree_sitter_python::LANGUAGE))),
        "go" => Some(("go", Language::from(tree_sitter_go::LANGUAGE))),
        "js" | "jsx" => Some((
            "javascript",
            Language::from(tree_sitter_javascript::LANGUAGE),
        )),
        "ts" => Some((
            "typescript",
            Language::from(tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
        )),
        "tsx" => Some((
            "typescript",
            Language::from(tree_sitter_typescript::LANGUAGE_TSX),
        )),
        "java" => Some(("java", Language::from(tree_sitter_java::LANGUAGE))),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => {
            Some(("cpp", Language::from(tree_sitter_cpp::LANGUAGE)))
        }
        _ => None,
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ai_ast_structural_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn language_for_rust_path_is_detected() {
        let lang = language_for_path(Path::new("/tmp/foo.rs"));
        assert!(lang.is_some());
        assert_eq!(lang.unwrap().0, "rust");
    }

    #[test]
    fn truncates_long_capture_text() {
        let s = truncate_chars(&"x".repeat(200), 10);
        assert!(s.starts_with("xxxxxxxxxx"));
        assert!(s.ends_with("..."));
    }

    #[test]
    fn rust_intent_query_is_generated() {
        let query = structural_query_for_intent("rust", "find_functions")
            .unwrap()
            .unwrap();
        assert!(query.contains("function_item"));
    }

    #[test]
    fn unsupported_intent_returns_error() {
        let err = structural_query_for_intent("rust", "find_whatever").unwrap_err();
        assert!(err.contains("Unsupported structural intent"));
    }

    #[test]
    fn collect_target_files_skips_hidden_and_dependency_dirs() {
        let dir = make_temp_dir("skip_dirs");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("target/debug")).unwrap();
        fs::create_dir_all(dir.join(".opencode/cache")).unwrap();
        fs::write(dir.join("src/lib.rs"), "fn keep() {}\n").unwrap();
        fs::write(dir.join("target/debug/generated.rs"), "fn generated() {}\n").unwrap();
        fs::write(
            dir.join(".opencode/cache/generated.js"),
            "function generated() {}\n",
        )
        .unwrap();

        let files = collect_target_files(&dir, None).unwrap();
        let rendered = files
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("src/lib.rs"), "{}", rendered);
        assert!(
            !rendered.contains("target/debug/generated.rs"),
            "{}",
            rendered
        );
        assert!(
            !rendered.contains(".opencode/cache/generated.js"),
            "{}",
            rendered
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn structural_search_caps_broad_results() {
        let dir = make_temp_dir("limit");
        let file = dir.join("many.rs");
        let mut source = String::new();
        for idx in 0..(MAX_STRUCTURAL_MATCHES + 50) {
            source.push_str(&format!("fn f_{idx}() {{}}\n"));
        }
        fs::write(&file, source).unwrap();

        let result = execute_structural_search(
            StructuralSearch::Intent("find_functions"),
            &dir.to_string_lossy(),
            None,
            StructuralFilters::default(),
        )
        .unwrap();

        let names = result.matches("@name =").count();
        assert_eq!(names, MAX_STRUCTURAL_MATCHES, "{}", result);
        assert!(result.contains("match limit reached"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn name_filter_matches_name_capture() {
        let item = StructuralMatch {
            file_path: "/tmp/foo.rs".to_string(),
            line: 1,
            captures: vec![("name".to_string(), "render_widget".to_string())],
        };
        assert!(match_filters(
            &item,
            StructuralFilters {
                name: Some("render"),
                contains_text: None,
                call_kind: None,
                receiver: None,
                qualified_name: None,
            }
        ));
        assert!(!match_filters(
            &item,
            StructuralFilters {
                name: Some("parse"),
                contains_text: None,
                call_kind: None,
                receiver: None,
                qualified_name: None,
            }
        ));
    }

    #[test]
    fn contains_text_filter_matches_any_capture() {
        let item = StructuralMatch {
            file_path: "/tmp/foo.rs".to_string(),
            line: 1,
            captures: vec![
                ("name".to_string(), "render_widget".to_string()),
                ("body".to_string(), "println!(\"hello\")".to_string()),
            ],
        };
        assert!(match_filters(
            &item,
            StructuralFilters {
                name: None,
                contains_text: Some("println"),
                call_kind: None,
                receiver: None,
                qualified_name: None,
            }
        ));
    }

    #[test]
    fn call_name_normalization_uses_last_identifier() {
        assert_eq!(normalize_call_name("rust", "foo::bar::baz"), "baz");
        assert_eq!(normalize_call_name("javascript", "obj?.render"), "render");
    }

    #[test]
    fn find_calls_normalization_keeps_raw_name() {
        let item = StructuralMatch {
            file_path: "/tmp/foo.rs".to_string(),
            line: 1,
            captures: vec![("name".to_string(), "foo.bar.baz".to_string())],
        };
        let normalized =
            normalize_match("javascript", StructuralSearch::Intent("find_calls"), item);
        assert_eq!(
            normalized.captures[0],
            ("name".to_string(), "baz".to_string())
        );
        assert!(
            normalized
                .captures
                .contains(&("call_kind".to_string(), "method_call".to_string()))
        );
        assert!(
            normalized
                .captures
                .contains(&("qualified_name".to_string(), "foo.bar.baz".to_string()))
        );
        assert!(
            normalized
                .captures
                .contains(&("receiver".to_string(), "foo.bar".to_string()))
        );
        assert_eq!(
            normalized
                .captures
                .iter()
                .find(|(k, _)| k == "raw_name")
                .cloned()
                .unwrap(),
            ("raw_name".to_string(), "foo.bar.baz".to_string())
        );
    }

    #[test]
    fn constructor_calls_are_marked_explicitly() {
        let item = StructuralMatch {
            file_path: "/tmp/Foo.java".to_string(),
            line: 1,
            captures: vec![("constructor_name".to_string(), "Foo".to_string())],
        };
        let normalized = normalize_match("java", StructuralSearch::Intent("find_calls"), item);
        assert!(
            normalized
                .captures
                .contains(&("call_kind".to_string(), "constructor_call".to_string()))
        );
        assert!(
            normalized
                .captures
                .contains(&("name".to_string(), "Foo".to_string()))
        );
    }

    #[test]
    fn call_metadata_filters_match_normalized_fields() {
        let item = StructuralMatch {
            file_path: "/tmp/foo.rs".to_string(),
            line: 1,
            captures: vec![
                ("name".to_string(), "render".to_string()),
                ("call_kind".to_string(), "method_call".to_string()),
                ("receiver".to_string(), "app.view".to_string()),
                ("qualified_name".to_string(), "app.view.render".to_string()),
            ],
        };
        assert!(match_filters(
            &item,
            StructuralFilters {
                name: None,
                contains_text: None,
                call_kind: Some("method_call"),
                receiver: Some("app.view"),
                qualified_name: Some("render"),
            }
        ));
        assert!(!match_filters(
            &item,
            StructuralFilters {
                name: None,
                contains_text: None,
                call_kind: Some("constructor_call"),
                receiver: None,
                qualified_name: None,
            }
        ));
    }
}
