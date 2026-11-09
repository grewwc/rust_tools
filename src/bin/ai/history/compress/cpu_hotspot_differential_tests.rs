//! Byte-level parity against the pre-optimization overlap and archive lookup loops.

use super::super::dedup::dedup_overlapping_read_file_results;
use super::*;
use rustc_hash::{FxHashMap, FxHashSet};

struct ReferenceRead {
    message_idx: usize,
    tool_call_id: String,
    path: String,
    lines: Vec<(usize, String)>,
}

fn reference_overlap_dedup(
    messages: &mut [Message],
    signatures: &FxHashMap<String, (String, String)>,
    protected_indices: &FxHashSet<usize>,
    protected_ids: &FxHashSet<String>,
) {
    let mut prior_results = Vec::<ReferenceRead>::new();
    for idx in tool_message_indices(messages) {
        let Some(id) = messages[idx].tool_call_id.clone() else {
            continue;
        };
        let Some((tool_name, args)) = signatures.get(&id) else {
            continue;
        };
        if tool_name != "read_file" {
            continue;
        }
        let Some(path) = serde_json::from_str::<Value>(args)
            .ok()
            .and_then(|value| value.get("file_path")?.as_str().map(str::to_string))
        else {
            continue;
        };
        let text = value_to_string(&messages[idx].content);
        let lines = text
            .lines()
            .map(|line| {
                let (number, content) = line.split_once('\t')?;
                Some((number.trim().parse::<usize>().ok()?, content.to_string()))
            })
            .collect::<Option<Vec<_>>>();
        let Some(lines) = lines.filter(|lines| !lines.is_empty()) else {
            continue;
        };
        if protected_indices.contains(&idx) || protected_ids.contains(&id) {
            prior_results.push(ReferenceRead {
                message_idx: idx,
                tool_call_id: id.clone(),
                path,
                lines,
            });
            continue;
        }
        for prior in &mut prior_results {
            if protected_indices.contains(&prior.message_idx)
                || protected_ids.contains(&prior.tool_call_id)
                || prior.path != path
            {
                continue;
            }
            // The old implementation rebuilt this same map for every prior read.
            let later_by_number = lines
                .iter()
                .map(|(number, content)| (*number, content.as_str()))
                .collect::<FxHashMap<_, _>>();
            let mut overlapping = FxHashSet::default();
            for (number, content) in &prior.lines {
                let Some(later_content) = later_by_number.get(number) else {
                    continue;
                };
                if *later_content != content {
                    overlapping.clear();
                    break;
                }
                overlapping.insert(*number);
            }
            if overlapping.is_empty() {
                continue;
            }
            let removed = overlapping.len();
            prior
                .lines
                .retain(|(number, _)| !overlapping.contains(number));
            let rendered = if prior.lines.is_empty() {
                format!(
                    "[overlap dedup: all {removed} numbered lines are present verbatim in a later read_file result]"
                )
            } else {
                let mut output = format!(
                    "[overlap dedup: {removed} numbered lines are present verbatim in a later read_file result]\n"
                );
                for (number, content) in &prior.lines {
                    use std::fmt::Write as _;
                    writeln!(output, "{number:>6}\t{content}").unwrap();
                }
                output.pop();
                output
            };
            messages[prior.message_idx].content = Value::String(rendered);
        }
        prior_results.push(ReferenceRead {
            message_idx: idx,
            tool_call_id: id.clone(),
            path,
            lines,
        });
    }
}

fn message(role: &str, content: Value) -> Message {
    Message {
        role: role.to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

#[test]
fn numbered_overlap_index_matches_sequential_bytes() {
    for seed in 0usize..64 {
        let mut messages = vec![message(
            "user",
            Value::String("Keep original user text.".to_string()),
        )];
        let mut signatures = FxHashMap::default();
        let mut protected_indices = FxHashSet::default();
        let mut protected_ids = FxHashSet::default();
        for read in 0usize..32 {
            let id = format!("read-{read}");
            let start = (read * 7 + seed) % 9;
            let mut lines = (0..(read % 7 + 1))
                .map(|offset| {
                    let number = start + offset;
                    format!(
                        "{number:>6}\tline {number}: revision {} 支付",
                        (read + seed) % 3
                    )
                })
                .collect::<Vec<_>>();
            if (read + seed) % 5 == 0 {
                lines.reverse();
                lines.push(lines[0].clone());
            }
            let text = if (read + seed) % 13 == 0 {
                "not numbered output".to_string()
            } else {
                lines.join("\n")
            };
            let content = if (read + seed) % 11 == 0 {
                serde_json::json!([{"type": "text", "text": text}])
            } else {
                Value::String(text)
            };
            let mut result = message("tool", content);
            if (read + seed) % 17 != 0 {
                result.tool_call_id = Some(id.clone());
            }
            let idx = messages.len();
            messages.push(result);
            if (read + seed) % 4 == 0 {
                protected_indices.insert(idx);
            }
            if (read + seed) % 8 == 0 {
                protected_ids.insert(id.clone());
            }
            if (read + seed) % 19 != 0 {
                signatures.insert(
                    id,
                    (
                        if (read + seed) % 23 == 0 { "execute_command" } else { "read_file" }.to_string(),
                        serde_json::json!({"file_path": format!("/fixture/{}.rs", (read + seed) % 4)}).to_string(),
                    ),
                );
            }
        }
        let mut reference = messages.clone();
        for pass in 0..2 {
            reference_overlap_dedup(
                &mut reference,
                &signatures,
                &protected_indices,
                &protected_ids,
            );
            dedup_overlapping_read_file_results(
                &mut messages,
                &signatures,
                &protected_indices,
                &protected_ids,
            );
            assert_eq!(
                serde_json::to_vec(&messages).unwrap(),
                serde_json::to_vec(&reference).unwrap(),
                "seed={seed}, pass={pass}"
            );
        }
    }
}

fn reference_command_lookup(
    command: &str,
    overflow_dir: &Path,
    cwd: Option<&Path>,
    expected_output: &str,
) -> Option<PathBuf> {
    let preserved_dir = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .canonicalize()
        .ok()?;
    for entry in std::fs::read_dir(&preserved_dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        if !canonical.is_file() || canonical.parent() != Some(preserved_dir.as_path()) {
            continue;
        }
        let abs = canonical.to_string_lossy().into_owned();
        let mentioned = command.contains(&abs)
            || cwd
                .as_ref()
                .and_then(|cwd| canonical.strip_prefix(cwd).ok())
                .and_then(|relative| relative.to_str())
                .is_some_and(|relative| command.contains(relative))
            || canonical
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| command.contains(name));
        if mentioned
            && std::fs::read_to_string(&canonical)
                .is_ok_and(|archived| archived.trim() == expected_output.trim())
        {
            return Some(canonical);
        }
    }
    None
}

struct LookupAssets(PathBuf);

impl Drop for LookupAssets {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn command_archive_prefilter_matches_sequential_lookup() {
    let root = LookupAssets(
        std::env::temp_dir().join(format!("ai-archive-lookup-parity-{}", uuid::Uuid::new_v4())),
    );
    let dir = root.0.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    let evidence = "preserved evidence 支付\n";
    let first = dir.join("first.txt");
    let second = dir.join("second-支付.txt");
    std::fs::write(&first, evidence).unwrap();
    std::fs::write(&second, evidence).unwrap();
    std::fs::write(dir.join("different.txt"), "different evidence").unwrap();
    std::fs::create_dir(dir.join("directory.txt")).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&second, dir.join("alias.txt")).unwrap();
        let outside = root.0.join("outside.txt");
        std::fs::write(&outside, evidence).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("escape.txt")).unwrap();
        std::os::unix::fs::symlink(dir.join("missing.txt"), dir.join("broken.txt")).unwrap();
    }
    let commands = [
        "inspect revision 42".to_string(),
        "cat first.txt".to_string(),
        "cat second-支付.txt".to_string(),
        format!("cat '{}'; cat '{}'", first.display(), second.display()),
        format!("cat {PRESERVED_TOOL_OVERFLOW_DIR}/first.txt"),
        "awk '{print $1}' first.txt".to_string(),
        "cat different.txt first.txt second-支付.txt".to_string(),
        "cat alias.txt escape.txt broken.txt directory.txt".to_string(),
        "echo /unrelated/prefix-first.txt-suffix".to_string(),
    ];
    let canonical_first = first.canonicalize().unwrap();
    for command in commands {
        for cwd in [
            None,
            Some(root.0.as_path()),
            Some(dir.as_path()),
            Some(canonical_first.as_path()),
        ] {
            for output in [
                evidence,
                evidence.trim(),
                "transformed output",
                "different evidence",
            ] {
                assert_eq!(
                    archived_asset_path_in_command(&command, &root.0, cwd, output),
                    reference_command_lookup(&command, &root.0, cwd, output),
                    "command={command:?}, cwd={cwd:?}, output={output:?}"
                );
            }
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let odd = dir.join(std::ffi::OsString::from_vec(b"odd-\xff.txt".to_vec()));
        std::fs::write(&odd, evidence).unwrap();
        let command = format!("cat {}", odd.to_string_lossy());
        assert_eq!(
            archived_asset_path_in_command(&command, &root.0, None, evidence),
            reference_command_lookup(&command, &root.0, None, evidence)
        );
    }
}
