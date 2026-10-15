use super::{render_text_attachment_block, text_file_contents};
use std::fs;
use std::path::PathBuf;

fn make_temp_path(name: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ai-attachment-{}-{}.{}",
        name,
        uuid::Uuid::new_v4(),
        ext
    ));
    path
}

#[test]
fn attachment_block_keeps_file_boundaries_for_multiple_files() {
    let first = make_temp_path("first", "txt");
    let second = make_temp_path("second", "txt");
    fs::write(&first, "alpha").unwrap();
    fs::write(&second, "beta").unwrap();

    let rendered = text_file_contents(&[
        first.to_string_lossy().to_string(),
        second.to_string_lossy().to_string(),
    ])
    .unwrap();

    assert!(rendered.contains(&format!("[Attached text file: {}]", first.display())));
    assert!(rendered.contains(&format!("[Attached text file: {}]", second.display())));
    assert!(rendered.contains("[/Attached text file]"));

    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
}

#[test]
fn attachment_block_truncates_large_files_and_points_to_read_file() {
    let path = make_temp_path("large", "rs");
    let content = (1..=400)
        .map(|idx| format!("fn item_{idx}() {{}}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&path, &content).unwrap();

    let rendered = render_text_attachment_block(path.to_string_lossy().as_ref()).unwrap();

    assert!(
        rendered.contains("Attachment preview only"),
        "rendered: {rendered}"
    );
    assert!(rendered.contains("read_file("), "rendered: {rendered}");

    let _ = fs::remove_file(path);
}

#[test]
fn key_lines_keep_diff_marker_semantics_and_index_labels() {
    // Context lines carry no `+`/`-`, so tagging them `[ctx]` is the only thing that
    // stops a preview reader from reporting unchanged code as part of the change.
    // Lines are joined explicitly: a `\`-continued string literal strips the next
    // line's indentation, which would lose the context marker space under test.
    let diff = [
        "@@ -1,3 +1,3 @@",
        "     /// Pre-existing doc comment.",
        "+    /// Added doc comment.",
        "-    /// Removed doc comment.",
    ]
    .join("\n");
    let lines = super::extract_key_lines(&diff, 10);
    assert!(
        lines.contains(&"#1: [ctx] /// Pre-existing doc comment.".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"#2: [+] /// Added doc comment.".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"#3: [-] /// Removed doc comment.".to_string()),
        "{lines:?}"
    );

    // A `-`-prefixed line outside any hunk (a commit-message bullet) is not part of
    // the change and stays untagged even though the payload contains a diff.
    let mixed = [
        "commit 1234abcd",
        "",
        "- fixed the error path",
        "@@ -1,3 +1,3 @@",
        " pub fn main() {",
    ]
    .join("\n");
    let mixed_lines = super::extract_key_lines(&mixed, 10);
    assert!(
        mixed_lines
            .iter()
            .any(|line| line.ends_with("- fixed the error path")),
        "the bullet stays visible as an untagged line: {mixed_lines:?}"
    );

    // Content that is not a diff keeps its plain shape, labeled by index rather than
    // by a line number the preview cannot resolve.
    let code = "fn main() {\n    let x = 1;\n}\n";
    assert_eq!(
        super::extract_key_lines(code, 10),
        vec!["#0: fn main() {".to_string()]
    );
}

#[test]
fn diff_tags_follow_hunk_counts_instead_of_text_guesses() {
    // Every row here defeats a text-based hunk guess: a removed row whose content starts
    // with `-- ` (SQL/Lua/Ada comments), the `\ No newline` note git prints between the
    // last removed and added row, and a trailing list after the hunk's counts run out.
    let diff = [
        "diff --git a/schema.sql b/schema.sql",
        "@@ -1,2 +1,2 @@",
        "- -- legacy comment",
        "+ -- updated comment",
        "\\ No newline at end of file",
        "",
        "- trailing bullet, not a change",
    ]
    .join("\n");

    assert_eq!(
        super::diff_line_tags(&diff),
        vec![
            None,
            None,
            Some("[-]"),
            Some("[+]"),
            None,
            Some("[ctx]"),
            None,
        ]
    );
}

#[test]
fn indented_hunk_bodies_keep_their_changed_tags() {
    // A wrapper that indents sub-command output must not turn every added or removed row
    // into unchanged context: both classification and marker stripping read the trimmed
    // text instead of the first byte of the raw line.
    let diff = ["  @@ -1,2 +1,2 @@", "  -  old line", "  +  new line"].join("\n");
    assert_eq!(
        super::diff_line_tags(&diff),
        vec![None, Some("[-]"), Some("[+]")]
    );

    let indented_fn = ["  @@ -1,2 +1,2 @@", "  +  fn added() {}"].join("\n");
    assert_eq!(
        super::extract_key_lines(&indented_fn, 5),
        vec!["#1: [+] fn added() {}".to_string()]
    );
}
