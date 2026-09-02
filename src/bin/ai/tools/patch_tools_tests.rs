use super::{
    PatchEnvelopeOp, apply_inline_replace, apply_patch_target_paths_from_patch,
    apply_unified_patch, apply_unified_patch_with_hints, execute_apply_patch,
    file_path_from_unified_diff_header, parse_patch_envelope, parse_patch_envelopes,
    parse_unified_diff_header_target, parse_unified_hunks, strip_code_fence,
    truncated_patch_hint,
};
use crate::ai::test_support::ENV_LOCK;
use std::{fs, path::PathBuf};

fn make_temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ai_patch_tools_test_{}_{}",
        name,
        uuid::Uuid::new_v4()
    ));
    path
}

#[test]
fn apply_patch_schema_does_not_expose_legacy_dry_run() {
    let schema = crate::ai::tools::registry::tool_metadata::tool_parameters("apply_patch");
    assert!(schema["properties"].get("dry_run").is_none());
}

/// Offline replay: uses the actual apply_patch inputs the model issued in a real session (history.json),
/// reconstructs the "file truth at the time" from them, and runs them through the **current** code, printing each patch's real
/// success/failure. This is not a conventional assertion test — it verifies the "real call success rate", not the assertions I wrote.
///
/// Ignored by default; enable when needed with:
///   AI_PATCH_REPLAY_DIR=/tmp/patch_review cargo test --bin a replay_apply_patch -- --ignored --nocapture
/// The directory needs replay_manifest.json + rebuild/proc.rs + rebuild/session_pid.rs.
#[test]
#[ignore]
fn replay_apply_patch_from_session() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let Ok(dir) = std::env::var("AI_PATCH_REPLAY_DIR") else {
        eprintln!("AI_PATCH_REPLAY_DIR not set; skipping replay");
        return;
    };
    let dir = PathBuf::from(dir);
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("replay_manifest.json")).unwrap())
            .unwrap();

    // Absolute path prefix in the session, replaced with the temporary workspace.
    const OLD_PREFIX: &str = "/Users/bytedance/rust_tools/src/bin/ai/driver";
    let records = manifest.as_array().unwrap();
    let mut ok = 0usize;
    let total = records.len();
    for rec in records {
        let msg = rec["msg"].as_i64().unwrap();
        let session = rec["session"].as_str().unwrap();
        let patch = rec["patch"].as_str().unwrap();
        let dry_run = rec["dry_run"].as_bool().unwrap_or(false);

        // Each patch uses a brand-new rebuilt file (they must not pollute each other).
        let work = make_temp_path(&format!("replay_{msg}"));
        let commands = work.join("commands");
        fs::create_dir_all(&commands).unwrap();
        fs::copy(dir.join("rebuild/proc.rs"), commands.join("proc.rs")).unwrap();
        fs::copy(
            dir.join("rebuild/session_pid.rs"),
            work.join("session_pid.rs"),
        )
        .unwrap();

        let new_prefix = work.to_string_lossy().to_string();
        let patch_rewritten = patch.replace(OLD_PREFIX, &new_prefix);

        let args = serde_json::json!({ "patch": patch_rewritten, "dry_run": dry_run });
        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(work.clone(), || execute_apply_patch(&args));
        let now = if result.is_ok() { "OK" } else { "FAIL" };
        if result.is_ok() {
            ok += 1;
        }
        let detail = match &result {
            Ok(s) => s.lines().next().unwrap_or("").to_string(),
            Err(e) => e.lines().next().unwrap_or("").to_string(),
        };
        eprintln!("msg{msg}: session={session} current_code={now} | {detail}");
        let _ = fs::remove_dir_all(&work);
    }
    eprintln!("=== replay success: {ok}/{total} ===");
}

#[test]
fn parse_unified_hunks_treats_empty_hunk_line_as_context() {
    // Models often write empty context lines as fully blank lines with no leading space; these should be treated as empty context lines,
    // not an error. This matches `git apply`'s tolerance for empty context lines.
    let patch = "@@ -1,3 +1,3 @@\n foo\n\n bar\n";
    let hunks =
        parse_unified_hunks(patch).expect("empty hunk line should be treated as context");
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].lines.len(), 3);
}

#[test]
fn apply_unified_patch_tolerates_empty_context_line() {
    // Models often write empty context lines as empty strings (no leading space); apply_patch should match them normally.
    let original = "foo\n\nbar\n";
    let patch = "@@ -1,3 +1,3 @@\n foo\n\n-bar\n+baz\n";
    let result =
        apply_unified_patch(original, patch).expect("empty context line should be tolerated");
    assert_eq!(result, "foo\n\nbaz\n");
}

#[test]
fn apply_unified_patch_strips_trailing_cr_from_crlf_patch() {
    // CRLF patch: the trailing \r on Add lines must not be written into the file content.
    let original = "foo\nbar\n";
    let patch = "@@ -2,1 +2,1 @@\r\n-bar\r\n+baz\r\n";
    let result = apply_unified_patch(original, patch).expect("CRLF patch should be tolerated");
    assert_eq!(result, "foo\nbaz\n");
}

#[test]
fn apply_unified_patch_tolerates_empty_context_line_in_crlf_patch() {
    // Empty context lines in a CRLF patch (lines with only \r) should also be treated as empty context lines.
    let original = "foo\r\n\r\nbar\r\n";
    let patch = "@@ -1,3 +1,3 @@\r\n foo\r\n\r\r\n-bar\r\n+baz\r\n";
    let result = apply_unified_patch(original, patch)
        .expect("empty CRLF context line should be tolerated");
    // The original file is CRLF, but the patch's Add lines have already stripped \r; output is uniformly LF.
    assert_eq!(result, "foo\n\nbaz\n");
}

#[test]
fn parse_unified_hunks_strips_trailing_blank_context_between_hunks() {
    // Hunks are separated by blank lines (a readability convention). Previously the blank line was swallowed into hunk1 as a trailing
    // empty context line, spuriously requiring the original file to have a blank line at that position → context mismatch.
    // After the fix, that trailing blank line should be stripped, leaving hunk1 with only the remove+add lines.
    let patch = "@@ -1,1 +1,1 @@\n-a\n+b\n\n@@ -5,1 +5,1 @@\n-c\n+d\n";
    let hunks = parse_unified_hunks(patch).expect("blank separator should be tolerated");
    assert_eq!(hunks.len(), 2);
    assert_eq!(
        hunks[0].lines.len(),
        2,
        "hunk1 should not swallow the blank separator"
    );
}

#[test]
fn apply_unified_patch_multi_hunk_separated_by_blank_line() {
    // Reproduces a real high-frequency scenario: hunks separated by blank lines. Before the fix, hunk1 ended with an extra empty context
    // line, making the whole patch report context mismatch.
    let original = "a\nkeep1\nkeep2\nkeep3\nc\n";
    let patch = "@@ -1,1 +1,1 @@\n-a\n+b\n\n@@ -5,1 +5,1 @@\n-c\n+d\n";
    let result = apply_unified_patch(original, patch)
        .expect("multi-hunk patch separated by a blank line should apply");
    assert_eq!(result, "b\nkeep1\nkeep2\nkeep3\nd\n");
}

#[test]
fn apply_unified_patch_tolerates_trailing_blank_line_in_patch() {
    // The patch ends with extra blank lines (common model output). Before the fix, the trailing blank lines were merged into the last hunk
    // as empty context lines → match failure.
    let original = "line1\nline2\nline3\n";
    let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n\n";
    let result = apply_unified_patch(original, patch)
        .expect("trailing blank line in patch should be tolerated");
    assert_eq!(result, "line1\nchanged\nline3\n");
}

#[test]
fn apply_unified_patch_tolerates_envelope_end_marker() {
    // Models often mistakenly append envelope tail markers like `*** End Patch` at the end of a unified-diff hunk
    // (format mixing). These markers do not belong to unified-diff content; the current hunk should end silently,
    // not report invalid hunk line.
    let original = "line1\nline2\nline3\n";
    let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** End Patch\n";
    let result = apply_unified_patch(original, patch)
        .expect("trailing `*** End Patch` marker should be tolerated");
    assert_eq!(result, "line1\nchanged\nline3\n");
}

#[test]
fn apply_unified_patch_rejects_envelope_section_marker_with_hint() {
    // When a unified-diff hunk mixes in `*** Begin Patch` / `*** Update File:` opener or
    // section markers, the patch structure is confused. It should error with an explicit "format mixing" message guiding the model
    // to rebuild with one of the two formats, instead of a generic invalid hunk line.
    let original = "line1\nline2\nline3\n";
    let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** Begin Patch\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("mixed patch formats"), "err was: {err}");
    assert!(err.contains("*** Begin Patch"), "err was: {err}");
}

#[test]
fn apply_unified_patch_rejects_malformed_envelope_trailer_not_silently_applied() {
    // Safety property: when a patch contains `*** Begin Patch` / `*** Update File:` envelope
    // markers (i.e. a truncated envelope leaked into the unified-diff path), a trailing `*** End Patch`
    // must never be silently tolerated and the hunk applied to the file_path target that the envelope did not declare.
    // It must report a "format mixing" error for the model to rebuild. Even if `original` here happens to contain the same
    // context (the most dangerous coincidence), it must error rather than write.
    let original = "line1\nline2\nline3\n";
    let patch = "*** Begin Patch\n*** Update File: other.rs\n@@ -2,1 +2,1 @@\n-line2\n+changed\n*** End Patch\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("mixed patch formats"), "err was: {err}");
    assert!(err.contains("*** End Patch"), "err was: {err}");
}

#[test]
fn apply_unified_patch_applies_simple_hunk() {
    let original = "line1\nline2\nline3\n";
    let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n";
    let result = apply_unified_patch(original, patch).unwrap();
    assert_eq!(result, "line1\nchanged\nline3\n");
}

#[test]
fn apply_unified_patch_context_mismatch_includes_actual_content() {
    let original = "alpha\nbeta\ngamma\n";
    // Deleting content that does not exist should trigger a context mismatch with context.
    let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    assert!(
        err.lines()
            .next()
            .unwrap_or_default()
            .contains("Rebuild the patch from the current file text"),
        "first error line should include the recovery action: {err}"
    );
    // The error should echo the expected line and the actual file content so the model can self-correct.
    assert!(err.contains("not_present"), "err was: {err}");
    assert!(err.contains("beta"), "err was: {err}");
    // Should include a directly pasteable current text block with no line-number prefix.
    assert!(err.contains("<<<PATCH_TEXT"), "err was: {err}");
    assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
}

#[test]
fn apply_unified_patch_context_mismatch_reports_unicode_code_points() {
    // Uses a genuinely "non-confusable" Unicode difference to trigger the mismatch, verifying the error echoes the code point.
    // Note: smart quotes (U+201C/U+201D) and ASCII quotes are already tolerated by normalize_confusables normalization
    // (see apply_unified_patch_tolerates_confusable_quotes), so they can no longer serve as mismatch samples.
    // Here we use accented é (U+00E9) vs e (U+0065) -- not in the confusable normalization range; a real difference.
    let original = "let label = \"café\";\n";
    let patch = "@@ -1,1 +1,1 @@\n-let label = \"cafe\";\n+let label = \"changed\";\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    assert!(err.contains("U+00E9"), "err was: {err}");
    assert!(err.contains("U+0065"), "err was: {err}");
}

#[test]
fn apply_unified_patch_tolerates_confusable_quotes() {
    // P0: models often auto-replace ASCII quotes/hyphens with typographic smart quotes / en-dash.
    // Such purely typographic differences must not cause a context mismatch -- after normalize_confusables normalization they should match.
    // Key safety property: the context line outputs the original file content (actual), not the smart quote from the patch,
    // so the ASCII characters in the file are never "replaced" with smart quotes.
    let original = "let quote = \"hi\";\nlet dash = a - b;\n";
    // context lines use smart quotes (“ ”), remove lines use en-dash (– U+2013),
    // the file has ASCII quotes / ASCII hyphen -- after normalization all should match.
    let patch = "@@ -1,2 +1,2 @@\n let quote = “hi”;\n-let dash = a – b;\n+let dash = a - b;\n";
    let result = apply_unified_patch(original, patch)
        .expect("confusable smart quotes / en-dash should be tolerated");
    // context lines keep the original file's ASCII quotes; the remove en-dash matches the file's ASCII hyphen and is deleted;
    // add lines write the patch content (ASCII hyphen).
    assert_eq!(result, "let quote = \"hi\";\nlet dash = a - b;\n");
}

#[test]
fn apply_unified_patch_detects_ambiguous_match() {
    // The same line appears multiple times in the file, and the nominal position does not match; an ambiguity error should be reported.
    let original = "dup\nmid\ndup\ntail\n";
    let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
    // The ambiguity error should echo the current line text of each candidate position so the model can pick a unique anchor without re-reading.
    assert!(
        err.contains("Candidate locations"),
        "ambiguous error should echo candidate current lines: {err}"
    );
    assert!(err.contains("line 1"), "err was: {err}");
    assert!(err.contains("line 3"), "err was: {err}");
}

/// When the model wrongly writes `@@ -0,0 +1,3 @@` (insert at file start), normalize it to old_start=1,
/// instead of treating it as "no nominal line number" (old_start=0) and reporting a context mismatch with "declared line 0".
#[test]
fn parse_unified_hunks_normalizes_zero_declared_line_to_one() {
    let patch = "@@ -0,0 +1,3 @@\n+aaa\n+bbb\n+ccc\n";
    let hunks = parse_unified_hunks(patch).expect("@@ -0 should parse");
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].old_start, 1);
    // And can be applied directly at the top of an empty file.
    let result = apply_unified_patch("", patch).expect("insert at top should apply");
    assert_eq!(result, "aaa\nbbb\nccc");
}

/// `@@ -0` insertion at the top of an existing file should also work.
#[test]
fn apply_unified_patch_inserts_at_top_with_zero_declared_line() {
    let original = "first\nsecond\n";
    let patch = "@@ -0,0 +1,2 @@\n+head\n+top\n";
    let result =
        apply_unified_patch(original, patch).expect("@@ -0 insert at top should apply");
    assert_eq!(result, "head\ntop\nfirst\nsecond\n");
}

/// A pure-insertion hunk (only `+` lines) on a non-empty file is located only by line number with no content verification;
/// on success it should return a hint reminding the model to re-read the file after the change.
#[test]
fn apply_unified_patch_pure_insert_reports_line_number_hint() {
    let original = "first\nsecond\n";
    let patch = "@@ -2,0 +3,2 @@\n+mid1\n+mid2\n";
    let (result, hints) =
        apply_unified_patch_with_hints(original, patch).expect("pure insert should apply");
    assert_eq!(result, "first\nmid1\nmid2\nsecond\n");
    assert!(
        hints.iter().any(|h| h.contains("line number")),
        "pure insert should carry a line-number hint, hints were: {hints:?}"
    );
}

/// Hunks with context/remove lines go through content verification, so no pure-insertion hint is produced.
#[test]
fn apply_unified_patch_no_hint_for_context_anchored_hunks() {
    let original = "first\nsecond\n";
    let patch = "@@ -1,2 +1,2 @@\n first\n-second\n+changed\n";
    let (result, hints) =
        apply_unified_patch_with_hints(original, patch).expect("context hunk should apply");
    assert_eq!(result, "first\nchanged\n");
    assert!(
        hints.is_empty(),
        "context-anchored hunk should have no hints: {hints:?}"
    );
}

/// Pure insertion on an empty file is the normal flow for creating a new file; no hint is produced.
#[test]
fn apply_unified_patch_pure_insert_on_empty_file_has_no_hint() {
    let patch = "@@ -0,0 +1,2 @@\n+a\n+b\n";
    let (result, hints) =
        apply_unified_patch_with_hints("", patch).expect("add file should apply");
    assert_eq!(result, "a\nb");
    assert!(
        hints.is_empty(),
        "empty-file insert should have no hints: {hints:?}"
    );
}

/// The pure-insertion hint should be returned with the success message (format_patch_success appends a note).
#[test]
fn apply_patch_success_message_includes_pure_insert_hint() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("pure_insert_hint");
    let target = base.join("target.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&target, "first\nsecond\n").unwrap();

    let patch = "@@ -2,0 +3,2 @@\n+mid1\n+mid2\n";
    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let result = execute_apply_patch(&serde_json::json!({
            "patch": patch,
            "file_path": "target.txt",
        }))
        .expect("pure insert should succeed");
        assert!(result.contains("Successfully patched"), "result was: {result}");
        assert!(
            result.contains("line number"),
            "success message should carry the pure-insert hint: {result}"
        );
    });

    assert_eq!(fs::read_to_string(&target).unwrap(), "first\nmid1\nmid2\nsecond\n");
    let _ = fs::remove_dir_all(base);
}

/// Truncation heuristic: recognize unclosed envelopes, broken `***` markers, and bare hunk-header endings;
/// legal endings must not be misreported.
#[test]
fn truncated_patch_hint_heuristics() {
    assert!(
        truncated_patch_hint(
            "*** Begin Patch\n*** Update File: x.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        )
        .is_some(),
        "unclosed envelope should be flagged"
    );
    assert!(
        truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n@@").is_some(),
        "trailing bare @@ should be flagged"
    );
    assert!(
        truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n*** End Patc").is_some(),
        "partial *** marker should be flagged"
    );
    // Legal endings must not be misreported.
    assert!(truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n").is_none());
    assert!(
        truncated_patch_hint(
            "*** Begin Patch\n*** Update File: x.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n*** End Patch\n"
        )
        .is_none()
    );
}

/// When parsing an unclosed (truncated) envelope fails, the error should include a truncation hint and a patch_file
/// alternative path, so the model does not treat the truncated text as its own syntax error and retry repeatedly.
#[test]
fn apply_patch_unclosed_envelope_error_hints_truncation_and_patch_file() {
    let patch = "*** Begin Patch\n*** Update File: missing.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n";
    let args = serde_json::json!({ "patch": patch });
    let err = execute_apply_patch(&args).unwrap_err();
    assert!(err.contains("patch_file"), "err was: {err}");
    assert!(err.contains("cut off"), "err was: {err}");
}

/// When the patch is missing, give a truncation hint and the patch_file alternative path.
#[test]
fn apply_patch_missing_patch_error_hints_truncation_and_patch_file() {
    let args = serde_json::json!({});
    let err = execute_apply_patch(&args).unwrap_err();
    assert!(err.contains("patch parameter is missing"), "err was: {err}");
    assert!(err.contains("patch_file"), "err was: {err}");
    assert!(err.contains("truncated"), "err was: {err}");
}

/// An oversized inline patch should error out immediately and guide splitting, rather than parse with the defect.
#[test]
fn apply_patch_rejects_oversized_inline_patch() {
    let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "x".repeat(9_000));
    let args = serde_json::json!({ "patch": huge });
    let err = execute_apply_patch(&args).unwrap_err();
    assert!(err.contains("patch too large"), "err was: {err}");
    assert!(err.contains("patch_file"), "err was: {err}");
}

/// A patch_file carrying a large patch (>8K inline cap) should succeed: the inline limit applies only to inline patches;
/// otherwise the audit-flagged "recommended fallback path is actually unusable" contradiction cannot be resolved.
#[test]
fn apply_patch_large_patch_file_above_inline_limit_applies() {
    let temp = make_temp_path("patch_file_large");
    std::fs::create_dir_all(&temp).unwrap();
    let patch_path = temp.join("large.patch");
    let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "y".repeat(9_000));
    std::fs::write(&patch_path, &huge).unwrap();
    let target = temp.join("target.txt");
    std::fs::write(&target, "a\n").unwrap();
    let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
        let args = serde_json::json!({
            "patch_file": patch_path.to_string_lossy(),
            "file_path": target.to_string_lossy(),
        });
        execute_apply_patch(&args)
    });
    let out = result.expect("large patch_file should apply");
    assert!(out.contains("+1 -1"), "out was: {out}");
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        format!("{}\n", "y".repeat(9_000))
    );
}

/// An empty patch_file materialized by the tool bridge counts as "not provided" and must not block a valid inline patch.
#[test]
fn apply_patch_inline_accepts_empty_patch_file_placeholder() {
    let temp = make_temp_path("empty_patch_file_placeholder");
    std::fs::create_dir_all(&temp).unwrap();
    let target = temp.join("target.txt");
    std::fs::write(&target, "old\n").unwrap();
    let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
        let args = serde_json::json!({
            "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n",
            "patch_file": "",
            "file_path": target.to_string_lossy(),
        });
        execute_apply_patch(&args)
    });
    result.expect("empty patch_file placeholder should be absent");
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "new\n");
}

/// When both sources are empty, missing-patch must still be explicitly reported rather than attempting to execute empty content.
#[test]
fn apply_patch_empty_source_placeholders_report_missing_patch() {
    let args = serde_json::json!({ "patch": "", "patch_file": null });
    let err = execute_apply_patch(&args).unwrap_err();
    assert!(err.contains("missing or empty"), "err was: {err}");
}

/// A patch_file exceeding the loose safety cap (64K) is explicitly rejected with guidance to split.
#[test]
fn apply_patch_rejects_oversized_patch_file() {
    let temp = make_temp_path("patch_file_huge");
    std::fs::create_dir_all(&temp).unwrap();
    let patch_path = temp.join("huge.patch");
    let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "z".repeat(70_000));
    std::fs::write(&patch_path, &huge).unwrap();
    let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
        let args = serde_json::json!({ "patch_file": patch_path.to_string_lossy() });
        execute_apply_patch(&args)
    });
    let err = result.unwrap_err();
    assert!(err.contains("patch_file too large"), "err was: {err}");
}

/// patch_file reads the patch from a file under effective_cwd (sync_scope) and applies it.
#[test]
fn apply_patch_reads_patch_from_patch_file_under_cwd() {
    let temp = make_temp_path("patch_file_cwd");
    std::fs::create_dir_all(&temp).unwrap();
    let patch_path = temp.join("edit.patch");
    std::fs::write(&patch_path, "@@ -1,1 +1,1 @@\n-foo\n+bar\n").unwrap();
    let target = temp.join("target.txt");
    std::fs::write(&target, "foo\n").unwrap();
    let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
        let args = serde_json::json!({
            "patch": "",
            "patch_file": patch_path.to_string_lossy(),
            "file_path": target.to_string_lossy(),
        });
        execute_apply_patch(&args)
    });
    let out = result.expect("patch_file should apply");
    assert!(out.contains("+1 -1"), "out was: {out}");
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "bar\n");
}

/// A patch_file pointing outside cwd and not registered in the temp registry must be explicitly rejected.
#[test]
fn apply_patch_rejects_patch_file_outside_cwd_and_registry() {
    let outside =
        std::env::temp_dir().join(format!("ai_patch_outside_{}.patch", uuid::Uuid::new_v4()));
    std::fs::write(&outside, "@@ -1,1 +1,1 @@\n-foo\n+bar\n").unwrap();
    let args = serde_json::json!({ "patch_file": outside.to_string_lossy() });
    let err = execute_apply_patch(&args).unwrap_err();
    assert!(
        err.contains("not an allowed patch source"),
        "err was: {err}"
    );
}

/// On a context mismatch where the file contains no partial match at all (the no-partial-match branch),
/// it should also attach a directly pasteable current text block with no line-number prefix.
#[test]
fn apply_unified_patch_context_mismatch_emits_pasteable_block_without_partial_match() {
    let original = "alpha\nbeta\ngamma\n";
    // The expected block and the file content have zero overlap, triggering the no-partial-match branch.
    let patch = "@@ -1,2 +1,2 @@\n-zzz1\n-zzz2\n+repl\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    assert!(err.contains("<<<PATCH_TEXT"), "err was: {err}");
    assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
    // The pasteable block contains real original file lines, without a `<line>: ` prefix.
    let block = err
        .split("<<<PATCH_TEXT\n")
        .nth(1)
        .and_then(|rest| rest.split("\nPATCH_TEXT>>>").next())
        .unwrap_or_default();
    assert!(block.contains("alpha"), "block was: {block:?}");
    assert!(
        !block.contains(':'),
        "pasteable block must not carry line-number prefixes: {block:?}"
    );
}

/// "hunks out of order" is no longer a bare string: it should state the reason, give pasteable current text, and
/// suggest reordering by line number or switching to Replace in line. In a real session the model lost 4 rounds in a row on the bare error.
#[test]
fn apply_unified_patch_out_of_order_error_is_actionable() {
    // File: `first` before `last`. The patch writes the hunks in reverse order: first modifies `last` (line 4),
    // then `first` (line 1) -- the second hunk's match position falls before the cursor, triggering out of order.
    let original = "first\naaa\nbbb\nlast\n";
    let patch = concat!("@@\n-last\n+LAST\n", "@@\n-first\n+FIRST\n",);
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("hunks out of order"), "err was: {err}");
    // Should include actionable reorder / Replace in line suggestions and a pasteable text block.
    assert!(
        err.contains("ascending file line number"),
        "err should explain ordering rule: {err}"
    );
    assert!(
        err.contains("<<<PATCH_TEXT"),
        "err should echo current text: {err}"
    );
    assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
    assert!(
        err.contains("consumed through 1-based line 4"),
        "err should report the previous hunk's inclusive end line: {err}"
    );
    assert!(
        err.contains("must start at 1-based line 5 or later"),
        "err should report the next hunk's earliest start line: {err}"
    );
}

#[test]
fn apply_unified_patch_disambiguates_ambiguous_match_by_nearby_declared_line() {
    let original = "dup\nhead\nfiller1\nfiller2\nfiller3\ndup\ntail\n";
    // `dup` appears twice, but the hunk header claims line 5, clearly closer to the second candidate (line 6).
    let patch = "@@ -5,1 +5,1 @@\n-dup\n+changed\n";
    let result = apply_unified_patch(original, patch)
        .expect("nearby declared line should disambiguate repeated context");
    assert_eq!(
        result,
        "dup\nhead\nfiller1\nfiller2\nfiller3\nchanged\ntail\n"
    );
}

#[test]
fn apply_unified_patch_rejects_declared_line_when_not_clear_nearest() {
    let original = "dup\nleft\nmid\ndup\nright\n";
    // The nominal line 3 sits between two `dup`s with equal candidate distance; it must not guess.
    let patch = "@@ -3,1 +3,1 @@\n-dup\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
}

#[test]
fn apply_unified_patch_finds_unique_match_beyond_search_radius() {
    // The file has 150 lines; the only match is at line 130 (0-based 129).
    // The hunk header claims line 1, nominal=0, and find_hunk_offset's ±50 window searches [0,50),
    // which cannot find the match at 129. But all_hunk_match_positions can find the unique match across the whole file.
    // Previously the code ignored the forward.len()==1 result and fell back to find_hunk_offset, causing a false
    // "context mismatch"。
    let mut lines: Vec<String> = (0..130).map(|i| format!("filler{i}")).collect();
    lines.push("unique_target".to_string());
    lines.push("after_target".to_string());
    lines.extend((0..18).map(|i| format!("tail{i}")));
    let original = lines.join("\n") + "\n";

    let patch = "@@ -1,2 +1,2 @@\n-unique_target\n+changed\n+after_target\n";
    // Deliberately uses a wrong nominal line number (-1) to simulate stale line numbers
    let result = apply_unified_patch(&original, patch).unwrap_or_else(|err| {
        panic!("apply_patch should find unique match beyond ±50 radius, but got: {err}")
    });
    assert!(
        result.contains("changed"),
        "result should contain changed line: {result}"
    );
    assert!(
        result.contains("after_target"),
        "result should preserve after_target: {result}"
    );
    assert!(
        !result.contains("unique_target"),
        "result should not contain old line: {result}"
    );
}

#[test]
fn apply_unified_patch_tolerates_leading_indent_mismatch() {
    // Real high-frequency failure scenario: in markdown/nested lists, the model's recreated context line indentation differs from the original file
    // (here the patch is missing 2 leading spaces). Before the fix, lines_match only did trim_end,
    // with zero tolerance for leading whitespace → the whole file failed to locate → "context mismatch: patch hunk could not
    // be located". After the fix, when strict matching fails, the indent-ignoring fallback uniquely locates and applies.
    let original = "# Title\n\n  - item one\n  - item two\n";
    // Leading spaces = context prefix; context content "- item one", remove content "- item two"
    // are both missing 2 indentation spaces compared to the original file.
    let patch = "@@ -3,2 +3,2 @@\n - item one\n-- item two\n+- item two changed\n";
    let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
        panic!("indent-insensitive fallback should locate the hunk, got: {err}")
    });
    // Context lines keep the original file's indentation; only the remove/add target lines are replaced.
    assert_eq!(result, "# Title\n\n  - item one\n- item two changed\n");
}

#[test]
fn apply_unified_patch_indent_fallback_still_detects_ambiguity() {
    // The indent-ignoring fallback must not sacrifice safety: if ignoring indentation yields multiple matches, it must still report ambiguity,
    // not silently change the wrong place.
    let original = "  dup\nmid\n    dup\ntail\n";
    let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
}

#[test]
fn apply_unified_patch_strict_match_preferred_over_indent_fallback() {
    // When strict matching uniquely locates, the strict-match result must be used, preserving the original file's exact content,
    // and must not fall back just because an indentation variant exists.
    let original = "    exact\nother\n";
    let patch = "@@ -1,1 +1,1 @@\n-    exact\n+    replaced\n";
    let result = apply_unified_patch(original, patch).unwrap();
    assert_eq!(result, "    replaced\nother\n");
}

#[test]
fn apply_unified_patch_fuzzes_stale_context_when_remove_lines_are_unique() {
    // Real loop root cause: the model wrote the context line as stale/target-state content, but the remove line still
    // anchors precisely on the target. Context must not hard-reject the patch in this case.
    let original = "alpha current\nold target\nomega current\n";
    let patch = "\
@@ -1,3 +1,3 @@
 alpha stale
-old target
+new target
 omega stale
";
    let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
        panic!("unique remove anchor should tolerate stale context, got: {err}")
    });
    assert_eq!(result, "alpha current\nnew target\nomega current\n");
}

#[test]
fn apply_unified_patch_fuzzy_context_uses_remaining_context_to_disambiguate() {
    // When a remove line appears twice, fuzz can still score using other context lines; only a unique top score
    // may be applied, avoiding degradation into "modify the first identical remove line".
    let original = "alpha current\nold target\ntail one\nbeta current\nold target\ntail two\n";
    let patch = "\
@@ -1,3 +1,3 @@
 stale head
-old target
+new target
 tail one
";
    let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
        panic!("tail context should disambiguate fuzzy candidate, got: {err}")
    });
    assert_eq!(
        result,
        "alpha current\nnew target\ntail one\nbeta current\nold target\ntail two\n"
    );
}

#[test]
fn apply_unified_patch_rejects_fuzzy_context_when_remove_anchor_is_ambiguous() {
    let original = "alpha current\nold target\nbeta current\nold target\n";
    let patch = "\
@@ -1,2 +1,2 @@
 stale context
-old target
+new target
";
    // old_start=1 (1-based) → nominal=0; remove "old target" matches at line 1.
    // Even if all context misses, old_start can still disambiguate — it should apply successfully.
    let result = apply_unified_patch(original, patch).expect("should apply via nominal");
    assert_eq!(
        result, "alpha current\nnew target\nbeta current\nold target\n",
        "should replace the FIRST 'old target' (line 1), not the second (line 3)"
    );
}

#[test]
fn apply_unified_patch_fuzzy_context_rejects_when_nominal_not_in_candidates() {
    // When the position old_start points to is not in the candidate list, it should still be rejected.
    // original: line 0="old target", line 1="xxx", line 2="old target", line 3="yyy"
    // patch: @@ -2,1 +2,1 @@ — old_start=2 → nominal=1
    // The hunk has only a remove line (no context); remove "xxx" appears at line 1.
    // But another variant: multiple "old target" as remove, and old_start points to a position with no match.
    // original: line 0="old target", line 1="aaa", line 2="old target", line 3="bbb"
    // patch: @@ -3,1 +3,1 @@ — old_start=3 → nominal=2
    // remove "old target" matches line 0 (pos=0) and line 2 (pos=2);
    // nominal=2 is in the candidate list → it is accepted (correct behavior).
    // Changed to: old_start points to a line number that does not exist in the file.
    let original = "old target\naaa\nold target\nbbb\n";
    let patch = "@@ -5,1 +5,1 @@\n-old target\n+changed\n";
    // old_start=5 → nominal=4, but the file has only 4 lines (index 0-3).
    // candidates: pos=0 and pos=2 (old target matches). nominal=4 is not in candidates.
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
}

#[test]
fn apply_unified_patch_indent_fallback_reports_context_mismatch_when_absent() {
    // Even ignoring indentation, if the content itself does not exist, it should still report a context mismatch (echoing the actual content).
    let original = "alpha\nbeta\ngamma\n";
    let patch = "@@ -2,1 +2,1 @@\n-  not_present\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
}

#[test]
fn execute_apply_patch_accepts_path_alias_and_begin_patch_envelope() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("update").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+changed\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args).expect("apply_patch should accept path alias and envelope");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_update_envelope_without_hunk_header() {
    // The *** Begin Patch Update format omits the @@ header (Cursor/Aider style),
    // writing only +/−/space-prefixed lines. Models write this often; it must not report "no hunks found".
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("update_nohdr").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n alpha\n-beta\n+changed\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args)
            .expect("apply_patch should accept Update envelope without @@ header");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "alpha\nchanged\ngamma\n"
    );
    let _ = fs::remove_dir_all(base);
}
#[test]
fn apply_unified_patch_multi_hunk_with_stale_line_numbers() {
    // Two hunks, both nominal line numbers 1 (stale), but each target is unique in the file and ordered.
    // Verifies that cursor advancement + forward filtering work correctly across multiple hunks, without mis-matching the second
    // hunk onto the first hunk's target position.
    let mut lines: Vec<String> = (0..60).map(|i| format!("filler{i}")).collect();
    lines.push("target_a".to_string());
    lines.push("after_a".to_string());
    lines.extend((0..60).map(|i| format!("mid{i}")));
    lines.push("target_b".to_string());
    lines.push("after_b".to_string());
    let original = lines.join("\n") + "\n";

    let patch = "\
@@ -1,2 +1,2 @@
-target_a
+changed_a
+after_a
@@ -1,2 +1,2 @@
-target_b
+changed_b
+after_b
";
    let result = apply_unified_patch(&original, patch).unwrap_or_else(|err| {
        panic!("multi-hunk patch should succeed with stale line numbers, but got: {err}")
    });
    assert!(result.contains("changed_a"), "missing changed_a: {result}");
    assert!(result.contains("changed_b"), "missing changed_b: {result}");
    assert!(result.contains("after_a"), "missing after_a: {result}");
    assert!(result.contains("after_b"), "missing after_b: {result}");
    assert!(
        !result.contains("target_a"),
        "should not contain target_a: {result}"
    );
    assert!(
        !result.contains("target_b"),
        "should not contain target_b: {result}"
    );
    // The filler lines in between should remain unchanged
    assert!(
        result.contains("filler0"),
        "filler0 should remain: {result}"
    );
    assert!(result.contains("mid0"), "mid0 should remain: {result}");
}

#[test]
fn execute_apply_patch_supports_add_file_envelope_without_file_path_arg() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("add_parent");
    let path = base.join("new.txt");
    fs::create_dir_all(&base).unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": "*** Begin Patch\n*** Add File: new.txt\n+hello\n+world\n*** End Patch\n"
        });
        execute_apply_patch(&args)
            .expect("apply_patch should infer target from Add File envelope");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "hello\nworld");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_add_file_tolerates_empty_lines() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("add_empty");
    let path = base.join("new.txt");
    fs::create_dir_all(&base).unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": "*** Begin Patch\n*** Add File: new.txt\n+hello\n\n+world\n*** End Patch\n"
        });
        execute_apply_patch(&args)
            .expect("apply_patch should tolerate empty lines in Add File envelope");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "hello\n\nworld");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_streaming_dispatch_emits_progress() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("streaming").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": "@@ -2,1 +2,1 @@\n-beta\n+changed\n"
        });
        let mut streamed = Vec::new();
        let mut capture = |chunk: &[u8]| streamed.extend_from_slice(chunk);
        let result = crate::ai::tools::common::execute_tool_call_with_args_streaming(
            "call_apply_patch_streaming",
            "apply_patch",
            &args,
            &mut capture,
        )
        .expect("streaming apply_patch should succeed");

        let streamed = String::from_utf8(streamed).expect("streamed output must be utf-8");
        assert!(
            streamed.contains("parsing patch envelope"),
            "streamed: {streamed}"
        );
        assert!(streamed.contains("target:"), "streamed: {streamed}");
        assert!(
            streamed.contains("applying 1 hunk(s)"),
            "streamed: {streamed}"
        );
        assert!(streamed.contains("writing "), "streamed: {streamed}");
        assert!(
            streamed.contains(&format!("Successfully patched {};", path.display())),
            "streamed: {streamed}"
        );
        assert!(
            result
                .content
                .starts_with(&format!("Successfully patched {};", path.display())),
            "result.content: {}",
            result.content
        );
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
    let _ = fs::remove_file(&path);
}

#[test]
fn execute_apply_patch_rejects_mismatched_envelope_target() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("mismatch_parent");
    let path = base.join("a.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\n").unwrap();

    let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": "*** Begin Patch\n*** Update File: b.txt\n@@ -1,1 +1,1 @@\n-alpha\n+beta\n*** End Patch\n"
        });
        execute_apply_patch(&args).expect_err("mismatched target must be rejected")
    });

    // file_path is silently ignored; the envelope declares b.txt as the authoritative target; b.txt does not exist → report missing file.
    assert!(
        err.contains("b.txt"),
        "err should mention the envelope target path: {err}"
    );
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_update_envelope_rejects_missing_target_file() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("update_missing_parent");
    let path = base.join("missing.txt");
    fs::create_dir_all(&base).unwrap();

    let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n+hello\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args).expect_err("Update File must not create a missing file")
    });

    assert!(
        err.contains("Update File patch targets a missing file"),
        "err was: {err}"
    );
    assert!(!path.exists(), "missing target must not be created");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_tilde_path_matches_between_arg_and_envelope() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let home = PathBuf::from(std::env::var("HOME").expect("HOME must be set"));
    let unique = format!("ai_patch_tools_home_{}", uuid::Uuid::new_v4());
    let base = home.join(&unique);
    let path = base.join("tilde.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\n").unwrap();

    let rel = path
        .strip_prefix(&home)
        .expect("test path should be under HOME");
    let tilde_path = format!("~/{}", rel.display());

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": tilde_path.clone(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+changed\n*** End Patch\n",
                tilde_path
            )
        });
        execute_apply_patch(&args)
            .expect("matching `~` paths in arg and envelope should resolve to the same file");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn strip_code_fence_removes_backtick_wrapper() {
    let fenced = "```diff\n@@ -1,1 +1,1 @@\n-line2\n+changed\n```";
    assert_eq!(
        strip_code_fence(fenced),
        "@@ -1,1 +1,1 @@\n-line2\n+changed"
    );
    // ~~~ fences are stripped as well.
    let fenced_tilde = "~~~\n@@ -1,1 +1,1 @@\n-x\n+y\n~~~";
    assert_eq!(strip_code_fence(fenced_tilde), "@@ -1,1 +1,1 @@\n-x\n+y");
}

#[test]
fn strip_code_fence_leaves_unfenced_patch_untouched() {
    let raw = "@@ -1,1 +1,1 @@\n-x\n+y";
    assert_eq!(strip_code_fence(raw), raw);
    // When the closing fence is missing, do not strip, to avoid damaging a real patch whose content starts with ```.
    let no_close = "```diff\n@@ -1,1 +1,1 @@\n-x\n+y";
    assert_eq!(strip_code_fence(no_close), no_close);
    // Do not process when there are too few lines.
    assert_eq!(strip_code_fence("```\n```"), "```\n```");
}

#[test]
fn execute_apply_patch_strips_code_fence_around_unified_diff() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("fence_unified").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "line1\nline2\nline3\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": "```diff\n@@ -1,3 +1,3 @@\n line1\n-line2\n+changed\n line3\n```"
        });
        execute_apply_patch(&args)
            .expect("apply_patch should strip code fence around unified diff");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "line1\nchanged\nline3\n"
    );
    let _ = fs::remove_dir_all(base);
}

#[test]
fn file_path_from_unified_diff_header_reads_git_style_paths() {
    // The `+++ b/` side takes priority; strip the `b/` prefix.
    assert_eq!(
        file_path_from_unified_diff_header(
            "--- a/src/old.rs\n+++ b/src/new.rs\n@@ -1 +1 @@\n-x\n+y\n"
        )
        .as_deref(),
        Some("src/new.rs")
    );
    // Deletion case: `+++ /dev/null` is skipped, falling back to the `--- a/` side.
    assert_eq!(
        file_path_from_unified_diff_header(
            "--- a/src/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        )
        .as_deref(),
        Some("src/gone.rs")
    );
    // `diff --git a/… b/…` takes the b side; trailing TAB+timestamp is stripped.
    assert_eq!(
        file_path_from_unified_diff_header(
            "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\t2026-07-30\n+++ b/foo.rs\t2026-07-30\n@@ -1 +1 @@\n-a\n+b\n"
        )
        .as_deref(),
        Some("foo.rs")
    );
    // Absolute paths without an a/ b/ prefix are preserved as-is.
    assert_eq!(
        file_path_from_unified_diff_header("+++ /abs/path.rs\n@@ -1 +1 @@\n-a\n+b\n")
            .as_deref(),
        Some("/abs/path.rs")
    );
    // Without a diff header, return None (a bare `@@` hunk still requires an explicit file_path).
    assert_eq!(
        file_path_from_unified_diff_header("@@ -1 +1 @@\n-a\n+b\n"),
        None
    );
    // `---`/`+++` header parsing must stop before the first hunk, so body context lines are not mistaken for paths.
    assert_eq!(
        file_path_from_unified_diff_header("@@ -1 +1 @@\n +++ b/not-a-header.rs\n"),
        None
    );
    // Git writes paths with spaces in JSON/C-style quotes; decode first, then strip b/.
    assert_eq!(
        file_path_from_unified_diff_header(
            "--- \"a/src/old name.rs\"\n+++ \"b/src/new name.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
        )
        .as_deref(),
        Some("src/new name.rs")
    );
    // A single-file call must not silently accept a standard multi-file unified diff without `diff --git`;
    // each file header must be followed by its own hunk; on conflict, do not infer any target.
    assert_eq!(
        file_path_from_unified_diff_header(
            "--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
        ),
        None
    );
    // In a standard multi-file Git diff, the second `diff --git` comes after the first hunk; the conflict must still be recognized,
    // not silently applying the whole patch to the first file.
    assert_eq!(
        file_path_from_unified_diff_header(
            "diff --git a/one.rs b/one.rs\n--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/two.rs b/two.rs\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
        ),
        None
    );
    assert_eq!(
        parse_unified_diff_header_target(
            "diff --git a/one.rs b/one.rs\n--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/two.rs b/two.rs\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
        )
        .paths,
        vec!["one.rs", "two.rs"]
    );
    // Paths with spaces must be read per git's quoting/escaping rules, not split into half tokens by whitespace.
    assert_eq!(
        file_path_from_unified_diff_header(
            "diff --git \"a/foo bar.rs\" \"b/foo bar.rs\"\n--- \"a/foo bar.rs\"\n+++ \"b/foo bar.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
        )
        .as_deref(),
        Some("foo bar.rs")
    );
    // Even without `+++`/`---` fallback, quoted paths should parse correctly from `diff --git`.
    assert_eq!(
        file_path_from_unified_diff_header(
            "diff --git \"a/foo bar.rs\" \"b/foo bar.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
        )
        .as_deref(),
        Some("foo bar.rs")
    );
    // When `diff --git` and `+++` point to the same file (no quotes), it is not a multi-file conflict; parse normally.
    assert_eq!(
        file_path_from_unified_diff_header(
            "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1 +1 @@\n-a\n+b\n"
        )
        .as_deref(),
        Some("foo.rs")
    );
}

#[test]
fn execute_apply_patch_reads_path_from_git_diff_header_without_file_path_arg() {
    // Reproduces the first domino of a historical loop: the model writes a textbook git unified diff
    // (with its own `--- a/` `+++ b/` headers), but does not pass file_path. Previously the tool reported missing
    // file_path, forcing the model to keep changing formats in trial and error. After the fix, it should read the path from the diff header and succeed.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("git_diff_header").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "line1\nline2\nline3\n").unwrap();
    let file_name = path.file_name().unwrap().to_string_lossy().to_string();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        // The patch header uses a relative file name (resolved relative to cwd), and neither file_path/path is passed.
        let patch = format!(
            "--- a/{file_name}\n+++ b/{file_name}\n@@ -1,3 +1,3 @@\n line1\n-line2\n+changed\n line3\n"
        );
        let args = serde_json::json!({ "patch": patch });
        execute_apply_patch(&args)
            .expect("apply_patch should read target path from git-style diff header");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "line1\nchanged\nline3\n"
    );
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_missing_path_error_mentions_diff_header_option() {
    // When there is neither file_path, nor a diff header, nor an envelope, the error message should offer three ways out, one
    // being a git-style diff header, grounding the model to the correct next step.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let err = execute_apply_patch(&serde_json::json!({
        "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n",
    }))
    .expect_err("bare hunk without file_path must error");
    assert!(err.contains("missing file_path"), "err was: {err}");
    assert!(
        err.contains("git-style") && err.contains("+++ b/"),
        "error should mention the git-style diff-header option; err was: {err}"
    );
}

#[test]
fn execute_apply_patch_applies_multi_file_diff_automatically() {
    // Multi-file unified diff (git diff output style): no longer an error; auto-split by file and apply atomically.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_diff_auto");
    let a = base.join("one.txt");
    let b = base.join("two.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "old_a\n").unwrap();
    fs::write(&b, "old_b\n").unwrap();

    let patch = "diff --git a/one.txt b/one.txt\n--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-old_a\n+new_a\ndiff --git a/two.txt b/two.txt\n--- a/two.txt\n+++ b/two.txt\n@@ -1 +1 @@\n-old_b\n+new_b\n";
    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let result = execute_apply_patch(&serde_json::json!({
            // The multi-file diff carries its own paths; a redundant file_path should be ignored
            "file_path": "one.txt",
            "patch": patch,
        }))
        .expect("multi-file unified diff should apply automatically");
        assert!(
            result.starts_with("Successfully patched 2 files:"),
            "result: {result}"
        );
    });

    assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "new_b\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_applies_multi_file_diff_without_git_headers() {
    // A multi-file diff without `diff --git` headers, only `---`/`+++` pairs: split by adjacent file-header pairs.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_diff_no_git");
    let a = base.join("a.txt");
    let b = base.join("b.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "alpha\n").unwrap();
    fs::write(&b, "beta\n").unwrap();

    let patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-alpha\n+ALPHA\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-beta\n+BETA\n";
    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let result = execute_apply_patch(&serde_json::json!({ "patch": patch }))
            .expect("multi-file diff without git headers should apply");
        assert!(
            result.starts_with("Successfully patched 2 files:"),
            "result: {result}"
        );
    });

    assert_eq!(fs::read_to_string(&a).unwrap(), "ALPHA\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "BETA\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_multi_file_diff_same_file_sections_stack() {
    // Multiple sections for the same file (same-path stacking semantics, consistent with the envelope branch).
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_diff_stack");
    let a = base.join("a.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "alpha\nbeta\ngamma\n").unwrap();

    let patch = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-alpha\n+ALPHA\ndiff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -3 +3 @@\n-gamma\n+GAMMA\n";
    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let result = execute_apply_patch(&serde_json::json!({ "patch": patch }))
            .expect("same-file sections in multi-file diff should stack");
        assert!(
            !result.starts_with("Successfully patched 2 files:"),
            "same file should be committed once: {result}"
        );
    });

    assert_eq!(fs::read_to_string(&a).unwrap(), "ALPHA\nbeta\nGAMMA\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_multi_file_diff_is_atomic_on_failure() {
    // If any file's prepare fails, nothing is committed; previously prepared files are not written either.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_diff_atomic");
    let a = base.join("a.txt");
    let b = base.join("b.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "old_a\n").unwrap();
    fs::write(&b, "current_b\n").unwrap();

    let patch = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old_a\n+new_a\ndiff --git a/b.txt b/b.txt\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-missing_b\n+new_b\n";
    let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        execute_apply_patch(&serde_json::json!({ "patch": patch }))
            .expect_err("second file mismatch should abort whole multi-file diff")
    });

    assert!(
        err.contains("failed while preparing patch for"),
        "err was: {err}"
    );
    assert_eq!(fs::read_to_string(&a).unwrap(), "old_a\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "current_b\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn unified_header_parser_ignores_header_shaped_hunk_body_lines() {
    let patch =
        "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,2 @@\n--- old marker\n+++ new marker\n";
    assert_eq!(
        file_path_from_unified_diff_header(patch).as_deref(),
        Some("notes.txt")
    );
}

#[test]
fn execute_apply_patch_rejects_dev_null_deletion_with_actionable_guidance() {
    let patch =
        "diff --git a/old.rs b/old.rs\n--- a/old.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n";
    let err = execute_apply_patch(&serde_json::json!({ "patch": patch }))
        .expect_err("unified mode must not silently turn deletion into an empty file");
    assert!(err.contains("+++ /dev/null"), "err was: {err}");
    assert!(err.contains("*** Delete File:"), "err was: {err}");
}

#[test]
fn shared_target_extractor_covers_envelope_and_quoted_git_header() {
    assert_eq!(
        apply_patch_target_paths_from_patch(
            "*** Begin Patch\n*** Update File: src/a.rs\n*** Add File: src/b.rs\n*** End Patch"
        ),
        vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")]
    );
    assert_eq!(
        apply_patch_target_paths_from_patch(
            "diff --git \"a/src/old name.rs\" \"b/src/new name.rs\"\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        vec![PathBuf::from("src/new name.rs")]
    );
}

#[test]
fn execute_apply_patch_strips_code_fence_around_envelope() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("fence_envelope").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "line1\nline2\nline3\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": format!(
                "```\n*** Begin Patch\n*** Update File: {}\n line1\n-line2\n+changed\n line3\n*** End Patch\n```",
                path.display()
            )
        });
        execute_apply_patch(&args)
            .expect("apply_patch should strip code fence around envelope");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "line1\nchanged\nline3\n"
    );
    let _ = fs::remove_dir_all(base);
}

#[test]
fn parse_unified_hunks_error_message_names_expected_prefixes() {
    // When a context line is missing its leading space, the error message should explicitly state the expected prefix.
    let err = parse_unified_hunks("@@ -1,3 +1,3 @@\nline1\n-line2\n+changed\n line3")
        .expect_err("missing leading space on context line must error");
    assert!(
        err.contains("must start with") && err.contains("context"),
        "err was: {err}"
    );
}

// ── Fix 1: strip_code_fence should tolerate trailing blank lines after the closing fence ──

#[test]
fn strip_code_fence_tolerates_trailing_blank_lines() {
    // Models often emit one or more extra blank lines after the closing fence; previously strip_code_fence treated the last
    // blank line as `last`, decided it was not a closing fence, and gave up stripping, so the whole patch stayed wrapped in the code fence
    // and went into the parser with an error.
    let fenced = "```diff\n@@ -1,1 +1,1 @@\n-line2\n+changed\n```\n";
    assert_eq!(
        strip_code_fence(fenced),
        "@@ -1,1 +1,1 @@\n-line2\n+changed"
    );
    // Multiple trailing blank lines should also be tolerated
    let fenced_multi = "```\n*** Begin Patch\n*** End Patch\n```\n\n\n";
    assert_eq!(
        strip_code_fence(fenced_multi),
        "*** Begin Patch\n*** End Patch"
    );
}

// ── Fix 2: give a clear error when the hunk header is missing ──

#[test]
fn parse_unified_hunks_missing_header_gives_clear_error() {
    // When patch content lines exist but there is no hunk header, give an error clearer than "no hunks found".
    let err = parse_unified_hunks(" line1\n-line2\n+changed\n line3")
        .expect_err("patch without hunk header must error");
    assert!(err.contains("no hunk header found"), "err was: {err}");
    assert!(err.contains("content lines"), "err was: {err}");
}

// ── Fix 3: envelope Update synthesized headers use old_start=0 ──

#[test]
fn execute_apply_patch_update_envelope_without_header_does_not_match_at_line_1() {
    // When the file start happens to match the hunk's context lines, the nominal match with old_start=1 may wrongly hit
    // the file start instead of where the model actually wants to change. old_start=0 gives the same nominal=0,
    // but with clearer semantics: no nominal position, relying on a whole-file search for a unique location.
    // Here we verify that a unique match not at the file start is located correctly.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("update_nohdr_mid").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "filler\nalpha\nbeta\ngamma\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n alpha\n-beta\n+changed\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args)
            .expect("envelope without header should locate unique match mid-file");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "filler\nalpha\nchanged\ngamma\n"
    );
    let _ = fs::remove_dir_all(base);
}

// ── Fix 4: fill in bare line prefixes when the envelope Update has no hunk header ──

#[test]
fn execute_apply_patch_update_envelope_tolerates_bare_lines() {
    // In the envelope Update format (no hunk header), the model wrote bare lines without a +/-/ prefix;
    // they should get an automatic space prefix and be treated as context lines, instead of reporting "invalid hunk line".
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("update_bare").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\nalpha\n-beta\n+changed\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args)
            .expect("envelope with bare context line should be tolerated");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "alpha\nchanged\ngamma\n"
    );
    let _ = fs::remove_dir_all(base);
}

// ── Fix 5: context lines tolerate line-number prefixes ──

#[test]
fn apply_unified_patch_tolerates_line_number_prefix_in_context() {
    // The model copied context lines with line-number prefixes from grep-like output (e.g. `   42| `);
    // the IgnoreIndent fallback mode should strip the line-number prefix and match successfully.
    // read_file's real TAB format is covered separately by apply_unified_patch_tolerates_read_file_tab_prefix.
    let original = "line1\nline2\nline3\n";
    // context line " line1" was wrongly written by the model as " 1| line1" with a line-number prefix
    let patch = "@@ -1,3 +1,3 @@\n 1| line1\n-line2\n+changed\n line3\n";
    let result = apply_unified_patch(original, patch)
        .expect("line number prefix in context should be tolerated by indent fallback");
    // context lines should keep the original file content (without the line-number prefix)
    assert_eq!(result, "line1\nchanged\nline3\n");
}

#[test]
fn apply_unified_patch_tolerates_line_number_prefix_in_remove() {
    // The remove line also carries a line-number prefix and should be tolerated the same way.
    let original = "line1\ntarget\nline3\n";
    let patch = "@@ -1,3 +1,3 @@\n line1\n-2| target\n+changed\n line3\n";
    let result = apply_unified_patch(original, patch)
        .expect("line number prefix in remove line should be tolerated");
    assert_eq!(result, "line1\nchanged\nline3\n");
}

#[test]
fn apply_unified_patch_tolerates_read_file_tab_prefix() {
    // Reproduces a real failure scenario from history: the model copied read_file output line by line into the patch's
    // context / remove lines. read_file's real render format is `{:>6}\t{}` (right-aligned line number + TAB);
    // before the fix, strip_line_number_prefix did not recognize TAB, causing repeated context mismatches.
    let original = "fn foo() {\n    let x = 1;\n    x\n}\n";
    // Construct the line the model sees using exactly the same rendering as read_file, to avoid miscounting spaces by hand.
    let rf = |n: usize, s: &str| format!("{:>6}\t{}", n, s);
    let patch = format!(
        "@@ -1,4 +1,4 @@\n {}\n-{}\n+    let x = 2;\n {}\n {}\n",
        rf(1, "fn foo() {"),
        rf(2, "    let x = 1;"),
        rf(3, "    x"),
        rf(4, "}"),
    );
    let result = apply_unified_patch(original, &patch)
        .expect("read_file TAB line-number prefix must be tolerated in context/remove lines");
    // context lines keep the original file content (including indentation); only target lines are replaced.
    assert_eq!(result, "fn foo() {\n    let x = 2;\n    x\n}\n");
}

#[test]
fn apply_unified_patch_line_number_prefix_still_detects_ambiguity() {
    // Line-number-prefix tolerance must not sacrifice safety: if there are still multiple matches after stripping the number, report ambiguity.
    let original = "dup\ndup\ndup\n";
    // The nominal position is deliberately wrong, forcing a whole-file search; after stripping the number, context+remove = ["dup","dup"] matches multiple places
    let patch = "@@ -9,2 +9,2 @@\n 1| dup\n-dup\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
}

#[test]
fn strip_line_number_prefix_does_not_strip_code_lines() {
    // Conservative single-argument fallback: only recognizes `digits+\t` and `digits+separator+space`, to avoid mis-stripping.
    use super::strip_line_number_prefix;
    // read_file's real format: right-aligned line number + TAB (the root-cause scenario previously missed)
    assert_eq!(
        strip_line_number_prefix("     3\tuse std::fs;"),
        "use std::fs;"
    );
    // After TAB, keep the code's original indentation (strip only one TAB, do not touch content indentation)
    assert_eq!(
        strip_line_number_prefix("    42\t    let x = 1;"),
        "    let x = 1;"
    );
    // grep-like formats (separator + space) should be stripped
    assert_eq!(strip_line_number_prefix("   42| hello"), "hello");
    assert_eq!(strip_line_number_prefix("42: hello"), "hello");
    // `80:80` (colon without a following space) is not a line-number prefix and must not be stripped
    assert_eq!(strip_line_number_prefix("80:80"), "80:80");
    // `3.14` (dot without a following space) must not be stripped
    assert_eq!(strip_line_number_prefix("3.14"), "3.14");
    // Pure digit lines must not be stripped (no separator)
    assert_eq!(strip_line_number_prefix("42"), "42");
    // Digits immediately followed by letters must not be stripped (`42px`)
    assert_eq!(strip_line_number_prefix("42px"), "42px");
    // Lines not starting with a digit must not be stripped
    assert_eq!(strip_line_number_prefix("hello"), "hello");
}

#[test]
fn strip_number_prefix_anchored_is_separator_agnostic() {
    // Anchor-based: keyed to the real line, stripping the line-number column regardless of separator, with almost zero false positives.
    use super::strip_number_prefix_anchored;
    let actual = "    let x = 1;";
    // read_file TAB / grep `| ` / `: ` / space / `.` / `)` all compatible
    assert_eq!(
        strip_number_prefix_anchored("  42\t    let x = 1;", actual),
        actual
    );
    assert_eq!(
        strip_number_prefix_anchored("42|     let x = 1;", actual),
        actual
    );
    assert_eq!(
        strip_number_prefix_anchored("42:     let x = 1;", actual),
        actual
    );
    assert_eq!(
        strip_number_prefix_anchored("42     let x = 1;", actual),
        actual
    );
    assert_eq!(
        strip_number_prefix_anchored("42)     let x = 1;", actual),
        actual
    );
    // After removing the column, not equal to the real line → return as-is (no false strip)
    assert_eq!(
        strip_number_prefix_anchored("42\tsomething else", actual),
        "42\tsomething else"
    );
    // Not starting with a digit → return as-is
    assert_eq!(strip_number_prefix_anchored(actual, actual), actual);
}

// ── Large-block replacement: best-effort partial matching precisely locates the inconsistent line ──

#[test]
fn apply_unified_patch_large_block_mismatch_pinpoints_wrong_line() {
    // In a large-block replacement where only one line's content is reproduced inaccurately, the error message should precisely locate which line is inconsistent
    // (expected vs actual), not just say "context mismatch".
    let original = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n";
    // The remove block has 6 lines; line4 was mistyped by the model as lineX
    let patch =
        "@@ -2,6 +2,3 @@\n-line2\n-line3\n-lineX\n-line5\n-line6\n-line7\n+new2\n+new3\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    // Should report the best match position and match count
    assert!(err.contains("Best partial match"), "err was: {err}");
    assert!(err.contains("5/6 lines matched"), "err was: {err}");
    // Should precisely point out the inconsistent line: expected lineX but actual is line4
    assert!(
        err.contains("lineX"),
        "err should mention wrong expected line: {err}"
    );
    assert!(
        err.contains("line4"),
        "err should mention actual file line: {err}"
    );
}

#[test]
fn apply_unified_patch_absent_block_falls_back_to_nominal_window() {
    // The expected block does not exist in the file at all (no line partially matches); it should echo the expected lines and
    // the actual content near the nominal position, instead of taking the partial-match branch.
    let original = "alpha\nbeta\ngamma\n";
    let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    // When the block does not exist at all, there is no "Best partial match"
    assert!(!err.contains("Best partial match"), "err was: {err}");
    // Should echo the expected lines
    assert!(err.contains("not_present"), "err was: {err}");
    // Should show the actual content near the nominal position
    assert!(err.contains("beta"), "err was: {err}");
}

#[test]
fn apply_unified_patch_partial_match_uses_middle_line_anchor() {
    // The first line of the expected block is mistyped, but the middle lines are correct. The middle-line anchors should find the best match position,
    // and report the first line's inconsistency.
    let original = "aaa\nbbb\nccc\nddd\neee\n";
    // The first line "wrong" is not in the file, but "ccc", "ddd" are
    let patch = "@@ -1,3 +1,1 @@\n-wrong\n-ccc\n ddd\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("context mismatch"), "err was: {err}");
    // Should find a partial match via "ccc" or "ddd"
    assert!(err.contains("Best partial match"), "err was: {err}");
    assert!(err.contains("2/3 lines matched"), "err was: {err}");
    // Should point out the first line mismatch: expected "wrong" but actual is "bbb"
    assert!(
        err.contains("wrong"),
        "err should mention wrong expected line: {err}"
    );
    assert!(
        err.contains("bbb"),
        "err should mention actual file line: {err}"
    );
}

// ── Canonical *** Begin Patch envelope: bare @@ / @@ heading @@ headers without line numbers ──

#[test]
fn parse_unified_hunks_accepts_bare_at_header() {
    // The canonical envelope format uses bare `@@` to separate hunks, without `-N,M +N,M` line numbers.
    // Before the fix it reported "invalid hunk header".
    let patch = "@@\n foo\n-bar\n+baz\n";
    let hunks = parse_unified_hunks(patch).expect("bare @@ header should be accepted");
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].old_start, 0);
}

#[test]
fn parse_unified_hunks_accepts_at_header_with_heading() {
    // `@@ <context title> @@` should also be accepted; the nominal line number is treated as 0 (whole-file search locates it).
    let patch = "@@ fn foo() @@\n foo\n-bar\n+baz\n";
    let hunks = parse_unified_hunks(patch).expect("@@ heading @@ header should be accepted");
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].old_start, 0);
}

#[test]
fn apply_unified_patch_applies_bare_at_header_hunk() {
    // End-to-end: a hunk with a bare @@ header should be uniquely located and applied via whole-file search.
    let original = "alpha\nbeta\ngamma\n";
    let patch = "@@\n alpha\n-beta\n+changed\n";
    let result = apply_unified_patch(original, patch).expect("bare @@ hunk should apply");
    assert_eq!(result, "alpha\nchanged\ngamma\n");
}

#[test]
fn apply_unified_patch_bare_at_header_requires_unique_match() {
    // A bare @@ header has no nominal line number; old_start=0 must not be treated as a strong anchor at line 1.
    // If the context appears multiple times in the file, the model must be asked to add more context, avoiding silently changing the first position.
    // The exact-location stage already confirms ambiguity sufficiently; it must not keep guessing or silently pick the first position.
    let original = "alpha\nbeta\ngamma\nalpha\nbeta\ngamma\n";
    let patch = "@@\n alpha\n-beta\n+changed\n";
    let err = apply_unified_patch(original, patch).unwrap_err();
    assert!(err.contains("ambiguous patch"), "err was: {err}");
    assert!(err.contains("1, 4"), "err was: {err}");
}

#[test]
fn execute_apply_patch_envelope_with_bare_at_header() {
    // Reproduces the user report: canonical *** Begin Patch envelope + bare @@ header.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("envelope_bare_at").with_extension("txt");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "path": path.to_string_lossy(),
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n@@\n alpha\n-beta\n+changed\n*** End Patch\n",
                path.display()
            )
        });
        execute_apply_patch(&args)
            .expect("envelope with bare @@ header should apply");
    });

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "alpha\nchanged\ngamma\n"
    );
    let _ = fs::remove_dir_all(base);
}

// ======================== ReplaceInLine (P2) tests ========================

fn make_envelope(op: PatchEnvelopeOp, target: &str, body: &[&str]) -> super::PatchEnvelope {
    super::PatchEnvelope {
        op,
        target_path: target.to_string(),
        body_lines: body.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn inline_replace_basic() {
    // Basic: anchor locates the line, old->new exact replacement
    let original = "fn foo() {\n    let x = 42;\n    println!(\"{}\", x);\n}\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.rs",
        &["anchor: let x = 42;", "old: 42", "new: 99"],
    );
    let result = apply_inline_replace(original, &envelope).expect("basic replace should work");
    assert_eq!(
        result,
        "fn foo() {\n    let x = 99;\n    println!(\"{}\", x);\n}\n"
    );
}

#[test]
fn inline_replace_preserves_no_trailing_newline() {
    // When the file does not end with \n, no \n is added after the replacement
    let original = "hello world";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: hello", "old: world", "new: rust"],
    );
    let result = apply_inline_replace(original, &envelope).expect("should work");
    assert_eq!(result, "hello rust");
}

#[test]
fn inline_replace_preserves_trailing_newline() {
    let original = "hello world\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: hello", "old: world", "new: rust"],
    );
    let result = apply_inline_replace(original, &envelope).expect("should work");
    assert_eq!(result, "hello rust\n");
}

#[test]
fn inline_replace_anchor_tolerates_confusable() {
    // The anchor uses em-dash (—, U+2014), while the file has ASCII hyphen (-).
    // Anchor normalized matching should tolerate it, but old must still match exactly.
    let original = "the quick—brown fox\njumps over\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: the quick—brown fox", "old: fox", "new: dog"],
    );
    let result =
        apply_inline_replace(original, &envelope).expect("confusable anchor should match");
    assert_eq!(result, "the quick—brown dog\njumps over\n");
}

#[test]
fn inline_replace_old_tolerates_confusable() {
    // old has em-dash, the file has ASCII hyphen: after exact match fails,
    // the tolerant fallback (confusable 1:1 normalization) should locate and replace.
    let original = "the quick-brown fox\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: quick", "old: quick—brown", "new: slow-brown"],
    );
    let result =
        apply_inline_replace(original, &envelope).expect("confusable old should match");
    // The output is built from new, preserving the file's original content; only the matched range is replaced
    assert_eq!(result, "the slow-brown fox\n");
}

#[test]
fn inline_replace_old_tolerates_whitespace() {
    // old with leading/trailing whitespace (model indentation not reproduced exactly) -> tolerant match ignores leading/trailing whitespace
    let original = "let x = 42;\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.rs",
        &["anchor: let x", "old:   x = 42  ", "new: x = 99"],
    );
    let result =
        apply_inline_replace(original, &envelope).expect("whitespace-trimmed old should match");
    assert_eq!(result, "let x = 99;\n");
}

#[test]
fn inline_replace_old_not_found_mentions_line_prefix_hint() {
    // old copied from read_file also brought the line-number prefix -> must not match silently; error with a hint.
    // Tolerant matching does not strip prefixes (that would pollute file content); here we verify the error message has guidance.
    let original = "let x = 42;\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.rs",
        &["anchor: let x", "old:     1\tlet x = 42;", "new: let x = 99;"],
    );
    let err = apply_inline_replace(original, &envelope)
        .expect_err("old with line-number prefix should fail");
    assert!(
        err.contains("line-number prefix"),
        "error should hint at line-number prefix: {err}"
    );
}

#[test]
fn inline_replace_old_confusable_ambiguous() {
    // Exact match is zero (em-dash/en-dash are not hyphen); after normalization old appears in the line
    // multiple times -> error (instead of guessing one)
    let original = "a—b a–b\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: a—b", "old: a-b", "new: c-d"],
    );
    let err = apply_inline_replace(original, &envelope)
        .expect_err("old matching 2 positions after normalization should fail");
    assert!(
        err.contains("matches 2 positions"),
        "error should mention ambiguity after normalization: {err}"
    );
}

#[test]
fn inline_replace_anchor_not_unique() {
    // Anchor matches multiple lines -> error
    let original = "duplicate line\nduplicate line\nunique here\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: duplicate line", "old: duplicate", "new: unique"],
    );
    let err =
        apply_inline_replace(original, &envelope).expect_err("non-unique anchor should fail");
    assert!(
        err.contains("matched 2 lines"),
        "error should mention 2 matched lines: {err}"
    );
}

#[test]
fn inline_replace_anchor_not_found() {
    let original = "hello world\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: nonexistent", "old: world", "new: rust"],
    );
    let err =
        apply_inline_replace(original, &envelope).expect_err("missing anchor should fail");
    assert!(err.contains("anchor not found"), "error: {err}");
}

#[test]
fn inline_replace_old_not_unique_in_line() {
    // old appears multiple times within the line -> error
    let original = "foo bar foo baz\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: foo bar", "old: foo", "new: qux"],
    );
    let err =
        apply_inline_replace(original, &envelope).expect_err("non-unique old should fail");
    assert!(
        err.contains("appears 2 times"),
        "error should mention 2 occurrences: {err}"
    );
}

#[test]
fn inline_replace_old_equals_new() {
    let original = "hello world\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: hello", "old: world", "new: world"],
    );
    let err = apply_inline_replace(original, &envelope).expect_err("old==new should fail");
    assert!(err.contains("identical"), "error: {err}");
}

#[test]
fn inline_replace_missing_field() {
    let original = "hello world\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.txt",
        &["anchor: hello", "old: world"],
    );
    let err = apply_inline_replace(original, &envelope).expect_err("missing new should fail");
    assert!(err.contains("missing `new:`"), "error: {err}");
}

#[test]
fn inline_replace_unicode_content() {
    // Replacing multi-byte UTF-8 content, verifying byte-index slicing safety
    let original = "let greeting = \"你好世界\";\n";
    let envelope = make_envelope(
        PatchEnvelopeOp::ReplaceInLine,
        "test.rs",
        &["anchor: greeting", "old: 你好", "new: 再见"],
    );
    let result =
        apply_inline_replace(original, &envelope).expect("unicode replace should work");
    assert_eq!(result, "let greeting = \"再见世界\";\n");
}

#[test]
fn inline_replace_parse_envelope() {
    // Verifies parse_patch_envelope recognizes the *** Replace in line: header
    let patch = "*** Begin Patch\n\
        *** Replace in line: src/main.rs\n\
        anchor: fn main()\n\
        old: println!\n\
        new: eprintln!\n\
        *** End Patch\n";
    let envelope = parse_patch_envelope(patch)
        .expect("should parse")
        .expect("should be Some");
    assert_eq!(envelope.op, PatchEnvelopeOp::ReplaceInLine);
    assert_eq!(envelope.target_path, "src/main.rs");
    assert_eq!(envelope.body_lines.len(), 3);
}

#[test]
fn inline_replace_via_execute_apply_patch() {
    // End-to-end: calls through execute_apply_patch, verifying the full path (including sandbox)
    let _guard = ENV_LOCK.lock();
    let path = make_temp_path("inline_e2e");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "the answer is 42\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": format!(
                "*** Begin Patch\n*** Replace in line: {}\nanchor: the answer\nold: 42\nnew: 99\n*** End Patch\n",
                path.to_string_lossy()
            ),
            "path": path.to_string_lossy(),
        });
        execute_apply_patch(&args).expect("e2e should succeed");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "the answer is 99\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn parse_patch_envelopes_accepts_multiple_sections() {
    let patch = "*** Begin Patch\n\
        *** Update File: src/a.rs\n\
        @@\n\
        -old_a\n\
        +new_a\n\
        \n\
        *** Add File: src/b.rs\n\
        +hello\n\
        *** End Patch\n";
    let envelopes = parse_patch_envelopes(patch)
        .expect("should parse")
        .expect("should be Some");
    assert_eq!(envelopes.len(), 2);
    assert_eq!(envelopes[0].target_path, "src/a.rs");
    assert_eq!(envelopes[1].target_path, "src/b.rs");
}

#[test]
fn execute_apply_patch_supports_multi_file_begin_patch_atomically() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_batch");
    let a = base.join("a.txt");
    let b = base.join("b.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "old_a\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            // Some models serialize unused optional string parameters as empty strings; treat them as not provided.
            "file_path": "",
            "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Add File: b.txt\n+hello\n+world\n*** End Patch\n"
        });
        let result = execute_apply_patch(&args).expect("multi-file Begin Patch should succeed");
        assert!(result.starts_with("Successfully patched 2 files:"), "result: {result}");
    });

    assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "hello\nworld");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_multi_file_ignores_redundant_file_path() {
    // Multi-file envelope + redundant file_path: models often still pass file_path in a multi-file envelope
    // (pointing at one of the files). file_path should be silently ignored, using each section's own path in the envelope.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_redundant_path");
    let a = base.join("a.txt");
    let b = base.join("b.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "old_a\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            // Redundant file_path should be silently ignored
            "file_path": a.to_string_lossy(),
            "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Add File: b.txt\n+hello\n+world\n*** End Patch\n"
        });
        let result = execute_apply_patch(&args).expect("multi-file Begin Patch with redundant file_path should succeed");
        assert!(result.starts_with("Successfully patched 2 files:"), "result: {result}");
    });

    assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "hello\nworld");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_multi_file_batch_is_atomic_on_failure() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("multi_file_atomic");
    let a = base.join("a.txt");
    let b = base.join("b.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&a, "old_a\n").unwrap();
    fs::write(&b, "current_b\n").unwrap();

    let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Update File: b.txt\n@@\n-missing_b\n+new_b\n*** End Patch\n"
        });
        execute_apply_patch(&args).expect_err("second file mismatch should abort whole batch")
    });

    assert!(
        err.contains("failed while preparing patch for"),
        "err was: {err}"
    );
    assert_eq!(fs::read_to_string(&a).unwrap(), "old_a\n");
    assert_eq!(fs::read_to_string(&b).unwrap(), "current_b\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_applies_repeated_same_file_sections_in_order() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("same_file_sections");
    let path = base.join("a.txt");
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-alpha\n+ALPHA\n*** Update File: a.txt\n@@\n-gamma\n+GAMMA\n*** End Patch\n"
        });
        let result = execute_apply_patch(&args)
            .expect("repeated same-file sections should apply sequentially");
        assert!(
            result.starts_with("Successfully patched "),
            "result: {result}"
        );
        assert!(
            !result.starts_with("Successfully patched 2 files:"),
            "same file should be committed once: {result}"
        );
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "ALPHA\nbeta\nGAMMA\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_can_update_file_created_earlier_in_same_patch() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let base = make_temp_path("same_file_add_update");
    let path = base.join("new.txt");
    fs::create_dir_all(&base).unwrap();

    crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        let args = serde_json::json!({
            "patch": "*** Begin Patch\n*** Add File: new.txt\n+alpha\n+beta\n*** Update File: new.txt\n@@\n-beta\n+changed\n*** End Patch\n"
        });
        execute_apply_patch(&args)
            .expect("Update File should see content added by an earlier same-file section");
    });

    assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn execute_apply_patch_legacy_dry_run_remains_non_mutating() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("dry_run");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "before\n").unwrap();

    let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        execute_apply_patch(&serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": "@@\n-before\n+after\n",
            "dry_run": true,
        }))
        .expect("legacy dry run should remain safe for old calls")
    });

    assert!(result.starts_with("Dry run succeeded; no files changed:"));
    assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn apply_unified_patch_ignores_context_only_hunks_when_ordering_changes() {
    let original = "first\nmiddle\nlast\n";
    let patch = "@@ -3,1 +3,1 @@\n last\n@@ -1,1 +1,1 @@\n-first\n+FIRST\n";

    let actual = apply_unified_patch(original, patch)
        .expect("context-only hunks must not advance the changed-hunk cursor");

    assert_eq!(actual, "FIRST\nmiddle\nlast\n");
}

#[test]
fn execute_apply_patch_rejects_unified_noop() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let path = make_temp_path("unified_noop");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "before\n").unwrap();

    let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
        execute_apply_patch(&serde_json::json!({
            "file_path": path.to_string_lossy(),
            "patch": "@@ -1,1 +1,1 @@\n before\n",
        }))
        .expect_err("a context-only unified diff must not report success")
    });

    assert!(err.contains("[NO_CHANGES]"), "err was: {err}");
    assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
    let _ = fs::remove_dir_all(base);
}

#[test]
fn prepared_patch_rejects_external_change_before_commit() {
    let path = make_temp_path("stale_patch");
    let base = path.parent().unwrap().to_path_buf();
    fs::create_dir_all(&base).unwrap();
    fs::write(&path, "before\n").unwrap();
    let store = super::FileStore::new(path.clone());
    let envelope = make_envelope(
        PatchEnvelopeOp::Update,
        &path.to_string_lossy(),
        &["@@", "-before", "+after"],
    );
    let prepared = super::prepare_patch_write(&path, &store, &envelope)
        .expect("matching patch should prepare");
    fs::write(&path, "changed_elsewhere\n").unwrap();

    let err = super::verify_patch_write_is_current(&prepared)
        .expect_err("a changed target must not be overwritten");
    assert!(err.contains("[FILE_CHANGED]"), "err: {err}");
    assert_eq!(fs::read_to_string(&path).unwrap(), "changed_elsewhere\n");
    let _ = fs::remove_dir_all(base);
}
