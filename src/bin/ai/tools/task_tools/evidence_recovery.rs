//! Full evidence recovery instructions shared by projection and task tools.

pub(crate) enum EvidenceRecoveryFormat {
    Compact,
    ToolCall,
}

const OMIT_PROJECTION: &str = "omitting fields/max_chars/cursor";

/// Generic instructions use the exact ID from the retained ledger or original request.
pub(crate) fn full_delivered_evidence_hint(format: EvidenceRecoveryFormat) -> String {
    let arguments = "source=\"delivered\", view=\"raw\", exact task_id";
    let call = match format {
        EvidenceRecoveryFormat::Compact => format!("task_evidence_read({arguments})"),
        EvidenceRecoveryFormat::ToolCall => format!("`task_evidence_read` ({arguments})"),
    };
    format!("{call}, {OMIT_PROJECTION}")
}

/// Preserve opaque IDs verbatim and reconstruct a complete read, never a prior projection.
pub(crate) fn task_evidence_recovery_hint(task_id: &str, source: &str) -> String {
    let arguments = serde_json::json!({"task_id": task_id, "source": source, "view": "raw"});
    format!("task_evidence_read({arguments}); use this exact task_id, {OMIT_PROJECTION}")
}

#[cfg(test)]
pub(crate) fn assert_full_evidence_recovery_hints(text: &str) {
    let mut checked = 0;
    for line in text
        .lines()
        .filter(|line| line.contains("task_evidence_read"))
    {
        if line.starts_with("Output preserved for tool ") {
            continue;
        }
        let call = line.split_once("task_evidence_read").unwrap().1;
        let call = call.trim_start_matches('`').trim_start();
        let Some(arguments) = call.strip_prefix('(') else {
            continue;
        };
        if arguments.starts_with('{') {
            let args = serde_json::Deserializer::from_str(arguments)
                .into_iter::<serde_json::Value>()
                .next()
                .unwrap()
                .unwrap();
            assert_eq!(args["view"], "raw", "{line}");
            assert!(args["task_id"].is_string(), "{line}");
            assert!(line.contains("exact task_id"), "{line}");
            for key in ["fields", "max_chars", "cursor"] {
                assert!(args.get(key).is_none(), "{line}");
            }
        } else {
            let arguments = arguments.split_once(')').unwrap().0;
            assert!(arguments.contains("source=\"delivered\""), "{line}");
            assert!(arguments.contains("view=\"raw\""), "{line}");
            assert!(arguments.contains("exact task_id"), "{line}");
            for key in ["fields", "max_chars", "cursor"] {
                assert!(!arguments.contains(&format!("{key}=")), "{line}");
            }
        }
        assert!(line.contains(OMIT_PROJECTION), "{line}");
        checked += 1;
    }
    assert!(checked > 0, "no complete recovery call found: {text}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_evidence_recovery_generator_inventory_uses_shared_contract() {
        fn check_static_instruction(text: &str) {
            let context = text.to_ascii_lowercase();
            if (text.contains("task_evidence_read(") || text.contains("`task_evidence_read` ("))
                && ["full", "recover", "fold", "truncat"]
                    .iter()
                    .any(|word| context.contains(word))
            {
                assert_full_evidence_recovery_hints(text);
            }
        }
        // Each producer must delegate its complete-read semantics. The inventory
        // also covers static mirrors, which cannot call Rust at document load time.
        let producers = [
            (
                include_str!("../../history/compress/tool_overflow.rs"),
                "task_evidence_review_lines",
                "task_evidence_recovery_hint",
            ),
            (
                include_str!("../../history/compress/tool_overflow.rs"),
                "build_tool_overflow_recall_lines",
                "task_evidence_recovery_hint",
            ),
            (
                include_str!("outstanding.rs"),
                "render_agent_team_outstanding_task_anchor",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../driver/turn_runtime/iteration.rs"),
                "refresh_outstanding_task_anchor",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../driver/turn_runtime/prepare.rs"),
                "verification_reminder_text",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../driver/turn_runtime/finalize.rs"),
                "format_full_subagent_result_for_parent",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../driver/turn_runtime/finalize.rs"),
                "format_agent_team_subagent_result_for_parent",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../history/task_evidence.rs"),
                "render_task_evidence_ledger",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("../../history/task_evidence.rs"),
                "render_agent_team_task_evidence_ledger",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("task_evidence.rs"),
                "render_summary",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("task_evidence.rs"),
                "render",
                "task_evidence_recovery_hint",
            ),
            (
                include_str!("verification.rs"),
                "inline_assessment",
                "full_delivered_evidence_hint",
            ),
            (
                include_str!("wait.rs"),
                "wait_verification_assessment",
                "full_delivered_evidence_hint",
            ),
        ];
        for (source, symbol, constructor) in producers {
            let body = source
                .split_once(&format!("fn {symbol}("))
                .unwrap_or_else(|| panic!("missing producer {symbol}"))
                .1
                .split_once("\n}")
                .unwrap()
                .0;
            assert!(
                body.contains(constructor),
                "{symbol} bypasses {constructor}"
            );
        }
        for format in [
            EvidenceRecoveryFormat::Compact,
            EvidenceRecoveryFormat::ToolCall,
        ] {
            assert_full_evidence_recovery_hints(&full_delivered_evidence_hint(format));
        }
        for source in ["delivered", "progress"] {
            let task_id = "Original-ID ) \"界\\worker";
            let hint = task_evidence_recovery_hint(task_id, source);
            assert_full_evidence_recovery_hints(&hint);
            let args = serde_json::Deserializer::from_str(hint.split_once('(').unwrap().1)
                .into_iter::<serde_json::Value>()
                .next()
                .unwrap()
                .unwrap();
            assert_eq!(args["task_id"], task_id);
            assert_eq!(args["source"], source);
            assert_eq!(args.as_object().unwrap().len(), 3);
        }
        let canonical = full_delivered_evidence_hint(EvidenceRecoveryFormat::Compact);
        let skill = include_str!("../../builtin_skills/agent-team.skill");
        assert!(skill.contains(&canonical));
        assert_full_evidence_recovery_hints(skill);
        for description in [
            include_str!("../../tool_descriptions/task_evidence_read.json"),
            include_str!("../../tool_descriptions/task_wait.json"),
        ] {
            let metadata: serde_json::Value = serde_json::from_str(description).unwrap();
            let description = metadata["description"].as_str().unwrap();
            assert!(description.contains(&canonical), "{description}");
            assert_full_evidence_recovery_hints(description);
        }
        let metadata: serde_json::Value = serde_json::from_str(include_str!(
            "../../tool_descriptions/task_evidence_read.json"
        ))
        .unwrap();
        assert_full_evidence_recovery_hints(
            metadata["parameters"]["properties"]["cursor"]["description"]
                .as_str()
                .unwrap(),
        );

        // Reject new handwritten prompt generators, including producers added
        // outside this inventory. Tool execution is not a prompt.
        let inline_call =
            regex::Regex::new(r#""[^"\n]*(?:task_evidence_read\(|`task_evidence_read` \()"#)
                .unwrap();
        let test_module =
            regex::Regex::new(r"(?m)^#\[cfg\(test\)\]\n(?:#\[[^\n]+\]\n)*mod ").unwrap();
        let mut pending = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/ai")];
        while let Some(path) = pending.pop() {
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                pending.extend(
                    std::fs::read_dir(&path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path()),
                );
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let name = path.file_name().unwrap().to_str().unwrap();
                if name.ends_with("_tests.rs")
                    || name == "tests.rs"
                    || name == "evidence_recovery.rs"
                {
                    continue;
                }
                let source = std::fs::read_to_string(&path).unwrap();
                let production = test_module
                    .find(&source)
                    .map_or(source.as_str(), |start| &source[..start.start()]);
                for (line_number, line) in production.lines().enumerate() {
                    if line.trim_start().starts_with("//") {
                        continue;
                    }
                    assert!(
                        !inline_call.is_match(line),
                        "handwritten evidence call at {}:{}: {line}",
                        path.display(),
                        line_number + 1
                    );
                }
            } else if path.extension().is_some_and(|extension| {
                extension == "json" || extension == "md" || extension == "skill"
            }) {
                let text = std::fs::read_to_string(&path).unwrap();
                if !text.contains("task_evidence_read") {
                    continue;
                }
                if path
                    .extension()
                    .is_some_and(|extension| extension == "json")
                {
                    let document: serde_json::Value = serde_json::from_str(&text).unwrap();
                    let mut values = vec![&document];
                    while let Some(value) = values.pop() {
                        match value {
                            serde_json::Value::String(text) => check_static_instruction(text),
                            serde_json::Value::Array(items) => values.extend(items),
                            serde_json::Value::Object(fields) => values.extend(fields.values()),
                            _ => {}
                        }
                    }
                } else {
                    check_static_instruction(&text);
                }
            }
        }
    }
}
