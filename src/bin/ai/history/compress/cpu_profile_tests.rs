//! Ignored, release-only CPU profile for one long-history compression.
//! The fixture and speculative fold copies can use several hundred MiB of RAM.
//! Run with cargo test --release --bin a cpu_profile_long_history -- --ignored --nocapture.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use super::*;
use crate::ai::types::{FunctionCall, ToolCall};

struct StageRecord {
    name: &'static str,
    calls: usize,
    elapsed: Duration,
}

thread_local! {
    static PROFILE: RefCell<Option<Vec<StageRecord>>> = const { RefCell::new(None) };
}

pub(super) struct StageTimer {
    name: &'static str,
    started: Option<Instant>,
}

impl StageTimer {
    pub(super) fn start(name: &'static str) -> Self {
        let started = PROFILE.with(|profile| profile.borrow().is_some().then(Instant::now));
        Self { name, started }
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        let Some(started) = self.started else {
            return;
        };
        let elapsed = started.elapsed();
        PROFILE.with(|profile| {
            let mut profile = profile.borrow_mut();
            let Some(records) = profile.as_mut() else {
                return;
            };
            if let Some(record) = records.iter_mut().find(|record| record.name == self.name) {
                record.calls += 1;
                record.elapsed += elapsed;
            } else {
                records.push(StageRecord {
                    name: self.name,
                    calls: 1,
                    elapsed,
                });
            }
        });
    }
}

struct ProfileAssets(PathBuf);

impl Drop for ProfileAssets {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn message(role: &str, content: String) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn long_history_fixture() -> Vec<Message> {
    let mut messages = Vec::with_capacity(13_000);
    messages.push(message(
        "system",
        "Stable project instructions.".to_string(),
    ));
    for index in 0..4 {
        messages.push(message(
            ROLE_INTERNAL_NOTE,
            format!(
                "长期记忆摘要（压缩保留）:\nEarlier task {} and its evidence.",
                index % 2
            ),
        ));
    }
    for _ in 0..2 {
        messages.push(message(
            ROLE_INTERNAL_NOTE,
            format!("{ARCHIVE_NOTE_PREFIX}\n{ARCHIVE_FILE_PATH_PREFIX}/fixture/earlier.md"),
        ));
    }

    let mut group = 0usize;
    for turn in 0..256 {
        let user_text = if turn % 16 == 0 || turn == 255 {
            format!(
                "User task {turn}: preserve this original instruction. {}",
                "context 支付 🎯 ".repeat(512)
            )
        } else {
            format!(
                "User task {turn}: inspect the next revision and explain the result. {}",
                "Keep the evidence. ".repeat(12)
            )
        };
        messages.push(message("user", user_text));
        let rounds = 24 + usize::from(turn < 224);
        for _ in 0..rounds {
            let id = format!("cpu-call-{group}");
            let ordinary = group % 4 == 0;
            let tool_name = if ordinary {
                "execute_command"
            } else {
                "read_file"
            };
            let arguments = if ordinary {
                serde_json::json!({"command": format!("inspect revision {group}")}).to_string()
            } else {
                let path = if group % 16 == 1 {
                    format!("/fixture/auxiliary-{}.rs", group % 3)
                } else {
                    "/fixture/shared.rs".to_string()
                };
                serde_json::json!({"file_path": path, "offset": 1, "limit": 8192}).to_string()
            };
            let mut assistant = message(
                "assistant",
                format!("Inspecting revision {group}; keep the changed evidence."),
            );
            assistant.tool_calls = Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: tool_name.to_string(),
                    arguments,
                },
            }]);
            messages.push(assistant);
            let line_count = if group % 257 == 0 { 4096 } else { 192 };
            let mut output = String::new();
            for line in 1..=line_count {
                if ordinary {
                    writeln!(output, "warning revision_{group} row_{line} completed /fixture/result/{group}/{line}.rs: inspect the changed value 支付").unwrap();
                } else {
                    // Distinct revisions overlap in line numbers but differ in bytes.
                    // This is the expensive no-dedup case, not repeated identical reads.
                    writeln!(output, "{line:>6}\tlet revision_{group}_row_{line} = \"evidence_{group}_{line} 支付\";").unwrap();
                }
            }
            output.pop();
            let mut result = message("tool", output);
            result.tool_call_id = Some(id);
            messages.push(result);
            group += 1;
        }
    }
    messages.push(message(
        "assistant",
        "Latest task result and retained evidence.".to_string(),
    ));
    assert_eq!(messages.len(), 13_000);
    assert_eq!(user_input_turn_count(&messages), 256);
    messages
}

#[test]
#[ignore = "release-only long-history CPU profile; uses several hundred MiB"]
fn cpu_profile_long_history() {
    assert!(
        !cfg!(debug_assertions),
        "run this ignored test with --release"
    );
    let assets = ProfileAssets(
        std::env::temp_dir().join(format!("ai-compress-cpu-{}", uuid::Uuid::new_v4())),
    );
    std::fs::create_dir_all(&assets.0).unwrap();
    let fixture = long_history_fixture();
    let chars = messages_total_chars(&fixture);
    assert!(chars > 200_000);
    println!(
        "CPU_PROFILE fixture messages={} user_turns=256 tool_names=read_file,execute_command billable_chars={chars} max_chars=200000",
        fixture.len()
    );
    let protected_user = fixture
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .unwrap()
        .clone();

    for run in 1..=2 {
        let input = fixture.clone();
        PROFILE.with(|profile| *profile.borrow_mut() = Some(Vec::new()));
        let started = Instant::now();
        let input = profile_compression_stage!(
            "coalesce_summary_notes",
            coalesce_accumulated_summary_notes(input)
        );
        let outcome = shrink_messages_to_fit_with_summary_outcome(
            input,
            200_000,
            4_000,
            Some(&assets.0),
            None,
            &rustc_hash::FxHashSet::default(),
        );
        let total = started.elapsed();
        let records = PROFILE.with(|profile| profile.borrow_mut().take().unwrap());
        println!(
            "CPU_PROFILE run={run} archive_state={} total_ms={:.3} output_messages={} output_chars={} status={:?}",
            if run == 1 { "cold" } else { "warm" },
            total.as_secs_f64() * 1000.0,
            outcome.messages.len(),
            outcome.after_chars,
            outcome.status
        );
        for record in records {
            println!(
                "CPU_PROFILE stage={:<30} calls={:<4} elapsed_ms={:.3}",
                record.name,
                record.calls,
                record.elapsed.as_secs_f64() * 1000.0
            );
        }
        assert_eq!(outcome.status, ContextCompressionStatus::Complete);
        assert!(outcome.messages.contains(&protected_user));
        // The directory is the only process-dependent input to recovery pointers.
        // Report a normalized digest for before/after comparisons; differential
        // tests compare the actual serialized bytes of the optimized passes.
        let output = serde_json::to_string(&outcome.messages)
            .unwrap()
            .replace(assets.0.to_str().unwrap(), "PROFILE_ASSETS");
        println!(
            "CPU_PROFILE run={run} normalized_output_sha256={}",
            content_sha256_hex(output.as_bytes())
        );
    }
}
