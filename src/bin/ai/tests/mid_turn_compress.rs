//! Mid-turn compression tests (history::mid_turn_compress).

use rust_tools::cw::SkipSet;
use serde_json::Value;

use super::super::{
    history::{Message, mid_turn_compress},
    types::{FunctionCall, ToolCall},
};

#[test]
fn mid_turn_compress_preserves_latest_user_message() {
    let latest_user = "请继续修复 request.rs 的流式中断问题";
    let filler = "x".repeat(12000);
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("早期需求：实现 streaming".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(latest_user.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到，我继续处理".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, before, after) = mid_turn_compress(messages, 4000, None, None);
    assert!(after <= before, "compression should not expand payload");

    let has_latest_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(latest_user));
    assert!(
        has_latest_user,
        "mid-turn compression must preserve the latest user message"
    );
}

#[test]
fn mid_turn_compress_spills_non_compressible_outputs_when_overflow_dir_present() {
    // Regression test: once mid-turn compression receives an overflow_dir, large outputs of "incompressible"
    // tools like read_file must spill with zero compression into the session file + leave a preview stub, actually reducing the character count.
    // Historical bug: mid-turn passed None overflow_dir, so such outputs could be neither pruned nor spilled;
    // they just piled up in the context verbatim, shaving only a few K per turn (the user-reported "compression does nothing").
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-midturn-overflow-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    // 10 read_file groups, each result over 8000 chars and all distinct, so byte-identical
    // dedup never hits first. The first 6 groups sit before the real user message, simulating "last turn's
    // uncompressed read_file outputs" — outside the current-turn protected window and outside the recent
    // keep_recent groups, they should spill with zero compression.
    for i in 0..6usize {
        let id = format!("call_{i}");
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: format!(
                        r#"{{"filePath":"src/lib.rs","startLine":{},"endLine":{}}}"#,
                        i + 1,
                        i + 40
                    ),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String(format!("chunk-{i:02}\n{}", "y".repeat(8000))),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    // Real user turn boundary: read_file results after it belong to the current turn, protected by
    // precision and kept in full as the most recent tool group, verifying "the current turn is not collateral damage".
    messages.push(Message {
        role: "user".to_string(),
        content: Value::String("continue".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    for i in 6..10usize {
        let id = format!("call_{i}");
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: format!(
                        r#"{{"filePath":"src/lib.rs","startLine":{},"endLine":{}}}"#,
                        i + 1,
                        i + 40
                    ),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String(format!("chunk-{i:02}\n{}", "y".repeat(8000))),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    let before = messages
        .iter()
        .map(|m| m.content.as_str().map(|s| s.chars().count()).unwrap_or(0))
        .sum::<usize>();
    let (compressed, reported_before, reported_after) =
        mid_turn_compress(messages, 36_000, Some(overflow_dir.as_path()), None);

    assert!(reported_before >= before);
    assert!(
        reported_after < reported_before,
        "mid-turn compression with overflow_dir must shrink payload \
         (before={reported_before}, after={reported_after})"
    );
    // Contract: after an incompressible read_file result is compressed, a readable file_path recall anchor must remain,
    // and the session archive file it points to must really exist with the full text saved zero-compression. The recall anchor has two legal
    // shapes depending on the compression path hit (both carry a `file_path:` pointing at the archive; assert the recall contract, not the
    // literal wording of one path):
    //   - spill stub：`Output preserved for tool `read_file` ... - file_path: ...`
    //   - fold note ：`compressed_tool_round: ... - read_file => - archive_file_path: ...`
    //     (secondary folding deliberately renames the internal overflow path to `archive_file_path` so it cannot pose as
    //     the plain `file_path` primary lead; the `- original_file_path:` in the same note points at the source
    //     file, not the archive, and must not be treated as the archive pointer.)
    //
    // Note: do not use substring splitting like `split("file_path: ").nth(1)` — in the fold note
    // `- original_file_path:` comes before `- archive_file_path:`, so substring splitting would mistake the
    // source file path for the archive path (historical bug: read src/lib.rs at only 1305 bytes).
    let file_path = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            let candidates: Vec<&str> = text
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    line.strip_prefix("- archive_file_path: ")
                        .or_else(|| line.strip_prefix("- file_path: "))
                        .map(str::trim)
                })
                .collect();
            // Prefer the per-tool archive carrying complete zero-compression content; if every candidate is under the threshold
            // (in the stub hit directly by the spill scenario, file_path already points at the full file), fall back to the first candidate.
            let full = candidates
                .iter()
                .copied()
                .find(|c| std::fs::read(c).map(|b| b.len() >= 8000).unwrap_or(false));
            let hit = full.or_else(|| candidates.into_iter().next())?;
            Some(hit)
        })
        .unwrap_or_else(|| {
            panic!("expected read_file recall archive path after mid-turn compression: {compressed:#?}")
        });
    let archived = std::fs::read(file_path).unwrap_or_else(|e| {
        panic!("overflow file referenced by recall anchor should exist: {file_path}: {e}")
    });
    assert!(
        archived.len() >= 8000,
        "archived read_file output must preserve full content (zero-compression), got {} bytes",
        archived.len()
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn large_image_does_not_evict_tool_history_from_budget() {
    // Regression test: one large base64 image must not squeeze the agent's tool results (working memory)
    // out of the context. Historical bug: value_len_chars billed by base64 text length,
    // so a ~900K-char image ballooned messages_total_chars and the compression pipeline
    // deleted tool results every turn -> agent amnesia -> repeating the same exploration over and over.
    let huge_base64 = "A".repeat(900_000);
    let image_content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": { "url": format!("data:image/png;base64,{huge_base64}") }
        }
    ]);

    let messages = vec![
        Message {
            role: "user".to_string(),
            content: image_content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("我先探索代码结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("read_file 结果：found memo.rs at src/bin/re/memo".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("继续实现".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    // soft_threshold 36K: if the image were still billed by base64 length (900K), it would be judged over budget and
    // trigger compression, deleting tool results. After the fix the image counts only ~1K, keeping the total well under the threshold.
    let (compressed, before, after) = mid_turn_compress(messages, 36_000, None, None);
    assert!(
        before <= 36_000,
        "image must not dominate the char budget (got {before})"
    );
    assert_eq!(
        before, after,
        "no compression should trigger for image-only payload"
    );

    let kept_tool_result = compressed.iter().any(|m| {
        m.role == "tool"
            && m.content.as_str() == Some("read_file 结果：found memo.rs at src/bin/re/memo")
    });
    assert!(
        kept_tool_result,
        "tool result (agent working memory) must survive; otherwise the agent re-explores"
    );

    let image_intact = compressed.iter().any(|m| {
        m.content
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("image_url"))
            .and_then(|iu| iu.get("url"))
            .and_then(|u| u.as_str())
            .map(|u| u.len() > 100_000)
            .unwrap_or(false)
    });
    assert!(
        image_intact,
        "image content itself must remain zero-compressed"
    );
}

#[test]
fn mid_turn_compress_preserves_recent_two_user_messages() {
    let previous_user = "先定位 streaming 中断的根因";
    let latest_user = "再补上修复并验证回归";
    let filler = "x".repeat(14_000);
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("更早需求：梳理模块结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(previous_user.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(latest_user.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到，我会按顺序处理".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, before, after) = mid_turn_compress(messages, 4_000, None, None);
    assert!(after <= before, "compression should not expand payload");

    let has_previous_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(previous_user));
    let has_latest_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(latest_user));

    assert!(
        has_previous_user,
        "mid-turn compression must preserve the previous user turn"
    );
    assert!(
        has_latest_user,
        "mid-turn compression must preserve the latest user turn"
    );
}

#[test]
fn mid_turn_compress_prefers_three_recent_user_turns_when_context_is_small_enough() {
    let user2 = "第二阶段：定位流式卡住点";
    let user3 = "第三阶段：修复压缩策略";
    let user4 = "第四阶段：补测试并复盘";
    let filler = "x".repeat(8_000);

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("第一阶段：读取代码结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user2.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user3.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user4.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到，按 2->3->4 顺序执行".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, _before, _after) = mid_turn_compress(messages, 4_000, None, None);

    let has_user2 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user2));
    let has_user3 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user3));
    let has_user4 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user4));

    assert!(
        has_user2,
        "should preserve previous-2 user turn when context is moderate"
    );
    assert!(
        has_user3,
        "should preserve previous-1 user turn when context is moderate"
    );
    assert!(
        has_user4,
        "should preserve latest user turn when context is moderate"
    );
}

#[test]
fn mid_turn_compress_keeps_tool_pairs_consistent() {
    let huge = "x".repeat(18_000);
    let tool_call = ToolCall {
        id: "call_pair_1".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{\"file_path\":\"src/main.rs\"}".to_string(),
        },
    };

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("先分析历史错误".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![tool_call.clone()]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String(huge.clone()),
            tool_calls: None,
            tool_call_id: Some(tool_call.id.clone()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("我继续排查".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("请继续修复并验证".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(huge),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, _before, _after) = mid_turn_compress(messages, 4_000, None, None);

    let mut assistant_tool_ids = SkipSet::default();
    for message in &compressed {
        if message.role == "assistant" {
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    assistant_tool_ids.insert(call.id.clone());
                }
            }
        }
    }

    let mut tool_message_ids = SkipSet::default();
    for message in &compressed {
        if message.role == "tool" {
            if let Some(id) = &message.tool_call_id {
                tool_message_ids.insert(id.clone());
            }
        }
    }

    for id in &assistant_tool_ids {
        assert!(
            tool_message_ids.contains(id),
            "assistant tool_call '{id}' must have a paired tool message"
        );
    }

    for id in &tool_message_ids {
        assert!(
            assistant_tool_ids.contains(id),
            "tool message '{id}' must be referenced by an assistant tool_call"
        );
    }
}
