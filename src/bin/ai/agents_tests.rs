use super::{
    AgentModelTier, BUILTIN_AGENTS, load_project_instruction_docs_from, parse_agent_front_matter,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!(
        "rust_tools_agents_{name}_{}_{}",
        std::process::id(),
        nanos
    ));
    path
}

#[test]
fn parses_model_tier_from_front_matter() {
    let content = r#"---
name: test-agent
description: Fast read-only codebase exploration
mode: subagent
model_tier: light
---

Read the codebase and summarize findings.
"#;

    let agent = parse_agent_front_matter(content).unwrap();
    assert_eq!(agent.name, "test-agent");
    assert_eq!(agent.model_tier, Some(AgentModelTier::Light));
}

#[test]
fn rejects_invalid_model_tier_in_front_matter() {
    let content = r#"---
name: bad
description: invalid tier
model_tier: giant
---

noop
"#;

    let err = parse_agent_front_matter(content).unwrap_err();
    assert!(err.contains("invalid model_tier"));
}

#[test]
fn parses_disable_mcp_tools_from_front_matter() {
    let content = r#"---
name: build
description: Development agent
disable_mcp_tools: true
---

Build things.
"#;

    let agent = parse_agent_front_matter(content).unwrap();

    assert!(agent.disable_mcp_tools);
}

#[test]
fn builtin_agents_do_not_mount_mcp_tools_by_default() {
    for (filename, content) in BUILTIN_AGENTS {
        let agent = parse_agent_front_matter(content).unwrap();
        assert!(
            agent.disable_mcp_tools,
            "{filename} should use progressive MCP loading instead of mounting every MCP tool"
        );
    }
}

#[test]
fn project_instruction_docs_include_root_and_nested_scope() {
    let root = temp_dir("project_docs");
    let nested = root.join("packages/app/src");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(&nested).unwrap();
    fs::write(root.join("AGENTS.md"), "# Root rules\nUse pnpm.\n").unwrap();
    fs::write(
        root.join("packages/app/CLAUDE.md"),
        "# App rules\nRun app tests only.\n",
    )
    .unwrap();

    let docs = load_project_instruction_docs_from(&nested);
    assert_eq!(docs.len(), 2);
    assert!(docs[0].path.ends_with("AGENTS.md"));
    assert!(docs[0].content.contains("Use pnpm."));
    assert!(docs[1].path.ends_with("CLAUDE.md"));
    assert!(docs[1].content.contains("Run app tests only."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_fall_back_to_doc_ancestors_without_repo_markers() {
    let root = temp_dir("project_docs_nomarker");
    let nested = root.join("services/api/src");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        root.join("claude.md"),
        "# Project rules\nPrefer make targets.\n",
    )
    .unwrap();

    let docs = load_project_instruction_docs_from(&nested);
    assert_eq!(docs.len(), 1);
    assert!(docs[0].path.ends_with("claude.md"));
    assert!(docs[0].content.contains("Prefer make targets."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_cache_invalidates_on_content_change() {
    // 该测试锁住缓存语义：只要文件 mtime/len 变化就必须重新读盘，缓存
    // 不能让 LLM 看到旧版指令。
    let root = temp_dir("project_docs_cache");
    fs::create_dir_all(root.join(".git")).unwrap();
    let agents_md = root.join("AGENTS.md");
    fs::write(&agents_md, "v1: use pnpm.\n").unwrap();

    let first = load_project_instruction_docs_from(&root);
    assert_eq!(first.len(), 1);
    assert!(first[0].content.contains("v1: use pnpm."));

    // 同样输入再调一次，结果应等价（命中缓存或不命中都允许，只要内容一致）。
    let cached = load_project_instruction_docs_from(&root);
    assert_eq!(cached, first);

    // 改文件并睡眠确保 mtime 推进；同时显式让 len 变化，双重保险触发
    // fingerprint 失配。
    std::thread::sleep(std::time::Duration::from_millis(1100));
    fs::write(
        &agents_md,
        "v2: use cargo and longer content for len change.\n",
    )
    .unwrap();

    let after = load_project_instruction_docs_from(&root);
    assert_eq!(after.len(), 1);
    assert!(
        after[0].content.contains("v2: use cargo"),
        "cache must invalidate on file change, got: {}",
        after[0].content
    );

    let _ = fs::remove_dir_all(root);
}
