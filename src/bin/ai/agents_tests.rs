use super::{
    AgentModelTier, BUILTIN_AGENTS, ProjectInstructionDoc, load_project_instruction_docs_from,
    limit_project_instruction_docs, load_scoped_project_instruction_docs_for_target_priority_from,
    load_scoped_project_instruction_docs_for_target_tiers_from,
    load_scoped_project_instruction_docs_for_targets_from, parse_agent_front_matter,
};
use crate::ai::test_support::ENV_LOCK;
use std::fs;
use std::path::{Path, PathBuf};
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

/// Run `f` with HOME redirected to `fake_home` (so `get_config_dir()` resolves under the
/// temp tree instead of the real user config), then restore the original value. Tests that
/// touch the user-config instruction dir must hold `ENV_LOCK` (HOME is process-global).
fn with_fake_home<T>(fake_home: &Path, f: impl FnOnce() -> T) -> T {
    let old_home = std::env::var_os("HOME");
    unsafe { std::env::set_var("HOME", fake_home) };
    let result = f();
    match old_home {
        Some(value) => unsafe { std::env::set_var("HOME", value) },
        None => unsafe { std::env::remove_var("HOME") },
    }
    result
}

#[test]
fn builtin_audit_agents_request_progress_checkpoints() {
    for (name, source) in BUILTIN_AGENTS.iter().filter(|(name, _)| name.starts_with("audit")) {
        let manifest = parse_agent_front_matter(source).expect("audit agent manifest parses");
        assert!(
            manifest.audit_checkpoints,
            "{name} must request AUDIT_CHECKPOINT progress records"
        );
    }
    // Anything that is not an audit agent keeps the default, so the task path does not
    // append the audit protocol to its prompt.
    let (name, source) = *BUILTIN_AGENTS
        .iter()
        .find(|(name, _)| *name == "build.agent")
        .expect("the build agent is registered");
    assert!(
        !parse_agent_front_matter(source)
            .expect("build agent manifest parses")
            .audit_checkpoints,
        "{name} must not request audit progress records"
    );
}

#[test]
fn manifest_json_without_the_progress_flag_keeps_loading() {
    // The manifest is `Deserialize`, and every other field defaults on absence, so this flag
    // must follow: a manifest written before it existed still has to load.
    let (_, source) = BUILTIN_AGENTS
        .iter()
        .find(|(name, _)| *name == "audit.agent")
        .expect("the audit agent is registered");
    let parsed = parse_agent_front_matter(source).expect("audit agent manifest parses");
    let mut json = serde_json::to_value(&parsed).expect("manifest serializes");
    json.as_object_mut()
        .expect("manifest JSON is an object")
        .remove("audit_checkpoints");

    let manifest: super::AgentManifest =
        serde_json::from_value(json).expect("a manifest without the flag still deserializes");

    assert!(!manifest.audit_checkpoints);
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
fn builtin_audit_agents_opt_out_of_task_auto_selection() {
    // Audit agents ship a fixed output contract (<audit_report>) and a read-only
    // default, so auto-selection must never hand them writing work; the
    // general-purpose build and sharp agents stay eligible.
    for (filename, content) in BUILTIN_AGENTS {
        let agent = parse_agent_front_matter(content).unwrap();
        let expect_eligible = matches!(agent.name.as_str(), "build" | "sharp");
        assert_eq!(
            agent.auto_select, expect_eligible,
            "{filename} (agent {}): expected auto_select={expect_eligible}",
            agent.name
        );
    }
}

#[test]
fn builtin_sharp_agent_pins_no_task_tools_resident() {
    // Explicitly named tools ride along in every turn, and resident task_*
    // tools inject the subagent-orchestration prose ("delegate every step",
    // "task_status before finishing"), which defeats a decisive single-pass
    // agent. Sharp keeps core only; task tools load on demand via enable_tools.
    let (_, content) = BUILTIN_AGENTS
        .iter()
        .find(|(filename, _)| *filename == "sharp.agent")
        .expect("sharp agent should be registered");
    let agent = parse_agent_front_matter(content).unwrap();
    assert!(
        agent.tools.is_empty(),
        "sharp must not pin resident tools, got {:?}",
        agent.tools
    );
    assert_eq!(agent.tool_groups, vec!["core".to_string()]);
}

#[test]
fn builtin_build_agent_prompt_preserves_end_to_end_behavior_tracing() {
    let (_, content) = BUILTIN_AGENTS
        .iter()
        .find(|(filename, _)| *filename == "build.agent")
        .expect("build agent should be registered");
    let agent = parse_agent_front_matter(content).unwrap();

    assert!(agent.prompt.contains("Trace behavior before editing"));
    assert!(
        agent
            .prompt
            .contains("transformations, branches, retries, and consumers")
    );
    assert!(
        agent
            .prompt
            .contains("do not infer a value's meaning or completeness")
    );
    assert!(
        agent
            .prompt
            .contains("Prove behavior, not just compilation")
    );
    // Analysis-only answers carry the same grounding duty as code changes,
    // without demoting stable knowledge to inference.
    assert!(agent.prompt.contains("Ground analysis answers too"));
    assert!(agent.prompt.contains("or mark it as inference"));
    assert!(agent.prompt.contains("stable general knowledge and stated derivations need no such tie"));
}

#[test]
fn builtin_audit_agent_enforces_evidence_driven_review() {
    let (_, content) = BUILTIN_AGENTS
        .iter()
        .find(|(filename, _)| *filename == "audit.agent")
        .expect("audit agent should be registered");
    let agent = parse_agent_front_matter(content).unwrap();

    assert_eq!(agent.name, "audit");
    assert!(agent.is_primary());
    assert!(agent.is_subagent());
    assert_eq!(agent.model_tier, Some(AgentModelTier::Heavy));
    assert_eq!(agent.max_steps, Some(256));
    assert!(agent.disable_mcp_tools);
    assert!(agent.prompt.contains("Falsify candidate findings"));
    assert!(
        agent
            .prompt
            .contains("An unresolved hypothesis is not a finding")
    );
    assert!(
        agent
            .prompt
            .contains("newly introduced behavior from pre-existing behavior")
    );
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
fn project_instruction_docs_include_user_config_dir_for_project() {
    // `~/.config/rust_tools/<project-name>/agents.md` must be loaded exactly like the
    // repo-root instruction docs, keyed by the leaf name of the project root.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let root = temp_dir("project_cfg_docs");
    let nested = root.join("packages/app/src");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(&nested).unwrap();
    fs::write(root.join("AGENTS.md"), "# Root rules\n").unwrap();

    // Fake HOME so get_config_dir() resolves under <root>/home/.config.
    let fake_home = root.join("home");
    let project_name = root.file_name().unwrap().to_string_lossy().into_owned();
    let config_dir = fake_home
        .join(".config")
        .join("rust_tools")
        .join(&project_name);
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(config_dir.join("agents.md"), "# User project rules\n").unwrap();

    let docs = with_fake_home(&fake_home, || load_project_instruction_docs_from(&nested));

    let user_doc = docs
        .iter()
        .find(|doc| doc.path.ends_with("agents.md") && doc.content.contains("User project rules"));
    assert!(
        user_doc.is_some(),
        "user config agents.md must be loaded for the project, got docs: {:?}",
        docs.iter().map(|d| &d.path).collect::<Vec<_>>()
    );
    assert!(
        user_doc.is_some_and(|doc| doc.path.contains("rust_tools")),
        "config doc path must live under the config dir: {:?}",
        user_doc.map(|d| &d.path)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_config_dir_prioritized_over_project_root_same_family() {
    // When ~/.config/rust_tools/<project>/ provides an instruction file (e.g. agents.md),
    // and the project root has the same family (e.g. AGENTS.md), BOTH must load, with
    // the config copy first: priority on conflict, not shadowing. A different family
    // (e.g. CLAUDE.md) must survive untouched.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let root = temp_dir("project_cfg_priority");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join("AGENTS.md"), "# Repo root rules\n").unwrap();
    fs::write(root.join("CLAUDE.md"), "# Repo claude rules\n").unwrap();

    let fake_home = root.join("home");
    let project_name = root.file_name().unwrap().to_string_lossy().into_owned();
    let config_dir = fake_home
        .join(".config")
        .join("rust_tools")
        .join(&project_name);
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(config_dir.join("agents.md"), "# User pinned rules\n").unwrap();

    let docs = with_fake_home(&fake_home, || load_project_instruction_docs_from(&root));

    let repo_idx = docs
        .iter()
        .position(|doc| doc.content.contains("Repo root rules"))
        .expect("project-root AGENTS.md must still load");
    let cfg_idx = docs
        .iter()
        .position(|doc| doc.content.contains("User pinned rules"))
        .expect("config agents.md must load");
    assert!(
        cfg_idx < repo_idx,
        "config copy must come before the project-root copy on same-family conflict, got: {:?}",
        docs.iter().map(|d| &d.path).collect::<Vec<_>>()
    );
    assert!(
        docs.iter()
            .any(|doc| doc.content.contains("Repo claude rules")),
        "a different family (CLAUDE.md) must still load"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_cache_invalidates_on_user_config_change() {
    // The cache fingerprint covers every file in the search scope, including the user
    // config instruction dir (~/.config/rust_tools/<project>/agents.md); rewriting that
    // file must invalidate the cache exactly like a repo-root AGENTS.md change.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let root = temp_dir("project_cfg_cache");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join("AGENTS.md"), "# Root rules\n").unwrap();

    let fake_home = root.join("home");
    let project_name = root.file_name().unwrap().to_string_lossy().into_owned();
    let config_dir = fake_home
        .join(".config")
        .join("rust_tools")
        .join(&project_name);
    fs::create_dir_all(&config_dir).unwrap();
    let cfg_md = config_dir.join("agents.md");
    fs::write(&cfg_md, "cfg-v1: use pnpm.\n").unwrap();

    let first = with_fake_home(&fake_home, || load_project_instruction_docs_from(&root));
    assert!(
        first.iter().any(|doc| doc.content.contains("cfg-v1")),
        "first load must include the user config doc, got: {:?}",
        first.iter().map(|d| &d.path).collect::<Vec<_>>()
    );

    // Rewrite only the config file: the mtime advances and the length changes, so the
    // fingerprint mismatch must force a reload even though the repo-root files are unchanged.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    fs::write(
        &cfg_md,
        "cfg-v2: use cargo and longer content for len change.\n",
    )
    .unwrap();

    let after = with_fake_home(&fake_home, || load_project_instruction_docs_from(&root));
    assert!(
        after.iter().any(|doc| doc.content.contains("cfg-v2")),
        "cache must invalidate when the user config instruction file changes, got: {:?}",
        after.iter().map(|d| &d.path).collect::<Vec<_>>()
    );
    assert!(
        !after.iter().any(|doc| doc.content.contains("cfg-v1")),
        "stale user config content must not survive a cache miss"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_do_not_load_other_projects_config_dir() {
    // The config dir is keyed by the leaf name of the *current* project root; instructions
    // stored under a different project's config dir must never leak into this project.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let root = temp_dir("project_cfg_negative");
    let fake_home = root.join("home");
    let cfg_other = fake_home
        .join(".config")
        .join("rust_tools")
        .join("other_project");
    fs::create_dir_all(&cfg_other).unwrap();
    fs::write(cfg_other.join("agents.md"), "# Other project rules\n").unwrap();

    let project = root.join("current_project");
    fs::create_dir_all(project.join(".git")).unwrap();
    fs::write(project.join("AGENTS.md"), "# Current project rules\n").unwrap();

    let docs = with_fake_home(&fake_home, || load_project_instruction_docs_from(&project));
    assert_eq!(
        docs.len(),
        1,
        "other project's config docs must not load, got: {:?}",
        docs.iter().map(|d| &d.path).collect::<Vec<_>>()
    );
    assert!(docs[0].content.contains("Current project rules"));
    assert!(
        !docs
            .iter()
            .any(|doc| doc.content.contains("Other project rules")),
        "config instructions for a different project must not be loaded"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_without_home_skip_user_config_dir() {
    // With HOME unset, get_config_dir() yields nothing, so the user config dir must be
    // skipped entirely: only the repo-root docs load, without a panic or a bogus path.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let root = temp_dir("project_docs_no_home");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join("AGENTS.md"), "# Root rules\n").unwrap();

    let old_home = std::env::var_os("HOME");
    unsafe { std::env::remove_var("HOME") };
    let docs = load_project_instruction_docs_from(&root);
    match old_home {
        Some(value) => unsafe { std::env::set_var("HOME", value) },
        None => unsafe { std::env::remove_var("HOME") },
    }

    assert_eq!(
        docs.len(),
        1,
        "only the repo-root AGENTS.md must load without HOME, got: {:?}",
        docs.iter().map(|d| &d.path).collect::<Vec<_>>()
    );
    assert!(docs[0].content.contains("Root rules"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instruction_docs_cache_invalidates_on_content_change() {
    // This test locks in the cache semantics: whenever the file's mtime/len changes, it
    // must be re-read from disk so the cache can never expose stale instructions to the LLM.
    let root = temp_dir("project_docs_cache");
    fs::create_dir_all(root.join(".git")).unwrap();
    let agents_md = root.join("AGENTS.md");
    fs::write(&agents_md, "v1: use pnpm.\n").unwrap();

    let first = load_project_instruction_docs_from(&root);
    assert_eq!(first.len(), 1);
    assert!(first[0].content.contains("v1: use pnpm."));

    // Calling again with the same input must give an equivalent result (a cache hit or
    // miss are both fine, as long as the content matches).
    let cached = load_project_instruction_docs_from(&root);
    assert_eq!(cached, first);

    // Rewrite the file and sleep to make sure the mtime advances; also change the length
    // explicitly as a second guarantee, so the fingerprint mismatch triggers.
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

#[test]
fn target_scoped_instruction_docs_add_only_nested_rules() {
    let root = temp_dir("target_scoped_docs");
    let target = root.join("src/bin/ai/driver/iteration.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    fs::write(root.join("src/bin/ai/AGENTS.md"), "ai rules\n").unwrap();
    fs::write(root.join("src/bin/ai/driver/AGENTS.md"), "driver rules\n").unwrap();
    fs::write(&target, "// source\n").unwrap();

    let docs =
        load_scoped_project_instruction_docs_for_targets_from(&root, std::slice::from_ref(&target));

    assert_eq!(docs.len(), 2);
    assert!(docs[0].path.ends_with("src/bin/ai/AGENTS.md"));
    assert!(docs[0].content.contains("ai rules"));
    assert!(docs[1].path.ends_with("src/bin/ai/driver/AGENTS.md"));
    assert!(docs[1].content.contains("driver rules"));
    assert!(docs.iter().all(|doc| !doc.content.contains("root rules")));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn target_scoped_instruction_budget_preserves_deepest_rules_first() {
    let root = temp_dir("target_scoped_budget");
    let target = root.join("src/bin/ai/driver/deep/iteration.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    fs::write(
        root.join("src/bin/ai/AGENTS.md"),
        format!("ai-start\n{}\nai-tail\n", "a".repeat(7_900)),
    )
    .unwrap();
    fs::write(
        root.join("src/bin/ai/driver/AGENTS.md"),
        format!("driver-start\n{}\ndriver-tail\n", "d".repeat(7_900)),
    )
    .unwrap();
    fs::write(
        root.join("src/bin/ai/driver/deep/AGENTS.md"),
        format!("deep-start\n{}\ndeep-tail\n", "z".repeat(7_900)),
    )
    .unwrap();
    fs::write(&target, "// source\n").unwrap();

    let docs =
        load_scoped_project_instruction_docs_for_targets_from(&root, std::slice::from_ref(&target));

    assert_eq!(docs.len(), 3);
    assert!(docs[0].path.ends_with("src/bin/ai/AGENTS.md"));
    assert!(docs[0].content.contains("ai-start"));
    assert!(!docs[0].content.contains("ai-tail"));
    assert!(docs[1].path.ends_with("src/bin/ai/driver/AGENTS.md"));
    assert!(docs[1].content.contains("driver-tail"));
    assert!(docs[2].path.ends_with("src/bin/ai/driver/deep/AGENTS.md"));
    assert!(docs[2].content.contains("deep-tail"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn target_scoped_instruction_budget_prioritizes_pending_mutation() {
    let root = temp_dir("target_scoped_priority");
    let required_target = root.join("required/file.rs");
    let observed_target = root.join("observed/deep/file.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(required_target.parent().unwrap()).unwrap();
    fs::create_dir_all(observed_target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    fs::write(root.join("required/AGENTS.md"), "required mutation rule\n").unwrap();
    fs::write(
        root.join("observed/AGENTS.md"),
        format!("observed-parent\n{}", "p".repeat(8_000)),
    )
    .unwrap();
    fs::write(
        root.join("observed/deep/AGENTS.md"),
        format!("observed-deep\n{}", "d".repeat(8_000)),
    )
    .unwrap();
    fs::write(&required_target, "// required\n").unwrap();
    fs::write(&observed_target, "// observed\n").unwrap();

    let docs = load_scoped_project_instruction_docs_for_target_priority_from(
        &root,
        std::slice::from_ref(&required_target),
        std::slice::from_ref(&observed_target),
    );

    let required = docs
        .iter()
        .find(|doc| doc.path.ends_with("required/AGENTS.md"))
        .expect("pending mutation rules must not be starved by observed targets");
    assert_eq!(required.content, "required mutation rule");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn resident_scoped_instructions_outrank_new_observed_targets_but_not_pending_mutations() {
    let root = temp_dir("target_scoped_resident_tier");
    let resident_dirs = [root.join("resident_a"), root.join("resident_b")];
    let observed_target = root.join("observed/file.rs");
    let required_target = root.join("required/file.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    for dir in resident_dirs
        .iter()
        .chain([&root.join("observed"), &root.join("required")])
    {
        fs::create_dir_all(dir).unwrap();
    }
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    // Two resident documents near the per-document cap fill the budget.
    let resident_rules = format!("resident-start\n{}\nresident-tail", "r".repeat(7_900));
    for dir in &resident_dirs {
        fs::write(dir.join("AGENTS.md"), &resident_rules).unwrap();
    }
    fs::write(
        root.join("observed/AGENTS.md"),
        format!("observed-start\n{}\nobserved-tail", "o".repeat(7_900)),
    )
    .unwrap();
    fs::write(
        root.join("required/AGENTS.md"),
        format!("required-start\n{}\nrequired-tail", "q".repeat(7_900)),
    )
    .unwrap();
    fs::write(&observed_target, "// observed\n").unwrap();
    fs::write(&required_target, "// required\n").unwrap();

    // A newly observed target competes for the remaining budget only: the
    // resident documents keep every byte they were rendered with.
    let docs = load_scoped_project_instruction_docs_for_target_tiers_from(
        &root,
        &[],
        std::slice::from_ref(&observed_target),
        &resident_dirs,
    );
    let resident = docs
        .iter()
        .filter(|doc| doc.path.contains("/resident_"))
        .collect::<Vec<_>>();
    assert_eq!(resident.len(), 2, "resident rules must not be evicted by observed targets");
    assert!(resident.iter().all(|doc| doc.content == resident_rules));
    assert!(
        docs.iter()
            .filter(|doc| doc.path.ends_with("observed/AGENTS.md"))
            .all(|doc| !doc.content.contains("observed-tail")),
        "observed rules only fill the remaining budget"
    );

    // A pending mutation still outranks resident rules, so preflight can
    // always unblock it.
    let docs = load_scoped_project_instruction_docs_for_target_tiers_from(
        &root,
        std::slice::from_ref(&required_target),
        std::slice::from_ref(&observed_target),
        &resident_dirs,
    );
    let required = docs
        .iter()
        .find(|doc| doc.path.ends_with("required/AGENTS.md"))
        .expect("pending mutation rules must load first");
    assert!(required.content.contains("required-tail"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn target_scoped_instruction_docs_ignore_paths_outside_project() {
    let root = temp_dir("target_scoped_outside");
    let outside = temp_dir("target_scoped_external").join("file.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    fs::write(
        outside.parent().unwrap().join("AGENTS.md"),
        "outside rules\n",
    )
    .unwrap();
    fs::write(&outside, "// source\n").unwrap();

    let docs = load_scoped_project_instruction_docs_for_targets_from(
        &root,
        std::slice::from_ref(&outside),
    );

    assert!(docs.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(outside.parent().unwrap());
}

#[test]
fn project_instruction_docs_budget_truncates_and_omits_in_order() {
    // The total budget applies in vector order: a doc that partially fits is
    // truncated with a trailing `…` (reported on stderr), and docs beyond the
    // budget are omitted entirely (also reported on stderr). The returned
    // vector is what the prompt sees; this test pins that behavior.
    let docs = vec![
        ProjectInstructionDoc {
            path: "/tmp/budget_a.md".into(),
            content: "a".repeat(300),
        },
        ProjectInstructionDoc {
            path: "/tmp/budget_b.md".into(),
            content: "b".repeat(500),
        },
        ProjectInstructionDoc {
            path: "/tmp/budget_c.md".into(),
            content: "c".repeat(200),
        },
    ];

    // 600 chars: a fits, b keeps 300 chars plus the ellipsis, c is omitted.
    let limited = limit_project_instruction_docs(docs, 600);
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0].content, "a".repeat(300));
    assert_eq!(limited[1].content.chars().count(), 301);
    assert!(limited[1].content.ends_with('…'));

    // Budget exhausted before the second doc: only the first survives.
    let docs = vec![
        ProjectInstructionDoc {
            path: "/tmp/budget_a.md".into(),
            content: "a".repeat(300),
        },
        ProjectInstructionDoc {
            path: "/tmp/budget_b.md".into(),
            content: "b".repeat(50),
        },
    ];
    let limited = limit_project_instruction_docs(docs, 100);
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].content.chars().count(), 101);
}
