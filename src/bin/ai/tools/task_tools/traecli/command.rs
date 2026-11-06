//! TraeCLI executable, model, argv, and stdin preparation.
//! Process execution and lifecycle stay in the parent runner.

use std::path::{Path, PathBuf};

use super::{ExternalSubagentSpec, TRAECLI_BUILTIN_DEFAULT_MODEL};

pub(super) fn traecli_binary() -> String {
    let raw = std::env::var("TRAECLI_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "traecli".to_string());
    let path = Path::new(&raw);
    if path.components().count() > 1 && path.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            return cwd.join(path).to_string_lossy().into_owned();
        }
    }
    raw
}

/// Cheap preflight used when a task is admitted: an absent binary is a
/// definite startup failure, so `task_spawn` can report it in its immediate
/// per-task status instead of waiting for the runner's startup watchdog.
pub(super) fn traecli_binary_available() -> bool {
    binary_available(&traecli_binary())
}

pub(super) fn binary_available(binary: &str) -> bool {
    let path = Path::new(&binary);
    if path.components().count() > 1 {
        return path.is_file();
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

/// Resolve the TraeCLI model: explicit per-call value first, then
/// `TRAECLI_MODEL`, then `ai.traecli.default_model`, otherwise the built-in
/// default. Never fed from the parent model name: the two model registries
/// use different naming.
pub(super) fn traecli_model(spec: &ExternalSubagentSpec) -> String {
    traecli_model_for(spec.model.as_deref())
}

/// Resolve the effective TraeCLI model from a per-call or session pick without
/// requiring a full spec. Task preparation prefers the per-call pick over the
/// session pick before passing it here; environment/config defaults follow.
pub(super) fn traecli_model_for(explicit: Option<&str>) -> String {
    let env = std::env::var("TRAECLI_MODEL").ok();
    let config = crate::commonw::configw::get_all_config()
        .get_opt(crate::ai::config_schema::AiConfig::TRAECLI_DEFAULT_MODEL);
    resolve_traecli_model(explicit, env.as_deref(), config.as_deref())
}

/// Pure precedence chain behind [`traecli_model`], kept separate so tests do
/// not depend on process environment or user config.
pub(super) fn resolve_traecli_model(
    explicit: Option<&str>,
    env: Option<&str>,
    config: Option<&str>,
) -> String {
    for candidate in [explicit, env, config] {
        if let Some(name) = candidate.map(str::trim).filter(|s| !s.is_empty()) {
            return name.to_string();
        }
    }
    TRAECLI_BUILTIN_DEFAULT_MODEL.to_string()
}

/// Build the headless TraeCLI command for a subagent run.
///
/// Fresh runs are `exec --json` with `--skip-git-repo-check`, an explicit
/// `-C` working root, and a sandbox derived from workspace access. Runs are
/// deliberately not `--ephemeral` so follow-up tasks can resume them via
/// `backend_continue_from`. Resume runs are `exec resume --json
/// --skip-git-repo-check --all <session>`: resume inherits the recorded
/// working root and sandbox, so `-C`/`-s` are omitted, and skills are not
/// re-prepended (the session already has them). `--permission-mode` stays
/// omitted (see module docs). Never YOLO.
pub(super) fn build_traecli_args(spec: &ExternalSubagentSpec) -> Vec<String> {
    if let Some(session) = spec
        .resume_session
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let mut args = vec![
            "exec".to_string(),
            "resume".to_string(),
            "--json".to_string(),
            "--skip-git-repo-check".to_string(),
            "--all".to_string(),
            session.to_string(),
            "-m".to_string(),
            traecli_model(spec),
        ];
        push_effort_arg(&mut args, spec);
        return args;
    }
    let cwd = normalize_cwd(&spec.cwd);
    let mut args = vec![
        "exec".to_string(),
        "--json".to_string(),
        "--skip-git-repo-check".to_string(),
        "-C".to_string(),
        cwd.to_string_lossy().into_owned(),
        "-s".to_string(),
        if spec.workspace_allowed {
            "workspace-write"
        } else {
            "read-only"
        }
        .to_string(),
    ];
    args.push("-m".to_string());
    args.push(traecli_model(spec));
    push_effort_arg(&mut args, spec);
    push_image_args(&mut args, spec);
    args
}

/// Prompt bytes written to the child's stdin. Fresh runs prepend the inherited
/// skill overlay; resume runs send only the follow-up because the persisted
/// session already holds the original prompt and skills.
pub(super) fn prompt_payload(spec: &ExternalSubagentSpec) -> String {
    let resuming = spec
        .resume_session
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let mut payload = if resuming {
        append_original_image_paths(spec.prompt.clone(), &spec.image_files)
    } else {
        full_prompt(spec)
    };
    if spec.phase_slot.is_some() {
        payload.push_str(
            "\n\nDuring long-running work, publish saved findings in a separate progress message \
             using this format:\n<context_checkpoint>\nsummary: Short finding summary\n\
             Checkpoint body with the evidence gathered so far.\n</context_checkpoint>\n\
             Keep each checkpoint concise. Send the final answer separately after finishing.",
        );
    }
    payload
}

/// Normalize effort for both TraeCLI argv and display. TraeCLI has no "off"
/// tier, so `none`/`minimal` clamp to `low`; other supported tiers pass through.
/// Unknown values are omitted so the backend default wins.
pub(super) fn normalize_traecli_reasoning_effort(effort: &str) -> Option<&str> {
    match effort.trim() {
        "none" | "minimal" => Some("low"),
        effort @ ("low" | "medium" | "high" | "xhigh" | "max") => Some(effort),
        _ => None,
    }
}

fn push_effort_arg(args: &mut Vec<String>, spec: &ExternalSubagentSpec) {
    let Some(mapped) = spec
        .reasoning_effort
        .as_deref()
        .and_then(normalize_traecli_reasoning_effort)
    else {
        return;
    };
    args.push("-c".to_string());
    args.push(format!("model_reasoning_effort={mapped}"));
}

/// Attach the parent turn's original images to the child's first request.
/// Codex's `exec` accepts one `-i/--image <FILE>` per image; resume runs
/// already carry them in the persisted session, so only the path hint is
/// re-sent there.
fn push_image_args(args: &mut Vec<String>, spec: &ExternalSubagentSpec) {
    for path in &spec.image_files {
        args.push("-i".to_string());
        args.push(path.clone());
    }
}

/// Later turns can compact the original request down to a summary. Keep the
/// absolute paths in the follow-up prompt so the model can reopen the original
/// whenever the task still depends on its visual detail.
fn append_original_image_paths(mut prompt: String, image_files: &[String]) -> String {
    if image_files.is_empty() {
        return prompt;
    }
    prompt.push_str("\n\n## Original images\n");
    for path in image_files {
        prompt.push_str("- ");
        prompt.push_str(path);
        prompt.push('\n');
    }
    prompt.push_str("Use your image viewer on these paths when you need the original again.");
    prompt
}

/// Final prompt text: parent skill context first (mirroring the kernel path,
/// where skills are system-prompt overlays above the task), then the task.
pub(super) fn full_prompt(spec: &ExternalSubagentSpec) -> String {
    match spec
        .skills
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(skills) => format!("{skills}\n\n{}", spec.prompt),
        None => spec.prompt.clone(),
    }
}

pub(super) fn normalize_cwd(cwd: &Path) -> PathBuf {
    if cwd.as_os_str().is_empty() {
        std::env::temp_dir()
    } else {
        cwd.to_path_buf()
    }
}
