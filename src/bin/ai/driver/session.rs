//! Session startup / resume: suspended session preview, selection prompt,
//! and startup session choice resolution.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::io::IsTerminal;
use std::path::PathBuf;

use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::ai::{
    cli,
    history::{
        SessionStore, SuspendedSessionEntry, SuspendedSessionStore,
        format_suspended_timestamp_label,
    },
};

#[derive(Debug, Clone)]
pub(super) struct StartupSessionChoice {
    pub(super) active_persona: crate::ai::persona::PersonaProfile,
    pub(super) history_file: PathBuf,
    pub(super) session_id: String,
    pub(super) model: Option<String>,
    pub(super) startup_notice: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct SuspendedSessionPreview {
    pub(super) entry: SuspendedSessionEntry,
    pub(super) persona_label: String,
    pub(super) summary: Option<String>,
    pub(super) modified_label: Option<String>,
    pub(super) suspended_label: String,
}

pub(super) fn build_suspended_session_previews(
    entries: Vec<SuspendedSessionEntry>,
    persona_store: &crate::ai::persona::PersonaStore,
) -> Vec<SuspendedSessionPreview> {
    let personas = persona_store.list_personas().unwrap_or_default();
    let mut sessions_by_history: FxHashMap<PathBuf, Vec<_>> = FxHashMap::default();

    entries
        .into_iter()
        .map(|entry| {
            let persona_label = personas
                .iter()
                .find(|persona| persona.id == entry.persona_id)
                .map(|persona| persona.name.clone())
                .unwrap_or_else(|| entry.persona_id.clone());
            let session_info = sessions_by_history
                .entry(entry.history_file.clone())
                .or_insert_with(|| {
                    SessionStore::new(entry.history_file.as_path())
                        .list_sessions()
                        .unwrap_or_default()
                })
                .iter()
                .find(|session| session.id == entry.session_id);
            SuspendedSessionPreview {
                persona_label,
                summary: session_info.and_then(|session| session.summary.clone()),
                modified_label: session_info.and_then(|session| {
                    session
                        .modified_local
                        .as_ref()
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                }),
                suspended_label: format_suspended_timestamp_label(&entry.suspended_at),
                entry,
            }
        })
        .collect()
}

pub(super) fn prompt_select_suspended_session(
    previews: &[SuspendedSessionPreview],
) -> std::io::Result<Option<usize>> {
    if previews.is_empty() {
        return Ok(None);
    }
    if previews.len() == 1 || !std::io::stdin().is_terminal() {
        return Ok(Some(0));
    }

    println!(
        "[resume] 当前 terminal 有 {} 个挂起 session：",
        previews.len()
    );
    for (index, preview) in previews.iter().enumerate() {
        println!(
            "  {}. {}  persona={}  modified={}  suspended={}",
            index + 1,
            preview.entry.session_id,
            preview.persona_label,
            preview.modified_label.as_deref().unwrap_or("-"),
            preview.suspended_label
        );
        if let Some(summary) = preview
            .summary
            .as_deref()
            .filter(|summary| !summary.is_empty())
        {
            println!("     {summary}");
        }
    }

    loop {
        let input = crate::commonw::prompt::read_line(&format!(
            "选择要恢复的 session [1-{}，回车=1，n=新 session]: ",
            previews.len()
        ));
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(Some(0));
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower == "n" || lower == "new" {
            return Ok(None);
        }
        if let Ok(index) = trimmed.parse::<usize>()
            && (1..=previews.len()).contains(&index)
        {
            return Ok(Some(index - 1));
        }
        eprintln!(
            "[resume] 无效选择：请输入 1-{}，或输入 n 新建 session。",
            previews.len()
        );
    }
}

pub(super) fn build_resume_startup_notice(
    session_id: &str,
    remaining_suspended: usize,
    persona_fallback: bool,
) -> String {
    let mut notice = format!("[resume] 已恢复挂起 session: {session_id}");
    if persona_fallback {
        notice.push_str("（原 persona 不存在，已按当前 persona 打开）");
    }
    if remaining_suspended > 0 {
        notice.push_str(&format!(
            "；当前 terminal 还有 {} 个挂起 session，运行 `a --resume` 可继续选择。",
            remaining_suspended
        ));
    }
    notice.push_str("；运行 `a --new-session` 可强制新建 session。");
    notice
}

pub(super) fn should_resume_suspended_terminal_session(cli: &cli::ParsedCli) -> bool {
    if cli.new_session {
        return false;
    }
    if cli.resume {
        return true;
    }
    if cli.session.is_some() || cli.clear || !cli.args.is_empty() {
        return false;
    }
    if cli.help
        || cli.list_tools
        || cli.list_mcp_tools
        || cli.list_skills
        || cli.list_agents
        || cli.note_search
        || cli.note_flag
        || cli.note_delete.is_some()
        || cli.note_edit.is_some()
        || cli.consolidate_knowledge
        || cli.generate_completions
    {
        return false;
    }
    true
}

pub(super) fn resolve_startup_session_choice_with_selector<F>(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
    selector: F,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>>
where
    F: FnMut(&[SuspendedSessionPreview]) -> std::io::Result<Option<usize>>,
{
    resolve_startup_session_choice_with_selector_inner(
        cli,
        config,
        persona_store,
        active_persona,
        selector,
    )
}

pub(super) fn resolve_startup_session_choice_with_selector_inner<F>(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
    mut selector: F,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>>
where
    F: FnMut(&[SuspendedSessionPreview]) -> std::io::Result<Option<usize>>,
{
    if cli.resume && cli.session.is_some() {
        return Err("`--resume` 不能和 `--session` 同时使用".into());
    }
    if cli.resume && cli.clear {
        return Err("`--resume` 不能和 `--clear` 同时使用".into());
    }
    if cli.resume && cli.new_session {
        return Err("`--resume` 不能和 `--new-session` 同时使用".into());
    }
    if cli.new_session && cli.session.is_some() {
        return Err("`--new-session` 不能和 `--session` 同时使用".into());
    }
    if cli.new_session && cli.clear {
        return Err("`--new-session` 不能和 `--clear` 同时使用".into());
    }

    let mut choice = StartupSessionChoice {
        history_file: crate::ai::persona::history_file_for_persona(
            config.base_history_file.as_path(),
            &active_persona.id,
        ),
        active_persona,
        session_id: cli
            .session
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(|id| id.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        model: None,
        startup_notice: None,
    };

    if !should_resume_suspended_terminal_session(cli) {
        return Ok(choice);
    }

    let suspended_store = SuspendedSessionStore::new();
    match suspended_store.list_current_terminal() {
        Ok(entries) if entries.is_empty() => {
            if cli.resume {
                choice.startup_notice = Some(
                    "[resume] 当前 terminal 没有可恢复的挂起 session，已创建新 session。"
                        .to_string(),
                );
            }
        }
        Ok(entries) => {
            let previews = build_suspended_session_previews(entries, persona_store);
            let selected_index = if previews.len() == 1 && !cli.resume {
                Some(0)
            } else {
                selector(&previews)?
            };
            let Some(selected_index) = selected_index else {
                choice.startup_notice = Some(format!(
                    "[resume] 已跳过当前 terminal 的 {} 个挂起 session，已创建新 session。运行 `a --resume` 可再次选择恢复。",
                    previews.len()
                ));
                return Ok(choice);
            };
            let selected = previews
                .get(selected_index)
                .ok_or_else(|| format!("invalid suspended session selection: {selected_index}"))?;
            let Some(entry) = suspended_store.take_selected_current_terminal(&selected.entry)?
            else {
                if cli.resume {
                    return Err("选中的挂起 session 已不存在，请重试".into());
                }
                choice.startup_notice =
                    Some("[resume] 选中的挂起 session 已不存在，已创建新 session。".to_string());
                return Ok(choice);
            };
            choice.history_file = entry.history_file.clone();
            choice.session_id = entry.session_id.clone();
            // 恢复挂起时保存的模型，而非使用默认模型
            choice.model = entry.model.clone();

            let remaining = previews.len().saturating_sub(1);
            let mut persona_fallback = false;
            match persona_store.list_personas() {
                Ok(personas) => {
                    if let Some(persona) = personas.into_iter().find(|p| p.id == entry.persona_id) {
                        choice.active_persona = persona;
                    } else {
                        persona_fallback = true;
                    }
                }
                Err(err) => {
                    eprintln!("[resume] failed to load personas: {}", err);
                }
            }
            choice.startup_notice = Some(build_resume_startup_notice(
                &choice.session_id,
                remaining,
                persona_fallback,
            ));
        }
        Err(err) => {
            if cli.resume {
                return Err(err.into());
            }
            if err.kind() != std::io::ErrorKind::Unsupported {
                eprintln!("[resume] 自动恢复已跳过：{}", err);
            }
        }
    }

    Ok(choice)
}

pub(super) fn resolve_startup_session_choice(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>> {
    resolve_startup_session_choice_with_selector_inner(
        cli,
        config,
        persona_store,
        active_persona,
        prompt_select_suspended_session,
    )
}
