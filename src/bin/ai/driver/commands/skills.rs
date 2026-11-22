use std::sync::Arc;

use crate::ai::{
    skills::SkillManifest,
    types::{App, ForcedSkillSource},
};

fn pending_skill_names(app: &App) -> Vec<&str> {
    app.forced_skills.iter().map(|s| s.as_str()).collect()
}

fn recent_skill_name(app: &App) -> Option<&str> {
    app.last_skill_bias
        .as_ref()
        .map(|memory| memory.skill_name.as_str())
}

fn current_skill_lines(app: &App) -> Vec<String> {
    let pending = pending_skill_names(app);
    let mut lines = Vec::new();
    if !pending.is_empty() {
        lines.push(format!(
            "current skills: {} (pending for next turn)",
            pending.join(", ")
        ));
    } else if let Some(name) = recent_skill_name(app) {
        lines.push(format!("current skill: {name} (recent active skill)"));
    }
    if !app.session_default_skills.is_empty() {
        lines.push(format!(
            "Pinned (session default): {}",
            app.session_default_skills.join(", ")
        ));
    }
    if lines.is_empty() {
        lines.push("No active skill.".to_string());
    }
    lines
}

fn resolve_skill_names(
    requested_names: &[&str],
    skill_manifests: &[SkillManifest],
) -> (Vec<String>, Vec<String>) {
    let mut names: Vec<String> = Vec::new();
    let mut not_found = Vec::new();
    for requested in requested_names {
        if let Some(skill) = skill_manifests
            .iter()
            .find(|skill| skill.name.eq_ignore_ascii_case(requested))
        {
            if !names
                .iter()
                .any(|name| name.eq_ignore_ascii_case(&skill.name))
            {
                names.push(skill.name.clone());
            }
        } else {
            not_found.push((*requested).to_string());
        }
    }
    (names, not_found)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SkillPinChanges {
    added: Vec<String>,
    already_pinned: Vec<String>,
    not_found: Vec<String>,
}

fn pin_skill_names(
    app: &mut App,
    requested_names: &[&str],
    skill_manifests: &[SkillManifest],
) -> SkillPinChanges {
    let (names, not_found) = resolve_skill_names(requested_names, skill_manifests);
    let mut changes = SkillPinChanges {
        not_found,
        ..SkillPinChanges::default()
    };
    for name in names {
        if app
            .session_default_skills
            .iter()
            .any(|stored| stored.eq_ignore_ascii_case(&name))
        {
            changes.already_pinned.push(name);
        } else {
            app.session_default_skills.push(name.clone());
            changes.added.push(name);
        }
    }
    changes
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SkillUnpinChanges {
    removed: Vec<String>,
    not_pinned: Vec<String>,
    agent_team_refused: bool,
}

fn unpin_skill_names(app: &mut App, requested_names: &[&str]) -> SkillUnpinChanges {
    let mut changes = SkillUnpinChanges::default();
    let mut seen: Vec<&str> = Vec::new();
    for requested in requested_names {
        if seen.iter().any(|name| name.eq_ignore_ascii_case(requested)) {
            continue;
        }
        seen.push(requested);
        if requested.eq_ignore_ascii_case("agent-team") {
            changes.agent_team_refused = true;
            continue;
        }
        if let Some(stored) = app
            .session_default_skills
            .iter()
            .find(|name| name.eq_ignore_ascii_case(requested))
            .cloned()
        {
            app.session_default_skills
                .retain(|name| !name.eq_ignore_ascii_case(requested));
            changes.removed.push(stored);
        } else {
            changes.not_pinned.push((*requested).to_string());
        }
    }
    changes
}

fn is_session_default_skill_action(name: &str) -> bool {
    name.eq_ignore_ascii_case("pin") || name.eq_ignore_ascii_case("unpin")
}

// Phone composers send multi-line input as one command string, so a newline
// ends the pin/unpin skill-name arguments.
fn split_pin_trailing_question(arguments: &str) -> (&str, Option<&str>) {
    let Some((names, question)) = arguments.split_once('\n') else {
        return (arguments, None);
    };
    let names = names.strip_suffix('\r').unwrap_or(names);
    let question = question.trim();
    (names, (!question.is_empty()).then_some(question))
}

pub fn try_handle_skills_command(
    app: &mut App,
    input: &str,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "skills" && cmd != "skill" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    let (rest_tokens, trailing_question): (Vec<&str>, Option<&str>) =
        if matches!(action, "pin" | "unpin") {
            let action_and_arguments = normalized.trim_start()[cmd.len()..].trim_start();
            let (names, question) =
                split_pin_trailing_question(&action_and_arguments[action.len()..]);
            (names.split_whitespace().collect(), question)
        } else {
            (parts.collect(), None)
        };

    match action {
        "help" | "h" | "--help" => {
            println!("Skill management commands:");
            println!();
            println!("  /skills              list all available skills");
            println!("  /skills list         list all available skills");
            println!("  /skills current      show pending/recent and pinned skills");
            println!("  /skills use <name>...  force the specified skills on the next turn only");
            println!("  /skills pin <name>...  pin skills as defaults for this session");
            println!("  /skills unpin <name>... remove pinned session defaults");
            println!("  /skills <name>...      one-turn shorthand for /skills use (also accepts a");
            println!("                         trailing question: /skills <name>... <question>)");
            println!("  /skills help         show this help");
            println!();
            println!(
                "Pinned defaults persist when this session is reopened; explicit picks add for one turn."
            );
            println!("Use /reset to unpin agent-team and leave the /use <backend> mode.");
            println!();
        }
        "list" | "ls" | "" => {
            let skills = &**skill_manifests;
            let pending = pending_skill_names(app);
            let recent = recent_skill_name(app);
            if skills.is_empty() {
                println!("No skills available.");
            } else {
                for s in skills {
                    let is_pending = pending
                        .iter()
                        .any(|name| name.eq_ignore_ascii_case(&s.name));
                    let is_recent = recent.is_some_and(|name| name.eq_ignore_ascii_case(&s.name));
                    let is_pinned = app
                        .session_default_skills
                        .iter()
                        .any(|name| name.eq_ignore_ascii_case(&s.name));
                    let mark = if is_pending || is_recent { "*" } else { " " };
                    let mut extras = Vec::new();
                    if !s.description.trim().is_empty() {
                        extras.push(format!("· {}", s.description.trim()));
                    }
                    if is_pending {
                        extras.push("[pending]".to_string());
                    }
                    if is_recent {
                        extras.push("[recent]".to_string());
                    }
                    if is_pinned {
                        extras.push("[pinned]".to_string());
                    }
                    if extras.is_empty() {
                        println!("{}  {}", mark, s.name);
                    } else {
                        println!("{}  {}  {}", mark, s.name, extras.join(" "));
                    }
                }
            }
        }
        "current" | "cur" => {
            for line in current_skill_lines(app) {
                println!("{line}");
            }
        }
        "pin" => {
            if rest_tokens.is_empty() {
                println!("Usage: /skills pin <skill-name> [<skill-name>...]");
            } else {
                let changes = pin_skill_names(app, &rest_tokens, skill_manifests);
                for missing in &changes.not_found {
                    println!("Skill '{missing}' not found.");
                }
                if !changes.added.is_empty() {
                    println!(
                        "Pinned skills for this session: {}",
                        changes.added.join(", ")
                    );
                }
                if !changes.already_pinned.is_empty() {
                    println!("Already pinned: {}", changes.already_pinned.join(", "));
                }
                if !changes.added.is_empty() {
                    super::session::persist_session_runtime_state(app);
                }
            }
        }
        "unpin" => {
            if rest_tokens.is_empty() {
                println!("Usage: /skills unpin <skill-name> [<skill-name>...]");
            } else {
                let changes = unpin_skill_names(app, &rest_tokens);
                if changes.agent_team_refused {
                    println!(
                        "Skill 'agent-team' is required by /use <backend>; use /reset to leave the mode."
                    );
                }
                if !changes.removed.is_empty() {
                    println!("Unpinned: {}", changes.removed.join(", "));
                }
                if !changes.not_pinned.is_empty() {
                    println!("Not pinned: {}", changes.not_pinned.join(", "));
                }
                if !changes.removed.is_empty() {
                    super::session::persist_session_runtime_state(app);
                }
            }
        }
        "use" | "select" | "switch" => {
            if rest_tokens.is_empty() {
                println!("Usage: /skills use <skill-name>");
                println!("Available skills:");
                for s in &**skill_manifests {
                    println!(
                        "  {}  {}",
                        s.name,
                        if s.description.trim().is_empty() {
                            String::new()
                        } else {
                            format!("· {}", s.description.trim())
                        }
                    );
                }
                return Ok(true);
            }

            // Resolve the names one by one (case-insensitive, deduplicated);
            // unmatched names get their own hint, and one invalid name should
            // not sink the whole set.
            let (names, not_found) = resolve_skill_names(&rest_tokens, skill_manifests);
            for missing in &not_found {
                println!("Skill '{missing}' not found.");
            }
            if names.is_empty() {
                println!();
                println!("Usage: /skills use <skill-name> [<skill-name>...]");
                println!("Available skills:");
                for s in &**skill_manifests {
                    println!(
                        "  {}  {}",
                        s.name,
                        if s.description.trim().is_empty() {
                            String::new()
                        } else {
                            format!("· {}", s.description.trim())
                        }
                    );
                }
            } else {
                app.forced_skills = names.clone();
                app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandNextTurn);
                println!("Skills selected for next turn: {}", names.join(", "));
                println!(
                    "Ask your next question naturally, or mention @skills:{} inline.",
                    names.join(",")
                );
            }
        }
        // Implicit selection: /skills <skillname> applies directly without the
        // use keyword.
        _ => {
            // Greedily collect consecutive tokens that match a skill name as the
            // skill set (keeping input order, normalized by manifest,
            // deduplicated); the remaining tokens become this turn's question.
            // Also supports a skill name glued to the question text (e.g.
            // `/skill code-review` with the question glued on, no space).
            let mut tokens = Vec::with_capacity(rest_tokens.len() + 1);
            tokens.push(action);
            tokens.extend(rest_tokens);
            let (names, rest) = resolve_implicit_selection(&tokens, skill_manifests);

            if names.is_empty() {
                println!("Unknown /skills subcommand: {action}");
                println!();
                println!(
                    "Usage: /skills [list|current|use <name>...|pin <name>...|unpin <name>...|help]"
                );
                println!();
                println!("Available skills:");
                for s in &**skill_manifests {
                    println!(
                        "  {}  {}",
                        s.name,
                        if s.description.trim().is_empty() {
                            String::new()
                        } else {
                            format!("· {}", s.description.trim())
                        }
                    );
                }
            } else if let Some(rest) = rest {
                // /skills <name>... <rest>: rest becomes this turn's question.
                app.forced_skills = names;
                app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandInline);
                app.forced_question = Some(rest);
                return Ok(true);
            } else {
                app.forced_skills = names.clone();
                app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandNextTurn);
                println!("Skills selected for next turn: {}", names.join(", "));
                println!(
                    "Ask your next question naturally, or mention @skills:{} inline.",
                    names.join(",")
                );
            }
        }
    }
    if let Some(question) = trailing_question {
        app.forced_question = Some(question.to_string());
    }
    Ok(true)
}

/// Parse implicit multi-skill selection of the form /skills <name>... <rest...>.
/// Greedily collects **consecutive** tokens that match a skill name as the skill
/// set (case-insensitive, normalized by manifest, deduplicated, input order
/// preserved); the remaining tokens become this turn's question.
/// A token with no whole-name match may still have an ASCII identifier prefix
/// (letters, digits, `-`, or `_`) matching a skill. Its suffix and subsequent
/// tokens then become the question.
/// Returns (normalized name list, rest).
fn resolve_implicit_selection<'m>(
    tokens: &[&'m str],
    skill_manifests: &[SkillManifest],
) -> (Vec<String>, Option<String>) {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_';
    let mut names: Vec<String> = Vec::new();
    let mut rest_tokens: Vec<&'m str> = Vec::new();

    let push_name = |name: &str, names: &mut Vec<String>| {
        if !names.iter().any(|n| n == name) {
            names.push(name.to_string());
        }
    };

    let mut iter = tokens.iter();
    while let Some(token) = iter.next() {
        if let Some(skill) = skill_manifests.iter().find(|s| {
            !is_session_default_skill_action(&s.name) && s.name.eq_ignore_ascii_case(token)
        }) {
            push_name(&skill.name, &mut names);
            continue;
        }
        // No whole-token match: check the glued form (ident prefix is a skill name).
        let mut consumed_as_skill = false;
        if let Some(pos) = token.find(|c: char| !is_ident(c)) {
            let prefix = &token[..pos];
            if let Some(skill) = skill_manifests.iter().find(|s| {
                !is_session_default_skill_action(&s.name) && s.name.eq_ignore_ascii_case(prefix)
            }) {
                push_name(&skill.name, &mut names);
                rest_tokens.push(&token[pos..]);
                consumed_as_skill = true;
            }
        }
        if !consumed_as_skill {
            rest_tokens.push(token);
        }
        // From this token on (including its glued body), everything is question text.
        rest_tokens.extend(iter.clone());
        break;
    }

    let rest = if rest_tokens.is_empty() {
        None
    } else {
        Some(rest_tokens.join(" "))
    };
    (names, rest)
}

/// Parse one input line as an embedded `/skills` selection for combined
/// multi-line sends (`/task <goal>` plus a `/skills <name>...` line, in
/// either order).
///
/// Accepts `/skills`, `:skills`, `/skill` and `:skill` heads with an optional
/// `use`/`select`/`switch` keyword. Read-only views (`list`, `current`,
/// `help`, ...), session-default commands (`pin`, `unpin`), and a bare head
/// with no names return `None` so those lines stay question text. Names come
/// back raw in input order; validation stays
/// with turn activation, which resolves each name against the manifests and
/// reports unknown ones without sinking the whole set.
pub(crate) fn parse_embedded_skills_line(line: &str) -> Option<Vec<String>> {
    let head = line
        .trim()
        .strip_prefix('/')
        .or_else(|| line.trim().strip_prefix(':'))?;
    let mut words = head.split_whitespace();
    match words.next()? {
        "skills" | "skill" => {}
        _ => return None,
    }
    let mut words = words.peekable();
    if matches!(
        words
            .peek()
            .map(|word| word.to_ascii_lowercase())
            .as_deref(),
        Some("use" | "select" | "switch")
    ) {
        words.next();
    }
    let names: Vec<String> = words
        .map(|word| {
            word.trim_matches(|c| c == ',' || c == ';')
                .trim_start_matches('@')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect();
    let first_is_reserved = names.first().is_some_and(|name| {
        is_session_default_skill_action(name)
            || matches!(
                name.to_ascii_lowercase().as_str(),
                "list" | "ls" | "current" | "cur" | "help" | "h" | "--help"
            )
    });
    if names.is_empty() || first_is_reserved {
        return None;
    }
    Some(names)
}

#[cfg(test)]
mod tests {
    use super::{
        SkillManifest, SkillPinChanges, current_skill_lines, pin_skill_names,
        resolve_implicit_selection, split_pin_trailing_question, try_handle_skills_command,
        unpin_skill_names,
    };
    use crate::ai::types::{ForcedSkillSource, PendingSkillContinuation, SkillBiasMemory};
    use std::sync::Arc;

    fn skill(name: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: String::new(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            delegation: None,
            model_invocable: true,
            mcp_servers: Vec::new(),
            disable_builtin_tools: false,
            disable_mcp_tools: false,
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            excludes: Vec::new(),
            parent: None,
            source_path: None,
            resource_path: None,
        }
    }

    fn manifests() -> Vec<SkillManifest> {
        vec![
            skill("code-review"),
            skill("docs-review"),
            skill("bytedcli"),
        ]
    }

    fn resolve(input: &str) -> (Vec<String>, Option<String>) {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        resolve_implicit_selection(&tokens, &manifests())
    }

    #[test]
    fn split_pin_trailing_question_without_newline_preserves_arguments() {
        for arguments in ["", "code docs-review", " \tcode 请检查 \t"] {
            assert_eq!(split_pin_trailing_question(arguments), (arguments, None));
        }
    }

    #[test]
    fn split_pin_trailing_question_lf_keeps_remaining_lines() {
        assert_eq!(
            split_pin_trailing_question("code\n请检查\nbytedcli 自动补全。"),
            ("code", Some("请检查\nbytedcli 自动补全。"))
        );
    }

    #[test]
    fn split_pin_trailing_question_crlf_keeps_remaining_lines() {
        assert_eq!(
            split_pin_trailing_question("code\r\n请检查\r\n后续问题"),
            ("code", Some("请检查\r\n后续问题"))
        );
    }

    #[test]
    fn split_pin_trailing_question_without_prose_has_no_question() {
        for arguments in ["code\n", "code\r\n", "code\n \t\r\n"] {
            assert_eq!(split_pin_trailing_question(arguments), ("code", None));
        }
    }

    #[test]
    fn split_pin_trailing_question_preserves_name_spacing_and_trims_question() {
        assert_eq!(
            split_pin_trailing_question(" \tcode docs-review \t\r\n \t请检查\n后续问题 \r\n"),
            (" \tcode docs-review \t", Some("请检查\n后续问题"))
        );
    }

    #[test]
    fn pin_trailing_question_dispatches_pin_and_unpin_without_consuming_prose() {
        let _env = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root =
            std::env::temp_dir().join(format!("skills-pin-question-{}", uuid::Uuid::new_v4()));
        let cases = [
            (" CODE-REVIEW code-review missing", None, true),
            (
                " CODE-REVIEW code-review missing\n bytedcli 请检查\n第二行 ",
                Some("bytedcli 请检查\n第二行"),
                true,
            ),
            (
                " CODE-REVIEW code-review missing\r\n bytedcli 请检查\r\n第二行 ",
                Some("bytedcli 请检查\r\n第二行"),
                true,
            ),
            (" CODE-REVIEW code-review missing\n \t\r\n", None, true),
            (
                "\n bytedcli 请检查\n第二行 ",
                Some("bytedcli 请检查\n第二行"),
                false,
            ),
            (
                "\r\n bytedcli 请检查\r\n第二行 ",
                Some("bytedcli 请检查\r\n第二行"),
                false,
            ),
        ];
        for action in ["pin", "unpin"] {
            for (arguments, question, has_names) in cases {
                let mut app = crate::ai::driver::tests::test_app("build");
                app.config.history_file = root.join("history.sqlite");
                app.session_id = "pin-question".to_string();
                app.session_default_skills = vec!["bytedcli".to_string()];
                if action == "unpin" {
                    app.session_default_skills.push("code-review".to_string());
                }
                let mut expected = app.session_default_skills.clone();
                if has_names {
                    expected = if action == "pin" {
                        vec!["bytedcli".to_string(), "code-review".to_string()]
                    } else {
                        vec!["bytedcli".to_string()]
                    };
                }
                app.forced_skills = vec!["docs-review".to_string()];
                app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandNextTurn);
                app.forced_question = Some("pending question".to_string());
                let mut available = Arc::new(manifests());
                let input = format!("/skills {action}{arguments}");
                assert!(try_handle_skills_command(&mut app, &input, &mut available).unwrap());
                assert_eq!(app.session_default_skills, expected, "{input}");
                assert_eq!(
                    app.forced_question.as_deref(),
                    question.or(Some("pending question")),
                    "{input}"
                );
                assert_eq!(app.forced_skills, ["docs-review"]);
                assert_eq!(
                    app.forced_skill_source,
                    Some(ForcedSkillSource::SkillsCommandNextTurn)
                );
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn pin_resolves_canonical_names_and_applies_known_subset() {
        let mut app = crate::ai::driver::tests::test_app("build");
        app.session_default_skills = vec!["CODE-REVIEW".to_string()];
        app.forced_skills = vec!["bytedcli".to_string()];
        app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandNextTurn);
        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec!["docs-review".to_string()],
        });

        let changes = pin_skill_names(
            &mut app,
            &[
                "Docs-Review",
                "docs-review",
                "Code-Review",
                "ByteDcli",
                "missing",
            ],
            &manifests(),
        );
        assert_eq!(changes.added, ["docs-review", "bytedcli"]);
        assert_eq!(changes.already_pinned, ["code-review"]);
        assert_eq!(changes.not_found, ["missing"]);
        assert_eq!(
            app.session_default_skills,
            ["CODE-REVIEW", "docs-review", "bytedcli"]
        );
        assert_eq!(app.forced_skills, ["bytedcli"]);
        assert_eq!(
            app.forced_skill_source,
            Some(ForcedSkillSource::SkillsCommandNextTurn)
        );
        assert_eq!(
            app.pending_skill_continuation.as_ref().unwrap().skill_names,
            ["docs-review"]
        );

        let repeated = pin_skill_names(
            &mut app,
            &["DOCS-REVIEW", "docs-review", "BYTEDCLI", "code-review"],
            &manifests(),
        );
        assert!(repeated.added.is_empty());
        assert_eq!(
            repeated.already_pinned,
            ["docs-review", "bytedcli", "code-review"]
        );
        assert!(repeated.not_found.is_empty());
        assert_eq!(
            app.session_default_skills,
            ["CODE-REVIEW", "docs-review", "bytedcli"]
        );
    }

    #[test]
    fn unpin_removes_case_variants_and_reports_not_pinned() {
        let mut app = crate::ai::driver::tests::test_app("build");
        app.session_default_skills = vec![
            "code-review".to_string(),
            "DOCS-REVIEW".to_string(),
            "docs-review".to_string(),
            "bytedcli".to_string(),
            "retired-skill".to_string(),
        ];
        let names = ["docs-review", "DOCS-REVIEW", "missing", "RETIRED-SKILL"];
        let changes = unpin_skill_names(&mut app, &names);
        assert_eq!(changes.removed, ["DOCS-REVIEW", "retired-skill"]);
        assert_eq!(changes.not_pinned, ["missing"]);
        assert!(!changes.agent_team_refused);
        assert_eq!(app.session_default_skills, ["code-review", "bytedcli"]);

        let repeated = unpin_skill_names(&mut app, &names);
        assert!(repeated.removed.is_empty());
        assert_eq!(
            repeated.not_pinned,
            ["docs-review", "missing", "RETIRED-SKILL"]
        );
        assert_eq!(app.session_default_skills, ["code-review", "bytedcli"]);
    }

    #[test]
    fn unpin_refuses_agent_team_and_still_removes_other_names() {
        let mut app = crate::ai::driver::tests::test_app("build");
        app.session_default_skills = vec!["AGENT-TEAM".to_string(), "code-review".to_string()];
        app.forced_skills = vec!["bytedcli".to_string()];
        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec!["docs-review".to_string()],
        });
        let changes = unpin_skill_names(&mut app, &["agent-team", "CODE-REVIEW", "AgEnT-TeAm"]);
        assert!(changes.agent_team_refused);
        assert_eq!(changes.removed, ["code-review"]);
        assert!(changes.not_pinned.is_empty());
        assert_eq!(app.session_default_skills, ["AGENT-TEAM"]);
        assert_eq!(app.forced_skills, ["bytedcli"]);
        assert_eq!(
            app.pending_skill_continuation.as_ref().unwrap().skill_names,
            ["docs-review"]
        );

        let repeated = unpin_skill_names(&mut app, &["agent-team"]);
        assert!(repeated.agent_team_refused);
        assert!(repeated.removed.is_empty());
        assert_eq!(app.session_default_skills, ["AGENT-TEAM"]);
    }

    #[test]
    fn pin_and_unpin_without_arguments_leave_runtime_state_unchanged() {
        let mut app = crate::ai::driver::tests::test_app("build");
        app.session_default_skills = vec!["code-review".to_string()];
        app.forced_skills = vec!["bytedcli".to_string()];
        app.forced_skill_source = Some(ForcedSkillSource::SkillsCommandNextTurn);
        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec!["docs-review".to_string()],
        });
        let mut available = Arc::new(vec![skill("pin"), skill("unpin"), skill("code-review")]);
        assert_eq!(
            pin_skill_names(&mut app, &[], &available),
            SkillPinChanges::default()
        );
        for command in ["/skills pin", "/skills unpin"] {
            assert!(try_handle_skills_command(&mut app, command, &mut available).unwrap());
            assert_eq!(app.session_default_skills, ["code-review"]);
            assert_eq!(app.forced_skills, ["bytedcli"]);
            assert_eq!(
                app.forced_skill_source,
                Some(ForcedSkillSource::SkillsCommandNextTurn)
            );
            assert_eq!(
                app.pending_skill_continuation.as_ref().unwrap().skill_names,
                ["docs-review"]
            );
            assert!(app.forced_question.is_none());
        }
    }

    #[test]
    fn pin_and_unpin_are_reserved_in_implicit_selection() {
        let available = vec![skill("pin"), skill("unpin"), skill("code-review")];
        for action in ["pin", "PIN", "unpin", "UNPIN", "pin请检查", "unpin请检查"] {
            let (names, rest) = resolve_implicit_selection(&[action, "code-review"], &available);
            assert!(names.is_empty(), "{action} must not select a manifest");
            assert_eq!(rest, Some(format!("{action} code-review")));
        }
        let (names, rest) =
            resolve_implicit_selection(&["code-review", "pin", "unpin"], &available);
        assert_eq!(names, ["code-review"]);
        assert_eq!(rest.as_deref(), Some("pin unpin"));
    }

    #[test]
    fn current_reports_pinned_defaults_after_existing_selection_lines() {
        let mut app = crate::ai::driver::tests::test_app("build");
        assert_eq!(current_skill_lines(&app), ["No active skill."]);
        app.session_default_skills = vec!["code-review".to_string(), "docs-review".to_string()];
        assert_eq!(
            current_skill_lines(&app),
            ["Pinned (session default): code-review, docs-review"]
        );

        app.forced_skills = vec!["bytedcli".to_string()];
        assert_eq!(
            current_skill_lines(&app),
            [
                "current skills: bytedcli (pending for next turn)",
                "Pinned (session default): code-review, docs-review",
            ]
        );
        app.forced_skills.clear();
        app.last_skill_bias = Some(SkillBiasMemory {
            skill_name: "bytedcli".to_string(),
            question: String::new(),
        });
        assert_eq!(
            current_skill_lines(&app),
            [
                "current skill: bytedcli (recent active skill)",
                "Pinned (session default): code-review, docs-review",
            ]
        );
    }

    #[test]
    fn single_skill_with_question() {
        // /skill code-review <question>
        let (names, rest) = resolve("code-review 帮我review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review"));
    }

    #[test]
    fn single_skill_no_question() {
        // /skill code-review
        let (names, rest) = resolve("code-review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest, None);
    }

    #[test]
    fn multi_skills_no_question() {
        // /skills code-review docs-review
        let (names, rest) = resolve("code-review docs-review");
        assert_eq!(names, vec!["code-review", "docs-review"]);
        assert_eq!(rest, None);
    }

    #[test]
    fn multi_skills_with_question() {
        // /skills code-review docs-review <question>
        let (names, rest) = resolve("code-review docs-review 帮我看看这段");
        assert_eq!(names, vec!["code-review", "docs-review"]);
        assert_eq!(rest.as_deref(), Some("帮我看看这段"));
    }

    #[test]
    fn duplicate_names_deduplicated() {
        // /skills code-review Code-Review → deduplicated and case-normalized
        let (names, rest) = resolve("code-review Code-Review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest, None);
    }

    #[test]
    fn unknown_token_starts_question() {
        // The first non-matching token starts the question:
        // /skills code-review <unknown> <question>
        let (names, rest) = resolve("code-review 不存在的skill 问题");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("不存在的skill 问题"));
    }

    #[test]
    fn glued_question_after_multi_skills() {
        // /skills code-review docs-review + glued question → question text glued after docs-review
        let (names, rest) = resolve("code-review docs-review帮我看看");
        assert_eq!(names, vec!["code-review", "docs-review"]);
        assert_eq!(rest.as_deref(), Some("帮我看看"));
    }

    #[test]
    fn glued_question_with_mixed_case() {
        // /skill Code-Review + glued question → the remainder after stripping the glued name
        let (names, rest) = resolve("Code-Review帮我review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review"));
    }

    #[test]
    fn question_repeating_skill_name() {
        // Duplicate skill names are deduplicated; the glued suffix is preserved once.
        let (names, rest) = resolve("code-review code-review帮我review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review"));
    }

    #[test]
    fn multi_word_question() {
        // /skill code-review <multi-word question>
        let (names, rest) = resolve("code-review 帮我review 这段代码 并给出建议");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review 这段代码 并给出建议"));
    }

    #[test]
    fn multiline_question() {
        // /skills bytedcli <question>
        let (names, rest) = resolve("bytedcli 帮我检查最近的飞书消息");
        assert_eq!(names, vec!["bytedcli"]);
        assert_eq!(rest.as_deref(), Some("帮我检查最近的飞书消息"));
    }

    #[test]
    fn unknown_subcommand_no_names() {
        // No skill matches: the resolver keeps the token as question text, while
        // the command rejects the selection because the name list is empty.
        let (names, rest) = resolve("fizzbuzz");
        assert!(names.is_empty());
        assert_eq!(rest.as_deref(), Some("fizzbuzz"));
    }

    #[test]
    fn embedded_skills_line_shapes() {
        let parse = super::parse_embedded_skills_line;
        assert_eq!(
            parse("/skills code-review docs-review").unwrap(),
            vec!["code-review", "docs-review"]
        );
        assert_eq!(
            parse("  :skills use code-review  ").unwrap(),
            vec!["code-review"]
        );
        assert_eq!(parse("/skill code-review").unwrap(), vec!["code-review"]);
        // Read-only views and bare heads are not selections: they stay text.
        for line in [
            "/skills",
            "/skills list",
            "/skills current",
            "/skills help",
            "/skills use",
            "/skills pin",
            "/skills pin code-review",
            ":skills PIN code-review",
            "/skills unpin code-review",
            "just some prose",
            "/task fix bug",
        ] {
            assert_eq!(parse(line), None, "{line:?} must not parse as a selection");
        }
    }
}
