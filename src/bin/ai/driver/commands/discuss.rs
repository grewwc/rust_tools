use colored::Colorize;

use crate::ai::{driver::discuss, history::SessionStore, types::App};

use super::{apply_embedded_skills, split_combined_command};

/// Default member count for a bare `/discuss <topic>`.
const DEFAULT_DISCUSS_MEMBERS: usize = 3;

/// Builds the host prompt that drives the multi-session discussion after
/// `/discuss` sets it as the forced question (same mechanism as `/goal`).
pub(crate) fn build_discuss_host_prompt(topic: &str, member_count: usize) -> String {
    format!(
        include_str!("prompts/discuss_host_initial.md"),
        topic = topic,
        member_count = member_count
    )
}

/// Parsed `/discuss` invocation.
#[derive(Debug, PartialEq, Eq)]
enum DiscussCommand {
    Start { topic: String, member_count: usize },
    Status { room: Option<String> },
    Help,
}

/// Detailed help for `/discuss` (also reachable via `/help discuss`).
pub(crate) fn print_discuss_help() {
    println!("Host a multi-session debate across worker sessions.");
    println!("  /discuss [N] <topic>      start a debate (1-8 members, default 3)");
    println!("  /discuss status [room]    show discussion progress (read-only)");
    println!("  /discuss help             show this help");
    println!("Combine with skills in one send: put /discuss first, /skills lines after.");
}

/// Bare host-tool names that must not silently become a topic: a lone
/// `collect`/`broadcast`/… is tool-name confusion, not a one-word topic.
const TOOL_NAME_WORDS: &[&str] = &["start", "assign", "collect", "broadcast", "close", "end"];

/// Parse a `/discuss` (or `:discuss`) invocation. `None` when `input` is not
/// a discuss command at all; `Some(Err)` carries a usage message to print.
/// A leading `status` selects the read-only progress view; otherwise a
/// leading integer 1-8 is the member count and the rest is the topic.
fn parse_discuss_command(input: &str) -> Option<Result<DiscussCommand, String>> {
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix("/discuss")
        .or_else(|| trimmed.strip_prefix(":discuss"))?;
    // Word boundary: `/discussion` is not `/discuss`.
    let rest = match rest.chars().next() {
        None => String::new(),
        Some(c) if c.is_whitespace() => rest.trim().to_string(),
        Some(_) => return None,
    };
    if rest.is_empty() {
        return Some(Err(
            "Usage: /discuss [members] <topic> | /discuss status [room]".to_string(),
        ));
    }
    if matches!(rest.as_str(), "help" | "h" | "--help") {
        return Some(Ok(DiscussCommand::Help));
    }
    if rest == "status" {
        return Some(Ok(DiscussCommand::Status { room: None }));
    }
    if let Some(room) = rest.strip_prefix("status ") {
        let room = room.trim();
        return Some(Ok(DiscussCommand::Status {
            room: if room.is_empty() {
                None
            } else {
                Some(room.to_string())
            },
        }));
    }
    if TOOL_NAME_WORDS.contains(&rest.as_str()) {
        return Some(Err(format!(
            "'{rest}' is a host tool, not a slash command — tools run inside the turn. \
             Say what you want in words, or run /discuss status [room] to view progress."
        )));
    }
    let mut words = rest.split_whitespace();
    let first = words.next().unwrap_or("");
    if let Ok(count) = first.parse::<usize>() {
        let topic: String = words.collect::<Vec<_>>().join(" ");
        if topic.is_empty() {
            return Some(Err(
                "Usage: /discuss [members] <topic> — a topic is required.".to_string(),
            ));
        }
        if count == 0 || count > discuss::MAX_DISCUSS_MEMBERS {
            return Some(Err(format!(
                "Member count must be 1-{}.",
                discuss::MAX_DISCUSS_MEMBERS
            )));
        }
        return Some(Ok(DiscussCommand::Start {
            topic,
            member_count: count,
        }));
    }
    Some(Ok(DiscussCommand::Start {
        topic: rest,
        member_count: DEFAULT_DISCUSS_MEMBERS,
    }))
}

/// Handles the `/discuss` interactive command. See the parse function for the
/// grammar; starting a discussion sets the forced question to the host prompt
/// and lets the turn run the `discuss_*` tools, with the host narrating each
/// round in prose so progress stays visible in this session.
pub fn try_handle_discuss_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Combined send: `/discuss <topic>` plus `/skills <name>...` lines in
    // either order. Discuss stays the main command; the picked skills ride
    // the host turn.
    let Some((main_input, embedded)) = split_combined_command(input, &["/discuss", ":discuss"])
    else {
        return Ok(false);
    };
    let Some(parsed) = parse_discuss_command(&main_input) else {
        return Ok(false);
    };
    let cmd = match parsed {
        Ok(cmd) => cmd,
        Err(usage) => {
            println!("{usage}");
            return Ok(true);
        }
    };
    match cmd {
        DiscussCommand::Start {
            topic,
            member_count,
        } => {
            let prompt = build_discuss_host_prompt(&topic, member_count);
            app.forced_question = Some(prompt);
            apply_embedded_skills(app, &embedded, true);
            println!(
                "{} Hosting a discussion with {} members. Topic: {}",
                "[discuss]".cyan().bold(),
                member_count,
                topic
            );
            Ok(true)
        }
        DiscussCommand::Status { room } => {
            apply_embedded_skills(app, &embedded, false);
            let store = SessionStore::new(app.config.history_file.as_path());
            let history_file = store.session_history_file(&app.session_id);
            let room = match room {
                Some(room) => room,
                None => {
                    let rooms = discuss::list_rooms(&history_file);
                    match rooms.len() {
                        0 => {
                            println!(
                                "{} This session owns no discussion rooms yet. Start one with /discuss <topic>.",
                                "[discuss]".dimmed()
                            );
                            return Ok(true);
                        }
                        1 => rooms.into_iter().next().unwrap_or_default(),
                        _ => {
                            println!(
                                "{} Multiple rooms: {}. Rerun with /discuss status <room>.",
                                "[discuss]".yellow(),
                                rooms.join(", ")
                            );
                            return Ok(true);
                        }
                    }
                }
            };
            match discuss::room_status(&history_file, &room) {
                Ok(st) => println!("{}", discuss::format_room_status(&st)),
                Err(error) => println!("{} {error}", "[discuss]".red()),
            }
            Ok(true)
        }
        DiscussCommand::Help => {
            apply_embedded_skills(app, &embedded, false);
            print_discuss_help();
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_discuss_input_is_not_claimed() {
        assert!(parse_discuss_command("hello world").is_none());
        assert!(parse_discuss_command("/help").is_none());
        // Word boundary: `/discussion` is a different word.
        assert!(parse_discuss_command("/discussion x").is_none());
    }

    #[test]
    fn bare_discuss_reports_usage() {
        assert!(parse_discuss_command("/discuss").unwrap().is_err());
        assert!(parse_discuss_command("  :discuss  ").unwrap().is_err());
    }

    #[test]
    fn bare_topic_uses_default_member_count() {
        assert_eq!(
            parse_discuss_command("/discuss tabs vs spaces").unwrap().unwrap(),
            DiscussCommand::Start {
                topic: "tabs vs spaces".to_string(),
                member_count: DEFAULT_DISCUSS_MEMBERS,
            }
        );
    }

    #[test]
    fn leading_count_sets_members() {
        assert_eq!(
            parse_discuss_command("/discuss 2 tabs vs spaces")
                .unwrap()
                .unwrap(),
            DiscussCommand::Start {
                topic: "tabs vs spaces".to_string(),
                member_count: 2,
            }
        );
    }

    #[test]
    fn count_without_topic_or_out_of_range_is_usage_error() {
        assert!(parse_discuss_command("/discuss 3").unwrap().is_err());
        assert!(parse_discuss_command("/discuss 0 x").unwrap().is_err());
        assert!(parse_discuss_command("/discuss 9 x").unwrap().is_err());
    }

    #[test]
    fn status_forms_parse() {
        assert_eq!(
            parse_discuss_command("/discuss status").unwrap().unwrap(),
            DiscussCommand::Status { room: None }
        );
        assert_eq!(
            parse_discuss_command("/discuss status myroom")
                .unwrap()
                .unwrap(),
            DiscussCommand::Status {
                room: Some("myroom".to_string())
            }
        );
    }

    #[test]
    fn host_prompt_carries_topic_and_count() {
        let prompt = build_discuss_host_prompt("tabs vs spaces", 2);
        assert!(prompt.contains("tabs vs spaces"));
        assert!(prompt.contains("2"));
        assert!(prompt.contains("discuss_start"));
    }

    #[test]
    fn bare_tool_names_are_usage_errors_not_topics() {
        for word in ["start", "assign", "collect", "broadcast", "close", "end"] {
            let error = parse_discuss_command(&format!("/discuss {word}"))
                .unwrap()
                .unwrap_err();
            assert!(
                error.contains("/discuss status"),
                "unexpected hint for {word}: {error}"
            );
        }
        // Multi-word topics starting with a tool name still parse as topics.
        assert_eq!(
            parse_discuss_command("/discuss collect pros and cons")
                .unwrap()
                .unwrap(),
            DiscussCommand::Start {
                topic: "collect pros and cons".to_string(),
                member_count: DEFAULT_DISCUSS_MEMBERS,
            }
        );
    }

    fn test_app() -> App {
        use std::sync::{Arc, atomic::AtomicBool};
        crate::ai::types::App {
            cli: crate::ai::cli::ParsedCli::default(),
            scoped_preflight_required: Vec::new(),
            config: crate::ai::types::AppConfig {
                api_key: String::new(),
                base_history_file: std::path::PathBuf::new(),
                history_file: std::path::PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 8000,
                history_keep_last: 10,
                history_summary_max_chars: 4000,
                intent_model: None,
            },
            session_id: "test".to_string(),
            session_history_file: std::path::PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: "test".to_string(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: Vec::new(),
            forced_skill_source: None,
            pending_skill_continuation: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: Vec::new(),
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            turn_tool_errors: Default::default(),
            stale_patch_targets: Default::default(),
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
        }
    }

    #[test]
    fn combined_send_selects_skills_on_start() {
        use crate::ai::types::ForcedSkillSource;
        for input in [
            "/discuss tabs vs spaces\n/skills s1 s2",
            "/skills use s1 s2\n/discuss tabs vs spaces",
        ] {
            let mut app = test_app();
            assert!(try_handle_discuss_command(&mut app, input).unwrap());
            let question = app.forced_question.expect("start must set a host prompt");
            assert!(question.contains("tabs vs spaces"), "topic lost: {question}");
            assert!(!question.contains("s1"), "skills line leaked: {question}");
            assert_eq!(app.forced_skills, vec!["s1", "s2"]);
            assert_eq!(
                app.forced_skill_source,
                Some(ForcedSkillSource::SkillsCommandInline)
            );
        }
    }

    #[test]
    fn combined_status_keeps_skills_for_next_turn() {
        use crate::ai::types::ForcedSkillSource;
        let mut app = test_app();
        assert!(try_handle_discuss_command(&mut app, "/discuss status\n/skills s1").unwrap());
        assert!(app.forced_question.is_none());
        assert_eq!(app.forced_skills, vec!["s1"]);
        assert_eq!(
            app.forced_skill_source,
            Some(ForcedSkillSource::SkillsCommandNextTurn)
        );
    }

    #[test]
    fn prose_containing_discuss_is_not_claimed() {
        let mut app = test_app();
        assert!(!try_handle_discuss_command(&mut app, "hello\n/discuss x").unwrap());
        assert!(app.forced_question.is_none());
        assert!(app.forced_skills.is_empty());
    }
}