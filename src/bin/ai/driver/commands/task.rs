use colored::Colorize;

use crate::ai::{driver::task, history::SessionStore, types::App};

/// Builds the host prompt that drives the multi-worker task room after
/// `/task` sets it as the forced question (same mechanism as `/goal`).
/// A bare `/task <goal>` leaves the team size to the host: it decomposes the
/// goal first, then passes its chosen count to `task_start`.
pub(crate) fn build_task_host_prompt(goal: &str, worker_count: Option<usize>) -> String {
    let team_size = match worker_count {
        Some(n) => format!(
            "{n} workers (fixed on the command line — pass worker_count={n} to task_start)"
        ),
        None => "undecided — decompose the goal and choose 1-8 workers yourself, \
             then pass your count to task_start"
            .to_string(),
    };
    format!(
        include_str!("prompts/task_host_initial.md"),
        goal = goal,
        team_size = team_size
    )
}

/// Parsed `/task` invocation.
#[derive(Debug, PartialEq, Eq)]
enum TaskCommand {
    Start { goal: String, worker_count: Option<usize> },
    Status { room: Option<String> },
}

/// Bare host-tool names that must not silently become a goal: a lone
/// `collect`/`close`/… is tool-name confusion, not a one-word goal.
const TOOL_NAME_WORDS: &[&str] = &["start", "assign", "collect", "broadcast", "close", "end"];

/// Parse a `/task` (or `:task`) invocation. `None` when `input` is not
/// a task command at all; `Some(Err)` carries a usage message to print.
/// A leading `status` selects the read-only progress view; otherwise a
/// leading integer 1-8 is the worker count and the rest is the goal.
fn parse_task_command(input: &str) -> Option<Result<TaskCommand, String>> {
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix("/task")
        .or_else(|| trimmed.strip_prefix(":task"))?;
    // Word boundary: `/tasks` is not `/task`.
    let rest = match rest.chars().next() {
        None => String::new(),
        Some(c) if c.is_whitespace() => rest.trim().to_string(),
        Some(_) => return None,
    };
    if rest.is_empty() {
        return Some(Err(
            "Usage: /task [workers] <goal> | /task status [room]".to_string(),
        ));
    }
    if rest == "status" {
        return Some(Ok(TaskCommand::Status { room: None }));
    }
    if let Some(room) = rest.strip_prefix("status ") {
        let room = room.trim();
        return Some(Ok(TaskCommand::Status {
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
             Say what you want in words, or run /task status [room] to view progress."
        )));
    }
    let mut words = rest.split_whitespace();
    let first = words.next().unwrap_or("");
    if let Ok(count) = first.parse::<usize>() {
        let goal: String = words.collect::<Vec<_>>().join(" ");
        if goal.is_empty() {
            return Some(Err(
                "Usage: /task [workers] <goal> — a goal is required.".to_string(),
            ));
        }
        if count == 0 || count > task::MAX_TASK_WORKERS {
            return Some(Err(format!(
                "Worker count must be 1-{}.",
                task::MAX_TASK_WORKERS
            )));
        }
        return Some(Ok(TaskCommand::Start {
            goal,
            worker_count: Some(count),
        }));
    }
    Some(Ok(TaskCommand::Start {
        goal: rest,
        worker_count: None,
    }))
}

/// Handles the `/task` interactive command. See the parse function for the
/// grammar; starting a task sets the forced question to the host prompt
/// and lets the turn run the `task_*` tools, with the host narrating each
/// round in prose so progress stays visible in this session.
pub fn try_handle_task_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(parsed) = parse_task_command(input) else {
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
        TaskCommand::Start {
            goal,
            worker_count,
        } => {
            let prompt = build_task_host_prompt(&goal, worker_count);
            app.forced_question = Some(prompt);
            match worker_count {
                Some(n) => println!(
                    "{} Splitting the work across {} workers. Goal: {}",
                    "[task]".cyan().bold(),
                    n,
                    goal
                ),
                None => println!(
                    "{} Goal: {} (you decide the team size)",
                    "[task]".cyan().bold(),
                    goal
                ),
            }
            Ok(true)
        }
        TaskCommand::Status { room } => {
            let store = SessionStore::new(app.config.history_file.as_path());
            let history_file = store.session_history_file(&app.session_id);
            let room = match room {
                Some(room) => room,
                None => {
                    let rooms = task::list_rooms(&history_file);
                    match rooms.len() {
                        0 => {
                            println!(
                                "{} This session owns no task rooms yet. Start one with /task <goal>.",
                                "[task]".dimmed()
                            );
                            return Ok(true);
                        }
                        1 => rooms.into_iter().next().unwrap_or_default(),
                        _ => {
                            println!(
                                "{} Multiple rooms: {}. Rerun with /task status <room>.",
                                "[task]".yellow(),
                                rooms.join(", ")
                            );
                            return Ok(true);
                        }
                    }
                }
            };
            match task::room_status(&history_file, &room) {
                Ok(st) => println!("{}", task::format_room_status(&st)),
                Err(error) => println!("{} {error}", "[task]".red()),
            }
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_task_input_is_not_claimed() {
        assert!(parse_task_command("hello world").is_none());
        assert!(parse_task_command("/help").is_none());
        // Word boundary: `/tasks` is a different word.
        assert!(parse_task_command("/tasks x").is_none());
        // The discuss command is not claimed either.
        assert!(parse_task_command("/discuss x").is_none());
    }

    #[test]
    fn bare_task_reports_usage() {
        assert!(parse_task_command("/task").unwrap().is_err());
        assert!(parse_task_command("  :task  ").unwrap().is_err());
    }

    #[test]
    fn bare_goal_leaves_team_size_to_host() {
        assert_eq!(
            parse_task_command("/task fix login retry").unwrap().unwrap(),
            TaskCommand::Start {
                goal: "fix login retry".to_string(),
                worker_count: None,
            }
        );
    }

    #[test]
    fn leading_count_sets_workers() {
        assert_eq!(
            parse_task_command("/task 2 fix login retry")
                .unwrap()
                .unwrap(),
            TaskCommand::Start {
                goal: "fix login retry".to_string(),
                worker_count: Some(2),
            }
        );
    }

    #[test]
    fn count_without_goal_or_out_of_range_is_usage_error() {
        assert!(parse_task_command("/task 3").unwrap().is_err());
        assert!(parse_task_command("/task 0 x").unwrap().is_err());
        assert!(parse_task_command("/task 9 x").unwrap().is_err());
    }

    #[test]
    fn status_forms_parse() {
        assert_eq!(
            parse_task_command("/task status").unwrap().unwrap(),
            TaskCommand::Status { room: None }
        );
        assert_eq!(
            parse_task_command("/task status my-room").unwrap().unwrap(),
            TaskCommand::Status {
                room: Some("my-room".to_string())
            }
        );
    }

    #[test]
    fn bare_tool_names_are_usage_errors_not_goals() {
        for word in ["start", "assign", "collect", "broadcast", "close", "end"] {
            let error = parse_task_command(&format!("/task {word}"))
                .unwrap()
                .unwrap_err();
            assert!(
                error.contains("/task status"),
                "unexpected hint for {word}: {error}"
            );
        }
        // Multi-word goals starting with a tool name still parse as goals.
        assert_eq!(
            parse_task_command("/task collect the results")
                .unwrap()
                .unwrap(),
            TaskCommand::Start {
                goal: "collect the results".to_string(),
                worker_count: None,
            }
        );
    }
}
