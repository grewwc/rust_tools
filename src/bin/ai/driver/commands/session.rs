use uuid::Uuid;

use crate::ai::{history::SessionStore, types::App};

pub fn try_handle_session_command(
    app: &mut App,
    input: &str,
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
    if cmd != "sessions" && cmd != "session" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();

    match action {
        "help" | "h" => {
            println!("sessions commands:");
            println!("  /sessions");
            println!("  /sessions list");
            println!("  /sessions current");
            println!("  /sessions new");
            println!("  /sessions use <id>");
            println!("  /sessions delete <id>");
            println!("  /sessions clear-all");
            println!("  /sessions export <id> [output.md]");
            println!("  /sessions export-current [output.md]");
            println!("  /sessions export-last [output.md]");
        }
        "list" | "ls" => {
            let sessions = store.list_sessions()?;
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                for s in sessions {
                    let mark = if s.id == app.session_id { "*" } else { " " };
                    let time = s
                        .modified_local
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let prompt = s
                        .first_user_prompt
                        .as_deref()
                        .map(sanitize_session_prompt)
                        .filter(|v| !v.is_empty())
                        .unwrap_or_else(|| "-".to_string());
                    let prompt = truncate_session_prompt(&prompt, 80);
                    println!(
                        "{mark} {:<36}  {}  {:>8}B  {}",
                        s.id, time, s.size_bytes, prompt
                    );
                }
            }
        }
        "current" | "cur" => {
            println!("session: {}", app.session_id);
            println!("history: {}", app.session_history_file.display());
            let first = store.first_user_prompt(&app.session_id).unwrap_or(None);
            if let Some(v) = first {
                let prompt = sanitize_session_prompt(&v);
                if !prompt.is_empty() {
                    println!("first: {}", truncate_session_prompt(&prompt, 160));
                }
            }
        }
        "new" | "create" => {
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Switched to new session: {}", new_id);
        }
        "use" | "select" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions use <id>");
                return Ok(true);
            };
            app.session_id = id.to_string();
            app.session_history_file = store.session_history_file(id);
            println!("Switched session: {}", id);
            let first = store.first_user_prompt(id).unwrap_or(None);
            if let Some(v) = first {
                let prompt = sanitize_session_prompt(&v);
                if !prompt.is_empty() {
                    println!("first: {}", truncate_session_prompt(&prompt, 160));
                }
            }
        }
        "delete" | "del" | "rm" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions delete <id>");
                return Ok(true);
            };
            let deleted = store.delete_session(id)?;
            if deleted {
                if id == app.session_id {
                    let new_id = Uuid::new_v4().to_string();
                    app.session_id = new_id.clone();
                    app.session_history_file = store.session_history_file(&new_id);
                    println!(
                        "Deleted current session. Switched to new session: {}",
                        new_id
                    );
                } else {
                    println!("Deleted session: {}", id);
                }
            } else {
                println!("Session not found: {}", id);
            }
        }
        "export" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions export <id> [output.md]");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(id, output_path) {
                Ok(()) => {
                    println!("Exported session '{}' to '{}'", id, output_path.display());
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-current" | "export-cur" => {
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&app.session_id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported current session '{}' to '{}'",
                        app.session_id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-last" | "export-latest" => {
            let sessions = store.list_sessions()?;
            let Some(last) = sessions.first() else {
                println!("No sessions found to export.");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&last.id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported latest session '{}' to '{}'",
                        last.id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "clear-all" | "clear_all" | "clear" | "wipe" => {
            let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(
                "Delete ALL sessions? (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            let deleted = store.clear_all_sessions()?;
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Deleted {deleted} session(s). Switched to new session: {new_id}");
        }
        _ => {
            println!("unknown action: {}. try: /sessions help", action);
        }
    }
    Ok(true)
}

fn sanitize_session_prompt(s: &str) -> String {
    let mut out = String::new();
    let mut last_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
}

fn truncate_session_prompt(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("...");
    out
}
