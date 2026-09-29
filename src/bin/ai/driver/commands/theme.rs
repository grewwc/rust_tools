//! `/theme` — live terminal theme switching.
//!
//! The switch is session-scoped and applies immediately: it sets an in-memory
//! override in `theme.rs` (which also survives the turn-boundary `refresh()`),
//! so the next terminal output already uses the new palette. Nothing is
//! written to the config file — set `ai.theme` in config to persist a choice.
use std::{error::Error, fs};

use crate::ai::{config_schema::AiConfig, theme};
use crate::commonw::{configw, utils::expanduser};

/// Subcommand literals offered by tab completion (`/theme <TAB>`); theme names
/// complete alongside these (see `completion.rs`).
pub(crate) const THEME_SUBCOMMANDS: &[&str] = &["list", "current", "clear", "file", "help"];

/// Handle `/theme` / `:theme`. Returns `Ok(false)` when the input is a
/// different command; every `/theme` form prints its own result and returns
/// `Ok(true)` (unknown names print the list rather than falling through).
pub fn try_handle_theme_command(input: &str) -> Result<bool, Box<dyn Error>> {
    let trimmed = input.trim();
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
    if cmd != "theme" {
        return Ok(false);
    }
    let arg = normalized[cmd.len()..].trim();
    if arg.is_empty() || matches!(arg, "list" | "ls") {
        print_theme_list();
        return Ok(true);
    }
    if matches!(arg, "help" | "h") {
        print_theme_help();
        return Ok(true);
    }
    if matches!(arg, "current" | "cur") {
        print_current_theme();
        return Ok(true);
    }
    if matches!(arg, "clear" | "reset") {
        theme::clear_override();
        println!(
            "Theme override cleared — back to config ({}).",
            config_theme_source()
        );
        return Ok(true);
    }
    // `/theme file <path>`: load a custom theme JSON for this session.
    let mut words = arg.split_whitespace();
    if words.next() == Some("file") {
        let path = arg["file".len()..].trim();
        if path.is_empty() {
            println!("Usage: /theme file <path-to-theme.json>");
            return Ok(true);
        }
        return apply_file(path);
    }
    apply_name(arg.split_whitespace().next().unwrap_or(arg))
}

fn apply_name(name: &str) -> Result<bool, Box<dyn Error>> {
    match theme::theme_from_name(name) {
        Some(resolved) => {
            theme::set_override(name.to_string(), resolved);
            println!("Switched theme: {name} (session-only; /theme clear to revert).");
        }
        None => {
            println!("Unknown theme: {name}");
            print_theme_list();
        }
    }
    Ok(true)
}

fn apply_file(path: &str) -> Result<bool, Box<dyn Error>> {
    let expanded = expanduser(path);
    match fs::read_to_string(expanded.as_ref()) {
        Ok(content) => match theme::Theme::from_json(&content) {
            Some(resolved) => {
                theme::set_override(format!("file:{path}"), resolved);
                println!("Switched theme from file: {path} (session-only).");
            }
            None => println!("Not a valid theme JSON: {path}"),
        },
        Err(err) => println!("Cannot read theme file {path}: {err}"),
    }
    Ok(true)
}

/// Label of the currently active theme: the live override when set, otherwise
/// the config source.
fn active_label() -> String {
    theme::override_label().unwrap_or_else(config_theme_source)
}

fn config_theme_source() -> String {
    let cfg = configw::get_all_config();
    if let Some(path) = cfg.get_opt(AiConfig::THEME_FILE) {
        return format!("file:{path}");
    }
    let name = cfg.get_opt(AiConfig::THEME).unwrap_or_default();
    if name.is_empty() {
        "default".to_string()
    } else {
        name
    }
}

fn print_theme_list() {
    let active = active_label();
    println!("Available themes (session-only switch via /theme <name>):");
    for name in theme::available_theme_names() {
        if name == active {
            println!("  {name} (active)");
        } else {
            println!("  {name}");
        }
    }
}

fn print_current_theme() {
    println!("Current theme: {}", active_label());
}

fn print_theme_help() {
    println!("Theme commands:");
    println!();
    println!("  /theme                    list available themes");
    println!("  /theme <name>             switch theme live (session-only)");
    println!("  /theme file <path.json>   load a custom theme file (session-only)");
    println!("  /theme current            show current theme");
    println!("  /theme clear              drop the session override, back to config");
    println!();
    println!("  Set ai.theme in config to persist a choice across sessions.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    #[test]
    fn theme_command_ignores_other_commands() {
        assert!(!try_handle_theme_command("/model foo").unwrap());
        assert!(!try_handle_theme_command("theme monokai").unwrap());
        assert!(!try_handle_theme_command("/themes").unwrap());
        assert!(!try_handle_theme_command("").unwrap());
    }

    #[test]
    fn theme_command_lists_on_bare_and_list() {
        assert!(try_handle_theme_command("/theme").unwrap());
        assert!(try_handle_theme_command("/theme list").unwrap());
        assert!(try_handle_theme_command(":theme help").unwrap());
        assert!(try_handle_theme_command("/theme current").unwrap());
    }

    #[test]
    fn theme_command_switches_live_and_clears() {
        let _guard = env_guard();
        theme::clear_override();
        // Snapshot the config-resolved palette (whatever the ambient config
        // is) so the test never depends on the machine's `~/.configW`.
        let before = theme::current().clone();
        let monokai = theme::theme_from_name("monokai").expect("monokai must parse");

        assert!(try_handle_theme_command("/theme monokai").unwrap());
        assert_eq!(theme::override_label().as_deref(), Some("monokai"));
        assert_eq!(*theme::current(), monokai);

        assert!(try_handle_theme_command("/theme clear").unwrap());
        assert_eq!(theme::override_label(), None);
        assert_eq!(*theme::current(), before);
    }

    #[test]
    fn theme_command_rejects_unknown_name_without_changing_active() {
        let _guard = env_guard();
        theme::clear_override();
        assert!(try_handle_theme_command("/theme monokai").unwrap());
        let active = theme::current().code_background;

        assert!(try_handle_theme_command("/theme no-such-theme").unwrap());
        assert_eq!(theme::override_label().as_deref(), Some("monokai"));
        assert_eq!(theme::current().code_background, active);

        theme::clear_override();
    }

    #[test]
    fn theme_command_rejects_missing_and_unreadable_file() {
        let _guard = env_guard();
        theme::clear_override();
        assert!(try_handle_theme_command("/theme file").unwrap());
        assert!(try_handle_theme_command("/theme file /no/such/theme.json").unwrap());
        assert_eq!(theme::override_label(), None);
    }
}
