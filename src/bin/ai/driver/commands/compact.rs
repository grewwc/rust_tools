use crate::ai::{history, types::App};

pub(crate) fn compact_command_args(input: &str) -> Option<&str> {
    let command = input.trim().strip_prefix('/').or_else(|| input.trim().strip_prefix(':'))?;
    let (name, args) = command.split_once(char::is_whitespace).unwrap_or((command, ""));
    (name == "compact").then_some(args.trim())
}

/// Dispatch before the normal model turn, with session context established.
/// This async entry point must not be nested in the synchronous local dispatcher.
pub(crate) async fn try_handle_compact_command(
    app: &App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(args) = compact_command_args(input) else {
        return Ok(false);
    };
    if !args.is_empty() {
        println!("Usage: /compact (no arguments). Compacts context, not canonical history.");
        return Ok(true);
    }
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()?;
    let result = history::compact_session_history_manually_with_app(app, Some(&cwd)).await?;
    if result.persisted {
        println!("Context compacted: {} -> {} chars (summary inserted: {}). Canonical history unchanged.",
            result.before_chars, result.after_chars, result.summary_inserted);
    } else {
        println!("No smaller safely archived context available ({} chars). History unchanged.", result.before_chars);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_command_requires_exact_name_and_preserves_argument_validation() {
        assert_eq!(compact_command_args("  /compact  "), Some(""));
        assert_eq!(compact_command_args(":compact"), Some(""));
        assert_eq!(compact_command_args("/compact\tunexpected"), Some("unexpected"));
        for input in ["compact", "/compaction", "/compact-more", "please /compact"] {
            assert_eq!(compact_command_args(input), None);
        }
    }
}