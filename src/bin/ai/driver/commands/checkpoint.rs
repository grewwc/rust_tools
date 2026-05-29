//! `/checkpoint`（别名 `/cp`）交互命令：保存 / 列出 / 回滚 / 删除当前
//! session 的对话历史检查点。

use crate::ai::{history::CheckpointStore, types::App};

pub fn try_handle_checkpoint_command(
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
    if cmd != "checkpoint" && cmd != "cp" {
        return Ok(false);
    }

    let action = parts.next().unwrap_or("list");
    let store = CheckpointStore::new(app.config.history_file.as_path(), &app.session_id);

    match action {
        "help" | "h" => {
            print_help();
        }
        "save" | "s" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint save <name>");
                return Ok(true);
            };
            match store.save(name) {
                Ok(_) => println!("Saved checkpoint '{name}'."),
                Err(err) => println!("Failed to save checkpoint: {err}"),
            }
        }
        "list" | "ls" | "" => {
            let checkpoints = store.list()?;
            if checkpoints.is_empty() {
                println!("No checkpoints for the current session.");
            } else {
                println!("Checkpoints (current session):");
                for cp in checkpoints {
                    let when = cp
                        .modified_local
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "-".to_string());
                    println!("  {:<24} {:>8} bytes  {}", cp.name, cp.size_bytes, when);
                }
            }
        }
        "rollback" | "restore" | "rb" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint rollback <name>");
                return Ok(true);
            };
            match store.rollback(name) {
                Ok(()) => println!(
                    "Rolled back current session to checkpoint '{name}'. \
                     New turns will continue from this restored history."
                ),
                Err(err) => println!("Failed to rollback: {err}"),
            }
        }
        "delete" | "del" | "rm" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint delete <name>");
                return Ok(true);
            };
            match store.delete(name) {
                Ok(true) => println!("Deleted checkpoint '{name}'."),
                Ok(false) => println!("Checkpoint '{name}' does not exist."),
                Err(err) => println!("Failed to delete checkpoint: {err}"),
            }
        }
        other => {
            println!("Unknown checkpoint action: '{other}'");
            print_help();
        }
    }
    Ok(true)
}

fn print_help() {
    println!("Checkpoint commands (current session):");
    println!();
    println!("  /checkpoint save <name>       save current history as a checkpoint");
    println!("  /checkpoint list              list checkpoints (default)");
    println!("  /checkpoint rollback <name>   restore history from a checkpoint");
    println!("  /checkpoint delete <name>     delete a checkpoint");
    println!();
    println!("  Alias: /cp");
}
