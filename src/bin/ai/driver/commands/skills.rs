use std::sync::Arc;

use crate::ai::{skills::SkillManifest, types::App};

fn pending_skill_name(app: &App) -> Option<&str> {
    app.forced_skill.as_deref()
}

fn recent_skill_name(app: &App) -> Option<&str> {
    app.last_skill_bias
        .as_ref()
        .map(|memory| memory.skill_name.as_str())
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
    // 先把 normalized 拷出来，后续可能需要提取 "skills <name>" 之后的部分
    let normalized_for_rest = normalized.to_string();
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "skills" && cmd != "skill" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");

    match action {
        "help" | "h" | "--help" => {
            println!("Skill management commands:");
            println!();
            println!("  /skills              list all available skills");
            println!("  /skills list         list all available skills");
            println!("  /skills current      show pending/recent skill selection");
            println!("  /skills use <name>   force the specified skill on the next turn");
            println!("  /skills help         show this help");
            println!();
        }
        "list" | "ls" | "" => {
            let skills = &**skill_manifests;
            let pending = pending_skill_name(app);
            let recent = recent_skill_name(app);
            if skills.is_empty() {
                println!("No skills available.");
            } else {
                for s in skills {
                    let is_pending = pending.is_some_and(|name| name.eq_ignore_ascii_case(&s.name));
                    let is_recent = recent.is_some_and(|name| name.eq_ignore_ascii_case(&s.name));
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
                    if extras.is_empty() {
                        println!("{}  {}", mark, s.name);
                    } else {
                        println!("{}  {}  {}", mark, s.name, extras.join(" "));
                    }
                }
            }
        }
        "current" | "cur" => {
            if let Some(name) = pending_skill_name(app) {
                println!("current skill: {name} (pending for next turn)");
            } else if let Some(name) = recent_skill_name(app) {
                println!("current skill: {name} (recent active skill)");
            } else {
                println!("No active skill.");
            }
        }
        "use" | "select" | "switch" => {
            let Some(skill_name) = parts.next() else {
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
            };

            // 查找 skill（大小写不敏感）
            let found = (**skill_manifests)
                .iter()
                .find(|s| s.name.eq_ignore_ascii_case(skill_name))
                .map(|s| s.name.clone());

            match found {
                Some(name) => {
                    app.forced_skill = Some(name.clone());
                    println!("Skill selected for next turn: {name}");
                    println!("Ask your next question naturally, or mention @skills:{name} inline.");
                }
                None => {
                    println!("Skill '{skill_name}' not found.");
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
                }
            }
        }
        // 隐式选择：输入 /skills <skillname> 直接应用，无需 use 关键字
        _ => {
            let action = action; // 重新绑定，避免 match 吃掉 action

            // 检查是否有更多参数（即 /skills <name> <rest>）
            let has_rest = parts.clone().peekable().peek().is_some();

            // 查找 skill（大小写不敏感）
            let found = (**skill_manifests)
                .iter()
                .find(|s| {
                    // 如果有多余参数，action 仅在 skill 名字本身范围内匹配
                    s.name.eq_ignore_ascii_case(action)
                })
                .map(|s| s.name.clone());

            match found {
                Some(name) => {
                    if has_rest {
                        // /skills <name> <rest>：提取 rest 作为本轮问题
                        let rest =
                            extract_rest_after_skill_name(&normalized_for_rest, &name, &action);
                        app.forced_skill = Some(name);
                        app.forced_question = rest;
                        return Ok(true);
                    }
                    app.forced_skill = Some(name.clone());
                    println!("Skill selected for next turn: {name}");
                    println!("Ask your next question naturally, or mention @skills:{name} inline.");
                }
                None => {
                    println!("Unknown /skills subcommand: {action}");
                    println!();
                    println!("Usage: /skills [list|current|use <name>|help]");
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
                }
            }
        }
    }
    Ok(true)
}

/// 从 /skills <name> <rest...> 中提取 <rest...> 部分。
/// normalized 形如 "skills name rest rest..."
fn extract_rest_after_skill_name(
    normalized: &str,
    skill_name: &str,
    action: &str,
) -> Option<String> {
    // 按空格分割，找到 skill 名称后面的部分
    let after_prefix = normalized
        .strip_prefix("skills ")
        .or_else(|| normalized.strip_prefix("skill "))?;

    // 去掉匹配到的 skill 名称本身
    let after_skill_name = if let Some(rest) = after_prefix.strip_prefix(action) {
        rest.trim()
    } else if let Some(rest) = after_prefix.strip_prefix(skill_name) {
        rest.trim()
    } else {
        return None;
    };

    if after_skill_name.is_empty() {
        None
    } else {
        Some(after_skill_name.to_string())
    }
}
