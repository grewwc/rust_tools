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
                        let rest = extract_rest_after_skill_name(&normalized_for_rest);
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
fn extract_rest_after_skill_name(normalized: &str) -> Option<String> {
    // 跳过前导 "skill"/"skills"，下一个 token 承载 skill 名。skill 名是 kebab-case
    // 标识符（ASCII 字母/数字/`-`/`_`）；该 token 中第一个非标识符字符起，即 skill 名
    // 后「无空格粘连」的问题正文（如 `/skill Code-Review帮我review`）。若无粘连，问题
    // 正文是随后的其余 token。用字符集边界而非 strip_prefix，既能剥出粘连问题，又不会
    // 误伤「问题恰好以 skill 名开头」的场景（如 `/skill code-review code-review帮我review`
    // 要完整保留 `code-review帮我review`）。
    let mut tokens = normalized.split_whitespace();
    tokens.next()?; // "skill" / "skills"
    let name_token = tokens.next()?; // 承载 skill 名的 token
    let trailing: String = tokens.collect::<Vec<&str>>().join(" ");

    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_';
    let glued = name_token
        .find(|c: char| !is_ident(c))
        .map(|pos| &name_token[pos..])
        .unwrap_or("");

    let rest = match (glued.is_empty(), trailing.is_empty()) {
        (true, true) => return None,
        (true, false) => trailing,
        (false, true) => glued.to_string(),
        (false, false) => format!("{glued} {trailing}"),
    };
    Some(rest)
}

#[cfg(test)]
mod tests {
    use super::extract_rest_after_skill_name;

    #[test]
    fn rest_after_simple_skill_name() {
        // /skill code-review 帮我review
        let r = extract_rest_after_skill_name("skill code-review 帮我review");
        assert_eq!(r.as_deref(), Some("帮我review"));
    }

    #[test]
    fn rest_after_no_question() {
        // /skill code-review
        let r = extract_rest_after_skill_name("skill code-review");
        assert_eq!(r, None);
    }

    #[test]
    fn rest_preserved_when_question_starts_with_skill_name() {
        // 之前会返回 None 的场景：用户问题以 skill 名开头
        // /skill code-review code-review帮我review
        let r = extract_rest_after_skill_name("skill code-review code-review帮我review");
        assert_eq!(r.as_deref(), Some("code-review帮我review"));
    }

    #[test]
    fn rest_with_mixed_case_action() {
        // action 与 skill_name 大小写不同
        // /skill Code-Review帮我review  → action="Code-Review"
        let r = extract_rest_after_skill_name("skill Code-Review帮我review");
        assert_eq!(r.as_deref(), Some("帮我review"));
    }

    #[test]
    fn rest_with_multi_word_question() {
        // /skill code-review 帮我review 这段代码 并给出建议
        let r = extract_rest_after_skill_name("skill code-review 帮我review 这段代码 并给出建议");
        assert_eq!(r.as_deref(), Some("帮我review 这段代码 并给出建议"));
    }
}
