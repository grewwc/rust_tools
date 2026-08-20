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
    // 剩余 token 一次性收集：use/隐式选择都需要完整消费参数列表。
    let rest_tokens: Vec<&str> = parts.collect();

    match action {
        "help" | "h" | "--help" => {
            println!("Skill management commands:");
            println!();
            println!("  /skills              list all available skills");
            println!("  /skills list         list all available skills");
            println!("  /skills current      show pending/recent skill selection");
            println!("  /skills use <name>...  force the specified skills on the next turn");
            println!("  /skills <name>...      shorthand for /skills use (also accepts a");
            println!("                         trailing question: /skills <name>... <question>)");
            println!("  /skills help         show this help");
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
            let pending = pending_skill_names(app);
            if !pending.is_empty() {
                println!(
                    "current skills: {} (pending for next turn)",
                    pending.join(", ")
                );
            } else if let Some(name) = recent_skill_name(app) {
                println!("current skill: {name} (recent active skill)");
            } else {
                println!("No active skill.");
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

            // 多个名字逐个解析（大小写不敏感、去重），未命中的单独提示，
            // 某个名字失效不应拖垮整个集合。
            let mut names: Vec<String> = Vec::new();
            let mut not_found: Vec<String> = Vec::new();
            for skill_name in &rest_tokens {
                if let Some(skill) = (**skill_manifests)
                    .iter()
                    .find(|s| s.name.eq_ignore_ascii_case(skill_name))
                {
                    if !names.iter().any(|n| n == &skill.name) {
                        names.push(skill.name.clone());
                    }
                } else {
                    not_found.push((*skill_name).to_string());
                }
            }
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
        // 隐式选择：输入 /skills <skillname> 直接应用，无需 use 关键字
        _ => {
            // 贪婪收集连续匹配 skill 名的 token 作为技能集（保持输入顺序、按
            // manifest 规范化、去重），其余 token 作为本轮问题正文；也支持
            // skill 名与问题正文粘连（如 `/skill code-review帮我review`）。
            let mut tokens = Vec::with_capacity(rest_tokens.len() + 1);
            tokens.push(action);
            tokens.extend(rest_tokens);
            let (names, rest) = resolve_implicit_selection(&tokens, skill_manifests);

            if names.is_empty() {
                println!("Unknown /skills subcommand: {action}");
                println!();
                println!("Usage: /skills [list|current|use <name>...|help]");
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
                // /skills <name>... <rest>：rest 作为本轮问题
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
    Ok(true)
}

/// 解析 /skills <name>... <rest...> 的隐式多 skill 选择。
/// 贪婪收集**连续**匹配 skill 名的 token 作为技能集（大小写不敏感、按 manifest
/// 规范化、去重、保持输入顺序），其余 token 作为本轮问题正文。
/// skill 名是 kebab-case 标识符（ASCII 字母/数字/`-`/`_`）；一个 token 若整体
/// 不匹配但其 ident 前缀匹配某个 skill 名，则视为「名字 + 无空格粘连的问题正文」
/// （如 `/skill Code-Review帮我review`），粘连部分连同后续 token 一起作为问题。
/// 返回 (规范名列表, rest)。
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
        if let Some(skill) = skill_manifests
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(token))
        {
            push_name(&skill.name, &mut names);
            continue;
        }
        // 整体不匹配：检查粘连形式（ident 前缀是 skill 名）。
        let mut consumed_as_skill = false;
        if let Some(pos) = token.find(|c: char| !is_ident(c)) {
            let prefix = &token[..pos];
            if let Some(skill) = skill_manifests
                .iter()
                .find(|s| s.name.eq_ignore_ascii_case(prefix))
            {
                push_name(&skill.name, &mut names);
                rest_tokens.push(&token[pos..]);
                consumed_as_skill = true;
            }
        }
        if !consumed_as_skill {
            rest_tokens.push(token);
        }
        // 该 token 起（含其粘连正文）之后全部是问题正文。
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

#[cfg(test)]
mod tests {
    use super::{resolve_implicit_selection, SkillManifest};

    fn skill(name: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: String::new(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
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
        vec![skill("code-review"), skill("docs-review"), skill("bytedcli")]
    }

    fn resolve(input: &str) -> (Vec<String>, Option<String>) {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        resolve_implicit_selection(&tokens, &manifests())
    }

    #[test]
    fn single_skill_with_question() {
        // /skill code-review 帮我review
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
        // /skills code-review docs-review 帮我看看这段
        let (names, rest) = resolve("code-review docs-review 帮我看看这段");
        assert_eq!(names, vec!["code-review", "docs-review"]);
        assert_eq!(rest.as_deref(), Some("帮我看看这段"));
    }

    #[test]
    fn duplicate_names_deduplicated() {
        // /skills code-review Code-Review → 去重且规范化大小写
        let (names, rest) = resolve("code-review Code-Review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest, None);
    }

    #[test]
    fn unknown_token_starts_question() {
        // 未命中 token 起视为问题正文：/skills code-review 不存在的skill 问题
        let (names, rest) = resolve("code-review 不存在的skill 问题");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("不存在的skill 问题"));
    }

    #[test]
    fn glued_question_after_multi_skills() {
        // /skills code-review docs-review帮我看看 → docs-review 后粘连问题正文
        let (names, rest) = resolve("code-review docs-review帮我看看");
        assert_eq!(names, vec!["code-review", "docs-review"]);
        assert_eq!(rest.as_deref(), Some("帮我看看"));
    }

    #[test]
    fn glued_question_with_mixed_case() {
        // /skill Code-Review帮我review → 粘连剥出后剩余作为问题
        let (names, rest) = resolve("Code-Review帮我review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review"));
    }

    #[test]
    fn question_repeating_skill_name() {
        // /skill code-review code-review帮我review：重复的 skill 名去重，
        // 粘连正文只保留一次（不再把整段当作问题）。
        let (names, rest) = resolve("code-review code-review帮我review");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review"));
    }

    #[test]
    fn multi_word_question() {
        // /skill code-review 帮我review 这段代码 并给出建议
        let (names, rest) = resolve("code-review 帮我review 这段代码 并给出建议");
        assert_eq!(names, vec!["code-review"]);
        assert_eq!(rest.as_deref(), Some("帮我review 这段代码 并给出建议"));
    }

    #[test]
    fn multiline_question() {
        // /skills bytedcli\n帮我检查最近的飞书消息
        let (names, rest) = resolve("bytedcli 帮我检查最近的飞书消息");
        assert_eq!(names, vec!["bytedcli"]);
        assert_eq!(rest.as_deref(), Some("帮我检查最近的飞书消息"));
    }

    #[test]
    fn unknown_subcommand_no_names() {
        // 不是任何 skill 名 → 无技能、无问题（命令层走 Unknown 分支）
        let (names, rest) = resolve("fizzbuzz");
        assert!(names.is_empty());
        assert_eq!(rest.as_deref(), Some("fizzbuzz"));
    }
}
