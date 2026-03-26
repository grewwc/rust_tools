use crate::ai::skills::SkillManifest;

const SKILL_MATCH_THRESHOLD: i64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SkillScore {
    evidence: i64,
    total: i64,
}

pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    let input_norm = normalize(input);
    if input_norm.is_empty() {
        return None;
    }
    let input_tokens = tokenize(&input_norm);

    let mut best: Option<(&SkillManifest, SkillScore)> = None;

    for skill in skills {
        let score = score_skill(skill, &input_norm, &input_tokens);
        if score.evidence < SKILL_MATCH_THRESHOLD {
            continue;
        }
        match best {
            None => best = Some((skill, score)),
            Some((best_skill, best_score)) => {
                if score.total > best_score.total
                    || (score.total == best_score.total && skill.priority > best_skill.priority)
                {
                    best = Some((skill, score));
                }
            }
        }
    }

    best.map(|(skill, _)| skill)
}

fn score_skill(skill: &SkillManifest, input_norm: &str, input_tokens: &[String]) -> SkillScore {
    let mut evidence = 0i64;
    let mut matched_trigger_count = 0i64;

    for trigger in &skill.triggers {
        let trigger_norm = normalize(trigger);
        if trigger_norm.is_empty() {
            continue;
        }
        if input_norm.contains(&trigger_norm) {
            matched_trigger_count += 1;
            evidence += 100;
            // Reward longer exact-phrase triggers a bit more; this stays generic.
            evidence += (trigger_norm.chars().count() as i64).min(40);
            continue;
        }

        let trigger_tokens = tokenize(&trigger_norm);
        let overlap = token_overlap(input_tokens, &trigger_tokens);
        if overlap >= 2 {
            evidence += overlap as i64 * 18;
        }
    }

    if matched_trigger_count > 1 {
        evidence += (matched_trigger_count - 1) * 20;
    }

    let mut skill_text = String::new();
    skill_text.push_str(&skill.name);
    skill_text.push(' ');
    skill_text.push_str(&skill.description);
    skill_text.push(' ');
    for t in &skill.triggers {
        skill_text.push_str(t);
        skill_text.push(' ');
    }
    let skill_tokens = tokenize(&normalize(&skill_text));
    let overlap = token_overlap(input_tokens, &skill_tokens);
    evidence += overlap as i64 * 10;

    SkillScore {
        evidence,
        total: evidence + skill.priority as i64,
    }
}

fn token_overlap(a: &[String], b: &[String]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let b = b.iter().collect::<std::collections::BTreeSet<_>>();
    let mut count = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for token in a {
        if token.len() <= 1 {
            continue;
        }
        if !seen.insert(token) {
            continue;
        }
        if b.contains(token) {
            count += 1;
        }
    }
    count
}

fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut ascii = String::new();
    let mut cjk = String::new();

    let flush_ascii = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.is_empty() {
            out.push(std::mem::take(buf));
        }
    };
    let flush_cjk = |buf: &mut String, out: &mut Vec<String>| {
        if buf.is_empty() {
            return;
        }
        let segment = std::mem::take(buf);
        let chars = segment.chars().collect::<Vec<_>>();
        if chars.len() == 1 {
            out.push(segment);
            return;
        }
        for window in chars.windows(2) {
            out.push(window.iter().collect());
        }
        out.push(chars.iter().collect());
    };

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            flush_cjk(&mut cjk, &mut out);
            ascii.push(ch);
        } else if is_cjk(ch) {
            flush_ascii(&mut ascii, &mut out);
            cjk.push(ch);
        } else {
            flush_ascii(&mut ascii, &mut out);
            flush_cjk(&mut cjk, &mut out);
        }
    }
    flush_ascii(&mut ascii, &mut out);
    flush_cjk(&mut cjk, &mut out);
    out
}

fn normalize(input: &str) -> String {
    input
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_punctuation() {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
}

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str, triggers: &[&str], priority: i32) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: Some("system".to_string()),
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            prompt: String::new(),
            system_prompt: None,
            examples: Vec::new(),
            triggers: triggers.iter().map(|s| s.to_string()).collect(),
            priority,
            source_path: None,
        }
    }

    #[test]
    fn priority_alone_does_not_match_skill() {
        let skills = vec![skill("openclaw", "agent", &["openclaw"], 30)];
        assert!(match_skill(&skills, "随便聊聊今天吃什么").is_none());
    }

    #[test]
    fn review_style_input_prefers_code_review_over_priority_fallback() {
        let skills = vec![
            skill("openclaw", "OpenClaw-like autonomous tool-using agent", &["openclaw"], 30),
            skill(
                "code-review",
                "Review code for quality, security, and best practices",
                &["帮我看一下", "看一下", "有问题吗"],
                10,
            ),
        ];
        let matched = match_skill(&skills, "你帮我看看这个makefile是不是有问题").unwrap();
        assert_eq!(matched.name, "code-review");
    }

    #[test]
    fn explicit_openclaw_trigger_still_matches() {
        let skills = vec![skill("openclaw", "agent", &["openclaw模式", "开启openclaw"], 30)];
        let matched = match_skill(&skills, "帮我开启openclaw模式").unwrap();
        assert_eq!(matched.name, "openclaw");
    }
}
