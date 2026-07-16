//! 技能路由（skill routing）逻辑。
//!
//! 通过辅助模型决定当前请求是否需要激活某个已注册 skill：
//! - `select_skill_via_model`：对外入口，分块路由 + 置信度阈值
//! - `select_skill_candidate_via_model`：单块候选选择
//! - `parse_router_output` / `extract_router_content`：解析辅助模型 JSON 响应
//! - `strip_json_fence`：去除 ```json 围栏

use std::time::{Duration, Instant};

use serde_json::Value;

use crate::ai::history::Message;
use crate::ai::types::App;
use crate::commonw::configw;

use super::builder::build_request_body;
use super::error::{
    api_key_for_request_model, apply_request_auth, control_model_for_aux_tasks,
    endpoint_for_request_model,
};
use crate::ai::skills::SkillManifest;

pub(crate) fn strip_json_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start_matches('\n').trim_start_matches('\r');
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim();
        }
    }
    trimmed
}

fn parse_router_output(s: &str) -> (Option<String>, f64) {
    let s = strip_json_fence(s);
    let candidate = if let (Some(l), Some(r)) = (s.find('{'), s.rfind('}'))
        && r >= l
    {
        &s[l..=r]
    } else {
        s
    };
    let v: Value = match serde_json::from_str(candidate) {
        Ok(v) => v,
        Err(_) => return (None, 0.0),
    };
    let name = v
        .get("skill")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let confidence = v.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if name.is_empty() || name == "none" || name == "null" {
        (None, confidence)
    } else {
        (Some(name), confidence)
    }
}

pub(crate) fn extract_router_content(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output").and_then(|o| o.get("choices")))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    match content {
        Value::String(s) => Some(s.to_string()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct SkillRouterDecision {
    skill: Option<String>,
    confidence: f64,
}

async fn select_skill_candidate_via_model(
    app: &App,
    question: &str,
    skills: &[SkillManifest],
) -> Option<SkillRouterDecision> {
    if question.trim().is_empty() || skills.is_empty() {
        return None;
    }

    let mut system_prompt = r#"You are a skill router for a code-focused assistant.
Your job is to decide whether the current request clearly needs one of the available skills.
Output schema: {"skill":"<exact skill name or empty>","confidence":0.0}
Rules:
- Route only when the request is explicitly about operating on source code, code artifacts, or a coding workflow that matches a listed skill.
- Abstain for general knowledge, documentation lookup, high-level discussion, non-code work, or ambiguous requests.
- Prefer abstaining over misrouting when the evidence is weak.
- Use only the exact skill names listed below.
- Return only valid JSON.
Skills:
"#.to_string();

    for s in skills {
        let desc = if s.description.trim().is_empty() {
            "(no description)".to_string()
        } else {
            s.description.trim().to_string()
        };
        system_prompt.push_str(&format!("- {}: {}\n", s.name, desc));
    }

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system_prompt),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(question.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    // 辅助请求（skill 路由），15 秒超时兜底，理由同上。
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(15), send_future).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };
    if !response.status().is_success() {
        return None;
    }

    let text = match tokio::time::timeout(Duration::from_secs(15), response.text()).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v).unwrap_or_default();
    let (name, confidence) = parse_router_output(&content);
    Some(SkillRouterDecision {
        skill: name,
        confidence,
    })
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "R",
    "request::select_skill_via_model",
    "[DEBUG] model skill router started",
    "[DEBUG] model skill router finished",
    {
        "question_len": question.chars().count(),
        "skill_count": skills.len(),
    },
    {
        "selected": __agent_hang_result.as_deref(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) async fn select_skill_via_model(
    app: &mut App,
    _model: &str,
    question: &str,
    skills: &[SkillManifest],
) -> Option<String> {
    const SKILL_ROUTER_CHUNK_SIZE: usize = 32;
    let router_start = Instant::now();
    if question.trim().is_empty() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "R",
            "request::select_skill_via_model:empty_question",
            "[DEBUG] model skill router skipped empty question",
            {
                "elapsed_ms": router_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }
    if skills.is_empty() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "R",
            "request::select_skill_via_model:empty_skills",
            "[DEBUG] model skill router skipped empty skills",
            {
                "elapsed_ms": router_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }

    let cfg = configw::get_all_config();
    let threshold = cfg
        .get_opt("ai.skills.router_threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let decision = if skills.len() <= SKILL_ROUTER_CHUNK_SIZE {
        select_skill_candidate_via_model(app, question, skills).await
    } else {
        let mut chunk_best: Vec<(String, f64)> = Vec::new();
        for chunk in skills.chunks(SKILL_ROUTER_CHUNK_SIZE) {
            let Some(decision) = select_skill_candidate_via_model(app, question, chunk).await
            else {
                continue;
            };
            let Some(name) = decision.skill else {
                continue;
            };
            if let Some(existing) = chunk_best.iter_mut().find(|(n, _)| *n == name) {
                if decision.confidence > existing.1 {
                    existing.1 = decision.confidence;
                }
            } else {
                chunk_best.push((name, decision.confidence));
            }
        }
        chunk_best.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if chunk_best.is_empty() {
            None
        } else if chunk_best.len() == 1 {
            Some(SkillRouterDecision {
                skill: Some(chunk_best[0].0.clone()),
                confidence: chunk_best[0].1,
            })
        } else {
            let finalists = chunk_best
                .iter()
                .filter_map(|(name, _)| skills.iter().find(|s| s.name == *name).cloned())
                .collect::<Vec<_>>();
            select_skill_candidate_via_model(app, question, &finalists)
                .await
                .or_else(|| {
                    Some(SkillRouterDecision {
                        skill: Some(chunk_best[0].0.clone()),
                        confidence: chunk_best[0].1,
                    })
                })
        }
    };
    let Some(decision) = decision else {
        return None;
    };
    if decision.confidence >= threshold {
        decision.skill
    } else {
        None
    }
}
