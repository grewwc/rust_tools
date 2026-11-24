//! Best-effort page evidence around trusted CDP input; AppleScript retains its single observation.

use std::time::Duration;

use chromiumoxide::cdp::browser_protocol::target::{GetTargetsParams, TargetInfo};
use chromiumoxide::cdp::js_protocol::runtime::EvaluateParams;
use chromiumoxide::page::Page;
use serde_json::{Value, json};
use tokio::time::{Instant, sleep_until, timeout_at};

use crate::browser::BrowserSession;
use crate::contract::STATE_SCRIPT;

const CAPTURE_BUDGET: Duration = Duration::from_millis(150);
const SETTLE_BUDGET: Duration = Duration::from_millis(750);
const FINAL_OBSERVATION_BUDGET: Duration = Duration::from_millis(100);
const QUIET_WINDOW: Duration = Duration::from_millis(140);
const POLL_INTERVAL: Duration = Duration::from_millis(40);

// One counter per top-level document. No DOM attributes, record history, or DOM
// text are retained. A new document gets a different identity even on reload.
const PROBE_SCRIPT: &str = r#"(() => {
    const key = Symbol.for('mcp_browser.action_feedback');
    let probe = document[key];
    if (!probe) {
        probe = {documentId:`${performance.timeOrigin}:${Math.random()}`, revision:0};
        const observer = new MutationObserver(() => { ++probe.revision; });
        observer.observe(document, {subtree:true, childList:true, attributes:true, characterData:true});
        document[key] = probe;
    }
    return {documentId:probe.documentId, revision:probe.revision,
        state:{current_url:location.href, readyState:document.readyState, scroll:{x:scrollX,y:scrollY}}};
})()"#;

struct Probe {
    document_id: String,
    revision: u64,
    state: Value,
}

impl Probe {
    fn parse(sample: Value) -> Option<Self> {
        let state = sample.get("state")?;
        state.get("current_url")?.as_str()?;
        state.get("readyState")?.as_str()?;
        Some(Self {
            document_id: sample.get("documentId")?.as_str()?.to_owned(),
            revision: sample.get("revision")?.as_u64()?,
            state: state.clone(),
        })
    }

    fn dom_changed(&self, before: &Self) -> bool {
        self.document_id != before.document_id || self.revision != before.revision
    }

    fn resets_quiet_window(&self, previous: &Self) -> bool {
        self.dom_changed(previous)
            || self.state["current_url"] != previous.state["current_url"]
            || self.state["readyState"] != previous.state["readyState"]
    }

    fn ready(&self) -> bool {
        matches!(
            self.state["readyState"].as_str(),
            Some("interactive" | "complete")
        )
    }
}

pub(super) struct ActionFeedback {
    before: Option<Probe>,
    targets: Option<Vec<TargetInfo>>,
}

impl ActionFeedback {
    pub(super) async fn capture(session: &BrowserSession, deadline: Instant) -> Self {
        let now = Instant::now();
        // Optional capture must leave time for the actual input and its verdict.
        let budget = (deadline.saturating_duration_since(now) / 4).min(CAPTURE_BUDGET);
        if budget.is_zero() {
            return Self {
                before: None,
                targets: None,
            };
        }
        let deadline = now + budget;
        let (before, targets) = tokio::join!(
            timeout_at(deadline, probe(&session.page)),
            timeout_at(deadline, page_targets(session)),
        );
        Self {
            before: before.ok().flatten(),
            targets: targets.ok().flatten(),
        }
    }

    pub(super) async fn finish(self, session: &BrowserSession, deadline: Instant) -> String {
        let now = Instant::now();
        let budget = deadline.saturating_duration_since(now).min(SETTLE_BUDGET);
        let deadline = now + budget;
        let settle_deadline = deadline - FINAL_OBSERVATION_BUDGET.min(budget / 2);
        let mut latest: Option<Probe> = None;
        let mut quiet_since = now;
        let mut settled = false;
        if let Some(before) = &self.before {
            while Instant::now() < settle_deadline {
                let sample = match timeout_at(settle_deadline, probe(&session.page)).await {
                    Ok(Some(sample)) => sample,
                    Ok(None) => {
                        // Evaluation failed (for instance, the context navigated away).
                        // Fall back to one ordinary page-state read without retrying input.
                        latest = None;
                        break;
                    }
                    Err(_) => break,
                };
                let now = Instant::now();
                if sample.resets_quiet_window(latest.as_ref().unwrap_or(before)) {
                    quiet_since = now;
                }
                settled = sample.ready() && now.duration_since(quiet_since) >= QUIET_WINDOW;
                latest = Some(sample);
                if settled {
                    break;
                }
                sleep_until((now + POLL_INTERVAL).min(settle_deadline)).await;
            }
        }

        let dom_changed = self
            .before
            .as_ref()
            .zip(latest.as_ref())
            .map(|(before, after)| after.dom_changed(before));
        let (state, targets) = tokio::join!(
            async {
                if let Some(sample) = latest {
                    return Some(sample.state);
                }
                let value = timeout_at(deadline, evaluate(&session.page, STATE_SCRIPT))
                    .await
                    .ok()??;
                serde_json::from_str::<Value>(value.as_str()?)
                    .ok()
                    .filter(Value::is_object)
            },
            async {
                let before = self.targets?;
                let after = timeout_at(deadline, page_targets(session)).await.ok()??;
                Some(new_tabs(session.page.target_id().as_ref(), &before, after))
            },
        );
        let unknown = state.is_none();
        let mut state = state.unwrap_or_else(|| json!({}));
        state["dom_changed"] = json!(dom_changed);
        state["settled"] = json!(settled);
        state["new_tabs"] = json!(targets.unwrap_or_default());
        if unknown {
            format!("post_action_state=unknown\n{state}")
        } else {
            state.to_string()
        }
    }
}

async fn evaluate(page: &Page, expression: &str) -> Option<Value> {
    // Use the target's current default context; an action may replace it.
    let command = EvaluateParams::builder()
        .expression(expression)
        .return_by_value(true)
        .build()
        .ok()?;
    let response = page.execute(command).await.ok()?.result;
    if response.exception_details.is_some() {
        return None;
    }
    response.result.value
}

async fn probe(page: &Page) -> Option<Probe> {
    Probe::parse(evaluate(page, PROBE_SCRIPT).await?)
}

async fn page_targets(session: &BrowserSession) -> Option<Vec<TargetInfo>> {
    let response = session
        .browser
        .execute(GetTargetsParams::default())
        .await
        .ok()?;
    Some(
        response
            .result
            .target_infos
            .into_iter()
            .filter(|target| target.r#type == "page" && target.subtype.is_none())
            .collect(),
    )
}

fn new_tabs(opener: &str, before: &[TargetInfo], after: Vec<TargetInfo>) -> Vec<Value> {
    // Chrome reports an opener ID for target=_blank links, including noopener.
    // Scope the diff to this tab so an attached user's unrelated tabs stay out.
    after
        .into_iter()
        .filter(|target| {
            target
                .opener_id
                .as_ref()
                .is_some_and(|id| id.as_ref() == opener)
                && !before
                    .iter()
                    .any(|known| known.target_id == target.target_id)
        })
        .map(|target| json!({"id":target.target_id,"url":target.url}))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_fixes_action_probe_distinguishes_navigation_from_same_document_url_change() {
        let sample = |document, revision, url, ready| {
            Probe::parse(json!({
            "documentId":document,"revision":revision,"state":{"current_url":url,"readyState":ready}
        })).unwrap()
        };
        let before = sample("old", 0, "https://example.com/", "complete");
        let hash = sample("old", 0, "https://example.com/#open", "complete");
        assert!(!hash.dom_changed(&before));
        assert!(hash.resets_quiet_window(&before));
        let reload = sample("new", 0, "https://example.com/", "loading");
        assert!(reload.dom_changed(&before));
        assert!(!reload.ready());
        let parsed = sample("new", 0, "https://example.com/", "interactive");
        assert!(parsed.ready());
        assert!(parsed.resets_quiet_window(&reload));
        let mutation = sample("new", 1, "https://example.com/", "interactive");
        assert!(mutation.dom_changed(&parsed));
        assert!(!mutation.resets_quiet_window(&mutation));
    }

    #[test]
    fn browser_fixes_action_tabs_only_report_new_targets_opened_by_current_tab() {
        let target = |id: &str, opener: Option<&str>, url: &str| -> TargetInfo {
            serde_json::from_value(json!({"targetId":id,"type":"page","title":"","url":url,
                "attached":false,"openerId":opener,"canAccessOpener":false}))
            .unwrap()
        };
        let old = target("already-open", Some("current"), "about:blank");
        let before = vec![old.clone()];
        let after = vec![
            old,
            target("other-user-tab", None, "about:blank"),
            target("other-tab-popup", Some("other"), "about:blank"),
            target("popup", Some("current"), "https://example.com/"),
        ];
        assert_eq!(
            new_tabs("current", &before, after),
            vec![json!({"id":"popup","url":"https://example.com/"})]
        );
    }
}
