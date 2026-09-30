//! Session facts that explicit evidence offloading depends on.
//!
//! Prune eligibility is otherwise a pure function of the message list. Subagent
//! results break that assumption: the work behind them is finished only once the
//! parent has integrated the conclusion, and that fact lives in the durable task
//! evidence ledger rather than in any message. This module carries those
//! ledger-derived facts into the projection so eligibility stays a pure function.
//!
//! The default value authorizes nothing, so every caller that cannot read the
//! ledger keeps conditional results inline instead of unlocking their offloading.

use rustc_hash::FxHashSet;

/// Every runtime-minted task id starts with this prefix (`next_task_id`).
const TASK_ID_PREFIX: &str = "task_";

/// Markers the runtime itself writes into subagent results: `[task_id=X]` for
/// deliveries, `Task spawned: task_id=X` for spawns, `retry_of=X` /
/// `new_task_id=X` for retries, the `"task_id": "X"` JSON form used by
/// `task_spawn_batch` and by the `task_evidence_read` envelope, and the
/// `"runtime_task_id": "X"` / `"pool_task_id": "X"` fields that `manage_team`
/// and `run_agent_graph` render in their JSON output.
///
/// This list is the only producer of the id set, so a shape that is missing
/// here can never be authorized and its results stay inline permanently. Add a
/// marker only for fields carrying a runtime-minted `task_<id>`: ids from
/// another id space (for example graph team ids such as `g-<uuid>`) are dropped
/// by [`task_id_token`] and would only make more results un-prunable.
const TASK_ID_MARKERS: [&str; 8] = [
    "task_id=",
    "new_task_id=",
    "retry_of=",
    "\"task_id\":",
    "\"new_task_id\":",
    "\"retry_of\":",
    "\"runtime_task_id\":",
    "\"pool_task_id\":",
];

/// Every task id a tool result refers to.
///
/// Extraction is deliberately marker-based and accepts only the id charset the
/// runtime generates, so a payload that merely quotes a marker (with no id
/// after it) contributes nothing. A result whose shape yields no id is never
/// authorized, which is the fail-closed direction.
pub(super) fn referenced_task_ids(content: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for marker in TASK_ID_MARKERS {
        let mut rest = content;
        while let Some(offset) = rest.find(marker) {
            rest = &rest[offset + marker.len()..];
            let id = task_id_token(rest);
            if !id.is_empty() && !ids.iter().any(|known| known == id) {
                ids.push(id.to_string());
            }
        }
    }
    ids
}

fn task_id_token(rest: &str) -> &str {
    let rest = rest.trim_start_matches([' ', '"', '\'', ':']);
    let end = rest
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'))
        .unwrap_or(rest.len());
    let token = &rest[..end];
    // Only the ids the runtime itself mints (`task_<id>`) count. A marker
    // followed by prose must not yield a phantom id, which would make an
    // ordinary result un-prunable forever.
    if token.starts_with(TASK_ID_PREFIX) {
        token
    } else {
        ""
    }
}

/// Which subagent tasks the durable evidence ledger records as integrated.
///
/// Integration is the boundary between "live state" (a delivered result the
/// parent may still need verbatim) and "history" (a conclusion already
/// incorporated, whose full text stays retrievable from the ledger).
#[derive(Clone, Debug, Default)]
pub(crate) struct PruneAuthorization {
    integrated_task_ids: FxHashSet<String>,
}

impl PruneAuthorization {
    pub(crate) fn from_integrated_task_ids(integrated_task_ids: FxHashSet<String>) -> Self {
        Self {
            integrated_task_ids,
        }
    }

    /// Whether a subagent result may be unloaded.
    ///
    /// Requires that the result names at least one task (an unrecognized result
    /// cannot be proven integrated) and that *every* task it names is
    /// integrated, so a multi-task result still holding one un-integrated
    /// delivery stays inline in full.
    pub(crate) fn allows_subagent_result(&self, content: &str) -> bool {
        let ids = referenced_task_ids(content);
        !ids.is_empty() && ids.iter().all(|id| self.integrated_task_ids.contains(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[&str]) -> PruneAuthorization {
        PruneAuthorization::from_integrated_task_ids(ids.iter().map(|id| id.to_string()).collect())
    }

    #[test]
    fn extracts_ids_from_every_runtime_result_shape() {
        assert_eq!(
            referenced_task_ids("[task_id=task_abc]\nresult body"),
            vec!["task_abc".to_string()]
        );
        assert_eq!(
            referenced_task_ids("Task spawned: task_id=task_abc, pid=7, agent=x"),
            vec!["task_abc".to_string()]
        );
        assert_eq!(
            referenced_task_ids("Task retried: retry_of=task_old, new_task_id=task_new, pid=7")
                .len(),
            2
        );
        assert_eq!(
            referenced_task_ids("{\n  \"task_id\": \"task_abc\",\n  \"status\": \"delivered\"\n}"),
            vec!["task_abc".to_string()]
        );
        assert_eq!(
            referenced_task_ids("- task_id=task_abc status=done agent=x"),
            vec!["task_abc".to_string()]
        );
        assert_eq!(
            referenced_task_ids(
                "{\n  \"tasks\": [\n    {\n      \"runtime_task_id\": \"task_worker\",\n      \"state\": \"done\"\n    }\n  ]\n}"
            ),
            vec!["task_worker".to_string()]
        );
        assert_eq!(
            referenced_task_ids("{\n  \"pool_task_id\": \"task_pool\",\n  \"phase\": \"advance\"\n}"),
            vec!["task_pool".to_string()]
        );
    }

    #[test]
    fn ignores_shapes_without_a_concrete_id() {
        assert!(referenced_task_ids("no markers here").is_empty());
        assert!(referenced_task_ids("the task_id= field is documented").is_empty());
        assert!(referenced_task_ids("{\"task_id\": \"\"}").is_empty());
    }

    #[test]
    fn authorizes_graph_status_only_when_every_worker_task_is_integrated() {
        let content = "{\n  \"nodes\": [\n    { \"runtime_task_id\": \"task_a\", \"status\": \"done\" },\n    { \"runtime_task_id\": \"task_b\", \"status\": \"running\" }\n  ]\n}";
        // One worker still running: the snapshot is live state, not history.
        assert!(!set(&["task_a"]).allows_subagent_result(content));
        assert!(set(&["task_a", "task_b"]).allows_subagent_result(content));
    }

    #[test]
    fn authorizes_only_fully_integrated_results() {
        let authorization = set(&["task_a", "task_b"]);
        assert!(authorization.allows_subagent_result("[task_id=task_a]\nbody"));
        assert!(authorization.allows_subagent_result(
            "[task_id=task_a]\nbody\n[task_id=task_b]\nbody"
        ));
        // One un-integrated delivery keeps the whole result inline.
        assert!(!authorization.allows_subagent_result(
            "[task_id=task_a]\nbody\n[task_id=task_c]\nbody"
        ));
        assert!(!authorization.allows_subagent_result("plain tool output"));
    }

    #[test]
    fn default_authorization_denies_everything_conditional() {
        let authorization = PruneAuthorization::default();
        assert!(!authorization.allows_subagent_result("[task_id=task_a]\nbody"));
        assert!(!authorization.allows_subagent_result("plain tool output"));
    }
}