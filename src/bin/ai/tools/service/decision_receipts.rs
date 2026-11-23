//! Model-facing interface and binding checks for source-versioned receipts.

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::ai::driver::runtime_ctx;
use crate::ai::tools::registry::common::{
    ToolIrreversibleRegistration, ToolRegistration, ToolSpec, irreversible_tool,
};
use crate::ai::tools::storage::{
    decision_receipts::{Decision, Freshness, Ledger, Receipt, SourceSnapshot, Sources},
    file_store::FileStore,
};

pub(in crate::ai) const RECEIPT_ID_ARGUMENT: &str = "decision_receipt_id";
const LIST_LIMIT: usize = 8;
const PROJECTION_LIMIT: usize = 4;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "decision_receipt",
        description: "",
        execute,
    },
});

fn identifier(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 128
        && text
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}

fn sources(tool: &str, args: &Value, path: Option<&str>) -> Result<Sources, String> {
    if let Some(resolve) = irreversible_tool(tool).and_then(|policy| policy.sources) {
        if path.is_some() {
            return Err("Omit source_path; this tool derives its source from target_args.".into());
        }
        return resolve(args)
            .map(Sources::ToolTargets)
            .map_err(|error| error.to_string());
    }
    let path = path
        .filter(|s| !s.is_empty())
        .ok_or("source_path is required")?;
    let file = FileStore::new(PathBuf::from(path));
    file.validate_read_access()
        .map_err(|error| error.to_string())?;
    Ok(Sources::File(file.path().to_path_buf()))
}

fn locator(receipt: &Receipt) -> Value {
    json!({
        "id": receipt.id,
        "kind": receipt.decision.kind,
        "target_tool": receipt.decision.target_tool,
        "source_paths": receipt.decision.source.paths(),
        "created_at": receipt.created_at,
    })
}

fn verify_receipt_source(ledger: &Ledger, receipt: &Receipt) -> Result<SourceSnapshot, Freshness> {
    let decision = &receipt.decision;
    ledger.verify_source_with(receipt, || {
        // Resolution observes source state too. Record a durable invalidation
        // if the resolver cannot find or verify the registered target.
        if let Some(resolve) =
            irreversible_tool(&decision.target_tool).and_then(|tool| tool.sources)
        {
            if Sources::ToolTargets(resolve(&decision.target_args)?) != decision.source {
                return Err(std::io::Error::other(
                    "receipt does not match the current tool source targets",
                ));
            }
        } else if !matches!(decision.source, Sources::File(_)) {
            return Err(std::io::Error::other("receipt does not bind a source file"));
        }
        decision.source.snapshot()
    })
}

fn receipt_freshness(ledger: &Ledger, receipt: &Receipt) -> Freshness {
    verify_receipt_source(ledger, receipt)
        .err()
        .unwrap_or(Freshness::Fresh)
}

fn execute(args: &Value) -> Result<String, String> {
    let ledger = Ledger::current().ok_or("decision_receipt requires an active session")?;
    execute_in(&ledger, args)
}

fn execute_in(ledger: &Ledger, args: &Value) -> Result<String, String> {
    match args["op"].as_str() {
        Some("list") => {
            if let Some(id) = args["id"].as_str() {
                let receipt = ledger.read(id).map_err(|error| error.to_string())?;
                let freshness = receipt_freshness(ledger, &receipt);
                return Ok(json!({ "receipt": receipt, "freshness": freshness }).to_string());
            }
            let mut ids = ledger
                .recent_ids(args["before"].as_str(), LIST_LIMIT + 1)
                .map_err(|error| error.to_string())?;
            let next_before = (ids.len() > LIST_LIMIT).then(|| ids[LIST_LIMIT - 1].clone());
            ids.truncate(LIST_LIMIT);
            let entries = ids
                .iter()
                .map(|id| match ledger.read(id) {
                    Ok(receipt) => {
                        let mut entry = locator(&receipt);
                        entry["freshness"] = json!(receipt_freshness(ledger, &receipt));
                        entry
                    }
                    Err(_) => {
                        json!({ "id": id, "error": "Receipt could not be read or validated" })
                    }
                })
                .collect::<Vec<_>>();
            Ok(json!({ "receipts": entries, "next_before": next_before }).to_string())
        }
        Some(op @ ("snapshot" | "write")) => {
            let target_tool = args["target_tool"]
                .as_str()
                .filter(|s| identifier(s))
                .ok_or("target_tool must be an exact tool name")?;
            let target_args = &args["target_args"];
            if !target_args.is_object() || target_args.get(RECEIPT_ID_ARGUMENT).is_some() {
                return Err("target_args must be the exact planned argument object without decision_receipt_id".into());
            }
            let source = sources(target_tool, target_args, args["source_path"].as_str())?;
            if op == "snapshot" {
                let fingerprint = source
                    .fingerprint()
                    .map_err(|error| format!("cannot snapshot source: {error}"))?;
                return Ok(json!({
                    "source_paths": source.paths(),
                    "source_fingerprint": fingerprint,
                })
                .to_string());
            }
            let kind = args["kind"]
                .as_str()
                .filter(|s| identifier(s))
                .ok_or("kind must be an ASCII identifier")?;
            let fingerprint = args["source_fingerprint"]
                .as_str()
                .filter(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
                .ok_or("write requires source_fingerprint from snapshot")?;
            let numeric_params = args
                .get("numeric_params")
                .cloned()
                .unwrap_or_else(|| json!({}));
            if !numeric_params
                .as_object()
                .is_some_and(|params| params.values().all(|value| value.as_i64().is_some()))
            {
                return Err("numeric_params must be an object of signed integers".into());
            }
            let selected_ids = args
                .get("selected_ids")
                .map(|ids| serde_json::from_value::<Vec<String>>(ids.clone()))
                .transpose()
                .map_err(|_| "selected_ids must be an array of strings")?
                .unwrap_or_default();
            let receipt = ledger
                .write(Decision {
                    kind: kind.to_owned(),
                    target_tool: target_tool.to_owned(),
                    target_args: target_args.clone(),
                    cwd: runtime_ctx::effective_cwd().map_err(|error| error.to_string())?,
                    source,
                    source_fingerprint: fingerprint.to_owned(),
                    numeric_params,
                    selected_ids,
                })
                .map_err(|error| format!("cannot record receipt: {error}"))?;
            Ok(locator(&receipt).to_string())
        }
        _ => Err("op must be snapshot, write, or list".into()),
    }
}

/// Runtime-only binding to the receipt's exact arguments, paths, and fingerprint.
pub(in crate::ai) struct VerifiedDecision {
    policy: &'static ToolIrreversibleRegistration,
    ledger: Ledger,
    receipt: Receipt,
}

/// Authorization built at the execution boundary. The registered executor receives
/// the checked targets, including their identities, rather than resolving new paths.
pub(in crate::ai) struct AuthorizedDecision<'a> {
    policy: &'static ToolIrreversibleRegistration,
    args: &'a Value,
    source: SourceSnapshot,
}

impl AuthorizedDecision<'_> {
    pub(in crate::ai::tools) fn name(&self) -> &'static str {
        self.policy.name
    }

    pub(in crate::ai::tools) fn args(&self) -> &Value {
        self.args
    }

    pub(in crate::ai::tools) fn execute(&self) -> Result<String, String> {
        (self.policy.execute)(self.args, &self.source.targets)
    }
}

impl VerifiedDecision {
    fn check_source(&self, args: &Value) -> Result<SourceSnapshot, String> {
        let decision = &self.receipt.decision;
        if decision.target_tool != self.policy.name
            || decision.target_args != *args
            || runtime_ctx::effective_cwd().ok().as_ref() != Some(&decision.cwd)
        {
            return Err(
                "receipt does not match the tool, exact arguments, or working directory".into(),
            );
        }
        verify_receipt_source(&self.ledger, &self.receipt).map_err(|_| {
            format!(
                "receipt is stale: the source changed or could not be verified ({})",
                decision
                    .source
                    .paths()
                    .iter()
                    .map(|path| format!("{path:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    pub(in crate::ai) fn authorize_execution<'a>(
        &self,
        args: &'a Value,
    ) -> Result<AuthorizedDecision<'a>, String> {
        let source = self
            .check_source(args)
            .map_err(|reason| rejection(self.policy, args, &reason))?;
        Ok(AuthorizedDecision {
            policy: self.policy,
            args,
            source,
        })
    }
}

pub(in crate::ai) fn verify_before_dispatch(
    policy: &'static ToolIrreversibleRegistration,
    args: &mut Value,
) -> Result<VerifiedDecision, String> {
    let result: Result<VerifiedDecision, String> = (|| {
        let id = args
            .as_object_mut()
            .and_then(|args| args.remove(RECEIPT_ID_ARGUMENT))
            .and_then(|id| id.as_str().map(str::to_owned))
            .ok_or("missing decision_receipt_id".to_owned())?;
        let ledger = Ledger::current().ok_or("no active session ledger".to_owned())?;
        let receipt = ledger
            .read(&id)
            .map_err(|_| "receipt not found or invalid for this session".to_owned())?;
        let verified = VerifiedDecision {
            policy,
            ledger,
            receipt,
        };
        verified.check_source(args)?;
        Ok(verified)
    })();
    result.map_err(|reason| rejection(policy, args, &reason))
}

fn rejection(policy: &ToolIrreversibleRegistration, args: &Value, reason: &str) -> String {
    const REPORT_MISSING: &str = "No source target exists. Report the intended target as missing; do not retry the deletion while it is unavailable.";
    const REPORT_UNREADABLE: &str = "The intended source or a companion entry is unreadable or unverifiable. Report the source as unavailable; do not retry until it can be verified.";
    let unavailable = policy.sources.and_then(|resolve| match resolve(args) {
        Ok(paths) => {
            let sources = Sources::ToolTargets(paths);
            if sources.tool_targets_missing().is_ok_and(|missing| missing) {
                Some(REPORT_MISSING)
            } else if sources.fingerprint().is_err() {
                Some(REPORT_UNREADABLE)
            } else {
                None
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(REPORT_MISSING),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {
            Some("Correct the target arguments before attempting to verify them again.")
        }
        Err(_) => Some(REPORT_UNREADABLE),
    });
    let name = policy.name;
    if let Some(hint) = unavailable {
        return format!("Error: '{name}' was not dispatched: {reason}. {hint}");
    }
    let source_hint = if policy.sources.is_none() {
        " and the original source_path"
    } else {
        ""
    };
    format!(
        "Error: '{name}' was not dispatched: {reason}. To re-verify, call decision_receipt(op=snapshot) with \
         the exact intended target arguments{source_hint}, narrowly re-read the returned source paths, \
         and verify the selection. Then call decision_receipt(op=write) with that fingerprint and retry \
         with its decision_receipt_id. If needed, enable decision_receipt via enable_tools."
    )
}

/// Only navigation and already-known invalidations enter the model prefix. Live checks run
/// on list and dispatch, so rendering previews never hashes source data or claims freshness.
pub(in crate::ai) fn projection() -> Option<String> {
    projection_for(&Ledger::current()?)
}

fn projection_for(ledger: &Ledger) -> Option<String> {
    let ids = ledger.recent_ids(None, PROJECTION_LIMIT).ok()?;
    let rows = ids
        .iter()
        .filter_map(|id| ledger.read(id).ok())
        .filter(|receipt| {
            identifier(&receipt.decision.kind) && identifier(&receipt.decision.target_tool)
        })
        .map(|receipt| {
            let status = match ledger.known_invalidation(&receipt) {
                Some(Freshness::Changed) => "changed",
                Some(Freshness::Stale) => "stale",
                _ => "unchecked",
            };
            format!(
                "- {}: {} ({}, {status})",
                receipt.id, receipt.decision.target_tool, receipt.decision.kind
            )
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return None;
    }
    Some(format!(
        "## Decision Receipts\nSession-scoped navigation, not verified facts. Retrieve exact arguments, decision \
         parameters, and source freshness with `decision_receipt` (op=list, id=...). These locators do not assert freshness.\n{}",
        rows.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File, FileTimes};

    fn call(ledger: &Ledger, args: Value) -> Value {
        serde_json::from_str(&execute_in(ledger, &args).unwrap()).unwrap()
    }

    #[test]
    fn decision_receipt_pagination_reaches_older_entries_through_corrupt_receipts() {
        let root = std::env::temp_dir().join(format!("receipt-page-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let ledger = Ledger::new(&root.join("test.sqlite"), "test");
        let data = root.join("source.txt");
        fs::write(&data, "verified source").unwrap();
        let mut ids = (0..LIST_LIMIT + 2)
            .map(|_| {
                let source = Sources::File(data.clone());
                ledger
                    .write(Decision {
                        kind: "selection".into(),
                        target_tool: "example_create".into(),
                        target_args: json!({}),
                        cwd: root.clone(),
                        source_fingerprint: source.fingerprint().unwrap(),
                        source,
                        numeric_params: json!({}),
                        selected_ids: Vec::new(),
                    })
                    .unwrap()
                    .id
            })
            .collect::<Vec<_>>();
        ids.sort_unstable_by(|a, b| b.cmp(a));
        // Corrupt page one and its lookahead; only the last receipt is readable.
        for id in &ids[..=LIST_LIMIT] {
            fs::write(
                root.join("test.assets/decision-receipts")
                    .join(format!("{id}.json")),
                "not valid JSON",
            )
            .unwrap();
        }
        let first = call(&ledger, json!({"op": "list"}));
        let rows = first["receipts"].as_array().unwrap();
        assert_eq!(rows.len(), LIST_LIMIT);
        assert!(
            rows.iter()
                .zip(&ids)
                .all(|(row, id)| { row["id"].as_str() == Some(id) && row["error"].is_string() })
        );
        assert_eq!(first["next_before"], ids[LIST_LIMIT - 1]);
        let second = call(
            &ledger,
            json!({"op": "list", "before": first["next_before"]}),
        );
        assert_eq!(second["receipts"].as_array().unwrap().len(), 2);
        assert_eq!(second["receipts"][0]["id"], ids[LIST_LIMIT]);
        assert!(second["receipts"][0]["error"].is_string());
        assert_eq!(second["receipts"][1]["id"], ids[LIST_LIMIT + 1]);
        assert_eq!(second["receipts"][1]["freshness"], "fresh");
        assert!(second["next_before"].is_null());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn decision_receipt_detects_symlink_retarget_and_restore() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("receipt-symlink-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let first = root.join("first.csv");
        let other = root.join("other.csv");
        let link = root.join("scores.csv");
        fs::write(&first, "a,9\n").unwrap();
        fs::write(&other, "b,9\n").unwrap();
        symlink(&first, &link).unwrap();
        let ledger = Ledger::new(&root.join("test.sqlite"), "test");
        let mut args = json!({"op": "snapshot", "target_tool": "example_create", "target_args": {"id": "a"}, "source_path": link});
        let snapshot = call(&ledger, args.clone());
        args["op"] = json!("write");
        args["kind"] = json!("ranked_selection");
        args["source_fingerprint"] = snapshot["source_fingerprint"].clone();
        let saved = call(&ledger, args);

        // Keep the original link inode alive so recreating the same link name
        // cannot accidentally reuse it, and do not inspect the intermediate state.
        fs::rename(&link, root.join("old-scores.csv")).unwrap();
        symlink(&other, &link).unwrap();
        fs::remove_file(&link).unwrap();
        symlink(&first, &link).unwrap();
        assert_eq!(
            fs::canonicalize(&link).unwrap(),
            fs::canonicalize(&first).unwrap()
        );
        assert_eq!(
            call(&ledger, json!({"op": "list", "id": saved["id"]}))["freshness"],
            "changed"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decision_receipt_persists_and_detects_mtime_restored_rewrites() {
        let root = std::env::temp_dir().join(format!("decision-receipt-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let history = root.join("test.sqlite");
        let data = root.join("scores.csv");
        fs::write(&data, "id,score\na,9\nb,1\n").unwrap();
        let original_time = fs::metadata(&data).unwrap().modified().unwrap();
        let ledger = Ledger::new(&history, "test");
        let mut args = json!({"op": "snapshot", "target_tool": "example_create", "target_args": {"id": "a"}, "source_path": data});
        let snapshot = call(&ledger, args.clone());
        args["op"] = json!("write");
        args["kind"] = json!("ranked_selection");
        args["source_fingerprint"] = snapshot["source_fingerprint"].clone();
        args["numeric_params"] = json!({"sort_direction": -1, "tie_rule": 1});
        args["selected_ids"] = json!(["a"]);
        let saved = call(&ledger, args.clone());
        let id = saved["id"].as_str().unwrap();
        assert!(
            root.join("test.assets/decision-receipts")
                .join(format!("{id}.json"))
                .is_file()
        );
        drop(ledger);

        let ledger = Ledger::new(&history, "test");
        let page = call(&ledger, json!({"op": "list"}));
        assert_eq!(page["receipts"][0]["id"], id);
        assert_eq!(page["receipts"][0]["freshness"], "fresh");
        assert!(
            Ledger::new(&root.join("child.sqlite"), "test")
                .read(id)
                .is_err()
        );
        let detail = call(&ledger, json!({"op": "list", "id": id}));
        assert_eq!(detail["receipt"]["numeric_params"]["sort_direction"], -1);
        assert_eq!(detail["receipt"]["selected_ids"], json!(["a"]));
        let projected = projection_for(&ledger).unwrap();
        assert!(projected.contains(id));
        assert!(!projected.contains("sort_direction"));

        let original = fs::read_to_string(&data).unwrap();
        fs::write(&data, "id,score\na,1\nb,9\n").unwrap();
        File::options()
            .write(true)
            .open(&data)
            .unwrap()
            .set_times(FileTimes::new().set_modified(original_time))
            .unwrap();
        assert!(
            execute_in(&ledger, &args)
                .unwrap_err()
                .contains("source changed")
        );
        assert_eq!(
            call(&ledger, json!({"op": "list"}))["receipts"][0]["freshness"],
            "changed"
        );
        fs::write(&data, original).unwrap();
        File::options()
            .write(true)
            .open(&data)
            .unwrap()
            .set_times(FileTimes::new().set_modified(original_time))
            .unwrap();
        assert_eq!(
            call(&ledger, json!({"op": "list", "id": id}))["freshness"],
            "changed"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
