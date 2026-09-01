//! Regression tests for the manifest-driven validated-claims protocol.

use super::super::*;
use super::common::*;

fn user_message() -> Message {
    Message {
        role: "user".to_string(),
        content: Value::String("Compare the two artifacts".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn two_artifact_turn(left: &str, right: &str) -> Vec<Message> {
    vec![
        user_message(),
        assistant_tool_call_message(test_tool_call(
            "call_left",
            TEST_REPLAY_TOOL,
            serde_json::json!({"source": "left"}),
        )),
        tool_result_message("call_left", left),
        assistant_tool_call_message(test_tool_call(
            "call_right",
            TEST_REPLAY_TOOL,
            serde_json::json!({"source": "right"}),
        )),
        tool_result_message("call_right", right),
    ]
}

#[test]
fn two_distinct_read_only_artifacts_activate_manifest_protocol() {
    let turn = two_artifact_turn("requestId=abc", "requestId=abc");
    assert!(validated_claims_required(true, &turn));
    assert!(!validated_claims_required(false, &turn));
}

#[test]
fn registered_identity_key_can_prove_same_request() {
    let turn = two_artifact_turn("requestId=abc", "request_id=abc");
    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"left_request","artifact":"left","field":"requestId","value":"abc","evidence":"requestId=abc"},
            {"id":"right_request","artifact":"right","field":"request_id","value":"abc","evidence":"request_id=abc"}
        ],
        "comparisons":[{"left_fact":"left_request","right_fact":"right_request"}],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}],
        "open_questions":[],
        "coverage_gaps":[]
    }</validated_claims>"#
        .to_string();

    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Allow
    );
    assert!(final_text.contains("Request identity for Artifact 1 and Artifact 2 is **the same**"));
    assert!(final_text.contains("`requestid` = `abc`"));
}

#[test]
fn shared_non_identity_values_cannot_prove_same_request() {
    let turn = two_artifact_turn("source=sqlQueryEditor", "source=sqlQueryEditor");
    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"left_source","artifact":"left","field":"source","value":"sqlQueryEditor","evidence":"source=sqlQueryEditor"},
            {"id":"right_source","artifact":"right","field":"source","value":"sqlQueryEditor","evidence":"source=sqlQueryEditor"}
        ],
        "comparisons":[{"left_fact":"left_source","right_fact":"right_source"}],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}],
        "open_questions":["These may belong to one request."],
        "coverage_gaps":[]
    }</validated_claims>"#
        .to_string();

    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Allow
    );
    assert!(final_text.contains("`source` has the same value"));
    assert!(
        final_text
            .contains("Request identity for Artifact 1 and Artifact 2 is **not established**")
    );
    assert!(final_text.contains("**Unverified:** These may belong to one request."));
    assert!(!final_text.contains("identity for Artifact 1 and Artifact 2 is **the same**"));
}

#[test]
fn ungrounded_fact_reopens_once_then_withholds() {
    let turn = two_artifact_turn("requestId=abc", "requestId=def");
    let invalid = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"invented","artifact":"left","field":"requestId","value":"zzz","evidence":"requestId=zzz"}
        ],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}]
    }</validated_claims>"#;
    let mut messages = turn.clone();
    let mut first = invalid.to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut first, false, 1, 8),
        ValidatedClaimsGateAction::Reopen
    );
    assert!(current_turn_has_internal_marker(
        &messages,
        VALIDATED_CLAIMS_RETRY_MARKER
    ));

    let mut second = invalid.to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut second, false, 2, 8),
        ValidatedClaimsGateAction::Warn
    );
    assert!(second.contains(VALIDATED_CLAIMS_WARNING));
    assert!(!second.contains("zzz"));
}

#[test]
fn successful_mutation_disables_investigation_protocol() {
    let mut turn = two_artifact_turn("requestId=abc", "requestId=abc");
    turn.push(assistant_tool_call_message(test_tool_call(
        "call_write",
        "apply_patch",
        serde_json::json!({"patch": "*** Begin Patch"}),
    )));
    turn.push(tool_result_message(
        "call_write",
        "Successfully patched 1 file",
    ));

    assert!(!validated_claims_required(true, &turn));
}

#[test]
fn unclassified_tool_results_cannot_bypass_the_gate() {
    let turn = vec![
        user_message(),
        assistant_tool_call_message(test_tool_call(
            "call_left",
            "unclassified_inspector",
            serde_json::json!({"source": "left"}),
        )),
        tool_result_message("call_left", "requestId=abc"),
        assistant_tool_call_message(test_tool_call(
            "call_right",
            "unclassified_inspector",
            serde_json::json!({"source": "right"}),
        )),
        tool_result_message("call_right", "requestId=abc"),
    ];
    assert!(validated_claims_required(true, &turn));

    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[],
        "facts":[],
        "comparisons":[],
        "relations":[],
        "open_questions":[],
        "coverage_gaps":["The inspection tool is not admissible as runtime-verified evidence."]
    }</validated_claims>"#
        .to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Allow
    );
    assert!(final_text.contains("No admissible field facts were available."));
    assert!(final_text.contains("**Unverified gap:**"));
}

#[test]
fn field_name_substrings_do_not_validate_identity_facts() {
    let turn = two_artifact_turn("parentRequestId=abc", "parentRequestId=abc");
    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"left_request","artifact":"left","field":"requestId","value":"abc","evidence":"parentRequestId=abc"},
            {"id":"right_request","artifact":"right","field":"requestId","value":"abc","evidence":"parentRequestId=abc"}
        ],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}]
    }</validated_claims>"#
        .to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Reopen
    );
}

#[test]
fn conflicting_identity_aliases_remain_unknown() {
    let turn = two_artifact_turn("requestId=abc reqId=def", "requestId=abc");
    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"left_request","artifact":"left","field":"requestId","value":"abc","evidence":"requestId=abc"},
            {"id":"left_alias","artifact":"left","field":"reqId","value":"def","evidence":"reqId=def"},
            {"id":"right_request","artifact":"right","field":"requestId","value":"abc","evidence":"requestId=abc"}
        ],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}],
        "open_questions":[],
        "coverage_gaps":[]
    }</validated_claims>"#
        .to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Allow
    );
    assert!(
        final_text
            .contains("Request identity for Artifact 1 and Artifact 2 is **not established**")
    );
    assert!(!final_text.contains("is **the same**"));
}

#[test]
fn identity_value_must_match_the_complete_encoded_value() {
    let turn = two_artifact_turn(r#"requestId="abc-left""#, r#"requestId="abc-right""#);
    let mut messages = turn.clone();
    let mut final_text = r#"<validated_claims>{
        "protocol":"validated_claims/v1",
        "artifacts":[
            {"id":"left","tool_call_id":"call_left"},
            {"id":"right","tool_call_id":"call_right"}
        ],
        "facts":[
            {"id":"left_request","artifact":"left","field":"requestId","value":"abc","evidence":"requestId=\"abc-left\""},
            {"id":"right_request","artifact":"right","field":"requestId","value":"abc","evidence":"requestId=\"abc-right\""}
        ],
        "relations":[{"left_artifact":"left","right_artifact":"right","scope":"request"}]
    }</validated_claims>"#
        .to_string();
    assert_eq!(
        validated_claims_gate_action(true, &mut messages, &turn, &mut final_text, false, 1, 8),
        ValidatedClaimsGateAction::Reopen
    );
}
