use super::super::builder::{
    RequestModelInputs, build_request_body_from_parts, resolve_wire_enable_search_from_parts,
};
use super::super::protocol::{
    build_http_body_for_json_messages_from_parts, build_http_body_for_request_from_parts,
};
use crate::ai::history::Message;
use crate::ai::model_names::ReasoningEffortWire;
use crate::ai::provider::{ALIBABA_DEFAULT_ENDPOINT, ApiProvider, adapter_for};
use crate::ai::request_protocol::RequestProtocolDialect;
use serde_json::{Value, json};

const MODEL: &str = "synthetic-compatible-wire";
const PLAIN_ENDPOINT: &str = "https://gateway.example/v1/chat/completions";
const MIXED_CASE_DASHSCOPE: &str =
    "  HTTPS://DASHSCOPE.ALIYUNCS.COM/compatible-mode/v1/chat/completions  ";
const OVERLAPPING_ENDPOINT: &str =
    "  HTTPS://OPENROUTER.AI/api/v1/chat/completions?upstream=DASHSCOPE.ALIYUNCS.COM  ";
const DEEPSEEK_ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
const SEARCH: [Option<bool>; 3] = [Some(true), Some(false), None];
const EFFORT: [Option<&str>; 3] = [Some("high"), Some("none"), None];
const OMIT: [&str; 3] = ["", "", ""];
const SEARCH_FIELDS: [&str; 3] = [r#","enable_search":true"#, r#","enable_search":false"#, ""];
const TOP_LEVEL_FIELDS: [&str; 3] = [
    r#","reasoning_effort":"high""#,
    r#","reasoning_effort":"none""#,
    "",
];
const NESTED_FIELDS: [&str; 3] = [
    r#","reasoning":{"effort":"high"}"#,
    r#","reasoning":{"effort":"none"}"#,
    "",
];
const RESPONSES_TOP_LEVEL_FIELDS: [&str; 3] = [
    r#","reasoning":{"effort":"high","summary":"auto"}"#,
    r#","reasoning":{"effort":"none","summary":"auto"}"#,
    "",
];
const DASHSCOPE_THINKING: [&str; 2] = [r#","enable_thinking":true"#, r#","enable_thinking":false"#];
const DEEPSEEK_THINKING: [&str; 2] = [
    r#","thinking":{"type":"enabled"}"#,
    r#","thinking":{"type":"disabled"}"#,
];

/// Literal wire expectations, independent of adapter and dialect helpers.
#[derive(Clone, Copy, Debug)]
struct WireFixture {
    adapter: ApiProvider,
    endpoint: &'static str,
    effort_wire: Option<ReasoningEffortWire>,
    thinking: [&'static str; 2],
    search: [&'static str; 3],
    effort: [&'static str; 3],
    responses_effort: [&'static str; 3],
}

const PLAIN: WireFixture = WireFixture {
    adapter: ApiProvider::Compatible,
    endpoint: PLAIN_ENDPOINT,
    effort_wire: None,
    thinking: ["", ""],
    search: OMIT,
    effort: TOP_LEVEL_FIELDS,
    responses_effort: RESPONSES_TOP_LEVEL_FIELDS,
};

const FIXTURES: &[WireFixture] = &[
    PLAIN,
    WireFixture {
        endpoint: ALIBABA_DEFAULT_ENDPOINT,
        thinking: DASHSCOPE_THINKING,
        search: SEARCH_FIELDS,
        effort: OMIT,
        responses_effort: OMIT,
        ..PLAIN
    },
    WireFixture {
        endpoint: MIXED_CASE_DASHSCOPE,
        thinking: DASHSCOPE_THINKING,
        search: SEARCH_FIELDS,
        effort: OMIT,
        responses_effort: OMIT,
        ..PLAIN
    },
    WireFixture {
        endpoint: MIXED_CASE_DASHSCOPE,
        effort_wire: Some(ReasoningEffortWire::TopLevel),
        thinking: DASHSCOPE_THINKING,
        search: SEARCH_FIELDS,
        ..PLAIN
    },
    WireFixture {
        effort_wire: Some(ReasoningEffortWire::Nested),
        effort: NESTED_FIELDS,
        responses_effort: NESTED_FIELDS,
        ..PLAIN
    },
    WireFixture {
        endpoint: OVERLAPPING_ENDPOINT,
        search: SEARCH_FIELDS,
        effort: NESTED_FIELDS,
        responses_effort: NESTED_FIELDS,
        ..PLAIN
    },
    WireFixture {
        endpoint: OVERLAPPING_ENDPOINT,
        effort_wire: Some(ReasoningEffortWire::TopLevel),
        search: SEARCH_FIELDS,
        ..PLAIN
    },
    WireFixture {
        adapter: ApiProvider::OpenAi,
        endpoint: MIXED_CASE_DASHSCOPE,
        thinking: DASHSCOPE_THINKING,
        effort: OMIT,
        responses_effort: OMIT,
        ..PLAIN
    },
    WireFixture {
        adapter: ApiProvider::OpenAi,
        ..PLAIN
    },
    WireFixture {
        adapter: ApiProvider::Alibaba,
        endpoint: OVERLAPPING_ENDPOINT,
        ..PLAIN
    },
    WireFixture {
        adapter: ApiProvider::Alibaba,
        endpoint: ALIBABA_DEFAULT_ENDPOINT,
        thinking: DASHSCOPE_THINKING,
        search: SEARCH_FIELDS,
        effort: OMIT,
        responses_effort: OMIT,
        ..PLAIN
    },
    WireFixture {
        adapter: ApiProvider::OpenCode,
        ..PLAIN
    },
    WireFixture {
        endpoint: DEEPSEEK_ENDPOINT,
        thinking: DEEPSEEK_THINKING,
        effort: OMIT,
        responses_effort: OMIT,
        ..PLAIN
    },
    WireFixture {
        endpoint: DEEPSEEK_ENDPOINT,
        effort_wire: Some(ReasoningEffortWire::TopLevel),
        thinking: DEEPSEEK_THINKING,
        ..PLAIN
    },
];

impl WireFixture {
    fn inputs(self, dialect: RequestProtocolDialect) -> RequestModelInputs {
        RequestModelInputs {
            request_model: MODEL.to_string(),
            endpoint: self.endpoint.to_string(),
            adapter: self.adapter,
            effort_wire: self.effort_wire,
            dialect,
            max_output: None,
            encrypted_replay: false,
            context_window: 256_000,
        }
    }
}

fn messages() -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }]
}

#[test]
fn main_body_bytes_and_preview_search_preserve_endpoint_precedence() {
    assert_eq!(
        adapter_for(ApiProvider::Compatible, OVERLAPPING_ENDPOINT).label(),
        "openrouter"
    );
    let messages = messages();
    for fixture in FIXTURES {
        let inputs = fixture.inputs(RequestProtocolDialect::ChatCompletions);
        for (thinking_index, enable_thinking) in [true, false].into_iter().enumerate() {
            for (search_index, enable_search) in SEARCH.into_iter().enumerate() {
                for (effort_index, effort) in EFFORT.into_iter().enumerate() {
                    let mut request = build_request_body_from_parts(
                        &inputs,
                        &messages,
                        false,
                        enable_thinking,
                        enable_search,
                        None,
                        None,
                        effort,
                        None,
                        None,
                        None,
                    );
                    let actual = build_http_body_for_request_from_parts(
                        inputs.adapter,
                        inputs.dialect,
                        &inputs.endpoint,
                        &mut request,
                    );
                    let expected = format!(
                        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":false{}{}{}}}"#,
                        fixture.thinking[thinking_index],
                        fixture.search[search_index],
                        fixture.effort[effort_index],
                    );
                    assert_eq!(actual, expected.as_bytes(), "{fixture:?}");
                    // Budget preview uses this same resolver without building a body.
                    assert_eq!(
                        resolve_wire_enable_search_from_parts(
                            inputs.adapter,
                            inputs.dialect,
                            &inputs.endpoint,
                            enable_search,
                        ),
                        request.enable_search,
                        "preview/body search parity: {fixture:?}",
                    );
                }
            }
        }
    }
}

#[test]
fn auxiliary_body_bytes_preserve_endpoint_and_declared_effort_precedence() {
    let messages = [json!({"role": "user", "content": "hi"})];
    for fixture in FIXTURES {
        let inputs = fixture.inputs(RequestProtocolDialect::ChatCompletions);
        for stream in [true, false] {
            for include_usage in [true, false] {
                for (effort_index, effort) in EFFORT.into_iter().enumerate() {
                    let actual = build_http_body_for_json_messages_from_parts(
                        &inputs,
                        &messages,
                        stream,
                        effort,
                        include_usage,
                    );
                    let usage = if stream && include_usage {
                        r#","stream_options":{"include_usage":true}"#
                    } else {
                        ""
                    };
                    let expected = format!(
                        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"stream":{stream}{}{}{usage}}}"#,
                        fixture.thinking[1], fixture.effort[effort_index],
                    );
                    assert_eq!(actual, expected.as_bytes(), "{fixture:?}");
                }
            }
        }
    }
}

#[test]
fn responses_body_bytes_preserve_the_caller_search_flag() {
    let messages = messages();
    for fixture in FIXTURES {
        let inputs = fixture.inputs(RequestProtocolDialect::Responses);
        for (search_index, enable_search) in SEARCH.into_iter().enumerate() {
            for (effort_index, effort) in EFFORT.into_iter().enumerate() {
                let mut request = build_request_body_from_parts(
                    &inputs,
                    &messages,
                    false,
                    true,
                    enable_search,
                    None,
                    None,
                    effort,
                    None,
                    None,
                    None,
                );
                let actual = build_http_body_for_request_from_parts(
                    inputs.adapter,
                    inputs.dialect,
                    &inputs.endpoint,
                    &mut request,
                );
                let search = [r#","tools":[{"type":"web_search"}]"#, "", ""][search_index];
                let expected = format!(
                    r#"{{"model":"{MODEL}","input":[{{"role":"user","content":[{{"type":"input_text","text":"hi"}}]}}],"stream":false{search}{}}}"#,
                    fixture.responses_effort[effort_index],
                );
                assert_eq!(actual, expected.as_bytes(), "{fixture:?}");
                assert_eq!(request.enable_search, enable_search, "{fixture:?}");
            }
        }
    }
}
