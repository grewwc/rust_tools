use super::*;

#[test]
fn normalize_tool_call_arguments_rejects_incomplete_json_and_canonicalizes_empty() {
    assert_eq!(normalize_tool_call_arguments(""), Some("{}".to_string()));
    assert_eq!(
        normalize_tool_call_arguments(" {\"command\":\"pwd\"} "),
        Some("{\"command\":\"pwd\"}".to_string())
    );
    assert_eq!(normalize_tool_call_arguments("{\"command\":"), None);
}

#[test]
fn collect_valid_tool_calls_reports_drop_on_incomplete_arguments() {
    let mut builders: rust_tools::cw::SkipMap<usize, ToolCallBuilder> =
        rust_tools::cw::SkipMap::default();
    // Simulate a large write_file hitting the output cap: arguments JSON cut in half and unrepairable.
    builders.insert(
        0,
        ToolCallBuilder {
            function_name: "write_file".to_string(),
            arguments: "{\"path\":\"/tmp/x\",\"content\":\"aaa".to_string(),
            ..Default::default()
        },
    );
    let (calls, dropped) = collect_valid_tool_calls(&mut builders);
    assert!(calls.is_empty(), "半截 JSON 应被丢弃");
    assert!(dropped, "发生丢弃时应返回 dropped=true");
}

#[test]
fn collect_valid_tool_calls_no_drop_on_valid_arguments() {
    let mut builders: rust_tools::cw::SkipMap<usize, ToolCallBuilder> =
        rust_tools::cw::SkipMap::default();
    builders.insert(
        0,
        ToolCallBuilder {
            function_name: "read_file".to_string(),
            arguments: "{\"path\":\"/tmp/x\"}".to_string(),
            ..Default::default()
        },
    );
    let (calls, dropped) = collect_valid_tool_calls(&mut builders);
    assert_eq!(calls.len(), 1);
    assert!(!dropped, "合法 JSON 不应触发 dropped");
}

#[test]
fn recover_inline_tool_calls_handles_bare_object() {
    // Simulate qwen3.7-max emitting a tool call as content.
    let raw = r#"{"name":"read_file","arguments":{"path":"/tmp/x"}}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
    assert_eq!(calls[0].tool_type, "function");
}

#[test]
fn recover_inline_tool_calls_handles_arguments_as_json_string() {
    let raw = r#"{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_fenced_code_block() {
    let raw = "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp/x\"}}\n```";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].function.name, "read_file");
}

#[test]
fn recover_inline_tool_calls_handles_tool_call_xml_wrapper() {
    let raw = r#"<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].function.name, "read_file");
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_json_body() {
    // The Hermes/Qwen XML shape the model actually emitted in the screenshot (body is JSON).
    let raw = "<tool_call>\n<function=read_file>\n{\"path\":\"/tmp/x\"}\n</function>\n</tool_call>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_parameter_tags() {
    let raw = "<function=read_file><parameter=path>/tmp/x</parameter><parameter=limit>200</parameter></function>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    // Numeric arguments must be recognized as JSON numbers, not strings.
    assert_eq!(args["limit"], 200);
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_no_args() {
    let raw = "<tool_call><function=list_agents></function></tool_call>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_agents");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_parallel_calls() {
    let raw = "<function=read_file>{\"path\":\"/a\"}</function><function=read_file>{\"path\":\"/b\"}</function>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.arguments, r#"{"path":"/a"}"#);
    assert_eq!(calls[1].function.arguments, r#"{"path":"/b"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_array_of_calls() {
    let raw = r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "a");
    assert_eq!(calls[1].function.name, "b");
    assert_eq!(calls[1].function.arguments, r#"{"x":1}"#);
}

#[test]
fn recover_inline_tool_calls_handles_openai_function_wrapper() {
    let raw = r#"{"id":"call_123","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].id, "call_123");
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_rejects_plain_text() {
    // Plain text answers must never be misidentified as a tool call.
    assert!(recover_inline_tool_calls("Hello world").is_none());
    assert!(recover_inline_tool_calls("").is_none());
    // name without arguments, and name not in the known object set — strictly this should not parse,
    // but for compatibility we still recognize a bare name on its own. Below are the true negative samples:
    assert!(recover_inline_tool_calls("{\"foo\":\"bar\"}").is_none());
    assert!(recover_inline_tool_calls("12345").is_none());
    // String-form args must themselves be valid JSON, otherwise reject.
    assert!(recover_inline_tool_calls(r#"{"name":"x","arguments":"not-json"}"#).is_none());
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_parameter_tags() {
    // The Anthropic style deepseek-v4-flash actually emits: <invoke name=...>/<parameter name=...>.
    let raw = r#"<function_calls><invoke name="read_file"><parameter name="path">/tmp/x</parameter><parameter name="limit">200</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    assert_eq!(args["limit"], 200);
}

#[test]
fn recover_inline_tool_calls_respects_anthropic_string_attr() {
    // string="true" must keep values that look like JSON scalars (true/123/null) as strings,
    // instead of auto-parsing them into bool/number. string="false" parses as JSON as usual.
    let raw = r#"<function_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="dry_run" string="true">true</parameter><parameter name="count" string="true">123</parameter><parameter name="tools" string="false">["a","b"]</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["operation"], "enable");
    assert_eq!(
        args["dry_run"], "true",
        "string=\"true\" must keep 'true' as string"
    );
    assert_eq!(
        args["count"], "123",
        "string=\"true\" must keep '123' as string"
    );
    assert_eq!(args["tools"][0], "a");
    assert_eq!(args["tools"][1], "b");
}

#[test]
fn recover_inline_tool_calls_matches_anthropic_string_attr_by_exact_name() {
    let raw = r#"<function_calls><invoke name="enable_tools"><parameter name="string_value" string="true">123</parameter><parameter name="count" notstring="true">456</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("expected recovered tool call");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["string_value"], "123");
    assert_eq!(args["count"], 456);
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_namespaced_tags() {
    // With a namespace prefix (antml:) and no outer wrapper.
    let raw = r#"<invoke name="list_agents"></invoke>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_agents");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn recover_inline_tool_calls_respects_anthropic_xml_string_attr() {
    // DSML `string="true"`: the value must stay a string even when it looks like a JSON scalar.
    // Covers the output shape the user reported from deepseek with MCP tools enabled.
    let raw = r#"<tool_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="tools" string="false">["mcp_excel_open_workbook"]</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "enable_tools");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // string="true" -> "enable" must remain a string, never treated as an identifier.
    assert_eq!(args["operation"], "enable");
    assert!(
        args["operation"].is_string(),
        "string=\"true\" 必须保持字符串"
    );
    // string="false" -> the array JSON parses normally.
    assert_eq!(args["tools"][0], "mcp_excel_open_workbook");
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_parallel_calls() {
    let raw = r#"<tool_calls><invoke name="read_file"><parameter name="path">/a</parameter></invoke><invoke name="read_file"><parameter name="path">/b</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let b: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a["path"], "/a");
    assert_eq!(b["path"], "/b");
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_string_true_attr() {
    // DSML `string="true"`: the value stays a string even when it looks like a JSON scalar.
    let raw = r#"<tool_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="tools" string="false">["read_file","write_file"]</parameter><parameter name="verbose" string="true">true</parameter><parameter name="count" string="true">123</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "enable_tools");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["operation"], "enable", "string=true -> 字符串");
    assert!(args["operation"].is_string());
    assert_eq!(
        args["tools"],
        serde_json::json!(["read_file", "write_file"]),
        "string=false -> 原生 JSON 数组"
    );
    assert_eq!(
        args["verbose"], "true",
        "看起来像 bool 但 string=true -> 字符串 \"true\""
    );
    assert!(args["verbose"].is_string());
    assert_eq!(
        args["count"], "123",
        "看起来像数字但 string=true -> 字符串 \"123\""
    );
    assert!(args["count"].is_string());
}

#[test]
fn recover_inline_tool_calls_handles_bare_registered_xml_with_raw_string_body() {
    let raw = r#"<execute_command>cd /tmp && pwd</execute_command>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover bare xml tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "execute_command");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "cd /tmp && pwd");
}

#[test]
fn anthropic_xml_streamer_suppresses_markup_and_emits_events() {
    let mut streamer = super::super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let (cleaned, events) = streamer.push(
            r#"Let me check.<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>"#,
        );
    // The invoke markers stay hidden; only the leading prose is kept.
    assert_eq!(cleaned, "Let me check.");
    // Emits Begin/Args/End events, consistent with the internal tool_call pipeline.
    assert_eq!(events.len(), 3);
    match (&events[0], &events[1], &events[2]) {
        (
            InternalToolCallStreamEvent::Begin(name),
            InternalToolCallStreamEvent::Args(args),
            InternalToolCallStreamEvent::End,
        ) => {
            assert_eq!(name, "read_file");
            let v: serde_json::Value = serde_json::from_str(args).unwrap();
            assert_eq!(v["path"], "/tmp/x");
        }
        _ => panic!("unexpected events: {events:?}"),
    }
}

#[test]
fn anthropic_xml_streamer_handles_split_chunks() {
    let mut streamer = super::super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let mut all_events = Vec::new();
    let mut all_cleaned = String::new();
    for chunk in [
        "pre <inv",
        "oke name=\"read_file\"><parameter name=\"pa",
        "th\">/tmp/x</parameter></in",
        "voke> post",
    ] {
        let (cleaned, events) = streamer.push(chunk);
        all_cleaned.push_str(&cleaned);
        all_events.extend(events);
    }
    assert_eq!(all_cleaned, "pre  post");
    assert_eq!(all_events.len(), 3);
    match &all_events[0] {
        InternalToolCallStreamEvent::Begin(name) => assert_eq!(name, "read_file"),
        other => panic!("unexpected first event: {other:?}"),
    }
}

#[test]
fn anthropic_xml_streamer_leaves_prose_angle_brackets_intact() {
    let mut streamer = super::super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let (cleaned, events) = streamer.push("a < b and c > d, also <div> here");
    assert_eq!(cleaned, "a < b and c > d, also <div> here");
    assert!(events.is_empty());
}
