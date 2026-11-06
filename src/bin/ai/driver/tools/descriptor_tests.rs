use super::{
    build_tool_cache_key, collect_tool_cache_file_fingerprints, normalize_for_validation,
    prepare_tool_call, should_store_or_load_tool_cache,
};
use crate::ai::mcp::McpClient;
use crate::ai::tools::registry::common::ToolRegistration;
use crate::ai::tools::registry::tool_metadata::tool_parameters;
use crate::ai::types::{FunctionCall, ToolCall};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
fn descriptor_aliases_preserve_validation_output_and_raw_arguments() {
    let mcp = McpClient::new();
    // Fixed expectations encode the pre-registration normalization contract.
    for (name, args, expected, accepted) in [
        (
            "read_file",
            json!({"path": "src/main.rs", "offset": null}),
            json!({"file_path": "src/main.rs"}),
            true,
        ),
        (
            "write_file",
            json!({"path": "src/main.rs", "content": "hi"}),
            json!({"file_path": "src/main.rs", "content": "hi"}),
            true,
        ),
        (
            "apply_patch",
            json!({"path": "target.rs", "patch": null, "patch_file": "/tmp/compat.patch"}),
            json!({"file_path": "target.rs", "patch_file": "/tmp/compat.patch"}),
            true,
        ),
        (
            "send_side_note",
            json!({"content": "steer left", "target_task_id": "first", "target": "second"}),
            json!({"note": "steer left", "task_id": "first"}),
            true,
        ),
        (
            "send_side_note",
            json!({"note": "steer left", "target": "second"}),
            json!({"note": "steer left", "task_id": "second"}),
            true,
        ),
        (
            "send_side_note",
            json!({"note": "canonical", "content": "alias", "task_id": "canonical", "target_task_id": "first", "target": "second"}),
            json!({"note": "canonical", "task_id": "canonical"}),
            true,
        ),
        (
            "send_side_note",
            json!({"content": "steer left", "task_id": null, "target_task_id": null, "target": "second"}),
            json!({"note": "steer left", "task_id": "second"}),
            true,
        ),
        (
            "read_file",
            json!({"file_path": "canonical.rs", "path": "alias.rs"}),
            json!({"file_path": "canonical.rs"}),
            true,
        ),
        (
            "read_file",
            json!({"file_path": null, "path": "alias.rs"}),
            json!({"file_path": null}),
            false,
        ),
        ("read_file", json!({"path": null}), json!({}), false),
        (
            "read_file",
            json!(["src/main.rs"]),
            json!(["src/main.rs"]),
            false,
        ),
    ] {
        let original = args.clone();
        assert_eq!(
            normalize_for_validation(name, &args, &tool_parameters(name)),
            expected,
            "{name}: {args}"
        );
        assert_eq!(args, original);
        let call = ToolCall {
            id: "descriptor-alias-test".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        };
        let prepared = prepare_tool_call(&mcp, &call, None);
        if accepted {
            assert_eq!(prepared.unwrap().args, original, "{name}");
        } else {
            assert!(prepared.is_err(), "{name}: {args}");
        }
    }
    // Persistent keys use raw spellings, even when validation equates them.
    assert_ne!(
        build_tool_cache_key("read_file", &json!({"path": "source.rs"})),
        build_tool_cache_key("read_file", &json!({"file_path": "source.rs"}))
    );
}

#[test]
fn descriptor_defaults_have_no_aliases_or_persistent_cache() {
    let fixture = CacheFixture::new();
    let args = json!({
        "path": fixture.0.join("source.txt"),
        "content": "note",
        "target_task_id": "first",
        "target": "second",
    });
    let schema = json!({
        "type": "object",
        "properties": {"file_path": {}, "note": {}, "task_id": {}},
    });
    for name in [
        "unregistered_read_file",
        "READ_FILE",
        "mcp_example_read_file",
        "knowledge_search",
        "tree",
        "read_mailbox",
    ] {
        assert_eq!(
            normalize_for_validation(name, &args, &schema),
            args,
            "{name}"
        );
        assert!(!should_store_or_load_tool_cache(name, &args), "{name}");
        assert!(collect_tool_cache_file_fingerprints(name, &args).is_empty());
    }
}

#[test]
fn descriptor_cache_qualification_preserves_file_and_argument_precedence() {
    let fixture = CacheFixture::new();
    let source = fixture.0.join("source.txt");
    let missing = fixture.0.join("missing.txt");
    for (args, eligible) in [
        (json!({"file_path": source}), true),
        (json!({"path": source}), true),
        (json!({"file_path": source, "path": missing}), true),
        (json!({"file_path": fixture.0.join("empty.txt")}), true),
        (json!({"file_path": missing}), false),
        (json!({"file_path": fixture.0}), false),
        (json!({"file_path": missing, "path": source}), false),
        (json!({"file_path": null, "path": source}), false),
        (json!({"file_path": 7, "path": source}), false),
        (json!({"file_path": "", "path": source}), false),
        (json!({}), false),
    ] {
        assert_eq!(
            should_store_or_load_tool_cache("read_file", &args),
            eligible,
            "{args}"
        );
        assert_eq!(
            collect_tool_cache_file_fingerprints("read_file", &args).len(),
            usize::from(eligible)
        );
    }
    let args = json!({"file_path": source});
    for reg in inventory::iter::<ToolRegistration> {
        assert_eq!(
            should_store_or_load_tool_cache(reg.spec.name, &args),
            reg.spec.name == "read_file",
            "{} must retain its existing persistent-cache eligibility",
            reg.spec.name
        );
    }

    let modified = UNIX_EPOCH + Duration::from_secs(123);
    fs::File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let raw_path = fixture.0.join(".").join("source.txt");
    let fingerprints =
        collect_tool_cache_file_fingerprints("read_file", &json!({"path": raw_path}));
    assert_eq!(fingerprints.len(), 1);
    assert_eq!(fingerprints[0].path, raw_path.to_string_lossy());
    assert_eq!(fingerprints[0].size, 5);
    assert_eq!(fingerprints[0].modified_ms, Some(123_000));
}

struct CacheFixture(PathBuf);

impl CacheFixture {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let fixture = Self(std::env::temp_dir().join(format!(
            "rust_tools_descriptor_{}_{nanos}",
            std::process::id()
        )));
        fs::create_dir(&fixture.0).unwrap();
        fs::write(fixture.0.join("source.txt"), "hello").unwrap();
        fs::write(fixture.0.join("empty.txt"), "").unwrap();
        fixture
    }
}

impl Drop for CacheFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
