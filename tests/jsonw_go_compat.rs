use rust_tools::jsonw::{Json, ParseOptions, diff_json, sanitize_json_input};

#[test]
fn test_parse_with_comments_and_control_chars() {
    let s = "{\n  \"a\": 1, // comment\n  \"b\": \"x//y\"\n}\u{0001}";
    let sanitized = sanitize_json_input(s, ParseOptions::default());
    let j = Json::from_str(
        &sanitized,
        ParseOptions {
            allow_comment: false,
            remove_special_chars: false,
        },
    )
    .unwrap();
    assert_eq!(j.value()["a"], 1);
    assert_eq!(j.value()["b"], "x//y");
}

#[test]
fn test_diff_json_sort_arrays() {
    let old = serde_json::json!({"a":[3,2,1]});
    let new = serde_json::json!({"a":[1,2,4]});
    let diff = diff_json(&old, &new, true);
    assert!(diff.iter().any(|d| d.key == "a.2"));
}

#[test]
fn test_object_set_get_keys_contains() {
    let mut j = Json::new(serde_json::json!({}));
    assert!(!j.set("a", 1));
    assert!(j.contains_key("a"));
    assert_eq!(j.get_i64("a"), 1);
    assert_eq!(j.keys(), vec!["a"]);
}

#[test]
fn test_array_add_get_index_extract_abskey() {
    let mut j = Json::new(serde_json::Value::Null);
    assert!(j.add(serde_json::json!({"x": 1})));
    assert!(j.add(serde_json::json!({"x": 2})));

    let x0 = j.get_index(0).unwrap();
    assert_eq!(x0.get_i64("x"), 1);

    let extracted = j.extract("x");
    assert_eq!(extracted.value(), &serde_json::json!([1, 2]));

    let paths = j.abs_key("x");
    assert!(paths.iter().any(|p| p.ends_with("->x")));
}
