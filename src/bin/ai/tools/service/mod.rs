use serde_json::Value;

pub(crate) mod audit;
pub(crate) mod changes;
pub(crate) mod command;
pub(crate) mod command_session;
pub(crate) mod discuss;
pub(crate) mod file;
pub(crate) mod memory;
pub(crate) mod reads;
pub(crate) mod session_distill;
pub(crate) mod side_note;
pub(crate) mod speech;
pub(crate) mod talk;
pub(crate) mod task;

/// Parse a count-like argument as an unsigned integer within `min..=max`.
///
/// Tool schemas describe counts as integers, so some models emit `2.0`; accept
/// finite integral floats and decimal strings, but never truncate fractional
/// or negative values. A present-but-invalid value is an error, not a missing
/// value: callers must not turn "received 2.5" into "no count supplied".
pub(crate) fn arg_u64(
    args: &Value,
    keys: &[&str],
    min: u64,
    max: u64,
) -> Result<Option<u64>, String> {
    for key in keys {
        let Some(value) = args.get(*key) else {
            continue;
        };
        // Optional fields are commonly sent as null; treat that as absent so
        // an alias can still supply the value.
        if value.is_null() {
            continue;
        }
        let Some(n) = value_as_u64(value) else {
            return Err(format!(
                "{key} must be an integer between {min} and {max} (received {value})"
            ));
        };
        if n < min || n > max {
            return Err(format!(
                "{key} must be an integer between {min} and {max} (received {value})"
            ));
        }
        return Ok(Some(n));
    }
    Ok(None)
}

/// Parse a required-or-optional string argument.
///
/// A present non-string value is an error; null and empty strings are treated
/// as absent so aliases can still supply a value.
pub(crate) fn arg_str<'a>(args: &'a Value, keys: &[&str]) -> Result<Option<&'a str>, String> {
    for key in keys {
        let Some(value) = args.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let Some(s) = value.as_str() else {
            return Err(format!("{key} must be a string (received {value})"));
        };
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        return Ok(Some(s));
    }
    Ok(None)
}

/// Parse an array of strings, rejecting non-array containers and non-string
/// items instead of silently dropping them.
///
/// `label` names the canonical argument in errors. Empty arrays and nulls are
/// absent; when `reject_empty` is set, empty items are errors too.
pub(crate) fn arg_string_list(
    args: &Value,
    keys: &[&str],
    label: &str,
    reject_empty: bool,
) -> Result<Option<Vec<String>>, String> {
    for key in keys {
        let Some(value) = args.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let Some(arr) = value.as_array() else {
            return Err(format!(
                "{label} must be an array of strings (received {value})"
            ));
        };
        let mut out = Vec::with_capacity(arr.len());
        for (i, item) in arr.iter().enumerate() {
            let Some(s) = item.as_str() else {
                return Err(format!("{label}[{i}] must be a string (received {item})"));
            };
            let s = s.trim();
            if reject_empty && s.is_empty() {
                return Err(format!("{label}[{i}] must not be empty"));
            }
            out.push(s.to_string());
        }
        // An empty array is "absent", so an alias can still supply a value,
        // matching the null and empty-string behavior of the other helpers.
        if out.is_empty() {
            continue;
        }
        return Ok(Some(out));
    }
    Ok(None)
}

/// Parse a boolean flag; wrong types are errors, not defaults.
pub(crate) fn arg_bool(args: &Value, key: &str, default: bool) -> Result<bool, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(other) => Err(format!("{key} must be a boolean (received {other})")),
    }
}

fn value_as_u64(value: &Value) -> Option<u64> {
    if let Some(n) = value.as_u64() {
        return Some(n);
    }
    if let Some(n) = value.as_i64() {
        return u64::try_from(n).ok();
    }
    if let Some(f) = value.as_f64() {
        return integral_float_to_u64(f);
    }
    let text = value.as_str()?.trim();
    if let Ok(n) = text.parse::<u64>() {
        return Some(n);
    }
    integral_float_to_u64(text.parse::<f64>().ok()?)
}

fn integral_float_to_u64(value: f64) -> Option<u64> {
    // `u64::MAX as f64` rounds up to 2^64; require strictly below it so the
    // cast cannot saturate.
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64 {
        Some(value as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{arg_bool, arg_str, arg_string_list, arg_u64};

    #[test]
    fn arg_u64_accepts_integral_numbers_and_strings() {
        let keys = ["worker_count"];
        assert_eq!(arg_u64(&json!({"worker_count": 2}), &keys, 1, 8), Ok(Some(2)));
        assert_eq!(arg_u64(&json!({"worker_count": 2.0}), &keys, 1, 8), Ok(Some(2)));
        assert_eq!(arg_u64(&json!({"worker_count": "2"}), &keys, 1, 8), Ok(Some(2)));
        assert_eq!(arg_u64(&json!({"worker_count": "2.0"}), &keys, 1, 8), Ok(Some(2)));
        assert_eq!(arg_u64(&json!({"worker_count": " 3 "}), &keys, 1, 8), Ok(Some(3)));
    }

    #[test]
    fn arg_u64_rejects_invalid_values_instead_of_treating_them_as_missing() {
        let keys = ["worker_count"];
        for args in [
            json!({"worker_count": 2.5}),
            json!({"worker_count": "2.5"}),
            json!({"worker_count": -1}),
            json!({"worker_count": "-1.0"}),
            json!({"worker_count": true}),
            json!({"worker_count": "many"}),
            json!({"worker_count": ""}),
        ] {
            let err = arg_u64(&args, &keys, 1, 8).unwrap_err();
            assert!(err.contains("worker_count"), "{err}");
            assert!(err.contains("between 1 and 8"), "{err}");
        }
        assert_eq!(arg_u64(&json!({"worker_count": null}), &keys, 1, 8), Ok(None));
        let err = arg_u64(&json!({"worker_count": 2.5}), &keys, 1, 8).unwrap_err();
        assert!(err.contains("received 2.5"), "{err}");
    }

    #[test]
    fn arg_u64_rejects_out_of_range_values() {
        let keys = ["worker_count"];
        let err = arg_u64(&json!({"worker_count": 0}), &keys, 1, 8).unwrap_err();
        assert!(err.contains("received 0"), "{err}");
        let err = arg_u64(&json!({"worker_count": 9}), &keys, 1, 8).unwrap_err();
        assert!(err.contains("received 9"), "{err}");
        assert_eq!(arg_u64(&json!({"worker_count": 8}), &keys, 1, 8), Ok(Some(8)));
    }

    #[test]
    fn arg_u64_checks_alias_only_when_canonical_is_absent_or_null() {
        let keys = ["worker_count", "count"];
        // A present-but-invalid canonical key must not be masked by an alias.
        assert!(arg_u64(&json!({"worker_count": 2.5, "count": 3}), &keys, 1, 8).is_err());
        assert_eq!(
            arg_u64(&json!({"worker_count": null, "count": 3}), &keys, 1, 8),
            Ok(Some(3))
        );
        assert_eq!(arg_u64(&json!({}), &keys, 1, 8), Ok(None));
    }

    #[test]
    fn arg_str_rejects_non_string_and_treats_null_or_empty_as_absent() {
        assert_eq!(arg_str(&json!({"goal": "  x  "}), &["goal"]), Ok(Some("x")));
        let err = arg_str(&json!({"goal": 123}), &["goal"]).unwrap_err();
        assert!(err.contains("goal must be a string"), "{err}");
        assert_eq!(arg_str(&json!({"goal": null}), &["goal"]), Ok(None));
        assert_eq!(
            arg_str(&json!({"goal": "", "task": "x"}), &["goal", "task"]),
            Ok(Some("x"))
        );
    }

    #[test]
    fn arg_string_list_rejects_bad_containers_and_items() {
        assert_eq!(
            arg_string_list(&json!({"workers": ["a", " b "]}), &["workers"], "workers", true),
            Ok(Some(vec!["a".to_string(), "b".to_string()]))
        );
        let err =
            arg_string_list(&json!({"workers": "abc"}), &["workers"], "workers", true).unwrap_err();
        assert!(err.contains("workers must be an array of strings"), "{err}");
        let err = arg_string_list(&json!({"workers": ["a", 123]}), &["workers"], "workers", true)
            .unwrap_err();
        assert!(err.contains("workers[1] must be a string"), "{err}");
        let err = arg_string_list(&json!({"workers": ["a", ""]}), &["workers"], "workers", true)
            .unwrap_err();
        assert!(err.contains("workers[1] must not be empty"), "{err}");
        assert_eq!(
            arg_string_list(&json!({"roles": ["pro", ""]}), &["roles"], "roles", false),
            Ok(Some(vec!["pro".to_string(), String::new()]))
        );
        assert_eq!(
            arg_string_list(&json!({"workers": []}), &["workers"], "workers", true),
            Ok(None)
        );
        assert_eq!(
            arg_string_list(
                &json!({"scope_files": [], "scope": ["a.rs"]}),
                &["scope_files", "scope"],
                "scope_files",
                true
            ),
            Ok(Some(vec!["a.rs".to_string()]))
        );
        assert!(
            arg_string_list(
                &json!({"roles": [], "stances": "pro"}),
                &["roles", "stances"],
                "roles",
                false
            )
            .is_err()
        );
    }

    #[test]
    fn arg_bool_rejects_non_boolean_values() {
        assert_eq!(arg_bool(&json!({}), "wake", true), Ok(true));
        assert_eq!(arg_bool(&json!({"wake": false}), "wake", true), Ok(false));
        let err = arg_bool(&json!({"wake": "false"}), "wake", true).unwrap_err();
        assert!(err.contains("wake must be a boolean"), "{err}");
    }
}
