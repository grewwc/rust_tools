use serde_json::Value;

use crate::jsonw::sort;
use crate::jsonw::types::DiffEntry;

pub fn diff_json(old: &Value, new: &Value, sort_arrays: bool) -> Vec<DiffEntry> {
    let mut out = Vec::new();
    collect_diff_entries("", old, new, sort_arrays, &mut out);
    out
}

fn collect_diff_entries(
    path: &str,
    old: &Value,
    new: &Value,
    sort_arrays: bool,
    out: &mut Vec<DiffEntry>,
) {
    if old.is_null() && new.is_null() {
        return;
    }

    match (old, new) {
        (Value::Object(o1), Value::Object(o2)) => {
            let mut keys: Vec<&str> = o1.keys().chain(o2.keys()).map(|k| k.as_str()).collect();
            keys.sort_unstable();
            keys.dedup();

            for k in keys {
                let p = if path.is_empty() {
                    k.to_string()
                } else {
                    format!("{path}.{k}")
                };
                let v1 = o1.get(k).unwrap_or(&Value::Null);
                let v2 = o2.get(k).unwrap_or(&Value::Null);
                if v1.is_null() ^ v2.is_null() {
                    out.push(DiffEntry {
                        key: p,
                        old: v1.clone(),
                        new: v2.clone(),
                    });
                    continue;
                }
                collect_diff_entries(&p, v1, v2, sort_arrays, out);
            }
        }
        (Value::Array(a1), Value::Array(a2)) => {
            let mut left = a1.clone();
            let mut right = a2.clone();
            if sort_arrays {
                left.sort_by_key(sort::json_scalar_string_key);
                right.sort_by_key(sort::json_scalar_string_key);
            }

            let max_len = left.len().max(right.len());
            for i in 0..max_len {
                let p = if path.is_empty() {
                    format!("{i}")
                } else {
                    format!("{path}.{i}")
                };
                let v1 = left.get(i).unwrap_or(&Value::Null);
                let v2 = right.get(i).unwrap_or(&Value::Null);
                if v1.is_null() ^ v2.is_null() {
                    out.push(DiffEntry {
                        key: p,
                        old: v1.clone(),
                        new: v2.clone(),
                    });
                    continue;
                }
                collect_diff_entries(&p, v1, v2, sort_arrays, out);
            }
        }
        _ => {
            if old != new {
                out.push(DiffEntry {
                    key: path.to_string(),
                    old: old.clone(),
                    new: new.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn test_diff_basic() {
        let old: Value = serde_json::json!({"a": 1, "b": [2,3]});
        let new: Value = serde_json::json!({"a": 2, "b": [2,4]});
        let diff = diff_json(&old, &new, false);
        assert!(diff.iter().any(|d| d.key == "a"));
        assert!(diff.iter().any(|d| d.key == "b.1"));
    }
}
