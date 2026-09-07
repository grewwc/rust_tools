use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::jsonw::types::DiffEntry;

/// Compares two JSON values and returns a flat list of differences.
///
/// Objects are compared key by key. Array elements are aligned by content,
/// not by index: items that appear in both arrays (canonically equal) cancel
/// out regardless of their position, so pure reordering produces no diff
/// entries. Leftover (unmatched) items are paired positionally and compared
/// recursively, so a modified item still yields field-level entries; the
/// unpaired tail is reported as removed / added.
///
/// Entries whose key carries the `  [old]` suffix exist only on the old side
/// (`new` is `Null`); `  [new]` marks items that exist only on the new side
/// (`old` is `Null`). This matches the go_tools jsondiff output format.
pub fn diff_json(old: &Value, new: &Value) -> Vec<DiffEntry> {
    let mut out = Vec::new();
    collect_diff_entries("", old, new, &mut out);
    out
}

fn collect_diff_entries(path: &str, old: &Value, new: &Value, out: &mut Vec<DiffEntry>) {
    if old.is_null() && new.is_null() {
        return;
    }

    match (old, new) {
        (Value::Object(o1), Value::Object(o2)) => {
            let mut keys: Vec<&str> = o1.keys().chain(o2.keys()).map(String::as_str).collect();
            keys.sort_unstable();
            keys.dedup();

            for k in keys {
                let p = join_path(path, k);
                let v1 = o1.get(k).unwrap_or(&Value::Null);
                let v2 = o2.get(k).unwrap_or(&Value::Null);
                collect_diff_entries(&p, v1, v2, out);
            }
        }
        (Value::Array(a1), Value::Array(a2)) => {
            diff_arrays(path, a1, a2, out);
        }
        _ => {
            if old == new {
                return;
            }
            // Present on one side only (missing key / array tail): mark the
            // entry so removed vs added is distinguishable from a replacement.
            if old.is_null() {
                out.push(DiffEntry {
                    key: format!("{path}  [new]"),
                    old: Value::Null,
                    new: new.clone(),
                });
            } else if new.is_null() {
                out.push(DiffEntry {
                    key: format!("{path}  [old]"),
                    old: old.clone(),
                    new: Value::Null,
                });
            } else {
                out.push(DiffEntry {
                    key: path.to_string(),
                    old: old.clone(),
                    new: new.clone(),
                });
            }
        }
    }
}

/// Diffs two arrays position-independently.
///
/// Matching is a multiset comparison on each item's canonical form, so equal
/// items pair up even when moved. Unmatched leftovers are paired by their
/// relative order and recursed into (paths anchor on the old array's index);
/// whatever remains unpaired is reported with `  [old]` / `  [new]` markers.
fn diff_arrays(path: &str, a1: &[Value], a2: &[Value], out: &mut Vec<DiffEntry>) {
    // Queue of positions on the right side for each canonical form.
    let mut right_by_key: FxHashMap<String, VecDeque<usize>> = FxHashMap::default();
    for (j, item) in a2.iter().enumerate() {
        right_by_key
            .entry(canonical_form(item))
            .or_default()
            .push_back(j);
    }

    // Consume one right-side position for every left item that has an exactly
    // equal partner, in any position.
    let mut consumed = vec![false; a2.len()];
    let mut left_unmatched: Vec<usize> = Vec::new();
    for (i, item) in a1.iter().enumerate() {
        let matched = match right_by_key.get_mut(&canonical_form(item)) {
            Some(queue) => match queue.pop_front() {
                Some(j) => {
                    consumed[j] = true;
                    true
                }
                None => false,
            },
            None => false,
        };
        if !matched {
            left_unmatched.push(i);
        }
    }
    let right_unmatched: Vec<usize> = (0..a2.len()).filter(|&j| !consumed[j]).collect();

    let pair_count = left_unmatched.len().min(right_unmatched.len());
    for t in 0..pair_count {
        let i = left_unmatched[t];
        let j = right_unmatched[t];
        collect_diff_entries(&join_index(path, i), &a1[i], &a2[j], out);
    }
    for &i in &left_unmatched[pair_count..] {
        out.push(DiffEntry {
            key: format!("{}  [old]", join_index(path, i)),
            old: a1[i].clone(),
            new: Value::Null,
        });
    }
    for &j in &right_unmatched[pair_count..] {
        out.push(DiffEntry {
            key: format!("{}  [new]", join_index(path, j)),
            old: Value::Null,
            new: a2[j].clone(),
        });
    }
}

fn join_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn join_index(path: &str, idx: usize) -> String {
    if path.is_empty() {
        idx.to_string()
    } else {
        format!("{path}.{idx}")
    }
}

/// Deterministic serialization used for content matching.
///
/// Object keys are sorted because key order is never significant in JSON
/// (serde_json maps preserve insertion order when `preserve_order` is on).
/// Array element order is kept: reordering a nested array is a real content
/// change, while top-level array elements are realigned by `diff_arrays`.
/// Strings stay JSON-quoted, which keeps the encoding injective (e.g. the
/// string `"[1]"` and the array `[1]` produce different forms).
fn canonical_form(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(&str, &Value)> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            let mut s = String::from("{");
            for (idx, (k, v)) in entries.iter().enumerate() {
                if idx > 0 {
                    s.push(',');
                }
                s.push_str(&serde_json::to_string(k).unwrap_or_default());
                s.push(':');
                s.push_str(&canonical_form(v));
            }
            s.push('}');
            s
        }
        Value::Array(items) => {
            let mut s = String::from("[");
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    s.push(',');
                }
                s.push_str(&canonical_form(item));
            }
            s.push(']');
            s
        }
        other => serde_json::to_string(other).unwrap_or_default(),
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
        let diff = diff_json(&old, &new);
        assert!(diff.iter().any(|d| d.key == "a"));
        assert!(diff.iter().any(|d| d.key == "b.1"));
    }

    #[test]
    fn reordered_arrays_produce_no_entries() {
        let old = serde_json::json!({"a": [1, 2, 3]});
        let new = serde_json::json!({"a": [3, 1, 2]});
        assert!(diff_json(&old, &new).is_empty());
    }

    #[test]
    fn inserted_item_is_reported_once_with_new_marker() {
        let old = serde_json::json!({"rules": [{"id": 1}, {"id": 2}]});
        let new = serde_json::json!({"rules": [{"id": 9}, {"id": 1}, {"id": 2}]});
        let diff = diff_json(&old, &new);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].key, "rules.0  [new]");
        assert_eq!(diff[0].new, serde_json::json!({"id": 9}));
        assert!(diff[0].old.is_null());
    }

    #[test]
    fn removed_item_is_reported_once_with_old_marker() {
        let old = serde_json::json!([1, 2, 3]);
        let new = serde_json::json!([1, 3]);
        let diff = diff_json(&old, &new);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].key, "1  [old]");
        assert_eq!(diff[0].old, 2);
        assert!(diff[0].new.is_null());
    }

    #[test]
    fn modified_item_keeps_field_level_diff_despite_reorder() {
        let old = serde_json::json!([{"id": 1, "v": 1}, {"id": 2, "v": 2}]);
        let new = serde_json::json!([{"id": 2, "v": 2}, {"id": 1, "v": 9}]);
        let diff = diff_json(&old, &new);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].key, "0.v");
        assert_eq!(diff[0].old, 1);
        assert_eq!(diff[0].new, 9);
    }

    #[test]
    fn object_key_order_does_not_affect_matching() {
        let old = serde_json::json!({"a": {"x": 1, "y": 2}});
        let new = serde_json::json!({"a": {"y": 2, "x": 1}});
        assert!(diff_json(&old, &new).is_empty());
    }

    #[test]
    fn duplicate_items_are_matched_as_multiset() {
        let old = serde_json::json!([1, 1, 2]);
        let new = serde_json::json!([1, 2, 2]);
        let diff = diff_json(&old, &new);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].key, "1");
        assert_eq!(diff[0].old, 1);
        assert_eq!(diff[0].new, 2);
    }

    #[test]
    fn missing_key_entries_carry_old_new_markers() {
        let old = serde_json::json!({"a": 1});
        let new = serde_json::json!({"b": 2});
        let diff = diff_json(&old, &new);
        assert_eq!(diff.len(), 2);
        assert_eq!(diff[0].key, "a  [old]");
        assert_eq!(diff[1].key, "b  [new]");
    }

    #[test]
    fn canonical_form_is_injective_across_types() {
        let s = serde_json::json!({"a": "[1]"});
        let a = serde_json::json!({"a": [1]});
        assert_ne!(canonical_form(&s), canonical_form(&a));
    }
}
