use std::cmp::Ordering;

use serde_json::Value;

pub fn compare_json_values_by_scalar_string(a: &Value, b: &Value) -> Ordering {
    json_scalar_string_key(a).cmp(&json_scalar_string_key(b))
}

pub fn json_scalar_string_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        _ => String::new(),
    }
}
