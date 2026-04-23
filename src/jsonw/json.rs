use std::{cmp::Ordering, fs, io::Read, path::Path};

use serde::Serialize;
use serde_json::Value;

use crate::{
    clipboardw::string_content, jsonw::sanitize::sanitize_json_input, jsonw::sort,
    jsonw::types::ParseOptions,
};

use super::types::Json;

impl Json {
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    pub fn from_str(s: &str, options: ParseOptions) -> Result<Self, serde_json::Error> {
        let sanitized = sanitize_json_input(s, options);
        let value: Value = serde_json::from_str(&sanitized)?;
        Ok(Self { value })
    }

    pub fn from_bytes(
        data: &[u8],
        options: ParseOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s = String::from_utf8_lossy(data);
        Ok(Self::from_str(&s, options)?)
    }

    pub fn from_reader<R: Read>(
        mut reader: R,
        options: ParseOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut s = String::new();
        reader.read_to_string(&mut s)?;
        Ok(Self::from_str(&s, options)?)
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        options: ParseOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s = fs::read_to_string(path)?;
        Ok(Self::from_str(&s, options)?)
    }

    pub fn from_clipboard(options: ParseOptions) -> Result<Self, serde_json::Error> {
        let s = string_content::get_clipboard_content();
        Self::from_str(&s, options)
    }

    pub fn is_array(&self) -> bool {
        self.value.is_array()
    }

    pub fn is_object(&self) -> bool {
        self.value.is_object()
    }

    pub fn len(&self) -> usize {
        match &self.value {
            Value::Array(a) => a.len(),
            Value::Object(o) => o.len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scalar(&self) -> Option<&Value> {
        match &self.value {
            Value::Array(_) | Value::Object(_) => None,
            _ => Some(&self.value),
        }
    }

    pub fn raw_value(&self) -> &Value {
        &self.value
    }

    pub fn contains_key(&self, key: &str) -> bool {
        match &self.value {
            Value::Object(o) => o.contains_key(key),
            Value::Array(a) => key.parse::<usize>().is_ok_and(|i| i < a.len()),
            _ => false,
        }
    }

    pub fn keys(&self) -> Vec<String> {
        match &self.value {
            Value::Object(o) => o.keys().cloned().collect(),
            Value::Array(a) => (0..a.len()).map(|i| i.to_string()).collect(),
            _ => Vec::new(),
        }
    }

    pub fn for_each_key<F: FnMut(&str)>(&self, mut f: F) {
        match &self.value {
            Value::Object(o) => {
                for k in o.keys() {
                    f(k);
                }
            }
            Value::Array(a) => {
                for i in 0..a.len() {
                    let k = i.to_string();
                    f(&k);
                }
            }
            _ => {}
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match &self.value {
            Value::Object(o) => o.get(key),
            Value::Array(a) => {
                let idx = key.parse::<usize>().ok()?;
                a.get(idx)
            }
            _ => None,
        }
    }

    pub fn get_or_default<'a>(&'a self, key: &str, default: &'a Value) -> &'a Value {
        if let Some(v) = self.get(key) {
            return v;
        }
        default
    }

    pub fn get_json(&self, key: &str) -> Option<Json> {
        self.get(key).map(|v| Json::new(v.clone()))
    }

    pub fn get_index(&self, idx: usize) -> Option<Json> {
        match &self.value {
            Value::Array(a) => a.get(idx).map(|v| Json::new(v.clone())),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    pub fn get_i64(&self, key: &str) -> i64 {
        self.get(key)
            .and_then(|v| v.as_i64())
            .or_else(|| self.get(key).and_then(|v| v.as_u64().map(|x| x as i64)))
            .unwrap_or_default()
    }

    pub fn get_f64(&self, key: &str) -> f64 {
        self.get(key).and_then(|v| v.as_f64()).unwrap_or_default()
    }

    pub fn get_bool(&self, key: &str) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    pub fn set_value(&mut self, key: &str, value: Value) -> bool {
        match &mut self.value {
            Value::Object(o) => {
                let existed = o.contains_key(key);
                o.insert(key.to_string(), value);
                existed
            }
            Value::Null => {
                let mut o = serde_json::Map::new();
                o.insert(key.to_string(), value);
                self.value = Value::Object(o);
                false
            }
            _ => false,
        }
    }

    pub fn set<T: Serialize>(&mut self, key: &str, value: T) -> bool {
        let v = serde_json::to_value(value).unwrap_or(Value::Null);
        self.set_value(key, v)
    }

    pub fn add_value(&mut self, value: Value) -> bool {
        match &mut self.value {
            Value::Array(a) => {
                a.push(value);
                true
            }
            Value::Null => {
                self.value = Value::Array(vec![value]);
                true
            }
            Value::Object(o) if o.is_empty() => {
                self.value = Value::Array(vec![value]);
                true
            }
            _ => false,
        }
    }

    pub fn add<T: Serialize>(&mut self, value: T) -> bool {
        let v = serde_json::to_value(value).unwrap_or(Value::Null);
        self.add_value(v)
    }

    pub fn string(&self) -> String {
        self.string_with_indent("", "")
    }

    pub fn string_with_indent(&self, prefix: &str, indent: &str) -> String {
        let s = if indent.is_empty() {
            serde_json::to_string(&self.value).unwrap_or_default()
        } else {
            let mut out = Vec::new();
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indent.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
            self.value.serialize(&mut ser).ok();
            String::from_utf8_lossy(&out).into_owned()
        };
        let s = s
            .replace(r"\u0026", "&")
            .replace(r"\u003c", "<")
            .replace(r"\u003e", ">");

        if prefix.is_empty() {
            return format!("{s}\n");
        }

        let mut res = String::new();
        for line in s.lines() {
            res.push_str(prefix);
            res.push_str(line);
            res.push('\n');
        }
        res
    }

    pub fn to_pretty_string(&self) -> String {
        self.string_with_indent("", "  ")
    }

    pub fn to_compact_string(&self) -> String {
        serde_json::to_string(&self.value).unwrap_or_default()
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P, pretty: bool) -> std::io::Result<()> {
        let s = if pretty {
            self.to_pretty_string()
        } else {
            self.to_compact_string()
        };
        fs::write(path, s)
    }

    pub fn abs_key(&self, key: &str) -> Vec<String> {
        let mut out = Vec::new();
        collect_abs_key_paths(&self.value, key, "root", &mut out);
        out.sort();
        out.dedup();
        out
    }

    pub fn extract(&self, key: &str) -> Json {
        if let Some((root_path, selected_fields)) = parse_bracket_field_list_selector(key) {
            return extract_multi_select_fields(self, &root_path, &selected_fields);
        }
        extract_by_dot_path(self, key)
    }

    pub fn raw_data(&self) -> Value {
        self.value.clone()
    }

    pub fn get_int(&self, key: &str) -> i64 {
        self.get_i64(key)
    }

    pub fn get_float(&self, key: &str) -> f64 {
        self.get_f64(key)
    }

    pub fn get_value(&self, key: &str) -> Value {
        self.get(key).cloned().unwrap_or(Value::Null)
    }

    pub fn get_or_default_value(&self, key: &str, default: Value) -> Value {
        self.get(key).cloned().unwrap_or(default)
    }

    pub fn sort_array_by<F>(&mut self, mut cmp: F) -> Result<(), &'static str>
    where
        F: FnMut(&Value, &Value) -> Ordering,
    {
        match &mut self.value {
            Value::Array(a) => {
                a.sort_by(|x, y| cmp(x, y));
                Ok(())
            }
            _ => Err("json is not array"),
        }
    }

    pub fn sort_array_default(&mut self) -> Result<(), &'static str> {
        self.sort_array_by(sort::compare_json_values_by_scalar_string)
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

fn collect_abs_key_paths(v: &Value, key: &str, curr_path: &str, out: &mut Vec<String>) {
    let sep = "->";
    match v {
        Value::Array(a) => {
            for (i, sub) in a.iter().enumerate() {
                collect_abs_key_paths(sub, key, &format!("{curr_path}{sep}{i}"), out);
            }
        }
        Value::Object(o) => {
            if o.contains_key(key) {
                out.push(format!("{curr_path}{sep}{key}"));
            }
            for (k, sub) in o.iter() {
                collect_abs_key_paths(sub, key, &format!("{curr_path}{sep}{k}"), out);
            }
        }
        _ => {}
    }
}

fn parse_bracket_field_list_selector(selector: &str) -> Option<(String, Vec<String>)> {
    let idx = selector.rfind('.')?;
    let (root, last) = selector.split_at(idx);
    let last = last[1..].trim();
    if !(last.starts_with('[') && last.ends_with(']')) {
        return None;
    }
    if !last.contains(',') {
        return None;
    }
    let inside = &last[1..last.len() - 1];
    let fields = inside
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return None;
    }
    Some((root.to_string(), fields))
}

fn extract_multi_select_fields(root: &Json, root_path: &str, fields: &[String]) -> Json {
    let mut rows: Vec<serde_json::Map<String, Value>> = Vec::new();
    for field in fields {
        let abs_path = if root_path.is_empty() {
            field.to_string()
        } else {
            format!("{root_path}.{field}")
        };
        let extracted = root.extract(&abs_path);
        let items = collect_values_as_items(&extracted.value);
        for (i, item) in items.into_iter().enumerate() {
            if rows.len() <= i {
                rows.push(serde_json::Map::new());
            }
            rows[i].insert(abs_path.clone(), item);
        }
    }
    Json::new(Value::Array(rows.into_iter().map(Value::Object).collect()))
}

fn extract_by_dot_path(root: &Json, path: &str) -> Json {
    let mut curr = root.clone();
    if path.is_empty() {
        return curr;
    }
    for token in path.split('.').filter(|p| !p.trim().is_empty()) {
        curr = extract_one_level(&curr, token.trim());
    }
    curr
}

fn extract_one_level(curr: &Json, key: &str) -> Json {
    match &curr.value {
        Value::Array(a) => {
            if a.len() == 1 {
                return extract_one_level(&Json::new(a[0].clone()), key);
            }
            let mut out: Vec<Value> = Vec::new();
            for sub in a {
                let j = extract_one_level(&Json::new(sub.clone()), key);
                match &j.value {
                    Value::Array(arr) => {
                        out.extend(flatten_nested_arrays(arr));
                    }
                    Value::Object(_) => out.push(j.value),
                    Value::Null => {}
                    _ => out.push(j.value),
                }
            }
            if out.is_empty() {
                Json::default()
            } else {
                Json::new(Value::Array(out))
            }
        }
        Value::Object(o) => match o.get(key) {
            Some(v) => Json::new(v.clone()),
            None => Json::default(),
        },
        _ => Json::default(),
    }
}

fn flatten_nested_arrays(arr: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for v in arr {
        match v {
            Value::Array(inner) => out.extend(flatten_nested_arrays(inner)),
            _ => out.push(v.clone()),
        }
    }
    out
}

fn collect_values_as_items(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.clone(),
        Value::Object(o) => o.values().cloned().collect(),
        Value::Null => Vec::new(),
        _ => vec![v.clone()],
    }
}
