use crate::common::types::{FastMap, FastSet};
use crate::strw::split;

pub fn add_quote(slice: &[String]) -> Vec<String> {
    slice.iter().map(|s| format!("{s:?}")).collect()
}

pub fn map_to_string(m: &FastMap<String, String>) -> String {
    let mut out = String::new();
    for (k, v) in m {
        if v.trim().contains(' ') {
            out.push_str(&format!(" {k} \"{v}\" "));
        } else {
            out.push_str(&format!(" {k} {v} "));
        }
    }
    out
}

pub fn format_file_extensions(extensions: &str) -> FastSet<String> {
    let mut out = FastSet::default();
    let normalized = extensions.replace(',', " ");
    for ext in split::split_no_empty(&normalized, " ") {
        let e = ext.trim();
        if e.is_empty() {
            continue;
        }
        if e.starts_with('.') {
            out.insert(e.to_string());
        } else {
            out.insert(format!(".{e}"));
        }
    }
    out
}
