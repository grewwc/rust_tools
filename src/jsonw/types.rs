use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Json {
    pub(crate) value: Value,
}

impl Default for Json {
    fn default() -> Self {
        Self {
            value: Value::Object(serde_json::Map::new()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParseOptions {
    pub allow_comment: bool,
    pub remove_special_chars: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            allow_comment: true,
            remove_special_chars: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffEntry {
    pub key: String,
    pub old: Value,
    pub new: Value,
}
