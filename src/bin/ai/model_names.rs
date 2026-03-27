use std::path::PathBuf;
use std::sync::LazyLock;

use crate::common::utils::expanduser;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelDef {
    pub key: String,
    pub name: String,
    pub is_vl: bool,
    pub search_enabled: bool,
    pub tools_default_enabled: bool,
    #[serde(default)]
    pub enable_thinking: bool,
}

static MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_models);
static BY_NAME: LazyLock<std::collections::HashMap<String, usize>> = LazyLock::new(build_name_index);

fn config_path() -> PathBuf {
    let home = expanduser("~/.config/rust_tools/models.json");
    let home_path: PathBuf = match home {
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
    };
    if home_path.exists() {
        return home_path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models.json")
}

fn load_models() -> Vec<ModelDef> {
    let path = config_path();
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
        std::process::exit(1);
    })
}

fn build_name_index() -> std::collections::HashMap<String, usize> {
    let mut index = std::collections::HashMap::new();
    for (i, m) in MODELS.iter().enumerate() {
        index.insert(m.name.to_lowercase(), i);
    }
    index
}

pub fn all() -> &'static [ModelDef] {
    &MODELS
}

pub fn find_by_name(name: &str) -> Option<&'static ModelDef> {
    BY_NAME.get(&name.trim().to_lowercase()).map(|&i| &MODELS[i])
}

pub fn find_by_key(key: &str) -> Option<&'static ModelDef> {
    MODELS.iter().find(|m| m.key == key)
}
