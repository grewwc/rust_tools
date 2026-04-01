use std::path::PathBuf;
use std::sync::LazyLock;

use rust_tools::commonw::FastMap;
use rust_tools::cw::SkipSet;
use crate::commonw::utils::expanduser;

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

static USER_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_user_models);
static BUILTIN_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_builtin_models);
static USER_BY_NAME: LazyLock<FastMap<String, usize>> =
    LazyLock::new(build_user_name_index);
static BUILTIN_BY_NAME: LazyLock<FastMap<String, usize>> =
    LazyLock::new(build_builtin_name_index);

fn user_config_path() -> PathBuf {
    let home = expanduser("~/.config/rust_tools/models.json");
    match home {
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
    }
}

fn builtin_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models.json")
}

fn load_models_from_path(path: &PathBuf) -> Vec<ModelDef> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
        Vec::new()
    })
}

fn load_user_models() -> Vec<ModelDef> {
    let path = user_config_path();
    if path.exists() {
        load_models_from_path(&path)
    } else {
        Vec::new()
    }
}

fn load_builtin_models() -> Vec<ModelDef> {
    let path = builtin_config_path();
    if !path.exists() {
        eprintln!(
            "[model_names] builtin models.json not found at {}",
            path.display()
        );
        std::process::exit(1);
    }
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
        std::process::exit(1);
    })
}

fn build_user_name_index() -> FastMap<String, usize> {
    let mut index = FastMap::default();
    for (i, m) in USER_MODELS.iter().enumerate() {
        index.insert(m.name.clone().to_lowercase(), i);
    }
    index
}

fn build_builtin_name_index() -> FastMap<String, usize> {
    let mut index = FastMap::default();
    for (i, m) in BUILTIN_MODELS.iter().enumerate() {
        index.insert(m.name.clone().to_lowercase(), i);
    }
    index
}

pub fn all() -> Vec<&'static ModelDef> {
    let mut seen = SkipSet::new(16);
    let mut result = Vec::new();

    for m in USER_MODELS.iter() {
        let key = m.name.to_lowercase();
        if seen.insert(key) {
            result.push(m);
        }
    }

    for m in BUILTIN_MODELS.iter() {
        let key = m.name.to_lowercase();
        if seen.insert(key) {
            result.push(m);
        }
    }

    result
}

pub fn find_by_name(name: &str) -> Option<&'static ModelDef> {
    let name_lower = name.trim().to_lowercase();

    if let Some(&i) = USER_BY_NAME.get(&name_lower) {
        return Some(&USER_MODELS[i]);
    }

    BUILTIN_BY_NAME
        .get(&name_lower)
        .map(|&i| &BUILTIN_MODELS[i])
}

pub fn find_by_key(key: &str) -> Option<&'static ModelDef> {
    USER_MODELS
        .iter()
        .find(|m| m.key == key)
        .or_else(|| BUILTIN_MODELS.iter().find(|m| m.key == key))
}
