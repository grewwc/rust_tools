use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use chrono::Local;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::commonw::{
    configw,
    utils::{expanduser, get_config_dir, open_file_for_write_truncate},
};

pub(in crate::ai) const DEFAULT_PERSONA_ID: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(in crate::ai) struct PersonaProfile {
    pub(in crate::ai) id: String,
    pub(in crate::ai) name: String,
    #[serde(default)]
    pub(in crate::ai) avatar: String,
    #[serde(default)]
    pub(in crate::ai) prompt: String,
    #[serde(default)]
    pub(in crate::ai) created_at: String,
    #[serde(default)]
    pub(in crate::ai) updated_at: String,
    #[serde(default)]
    pub(in crate::ai) last_session_id: Option<String>,
}

impl PersonaProfile {
    pub(in crate::ai) fn is_default(&self) -> bool {
        self.id == DEFAULT_PERSONA_ID
    }

    pub(in crate::ai) fn prompt_summary(&self) -> String {
        let first = self
            .prompt
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("-");
        truncate_chars(first, 48)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersonaRegistry {
    #[serde(default)]
    active_persona_id: Option<String>,
    #[serde(default)]
    personas: Vec<PersonaProfile>,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct PersonaDeleteResult {
    pub(in crate::ai) removed: PersonaProfile,
    pub(in crate::ai) active_after: PersonaProfile,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct PersonaStore {
    path: PathBuf,
}

pub(in crate::ai) fn default_persona() -> PersonaProfile {
    PersonaProfile {
        id: DEFAULT_PERSONA_ID.to_string(),
        name: DEFAULT_PERSONA_ID.to_string(),
        avatar: String::new(),
        prompt: String::new(),
        created_at: String::new(),
        updated_at: String::new(),
        last_session_id: None,
    }
}

pub(in crate::ai) fn history_file_for_persona(base_history_file: &Path, persona_id: &str) -> PathBuf {
    if persona_id == DEFAULT_PERSONA_ID {
        return base_history_file.to_path_buf();
    }
    insert_suffix_before_extension(base_history_file, &format!(".persona-{}", sanitize_path_fragment(persona_id)))
}

pub(in crate::ai) fn memory_file_for_persona(persona_id: &str) -> PathBuf {
    let base = resolve_default_memory_file();
    if persona_id == DEFAULT_PERSONA_ID {
        return base;
    }
    insert_suffix_before_extension(&base, &format!(".persona-{}", sanitize_path_fragment(persona_id)))
}

pub(in crate::ai) fn cleanup_persona_storage(
    base_history_file: &Path,
    persona_id: &str,
) -> io::Result<()> {
    if persona_id == DEFAULT_PERSONA_ID {
        return Ok(());
    }

    let history_file = history_file_for_persona(base_history_file, persona_id);
    remove_path_if_exists(&history_file)?;
    remove_path_if_exists(&sessions_root_from_history_file(&history_file))?;

    let memory_file = memory_file_for_persona(persona_id);
    remove_path_if_exists(&memory_file)?;
    remove_suffixed_files(&memory_file)?;
    Ok(())
}

impl PersonaStore {
    pub(in crate::ai) fn new() -> Self {
        let path = get_config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("rust_tools")
            .join("personas.json");
        Self { path }
    }

    #[cfg(test)]
    pub(in crate::ai) fn for_tests_with_path(path: PathBuf) -> Self {
        Self { path }
    }

    pub(in crate::ai) fn list_personas(&self) -> io::Result<Vec<PersonaProfile>> {
        let registry = self.load_registry()?;
        let mut personas = Vec::with_capacity(registry.personas.len() + 1);
        personas.push(default_persona());
        personas.extend(
            registry
                .personas
                .into_iter()
                .filter(|persona| !persona.is_default()),
        );
        Ok(personas)
    }

    pub(in crate::ai) fn active_persona(&self) -> io::Result<PersonaProfile> {
        let registry = self.load_registry()?;
        Ok(resolve_active_persona(&registry))
    }

    pub(in crate::ai) fn create_persona(
        &self,
        name: &str,
        avatar: Option<&str>,
        prompt: &str,
    ) -> io::Result<PersonaProfile> {
        let name = name.trim();
        if name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "persona name cannot be empty",
            ));
        }
        if name.eq_ignore_ascii_case(DEFAULT_PERSONA_ID) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "persona name 'default' is reserved",
            ));
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "persona prompt cannot be empty",
            ));
        }

        let avatar = normalize_avatar(avatar)?;
        let mut registry = self.load_registry()?;
        if registry
            .personas
            .iter()
            .any(|persona| eq_selector(&persona.name, name))
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("persona '{name}' already exists"),
            ));
        }
        if let Some(avatar_value) = avatar.as_deref()
            && registry
                .personas
                .iter()
                .any(|persona| eq_selector(&persona.avatar, avatar_value))
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("avatar '{avatar_value}' is already in use"),
            ));
        }

        let now = Local::now().to_rfc3339();
        let persona = PersonaProfile {
            id: format!("persona-{}", Uuid::new_v4().simple()),
            name: name.to_string(),
            avatar: avatar.unwrap_or_default(),
            prompt: prompt.to_string(),
            created_at: now.clone(),
            updated_at: now,
            last_session_id: None,
        };
        registry.personas.push(persona.clone());
        self.save_registry(&registry)?;
        Ok(persona)
    }

    pub(in crate::ai) fn set_active_persona(&self, selector: &str) -> io::Result<PersonaProfile> {
        let mut registry = self.load_registry()?;
        let persona = find_persona_in_registry(&registry, selector)?
            .unwrap_or_else(default_persona);
        registry.active_persona_id = if persona.is_default() {
            None
        } else {
            Some(persona.id.clone())
        };
        self.save_registry(&registry)?;
        Ok(persona)
    }

    pub(in crate::ai) fn delete_persona(&self, selector: &str) -> io::Result<PersonaDeleteResult> {
        let trimmed = selector.trim();
        if trimmed.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "missing persona selector",
            ));
        }
        if trimmed.eq_ignore_ascii_case(DEFAULT_PERSONA_ID) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "default persona cannot be deleted",
            ));
        }

        let mut registry = self.load_registry()?;
        let Some(index) = registry
            .personas
            .iter()
            .position(|persona| matches_selector(persona, trimmed))
        else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("persona '{trimmed}' not found"),
            ));
        };

        let removed = registry.personas.remove(index);
        let active_after = if registry.active_persona_id.as_deref() == Some(removed.id.as_str()) {
            registry.active_persona_id = None;
            default_persona()
        } else {
            resolve_active_persona(&registry)
        };
        self.save_registry(&registry)?;
        Ok(PersonaDeleteResult {
            removed,
            active_after,
        })
    }

    pub(in crate::ai) fn remember_session(
        &self,
        persona_id: &str,
        session_id: &str,
    ) -> io::Result<()> {
        if persona_id == DEFAULT_PERSONA_ID || session_id.trim().is_empty() {
            return Ok(());
        }

        let mut registry = self.load_registry()?;
        let Some(persona) = registry.personas.iter_mut().find(|persona| persona.id == persona_id) else {
            return Ok(());
        };
        persona.last_session_id = Some(session_id.trim().to_string());
        persona.updated_at = Local::now().to_rfc3339();
        self.save_registry(&registry)
    }

    fn load_registry(&self) -> io::Result<PersonaRegistry> {
        if !self.path.exists() {
            return Ok(PersonaRegistry::default());
        }
        let content = fs::read_to_string(&self.path)?;
        serde_json::from_str(&content).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse personas: {err}"),
            )
        })
    }

    fn save_registry(&self, registry: &PersonaRegistry) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(registry).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to serialize personas: {err}"),
            )
        })?;
        let mut file = open_file_for_write_truncate(&self.path, 0o600)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
}

fn resolve_active_persona(registry: &PersonaRegistry) -> PersonaProfile {
    registry
        .active_persona_id
        .as_deref()
        .and_then(|id| {
            registry
                .personas
                .iter()
                .find(|persona| persona.id == id)
                .cloned()
        })
        .unwrap_or_else(default_persona)
}

fn find_persona_in_registry(
    registry: &PersonaRegistry,
    selector: &str,
) -> io::Result<Option<PersonaProfile>> {
    let trimmed = selector.trim();
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing persona selector",
        ));
    }
    if trimmed.eq_ignore_ascii_case(DEFAULT_PERSONA_ID) {
        return Ok(Some(default_persona()));
    }
    Ok(registry
        .personas
        .iter()
        .find(|persona| matches_selector(persona, trimmed))
        .cloned())
}

fn matches_selector(persona: &PersonaProfile, selector: &str) -> bool {
    eq_selector(&persona.id, selector) || eq_selector(&persona.name, selector)
}

fn eq_selector(left: &str, right: &str) -> bool {
    left.trim().to_lowercase() == right.trim().to_lowercase()
}

fn normalize_avatar(avatar: Option<&str>) -> io::Result<Option<String>> {
    let Some(avatar) = avatar.map(str::trim).filter(|avatar| !avatar.is_empty()) else {
        return Ok(None);
    };
    if avatar.chars().count() > 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "avatar is too long (max 16 chars)",
        ));
    }
    Ok(Some(avatar.to_string()))
}

fn resolve_default_memory_file() -> PathBuf {
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(expanduser(path).as_ref());
        }
    }
    let cfg = configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    PathBuf::from(expanduser(&raw).as_ref())
}

fn sessions_root_from_history_file(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let name = history_file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("history");
    parent.join(format!("{name}.sessions"))
}

fn insert_suffix_before_extension(path: &Path, suffix: &str) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("history");
    let ext = path.extension().and_then(|ext| ext.to_str());
    let file_name = match ext {
        Some(ext) if !ext.is_empty() => format!("{stem}{suffix}.{ext}"),
        _ => format!("{stem}{suffix}"),
    };
    parent.join(file_name)
}

fn sanitize_path_fragment(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "persona".to_string()
    } else {
        out
    }
}

fn remove_path_if_exists(path: &Path) -> io::Result<()> {
    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn remove_suffixed_files(base_file: &Path) -> io::Result<()> {
    let Some(parent) = base_file.parent() else {
        return Ok(());
    };
    let Some(base_name) = base_file.file_name().and_then(|name| name.to_str()) else {
        return Ok(());
    };
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name != base_name && name.starts_with(&format!("{base_name}.")) {
            remove_path_if_exists(&entry.path())?;
        }
    }
    Ok(())
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out = String::with_capacity(limit + 1);
    for ch in text.chars().take(limit) {
        out.push(ch);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> PersonaStore {
        let dir = std::env::temp_dir().join(format!("rust-tools-personas-{}", Uuid::new_v4()));
        PersonaStore::for_tests_with_path(dir.join("personas.json"))
    }

    #[test]
    fn history_path_for_default_persona_preserves_base() {
        let base = PathBuf::from("/tmp/history.sqlite");
        assert_eq!(history_file_for_persona(&base, DEFAULT_PERSONA_ID), base);
    }

    #[test]
    fn history_path_for_named_persona_gets_isolated_suffix() {
        let base = PathBuf::from("/tmp/history.sqlite");
        let path = history_file_for_persona(&base, "persona-123");
        assert_eq!(path, PathBuf::from("/tmp/history.persona-persona-123.sqlite"));
    }

    #[test]
    fn create_persona_rejects_duplicate_name() {
        let store = temp_store();
        store
            .create_persona("Alice", Some("A"), "You are Alice.")
            .unwrap();
        let err = store
            .create_persona("alice", Some("B"), "You are another Alice.")
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn delete_current_persona_falls_back_to_default() {
        let store = temp_store();
        let persona = store
            .create_persona("Reviewer", Some("R"), "You are a reviewer.")
            .unwrap();
        store.set_active_persona("Reviewer").unwrap();
        let result = store.delete_persona(&persona.id).unwrap();
        assert_eq!(result.removed.name, "Reviewer");
        assert!(result.active_after.is_default());
    }

    #[test]
    fn remember_session_updates_persona_binding() {
        let store = temp_store();
        let persona = store
            .create_persona("Planner", None, "You are a planner.")
            .unwrap();
        store.remember_session(&persona.id, "session-1").unwrap();
        let all = store.list_personas().unwrap();
        let planner = all
            .into_iter()
            .find(|item| item.name == "Planner")
            .expect("planner persona should exist");
        assert_eq!(planner.last_session_id.as_deref(), Some("session-1"));
    }
}
