use std::fs;
use std::path::{Path, PathBuf};

pub(crate) struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn validate_access(&self) -> Result<(), String> {
        if !self.path.is_absolute() {
            return Err("file_path must be absolute".to_string());
        }
        if is_sensitive_fs_path(&self.path) {
            return Err("Access blocked: sensitive path".to_string());
        }
        Ok(())
    }

    pub(crate) fn ensure_exists(&self) -> Result<(), String> {
        if !self.path.exists() {
            return Err(format!("File not found: {}", self.path.display()));
        }
        Ok(())
    }

    pub(crate) fn read_to_string(&self) -> Result<String, String> {
        fs::read_to_string(&self.path).map_err(|e| format!("Failed to read file: {}", e))
    }

    pub(crate) fn write_all(&self, content: &str) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        fs::write(&self.path, content).map_err(|e| format!("Failed to write file: {}", e))
    }
}

fn is_sensitive_fs_path(path: &Path) -> bool {
    let rendered = path.to_string_lossy();
    let rendered = rendered.as_ref();
    if rendered.contains("/.ssh/")
        || rendered.ends_with("/.ssh")
        || rendered.contains("/.gnupg/")
        || rendered.ends_with("/.gnupg")
        || rendered.contains("/.aws/")
        || rendered.ends_with("/.aws")
        || rendered.contains("/.kube/")
        || rendered.ends_with("/.kube")
        || rendered.contains("/.configW")
        || rendered.ends_with("/.configW")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        name,
        "id_rsa"
            | "id_rsa.pub"
            | "id_ed25519"
            | "id_ed25519.pub"
            | "authorized_keys"
            | "known_hosts"
            | ".netrc"
            | ".npmrc"
            | ".pypirc"
            | ".git-credentials"
            | "credentials"
            | "config.json"
    )
}
