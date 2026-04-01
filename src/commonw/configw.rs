use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

use crate::commonw::types::FastMap;
use crate::commonw::utils::expanduser;
use crate::strw::split::split_by_str_keep_quotes;

#[derive(Debug, Clone, Default)]
pub struct ConfigW {
    entries: Vec<(String, String)>,
    index: FastMap<String, usize>,
}

impl ConfigW {
    pub fn parse(content: &str) -> Self {
        let mut cfg = Self::default();
        for line in content.lines() {
            cfg.parse_line(line);
        }
        cfg
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut cfg = Self::default();
        for line in reader.lines() {
            cfg.parse_line(&line?);
        }
        Ok(cfg)
    }

    pub fn get(&self, key: &str, default: &str) -> String {
        self.get_opt(key).unwrap_or_else(|| default.to_string())
    }

    pub fn get_opt(&self, key: &str) -> Option<String> {
        let idx = *self.index.get(key)?;
        let (_, v) = self.entries.get(idx)?;
        Some(normalize_value(v))
    }

    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    fn parse_line(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        if trimmed.starts_with('#') || trimmed.starts_with("//") {
            return;
        }

        let parts = split_by_str_keep_quotes(line, "=", "\"", false);
        if parts.is_empty() {
            return;
        }
        let key = parts[0].trim().to_string();
        if key.is_empty() {
            return;
        }
        let val = parts
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        if let Some(&idx) = self.index.get(&key) {
            self.entries[idx] = (key, val);
            return;
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, val));
    }
}

fn normalize_value(v: &str) -> String {
    let mut out = v.trim().to_string();
    out = out
        .trim_start_matches('\'')
        .trim_end_matches('\'')
        .to_string();
    out = out
        .trim_start_matches('"')
        .trim_end_matches('"')
        .to_string();
    out.trim().to_string()
}

static CACHE: LazyLock<RwLock<Option<ConfigW>>> = LazyLock::new(|| RwLock::new(None));

pub fn config_path() -> PathBuf {
    if let Ok(path) = std::env::var("CONFIGW_PATH") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(expanduser(path).as_ref());
        }
    }
    PathBuf::from(expanduser("~/.configW").as_ref())
}

pub fn refresh() {
    if let Ok(mut lock) = CACHE.write() {
        *lock = None;
    }
}

pub fn get_all_config() -> ConfigW {
    if let Ok(lock) = CACHE.read()
        && let Some(cfg) = lock.clone()
    {
        return cfg;
    }

    let cfg = ConfigW::from_file(config_path()).unwrap_or_default();
    if let Ok(mut lock) = CACHE.write() {
        *lock = Some(cfg.clone());
    }
    cfg
}

pub fn get_config(key: &str, default: &str) -> String {
    let cfg = get_all_config();
    cfg.get(key, default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_line_quotes_and_comments() {
        let s = r#"
// comment
# comment2
re.remote.host=user@10.0.0.1:22
memo.sqlite="~/.go_tools_memo.sqlite3"
k1='v1'
k2=v2
k3="a=b=c"
"#;
        let cfg = ConfigW::parse(s);
        assert_eq!(cfg.get("re.remote.host", ""), "user@10.0.0.1:22");
        assert_eq!(cfg.get("memo.sqlite", ""), "~/.go_tools_memo.sqlite3");
        assert_eq!(cfg.get("k1", ""), "v1");
        assert_eq!(cfg.get("k2", ""), "v2");
        assert_eq!(cfg.get("k3", ""), "a=b=c");
    }
}
