use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TokenStore {
    pub(super) user_access_token: Option<String>,
    pub(super) user_access_token_expires_at_epoch_ms: Option<i64>,
    pub(super) refresh_token: Option<String>,
    pub(super) refresh_token_expires_in: Option<i64>,
    pub(super) updated_at_epoch_ms: Option<i64>,
}

pub(super) fn token_store_path() -> PathBuf {
    if let Ok(v) = std::env::var("FEISHU_TOKEN_STORE_PATH") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.token_store") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return PathBuf::from(rust_tools::commonw::utils::expanduser(&v).as_ref());
        }
    }
    PathBuf::from(
        rust_tools::commonw::utils::expanduser("~/.config/rust_tools/feishu_token.json").as_ref(),
    )
}

pub(super) fn load_token_store() -> Result<TokenStore, JsonRpcErr> {
    let path = token_store_path();
    let content = fs::read_to_string(&path).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })?;
    serde_json::from_str::<TokenStore>(&content).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })
}

pub(super) fn save_token_store(store: &TokenStore) -> Result<(), JsonRpcErr> {
    let path = token_store_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create token store directory",
                Some(json!({ "path": parent.display().to_string(), "error": e.to_string() })),
            )
        })?;
    }
    let s = serde_json::to_string_pretty(store).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to serialize token store",
            Some(json!({ "error": e.to_string() })),
        )
    })?;

    // 原子落盘：先写到同目录的临时文件（创建时即用 0o600 打开，杜绝"先默认权限
    // 落盘、再收权限"之间 token 短暂全局可读的窗口），再 rename 覆盖目标。
    // rename 同一文件系统内是原子的，写到一半崩溃也不会损坏既有 token store。
    let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));
    let write_tmp = || -> io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp_path)?;
        f.write_all(s.as_bytes())?;
        f.flush()?;
        Ok(())
    };
    write_tmp().map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        json_rpc_error(
            -32000,
            "Failed to write token store",
            Some(json!({ "path": tmp_path.display().to_string(), "error": e.to_string() })),
        )
    })?;

    fs::rename(&tmp_path, &path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        json_rpc_error(
            -32000,
            "Failed to write token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })?;
    Ok(())
}

pub(super) fn epoch_ms_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now();
    now.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub(super) fn epoch_ms_from_instant(instant: Instant) -> i64 {
    let now_instant = Instant::now();
    let now_ms = epoch_ms_now();
    if instant <= now_instant {
        return now_ms;
    }
    let delta = instant.duration_since(now_instant).as_millis() as i64;
    now_ms.saturating_add(delta)
}
