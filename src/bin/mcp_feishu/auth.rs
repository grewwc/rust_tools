use super::*;

pub(super) struct CachedToken {
    token: String,
    expires_at: Instant,
}

pub(super) static USER_TOKEN_CACHE: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();
pub(super) fn resolve_base_url() -> String {
    if let Ok(v) = std::env::var("FEISHU_BASE_URL") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.base_url") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    "https://open.feishu.cn".to_string()
}

pub(super) fn resolve_accounts_base_url(base_url: &str) -> String {
    if let Ok(v) = std::env::var("FEISHU_ACCOUNTS_BASE_URL") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.accounts_base_url") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    if base_url.contains("larksuite") {
        "https://accounts.larksuite.com".to_string()
    } else {
        "https://accounts.feishu.cn".to_string()
    }
}

pub(super) fn resolve_user_access_token() -> Option<String> {
    if let Ok(v) = std::env::var("FEISHU_USER_ACCESS_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() && v.starts_with("u-") {
            return Some(v);
        }
    }
    if let Ok(v) = std::env::var("FEISHU_ACCESS_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() && v.starts_with("u-") {
            return Some(v);
        }
    }
    let cfg = configw::get_all_config();
    cfg.get_opt("feishu.user_access_token")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && v.starts_with("u-"))
}

pub(super) fn resolve_refresh_token() -> Option<String> {
    if let Ok(v) = std::env::var("FEISHU_REFRESH_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    let cfg = configw::get_all_config();
    let v = cfg
        .get_opt("feishu.refresh_token")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if v.is_some() {
        return v;
    }
    load_token_store().ok().and_then(|s| s.refresh_token)
}

pub(super) fn resolve_stored_user_access_token() -> Option<(String, Option<i64>)> {
    let store = load_token_store().ok()?;
    let token = store.user_access_token?.trim().to_string();
    if token.is_empty() || !token.starts_with("u-") {
        return None;
    }
    let expires_at = store.user_access_token_expires_at_epoch_ms;
    if let Some(epoch_ms) = expires_at
        && epoch_ms > 0
        && epoch_ms <= epoch_ms_now() + 300_000
    {
        return None;
    }
    Some((token, expires_at))
}

pub(super) fn resolve_client_credentials() -> Option<(String, String)> {
    let env_id = std::env::var("FEISHU_CLIENT_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .or_else(|| {
            std::env::var("FEISHU_APP_ID")
                .ok()
                .map(|v| v.trim().to_string())
        });
    let env_secret = std::env::var("FEISHU_CLIENT_SECRET")
        .ok()
        .map(|v| v.trim().to_string())
        .or_else(|| {
            std::env::var("FEISHU_APP_SECRET")
                .ok()
                .map(|v| v.trim().to_string())
        });
    if let (Some(id), Some(secret)) = (env_id, env_secret)
        && !id.is_empty()
        && !secret.is_empty()
    {
        return Some((id, secret));
    }

    let cfg = configw::get_all_config();
    let id = cfg
        .get_opt("feishu.client_id")
        .map(|v| v.trim().to_string())
        .or_else(|| cfg.get_opt("feishu.app_id").map(|v| v.trim().to_string()));
    let secret = cfg
        .get_opt("feishu.client_secret")
        .map(|v| v.trim().to_string())
        .or_else(|| {
            cfg.get_opt("feishu.app_secret")
                .map(|v| v.trim().to_string())
        });
    match (id, secret) {
        (Some(id), Some(secret)) if !id.is_empty() && !secret.is_empty() => Some((id, secret)),
        _ => None,
    }
}

pub(super) fn get_user_access_token_cached(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    let now = Instant::now();
    if let Ok(guard) = cache.lock()
        && let Some(cached) = guard.as_ref()
        && cached.expires_at > now + Duration::from_secs(300)
    {
        return Ok(cached.token.clone());
    }
    refresh_user_access_token_and_cache(client, base_url)
}

pub(super) fn cache_user_access_token(token: &str, expires_at_epoch_ms: Option<i64>) {
    if token.trim().is_empty() {
        return;
    }
    let expires_at = match expires_at_epoch_ms {
        Some(ms) if ms > epoch_ms_now() => {
            Instant::now() + Duration::from_millis(ms.saturating_sub(epoch_ms_now()) as u64)
        }
        _ => Instant::now() + Duration::from_secs(600),
    };
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken {
            token: token.trim().to_string(),
            expires_at,
        });
    }
}

pub(super) fn acquire_user_access_token(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    if let Some(tok) = resolve_user_access_token() {
        return Ok(tok);
    }
    if let Some((tok, expires_at_epoch_ms)) = resolve_stored_user_access_token() {
        cache_user_access_token(&tok, expires_at_epoch_ms);
        return Ok(tok);
    }
    get_user_access_token_cached(client, base_url)
}

pub(super) fn with_user_access_token<T, F>(
    client: &Client,
    base_url: &str,
    missing_message: &str,
    mut op: F,
) -> Result<T, JsonRpcErr>
where
    F: FnMut(&str) -> Result<T, JsonRpcErr>,
{
    let primary = acquire_user_access_token(client, base_url).map_err(|e| {
        json_rpc_error(
            -32000,
            missing_message,
            Some(json!({
                "detail": e.message,
                "next_steps": ["oauth_authorize_url", "oauth_wait_local_code", "oauth_exchange_code"],
                "token_store": token_store_path().display().to_string()
            })),
        )
    })?;

    let mut err = match op(&primary) {
        Ok(v) => return Ok(v),
        Err(err) => err,
    };
    if !is_invalid_user_access_token_error(&err) {
        return Err(err);
    }

    if let Some((stored, expires_at_epoch_ms)) = resolve_stored_user_access_token()
        && stored != primary
    {
        cache_user_access_token(&stored, expires_at_epoch_ms);
        match op(&stored) {
            Ok(v) => return Ok(v),
            Err(next_err) => {
                if !is_invalid_user_access_token_error(&next_err) {
                    return Err(next_err);
                }
                err = next_err;
            }
        }
    }

    if let Ok(refreshed) = refresh_user_access_token_and_cache(client, base_url)
        && refreshed != primary
    {
        match op(&refreshed) {
            Ok(v) => return Ok(v),
            Err(next_err) => err = next_err,
        }
    }

    if is_invalid_user_access_token_error(&err) {
        Err(invalid_user_access_token_error(err))
    } else {
        Err(err)
    }
}

pub(super) fn is_invalid_user_access_token_error(err: &JsonRpcErr) -> bool {
    if err.message.contains("Invalid access token") {
        return true;
    }
    extract_feishu_error_code(err.data.as_ref()) == Some(99991668)
}

pub(super) fn extract_feishu_error_code(data: Option<&Value>) -> Option<i64> {
    let data = data?;
    if let Some(code) = data.get("feishu_code").and_then(|v| v.as_i64()) {
        return Some(code);
    }
    if let Some(code) = data.get("code").and_then(|v| v.as_i64()) {
        return Some(code);
    }
    let body = data.get("body").and_then(|v| v.as_str())?;
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("code").and_then(|x| x.as_i64()))
}

pub(super) fn extract_feishu_error_message(data: Option<&Value>) -> Option<String> {
    let data = data?;
    if let Some(msg) = data.get("msg").and_then(|v| v.as_str()) {
        return Some(msg.trim().to_string());
    }
    let body = data.get("body").and_then(|v| v.as_str())?;
    serde_json::from_str::<Value>(body).ok().and_then(|v| {
        v.get("msg")
            .and_then(|x| x.as_str())
            .map(|s| s.trim().to_string())
    })
}

pub(super) fn invalid_user_access_token_error(err: JsonRpcErr) -> JsonRpcErr {
    let status = err
        .data
        .as_ref()
        .and_then(|v| v.get("status"))
        .and_then(|v| v.as_u64());
    let feishu_code = extract_feishu_error_code(err.data.as_ref());
    let msg = extract_feishu_error_message(err.data.as_ref()).unwrap_or(err.message);
    json_rpc_error(
        -32000,
        "Invalid access token. Provide a valid user_access_token or set refresh_token for automatic refresh.",
        Some(json!({
            "status": status,
            "feishu_code": feishu_code,
            "msg": msg,
            "token_store": token_store_path().display().to_string(),
            "next_steps": [
                "If you have refresh_token: call oauth_refresh_user_access_token, then update FEISHU_USER_ACCESS_TOKEN (or set FEISHU_REFRESH_TOKEN for auto refresh)",
                "Otherwise: run oauth_authorize_url -> oauth_wait_local_code -> oauth_exchange_code"
            ]
        })),
    )
}

pub(super) fn refresh_user_access_token_and_cache(
    client: &Client,
    base_url: &str,
) -> Result<String, JsonRpcErr> {
    let refresh_token = resolve_refresh_token().ok_or_else(|| {
        json_rpc_error(
            -32000,
            "Missing refresh_token for refreshing user_access_token",
            Some(json!({
                "env": ["FEISHU_REFRESH_TOKEN"],
                "config_keys": ["feishu.refresh_token"],
                "token_store": token_store_path().display().to_string()
            })),
        )
    })?;
    let refreshed = refresh_user_access_token_api(client, base_url, &refresh_token)?;
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken {
            token: refreshed.user_access_token.clone(),
            expires_at: refreshed.expires_at,
        });
    }
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(refreshed.user_access_token.clone()),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(refreshed.expires_at)),
        // 飞书刷新接口不保证轮换 refresh_token；未轮换时返回空串。此处若无条件写回
        // 会把存储里仍然有效的 refresh_token 清空，导致后续自动刷新彻底失效。故仅在
        // 拿到非空新值时才覆盖，与 exchange_code / refresh 工具两条路径保持一致。
        refresh_token: (!refreshed.refresh_token.is_empty())
            .then(|| refreshed.refresh_token.clone())
            .or_else(|| Some(refresh_token.clone())),
        refresh_token_expires_in: Some(refreshed.refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });
    Ok(refreshed.user_access_token)
}

pub(super) struct RefreshedUserToken {
    pub(super) user_access_token: String,
    pub(super) refresh_token: String,
    pub(super) expires_at: Instant,
    pub(super) refresh_expires_in: i64,
}

pub(super) fn refresh_user_access_token_api(
    client: &Client,
    base_url: &str,
    refresh_token: &str,
) -> Result<RefreshedUserToken, JsonRpcErr> {
    let Some((client_id, client_secret)) = resolve_client_credentials() else {
        return Err(json_rpc_error(
            -32000,
            "Missing Feishu client credentials (client_id/client_secret)",
            Some(json!({
                "env": [
                    "FEISHU_CLIENT_ID",
                    "FEISHU_CLIENT_SECRET",
                    "FEISHU_APP_ID",
                    "FEISHU_APP_SECRET"
                ],
                "config_keys": [
                    "feishu.client_id",
                    "feishu.client_secret",
                    "feishu.app_id",
                    "feishu.app_secret"
                ]
            })),
        ));
    };
    let url = format!(
        "{}/open-apis/authen/v2/oauth/token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    });
    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call refresh_access_token API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read refresh_access_token response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "refresh_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "refresh_access_token response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(
            -32000,
            "refresh_access_token API returned error code",
            Some(v),
        ));
    }
    let user_access_token = v
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let next_refresh_token = v
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = v
        .get("expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(60) as u64;
    let refresh_expires_in = v
        .get("refresh_token_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if user_access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in refresh response",
            Some(v),
        ));
    }
    Ok(RefreshedUserToken {
        user_access_token,
        refresh_token: next_refresh_token,
        expires_at: Instant::now() + Duration::from_secs(expires_in),
        refresh_expires_in,
    })
}
