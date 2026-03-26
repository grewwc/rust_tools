use std::io::{self, BufRead, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{fs, os::unix::fs::PermissionsExt};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use rust_tools::common::configw;

struct CachedToken {
    token: String,
    expires_at: Instant,
}

static APP_TOKEN_CACHE: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();
static USER_TOKEN_CACHE: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(_) => return,
        };
        if n == 0 {
            return;
        }
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(err) => {
                let _ = write_json_rpc_error(&mut stdout, None, -32700, &err.to_string(), None);
                continue;
            }
        };

        let id = req.get("id").cloned();
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned();

        let res = match method.as_str() {
            "initialize" => handle_initialize(),
            "notifications/initialized" => Ok(json!({})),
            "tools/list" => handle_tools_list(),
            "tools/call" => handle_tools_call(params),
            "resources/list" => Ok(json!({"resources": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            _ => Err(json_rpc_error(-32601, "Method not found", Some(json!({ "method": method })))),
        };

        match res {
            Ok(result) => {
                let _ = write_json_rpc_result(&mut stdout, id.as_ref(), result);
            }
            Err(err) => {
                let _ = write_json_rpc_error(
                    &mut stdout,
                    id.as_ref(),
                    err.code,
                    &err.message,
                    err.data,
                );
            }
        }
    }
}

fn handle_initialize() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
            "resources": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": "mcp-feishu",
            "version": "0.1.0"
        }
    }))
}

fn handle_tools_list() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "tools": [
            {
                "name": "docs_search",
                "description": "Search Feishu cloud docs by keyword (requires user_access_token; supports OAuth + local token store for auto refresh)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_key": { "type": "string", "description": "Search keyword" },
                        "count": { "type": "integer", "description": "Number of results (0-50)" },
                        "offset": { "type": "integer", "description": "Offset (offset + count < 200)" },
                        "owner_ids": { "type": "array", "items": { "type": "string" }, "description": "Owner open_ids" },
                        "chat_ids": { "type": "array", "items": { "type": "string" }, "description": "Chat IDs" },
                        "docs_types": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "doc/sheet/slides/bitable/mindnote/file"
                        }
                    },
                    "required": ["search_key"]
                }
            },
            {
                "name": "docs_get_text",
                "description": "Fetch plain text content for a Feishu doc/docx (raw_content). Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "docs_token": { "type": "string", "description": "docs_token from docs_search" },
                        "docs_type": { "type": "string", "description": "doc or docx" },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" }
                    },
                    "required": ["docs_token", "docs_type"]
                }
            },
            {
                "name": "docs_export_text",
                "description": "Export plain text content for a Feishu doc/docx to a local file and return the path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "docs_token": { "type": "string", "description": "docs_token from docs_search" },
                        "docs_type": { "type": "string", "description": "doc or docx" },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" },
                        "out_dir": { "type": "string", "description": "Optional output directory. Default: ~/.config/rust_tools/feishu_docs_text" }
                    },
                    "required": ["docs_token", "docs_type"]
                }
            },
            {
                "name": "oauth_authorize_url",
                "description": "Build Feishu OAuth authorize URL to obtain code (for user_access_token). You must configure redirect_uri in Feishu app console.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "redirect_uri": { "type": "string", "description": "Redirect URI configured in Feishu console. Default: http://127.0.0.1:8711/callback" },
                        "scope": { "type": "string", "description": "Scopes separated by space. Tip: include offline_access if you need refresh_token." },
                        "state": { "type": "string", "description": "Opaque state string for CSRF protection" },
                        "prompt": { "type": "string", "description": "Optional. Use \"consent\" to force explicit consent UI." }
                    }
                }
            },
            {
                "name": "oauth_wait_local_code",
                "description": "Start a local HTTP listener and wait for OAuth redirect to capture code. Use with redirect_uri=http://127.0.0.1:<port>/callback",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "port": { "type": "integer", "description": "Local port to listen on (default 8711)" },
                        "timeout_sec": { "type": "integer", "description": "Wait timeout in seconds (default 180)" }
                    }
                }
            },
            {
                "name": "oauth_exchange_code",
                "description": "Exchange OAuth code for user_access_token (requires app_id/app_secret). Returns user_access_token and refresh_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": { "type": "string", "description": "OAuth authorization code (valid ~5 minutes, single-use)" }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "oauth_refresh_user_access_token",
                "description": "Refresh user_access_token using refresh_token (requires app_id/app_secret).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh_token": { "type": "string", "description": "Refresh token. If omitted, uses FEISHU_REFRESH_TOKEN env or feishu.refresh_token in ~/.configW" }
                    }
                }
            }
        ]
    }))
}

fn handle_tools_call(params: Option<Value>) -> Result<Value, JsonRpcErr> {
    let params = params.unwrap_or_else(|| json!({}));
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

    match name.as_str() {
        "docs_search" => {
            let text = feishu_docs_search(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "docs_get_text" => {
            let text = feishu_docs_get_text(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "docs_export_text" => {
            let text = feishu_docs_export_text(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_authorize_url" => {
            let text = feishu_oauth_authorize_url(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_wait_local_code" => {
            let text = feishu_oauth_wait_local_code(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_exchange_code" => {
            let text = feishu_oauth_exchange_code(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_refresh_user_access_token" => {
            let text = feishu_oauth_refresh_user_access_token(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        _ => Err(json_rpc_error(
            -32602,
            "Invalid params: unknown tool name",
            Some(json!({ "name": name })),
        )),
    }
}

fn feishu_docs_search(args: &Value) -> Result<String, JsonRpcErr> {
    let base_url = resolve_base_url();

    let search_key = args
        .get("search_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if search_key.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: search_key is empty", None));
    }

    let mut body = json!({
        "search_key": search_key,
        "count": args.get("count").and_then(|v| v.as_i64()).unwrap_or(10).clamp(0, 50),
        "offset": args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0).max(0),
    });

    if let Some(v) = args.get("owner_ids") {
        body["owner_ids"] = v.clone();
    }
    if let Some(v) = args.get("chat_ids") {
        body["chat_ids"] = v.clone();
    }
    if let Some(v) = args.get("docs_types") {
        body["docs_types"] = v.clone();
    }

    let url = format!(
        "{}/open-apis/suite/docs-api/search/object",
        base_url.trim_end_matches('/')
    );
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .map_err(|e| json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() }))))?;

    let token = if let Some(tok) = resolve_user_access_token() {
        tok
    } else if let Ok(tok) = get_user_access_token_cached(&client, &base_url) {
        tok
    } else {
        return Err(json_rpc_error(
            -32000,
            "Missing user_access_token. docs-api search requires user_access_token. Use OAuth once, then keep refresh_token for automatic refresh.",
            Some(json!({
                "next_steps": [
                    "Call oauth_authorize_url to get an authorization URL",
                    "Open the URL in a browser, complete authorization",
                    "Call oauth_wait_local_code (or copy code from redirect URL)",
                    "Call oauth_exchange_code to get user_access_token + refresh_token",
                    "Then store FEISHU_REFRESH_TOKEN (and app_id/app_secret) so the token can be refreshed automatically"
                ],
                "config_keys": ["feishu.app_id", "feishu.app_secret", "feishu.user_access_token", "feishu.refresh_token"],
                "env": ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_USER_ACCESS_TOKEN", "FEISHU_REFRESH_TOKEN"]
            })),
        ));
    };

    let (mut status, mut text) = do_docs_search_request(&client, &url, &token, &body)?;
    if !status.is_success() {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
            if code == 99991668 {
                if let Ok(tok) = refresh_user_access_token_and_cache(&client, &base_url) {
                    let (s2, t2) = do_docs_search_request(&client, &url, &tok, &body)?;
                    status = s2;
                    text = t2;
                }
            }
        }
    }
    if !status.is_success() {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
            if code == 99991668 {
                return Err(json_rpc_error(
                    -32000,
                    "Invalid access token. Provide a valid user_access_token or set refresh_token for automatic refresh.",
                    Some(json!({
                        "status": status.as_u16(),
                        "feishu_code": code,
                        "msg": v.get("msg").cloned().unwrap_or(Value::Null),
                        "next_steps": [
                            "If you have refresh_token: call oauth_refresh_user_access_token, then update FEISHU_USER_ACCESS_TOKEN (or set FEISHU_REFRESH_TOKEN for auto refresh)",
                            "Otherwise: run oauth_authorize_url -> oauth_wait_local_code -> oauth_exchange_code"
                        ]
                    })),
                ));
            }
        }
        return Err(json_rpc_error(
            -32000,
            "Feishu API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text)
        .map_err(|e| json_rpc_error(-32000, "Feishu API response is not valid JSON", Some(json!({ "error": e.to_string(), "body": text }))))?;

    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "Feishu API returned error code",
            Some(v),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let total = data.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
    let has_more = data.get("has_more").and_then(|v| v.as_bool()).unwrap_or(false);
    let docs = data
        .get("docs_entities")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut out = String::new();
    out.push_str(&format!("total: {total}, has_more: {has_more}\n"));
    for (i, item) in docs.iter().enumerate() {
        let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let docs_type = item.get("docs_type").and_then(|v| v.as_str()).unwrap_or("");
        let token = item.get("docs_token").and_then(|v| v.as_str()).unwrap_or("");
        let owner_id = item.get("owner_id").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(&format!(
            "{}. [{}] {} (token: {}, owner_id: {})\n",
            i + 1,
            docs_type,
            title.trim(),
            token.trim(),
            owner_id.trim()
        ));
    }
    Ok(out.trim_end().to_string())
}

fn feishu_docs_get_text(args: &Value) -> Result<String, JsonRpcErr> {
    let docs_token = args
        .get("docs_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let docs_type = args
        .get("docs_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if docs_token.is_empty() || docs_type.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: docs_token/docs_type required",
            Some(json!({ "docs_token": docs_token, "docs_type": docs_type })),
        ));
    }

    let lang = args.get("lang").and_then(|v| v.as_i64()).unwrap_or(0).clamp(0, 2);

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() }))))?;

    let token = if let Some(tok) = resolve_user_access_token() {
        tok
    } else {
        get_user_access_token_cached(&client, &base_url).map_err(|e| {
            json_rpc_error(
                -32000,
                "Missing user_access_token. Fetch requires OAuth once.",
                Some(json!({
                    "detail": e.message,
                    "next_steps": ["oauth_authorize_url", "oauth_wait_local_code", "oauth_exchange_code"],
                    "token_store": token_store_path().display().to_string()
                })),
            )
        })?
    };

    let content = feishu_fetch_raw_content(&client, &base_url, &token, &docs_type, &docs_token, lang)?;
    Ok(content)
}

fn feishu_docs_export_text(args: &Value) -> Result<String, JsonRpcErr> {
    let docs_token = args
        .get("docs_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let docs_type = args
        .get("docs_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if docs_token.is_empty() || docs_type.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: docs_token/docs_type required",
            Some(json!({ "docs_token": docs_token, "docs_type": docs_type })),
        ));
    }
    let lang = args.get("lang").and_then(|v| v.as_i64()).unwrap_or(0).clamp(0, 2);

    let out_dir = args
        .get("out_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let out_dir = if out_dir.is_empty() {
        rust_tools::common::utils::expanduser("~/.config/rust_tools/feishu_docs_text")
            .as_ref()
            .to_string()
    } else {
        rust_tools::common::utils::expanduser(&out_dir).as_ref().to_string()
    };

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() }))))?;

    let token = if let Some(tok) = resolve_user_access_token() {
        tok
    } else {
        get_user_access_token_cached(&client, &base_url)?
    };

    let content = feishu_fetch_raw_content(&client, &base_url, &token, &docs_type, &docs_token, lang)?;

    let dir = PathBuf::from(&out_dir);
    fs::create_dir_all(&dir).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to create output directory",
            Some(json!({ "out_dir": out_dir, "error": e.to_string() })),
        )
    })?;
    let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));

    let safe_type = docs_type.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let safe_token = docs_token.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let file_path = dir.join(format!("{safe_type}_{safe_token}.txt"));
    fs::write(&file_path, &content).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to write exported file",
            Some(json!({ "file_path": file_path.display().to_string(), "error": e.to_string() })),
        )
    })?;
    let _ = fs::set_permissions(&file_path, fs::Permissions::from_mode(0o600));

    Ok(format!("exported: {}", file_path.display()))
}

fn feishu_fetch_raw_content(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    docs_type: &str,
    docs_token: &str,
    lang: i64,
) -> Result<String, JsonRpcErr> {
    let (url, is_docx) = match docs_type {
        "docx" => (
            format!(
                "{}/open-apis/docx/v1/documents/{}/raw_content?lang={}",
                base_url.trim_end_matches('/'),
                docs_token,
                lang
            ),
            true,
        ),
        "doc" => (
            format!(
                "{}/open-apis/doc/v2/{}/raw_content",
                base_url.trim_end_matches('/'),
                docs_token
            ),
            false,
        ),
        _ => {
            return Err(json_rpc_error(
                -32602,
                "Unsupported docs_type (only doc/docx supported for now)",
                Some(json!({ "docs_type": docs_type })),
            ));
        }
    };

    let resp = client
        .get(url)
        .header("Authorization", format!("Bearer {}", user_access_token.trim()))
        .header("Content-Type", if is_docx { "application/json; charset=utf-8" } else { "text/plain" })
        .send()
        .map_err(|e| json_rpc_error(-32000, "Failed to fetch raw_content", Some(json!({ "error": e.to_string() }))))?;
    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| json_rpc_error(-32000, "Failed to read raw_content response", Some(json!({ "error": e.to_string() }))))?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "raw_content API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| json_rpc_error(-32000, "raw_content response is not valid JSON", Some(json!({ "error": e.to_string(), "body": text }))))?;
    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(-32000, "raw_content API returned error code", Some(v)));
    }
    let content = v
        .get("data")
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Ok(content)
}

fn do_docs_search_request(
    client: &Client,
    url: &str,
    token: &str,
    body: &Value,
) -> Result<(reqwest::StatusCode, String), JsonRpcErr> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(body)
        .send()
        .map_err(|e| json_rpc_error(-32000, "Failed to call Feishu API", Some(json!({ "error": e.to_string() }))))?;

    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| json_rpc_error(-32000, "Failed to read response body", Some(json!({ "error": e.to_string() }))))?;
    Ok((status, text))
}

fn resolve_base_url() -> String {
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

fn resolve_accounts_base_url(base_url: &str) -> String {
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

fn resolve_user_access_token() -> Option<String> {
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

fn resolve_refresh_token() -> Option<String> {
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
    load_token_store()
        .ok()
        .and_then(|s| s.refresh_token)
}

fn resolve_app_credentials() -> Option<(String, String)> {
    let env_id = std::env::var("FEISHU_APP_ID").ok().map(|v| v.trim().to_string());
    let env_secret = std::env::var("FEISHU_APP_SECRET").ok().map(|v| v.trim().to_string());
    if let (Some(id), Some(secret)) = (env_id, env_secret) {
        if !id.is_empty() && !secret.is_empty() {
            return Some((id, secret));
        }
    }

    let cfg = configw::get_all_config();
    let id = cfg.get_opt("feishu.app_id").map(|v| v.trim().to_string());
    let secret = cfg.get_opt("feishu.app_secret").map(|v| v.trim().to_string());
    match (id, secret) {
        (Some(id), Some(secret)) if !id.is_empty() && !secret.is_empty() => Some((id, secret)),
        _ => None,
    }
}

fn get_user_access_token_cached(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    let now = Instant::now();
    if let Ok(guard) = cache.lock() {
        if let Some(cached) = guard.as_ref() {
            if cached.expires_at > now + Duration::from_secs(300) {
                return Ok(cached.token.clone());
            }
        }
    }
    refresh_user_access_token_and_cache(client, base_url)
}

fn refresh_user_access_token_and_cache(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
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
        refresh_token: Some(refreshed.refresh_token.clone()),
        refresh_token_expires_in: Some(refreshed.refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });
    Ok(refreshed.user_access_token)
}

struct RefreshedUserToken {
    user_access_token: String,
    refresh_token: String,
    expires_at: Instant,
    refresh_expires_in: i64,
}

fn refresh_user_access_token_api(
    client: &Client,
    base_url: &str,
    refresh_token: &str,
) -> Result<RefreshedUserToken, JsonRpcErr> {
    let app_access_token = get_app_access_token_cached(client, base_url)?;
    let url = format!(
        "{}/open-apis/authen/v1/refresh_access_token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    });
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", app_access_token))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| json_rpc_error(-32000, "Failed to call refresh_access_token API", Some(json!({ "error": e.to_string() }))))?;

    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| json_rpc_error(-32000, "Failed to read refresh_access_token response body", Some(json!({ "error": e.to_string() }))))?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "refresh_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| json_rpc_error(-32000, "refresh_access_token response is not valid JSON", Some(json!({ "error": e.to_string(), "body": text }))))?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(-32000, "refresh_access_token API returned error code", Some(v)));
    }
    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let user_access_token = data
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = data
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = data.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0).max(60) as u64;
    let refresh_expires_in = data.get("refresh_expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    if user_access_token.is_empty() {
        return Err(json_rpc_error(-32000, "Missing access_token in refresh response", Some(data)));
    }
    Ok(RefreshedUserToken {
        user_access_token,
        refresh_token,
        expires_at: Instant::now() + Duration::from_secs(expires_in),
        refresh_expires_in,
    })
}
fn get_app_access_token_cached(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    let cache = APP_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    let now = Instant::now();
    if let Ok(guard) = cache.lock() {
        if let Some(cached) = guard.as_ref() {
            if cached.expires_at > now + Duration::from_secs(300) {
                return Ok(cached.token.clone());
            }
        }
    }

    let Some((app_id, app_secret)) = resolve_app_credentials() else {
        return Err(json_rpc_error(
            -32000,
            "Missing Feishu app credentials (app_id/app_secret)",
            Some(json!({
                "env": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
                "config_keys": ["feishu.app_id", "feishu.app_secret"]
            })),
        ));
    };

    let (token, expires_at) = fetch_app_access_token(client, base_url, &app_id, &app_secret)?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken { token: token.clone(), expires_at });
    }
    Ok(token)
}

fn fetch_app_access_token(
    client: &Client,
    base_url: &str,
    app_id: &str,
    app_secret: &str,
) -> Result<(String, Instant), JsonRpcErr> {
    let url = format!(
        "{}/open-apis/auth/v3/app_access_token/internal",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "app_id": app_id,
        "app_secret": app_secret
    });

    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| json_rpc_error(-32000, "Failed to call app_access_token API", Some(json!({ "error": e.to_string() }))))?;

    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| json_rpc_error(-32000, "Failed to read app_access_token response body", Some(json!({ "error": e.to_string() }))))?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "app_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text)
        .map_err(|e| json_rpc_error(-32000, "app_access_token response is not valid JSON", Some(json!({ "error": e.to_string(), "body": text }))))?;
    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(-32000, "app_access_token API returned error code", Some(v)));
    }

    let token = v
        .get("app_access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if token.is_empty() {
        return Err(json_rpc_error(-32000, "app_access_token missing in response", Some(v)));
    }
    let expire = v.get("expire").and_then(|v| v.as_i64()).unwrap_or(0).max(60) as u64;
    let expires_at = Instant::now() + Duration::from_secs(expire);
    Ok((token, expires_at))
}

fn feishu_oauth_authorize_url(args: &Value) -> Result<String, JsonRpcErr> {
    let cfg = configw::get_all_config();
    let app_id = cfg
        .get_opt("feishu.app_id")
        .or_else(|| std::env::var("FEISHU_APP_ID").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| json_rpc_error(-32000, "Missing feishu.app_id / FEISHU_APP_ID", None))?;

    let base_url = resolve_base_url();
    let accounts_base = resolve_accounts_base_url(&base_url);

    let redirect_uri = args
        .get("redirect_uri")
        .and_then(|v| v.as_str())
        .unwrap_or("http://127.0.0.1:8711/callback")
        .trim()
        .to_string();
    if redirect_uri.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: redirect_uri is empty", None));
    }

    let scope = args
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("offline_access")
        .trim()
        .to_string();
    let state = args.get("state").and_then(|v| v.as_str()).unwrap_or("rust-tools-ai");
    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    let encoded_redirect = url_encode_component(&redirect_uri);
    let encoded_scope = url_encode_component(&scope);
    let encoded_state = url_encode_component(state);

    let mut url = format!(
        "{}/open-apis/authen/v1/authorize?client_id={}&response_type=code&redirect_uri={}&scope={}&state={}",
        accounts_base.trim_end_matches('/'),
        url_encode_component(&app_id),
        encoded_redirect,
        encoded_scope,
        encoded_state
    );
    if !prompt.trim().is_empty() {
        url.push_str("&prompt=");
        url.push_str(&url_encode_component(prompt.trim()));
    }
    Ok(url)
}

fn feishu_oauth_wait_local_code(args: &Value) -> Result<String, JsonRpcErr> {
    let port = args.get("port").and_then(|v| v.as_i64()).unwrap_or(8711).clamp(1, 65535) as u16;
    let timeout_sec = args.get("timeout_sec").and_then(|v| v.as_i64()).unwrap_or(180).clamp(1, 600) as u64;
    let addr = format!("127.0.0.1:{port}");

    let listener = TcpListener::bind(&addr)
        .map_err(|e| json_rpc_error(-32000, "Failed to bind local callback port", Some(json!({ "addr": addr, "error": e.to_string() }))))?;
    listener
        .set_nonblocking(true)
        .ok();

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout_sec);
        loop {
            if Instant::now() >= deadline {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let mut buf = [0u8; 2048];
                    let n = stream.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let code = extract_code_from_http_request(&req).unwrap_or_default();
                    let body = if code.is_empty() {
                        "<html><body>Missing code</body></html>"
                    } else {
                        "<html><body>OK. You can close this tab.</body></html>"
                    };
                    let _ = stream.write_all(format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    ).as_bytes());
                    let _ = stream.flush();
                    if !code.is_empty() {
                        let _ = tx.send(code);
                        return;
                    }
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    });

    match rx.recv_timeout(Duration::from_secs(timeout_sec)) {
        Ok(code) => Ok(format!("code: {code}\nport: {port}\npath: /callback")),
        Err(_) => Err(json_rpc_error(-32000, "Timeout waiting for OAuth code", Some(json!({ "port": port, "timeout_sec": timeout_sec })))),
    }
}

fn extract_code_from_http_request(req: &str) -> Option<String> {
    let first = req.lines().next()?.trim();
    let first = first.strip_prefix("GET ")?;
    let path = first.split_whitespace().next().unwrap_or("");
    let qidx = path.find('?')?;
    let query = &path[qidx + 1..];
    for part in query.split('&') {
        let mut it = part.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        if k == "code" && !v.trim().is_empty() {
            return url_decode_component(v);
        }
    }
    None
}

fn url_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match *b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char);
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn url_decode_component(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h1 = bytes[i + 1];
                let h2 = bytes[i + 2];
                let v1 = hex_value(h1)?;
                let v2 = hex_value(h2)?;
                out.push((v1 << 4) | v2);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn feishu_oauth_exchange_code(args: &Value) -> Result<String, JsonRpcErr> {
    let code = args
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if code.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: code is empty", None));
    }
    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() }))))?;

    let app_access_token = get_app_access_token_cached(&client, &base_url)?;
    let url = format!(
        "{}/open-apis/authen/v1/access_token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "authorization_code",
        "code": code
    });
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", app_access_token))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| json_rpc_error(-32000, "Failed to call user_access_token API", Some(json!({ "error": e.to_string() }))))?;

    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| json_rpc_error(-32000, "Failed to read user_access_token response body", Some(json!({ "error": e.to_string() }))))?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "user_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| json_rpc_error(-32000, "user_access_token response is not valid JSON", Some(json!({ "error": e.to_string(), "body": text }))))?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(-32000, "user_access_token API returned error code", Some(v)));
    }
    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let access_token = data.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let refresh_token = data.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let expires_in = data.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    let refresh_expires_in = data.get("refresh_expires_in").and_then(|v| v.as_i64()).unwrap_or(0);

    if access_token.is_empty() {
        return Err(json_rpc_error(-32000, "Missing access_token in response", Some(data)));
    }

    let expires_at = Instant::now() + Duration::from_secs(expires_in.max(60) as u64);
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(access_token),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(expires_at)),
        refresh_token: (!refresh_token.is_empty()).then_some(refresh_token),
        refresh_token_expires_in: Some(refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });

    Ok(format!(
        "OAuth exchange success.\n- Stored tokens in: {}\n- expires_in: {}\n- refresh_expires_in: {}\n\nYou can now call docs_search without setting FEISHU_USER_ACCESS_TOKEN.",
        token_store_path().display(),
        expires_in,
        refresh_expires_in
    ))
}

fn feishu_oauth_refresh_user_access_token(args: &Value) -> Result<String, JsonRpcErr> {
    let refresh_token = args
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = if !refresh_token.is_empty() {
        Some(refresh_token)
    } else {
        resolve_refresh_token()
    }
    .ok_or_else(|| json_rpc_error(-32000, "Missing refresh_token", None))?;

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() }))))?;

    let refreshed = refresh_user_access_token_api(&client, &base_url, &refresh_token)?;
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(refreshed.user_access_token.clone()),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(refreshed.expires_at)),
        refresh_token: (!refreshed.refresh_token.is_empty()).then_some(refreshed.refresh_token.clone()),
        refresh_token_expires_in: Some(refreshed.refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });
    Ok(format!(
        "Refresh success.\n- Stored tokens in: {}\n- expires_in: {}\n- refresh_expires_in: {}",
        token_store_path().display(),
        refreshed
            .expires_at
            .saturating_duration_since(Instant::now())
            .as_secs() as i64,
        refreshed.refresh_expires_in
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenStore {
    user_access_token: Option<String>,
    user_access_token_expires_at_epoch_ms: Option<i64>,
    refresh_token: Option<String>,
    refresh_token_expires_in: Option<i64>,
    updated_at_epoch_ms: Option<i64>,
}

fn token_store_path() -> PathBuf {
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
            return PathBuf::from(rust_tools::common::utils::expanduser(&v).as_ref());
        }
    }
    PathBuf::from(rust_tools::common::utils::expanduser("~/.config/rust_tools/feishu_token.json").as_ref())
}

fn load_token_store() -> Result<TokenStore, JsonRpcErr> {
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

fn save_token_store(store: &TokenStore) -> Result<(), JsonRpcErr> {
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
    let s = serde_json::to_string_pretty(store)
        .map_err(|e| json_rpc_error(-32000, "Failed to serialize token store", Some(json!({ "error": e.to_string() }))))?;

    fs::write(&path, s).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to write token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })?;

    let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    Ok(())
}

fn epoch_ms_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now();
    now.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn epoch_ms_from_instant(instant: Instant) -> i64 {
    let now_instant = Instant::now();
    let now_ms = epoch_ms_now();
    if instant <= now_instant {
        return now_ms;
    }
    let delta = instant.duration_since(now_instant).as_millis() as i64;
    now_ms.saturating_add(delta)
}

#[derive(Debug, Clone)]
struct JsonRpcErr {
    code: i64,
    message: String,
    data: Option<Value>,
}

fn json_rpc_error(code: i64, message: &str, data: Option<Value>) -> JsonRpcErr {
    JsonRpcErr {
        code,
        message: message.to_string(),
        data,
    }
}

fn write_json_rpc_result(out: &mut dyn Write, id: Option<&Value>, result: Value) -> io::Result<()> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "result": result
    });
    writeln!(out, "{}", payload.to_string())?;
    out.flush()
}

fn write_json_rpc_error(
    out: &mut dyn Write,
    id: Option<&Value>,
    code: i64,
    message: &str,
    data: Option<Value>,
) -> io::Result<()> {
    let mut err = json!({
        "code": code,
        "message": message
    });
    if let Some(d) = data {
        err["data"] = d;
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "error": err
    });
    writeln!(out, "{}", payload.to_string())?;
    out.flush()
}
