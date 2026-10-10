use super::*;

pub(super) fn feishu_oauth_authorize_url(args: &Value) -> Result<String, JsonRpcErr> {
    let cfg = configw::get_all_config();
    let client_id = cfg
        .get_opt("feishu.client_id")
        .or_else(|| cfg.get_opt("feishu.app_id"))
        .or_else(|| std::env::var("FEISHU_CLIENT_ID").ok())
        .or_else(|| std::env::var("FEISHU_APP_ID").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            json_rpc_error(
                -32000,
                "Missing feishu.client_id / FEISHU_CLIENT_ID",
                Some(json!({
                    "legacy_env": ["FEISHU_APP_ID"],
                    "legacy_config_keys": ["feishu.app_id"]
                })),
            )
        })?;

    let base_url = resolve_base_url();
    let accounts_base = resolve_accounts_base_url(&base_url);

    let redirect_uri = args
        .get("redirect_uri")
        .and_then(|v| v.as_str())
        .unwrap_or("http://127.0.0.1:8711/callback")
        .trim()
        .to_string();
    if redirect_uri.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: redirect_uri is empty",
            None,
        ));
    }

    let scope = args
        .get("scope")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(FEISHU_SCOPE);
    let state = args
        .get("state")
        .and_then(|v| v.as_str())
        .unwrap_or("rust-tools-ai");
    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    let encoded_redirect = url_encode_component(&redirect_uri);
    let encoded_scope = url_encode_component(scope);
    let encoded_state = url_encode_component(state);

    let mut url = format!(
        "{}/open-apis/authen/v1/authorize?client_id={}&response_type=code&redirect_uri={}&scope={}&state={}",
        accounts_base.trim_end_matches('/'),
        url_encode_component(&client_id),
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

pub(super) fn feishu_oauth_wait_local_code(args: &Value) -> Result<String, JsonRpcErr> {
    let port = args
        .get("port")
        .and_then(|v| v.as_i64())
        .unwrap_or(8711)
        .clamp(1, 65535) as u16;
    let timeout_sec = args
        .get("timeout_sec")
        .and_then(|v| v.as_i64())
        .unwrap_or(180)
        .clamp(1, 600) as u64;
    let mut listeners: Vec<TcpListener> = Vec::new();
    let addr4 = format!("127.0.0.1:{port}");
    match TcpListener::bind(&addr4) {
        Ok(l) => {
            l.set_nonblocking(true).ok();
            listeners.push(l);
        }
        Err(e) => {
            let _ = e;
        }
    }
    let addr6 = format!("[::1]:{port}");
    match TcpListener::bind(&addr6) {
        Ok(l) => {
            l.set_nonblocking(true).ok();
            listeners.push(l);
        }
        Err(e) => {
            let _ = e;
        }
    }
    if listeners.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Failed to bind local callback port",
            Some(json!({ "port": port, "addrs": [addr4, addr6] })),
        ));
    }

    let (tx, rx) = mpsc::channel::<String>();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let worker_stop_flag = Arc::clone(&stop_flag);
    let worker = std::thread::spawn(move || {
        // worker 不再自持 deadline：唯一权威超时源是主线程的 recv_timeout，
        // 避免"两个独立倒计时叠加"——worker 曾因自身 deadline 先到而在即将
        // send 已收到的 code 前退出、把授权码丢弃。worker 只在 stop_flag 置位时退出。
        loop {
            if worker_stop_flag.load(Ordering::Relaxed) {
                break;
            }
            let mut accepted: Option<TcpStream> = None;
            for listener in &listeners {
                if let Ok((stream, _)) = listener.accept() {
                    accepted = Some(stream);
                    break;
                }
            }

            let Some(mut stream) = accepted else {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            };

            let req = read_http_request(&mut stream);
            let code = parse_oauth_code_from_http_request(&req).unwrap_or_default();
            if !code.is_empty() {
                let body = "<html><body>OK. You can close this tab.</body></html>";
                let _ = write_http_response(&mut stream, body);
                let _ = tx.send(code);
                return;
            }

            let body = r#"<html><head><meta charset="utf-8"></head><body>
<div>Waiting for OAuth code...</div>
<script>
  (function () {
    try {
      var url = new URL(window.location.href);
      var code = url.searchParams.get('code');
      if (!code && window.location.hash && window.location.hash.length > 1) {
        var hash = window.location.hash.substring(1);
        var params = new URLSearchParams(hash);
        code = params.get('code');
      }
      if (code) {
        url.hash = '';
        url.searchParams.set('code', code);
        window.location.replace(url.toString());
        return;
      }
    } catch (e) {}
    document.body.innerHTML = '<div>Missing code. If you see a "code" in the URL, copy it and paste back into the CLI.</div>';
  })();
</script>
</body></html>"#;
            let _ = write_http_response(&mut stream, body);
        }
    });

    let result = match rx.recv_timeout(Duration::from_secs(timeout_sec)) {
        Ok(code) => Ok(format!("code: {code}\nport: {port}\npath: /callback")),
        Err(_) => Err(json_rpc_error(
            -32000,
            "Timeout waiting for OAuth code",
            Some(json!({ "port": port, "timeout_sec": timeout_sec })),
        )),
    };
    stop_flag.store(true, Ordering::Relaxed);
    let _ = worker.join();
    result
}

pub(super) fn read_http_request(stream: &mut TcpStream) -> String {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(8)));
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    let mut buf = [0u8; 1024];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if Instant::now() >= deadline || out.len() >= 16_384 {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                if out.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

pub(super) fn write_http_response(stream: &mut TcpStream, body: &str) -> io::Result<()> {
    let bytes = body.as_bytes();
    stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .as_bytes(),
    )?;
    stream.write_all(bytes)?;
    stream.flush()
}

pub(super) fn parse_oauth_code_from_http_request(req: &str) -> Option<String> {
    let first = req.lines().next()?.trim();
    let mut parts = first.split_whitespace();
    let _method = parts.next()?;
    let target = parts.next().unwrap_or("");
    if let Some(code) = parse_oauth_code_from_urlish(target) {
        return Some(code);
    }
    if let Some(idx) = req.find("\r\n\r\n") {
        let body = &req[idx + 4..];
        if let Some(code) = parse_oauth_code_from_query(body) {
            return Some(code);
        }
    }
    None
}

pub(super) fn parse_oauth_code_from_urlish(target: &str) -> Option<String> {
    let target = target.trim();
    if target.is_empty() {
        return None;
    }
    let without_fragment = target.split('#').next().unwrap_or(target);
    let qidx = without_fragment.find('?')?;
    let query = &without_fragment[qidx + 1..];
    parse_oauth_code_from_query(query)
}

pub(super) fn parse_oauth_code_from_query(query: &str) -> Option<String> {
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

pub(super) fn url_encode_component(s: &str) -> String {
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

pub(super) fn url_decode_component(s: &str) -> Option<String> {
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

pub(super) fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

pub(super) fn feishu_oauth_exchange_code(args: &Value) -> Result<String, JsonRpcErr> {
    let code = args
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let redirect_uri = args
        .get("redirect_uri")
        .and_then(|v| v.as_str())
        .unwrap_or("http://127.0.0.1:8711/callback")
        .trim()
        .to_string();
    if code.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: code is empty",
            None,
        ));
    }
    if redirect_uri.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: redirect_uri is empty",
            None,
        ));
    }
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
    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let url = format!(
        "{}/open-apis/authen/v2/oauth/token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri
    });
    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call user_access_token API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read user_access_token response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "user_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "user_access_token response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(
            -32000,
            "user_access_token API returned error code",
            Some(v),
        ));
    }
    let access_token = v
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = v
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = v.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    let refresh_expires_in = v
        .get("refresh_token_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    if access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in response",
            Some(v),
        ));
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

pub(super) fn feishu_oauth_refresh_user_access_token(args: &Value) -> Result<String, JsonRpcErr> {
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
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let refreshed = refresh_user_access_token_api(&client, &base_url, &refresh_token)?;
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(refreshed.user_access_token.clone()),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(refreshed.expires_at)),
        // 未轮换（返回空）时保留本次使用的 refresh_token，避免把有效凭据清空。
        refresh_token: (!refreshed.refresh_token.is_empty())
            .then(|| refreshed.refresh_token.clone())
            .or_else(|| Some(refresh_token.clone())),
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
