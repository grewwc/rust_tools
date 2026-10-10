use super::*;

pub(super) fn feishu_docs_get_text(args: &Value) -> Result<String, JsonRpcErr> {
    let docs_token = args
        .get("docs_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if docs_token.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: docs_token is required",
            None,
        ));
    }

    let docs_type = args
        .get("docs_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();

    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let max_rows = args
        .get("max_rows")
        .and_then(|v| v.as_i64())
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let max_cols = args
        .get("max_cols")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 200) as usize;
    let max_sheets = args
        .get("max_sheets")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .clamp(1, 20) as usize;

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| {
            let resolved_type = if docs_type.is_empty() {
                infer_docs_type(&docs_token)
            } else {
                docs_type.clone()
            };

            match resolved_type.as_str() {
                "wiki" => {
                    let (obj_type, obj_token) =
                        feishu_wiki_resolve_obj(&client, &base_url, token, &docs_token)?;
                    match obj_type.as_str() {
                        "doc" | "docx" => feishu_fetch_raw_content(
                            &client, &base_url, token, &obj_type, &obj_token, lang, None,
                        ),
                        "sheet" | "sheets" => feishu_fetch_sheet_preview_text(
                            &client, &base_url, token, &obj_token, max_rows, max_cols, max_sheets,
                        ),
                        other => Err(json_rpc_error(
                            -32000,
                            "Wiki node resolved to unsupported type",
                            Some(json!({ "obj_type": other, "obj_token": obj_token })),
                        )),
                    }
                }
                "sheet" | "sheets" => feishu_fetch_sheet_preview_text(
                    &client,
                    &base_url,
                    token,
                    &docs_token,
                    max_rows,
                    max_cols,
                    max_sheets,
                ),
                "doc" | "docx" => feishu_fetch_raw_content(
                    &client,
                    &base_url,
                    token,
                    &resolved_type,
                    &docs_token,
                    lang,
                    None,
                ),
                other => Err(json_rpc_error(
                    -32602,
                    "Unsupported docs_type",
                    Some(
                        json!({ "docs_type": other, "supported": ["doc", "docx", "wiki", "sheet"] }),
                    ),
                )),
            }
        },
    )
}

pub(super) fn infer_docs_type(token: &str) -> String {
    let t = token.trim();
    if t.len() < 10 {
        return "docx".to_string();
    }
    if t.starts_with("bascn") {
        return "sheet".to_string();
    }
    "docx".to_string()
}

pub(super) fn feishu_docs_get_text_by_url(args: &Value) -> Result<String, JsonRpcErr> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if url.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: url is empty", None));
    }

    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let max_rows = args
        .get("max_rows")
        .and_then(|v| v.as_i64())
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let max_cols = args
        .get("max_cols")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 200) as usize;
    let max_sheets = args
        .get("max_sheets")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .clamp(1, 20) as usize;

    let base_url = resolve_base_url_for_user_url(&url);
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let Some((kind, tok)) = parse_docs_url_kind_and_token(&url) else {
        return Err(json_rpc_error(
            -32602,
            "Unsupported URL: failed to extract docs token/type",
            Some(json!({ "url": url })),
        ));
    };
    let default_origin = extract_url_origin(&url);

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| match kind.as_str() {
            "wiki" => {
                let (docs_type, docs_token) =
                    feishu_wiki_resolve_obj(&client, &base_url, token, &tok)?;
                match docs_type.as_str() {
                    "doc" | "docx" => feishu_fetch_raw_content(
                        &client,
                        &base_url,
                        token,
                        &docs_type,
                        &docs_token,
                        lang,
                        default_origin.as_deref(),
                    ),
                    "sheet" => feishu_fetch_sheet_preview_text(
                        &client,
                        &base_url,
                        token,
                        &docs_token,
                        max_rows,
                        max_cols,
                        max_sheets,
                    ),
                    other => Err(json_rpc_error(
                        -32602,
                        "Unsupported wiki node object type (supported: doc/docx/sheet for now)",
                        Some(
                            json!({ "obj_type": other, "obj_token": docs_token, "node_token": tok }),
                        ),
                    )),
                }
            }
            "doc" | "docx" => feishu_fetch_raw_content(
                &client,
                &base_url,
                token,
                &kind,
                &tok,
                lang,
                default_origin.as_deref(),
            ),
            "sheet" => feishu_fetch_sheet_preview_text(
                &client, &base_url, token, &tok, max_rows, max_cols, max_sheets,
            ),
            other => Err(json_rpc_error(
                -32602,
                "Unsupported docs URL type (supported: wiki/doc/docx/sheets for now)",
                Some(json!({ "url": url, "parsed_type": other, "parsed_token": tok })),
            )),
        },
    )
}

pub(super) fn feishu_docs_export_text(args: &Value) -> Result<String, JsonRpcErr> {
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
    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let out_dir = args
        .get("out_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let out_dir = if out_dir.is_empty() {
        rust_tools::commonw::utils::expanduser("~/.config/rust_tools/feishu_docs_text")
            .as_ref()
            .to_string()
    } else {
        rust_tools::commonw::utils::expanduser(&out_dir)
            .as_ref()
            .to_string()
    };

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let content = with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| {
            feishu_fetch_raw_content(
                &client,
                &base_url,
                token,
                &docs_type,
                &docs_token,
                lang,
                None,
            )
        },
    )?;

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

pub(super) fn resolve_base_url_for_user_url(url: &str) -> String {
    let u = url.to_lowercase();
    if u.contains("larkoffice.com") || u.contains("larksuite.com") {
        "https://open.larksuite.com".to_string()
    } else {
        resolve_base_url()
    }
}

pub(super) fn parse_docs_url_kind_and_token(url: &str) -> Option<(String, String)> {
    let mut s = url.trim();
    if let Some(idx) = s.find("://") {
        s = &s[idx + 3..];
    }
    let s = s.split('?').next().unwrap_or(s);
    let s = s.split('#').next().unwrap_or(s);
    let path = if let Some(idx) = s.find('/') {
        &s[idx..]
    } else {
        s
    };
    let path = path.trim_start_matches('/');

    let segs = path
        .split('/')
        .filter(|p| !p.trim().is_empty())
        .collect::<Vec<_>>();
    if segs.len() < 2 {
        return None;
    }

    for i in 0..(segs.len().saturating_sub(1)) {
        let kind = segs[i].trim().to_lowercase();
        let token = segs[i + 1].trim();
        if token.is_empty() {
            continue;
        }
        let kind = match kind.as_str() {
            "wiki" => "wiki",
            "docx" => "docx",
            "doc" | "docs" => "doc",
            "sheets" | "sheet" => "sheet",
            _ => continue,
        };
        return Some((kind.to_string(), token.to_string()));
    }
    None
}

pub(super) fn feishu_wiki_resolve_obj(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    token: &str,
) -> Result<(String, String), JsonRpcErr> {
    let q_token = url_encode_component(token);
    let url = format!(
        "{}/open-apis/wiki/v2/spaces/get_node?token={}",
        base_url.trim_end_matches('/'),
        q_token
    );

    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call wiki get_node API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read wiki get_node response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "wiki get_node API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse wiki get_node JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "wiki get_node returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let node = data.get("node").cloned().unwrap_or_else(|| data.clone());

    let obj_type = node
        .get("obj_type")
        .and_then(|x| x.as_str())
        .or_else(|| node.get("objType").and_then(|x| x.as_str()))
        .unwrap_or("")
        .trim()
        .to_string();
    let obj_token = node
        .get("obj_token")
        .and_then(|x| x.as_str())
        .or_else(|| node.get("objToken").and_then(|x| x.as_str()))
        .unwrap_or("")
        .trim()
        .to_string();

    if obj_type.is_empty() || obj_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "wiki get_node response missing obj_type/obj_token",
            Some(json!({ "parsed": { "obj_type": obj_type, "obj_token": obj_token }, "body": v })),
        ));
    }

    Ok((obj_type, obj_token))
}

pub(super) fn feishu_fetch_sheet_preview_text(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
    max_rows: usize,
    max_cols: usize,
    max_sheets: usize,
) -> Result<String, JsonRpcErr> {
    let sheets = feishu_query_sheet_list(client, base_url, user_access_token, spreadsheet_token)?;
    if sheets.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "No sheets found in spreadsheet",
            Some(json!({ "spreadsheet_token": spreadsheet_token })),
        ));
    }

    let mut out = String::new();
    for (idx, (sheet_id, title)) in sheets.into_iter().take(max_sheets).enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!("[sheet] {} ({})\n", title.trim(), sheet_id.trim()));

        let col = col_letters(max_cols.saturating_sub(1));
        let range = format!("{}!A1:{}{}", sheet_id.trim(), col, max_rows);
        let values = feishu_sheets_read_range_values(
            client,
            base_url,
            user_access_token,
            spreadsheet_token,
            &range,
        )?;
        out.push_str(&format_values_as_tsv(&values));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }

    Ok(out.trim_end().to_string())
}

pub(super) fn feishu_query_sheet_list(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
) -> Result<Vec<(String, String)>, JsonRpcErr> {
    let url = format!(
        "{}/open-apis/sheets/v3/spreadsheets/{}/sheets/query",
        base_url.trim_end_matches('/'),
        spreadsheet_token.trim()
    );
    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to query sheets list",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read sheets list response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "Sheets list API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse sheets list JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "Sheets list returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let arr = data
        .get("sheets")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for item in arr {
        let sheet_id = item
            .get("sheet_id")
            .and_then(|x| x.as_str())
            .or_else(|| item.get("sheetId").and_then(|x| x.as_str()))
            .unwrap_or("")
            .trim()
            .to_string();
        let title = item
            .get("title")
            .and_then(|x| x.as_str())
            .or_else(|| item.get("name").and_then(|x| x.as_str()))
            .unwrap_or("")
            .trim()
            .to_string();
        if !sheet_id.is_empty() {
            out.push((sheet_id, title));
        }
    }
    Ok(out)
}

pub(super) fn feishu_sheets_read_range_values(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
    range: &str,
) -> Result<Value, JsonRpcErr> {
    let encoded_range = url_encode_component(range);
    let url = format!(
        "{}/open-apis/sheets/v2/spreadsheets/{}/values/{}?valueRenderOption=ToString&dateTimeRenderOption=FormattedString",
        base_url.trim_end_matches('/'),
        spreadsheet_token.trim(),
        encoded_range
    );

    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to read spreadsheet range",
                Some(json!({ "error": e.to_string(), "range": range })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read spreadsheet range response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "Spreadsheet range API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse spreadsheet range JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "Spreadsheet range returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let values = v
        .get("data")
        .and_then(|d| d.get("valueRange"))
        .and_then(|vr| vr.get("values"))
        .cloned()
        .unwrap_or_else(|| json!([]));
    Ok(values)
}

pub(super) fn format_values_as_tsv(values: &Value) -> String {
    let rows = values.as_array().cloned().unwrap_or_default();
    let mut out = String::new();
    for row in rows {
        let cells = row.as_array().cloned().unwrap_or_default();
        let mut first = true;
        for cell in cells {
            if !first {
                out.push('\t');
            }
            first = false;
            out.push_str(&cell_to_text(&cell));
        }
        out.push('\n');
    }
    out
}

pub(super) fn cell_to_text(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(n) = v.as_i64() {
        return n.to_string();
    }
    if let Some(n) = v.as_f64() {
        return n.to_string();
    }
    if let Some(obj) = v.as_object() {
        if let Some(t) = obj.get("text").and_then(|x| x.as_str()) {
            return t.to_string();
        }
        if obj.get("fileToken").and_then(|x| x.as_str()).is_some()
            || obj
                .get("float_image_token")
                .and_then(|x| x.as_str())
                .is_some()
            || obj
                .get("floatImageToken")
                .and_then(|x| x.as_str())
                .is_some()
        {
            return "[image]".to_string();
        }
        if let Some(t) = obj.get("type").and_then(|x| x.as_str())
            && t == "embed-image"
        {
            return "[image]".to_string();
        }
        if let Some(s) = obj.get("value").and_then(|x| x.as_str()) {
            return s.to_string();
        }
    }
    if v.is_null() {
        return String::new();
    }
    v.to_string()
}

pub(super) fn col_letters(mut idx: usize) -> String {
    let mut out = Vec::new();
    loop {
        let rem = idx % 26;
        out.push((b'A' + rem as u8) as char);
        if idx < 26 {
            break;
        }
        idx = idx / 26 - 1;
    }
    out.iter().rev().collect()
}

pub(super) fn feishu_fetch_raw_content(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    docs_type: &str,
    docs_token: &str,
    lang: i64,
    default_origin: Option<&str>,
) -> Result<String, JsonRpcErr> {
    let content = feishu_fetch_raw_content_api(
        client,
        base_url,
        user_access_token,
        docs_type,
        docs_token,
        lang,
    )?;

    if docs_type != "docx" {
        return Ok(content);
    }

    let blocks_text = feishu_fetch_docx_blocks_text(
        client,
        base_url,
        user_access_token,
        docs_token,
        default_origin,
    )?;
    if should_prefer_docx_blocks_render(&content, &blocks_text) {
        Ok(blocks_text)
    } else {
        Ok(content)
    }
}

pub(super) fn feishu_fetch_raw_content_api(
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
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header(
            "Content-Type",
            if is_docx {
                "application/json; charset=utf-8"
            } else {
                "text/plain"
            },
        )
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to fetch raw_content",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read raw_content response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "raw_content API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "raw_content response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "raw_content API returned error code",
            Some(v),
        ));
    }
    let content = v
        .get("data")
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Ok(content)
}

pub(super) fn feishu_fetch_docx_blocks_text(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    document_id: &str,
    default_origin: Option<&str>,
) -> Result<String, JsonRpcErr> {
    let mut page_token: Option<String> = None;
    let mut items = Vec::new();

    loop {
        let mut url = format!(
            "{}/open-apis/docx/v1/documents/{}/blocks?page_size=500",
            base_url.trim_end_matches('/'),
            document_id.trim()
        );
        if let Some(token) = page_token.as_deref()
            && !token.trim().is_empty()
        {
            url.push_str("&page_token=");
            url.push_str(&url_encode_component(token.trim()));
        }

        let resp = client
            .get(&url)
            .header(
                "Authorization",
                format!("Bearer {}", user_access_token.trim()),
            )
            .header("Content-Type", "application/json; charset=utf-8")
            .send()
            .map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to fetch docx blocks",
                    Some(json!({ "error": e.to_string(), "document_id": document_id })),
                )
            })?;
        let status = resp.status();
        let text = resp.text().map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to read docx blocks response",
                Some(json!({ "error": e.to_string(), "document_id": document_id })),
            )
        })?;
        if !status.is_success() {
            return Err(json_rpc_error(
                -32000,
                "docx blocks API returned non-success HTTP status",
                Some(
                    json!({ "status": status.as_u16(), "body": text, "document_id": document_id }),
                ),
            ));
        }

        let v: Value = serde_json::from_str(&text).map_err(|e| {
            json_rpc_error(
                -32000,
                "docx blocks response is not valid JSON",
                Some(json!({ "error": e.to_string(), "body": text, "document_id": document_id })),
            )
        })?;
        let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(
                -32000,
                "docx blocks API returned error code",
                Some(v),
            ));
        }

        let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
        if let Some(arr) = data.get("items").and_then(|v| v.as_array()) {
            items.extend(arr.iter().cloned());
        }

        let has_more = data
            .get("has_more")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let next_token = data
            .get("page_token")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if !has_more || next_token.is_none() {
            break;
        }
        page_token = next_token;
    }

    Ok(render_docx_blocks_as_text(&items, default_origin))
}
