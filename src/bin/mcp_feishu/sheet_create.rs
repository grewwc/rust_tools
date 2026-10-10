use super::*;

pub(super) fn feishu_sheet_create_from_csv(args: &Value) -> Result<String, JsonRpcErr> {
    let title = args
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let csv_content = args
        .get("csv_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let folder_token = args
        .get("folder_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if title.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: title is required",
            Some(json!({ "title": title })),
        ));
    }
    if csv_content.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: csv_content is required",
            None,
        ));
    }

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    // Step 1: Create spreadsheet
    let spreadsheet_token = with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Create spreadsheet requires OAuth once.",
        |token| {
            let mut create_body = json!({
                "title": title,
                "folder_token": folder_token
            });
            if folder_token.is_empty() {
                create_body.as_object_mut().unwrap().remove("folder_token");
            }

            let url = format!("{}/open-apis/sheets/v3/spreadsheets", base_url);
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token.trim()))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&create_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to create spreadsheet",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

            let _status = resp.status();
            let text = resp.text().map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to read response body",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let json: Value = serde_json::from_str(&text).map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to parse response JSON",
                    Some(json!({ "error": e.to_string(), "body": text })),
                )
            })?;

            let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            if code != 0 {
                let err_json = json.clone();
                return Err(json_rpc_error(
                    -32000,
                    "Failed to create spreadsheet",
                    Some(err_json),
                ));
            }

            let token = json
                .get("data")
                .and_then(|d| d.get("spreadsheet_token"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    let err_json = json.clone();
                    json_rpc_error(-32000, "No spreadsheet_token in response", Some(err_json))
                })?;

            Ok(token.to_string())
        },
    )?;

    // Step 2: Parse CSV and prepare values
    let mut values: Vec<Vec<String>> = Vec::new();
    for line in csv_content.lines() {
        let row: Vec<String> = parse_csv_line(line);
        values.push(row);
    }

    // Step 3: Batch update spreadsheet with values
    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Update spreadsheet requires OAuth.",
        |token| {
            let range = "Sheet1!A1";
            let update_body = json!({
                "value_range": {
                    "range": range,
                    "values": values
                }
            });

            let url = format!(
                "{}/open-apis/sheets/v2/spreadsheets/{}/values/batchUpdate",
                base_url, spreadsheet_token
            );
            let resp = client
                .put(&url)
                .header("Authorization", format!("Bearer {}", token.trim()))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&update_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to update spreadsheet values",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

            let _status = resp.status();
            let text = resp.text().map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to read response body",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let json: Value = serde_json::from_str(&text).map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to parse response JSON",
                    Some(json!({ "error": e.to_string(), "body": text })),
                )
            })?;

            let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            if code != 0 {
                return Err(json_rpc_error(
                    -32000,
                    "Failed to update spreadsheet values",
                    Some(json),
                ));
            }

            Ok(())
        },
    )?;

    // Generate URL
    let spreadsheet_url = format!(
        "{}/sheets/{}",
        resolve_docs_web_base_url(&base_url),
        spreadsheet_token
    );

    Ok(format!(
        "Created spreadsheet: {}\nToken: {}",
        spreadsheet_url, spreadsheet_token
    ))
}

pub(super) fn parse_csv_line(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current_cell = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current_cell.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current_cell.push(c);
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => {
                    cells.push(current_cell.trim().to_string());
                    current_cell = String::new();
                }
                _ => current_cell.push(c),
            }
        }
    }
    cells.push(current_cell.trim().to_string());
    cells
}

pub(super) fn make_text_element(content: &str, bold: bool, italic: bool, inline_code: bool) -> Value {
    let mut style = json!({});
    if bold {
        style["bold"] = json!(true);
    }
    if italic {
        style["italic"] = json!(true);
    }
    if inline_code {
        style["inline_code"] = json!(true);
    }
    json!({
        "text_run": {
            "content": content,
            "text_element_style": style
        }
    })
}

pub(super) fn parse_inline_elements(text: &str) -> Vec<Value> {
    let mut elements = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    let mut bold_depth = 0usize;
    let mut italic_depth = 0usize;
    let mut in_code = false;

    while let Some(c) = chars.next() {
        if in_code {
            if c == '`' {
                in_code = false;
                if !current.is_empty() {
                    elements.push(make_text_element(&current, false, false, true));
                    current.clear();
                }
            } else {
                current.push(c);
            }
            continue;
        }

        match c {
            '`' => {
                if !current.is_empty() {
                    elements.push(make_text_element(
                        &current,
                        bold_depth > 0,
                        italic_depth > 0,
                        false,
                    ));
                    current.clear();
                }
                in_code = true;
            }
            '*' => {
                let peeked = chars.peek();
                if peeked == Some(&'*') {
                    chars.next();
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if bold_depth > 0 {
                        bold_depth -= 1;
                    } else {
                        bold_depth += 1;
                    }
                } else {
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if italic_depth > 0 {
                        italic_depth -= 1;
                    } else {
                        italic_depth += 1;
                    }
                }
            }
            '_' => {
                let next_is_underscore = chars.peek() == Some(&'_');
                if next_is_underscore {
                    chars.next();
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if bold_depth > 0 {
                        bold_depth -= 1;
                    } else {
                        bold_depth += 1;
                    }
                } else {
                    current.push(c);
                }
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        elements.push(make_text_element(
            &current,
            bold_depth > 0,
            italic_depth > 0,
            in_code,
        ));
    }

    if elements.is_empty() {
        elements.push(make_text_element("", false, false, false));
    }

    elements
}
