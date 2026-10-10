use super::*;

pub(super) fn feishu_doc_create_from_markdown(args: &Value) -> Result<String, JsonRpcErr> {
    let title = args
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let markdown_content = args
        .get("markdown_content")
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
    if markdown_content.trim().is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: markdown_content is required",
            None,
        ));
    }

    let base_url = resolve_base_url();
    let client = feishu_docs_http_client()?;

    let document_id = with_user_access_token(
        client,
        &base_url,
        "Missing user_access_token. Create document requires OAuth once.",
        |token| {
            let mut create_body = json!({
                "title": title,
                "folder_token": folder_token
            });
            if folder_token.is_empty() {
                create_body.as_object_mut().unwrap().remove("folder_token");
            }

            let url = format!(
                "{}/open-apis/docx/v1/documents",
                base_url.trim_end_matches('/')
            );
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token.trim()))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&create_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to create document",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

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
                    "Failed to create document",
                    Some(err_json),
                ));
            }

            let doc_id = json
                .get("data")
                .and_then(|d| {
                    d.get("document_id").or_else(|| {
                        d.get("document")
                            .and_then(|document| document.get("document_id"))
                    })
                })
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    let err_json = json.clone();
                    json_rpc_error(-32000, "No document_id in response", Some(err_json))
                })?;

            Ok(doc_id.to_string())
        },
    )?;

    let block_ops = convert_markdown_to_docx_blocks(&markdown_content);
    let doc_url = format!(
        "{}/docx/{}",
        resolve_docs_web_base_url(&base_url),
        document_id
    );
    if block_ops.is_empty() {
        return Ok(empty_doc_result(&document_id, &doc_url));
    }

    // 飞书 children/descendant 接口对单次写入数量有限制，每批最多 50 个直接子块。
    const MAX_CHILDREN_PER_BATCH: usize = 50;

    let mut index: i64 = 0;
    let mut simple_buf: Vec<Value> = Vec::new();
    let mut desc_children_id: Vec<String> = Vec::new();
    let mut desc_descendants: Vec<Value> = Vec::new();
    let mut blocks_inserted: i64 = 0;

    for op in &block_ops {
        match op {
            BlockOp::Simple(block) => {
                if !desc_children_id.is_empty() {
                    flush_descendant_batch(
                        client,
                        &base_url,
                        &document_id,
                        &mut desc_children_id,
                        &mut desc_descendants,
                        &mut index,
                        &mut blocks_inserted,
                    )?;
                }
                simple_buf.push(block.clone());
                if simple_buf.len() >= MAX_CHILDREN_PER_BATCH {
                    flush_simple_batch(
                        client,
                        &base_url,
                        &document_id,
                        &mut simple_buf,
                        &mut index,
                        &mut blocks_inserted,
                    )?;
                }
            }
            BlockOp::Descendant {
                children_id,
                descendants,
            } => {
                if !simple_buf.is_empty() {
                    flush_simple_batch(
                        client,
                        &base_url,
                        &document_id,
                        &mut simple_buf,
                        &mut index,
                        &mut blocks_inserted,
                    )?;
                }
                if !desc_children_id.is_empty()
                    && desc_children_id.len() + children_id.len() > MAX_CHILDREN_PER_BATCH
                {
                    flush_descendant_batch(
                        client,
                        &base_url,
                        &document_id,
                        &mut desc_children_id,
                        &mut desc_descendants,
                        &mut index,
                        &mut blocks_inserted,
                    )?;
                }
                desc_children_id.extend(children_id.iter().cloned());
                desc_descendants.extend(descendants.iter().cloned());
                if desc_children_id.len() >= MAX_CHILDREN_PER_BATCH {
                    flush_descendant_batch(
                        client,
                        &base_url,
                        &document_id,
                        &mut desc_children_id,
                        &mut desc_descendants,
                        &mut index,
                        &mut blocks_inserted,
                    )?;
                }
            }
        }
    }

    if !simple_buf.is_empty() {
        flush_simple_batch(
            client,
            &base_url,
            &document_id,
            &mut simple_buf,
            &mut index,
            &mut blocks_inserted,
        )?;
    }
    if !desc_children_id.is_empty() {
        flush_descendant_batch(
            client,
            &base_url,
            &document_id,
            &mut desc_children_id,
            &mut desc_descendants,
            &mut index,
            &mut blocks_inserted,
        )?;
    }

    Ok(success_doc_result(&document_id, &doc_url, blocks_inserted))
}

/// 进程级共享 HTTP client，针对文档批量写入场景使用 60s 超时，避免大批量
/// payload 在 30s 限制下因网络抖动失败。
pub(super) fn feishu_docs_http_client() -> Result<&'static Client, JsonRpcErr> {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    if let Some(c) = CLIENT.get() {
        return Ok(c);
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    Ok(CLIENT.get_or_init(|| client))
}

pub(super) fn flush_simple_batch(
    client: &Client,
    base_url: &str,
    document_id: &str,
    buf: &mut Vec<Value>,
    index: &mut i64,
    blocks_inserted: &mut i64,
) -> Result<(), JsonRpcErr> {
    if buf.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(buf);
    let count = batch.len() as i64;
    let start_index = *index;
    with_user_access_token(
        client,
        base_url,
        "Missing user_access_token. Update document requires OAuth.",
        |token| {
            create_children_batch(client, base_url, token, document_id, &batch, start_index)?;
            Ok(())
        },
    )?;
    *index += count;
    *blocks_inserted += count;
    Ok(())
}

pub(super) fn flush_descendant_batch(
    client: &Client,
    base_url: &str,
    document_id: &str,
    children_id: &mut Vec<String>,
    descendants: &mut Vec<Value>,
    index: &mut i64,
    blocks_inserted: &mut i64,
) -> Result<(), JsonRpcErr> {
    if children_id.is_empty() {
        return Ok(());
    }
    let cids = std::mem::take(children_id);
    let descs = std::mem::take(descendants);
    let count = cids.len() as i64;
    let start_index = *index;
    with_user_access_token(
        client,
        base_url,
        "Missing user_access_token. Update document requires OAuth.",
        |token| {
            create_descendant_batch(
                client,
                base_url,
                token,
                document_id,
                &cids,
                &descs,
                start_index,
            )?;
            Ok(())
        },
    )?;
    *index += count;
    *blocks_inserted += count;
    Ok(())
}

pub(super) fn empty_doc_result(document_id: &str, doc_url: &str) -> String {
    json!({
        "document_id": document_id,
        "url": doc_url,
        "blocks_inserted": 0,
        "empty": true
    })
    .to_string()
}

pub(super) fn success_doc_result(document_id: &str, doc_url: &str, blocks_inserted: i64) -> String {
    json!({
        "document_id": document_id,
        "url": doc_url,
        "blocks_inserted": blocks_inserted
    })
    .to_string()
}

pub(super) fn create_children_batch(
    client: &Client,
    base_url: &str,
    token: &str,
    document_id: &str,
    children: &[Value],
    index: i64,
) -> Result<(), JsonRpcErr> {
    let url = format!(
        "{}/open-apis/docx/v1/documents/{}/blocks/{}/children",
        base_url.trim_end_matches('/'),
        document_id,
        document_id
    );
    let body = json!({
        "children": children,
        "index": index
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create document blocks",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

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
            "Failed to create document blocks",
            Some(json),
        ));
    }

    Ok(())
}

/// 飞书 descendant API 要求每个 descendant block 必须包含 parent_id 字段。
/// block 构造阶段不知道 document_id，因此在此处统一注入 parent_id：
/// - children_id 中的根级 block，parent_id = document_id
/// - 其余 block，通过遍历各 block 的 children 数组反查父 block_id
pub(super) fn inject_descendant_parent_ids(
    descendants: &[Value],
    children_id: &[String],
    document_id: &str,
) -> Vec<Value> {
    let mut parent_map: FastMap<String, String> = FastMap::default();
    for desc in descendants {
        if let Some(block_id) = desc.get("block_id").and_then(|v| v.as_str()) {
            if let Some(children) = desc.get("children").and_then(|v| v.as_array()) {
                for child in children {
                    if let Some(child_id) = child.as_str() {
                        parent_map.insert(child_id.to_string(), block_id.to_string());
                    }
                }
            }
        }
    }
    let children_set: FastSet<String> = children_id.iter().cloned().collect();
    descendants
        .iter()
        .map(|desc| {
            let mut d = desc.clone();
            if let Some(block_id) = d.get("block_id").and_then(|v| v.as_str()) {
                let parent = if children_set.contains(block_id) {
                    document_id.to_string()
                } else {
                    parent_map.get(block_id).cloned().unwrap_or_default()
                };
                d["parent_id"] = json!(parent);
            }
            d
        })
        .collect()
}

pub(super) fn create_descendant_batch(
    client: &Client,
    base_url: &str,
    token: &str,
    document_id: &str,
    children_id: &[String],
    descendants: &[Value],
    index: i64,
) -> Result<(), JsonRpcErr> {
    let enriched_descendants = inject_descendant_parent_ids(descendants, children_id, document_id);
    let url = format!(
        "{}/open-apis/docx/v1/documents/{}/blocks/{}/descendant",
        base_url.trim_end_matches('/'),
        document_id,
        document_id
    );
    let body = json!({
        "index": index,
        "children_id": children_id,
        "descendants": enriched_descendants
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create table descendant blocks",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

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
            "Failed to create table descendant blocks",
            Some(json),
        ));
    }

    Ok(())
}

pub(super) fn resolve_docs_web_base_url(base_url: &str) -> String {
    let host = base_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .split('/')
        .next()
        .unwrap_or("open.feishu.cn")
        .to_ascii_lowercase();

    match host.as_str() {
        "open.feishu.cn" => "https://www.feishu.cn".to_string(),
        "open.larksuite.com" | "open.larkoffice.com" => "https://www.larksuite.com".to_string(),
        _ if host.ends_with(".feishu.cn")
            || host.ends_with(".larksuite.com")
            || host.ends_with(".larkoffice.com") =>
        {
            format!("https://{}", host)
        }
        _ => "https://www.feishu.cn".to_string(),
    }
}
