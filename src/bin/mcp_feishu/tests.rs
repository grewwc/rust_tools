    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn invalid_token_err() -> JsonRpcErr {
        json_rpc_error(
            -32000,
            "raw_content API returned non-success HTTP status",
            Some(json!({
                "status": 400,
                "body": "{\"code\":99991668,\"msg\":\"Invalid access token for authorization. Please make a request with token attached.\"}"
            })),
        )
    }

    fn start_mock_http_server<F>(handler: F) -> String
    where
        F: Fn(String) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    continue;
                }
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                let body = handler(request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{}", addr)
    }

    #[test]
    fn render_docx_blocks_keeps_diagram_placeholder_near_heading() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["h2", "diagram"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "h2",
                "block_type": 4,
                "parent_id": "root",
                "heading2": {
                    "elements": [
                        { "text_run": { "content": "6.1 文字绘图", "text_element_style": {} } }
                    ]
                }
            }),
            json!({
                "block_id": "diagram",
                "block_type": 21,
                "parent_id": "root",
                "diagram": { "diagram_type": 1 }
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("## 6.1 文字绘图"));
        assert!(rendered.contains("[流程图]"));
        assert!(rendered.find("## 6.1 文字绘图") < rendered.find("[流程图]"));
    }

    #[test]
    fn prefer_blocks_render_when_special_blocks_would_be_lost() {
        let raw = "## 6.1 文字绘图";
        let rendered = "## 6.1 文字绘图\n[流程图]";
        assert!(should_prefer_docx_blocks_render(raw, rendered));
        assert!(!should_prefer_docx_blocks_render(raw, raw));
    }

    #[test]
    fn prefer_blocks_render_when_raw_content_flattens_code_block() {
        let raw = "Python 代码示例\n4 行 Python 代码\ndef greet(name): return f\"Hello, {name}!\"\nresult = greet(\"World\") print(result)";
        let rendered = "```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nresult = greet(\"World\")\nprint(result)\n```";
        assert!(should_prefer_docx_blocks_render(raw, rendered));
    }

    #[test]
    fn render_docx_blocks_keeps_board_placeholders_after_heading() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["h3", "board1", "board2"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "h3",
                "block_type": 5,
                "parent_id": "root",
                "heading3": {
                    "elements": [
                        { "text_run": { "content": "6.1. 系统架构图", "text_element_style": {} } }
                    ]
                }
            }),
            json!({
                "block_id": "board1",
                "block_type": 43,
                "parent_id": "root",
                "board": { "token": "board-token-1" }
            }),
            json!({
                "block_id": "board2",
                "block_type": 43,
                "parent_id": "root",
                "board": { "token": "board-token-2" }
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("### 6.1. 系统架构图"));
        assert!(rendered.contains("[文字绘图: board-token-1]"));
        assert!(rendered.contains("[文字绘图: board-token-2]"));
        assert!(rendered.find("### 6.1. 系统架构图") < rendered.find("[文字绘图: board-token-1]"));
    }

    #[test]
    fn render_docx_blocks_preserves_quote_and_nested_list_formatting() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["quote"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "quote",
                "block_type": 15,
                "parent_id": "root",
                "children": ["todo_nested"],
                "quote": { "elements": [{ "text_run": { "content": "注意事项", "text_element_style": {} } }] }
            }),
            json!({
                "block_id": "todo_nested",
                "block_type": 17,
                "parent_id": "quote",
                "children": [],
                "todo": {
                    "elements": [{ "text_run": { "content": "先检查输入", "text_element_style": {} } }],
                    "style": { "done": false }
                }
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("> 注意事项"));
        assert!(rendered.contains("> - [ ] 先检查输入"));
    }

    #[test]
    fn render_docx_blocks_uses_header_row_and_cell_line_breaks() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["table1"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "table1",
                "block_type": 31,
                "parent_id": "root",
                "table": {
                    "cells": ["cell1", "cell2"],
                    "property": { "row_size": 1, "column_size": 2, "header_row": false }
                }
            }),
            json!({
                "block_id": "cell1",
                "block_type": 32,
                "parent_id": "table1",
                "table_cell": {},
                "children": ["cell1_text", "cell1_text2"]
            }),
            json!({
                "block_id": "cell1_text",
                "block_type": 2,
                "parent_id": "cell1",
                "text": { "elements": [{ "text_run": { "content": "第一行", "text_element_style": {} } }] },
                "children": []
            }),
            json!({
                "block_id": "cell1_text2",
                "block_type": 2,
                "parent_id": "cell1",
                "text": { "elements": [{ "text_run": { "content": "第二行", "text_element_style": {} } }] },
                "children": []
            }),
            json!({
                "block_id": "cell2",
                "block_type": 32,
                "parent_id": "table1",
                "table_cell": {},
                "children": ["cell2_text"]
            }),
            json!({
                "block_id": "cell2_text",
                "block_type": 2,
                "parent_id": "cell2",
                "text": { "elements": [{ "text_run": { "content": "值", "text_element_style": {} } }] },
                "children": []
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("| 第一行<br>第二行 | 值 |"));
        assert!(!rendered.contains("| --- | --- |"));
    }

    #[test]
    fn render_docx_blocks_uses_ordered_sequence_when_available() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["ordered1"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "ordered1",
                "block_type": 13,
                "parent_id": "root",
                "ordered": {
                    "elements": [{ "text_run": { "content": "第三项", "text_element_style": {} } }],
                    "style": { "sequence": "3" }
                },
                "children": []
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("3. 第三项"));
    }

    #[test]
    fn detects_invalid_user_access_token_from_raw_body() {
        assert!(is_invalid_user_access_token_error(&invalid_token_err()));
    }

    #[test]
    fn with_user_access_token_falls_back_to_token_store_when_env_token_is_stale() {
        let _guard = env_lock().lock().unwrap();
        let token_store_path = format!(
            "/tmp/mcp_feishu_token_test_{}_{}.json",
            std::process::id(),
            epoch_ms_now()
        );
        let old_env_token_store_path = std::env::var("FEISHU_TOKEN_STORE_PATH").ok();
        let old_env_user_token = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        let _ = fs::remove_file(&token_store_path);

        unsafe {
            std::env::set_var("FEISHU_TOKEN_STORE_PATH", &token_store_path);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-stale-token");
        }

        save_token_store(&TokenStore {
            user_access_token: Some("u-fresh-token".to_string()),
            user_access_token_expires_at_epoch_ms: Some(epoch_ms_now() + 3_600_000),
            refresh_token: None,
            refresh_token_expires_in: None,
            updated_at_epoch_ms: Some(epoch_ms_now()),
        })
        .unwrap();

        let client = Client::builder().build().unwrap();
        let mut seen = Vec::new();
        let result = with_user_access_token(
            &client,
            "https://open.feishu.cn",
            "Missing user_access_token. Fetch requires OAuth once.",
            |token| {
                seen.push(token.to_string());
                if token == "u-stale-token" {
                    Err(invalid_token_err())
                } else {
                    Ok(token.to_string())
                }
            },
        )
        .unwrap();

        assert_eq!(result, "u-fresh-token");
        assert_eq!(seen, vec!["u-stale-token", "u-fresh-token"]);

        let _ = fs::remove_file(&token_store_path);
        unsafe {
            if let Some(v) = old_env_token_store_path {
                std::env::set_var("FEISHU_TOKEN_STORE_PATH", v);
            } else {
                std::env::remove_var("FEISHU_TOKEN_STORE_PATH");
            }
            if let Some(v) = old_env_user_token {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn docs_web_base_url_maps_open_feishu_to_www_feishu() {
        assert_eq!(
            resolve_docs_web_base_url("https://open.feishu.cn"),
            "https://www.feishu.cn"
        );
    }

    #[test]
    fn docs_web_base_url_maps_open_larksuite_to_www_larksuite() {
        assert_eq!(
            resolve_docs_web_base_url("https://open.larksuite.com"),
            "https://www.larksuite.com"
        );
    }

    #[test]
    fn prefer_blocks_render_for_markdown_table_and_todo() {
        assert!(should_prefer_docx_blocks_render(
            "任务列表 待办事项",
            "- [ ] 第一项\n- [x] 第二项"
        ));
        assert!(should_prefer_docx_blocks_render(
            "姓名 年龄 Alice 18",
            "| 姓名 | 年龄 |\n| --- | --- |\n| Alice | 18 |"
        ));
    }

    #[test]
    fn parse_markdown_ast_preserves_todo_items_state() {
        let ast = parse_markdown_ast("- [ ] first\n- [x] second");
        match &ast[0] {
            MdNode::TodoList { items } => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0].done, Some(false));
                assert_eq!(items[1].done, Some(true));
            }
            _ => panic!("unexpected node kind"),
        }
    }

    #[test]
    fn build_code_block_style_maps_common_languages_and_mermaid() {
        assert_eq!(
            build_code_block_style(Some("python"))["language"],
            serde_json::json!(49)
        );
        assert_eq!(
            build_code_block_style(Some("mermaid"))["language"],
            serde_json::json!(39)
        );
    }

    #[test]
    fn render_text_elements_preserves_links() {
        // 测试普通文本（无链接）
        let elements_no_link = json!([
            { "text_run": { "content": "普通文本" } }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_no_link), None),
            "普通文本"
        );

        // 测试带链接的文本
        let elements_with_link = json!([
            {
                "text_run": {
                    "content": "点击这里",
                    "text_style": {
                        "link": {
                            "url": "https://example.com"
                        }
                    }
                }
            }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_with_link), None),
            "[点击这里](https://example.com)"
        );

        // 测试混合文本（有链接和无链接）
        let elements_mixed = json!([
            { "text_run": { "content": "前面文字" } },
            {
                "text_run": {
                    "content": "链接文本",
                    "text_style": {
                        "link": {
                            "url": "https://test.com"
                        }
                    }
                }
            },
            { "text_run": { "content": "后面文字" } }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_mixed), None),
            "前面文字[链接文本](https://test.com)后面文字"
        );

        // 测试只有 URL 没有文本内容的情况
        let elements_url_only = json!([
            {
                "text_run": {
                    "content": "",
                    "text_style": {
                        "link": {
                            "url": "https://bare-url.com"
                        }
                    }
                }
            }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_url_only), None),
            "https://bare-url.com"
        );
    }

    fn with_temp_token_store<F: FnOnce()>(body: F) {
        let path = format!(
            "/tmp/mcp_feishu_token_test_{}_{}.json",
            std::process::id(),
            epoch_ms_now()
        );
        let old = std::env::var("FEISHU_TOKEN_STORE_PATH").ok();
        let _ = fs::remove_file(&path);
        unsafe {
            std::env::set_var("FEISHU_TOKEN_STORE_PATH", &path);
        }
        body();
        let _ = fs::remove_file(&path);
        unsafe {
            match old {
                Some(v) => std::env::set_var("FEISHU_TOKEN_STORE_PATH", v),
                None => std::env::remove_var("FEISHU_TOKEN_STORE_PATH"),
            }
        }
    }

    #[test]
    fn save_token_store_writes_atomically_with_owner_only_permissions() {
        let _guard = env_lock().lock().unwrap();
        with_temp_token_store(|| {
            save_token_store(&TokenStore {
                user_access_token: Some("u-secret".to_string()),
                user_access_token_expires_at_epoch_ms: Some(epoch_ms_now() + 3_600_000),
                refresh_token: Some("r-secret".to_string()),
                refresh_token_expires_in: Some(1_000_000),
                updated_at_epoch_ms: Some(epoch_ms_now()),
            })
            .unwrap();

            let path = token_store_path();
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(
                mode, 0o600,
                "token store must be owner-only readable/writable"
            );

            // 重写覆盖不应损坏文件，且仍保持 0o600。
            save_token_store(&TokenStore {
                user_access_token: Some("u-secret-2".to_string()),
                user_access_token_expires_at_epoch_ms: Some(epoch_ms_now() + 3_600_000),
                refresh_token: Some("r-secret-2".to_string()),
                refresh_token_expires_in: Some(1_000_000),
                updated_at_epoch_ms: Some(epoch_ms_now()),
            })
            .unwrap();
            let mode2 = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode2, 0o600);
            let reloaded = load_token_store().unwrap();
            assert_eq!(reloaded.user_access_token.as_deref(), Some("u-secret-2"));
            assert_eq!(reloaded.refresh_token.as_deref(), Some("r-secret-2"));
        });
    }

    #[test]
    fn refresh_response_without_rotation_preserves_existing_refresh_token() {
        let _guard = env_lock().lock().unwrap();
        let old_client_id = std::env::var("FEISHU_CLIENT_ID").ok();
        let old_client_secret = std::env::var("FEISHU_CLIENT_SECRET").ok();
        let old_base = std::env::var("FEISHU_BASE_URL").ok();
        // 模拟飞书刷新接口：返回新的 access_token 但 refresh_token 为空（未轮换）。
        let base = start_mock_http_server(|_req| {
            json!({
                "code": 0,
                "access_token": "u-new-access",
                "refresh_token": "",
                "expires_in": 7200,
                "refresh_token_expires_in": 0
            })
            .to_string()
        });
        with_temp_token_store(|| {
            unsafe {
                std::env::set_var("FEISHU_CLIENT_ID", "cli_test");
                std::env::set_var("FEISHU_CLIENT_SECRET", "secret_test");
                std::env::set_var("FEISHU_BASE_URL", &base);
                std::env::remove_var("FEISHU_REFRESH_TOKEN");
            }
            save_token_store(&TokenStore {
                user_access_token: Some("u-old".to_string()),
                user_access_token_expires_at_epoch_ms: Some(epoch_ms_now() + 1_000),
                refresh_token: Some("r-existing".to_string()),
                refresh_token_expires_in: Some(1_000_000),
                updated_at_epoch_ms: Some(epoch_ms_now()),
            })
            .unwrap();

            let client = Client::builder().build().unwrap();
            let token = refresh_user_access_token_and_cache(&client, &base).unwrap();
            assert_eq!(token, "u-new-access");

            let reloaded = load_token_store().unwrap();
            assert_eq!(
                reloaded.refresh_token.as_deref(),
                Some("r-existing"),
                "un-rotated refresh must keep the existing refresh_token, not wipe it"
            );
            assert_eq!(reloaded.user_access_token.as_deref(), Some("u-new-access"));
        });

        unsafe {
            match old_client_id {
                Some(v) => std::env::set_var("FEISHU_CLIENT_ID", v),
                None => std::env::remove_var("FEISHU_CLIENT_ID"),
            }
            match old_client_secret {
                Some(v) => std::env::set_var("FEISHU_CLIENT_SECRET", v),
                None => std::env::remove_var("FEISHU_CLIENT_SECRET"),
            }
            match old_base {
                Some(v) => std::env::set_var("FEISHU_BASE_URL", v),
                None => std::env::remove_var("FEISHU_BASE_URL"),
            }
        }
    }

    #[test]
    fn oauth_authorize_url_uses_custom_scope_when_provided() {
        let _guard = env_lock().lock().unwrap();
        let old_client_id = std::env::var("FEISHU_CLIENT_ID").ok();
        unsafe {
            std::env::set_var("FEISHU_CLIENT_ID", "cli_scope_test");
        }
        let custom = feishu_oauth_authorize_url(&json!({
            "redirect_uri": "http://127.0.0.1:8711/callback",
            "scope": "docx:document:readonly wiki:node:read"
        }))
        .unwrap();
        assert!(custom.contains("scope=docx%3Adocument%3Areadonly%20wiki%3Anode%3Aread"));

        let default_scope = feishu_oauth_authorize_url(&json!({
            "redirect_uri": "http://127.0.0.1:8711/callback"
        }))
        .unwrap();
        assert!(default_scope.contains(&url_encode_component(FEISHU_SCOPE)));

        unsafe {
            match old_client_id {
                Some(v) => std::env::set_var("FEISHU_CLIENT_ID", v),
                None => std::env::remove_var("FEISHU_CLIENT_ID"),
            }
        }
    }

    #[test]
    fn inject_parent_ids_for_blockquote_descendants() {
        let descendants = vec![
            json!({ "block_id": "blk_1", "block_type": 15, "quote": {}, "children": ["blk_2"] }),
            json!({ "block_id": "blk_2", "block_type": 2, "text": { "elements": [] }, "children": [] }),
        ];
        let enriched = inject_descendant_parent_ids(&descendants, &["blk_1".to_string()], "doc123");
        assert_eq!(enriched[0]["parent_id"], "doc123");
        assert_eq!(enriched[1]["parent_id"], "blk_1");
    }

    #[test]
    fn inject_parent_ids_for_table_descendants() {
        let descendants = vec![
            json!({ "block_id": "blk_1", "block_type": 31, "table": { "property": { "row_size": 2, "column_size": 2, "header_row": true } }, "children": ["blk_2", "blk_4"] }),
            json!({ "block_id": "blk_2", "block_type": 32, "table_cell": {}, "children": ["blk_3"] }),
            json!({ "block_id": "blk_3", "block_type": 2, "text": { "elements": [] }, "children": [] }),
            json!({ "block_id": "blk_4", "block_type": 32, "table_cell": {}, "children": ["blk_5"] }),
            json!({ "block_id": "blk_5", "block_type": 2, "text": { "elements": [] }, "children": [] }),
        ];
        let enriched = inject_descendant_parent_ids(&descendants, &["blk_1".to_string()], "doc456");
        assert_eq!(enriched[0]["parent_id"], "doc456");
        assert_eq!(enriched[1]["parent_id"], "blk_1");
        assert_eq!(enriched[2]["parent_id"], "blk_2");
        assert_eq!(enriched[3]["parent_id"], "blk_1");
        assert_eq!(enriched[4]["parent_id"], "blk_4");
    }

    #[test]
    fn inject_parent_ids_for_multiple_root_blocks() {
        let descendants = vec![
            json!({ "block_id": "blk_1", "block_type": 15, "quote": {}, "children": ["blk_2"] }),
            json!({ "block_id": "blk_2", "block_type": 2, "text": { "elements": [] }, "children": [] }),
            json!({ "block_id": "blk_3", "block_type": 15, "quote": {}, "children": ["blk_4"] }),
            json!({ "block_id": "blk_4", "block_type": 2, "text": { "elements": [] }, "children": [] }),
        ];
        let enriched = inject_descendant_parent_ids(
            &descendants,
            &["blk_1".to_string(), "blk_3".to_string()],
            "doc789",
        );
        assert_eq!(enriched[0]["parent_id"], "doc789");
        assert_eq!(enriched[1]["parent_id"], "blk_1");
        assert_eq!(enriched[2]["parent_id"], "doc789");
        assert_eq!(enriched[3]["parent_id"], "blk_3");
    }

    #[test]
    fn convert_markdown_quote_uses_empty_quote_container() {
        let ops = convert_markdown_to_docx_blocks("> 测试引用");
        match &ops[0] {
            BlockOp::Descendant { descendants, .. } => {
                let quote_block = &descendants[0];
                assert_eq!(quote_block["block_type"], 15);
                assert!(quote_block["quote"].as_object().unwrap().is_empty());
            }
            BlockOp::Simple(block) => {
                assert_eq!(block["block_type"], 15);
                assert!(block["quote"].as_object().unwrap().is_empty());
            }
        }
    }
