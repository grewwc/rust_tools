use std::io::Read;
use std::time::Duration;

use regex::Regex;
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

const HTTP_TOOL_TIMEOUT: Duration = Duration::from_secs(10);
const HTTP_SEARCH_TIMEOUT: Duration = Duration::from_secs(5);

fn params_web_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text."
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 20)."
            }
        },
        "required": ["query"]
    })
}

fn params_web_fetch() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "http/https URL to fetch. Localhost and private network targets are blocked; response body is capped at 512KB."
            }
        },
        "required": ["url"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "web_search",
        description: "Search the public web using DuckDuckGo for documentation and references. Returns up to num_results results with title, URL, and snippet.",
        parameters: params_web_search,
        execute: execute_web_search,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "web_fetch",
        description: "Fetch the raw response body of an http/https URL (10s timeout, 512KB cap). Blocks localhost/private network targets. Returns URL, status, content-type, and content.",
        parameters: params_web_fetch,
        execute: execute_web_fetch,
        groups: &["builtin"],
    }
});

/// Format search results as a readable string
fn format_search_results(hits: &[WebSearchHit]) -> String {
    if hits.is_empty() {
        return "No results found.".to_string();
    }

    let mut output = String::new();
    output.push_str(&format!("Found {} result(s):\n\n", hits.len()));

    for (i, hit) in hits.iter().enumerate() {
        output.push_str(&format!("{}. {}\n", i + 1, hit.title));
        output.push_str(&format!("   URL: {}\n", hit.url));
        if !hit.snippet.is_empty() {
            output.push_str(&format!("   Snippet: {}\n", hit.snippet));
        }
        output.push('\n');
    }

    output
}

pub(crate) fn execute_web_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query parameter")?;
    let num_results = args["num_results"].as_u64().unwrap_or(5) as usize;

    if query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let limit = if num_results == 0 {
        5
    } else {
        num_results.min(20)
    };

    match duckduckgo_search(query, limit) {
        Ok(hits) => {
            if hits.is_empty() {
                eprintln!("[web_search] No results from primary. Trying fallback...");
                duckduckgo_search_fallback(query, limit).map(|h| format_search_results(&h))
            } else {
                Ok(format_search_results(&hits))
            }
        }
        Err(e) => {
            eprintln!(
                "[web_search] Primary search failed: {}. Trying fallback...",
                e
            );
            match duckduckgo_search_fallback(query, limit) {
                Ok(hits) => Ok(format_search_results(&hits)),
                Err(e) => Err(format!(
                    "Search failed: {}. Try a different query or check network connectivity.",
                    e
                )),
            }
        }
    }
}

pub(crate) fn execute_web_fetch(args: &Value) -> Result<String, String> {
    let url = args["url"].as_str().ok_or("Missing url")?;
    let parsed = reqwest::Url::parse(url).map_err(|_| "Invalid url".to_string())?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err("Only http/https urls are allowed".to_string());
    }
    let Some(host) = parsed.host_str() else {
        return Err("Invalid url host".to_string());
    };
    let host_lc = host.to_lowercase();
    if host_lc == "localhost" || host_lc.ends_with(".localhost") || host_lc.ends_with(".local") {
        return Err("Blocked url host".to_string());
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_private() || v4.is_loopback() || v4.is_link_local() || v4.is_multicast()
            }
            std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unique_local(),
        };
        if blocked {
            return Err("Blocked url host".to_string());
        }
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TOOL_TIMEOUT)
        .connect_timeout(Duration::from_secs(5))
        .user_agent("Mozilla/5.0 (compatible; rust-tools/1.0)")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    const MAX_BYTES: usize = 512 * 1024;

    // Extract metadata before consuming response (clone to avoid borrow issues)
    let status = response.status().as_u16();
    let content_type: String = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/plain")
        .to_string();

    let mut buf = Vec::new();
    response
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read response: {}", e))?;

    let truncated = buf.len() > MAX_BYTES;
    if truncated {
        buf.truncate(MAX_BYTES);
    }
    let content = String::from_utf8_lossy(&buf).to_string();

    // Add metadata header
    let mut result = String::new();
    result.push_str(&format!("URL: {}\n", url));
    result.push_str(&format!("Status: {}\n", status));
    result.push_str(&format!("Content-Type: {}\n", content_type));
    result.push_str(&format!("Size: {} bytes", buf.len()));
    if truncated {
        result.push_str(" (truncated at 512KB)");
    }
    result.push_str("\n\n--- Content ---\n\n");
    result.push_str(&content);

    Ok(result)
}

/// Fallback search using DuckDuckGo lite version
fn duckduckgo_search_fallback(query: &str, limit: usize) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_SEARCH_TIMEOUT)
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    // Try the lite version of DuckDuckGo
    let response = client
        .get("https://lite.duckduckgo.com/lite/")
        .query(&[("q", query)])
        .send()
        .map_err(|e| format!("Failed to perform web search: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("Web search failed: HTTP {}", status.as_u16()));
    }

    let html = response
        .text()
        .map_err(|e| format!("Failed to read search response: {}", e))?;

    Ok(parse_duckduckgo_lite(&html, limit))
}

/// Parse DuckDuckGo lite HTML format
fn parse_duckduckgo_lite(html: &str, limit: usize) -> Vec<WebSearchHit> {
    let mut out = Vec::new();
    let lines: Vec<&str> = html.lines().collect();

    for i in 0..lines.len().saturating_sub(2) {
        if out.len() >= limit {
            break;
        }

        let line = lines[i].trim();
        if line.contains("<a") && line.contains("href=") {
            // Extract URL
            if let Some(url_start) = line.find("href=\"") {
                let url_rest = &line[url_start + 6..];
                if let Some(url_end) = url_rest.find('"') {
                    let raw_url = url_rest[..url_end].to_string();
                    let url = if raw_url.starts_with("http") {
                        raw_url
                    } else {
                        format!("https://lite.duckduckgo.com{}", raw_url)
                    };

                    // Extract title
                    let title = clean_html_text(line);

                    // Extract snippet (next line usually contains it)
                    let snippet = if i + 1 < lines.len() {
                        clean_html_text(lines[i + 1])
                    } else {
                        String::new()
                    };

                    if !title.is_empty() && !url.is_empty() {
                        out.push(WebSearchHit {
                            title,
                            url,
                            snippet,
                        });
                    }
                }
            }
        }
    }

    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchHit {
    title: String,
    url: String,
    snippet: String,
}

fn duckduckgo_search(query: &str, limit: usize) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_SEARCH_TIMEOUT)
        .connect_timeout(Duration::from_secs(3))
        .user_agent("Mozilla/5.0 (compatible; rust-tools/1.0)")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let response = client
        .get("https://duckduckgo.com/html/")
        .query(&[("q", query)])
        .send()
        .map_err(|e| format!("Failed to perform web search: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("Web search failed: HTTP {}", status.as_u16()));
    }

    let html = response
        .text()
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    Ok(parse_duckduckgo_html(&html, limit))
}

fn parse_duckduckgo_html(html: &str, limit: usize) -> Vec<WebSearchHit> {
    // Try multiple regex patterns for flexibility
    let title_re = Regex::new(
        r#"(?s)<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>"#,
    )
    .unwrap_or_else(|_| {
        Regex::new(
            r#"(?s)<a[^>]*href="(?P<url>[^"]+)"[^>]*class="result__a"[^>]*>(?P<title>.*?)</a>"#,
        )
        .unwrap()
    });

    let snippet_re = Regex::new(r#"(?s)<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(?P<snippet2>.*?)</div>"#)
        .ok()
        .or_else(|| Regex::new(r#"(?s)<div[^>]*>(?P<snippet>.*?)</div>"#).ok());

    let mut out = Vec::new();
    for m in title_re.captures_iter(html) {
        if out.len() >= limit {
            break;
        }
        let raw_url = m.name("url").map(|m| m.as_str()).unwrap_or("").to_string();
        let url = normalize_duckduckgo_url(&raw_url);
        let title_html = m.name("title").map(|m| m.as_str()).unwrap_or("");
        let title = clean_html_text(title_html);

        let mut snippet = String::new();
        if let Some(snippet_re) = snippet_re.as_ref() {
            let window_start = m.get(0).map(|m| m.end()).unwrap_or(0);
            let mut window_end = (window_start + 4000).min(html.len());
            while window_end > window_start && !html.is_char_boundary(window_end) {
                window_end -= 1;
            }
            let window = html.get(window_start..window_end).unwrap_or("");
            if let Some(caps) = snippet_re.captures(window) {
                let snippet_html = caps
                    .name("snippet")
                    .or_else(|| caps.name("snippet2"))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                snippet = clean_html_text(snippet_html);
            }
        }

        if title.trim().is_empty() {
            continue;
        }
        out.push(WebSearchHit {
            title,
            url,
            snippet,
        });
    }
    out
}

fn normalize_duckduckgo_url(url: &str) -> String {
    let decoded_url = decode_html_entities(url.trim());
    if let Some(decoded) = extract_duckduckgo_uddg(&decoded_url) {
        return decoded;
    }
    decoded_url
}

fn extract_duckduckgo_uddg(url: &str) -> Option<String> {
    let idx = url.find("uddg=")?;
    let rest = &url[idx + 5..];
    let value = rest.split('&').next().unwrap_or(rest);
    let decoded = percent_decode(value)?;
    if decoded.trim().is_empty() {
        None
    } else {
        Some(decoded)
    }
}

fn percent_decode(s: &str) -> Option<String> {
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

fn clean_html_text(s: &str) -> String {
    let without_tags = strip_html_tags(s);
    let decoded = decode_html_entities(&without_tags);
    decoded
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_html_entities(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < bytes.len() && bytes[end] != b';' {
            end += 1;
        }
        if end >= bytes.len() {
            out.push(b'&');
            i += 1;
            continue;
        }

        let entity_bytes = &bytes[i + 1..end];
        let decoded = std::str::from_utf8(entity_bytes)
            .ok()
            .and_then(decode_single_entity);

        if let Some(decoded) = decoded {
            out.extend_from_slice(decoded.as_bytes());
        } else {
            out.push(b'&');
            out.extend_from_slice(entity_bytes);
            out.push(b';');
        }
        i = end + 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn decode_single_entity(entity: &str) -> Option<String> {
    match entity {
        "amp" => Some("&".to_string()),
        "lt" => Some("<".to_string()),
        "gt" => Some(">".to_string()),
        "quot" => Some("\"".to_string()),
        "apos" => Some("'".to_string()),
        _ if entity.starts_with("#x") || entity.starts_with("#X") => {
            let hex = &entity[2..];
            let v = u32::from_str_radix(hex, 16).ok()?;
            char::from_u32(v).map(|c| c.to_string())
        }
        _ if entity.starts_with('#') => {
            let dec = &entity[1..];
            let v = dec.parse::<u32>().ok()?;
            char::from_u32(v).map(|c| c.to_string())
        }
        _ => None,
    }
}
