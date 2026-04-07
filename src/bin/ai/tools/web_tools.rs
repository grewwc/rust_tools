use std::collections::HashSet;
use std::io::Read;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use regex::Regex;
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

const HTTP_TOOL_TIMEOUT: Duration = Duration::from_secs(10);
const HTTP_SEARCH_TIMEOUT: Duration = Duration::from_secs(4);
const HTTP_SEARCH_TOTAL_BUDGET: Duration = Duration::from_secs(12);
const MAX_SEARCH_RETRIES: usize = 2;
const MAX_PUBLIC_SEARXNG_INSTANCES_PER_ATTEMPT: usize = 1;
const DEFAULT_NUM_RESULTS: usize = 10;
const MAX_NUM_RESULTS: usize = 20;
const CACHE_MAX_ENTRIES: usize = 128;
const CACHE_TTL_MS: i64 = 300_000; // 5 minutes

type SearchCache = Mutex<rust_tools::cw::LruCache<String, Result<Vec<WebSearchHit>, String>>>;

static SEARCH_CACHE: LazyLock<SearchCache> = LazyLock::new(|| {
    Mutex::new(rust_tools::cw::LruCache::with_ttl(
        CACHE_MAX_ENTRIES,
        CACHE_TTL_MS,
    ))
});

/// User-Agent pool for rotation
const USER_AGENTS: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
];

fn params_web_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text. Tips: use concise keywords instead of full sentences, mix English terms for technical queries, and use site:domain.com to target specific sites."
            },
            "region": {
                "type": "string",
                "description": "Search region code for localized results (e.g. 'cn-zh' for Chinese, 'us-en' for English, 'jp-jp' for Japanese). Default: 'wt-wt' (worldwide)."
            },
            "time_range": {
                "type": "string",
                "description": "Time range filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year), or omit for no filter."
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10, max: 20)."
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
        description: "Search the public web using DuckDuckGo for real-time information including weather, news, stock prices, current events, documentation, and references. Returns up to num_results results with title, URL, and snippet.",
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

    if query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let num_results = args["num_results"]
        .as_u64()
        .unwrap_or(DEFAULT_NUM_RESULTS as u64) as usize;
    let limit = if num_results == 0 {
        DEFAULT_NUM_RESULTS
    } else {
        num_results.min(MAX_NUM_RESULTS)
    };

    let region = args["region"].as_str().unwrap_or("wt-wt");
    let time_range = args["time_range"].as_str().unwrap_or("");
    eprintln!(
        "[web_search] Start query={:?}, region={}, time_range={}, limit={}",
        query, region, time_range, limit
    );

    // Check cache first
    let cache_key = format!("{}|{}|{}|{}", query, region, time_range, limit);
    if let Ok(mut cache) = SEARCH_CACHE.lock() {
        if let Some(cached_result) = cache.get_ref(&cache_key) {
            eprintln!("[web_search] Cache hit for query: {}", query);
            match cached_result {
                Ok(hits) => return Ok(format_search_results(&hits)),
                Err(e) => return Err(format!("Cached error: {}", e)),
            }
        }
    }

    // Try search with retries
    let result = search_with_retries(query, region, time_range, limit);

    // Cache the result (LruCache handles TTL internally)
    if let Ok(mut cache) = SEARCH_CACHE.lock() {
        cache.put(cache_key, result.clone());
    }

    match result {
        Ok(hits) => {
            if hits.is_empty() {
                eprintln!("[web_search] No results for query: {}", query);
                Err(format!(
                    "No results found for: {}. Try different keywords or check spelling.",
                    query
                ))
            } else {
                Ok(format_search_results(&hits))
            }
        }
        Err(e) => Err(format!(
            "Search failed: {}. Try a different query or check network connectivity.",
            e
        )),
    }
}

fn search_with_retries(
    query: &str,
    region: &str,
    time_range: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, String> {
    let mut last_error = String::new();
    let started_at = std::time::Instant::now();
    let deadline = started_at + HTTP_SEARCH_TOTAL_BUDGET;

    for attempt in 0..MAX_SEARCH_RETRIES {
        if started_at.elapsed() >= HTTP_SEARCH_TOTAL_BUDGET {
            break;
        }
        let Some(timeout) = remaining_search_timeout(deadline) else {
            break;
        };
        eprintln!(
            "[web_search] Attempt {}/{} (timeout {:?}, elapsed {:?})",
            attempt + 1,
            MAX_SEARCH_RETRIES,
            timeout,
            started_at.elapsed()
        );

        // Try primary search
        match duckduckgo_search(query, region, time_range, limit, timeout) {
            Ok(hits) if !hits.is_empty() => {
                eprintln!("[web_search] Success on primary (attempt {})", attempt + 1);
                return Ok(hits);
            }
            Ok(_) => {
                eprintln!(
                    "[web_search] Primary returned empty (attempt {})",
                    attempt + 1
                );
            }
            Err(e) => {
                eprintln!(
                    "[web_search] Primary failed (attempt {}): {}",
                    attempt + 1,
                    e
                );
                last_error = e;
            }
        }

        let Some(timeout) = remaining_search_timeout(deadline) else {
            break;
        };
        // Try fallback
        match duckduckgo_search_fallback(query, region, time_range, limit, timeout) {
            Ok(hits) if !hits.is_empty() => {
                eprintln!("[web_search] Success on fallback (attempt {})", attempt + 1);
                return Ok(hits);
            }
            Ok(_) => {
                eprintln!(
                    "[web_search] Fallback returned empty (attempt {})",
                    attempt + 1
                );
            }
            Err(e) => {
                eprintln!(
                    "[web_search] Fallback failed (attempt {}): {}",
                    attempt + 1,
                    e
                );
                last_error = e;
            }
        }

        let Some(timeout) = remaining_search_timeout(deadline) else {
            break;
        };
        // Try alternative search endpoint
        match duckduckgo_search_alternative(query, limit, timeout) {
            Ok(hits) if !hits.is_empty() => {
                eprintln!(
                    "[web_search] Success on alternative (attempt {})",
                    attempt + 1
                );
                return Ok(hits);
            }
            Ok(_) => {
                eprintln!(
                    "[web_search] Alternative returned empty (attempt {})",
                    attempt + 1
                );
            }
            Err(e) => {
                eprintln!(
                    "[web_search] Alternative failed (attempt {}): {}",
                    attempt + 1,
                    e
                );
                last_error = e;
            }
        }

        let Some(timeout) = remaining_search_timeout(deadline) else {
            break;
        };
        // Try SearXNG search (free, open-source metasearch engine)
        if let Some(instance) = std::env::var("SEARXNG_INSTANCE")
            .ok()
            .filter(|s| !s.is_empty())
        {
            match searxng_search(&instance, query, region, time_range, limit, timeout) {
                Ok(hits) if !hits.is_empty() => {
                    eprintln!("[web_search] Success on SearXNG (attempt {})", attempt + 1);
                    return Ok(hits);
                }
                Ok(_) => {
                    eprintln!(
                        "[web_search] SearXNG returned empty (attempt {})",
                        attempt + 1
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[web_search] SearXNG failed (attempt {}): {}",
                        attempt + 1,
                        e
                    );
                    last_error = e;
                }
            }
        } else {
            // Try public SearXNG instances as fallback
            for instance in SEARXNG_PUBLIC_INSTANCES
                .iter()
                .take(MAX_PUBLIC_SEARXNG_INSTANCES_PER_ATTEMPT)
            {
                let Some(timeout) = remaining_search_timeout(deadline) else {
                    break;
                };
                match searxng_search(instance, query, region, time_range, limit, timeout) {
                    Ok(hits) if !hits.is_empty() => {
                        eprintln!(
                            "[web_search] Success on public SearXNG {} (attempt {})",
                            instance,
                            attempt + 1
                        );
                        return Ok(hits);
                    }
                    Ok(_) => {
                        eprintln!(
                            "[web_search] Public SearXNG {} returned empty (attempt {})",
                            instance,
                            attempt + 1
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[web_search] Public SearXNG {} failed (attempt {}): {}",
                            instance,
                            attempt + 1,
                            e
                        );
                        last_error = e;
                    }
                }
            }
        }

        // Exponential backoff before retry (except for last attempt)
        if attempt < MAX_SEARCH_RETRIES - 1 && started_at.elapsed() < HTTP_SEARCH_TOTAL_BUDGET {
            let delay_ms = [100u64, 200, 400].get(attempt).copied().unwrap_or(400);
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }

    Err(format!(
        "All search methods failed within {:?}. Last error: {}",
        HTTP_SEARCH_TOTAL_BUDGET, last_error
    ))
}

fn remaining_search_timeout(deadline: std::time::Instant) -> Option<Duration> {
    let remaining = deadline.checked_duration_since(std::time::Instant::now())?;
    if remaining.is_zero() {
        return None;
    }
    Some(remaining.min(HTTP_SEARCH_TIMEOUT))
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
        .user_agent(USER_AGENTS[0])
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

/// Public SearXNG instances (no API key required, but may be rate-limited)
const SEARXNG_PUBLIC_INSTANCES: &[&str] = &[
    "https://search.bus-hit.me",
    "https://search.mdosch.de",
    "https://searx.be",
];

/// SearXNG search API (free, open-source metasearch engine)
fn searxng_search(
    base_url: &str,
    query: &str,
    region: &str,
    time_range: &str,
    limit: usize,
    timeout: Duration,
) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .user_agent(get_random_user_agent())
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    // SearXNG API: https://docs.searxng.org/dev/search_api.html
    let mut params = vec![
        ("q", query.to_string()),
        ("format", "json".to_string()),
        ("categories", "general".to_string()),
    ];

    if region != "wt-wt" {
        let lang = region.replace("wt-wt", "all").replace('-', "-");
        params.push(("language", lang));
    }

    if !time_range.is_empty() {
        let tr = match time_range {
            "d" => "day",
            "w" => "week",
            "m" => "month",
            "y" => "year",
            _ => time_range,
        };
        params.push(("time_range", tr.to_string()));
    }

    let response = client
        .get(format!("{}/search", base_url.trim_end_matches('/')))
        .query(&params)
        .send()
        .map_err(|e| format!("Failed to perform SearXNG search: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("SearXNG search failed: HTTP {}", status.as_u16()));
    }

    let json_str = response
        .text()
        .map_err(|e| format!("Failed to read SearXNG response: {}", e))?;

    Ok(parse_searxng_json(&json_str, limit))
}

/// Alternative search using DuckDuckGo HTML API
fn duckduckgo_search_alternative(
    query: &str,
    limit: usize,
    timeout: Duration,
) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .user_agent(get_random_user_agent())
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    // Try DDG API endpoint
    let response = client
        .get("https://api.duckduckgo.com/")
        .query(&[
            ("q", query),
            ("format", "json"),
            ("no_html", "1"),
            ("skip_disambig", "1"),
        ])
        .send()
        .map_err(|e| format!("Failed to perform web search: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("Web search failed: HTTP {}", status.as_u16()));
    }

    let json_str = response
        .text()
        .map_err(|e| format!("Failed to read search response: {}", e))?;

    Ok(parse_duckduckgo_api(&json_str, limit))
}

/// Fallback search using DuckDuckGo lite version
fn duckduckgo_search_fallback(
    query: &str,
    region: &str,
    time_range: &str,
    limit: usize,
    timeout: Duration,
) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .user_agent(get_random_user_agent())
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    // Try the lite version of DuckDuckGo
    let mut params = vec![("q", query)];
    if region != "wt-wt" {
        params.push(("kl", region));
    }
    if !time_range.is_empty() {
        params.push(("df", time_range));
    }

    let response = client
        .get("https://lite.duckduckgo.com/lite/")
        .query(&params)
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
    let mut seen_urls = HashSet::new();

    // Use a more robust regex for lite parsing
    let lite_result_re = Regex::new(
        r#"(?s)<a[^>]*class="result-link"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>"#,
    );

    if let Ok(re) = lite_result_re {
        for caps in re.captures_iter(html) {
            if out.len() >= limit {
                break;
            }

            let url = caps
                .name("url")
                .map(|m| m.as_str())
                .unwrap_or("")
                .to_string();
            let title = clean_html_text(caps.name("title").map(|m| m.as_str()).unwrap_or(""));

            if url.is_empty() || title.is_empty() || seen_urls.contains(&url) {
                continue;
            }
            seen_urls.insert(url.clone());

            out.push(WebSearchHit {
                title,
                url,
                snippet: String::new(),
            });
        }
    }

    // Fallback to line-based parsing if regex didn't work
    if out.is_empty() {
        let lines: Vec<&str> = html.lines().collect();
        for i in 0..lines.len().saturating_sub(2) {
            if out.len() >= limit {
                break;
            }

            let line = lines[i].trim();
            if line.contains("<a") && line.contains("href=") && line.contains("result-link") {
                if let Some(url_start) = line.find("href=\"") {
                    let url_rest = &line[url_start + 6..];
                    if let Some(url_end) = url_rest.find('"') {
                        let raw_url = url_rest[..url_end].to_string();
                        let url = if raw_url.starts_with("http") {
                            raw_url
                        } else {
                            format!("https://lite.duckduckgo.com{}", raw_url)
                        };

                        let title = clean_html_text(line);
                        let snippet = if i + 1 < lines.len() {
                            clean_html_text(lines[i + 1])
                        } else {
                            String::new()
                        };

                        if !title.is_empty() && !url.is_empty() && !seen_urls.contains(&url) {
                            seen_urls.insert(url.clone());
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
    }

    out
}

/// Parse DuckDuckGo JSON API response
fn parse_duckduckgo_api(json_str: &str, limit: usize) -> Vec<WebSearchHit> {
    let mut out = Vec::new();
    let mut seen_urls = HashSet::new();

    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
        // Extract AbstractURL and Abstract
        if let Some(abs_url) = value["AbstractURL"].as_str() {
            if !abs_url.is_empty() && seen_urls.insert(abs_url.to_string()) {
                out.push(WebSearchHit {
                    title: value["Heading"].as_str().unwrap_or("No title").to_string(),
                    url: abs_url.to_string(),
                    snippet: value["Abstract"].as_str().unwrap_or("").to_string(),
                });
            }
        }

        // Extract related topics
        if let Some(topics) = value["RelatedTopics"].as_array() {
            for topic in topics {
                if out.len() >= limit {
                    break;
                }

                if let Some(url) = topic["FirstURL"].as_str() {
                    let title = topic["Text"].as_str().unwrap_or("").to_string();
                    if !url.is_empty() && !title.is_empty() && seen_urls.insert(url.to_string()) {
                        out.push(WebSearchHit {
                            title,
                            url: url.to_string(),
                            snippet: String::new(),
                        });
                    }
                }

                // Handle nested topics
                if let Some(topics_arr) = topic["Topics"].as_array() {
                    for nested in topics_arr {
                        if out.len() >= limit {
                            break;
                        }

                        if let Some(url) = nested["FirstURL"].as_str() {
                            let title = nested["Text"].as_str().unwrap_or("").to_string();
                            if !url.is_empty()
                                && !title.is_empty()
                                && seen_urls.insert(url.to_string())
                            {
                                out.push(WebSearchHit {
                                    title,
                                    url: url.to_string(),
                                    snippet: String::new(),
                                });
                            }
                        }
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

fn duckduckgo_search(
    query: &str,
    region: &str,
    time_range: &str,
    limit: usize,
    timeout: Duration,
) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(5))
        .user_agent(get_random_user_agent())
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let mut params = vec![("q", query)];
    if region != "wt-wt" {
        params.push(("kl", region));
    }
    if !time_range.is_empty() {
        params.push(("df", time_range));
    }

    let response = client
        .get("https://duckduckgo.com/html/")
        .query(&params)
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
    let mut out = Vec::new();
    let mut seen_urls = HashSet::new();

    // Multiple patterns for robustness
    let title_patterns = [
        r#"(?s)<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>"#,
        r#"(?s)<a[^>]*href="(?P<url>[^"]+)"[^>]*class="result__a"[^>]*>(?P<title>.*?)</a>"#,
        r#"(?s)<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>"#,
    ];

    let snippet_patterns = [
        r#"(?s)<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>"#,
        r#"(?s)<div[^>]*class="result__snippet"[^>]*>(?P<snippet2>.*?)</div>"#,
    ];

    // Compile title regexes
    let title_res: Vec<Regex> = title_patterns
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect();

    if title_res.is_empty() {
        return out;
    }

    // Find all matches across all patterns
    let mut all_matches: Vec<(String, String)> = Vec::new();
    for re in &title_res {
        for caps in re.captures_iter(html) {
            let url = caps
                .name("url")
                .map(|m| m.as_str())
                .unwrap_or("")
                .to_string();
            let title = caps.name("title").map(|m| m.as_str()).unwrap_or("");
            let title = clean_html_text(title);

            if !title.is_empty() && !url.is_empty() {
                all_matches.push((url, title));
            }
        }
    }

    // Deduplicate and collect results
    for (raw_url, title) in all_matches {
        if out.len() >= limit {
            break;
        }

        let url = normalize_duckduckgo_url(&raw_url);
        if seen_urls.contains(&url) {
            continue;
        }
        seen_urls.insert(url.clone());

        // Try to find snippet near the match
        let mut snippet = String::new();
        let snippet_res: Vec<Regex> = snippet_patterns
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect();

        for re in &snippet_res {
            if let Some(caps) = re.captures(html) {
                let snippet_html = caps
                    .name("snippet")
                    .or_else(|| caps.name("snippet2"))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                snippet = clean_html_text(snippet_html);
                if !snippet.is_empty() {
                    break;
                }
            }
        }

        out.push(WebSearchHit {
            title,
            url,
            snippet,
        });
    }

    out
}

fn get_random_user_agent() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    USER_AGENTS[(timestamp as usize) % USER_AGENTS.len()]
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

/// Parse SearXNG JSON API response
fn parse_searxng_json(json_str: &str, limit: usize) -> Vec<WebSearchHit> {
    let mut out = Vec::new();
    let mut seen_urls = HashSet::new();

    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
        if let Some(results) = value["results"].as_array() {
            for result in results {
                if out.len() >= limit {
                    break;
                }

                if let Some(url) = result["url"].as_str() {
                    if seen_urls.insert(url.to_string()) {
                        out.push(WebSearchHit {
                            title: result["title"].as_str().unwrap_or("").to_string(),
                            url: url.to_string(),
                            snippet: result["content"].as_str().unwrap_or("").to_string(),
                        });
                    }
                }
            }
        }
    }

    out
}
