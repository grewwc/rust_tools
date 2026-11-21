use super::*;

use std::{convert::Infallible, io::Read, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    response::{IntoResponse, Sse, sse::Event},
    routing::get,
};
use flate2::{read::GzDecoder, write::GzDecoder as StreamingGzDecoder};
use futures_util::{StreamExt, stream};
use hyper::service::Service;
use hyper_util::service::TowerToHyperService;

fn compressed_router(router: Router) -> Router {
    router.layer(axum::middleware::from_fn(gzip))
}

async fn request(router: Router, method: Method, path: &str, encoding: Option<&str>) -> Response {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(encoding) = encoding {
        request = request.header(header::ACCEPT_ENCODING, encoding);
    }
    // Exercise the same Tower-to-Hyper adapter used by the TLS server.
    TowerToHyperService::new(router)
        .call(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn body(response: Response) -> Bytes {
    axum::body::to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap()
}

fn decode(bytes: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    GzDecoder::new(bytes).read_to_end(&mut decoded).unwrap();
    decoded
}

fn fixture_json() -> serde_json::Value {
    serde_json::json!({"ok": true, "message": "compressible payload ".repeat(128)})
}

async fn json_response() -> Response {
    let value = fixture_json();
    let length = serde_json::to_vec(&value).unwrap().len();
    let mut response = Json(value).into_response();
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&length.to_string()).unwrap(),
    );
    response
        .headers_mut()
        .insert(header::VARY, HeaderValue::from_static("Origin"));
    response
}

#[tokio::test]
async fn serve_gzip_json_round_trip_and_identity_negotiation() {
    let router = compressed_router(Router::new().route("/json", get(json_response)));
    let expected = serde_json::to_vec(&fixture_json()).unwrap();
    for encoding in [
        "gzip",
        "gzip, deflate, br",
        "GZip ; q=0.5",
        "br;q=1, gzip;q=0.1",
    ] {
        let response = request(router.clone(), Method::GET, "/json", Some(encoding)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_ENCODING], "gzip");
        assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
        let vary = response
            .headers()
            .get_all(header::VARY)
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(vary, ["Origin", "Accept-Encoding"]);
        let bytes = body(response).await;
        assert!(bytes.len() < expected.len());
        let decoded = decode(&bytes);
        assert_eq!(decoded, expected);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&decoded).unwrap(),
            fixture_json()
        );
    }
    for encoding in [
        None,
        Some("identity"),
        Some("br"),
        Some("*"),
        Some("gzip;q=0"),
        Some("gzip;q=invalid"),
        Some("gzip;q=2"),
        Some("xgzip"),
    ] {
        let response = request(router.clone(), Method::GET, "/json", encoding).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!response.headers().contains_key(header::CONTENT_ENCODING));
        assert_eq!(
            response.headers()[header::CONTENT_LENGTH],
            expected.len().to_string()
        );
        assert_eq!(body(response).await.as_ref(), expected);
    }
    let response = request(router, Method::HEAD, "/json", Some("gzip")).await;
    assert_eq!(response.headers()[header::CONTENT_ENCODING], "gzip");
    assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
    assert!(body(response).await.is_empty());
}

fn gated_sse_router() -> (Router, tokio::sync::oneshot::Sender<()>) {
    let (release, wait) = tokio::sync::oneshot::channel();
    let wait = Arc::new(std::sync::Mutex::new(Some(wait)));
    let router = Router::new().route(
        "/events",
        get(move || {
            let wait = wait.lock().unwrap().take().unwrap();
            async move {
                let first =
                    stream::once(async { Ok::<_, Infallible>(Event::default().data("first")) });
                let second = stream::once(async move {
                    wait.await.unwrap();
                    Ok::<_, Infallible>(Event::default().data("second"))
                });
                Sse::new(first.chain(second)).keep_alive(axum::response::sse::KeepAlive::default())
            }
        }),
    );
    (compressed_router(router), release)
}

#[tokio::test]
async fn serve_gzip_sse_flushes_before_stream_end() {
    let expected = b"data: first\n\ndata: second\n\n";
    for encoding in [None, Some("gzip")] {
        let (router, release) = gated_sse_router();
        let response = request(router, Method::GET, "/events", encoding).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
        assert_eq!(
            response.headers().get(header::CONTENT_ENCODING).is_some(),
            encoding.is_some()
        );
        let mut stream = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("the first event must arrive while the second is gated")
            .unwrap()
            .unwrap();
        let mut decoder = StreamingGzDecoder::new(Vec::new());
        let mut identity = Vec::new();
        if encoding.is_some() {
            decoder.write_all(&first).unwrap();
            decoder.flush().unwrap();
            assert_eq!(decoder.get_ref(), b"data: first\n\n");
        } else {
            identity.extend_from_slice(&first);
            assert_eq!(identity, b"data: first\n\n");
        }
        assert!(!release.is_closed(), "the stream has not finished");
        release.send(()).unwrap();
        while let Some(bytes) = stream.next().await {
            let bytes = bytes.unwrap();
            if encoding.is_some() {
                decoder.write_all(&bytes).unwrap();
            } else {
                identity.extend_from_slice(&bytes);
            }
        }
        let decoded = if encoding.is_some() {
            decoder.finish().unwrap()
        } else {
            identity
        };
        assert_eq!(decoded, expected);
    }
    eprintln!("SSE: first event decoded before releasing the second event (gzip and identity)");
}

#[tokio::test]
async fn serve_gzip_content_type_policy_and_passthrough() {
    let payload = "payload ".repeat(256);
    for (content_type, compressed) in [
        ("text/html; charset=utf-8", true),
        ("text/css", true),
        ("text/plain", true),
        ("text/markdown", true),
        ("text/javascript", true),
        ("application/javascript", true),
        ("application/json", true),
        ("application/problem+json", true),
        ("image/svg+xml", true),
        ("image/png", false),
        ("image/jpeg", false),
        ("image/webp", false),
        ("image/bmp", false),
        ("audio/wav", false),
        ("application/zip", false),
        ("application/vnd.android.package-archive", false),
        ("application/octet-stream", false),
    ] {
        let payload = payload.clone();
        let router = compressed_router(Router::new().route(
            "/payload",
            get(move || {
                let payload = payload.clone();
                async move { ([(header::CONTENT_TYPE, content_type)], payload) }
            }),
        ));
        let response = request(router, Method::GET, "/payload", Some("gzip")).await;
        assert_eq!(
            response.headers().contains_key(header::CONTENT_ENCODING),
            compressed,
            "{content_type}"
        );
        let bytes = body(response).await;
        let decoded = if compressed {
            decode(&bytes)
        } else {
            bytes.to_vec()
        };
        assert_eq!(decoded, "payload ".repeat(256).as_bytes());
    }
    for extra in [header::CONTENT_ENCODING, header::CONTENT_RANGE] {
        let router = compressed_router(Router::new().route(
            "/encoded",
            get(move || {
                let extra = extra.clone();
                async move {
                    let mut response = "already encoded".into_response();
                    response
                        .headers_mut()
                        .insert(extra, HeaderValue::from_static("identity"));
                    response
                }
            }),
        ));
        let response = request(router, Method::GET, "/encoded", Some("gzip")).await;
        assert_ne!(
            response
                .headers()
                .get(header::CONTENT_ENCODING)
                .and_then(|value| value.to_str().ok()),
            Some("gzip")
        );
        assert_eq!(body(response).await, "already encoded");
    }
}

#[tokio::test]
async fn serve_gzip_webui_assets_and_binary_downloads() {
    use super::super::info;

    let router = compressed_router(
        Router::new()
            .route("/", get(info::serve_app))
            .route("/app", get(info::serve_app))
            .route("/app/", get(info::serve_app))
            .route("/app.webmanifest", get(info::serve_app_manifest))
            .route("/app-icon.svg", get(info::serve_app_icon))
            .route("/app-icon-192.png", get(info::serve_app_icon_192))
            .route("/app-icon-512.png", get(info::serve_app_icon_512))
            .route("/app-sw.js", get(info::serve_app_service_worker))
            .route("/a-serve.apk", get(info::serve_android_apk)),
    );
    for path in [
        "/",
        "/app",
        "/app/",
        "/app.webmanifest",
        "/app-icon.svg",
        "/app-sw.js",
    ] {
        let identity = body(request(router.clone(), Method::GET, path, None).await).await;
        let response = request(router.clone(), Method::GET, path, Some("gzip, deflate, br")).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_ENCODING], "gzip");
        assert_eq!(response.headers()[header::VARY], "Accept-Encoding");
        assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
        if path == "/app-sw.js" {
            assert_eq!(response.headers()["service-worker-allowed"], "/");
            assert!(
                std::str::from_utf8(&identity)
                    .unwrap()
                    .contains("respondWith(fetch(event.request))")
            );
        }
        let compressed = body(response).await;
        assert_eq!(decode(&compressed), identity);
        eprintln!(
            "{path}: 200 Content-Encoding=gzip Vary=Accept-Encoding, {} -> {} bytes",
            identity.len(),
            compressed.len()
        );
    }
    for path in ["/app-icon-192.png", "/app-icon-512.png", "/a-serve.apk"] {
        let identity = body(request(router.clone(), Method::GET, path, None).await).await;
        let response = request(router.clone(), Method::GET, path, Some("gzip, deflate, br")).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!response.headers().contains_key(header::CONTENT_ENCODING));
        assert_eq!(body(response).await, identity);
        eprintln!("{path}: 200 identity, {} unchanged bytes", identity.len());
    }
}
