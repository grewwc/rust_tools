//! Negotiated response gzip shared by the HTTP and HTTPS transports.

use std::{
    io::Write,
    pin::Pin,
    task::{Context, Poll},
};

use axum::{
    body::{Body, Bytes, HttpBody},
    extract::Request,
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::Next,
    response::Response,
};
use flate2::{Compression, write::GzEncoder};
use hyper::body::{Frame, SizeHint};

// An allowlist keeps archives, APKs, images other than SVG, and opaque media
// out of the compressor even when their bodies are large.
const COMPRESSIBLE_TYPES: &[&str] = &[
    "application/javascript",
    "application/json",
    "image/svg+xml",
    "text/css",
    "text/event-stream",
    "text/html",
    "text/javascript",
    "text/markdown",
    "text/plain",
];

pub(super) async fn gzip(request: Request, next: Next) -> Response {
    let negotiated = accepts_gzip(request.headers());
    let head = request.method() == Method::HEAD;
    let mut response = next.run(request).await;
    if !compressible(response.headers())
        || response.headers().contains_key(header::CONTENT_ENCODING)
        || response.headers().contains_key(header::CONTENT_RANGE)
        || response.status().is_informational()
        || matches!(
            response.status(),
            StatusCode::NO_CONTENT | StatusCode::NOT_MODIFIED | StatusCode::PARTIAL_CONTENT
        )
        || (!head && response.body().is_end_stream())
    {
        return response;
    }

    // Vary also belongs on the identity response so caches do not reuse it
    // for a later request that negotiates a different representation.
    vary_accept_encoding(response.headers_mut());
    if !negotiated {
        return response;
    }
    response
        .headers_mut()
        .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    response.headers_mut().remove(header::CONTENT_LENGTH);
    // Axum derives HEAD's Content-Length from the body before stripping it.
    // Wrap HEAD too so that the compressed length stays unknown.
    let (parts, body) = response.into_parts();
    Response::from_parts(parts, Body::new(GzipBody::new(body)))
}

fn accepts_gzip(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|encoding| {
            let mut parts = encoding.split(';');
            if !parts
                .next()
                .unwrap_or_default()
                .trim()
                .eq_ignore_ascii_case("gzip")
            {
                // An explicit gzip token is required; wildcard negotiation
                // does not change the behavior of existing callers.
                return false;
            }
            parts.all(|parameter| {
                let Some((name, value)) = parameter.trim().split_once('=') else {
                    return false;
                };
                !name.trim().eq_ignore_ascii_case("q")
                    || value
                        .trim()
                        .parse::<f32>()
                        .is_ok_and(|q| q > 0.0 && q <= 1.0)
            })
        })
}

fn compressible(headers: &HeaderMap) -> bool {
    let Some(content_type) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    COMPRESSIBLE_TYPES.contains(&media_type.as_str())
        || media_type
            .strip_prefix("application/")
            .is_some_and(|subtype| subtype.ends_with("+json"))
}

fn vary_accept_encoding(headers: &mut HeaderMap) {
    let present = headers
        .get_all(header::VARY)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|name| name.trim() == "*" || name.trim().eq_ignore_ascii_case("accept-encoding"));
    if !present {
        headers.append(header::VARY, HeaderValue::from_static("Accept-Encoding"));
    }
}

struct GzipBody {
    inner: Body,
    encoder: Option<GzEncoder<Vec<u8>>>,
    trailers: Option<Frame<Bytes>>,
}

impl GzipBody {
    fn new(inner: Body) -> Self {
        Self {
            inner,
            encoder: Some(GzEncoder::new(Vec::new(), Compression::fast())),
            trailers: None,
        }
    }

    fn finish(&mut self) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let encoder = self.encoder.take().expect("unfinished gzip body");
        Poll::Ready(Some(
            encoder
                .finish()
                .map(|bytes| Frame::data(Bytes::from(bytes)))
                .map_err(axum::Error::new),
        ))
    }
}

impl HttpBody for GzipBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = self.get_mut();
        if let Some(trailers) = this.trailers.take() {
            return Poll::Ready(Some(Ok(trailers)));
        }
        if this.encoder.is_none() {
            return Poll::Ready(None);
        }
        match std::task::ready!(Pin::new(&mut this.inner).poll_frame(cx)) {
            Some(Ok(frame)) => match frame.into_data() {
                Ok(data) => {
                    let encoder = this.encoder.as_mut().expect("unfinished gzip body");
                    // Sync-flush after every frame, including SSE keep-alives:
                    // the client can decode an event before the next arrives.
                    // Drain output per frame instead of buffering the stream;
                    // compress small responses too, without a buffering probe.
                    if let Err(error) = encoder.write_all(&data).and_then(|_| encoder.flush()) {
                        this.encoder = None;
                        return Poll::Ready(Some(Err(axum::Error::new(error))));
                    }
                    Poll::Ready(Some(Ok(Frame::data(Bytes::from(std::mem::take(
                        encoder.get_mut(),
                    ))))))
                }
                Err(trailers) => {
                    // Finish the gzip member before forwarding HTTP trailers.
                    this.trailers = Some(trailers);
                    this.finish()
                }
            },
            Some(Err(error)) => {
                this.encoder = None;
                Poll::Ready(Some(Err(error)))
            }
            None => this.finish(),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.encoder.is_none() && self.trailers.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[cfg(test)]
mod tests;
