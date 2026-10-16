use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};

use axum::Router;
use hyper::server::conn::http1;
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ServeTlsConfig {
    pub(super) cert: String,
    pub(super) key: String,
}

pub(super) fn resolve_serve_tls(
    cert: &str,
    key: &str,
    fallback: Option<&ServeTlsConfig>,
) -> Result<Option<ServeTlsConfig>, String> {
    let cert = cert.trim();
    let key = key.trim();
    match (cert.is_empty(), key.is_empty()) {
        (true, true) => Ok(fallback.cloned()),
        (false, false) => Ok(Some(ServeTlsConfig {
            cert: cert.to_string(),
            key: key.to_string(),
        })),
        _ => Err("--serve-tls-cert and --serve-tls-key must be provided together".to_string()),
    }
}

pub(super) struct TlsServer {
    listener: TcpListener,
    acceptor: TlsAcceptor,
}

impl TlsServer {
    pub(super) async fn bind(
        bind: &str,
        tls: &ServeTlsConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let certs =
            CertificateDer::pem_file_iter(Path::new(&tls.cert))?.collect::<Result<Vec<_>, _>>()?;
        if certs.is_empty() {
            return Err(format!(
                "TLS certificate file contains no certificates: {}",
                tls.cert
            )
            .into());
        }
        let key = PrivateKeyDer::from_pem_file(Path::new(&tls.key))?;
        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;
        let listener = TcpListener::bind(bind).await?;
        Ok(Self {
            listener,
            acceptor: TlsAcceptor::from(Arc::new(config)),
        })
    }

    pub(super) fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    pub(super) async fn serve(self, app: Router) -> std::io::Result<()> {
        loop {
            let (stream, peer) = match self.listener.accept().await {
                Ok(accepted) => accepted,
                Err(err) => {
                    eprintln!("[serve] TCP accept failed: {err}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let acceptor = self.acceptor.clone();
            let app = app.clone();
            tokio::spawn(async move {
                let stream = match acceptor.accept(stream).await {
                    Ok(stream) => stream,
                    Err(err) => {
                        eprintln!("[serve] TLS handshake from {peer} failed: {err}");
                        return;
                    }
                };
                let service = TowerToHyperService::new(app);
                if let Err(err) = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .with_upgrades()
                    .await
                {
                    eprintln!("[serve] HTTPS connection from {peer} failed: {err}");
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_paths_must_be_supplied_as_a_pair() {
        assert!(resolve_serve_tls("cert.pem", "", None).is_err());
        assert!(resolve_serve_tls("", "key.pem", None).is_err());
    }

    #[test]
    fn explicit_tls_paths_override_the_restart_fallback() {
        let fallback = ServeTlsConfig {
            cert: "old-cert.pem".into(),
            key: "old-key.pem".into(),
        };
        let resolved = resolve_serve_tls("new-cert.pem", "new-key.pem", Some(&fallback))
            .expect("valid TLS pair")
            .expect("TLS enabled");
        assert_eq!(resolved.cert, "new-cert.pem");
        assert_eq!(resolved.key, "new-key.pem");
        assert_eq!(
            resolve_serve_tls("", "", Some(&fallback)),
            Ok(Some(fallback))
        );
    }
}
