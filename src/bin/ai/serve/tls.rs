use std::{
    collections::BTreeSet,
    fs,
    io::Write as _,
    net::{IpAddr, SocketAddr, UdpSocket},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;

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

const TLS_DIR: &str = "tls";
const CA_CERT_FILE: &str = "ca.pem";
const CA_KEY_FILE: &str = "ca-key.pem";
const SERVER_CERT_FILE: &str = "server.pem";
const SERVER_KEY_FILE: &str = "server-key.pem";
const CERT_RENEWAL_WINDOW_SECONDS: &str = "2592000";

#[derive(Debug, Eq, PartialEq)]
pub(super) enum TlsInitStatus {
    Generated,
    Reused,
    Reissued,
}

#[derive(Debug)]
pub(super) struct TlsInitResult {
    pub(super) status: TlsInitStatus,
    pub(super) tls: ServeTlsConfig,
    pub(super) ca_cert: PathBuf,
    pub(super) fingerprint: String,
    pub(super) subject_alt_names: Vec<String>,
}

fn serve_state_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(dir) = std::env::var_os("A_SERVE_STATE_DIR") {
        if dir.is_empty() {
            return Err("A_SERVE_STATE_DIR is set but empty".into());
        }
        return Ok(PathBuf::from(dir));
    }
    Ok(crate::commonw::utils::get_config_dir()
        .ok_or("cannot resolve user config dir")?
        .join("rust_tools"))
}

fn default_tls_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(serve_state_dir()?.join(TLS_DIR))
}

pub(super) fn default_serve_tls_config()
-> Result<Option<ServeTlsConfig>, Box<dyn std::error::Error>> {
    default_serve_tls_config_in(&default_tls_dir()?)
}

fn default_serve_tls_config_in(
    dir: &Path,
) -> Result<Option<ServeTlsConfig>, Box<dyn std::error::Error>> {
    let cert = dir.join(SERVER_CERT_FILE);
    let key = dir.join(SERVER_KEY_FILE);
    match (cert.is_file(), key.is_file()) {
        (false, false) => Ok(None),
        (true, true) => Ok(Some(ServeTlsConfig {
            cert: cert.to_string_lossy().into_owned(),
            key: key.to_string_lossy().into_owned(),
        })),
        _ => Err(format!(
            "incomplete managed TLS files under {}; run `a --serve-tls-init` to repair them",
            dir.display()
        )
        .into()),
    }
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

fn bind_host(bind: &str) -> Result<String, Box<dyn std::error::Error>> {
    let authority = bind
        .trim()
        .strip_prefix("https://")
        .or_else(|| bind.trim().strip_prefix("http://"))
        .unwrap_or(bind.trim())
        .split('/')
        .next()
        .unwrap_or_default();
    if authority.is_empty() {
        return Err("serve bind is empty".into());
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']').ok_or("invalid bracketed IPv6 serve bind")?;
        return Ok(rest[..end].to_string());
    }
    let host = authority
        .rsplit_once(':')
        .filter(|(_, port)| port.parse::<u16>().is_ok())
        .map(|(host, _)| host)
        .unwrap_or(authority);
    if host.trim().is_empty() {
        return Err("serve bind has no host".into());
    }
    Ok(host.trim().to_string())
}

fn primary_local_ip() -> Option<IpAddr> {
    for (bind, peer) in [
        ("0.0.0.0:0", "1.1.1.1:80"),
        ("[::]:0", "[2606:4700:4700::1111]:80"),
    ] {
        let Ok(socket) = UdpSocket::bind(bind) else {
            continue;
        };
        if socket.connect(peer).is_ok()
            && let Ok(addr) = socket.local_addr()
            && !addr.ip().is_unspecified()
            && !addr.ip().is_loopback()
        {
            return Some(addr.ip());
        }
    }
    None
}

fn subject_alt_names(bind: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let host = bind_host(bind)?;
    let mut names = BTreeSet::from([
        "DNS:localhost".to_string(),
        "IP:127.0.0.1".to_string(),
        "IP:::1".to_string(),
    ]);
    match host.parse::<IpAddr>() {
        Ok(ip) if ip.is_unspecified() => {
            let local = primary_local_ip().ok_or(
                "cannot discover a reachable local IP for a wildcard bind; pass --serve-bind <ip>:<port>",
            )?;
            names.insert(format!("IP:{local}"));
        }
        Ok(ip) => {
            names.insert(format!("IP:{ip}"));
        }
        Err(_) => {
            if !host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
            {
                return Err(format!("unsupported serve bind host for TLS: {host}").into());
            }
            names.insert(format!("DNS:{host}"));
        }
    }
    Ok(names.into_iter().collect())
}

fn run_openssl(args: &[&str]) -> Result<Output, Box<dyn std::error::Error>> {
    let output = Command::new("openssl").args(args).output().map_err(|err| {
        format!("failed to run openssl ({err}); install OpenSSL and retry `a --serve-tls-init`")
    })?;
    Ok(output)
}

fn openssl_ok(args: &[&str]) -> bool {
    run_openssl(args)
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn require_openssl(args: &[&str]) -> Result<Output, Box<dyn std::error::Error>> {
    let output = run_openssl(args)?;
    if output.status.success() {
        return Ok(output);
    }
    let message = String::from_utf8_lossy(&output.stderr);
    Err(format!("openssl {} failed: {}", args.join(" "), message.trim()).into())
}

fn create_stage_dir(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(dir)?;
    set_mode(dir, 0o700)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let stage = dir.join(format!(".init-{}-{nonce}", std::process::id()));
    fs::create_dir(&stage)?;
    set_mode(&stage, 0o700)?;
    Ok(stage)
}

fn set_mode(path: &Path, mode: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    #[cfg(not(unix))]
    let _ = (path, mode);
    Ok(())
}

fn publish_file(src: &Path, dst: &Path, mode: u32) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(src)?;
    let name = dst
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("invalid TLS file name")?;
    let tmp = dst.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    set_mode(&tmp, mode)?;
    if let Err(err) = fs::rename(&tmp, dst) {
        let _ = fs::remove_file(&tmp);
        return Err(err.into());
    }
    Ok(())
}

fn path_arg(path: &Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.to_str()
        .ok_or_else(|| format!("TLS path is not valid UTF-8: {}", path.display()).into())
}

fn certificate_matches_key(cert: &Path, key: &Path) -> bool {
    let Ok(cert) = run_openssl(&[
        "x509",
        "-in",
        path_arg(cert).unwrap_or(""),
        "-pubkey",
        "-noout",
    ]) else {
        return false;
    };
    let Ok(key) = run_openssl(&["pkey", "-in", path_arg(key).unwrap_or(""), "-pubout"]) else {
        return false;
    };
    cert.status.success() && key.status.success() && cert.stdout == key.stdout
}

fn validate_ca(cert: &Path, key: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !certificate_matches_key(cert, key) {
        return Err(
            "existing ca.pem and ca-key.pem do not form a key pair; refusing to replace the CA"
                .into(),
        );
    }
    let check = run_openssl(&[
        "x509",
        "-in",
        path_arg(cert)?,
        "-checkend",
        CERT_RENEWAL_WINDOW_SECONDS,
        "-noout",
    ])?;
    if !check.status.success() {
        return Err("existing serve CA is invalid or expires within 30 days; rotate it explicitly instead of silently replacing client trust".into());
    }
    Ok(())
}

fn generate_ca(stage: &Path) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let key = stage.join(CA_KEY_FILE);
    let cert = stage.join(CA_CERT_FILE);
    require_openssl(&[
        "genpkey",
        "-algorithm",
        "RSA",
        "-pkeyopt",
        "rsa_keygen_bits:4096",
        "-out",
        path_arg(&key)?,
    ])?;
    require_openssl(&[
        "req",
        "-x509",
        "-new",
        "-key",
        path_arg(&key)?,
        "-sha256",
        "-days",
        "3650",
        "-out",
        path_arg(&cert)?,
        "-subj",
        "/CN=a serve local CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
        "-addext",
        "subjectKeyIdentifier=hash",
    ])?;
    validate_ca(&cert, &key)?;
    Ok((cert, key))
}

/// A single subject alternative name extracted from a certificate's SAN
/// extension. DNS names are compared as strings; IP addresses are compared
/// after parsing so equivalent textual forms (e.g. `::1` vs
/// `0:0:0:0:0:0:0:1`) still match.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum SanEntry {
    Dns(String),
    Ip(IpAddr),
}

fn collect_san_entries(line: &str, entries: &mut BTreeSet<SanEntry>) {
    for token in line.split(',') {
        let token = token.trim();
        if let Some(name) = token.strip_prefix("DNS:") {
            if !name.is_empty() {
                entries.insert(SanEntry::Dns(name.to_string()));
            }
        } else if let Some(addr) = token.strip_prefix("IP Address:") {
            if let Ok(ip) = addr.parse::<IpAddr>() {
                entries.insert(SanEntry::Ip(ip));
            }
        }
    }
}

/// Read the `X509v3 Subject Alternative Name` extension from the
/// human-readable `openssl x509 -noout -text` output. Both LibreSSL and
/// OpenSSL 1.1/3 render it as a comma-separated `DNS:name` / `IP Address:addr`
/// list, so this avoids the `-checkip`/`-checkhost` options that only OpenSSL
/// supports.
fn parse_certificate_san(cert: &Path) -> Result<BTreeSet<SanEntry>, Box<dyn std::error::Error>> {
    let output = require_openssl(&["x509", "-in", path_arg(cert)?, "-noout", "-text"])?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut entries = BTreeSet::new();
    let mut in_san = false;
    for line in text.lines() {
        if in_san {
            let trimmed = line.trim();
            if trimmed == "critical" {
                continue;
            }
            if line.starts_with(char::is_whitespace)
                && (line.contains("DNS:") || line.contains("IP Address:"))
            {
                collect_san_entries(line, &mut entries);
                continue;
            }
            in_san = false;
        }
        if let Some((_, rest)) = line.split_once("Subject Alternative Name:") {
            collect_san_entries(rest, &mut entries);
            in_san = true;
        }
    }
    if entries.is_empty() {
        return Err("certificate has no parseable X509v3 Subject Alternative Name entries".into());
    }
    Ok(entries)
}

/// True when every expected name in `names` really appears in the
/// certificate's SAN extension. A certificate whose SAN cannot be read (no
/// extension, corrupt output, openssl failure) is treated as not current, so
/// the caller reissues it instead of failing hard; a missing name likewise
/// means the certificate must be reissued, never silently reused.
fn certificate_san_matches(
    cert: &Path,
    names: &[String],
) -> Result<bool, Box<dyn std::error::Error>> {
    let Ok(entries) = parse_certificate_san(cert) else {
        return Ok(false);
    };
    for name in names {
        let (kind, value) = name
            .split_once(':')
            .ok_or("invalid generated subjectAltName")?;
        let found = match kind {
            "IP" => value
                .parse::<IpAddr>()
                .map(|ip| entries.contains(&SanEntry::Ip(ip)))
                .unwrap_or(false),
            _ => entries.contains(&SanEntry::Dns(value.to_string())),
        };
        if !found {
            return Ok(false);
        }
    }
    Ok(true)
}

fn server_certificate_is_current(
    cert: &Path,
    key: &Path,
    ca_cert: &Path,
    names: &[String],
) -> Result<bool, Box<dyn std::error::Error>> {
    if !cert.is_file() || !key.is_file() || !certificate_matches_key(cert, key) {
        return Ok(false);
    }
    // Chain validity and the renewal window must not be relaxed.
    if !openssl_ok(&["verify", "-CAfile", path_arg(ca_cert)?, path_arg(cert)?])
        || !openssl_ok(&[
            "x509",
            "-in",
            path_arg(cert)?,
            "-checkend",
            CERT_RENEWAL_WINDOW_SECONDS,
            "-noout",
        ])
    {
        return Ok(false);
    }
    certificate_san_matches(cert, names)
}

fn generate_server_certificate(
    stage: &Path,
    ca_cert: &Path,
    ca_key: &Path,
    names: &[String],
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let key = stage.join(SERVER_KEY_FILE);
    let csr = stage.join("server.csr");
    let leaf = stage.join("server-leaf.pem");
    let chain = stage.join(SERVER_CERT_FILE);
    let extensions = stage.join("server-ext.cnf");
    require_openssl(&[
        "genpkey",
        "-algorithm",
        "RSA",
        "-pkeyopt",
        "rsa_keygen_bits:3072",
        "-out",
        path_arg(&key)?,
    ])?;
    require_openssl(&[
        "req",
        "-new",
        "-key",
        path_arg(&key)?,
        "-out",
        path_arg(&csr)?,
        "-subj",
        "/CN=a serve",
    ])?;
    fs::write(
        &extensions,
        format!(
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName={}\n",
            names.join(",")
        ),
    )?;
    let serial = format!(
        "0x{:x}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    require_openssl(&[
        "x509",
        "-req",
        "-in",
        path_arg(&csr)?,
        "-CA",
        path_arg(ca_cert)?,
        "-CAkey",
        path_arg(ca_key)?,
        "-set_serial",
        &serial,
        "-out",
        path_arg(&leaf)?,
        "-days",
        "825",
        "-sha256",
        "-extfile",
        path_arg(&extensions)?,
    ])?;
    let mut chain_bytes = fs::read(&leaf)?;
    if !chain_bytes.ends_with(b"\n") {
        chain_bytes.push(b'\n');
    }
    chain_bytes.extend_from_slice(&fs::read(ca_cert)?);
    fs::write(&chain, chain_bytes)?;
    if !server_certificate_is_current(&chain, &key, ca_cert, names)? {
        return Err("generated serve certificate failed validation".into());
    }
    Ok((chain, key))
}

fn ca_fingerprint(cert: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = require_openssl(&[
        "x509",
        "-in",
        path_arg(cert)?,
        "-noout",
        "-fingerprint",
        "-sha256",
    ])?;
    let line = String::from_utf8(output.stdout)?;
    Ok(line
        .trim()
        .split_once('=')
        .map(|(_, value)| value.trim())
        .unwrap_or(line.trim())
        .to_string())
}

pub(super) fn init_serve_tls(bind: &str) -> Result<TlsInitResult, Box<dyn std::error::Error>> {
    init_serve_tls_in(&default_tls_dir()?, bind)
}

fn init_serve_tls_in(dir: &Path, bind: &str) -> Result<TlsInitResult, Box<dyn std::error::Error>> {
    require_openssl(&["version"])?;
    let names = subject_alt_names(bind)?;
    fs::create_dir_all(dir)?;
    set_mode(dir, 0o700)?;

    let ca_cert = dir.join(CA_CERT_FILE);
    let ca_key = dir.join(CA_KEY_FILE);
    match (ca_cert.is_file(), ca_key.is_file()) {
        (true, true) => validate_ca(&ca_cert, &ca_key)?,
        (false, false) => {
            let stage = create_stage_dir(dir)?;
            let generated = generate_ca(&stage);
            if let Ok((cert, key)) = generated.as_ref() {
                publish_file(key, &ca_key, 0o600)?;
                publish_file(cert, &ca_cert, 0o644)?;
            }
            let _ = fs::remove_dir_all(&stage);
            generated?;
        }
        _ => {
            return Err(format!(
                "incomplete CA files under {}; restore the missing file because the CA will never be replaced automatically",
                dir.display()
            )
            .into());
        }
    }
    set_mode(&ca_key, 0o600)?;
    validate_ca(&ca_cert, &ca_key)?;

    let server_cert = dir.join(SERVER_CERT_FILE);
    let server_key = dir.join(SERVER_KEY_FILE);
    let server_existed = server_cert.is_file() && server_key.is_file();
    let status = if server_certificate_is_current(&server_cert, &server_key, &ca_cert, &names)? {
        TlsInitStatus::Reused
    } else {
        let stage = create_stage_dir(dir)?;
        let generated = generate_server_certificate(&stage, &ca_cert, &ca_key, &names);
        if let Ok((cert, key)) = generated.as_ref() {
            publish_file(key, &server_key, 0o600)?;
            publish_file(cert, &server_cert, 0o644)?;
        }
        let _ = fs::remove_dir_all(&stage);
        generated?;
        if server_existed {
            TlsInitStatus::Reissued
        } else {
            TlsInitStatus::Generated
        }
    };
    set_mode(&server_key, 0o600)?;

    Ok(TlsInitResult {
        status,
        tls: default_serve_tls_config_in(dir)?.ok_or("managed TLS files were not published")?,
        ca_cert: ca_cert.clone(),
        fingerprint: ca_fingerprint(&ca_cert)?,
        subject_alt_names: names,
    })
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

    #[test]
    fn tls_init_is_idempotent_and_reissues_only_the_server_certificate() {
        if !openssl_ok(&["version"]) {
            eprintln!("skipping TLS init test because openssl is unavailable");
            return;
        }
        let dir = std::env::temp_dir().join(format!("a-serve-tls-{}", uuid::Uuid::new_v4()));
        let first = init_serve_tls_in(&dir, "192.0.2.10:8080").expect("initial TLS generation");
        assert_eq!(first.status, TlsInitStatus::Generated);
        let ca_before = fs::read(dir.join(CA_CERT_FILE)).expect("read CA");
        let server_before = fs::read(dir.join(SERVER_CERT_FILE)).expect("read server certificate");

        let second = init_serve_tls_in(&dir, "192.0.2.10:8080").expect("idempotent TLS init");
        assert_eq!(second.status, TlsInitStatus::Reused);
        assert_eq!(fs::read(dir.join(CA_CERT_FILE)).unwrap(), ca_before);
        assert_eq!(fs::read(dir.join(SERVER_CERT_FILE)).unwrap(), server_before);

        let third = init_serve_tls_in(&dir, "192.0.2.11:8080").expect("server reissue");
        assert_eq!(third.status, TlsInitStatus::Reissued);
        assert_eq!(fs::read(dir.join(CA_CERT_FILE)).unwrap(), ca_before);
        assert_ne!(fs::read(dir.join(SERVER_CERT_FILE)).unwrap(), server_before);
        assert!(
            third
                .subject_alt_names
                .contains(&"IP:192.0.2.11".to_string())
        );

        #[cfg(unix)]
        assert_eq!(
            fs::metadata(dir.join(SERVER_KEY_FILE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tls_init_rejects_an_incomplete_ca_without_replacing_it() {
        let dir = std::env::temp_dir().join(format!("a-serve-tls-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(CA_CERT_FILE), b"do not replace").unwrap();
        let err = init_serve_tls_in(&dir, "192.0.2.10:8080").unwrap_err();
        assert!(err.to_string().contains("incomplete CA files"));
        assert_eq!(fs::read(dir.join(CA_CERT_FILE)).unwrap(), b"do not replace");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn san_matching_rejects_a_missing_expected_ip() {
        if !openssl_ok(&["version"]) {
            eprintln!("skipping SAN matching test because openssl is unavailable");
            return;
        }
        let dir = std::env::temp_dir().join(format!("a-serve-tls-san-{}", uuid::Uuid::new_v4()));
        let stage = create_stage_dir(&dir).expect("stage dir");
        let (ca_cert, ca_key) = generate_ca(&stage).expect("CA generation");
        let names = vec![
            "DNS:localhost".to_string(),
            "IP:127.0.0.1".to_string(),
            "IP:192.0.2.10".to_string(),
            "IP:::1".to_string(),
        ];
        let (cert, _key) =
            generate_server_certificate(&stage, &ca_cert, &ca_key, &names).expect("server cert");
        assert!(
            certificate_san_matches(&cert, &names).expect("SAN check succeeds for present names")
        );
        let missing = vec!["IP:192.0.2.11".to_string()];
        assert!(
            !certificate_san_matches(&cert, &missing).expect("SAN check runs for an absent IP")
        );
        let missing_dns = vec!["DNS:notlocalhost".to_string()];
        assert!(
            !certificate_san_matches(&cert, &missing_dns)
                .expect("SAN check runs for an absent DNS")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn damaged_server_certificate_is_reissued_instead_of_erroring() {
        if !openssl_ok(&["version"]) {
            eprintln!("skipping damaged certificate test because openssl is unavailable");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("a-serve-tls-damaged-{}", uuid::Uuid::new_v4()));
        let first = init_serve_tls_in(&dir, "192.0.2.10:8080").expect("initial TLS generation");
        assert_eq!(first.status, TlsInitStatus::Generated);
        let ca_before = fs::read(dir.join(CA_CERT_FILE)).expect("read CA");

        // Replace the server certificate with a CA-signed leaf that carries no
        // SAN extension at all: init must reissue it, not error out.
        let stage = create_stage_dir(&dir).expect("stage dir");
        let key = stage.join("nosan-key.pem");
        let csr = stage.join("nosan.csr");
        let nosan = stage.join("nosan.pem");
        require_openssl(&[
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            path_arg(&key).expect("key path"),
        ])
        .expect("SAN-less key generation");
        require_openssl(&[
            "req",
            "-new",
            "-key",
            path_arg(&key).expect("key path"),
            "-out",
            path_arg(&csr).expect("CSR path"),
            "-subj",
            "/CN=a serve",
        ])
        .expect("SAN-less CSR generation");
        // No -extfile: the leaf gets no extensions, hence no SAN.
        require_openssl(&[
            "x509",
            "-req",
            "-in",
            path_arg(&csr).expect("CSR path"),
            "-CA",
            path_arg(&dir.join(CA_CERT_FILE)).expect("CA path"),
            "-CAkey",
            path_arg(&dir.join(CA_KEY_FILE)).expect("CA key path"),
            "-set_serial",
            "0xfeed",
            "-out",
            path_arg(&nosan).expect("leaf path"),
            "-days",
            "825",
            "-sha256",
        ])
        .expect("SAN-less leaf generation");
        fs::copy(&key, dir.join(SERVER_KEY_FILE)).expect("install SAN-less key");
        fs::copy(&nosan, dir.join(SERVER_CERT_FILE)).expect("install SAN-less certificate");
        let _ = fs::remove_dir_all(&stage);

        let reissued =
            init_serve_tls_in(&dir, "192.0.2.10:8080").expect("reissue a SAN-less certificate");
        assert_eq!(reissued.status, TlsInitStatus::Reissued);
        assert_eq!(fs::read(dir.join(CA_CERT_FILE)).unwrap(), ca_before);

        // A corrupt certificate must also trigger a reissue instead of an error.
        fs::write(dir.join(SERVER_CERT_FILE), b"not a certificate").expect("corrupt certificate");
        let reissued =
            init_serve_tls_in(&dir, "192.0.2.10:8080").expect("reissue a corrupt certificate");
        assert_eq!(reissued.status, TlsInitStatus::Reissued);
        assert_eq!(fs::read(dir.join(CA_CERT_FILE)).unwrap(), ca_before);
        let _ = fs::remove_dir_all(dir);
    }
}
