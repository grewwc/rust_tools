//! Live TraeCLI worker-model discovery for command completion.

use std::{
    io::Read,
    process::{Command, Stdio},
    sync::{LazyLock, Mutex, mpsc},
    time::{Duration, Instant},
};

use rust_tools::commonw::FastSet;

const CACHE_TTL: Duration = Duration::from_secs(60);
/// Negative-cache TTL for failed/empty discoveries. A transient spawn failure
/// or timeout must not blank completion for the full success TTL, while
/// back-to-back Tab presses still must not respawn a missing CLI every time.
const FAILURE_CACHE_TTL: Duration = Duration::from_secs(5);
const DISCOVERY_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_OUTPUT_BYTES: u64 = 128 * 1024;

#[derive(Default)]
struct ModelCache {
    binary: String,
    refreshed_at: Option<Instant>,
    names: Vec<String>,
}

impl ModelCache {
    fn get_or_refresh(
        &mut self,
        binary: &str,
        now: Instant,
        discover: impl FnOnce() -> Vec<String>,
    ) -> Vec<String> {
        // Empty results are failures (spawn error, timeout, unparsable output):
        // re-probe after a short TTL instead of the full success TTL.
        let expired = match self.refreshed_at {
            None => true,
            Some(at) => {
                let ttl = if self.names.is_empty() {
                    FAILURE_CACHE_TTL
                } else {
                    CACHE_TTL
                };
                now.duration_since(at) >= ttl
            }
        };
        if self.binary != binary || expired {
            self.names = discover();
            self.binary = binary.to_string();
            self.refreshed_at = Some(now);
        }
        self.names.clone()
    }
}

static MODEL_CACHE: LazyLock<Mutex<ModelCache>> =
    LazyLock::new(|| Mutex::new(ModelCache::default()));

pub(crate) fn traecli_model_names() -> Vec<String> {
    #[cfg(test)]
    if let Some(names) = TEST_MODEL_NAMES.with(|names| names.borrow().clone()) {
        return names;
    }

    // A concurrent lookup must never wait for another completion's subprocess.
    let Ok(mut cache) = MODEL_CACHE.try_lock() else {
        return Vec::new();
    };
    let binary = super::traecli::traecli_binary();
    cache.get_or_refresh(&binary, Instant::now(), || discover_model_names(&binary))
}

/// The installed `traecli models` prints one case-sensitive identifier per
/// stdout line. Reject prose/diagnostics and deduplicate without changing order.
fn parse_model_names(output: &str) -> Vec<String> {
    let mut seen = FastSet::default();
    output
        .lines()
        .map(str::trim)
        .filter(|name| {
            name.bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:/".contains(&byte))
        })
        .filter(|name| seen.insert((*name).to_string()))
        .map(str::to_string)
        .collect()
}

/// Bound both runtime and stdout size. A hung CLI or inherited stdout pipe
/// must not hang the editor; failures yield no candidates.
fn discover_model_names(binary: &str) -> Vec<String> {
    let Ok(mut child) = Command::new(binary)
        .arg("models")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return Vec::new();
    };
    let deadline = Instant::now() + DISCOVERY_TIMEOUT;
    let result = (|| {
        let stdout = child.stdout.take()?;
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("traecli-models-output".to_string())
            .spawn(move || {
                let mut output = Vec::new();
                let result = stdout.take(MAX_OUTPUT_BYTES + 1).read_to_end(&mut output);
                let _ = sender.send(result.map(|_| output));
            })
            .ok()?;
        loop {
            if let Some(status) = child.try_wait().ok()? {
                if !status.success() {
                    return None;
                }
                let output = receiver
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                    .ok()?
                    .ok()?;
                if output.len() as u64 > MAX_OUTPUT_BYTES {
                    return None;
                }
                return String::from_utf8(output)
                    .ok()
                    .map(|output| parse_model_names(&output));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(10)));
        }
    })();
    if result.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result.unwrap_or_default()
}

#[cfg(test)]
thread_local! {
    static TEST_MODEL_NAMES: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_traecli_model_names_for_test<T>(
    names: Vec<String>,
    test: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<Vec<String>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_MODEL_NAMES.with(|names| *names.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(TEST_MODEL_NAMES.with(|value| value.replace(Some(names))));
    test()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traecli_model_list_parses_live_line_format() {
        let output = "Seed-Evolving\nSeed-2.1-Pro-0915\nDeepSeek-V4.1-Flash\nGPT-6.1-Sol\nGPT-5.5\nGemini-3.1-Pro-Preview\n";
        assert_eq!(
            parse_model_names(output),
            [
                "Seed-Evolving",
                "Seed-2.1-Pro-0915",
                "DeepSeek-V4.1-Flash",
                "GPT-6.1-Sol",
                "GPT-5.5",
                "Gemini-3.1-Pro-Preview",
            ]
        );
    }

    #[test]
    fn traecli_model_list_ignores_blanks_duplicates_and_diagnostics() {
        assert_eq!(
            parse_model_names(
                "\n GPT-5.5\r\nGPT-5.5\nWARNING: failed to load\n# Models\nGemini-3-Flash-Preview\n"
            ),
            ["GPT-5.5", "Gemini-3-Flash-Preview"]
        );
        assert!(parse_model_names("Error: model listing failed\n").is_empty());
    }

    #[test]
    fn traecli_model_list_caches_success_until_expiry_and_binary_changes() {
        let now = Instant::now();
        let mut cache = ModelCache::default();
        assert_eq!(
            cache.get_or_refresh("traecli", now, || vec!["M1".to_string()]),
            ["M1"]
        );
        assert_eq!(
            cache.get_or_refresh("traecli", now + Duration::from_secs(59), || panic!(
                "cached lookup must not run a process"
            )),
            ["M1"]
        );
        assert_eq!(
            cache.get_or_refresh("traecli", now + CACHE_TTL, || vec!["M2".to_string()]),
            ["M2"]
        );
        assert_eq!(
            cache.get_or_refresh("/other/traecli", now + CACHE_TTL, || vec!["M3".to_string()]),
            ["M3"]
        );
    }

    #[test]
    fn traecli_model_list_caches_failures_and_absent_binary_is_empty() {
        let now = Instant::now();
        let mut cache = ModelCache::default();
        assert!(cache.get_or_refresh("absent", now, Vec::new).is_empty());
        assert!(
            cache
                .get_or_refresh("absent", now + Duration::from_secs(1), || panic!(
                    "failed lookup must be cached"
                ))
                .is_empty()
        );
        // A transient failure must recover quickly, not linger for the full
        // success TTL: past the short failure TTL the next lookup re-probes.
        assert_eq!(
            cache.get_or_refresh("absent", now + FAILURE_CACHE_TTL, || vec![
                "M-recovered".to_string()
            ]),
            ["M-recovered"]
        );
        assert!(discover_model_names("/definitely/not/a/traecli-binary").is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn traecli_model_list_subprocess_success_failure_and_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let root =
            std::env::temp_dir().join(format!("traecli-model-list-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let binary = root.join("fake traecli");
        std::fs::write(&binary, "#!/bin/sh\nprintf 'GPT-5.5\\nGPT-6.1-Sol\\n'\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            discover_model_names(binary.to_str().unwrap()),
            ["GPT-5.5", "GPT-6.1-Sol"]
        );

        std::fs::write(&binary, "#!/bin/sh\nprintf 'partial-model\\n'\nexit 1\n").unwrap();
        assert!(discover_model_names(binary.to_str().unwrap()).is_empty());
        std::fs::write(&binary, "#!/bin/sh\nexec sleep 5\n").unwrap();
        let start = Instant::now();
        assert!(discover_model_names(binary.to_str().unwrap()).is_empty());
        assert!(start.elapsed() < Duration::from_secs(2));
        let _ = std::fs::remove_dir_all(root);
    }
}
