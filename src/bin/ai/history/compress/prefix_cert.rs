//! Versioned, snapshot-persistable evidence of a B1 identity projection.
//! The digest binds the admitted prefix, while the original B1 extension proof
//! decides whether a canonical tail can reuse it without changing any bytes.

use std::fmt::Write;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::prefix_reuse::{
    CACHE, CACHE_MAX_BYTES, CACHE_MAX_MESSAGES, PrefixReuseCache, PrefixReuseEntry, PrefixReuseKey,
    is_plain_tail_message, projection_bytes, retention_boundaries,
};
use super::{
    ContextCompressionOutcome, ContextCompressionStatus, Message, Path, PathBuf,
    messages_total_chars, retained_turn_start, user_input_turn_count,
};

/// Increment when B1 admission, extension, or retention-window rules change.
pub(in crate::ai) const PROOF_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct PrefixCertificatePolicy {
    pub(in crate::ai) max_chars: usize,
    pub(in crate::ai) keep_last: usize,
    pub(in crate::ai) summary_max_chars: usize,
    pub(in crate::ai) overflow_dir: Option<String>,
    pub(in crate::ai) cwd: Option<String>,
}

impl PrefixCertificatePolicy {
    /// Normalize only path components, matching B1's lexical Path equality.
    /// Parent components stay intact; no filesystem lookup or lossy conversion
    /// can change the cache key's meaning.
    pub(in crate::ai) fn new(
        max_chars: usize,
        keep_last: usize,
        summary_max_chars: usize,
        overflow_dir: Option<&Path>,
        cwd: Option<&Path>,
    ) -> Option<Self> {
        Some(Self {
            max_chars,
            keep_last,
            summary_max_chars,
            overflow_dir: match overflow_dir {
                Some(path) => Some(normalized_path(path)?),
                None => None,
            },
            cwd: match cwd {
                Some(path) => Some(normalized_path(path)?),
                None => None,
            },
        })
    }

    fn reuse_key(&self) -> Option<PrefixReuseKey> {
        for path in [&self.overflow_dir, &self.cwd].into_iter().flatten() {
            if normalized_path(Path::new(path))?.as_str() != path {
                return None;
            }
        }
        Some(PrefixReuseKey::new(
            self.max_chars,
            self.keep_last,
            self.summary_max_chars,
            self.overflow_dir.as_deref().map(Path::new),
            self.cwd.as_deref().map(Path::new),
        ))
    }

    fn prefix_bytes(&self, messages: &[Message]) -> Option<usize> {
        projection_bytes(messages)?
            .checked_add(
                self.overflow_dir
                    .as_deref()
                    .map_or(0, |path| Path::new(path).as_os_str().len()),
            )?
            .checked_add(
                self.cwd
                    .as_deref()
                    .map_or(0, |path| Path::new(path).as_os_str().len()),
            )
    }

    fn older_recent_split(&self, messages: &[Message]) -> Option<usize> {
        (self.keep_last > 0)
            .then(|| retained_turn_start(messages, self.keep_last.min(messages.len())))
    }
}

fn normalized_path(path: &Path) -> Option<String> {
    path.components()
        .collect::<PathBuf>()
        .to_str()
        .map(str::to_owned)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct PrefixCertificate {
    pub(in crate::ai) proof_version: u32,
    pub(in crate::ai) policy: PrefixCertificatePolicy,
    pub(in crate::ai) prefix_message_count: usize,
    pub(in crate::ai) prefix_sha256: String,
    pub(in crate::ai) chars: usize,
    /// B1's bounded projection-memory estimate, including both cache-key paths.
    pub(in crate::ai) bytes: usize,
    pub(in crate::ai) real_user_turns: usize,
    pub(in crate::ai) trim_start: usize,
    pub(in crate::ai) merge_start: usize,
    /// The older/recent split is explicitly null when keep_last is disabled.
    pub(in crate::ai) older_recent_split: Option<usize>,
}

/// Certify only the exact current entry of the process-local B1 cache.
pub(in crate::ai) fn prefix_certificate_from_cache(
    messages: &[Message],
    policy: &PrefixCertificatePolicy,
) -> Option<PrefixCertificate> {
    let key = policy.reuse_key()?;
    let (chars, bytes, real_user_turns, boundaries) = {
        let mut cache = CACHE.lock().ok()?;
        let entry = cache.entry.as_ref()?;
        if entry.key != key || entry.messages != messages {
            return None;
        }
        let metrics = (entry.chars, entry.bytes, entry.user_turns, entry.boundaries);
        // Even an admitted entry may have expired guidance. An empty extension
        // checks all B1 reuse gates while preserving the cached prefix exactly.
        if cache.try_reuse(messages, &key)? != metrics.0 {
            return None;
        }
        metrics
    };

    if chars != messages_total_chars(messages)
        || bytes != policy.prefix_bytes(messages)?
        || real_user_turns != user_input_turn_count(messages)
        || boundaries != retention_boundaries(messages, chars)
    {
        // Noncanonical path spelling can compare equal as a B1 key but have a
        // different memory estimate. Do not manufacture a replacement estimate.
        return None;
    }
    Some(PrefixCertificate {
        proof_version: PROOF_VERSION,
        policy: policy.clone(),
        prefix_message_count: messages.len(),
        prefix_sha256: prefix_sha256(messages)?,
        chars,
        bytes,
        real_user_turns,
        trim_start: boundaries.0,
        merge_start: boundaries.1,
        older_recent_split: policy.older_recent_split(messages),
    })
}

pub(in crate::ai) fn prefix_certificate_json_from_cache(
    messages: &[Message],
    policy: &PrefixCertificatePolicy,
) -> Option<String> {
    serde_json::to_string(&prefix_certificate_from_cache(messages, policy)?).ok()
}

impl PrefixCertificate {
    pub(in crate::ai) fn validate(
        &self,
        prefix: &[Message],
        tail: &[Message],
        policy: &PrefixCertificatePolicy,
    ) -> bool {
        self.validated_projection(prefix, tail, policy).is_some()
    }

    fn validated_projection(
        &self,
        prefix: &[Message],
        tail: &[Message],
        policy: &PrefixCertificatePolicy,
    ) -> Option<(Vec<Message>, usize)> {
        let key = policy.reuse_key()?;
        let message_count = prefix.len().checked_add(tail.len())?;
        if self.proof_version != PROOF_VERSION
            || self.policy != *policy
            || prefix.is_empty()
            || self.prefix_message_count != prefix.len()
            || message_count > CACHE_MAX_MESSAGES
            || self.chars != messages_total_chars(prefix)
            || self.bytes != policy.prefix_bytes(prefix)?
            || self.bytes > CACHE_MAX_BYTES
            || self.real_user_turns != user_input_turn_count(prefix)
            || (self.trim_start, self.merge_start) != retention_boundaries(prefix, self.chars)
            || self.older_recent_split != policy.older_recent_split(prefix)
            || self.prefix_sha256 != prefix_sha256(prefix)?
            || !tail.iter().all(is_plain_tail_message)
            || self.bytes.checked_add(projection_bytes(tail)?)? > CACHE_MAX_BYTES
        {
            return None;
        }

        let mut messages = Vec::with_capacity(message_count);
        messages.extend_from_slice(prefix);
        messages.extend_from_slice(tail);
        if self.older_recent_split != policy.older_recent_split(&messages) {
            return None;
        }

        // The versioned digest binds the prefix's completed admission proof,
        // including independent note windows and pure no-op passes. A local
        // entry reuses the ORIGINAL B1 extension proof: total budgets, real-user
        // limits, both retention boundaries, and full-input guidance filtering.
        // Its extension is disposable; neither the certificate nor CACHE moves.
        let mut cache = PrefixReuseCache {
            entry: Some(PrefixReuseEntry {
                key: policy.reuse_key()?,
                messages: prefix.to_vec(),
                chars: self.chars,
                bytes: self.bytes,
                user_turns: self.real_user_turns,
                boundaries: (self.trim_start, self.merge_start),
            }),
        };
        let chars = cache.try_reuse(&messages, &key)?;
        Some((messages, chars))
    }
}

/// A pure snapshot-prefix/canonical-tail gate; malformed evidence fails closed.
pub(in crate::ai) fn validate_certificate_json(
    json: &str,
    prefix: &[Message],
    tail: &[Message],
    policy: &PrefixCertificatePolicy,
) -> bool {
    serde_json::from_str::<PrefixCertificate>(json)
        .ok()
        .is_some_and(|certificate| certificate.validate(prefix, tail, policy))
}

/// Return the identity outcome only after the same pure certificate gate passes.
pub(in crate::ai) fn try_reuse_certificate_json(
    json: &str,
    prefix: &[Message],
    tail: &[Message],
    policy: &PrefixCertificatePolicy,
) -> Option<ContextCompressionOutcome> {
    let certificate = serde_json::from_str::<PrefixCertificate>(json).ok()?;
    let (messages, chars) = certificate.validated_projection(prefix, tail, policy)?;
    Some(ContextCompressionOutcome {
        messages,
        status: ContextCompressionStatus::Complete,
        before_chars: chars,
        after_chars: chars,
        max_chars: policy.max_chars,
    })
}

fn prefix_sha256(messages: &[Message]) -> Option<String> {
    let mut digest = Sha256::new();
    for message in messages {
        digest.update(serde_json::to_vec(message).ok()?);
    }
    let mut hex = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(hex, "{byte:02x}").ok()?;
    }
    Some(hex)
}
