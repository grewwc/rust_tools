//! Bounded, version-checked read cache for the overflow archive.
//!
//! `search_overflow` reads every candidate archive file once during scanning
//! (pass A) and re-reads only the selected files while rendering excerpts.
//! That second read used to hit the filesystem again; this cache serves both
//! reads from memory when the file is unchanged.
//!
//! Freshness is enforced by a caller-supplied [`FileVersion`] that the caller
//! observed on the SAME open file descriptor it read the body from. The cache
//! never touches the filesystem itself, so there is no TOCTOU gap between
//! "check the file" and "serve the body": a body is only ever paired with the
//! version its reader observed on the descriptor it read. A version mismatch
//! is a miss and the stale entry is dropped, so an equal-length in-place
//! replacement is still detected through its changed mtime.
//!
//! The cache is bounded (entry count and total bytes) with LRU eviction and
//! never stores files larger than 1 MiB, so it cannot balloon resident
//! memory. It is a pure performance layer, never a scan or byte budget: the
//! search still scans every archive file on every call, and absence results
//! remain based on the full scan.
//!
//! Bodies are shared as `Arc<str>`, so a cache hit is a reference-count bump
//! instead of a full-text copy, and recency is a per-entry monotonic counter,
//! so the hot read path is O(1) with a tiny critical section under the global
//! lock (eviction scans for the oldest counter, but only on the cold insert
//! path when a cap is exceeded).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

use rustc_hash::FxHashMap;

/// Maximum number of cached files.
const MAX_ENTRIES: usize = 512;
/// Maximum total cached content, in bytes.
const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
/// Files larger than this are never cached.
const MAX_FILE_BYTES: usize = 1 * 1024 * 1024;

/// Identity of one file version, as observed on an open file descriptor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FileVersion {
    /// File length in bytes, observed on the descriptor.
    pub len: u64,
    /// File modification time, observed on the descriptor.
    pub mtime: SystemTime,
    /// Owning device, observed on the descriptor (zero off Unix).
    pub dev: u64,
    /// Inode number, observed on the descriptor (zero off Unix).
    pub ino: u64,
    /// Metadata change time, seconds since the Unix epoch (zero off Unix).
    /// Updating the mtime itself advances the change time, so an in-place
    /// rewrite that restores the mtime is still a version mismatch.
    pub ctime_sec: i64,
    /// Metadata change time, sub-second nanoseconds (zero off Unix).
    pub ctime_nsec: i64,
    /// Device of the session root the file was opened through (zero off
    /// Unix). Binds the entry to the session directory so a path that merely
    /// reuses the same spelling after the directory was replaced cannot hit.
    pub root_dev: u64,
    /// Inode of the session root the file was opened through (zero off Unix).
    pub root_ino: u64,
}

struct CacheEntry {
    /// Version the body was read at; the body is served only for this exact
    /// version.
    version: FileVersion,
    size: usize,
    /// Shared body: hits clone the `Arc`, never the text.
    content: Arc<str>,
    /// Last-use stamp from the cache clock; higher means more recent. Unique
    /// per touch, so it totally orders entries by recency.
    last_used: u64,
}

struct OverflowCache {
    entries: FxHashMap<PathBuf, CacheEntry>,
    total_bytes: usize,
    /// Monotonic recency clock, advanced on every touch/insert.
    clock: u64,
    max_entries: usize,
    max_total_bytes: usize,
    max_file_bytes: usize,
    /// Test-only: bytes of body text copied by the store path, attributed
    /// to THIS instance. Each instance owns its counter, so
    /// independent-instance tests are immune to other modules' global-cache
    /// traffic. The global instance's counter is aliased lock-free (see
    /// `GLOBAL_COPY_BYTES`) so `store_text` can account copies without
    /// holding the global lock.
    #[cfg(test)]
    copied: Option<Arc<AtomicUsize>>,
}

impl OverflowCache {
    fn new() -> Self {
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut cache = Self::with_limits(MAX_ENTRIES, MAX_TOTAL_BYTES, MAX_FILE_BYTES);
        // The global instance's copy counter is aliased lock-free (see
        // `GLOBAL_COPY_BYTES`); tests observe the same atomic.
        #[cfg(test)]
        {
            cache.copied = Some((*GLOBAL_COPY_BYTES).clone());
        }
        cache
    }

    /// Limits are injectable so the eviction logic is testable without
    /// producing hundreds of files or megabytes of content.
    fn with_limits(max_entries: usize, max_total_bytes: usize, max_file_bytes: usize) -> Self {
        OverflowCache {
            entries: FxHashMap::default(),
            total_bytes: 0,
            clock: 0,
            max_entries,
            max_total_bytes,
            max_file_bytes,
            #[cfg(test)]
            copied: Some(Arc::new(AtomicUsize::new(0))),
        }
    }

    /// Advances the recency clock, returning the stamp for this touch.
    /// Wrapping is harmless: only the relative order of live stamps matters,
    /// and two live entries can never share a stamp within 2^64 touches.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn get(&mut self, path: &Path, version: FileVersion) -> Option<Arc<str>> {
        if version.len > self.max_file_bytes as u64 {
            return None;
        }
        // Stamp first: the relative order of hits is what matters, and an
        // unconsumed stamp on a miss path is harmless.
        let stamp = self.tick();
        let entry = self.entries.get_mut(path)?;
        if entry.version != version {
            let stale = self.entries.remove(path).expect("entry just observed");
            self.total_bytes = self.total_bytes.saturating_sub(stale.size);
            return None;
        }
        // LRU touch: O(1) stamp bump, no order-list scan.
        entry.last_used = stamp;
        Some(Arc::clone(&entry.content))
    }

    fn put(&mut self, path: PathBuf, version: FileVersion, content: Arc<str>) {
        if content.len() > self.max_file_bytes {
            return;
        }
        // A version whose length disagrees with the body can never be served
        // coherently; refuse rather than pair mismatched metadata with text.
        if version.len != content.len() as u64 {
            return;
        }
        self.remove(&path);
        let size = content.len();
        self.total_bytes += size;
        let stamp = self.tick();
        self.entries.insert(
            path,
            CacheEntry {
                version,
                size,
                content,
                last_used: stamp,
            },
        );
        self.evict();
    }

    fn remove(&mut self, path: &Path) {
        if let Some(entry) = self.entries.remove(path) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.size);
        }
    }

    /// Drops least-recently-used entries until both caps hold. The oldest-stamp
    /// scan is O(entries) but runs only on the insert path when a cap is
    /// exceeded, never on the hot read path.
    fn evict(&mut self) {
        while self.entries.len() > self.max_entries || self.total_bytes > self.max_total_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            self.remove(&oldest);
        }
    }
}

static CACHE: LazyLock<Mutex<OverflowCache>> = LazyLock::new(|| Mutex::new(OverflowCache::new()));

/// Test-only lock-free alias of the GLOBAL instance's copy counter:
/// `OverflowCache::new` points the instance field at this atomic, so
/// `store_text` can account its copy without holding the global lock.
#[cfg(test)]
static GLOBAL_COPY_BYTES: LazyLock<Arc<AtomicUsize>> =
    LazyLock::new(|| Arc::new(AtomicUsize::new(0)));

/// Returns the cached body of `path` when an entry exists for exactly this
/// file version; `None` otherwise (uncached, different version, oversized).
pub(crate) fn cached_text(path: &Path, version: FileVersion) -> Option<Arc<str>> {
    let mut cache = CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.get(path, version)
}

/// Phase 1 of the store path: enforce the per-file ceiling BEFORE copying,
/// then copy the body, accounting the copied bytes on `copied` when present
/// (test-only observability). Lock-free: the store path runs it before
/// taking the cache lock, so the copy provably happens outside the lock.
fn prepare_body(
    copied: Option<&AtomicUsize>,
    max_file_bytes: usize,
    content: &str,
) -> Option<Arc<str>> {
    if content.len() > max_file_bytes {
        return None; // rejected before a single byte is copied
    }
    let body: Arc<str> = Arc::from(content);
    if let Some(copied) = copied {
        copied.fetch_add(body.len(), Ordering::Relaxed);
    }
    Some(body)
}

/// Production store path shared by `store_text` and the instance-level
/// tests: the size gate and body copy run lock-free first, then the cache
/// lock is taken for lookup/eviction/insertion only.
fn store_body(
    cache: &Mutex<OverflowCache>,
    copied: Option<&AtomicUsize>,
    max_file_bytes: usize,
    path: PathBuf,
    version: FileVersion,
    content: &str,
) {
    let Some(body) = prepare_body(copied, max_file_bytes, content) else {
        return; // oversized: rejected before a single byte is copied
    };
    let mut guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.put(path, version, body);
}

/// Test-only: the global instance's copy counter, reachable without the
/// global lock so `store_text` can account its copy before the lock.
#[cfg(test)]
fn global_copy_counter() -> Option<&'static AtomicUsize> {
    Some(&**GLOBAL_COPY_BYTES)
}

#[cfg(not(test))]
fn global_copy_counter() -> Option<&'static AtomicUsize> {
    None
}

/// Stores `content` as the body of `path` at exactly `version`.
///
/// Subject to the entry/byte caps and the 1 MiB per-file ceiling. The size
/// ceiling is enforced before any copy is made — an oversized body is
/// rejected without allocating — and the body copy happens outside the
/// global lock, which only performs lookup, eviction, and insertion.
pub(crate) fn store_text(path: &Path, version: FileVersion, content: &str) {
    // The global cache always runs with the default per-file ceiling, so
    // the constant here matches `put`'s guard.
    store_body(
        &CACHE,
        global_copy_counter(),
        MAX_FILE_BYTES,
        path.to_path_buf(),
        version,
        content,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn version(len: usize, mtime: SystemTime) -> FileVersion {
        FileVersion {
            len: len as u64,
            mtime,
            dev: 0,
            ino: 0,
            ctime_sec: 0,
            ctime_nsec: 0,
            root_dev: 0,
            root_ino: 0,
        }
    }

    fn epoch() -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    #[test]
    fn cached_text_serves_matching_version_and_misses_on_mismatch() {
        let path = PathBuf::from("/virtual/a.txt");
        // Same length, different mtime: an equal-length in-place replacement.
        let v1 = version(4, epoch());
        let v2 = version(4, epoch() + Duration::from_secs(1));
        let mut cache = OverflowCache::with_limits(8, 4096, 4096);
        cache.put(path.clone(), v1, Arc::from("body"));
        assert_eq!(cache.get(&path, v1).as_deref(), Some("body"));
        // Unchanged version: the second read is a cache hit too.
        assert_eq!(cache.get(&path, v1).as_deref(), Some("body"));
        // Any version change is a miss, and the stale entry is dropped
        // rather than served.
        assert_eq!(cache.get(&path, v2), None);
        assert!(
            !cache.entries.contains_key(&path),
            "stale entry must be cleared on mismatch"
        );
        // Re-storing at the new version makes it servable again.
        cache.put(path.clone(), v2, Arc::from("body"));
        assert_eq!(cache.get(&path, v2).as_deref(), Some("body"));
    }

    #[test]
    fn store_refuses_text_whose_version_len_disagrees() {
        let path = PathBuf::from("/virtual/a.txt");
        let v = version(5, epoch());
        let mut cache = OverflowCache::with_limits(8, 4096, 4096);
        // Body length must equal version.len: a mismatch can never be served
        // coherently, so the store is refused and no entry appears.
        cache.put(path.clone(), v, Arc::from("old"));
        assert_eq!(cache.get(&path, v), None);
        assert!(!cache.entries.contains_key(&path));
    }

    #[test]
    fn entry_limit_evicts_least_recently_used() {
        let mut cache = OverflowCache::with_limits(4, 4096, 1024);
        let files: Vec<PathBuf> = (0..5)
            .map(|index| PathBuf::from(format!("/virtual/f{index}.txt")))
            .collect();
        for (index, path) in files.iter().enumerate() {
            let content = format!("content {index}");
            cache.put(
                path.clone(),
                version(content.len(), epoch()),
                Arc::from(content.as_str()),
            );
        }
        assert!(
            cache.get(&files[0], version(9, epoch())).is_none(),
            "oldest evicted"
        );
        assert_eq!(
            cache.get(&files[4], version(9, epoch())).as_deref(),
            Some("content 4"),
            "newest retained"
        );
        assert!(cache.entries.len() <= 4, "entry cap must hold");
        // Touching f2 makes f1 the next victim instead.
        assert_eq!(
            cache.get(&files[2], version(9, epoch())).as_deref(),
            Some("content 2")
        );
        cache.put(
            PathBuf::from("/virtual/f5.txt"),
            version(9, epoch()),
            Arc::from("content 5"),
        );
        assert!(
            cache.get(&files[1], version(9, epoch())).is_none(),
            "untouched evicted"
        );
        assert_eq!(
            cache.get(&files[2], version(9, epoch())).as_deref(),
            Some("content 2"),
            "touched entry survives"
        );
    }

    #[test]
    fn byte_limit_evicts_until_total_fits() {
        let mut cache = OverflowCache::with_limits(16, 30, 1024);
        let v = version(10, epoch());
        for index in 0..4 {
            cache.put(
                PathBuf::from(format!("/virtual/b{index}.txt")),
                v,
                Arc::from("0123456789"),
            );
        }
        // 4 x 10 bytes = 40 > 30: exactly one entry must go, oldest first.
        assert!(
            cache.get(&PathBuf::from("/virtual/b0.txt"), v).is_none(),
            "oldest evicted"
        );
        assert!(
            cache.get(&PathBuf::from("/virtual/b3.txt"), v).is_some(),
            "newest retained"
        );
        assert!(cache.total_bytes <= 30, "byte cap must hold");
    }

    #[test]
    fn oversized_files_are_never_cached() {
        let path = PathBuf::from("/virtual/big.txt");
        let content = "x".repeat(1025); // > max_file_bytes of 1024
        let v = version(content.len(), epoch());
        let mut cache = OverflowCache::with_limits(8, 4096, 1024);
        cache.put(path.clone(), v, Arc::from(content.as_str()));
        assert_eq!(cache.get(&path, v), None, "oversized content must not cache");
        assert_eq!(cache.total_bytes, 0);
    }

    #[test]
    fn global_api_hits_and_invalidates() {
        let path = PathBuf::from("/virtual/global.txt");
        let v1 = version(11, epoch());
        store_text(&path, v1, "global body");
        assert_eq!(cached_text(&path, v1).as_deref(), Some("global body"));
        // A different version invalidates: the entry is dropped, not served.
        let v2 = version(11, epoch() + Duration::from_secs(2));
        assert_eq!(cached_text(&path, v2), None);
        assert!(
            !CACHE
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entries
                .contains_key(&path),
            "stale entry cleared for this path"
        );
    }

    #[test]
    fn oversized_store_rejects_with_zero_copy() {
        // Independent instance in its own lock: the copy counter is
        // per-instance, so other modules' global-cache traffic cannot
        // pollute this assertion.
        let cache = Arc::new(Mutex::new(OverflowCache::with_limits(8, 4096, 1024)));
        let path = PathBuf::from("/virtual/big.txt");
        let content = "x".repeat(1025); // > max_file_bytes of 1024
        let v = version(content.len(), epoch());
        let copied = Arc::clone(
            cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .copied
                .as_ref()
                .unwrap(),
        );
        // The store path rejects the body before any copy...
        store_body(&cache, Some(&copied), 1024, path.clone(), v, &content);
        assert_eq!(
            copied.load(Ordering::SeqCst),
            0,
            "oversized body must be rejected without copying a single byte"
        );
        // ...and leaves no entry behind.
        let mut guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            guard.get(&path, v),
            None,
            "oversized content must not be cached"
        );
        assert_eq!(guard.total_bytes, 0);
    }

    #[test]
    fn store_copies_body_outside_the_lock() {
        // Independent instance in its own lock, fully isolated from the
        // global cache other modules' tests write to.
        let cache = Arc::new(Mutex::new(OverflowCache::with_limits(8, 4096, 4096)));
        let path = PathBuf::from("/virtual/lock.txt");
        let v = version(3, epoch());
        // Share the instance's copy counter and ceiling before locking, so
        // the store thread can run the copy phase without the instance lock.
        let (copied, ceiling) = {
            let guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            (Arc::clone(guard.copied.as_ref().unwrap()), guard.max_file_bytes)
        };
        // Hold the instance lock...
        let guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // ...and run the production store path on another thread. The body
        // is copied (counted) before the thread blocks on the lock for the
        // insert, so the counter must advance while the lock is still held.
        let spawned_path = path.clone();
        let spawned_cache = Arc::clone(&cache);
        let spawned_copied = Arc::clone(&copied);
        let spawned = std::thread::spawn(move || {
            store_body(
                &spawned_cache,
                Some(&spawned_copied),
                ceiling,
                spawned_path,
                v,
                "abc",
            );
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while copied.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "body copy did not happen outside the lock"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(copied.load(Ordering::SeqCst) >= 3, "copy counter counts body bytes");
        // Release the lock: the store thread inserts, and the entry is
        // servable.
        drop(guard);
        spawned.join().unwrap();
        assert_eq!(
            cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&path, v)
                .as_deref(),
            Some("abc")
        );
    }
}
