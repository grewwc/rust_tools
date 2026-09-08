use super::*;

/// Wording shared with the tools-side compare-and-swap fallback in
/// `src/bin/ai/tools/storage/file_store.rs` (`FILE_CHANGED_MSG`); keep the two in sync so
/// kernel-bound and unbound compare-and-swap failures surface identically.
const FILE_CHANGED_MSG: &str =
    "[FILE_CHANGED] file changed since it was last read; re-read and rebuild before retrying";

impl VfsOps for LocalOS {
    // SAFETY/PERF NOTE: the methods on this impl block call into blocking
    // `std::fs` while the global `SharedKernel` mutex is held. They must
    // therefore not be invoked from latency-sensitive async code paths that
    // expect non-blocking semantics — large reads/writes will stall every
    // other tenant of the kernel for the duration of the syscall. Use
    // out-of-band tooling (e.g. dedicated worker threads) for big files.
    fn vfs_read_to_string(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
    ) -> Result<String, VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("read.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        if !path.exists() {
            self.vfs_emit_trace("read.notfound", pid, path, 0, None);
            return Err(VfsError::NotFound(path.display().to_string()));
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?;
        let bytes = content.len() as u64;

        // charge fs_bytes (skip when the pid is missing or unconstrained; return QuotaExceeded when the rlimit is exceeded)
        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("read", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }
        Ok(content)
    }

    fn vfs_read_range(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        offset: u64,
        max_bytes: usize,
    ) -> Result<VfsReadRange, VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("read.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        if !path.exists() {
            self.vfs_emit_trace("read.notfound", pid, path, 0, None);
            return Err(VfsError::NotFound(path.display().to_string()));
        }

        // A zero-byte chunk can never contain a character, so it could never advance a paged
        // caller; reject it explicitly instead of returning a non-advancing result
        // (offset, empty content, hit_eof=false) that would make the caller loop forever.
        if max_bytes == 0 {
            return Err(VfsError::Io(
                "Failed to read file: max_bytes must be at least 1 to make progress".to_string(),
            ));
        }

        // Each call transfers at most `max_bytes`, so a caller streaming a large file through the
        // shared kernel lock holds it for a single bounded chunk rather than for the whole file.
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)
            .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?;
        let mut raw = Vec::with_capacity(max_bytes);
        let read = file
            .by_ref()
            .take(max_bytes as u64)
            .read_to_end(&mut raw)
            .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?;

        // `read < max_bytes` proves EOF, but a chunk that exactly fills `max_bytes` may still sit
        // at the true end of the file (remaining bytes == max_bytes). Peek one byte past the chunk
        // so `hit_eof` matches the documented contract — "a subsequent read at next_offset would
        // return empty content" — instead of forcing paged callers into a redundant final empty
        // read. The probe runs only on full chunks, never on short (final) reads.
        let mut probe = [0u8; 1];
        let probe_read = if read == max_bytes {
            file.read(&mut probe)
                .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?
        } else {
            0
        };
        let reached_eof = read < max_bytes || probe_read == 0;
        // Trim the tail at a character boundary when a multibyte char straddles the range end. The
        // partial char is reported to the caller by `next_offset` resuming at the char's start, so
        // the next read re-reads those few bytes together with the rest of the char and decodes
        // them correctly. A genuinely invalid byte anywhere, or an incomplete char at EOF, is an
        // error — mirroring the whole-file `read_to_string` behavior.
        let (content, consumed) = decode_range_utf8(&raw, reached_eof)?;
        let bytes = content.len() as u64;

        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("read", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }

        Ok(VfsReadRange {
            content,
            // Resume at the consumed (char-boundary-aligned) length, not at the raw read end: this
            // re-reads any trimmed partial char so the next chunk starts at a valid UTF-8 boundary.
            next_offset: offset + consumed as u64,
            // True only when the raw read hit EOF *and* nothing was trimmed (a trimmed tail means
            // the partial char still has to be re-read, i.e. more bytes follow).
            hit_eof: reached_eof && consumed as usize == read,
        })
    }

    fn vfs_write_all(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        content: &str,
    ) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("write.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| VfsError::Io(format!("Failed to create directory: {}", e)))?;
            }
        }
        std::fs::write(path, content)
            .map_err(|e| VfsError::Io(format!("Failed to write file: {}", e)))?;
        let bytes = content.len() as u64;

        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("write", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }
        Ok(())
    }

    fn vfs_write_if_unchanged(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        expected: Option<&str>,
        content: &str,
    ) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("write.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        // Existing-file comparisons are serialized with other operations on this kernel.
        // Creation uses O_EXCL/create_new below: a pre-check alone cannot protect against
        // an external process creating the path, including a dangling symlink.
        if let Some(expected) = expected {
            let current = match std::fs::read_to_string(path) {
                Ok(current) => Some(current),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                Err(err) => return Err(VfsError::Io(format!("Failed to read file: {err}"))),
            };
            if current.as_deref() != Some(expected) {
                self.vfs_emit_trace("write.stale", pid, path, 0, None);
                return Err(VfsError::Io(FILE_CHANGED_MSG.to_string()));
            }
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| VfsError::Io(format!("Failed to create directory: {}", e)))?;
            }
        }
        if expected.is_none() {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|err| {
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        self.vfs_emit_trace("write.stale", pid, path, 0, None);
                        VfsError::Io(FILE_CHANGED_MSG.to_string())
                    } else {
                        VfsError::Io(format!("Failed to create file: {err}"))
                    }
                })?;
            if let Err(write_err) = file.write_all(content.as_bytes()) {
                // The file was created by this call (create_new proved the path was absent),
                // so a failed write can only leave a partial prefix of `content` behind.
                // Remove it best-effort to restore the "absent" precondition so a batch
                // rollback treats this create as fully undone; a file that an external
                // writer replaced in the meantime is preserved by the prefix check.
                remove_partial_create_file(path, content);
                return Err(VfsError::Io(format!("Failed to write file: {write_err}")));
            }
        } else {
            std::fs::write(path, content)
                .map_err(|e| VfsError::Io(format!("Failed to write file: {}", e)))?;
        }
        let bytes = content.len() as u64;

        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("write", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }
        Ok(())
    }

    fn vfs_remove_if_unchanged(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        expected: Option<&str>,
    ) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        // Same single-lock-hold check-then-act contract as `vfs_write_if_unchanged`.
        let current = if path.exists() {
            Some(
                std::fs::read_to_string(path)
                    .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?,
            )
        } else {
            None
        };
        if current.as_deref() != expected {
            self.vfs_emit_trace("remove.stale", pid, path, 0, None);
            return Err(VfsError::Io(FILE_CHANGED_MSG.to_string()));
        }
        std::fs::remove_file(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;
        self.vfs_emit_trace("remove", pid, path, 0, None);
        Ok(())
    }

    fn vfs_stat(&mut self, path: &std::path::Path) -> Result<VfsStat, VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;
        Ok(VfsStat {
            size: meta.len(),
            is_file: meta.is_file(),
            is_dir: meta.is_dir(),
        })
    }

    fn vfs_remove_file(&mut self, path: &std::path::Path) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        std::fs::remove_file(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;
        self.vfs_emit_trace("remove", None, path, 0, None);
        Ok(())
    }
}

/// Best-effort cleanup for a `create_new` write that failed partway through `content`:
/// removes the file only while its current bytes are still a strict prefix of `content`
/// (i.e. the on-disk data can only be our own partial write). A file that an external
/// writer replaced with unrelated content is left intact, so a later rollback surfaces
/// a truthful failure instead of deleting another writer's data.
///
/// NOTE: the tools-side fallback in `storage/file_store.rs` keeps a behavior-identical
/// copy of this function for the unbound create path; keep the two in sync.
pub(super) fn remove_partial_create_file(path: &std::path::Path, content: &str) {
    let current = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    if !content.as_bytes().starts_with(&current) || current.len() == content.len() {
        // Either the file is not ours (not a prefix of what we wrote), or the write
        // actually completed in full (current == content): both cases must be left for
        // the regular compare-and-swap rollback to decide.
        return;
    }
    let _ = std::fs::remove_file(path);
}

/// Decodes a bounded read buffer as UTF-8 for `vfs_read_range`.
///
/// Returns `(decoded_string, consumed_bytes)`. When a multibyte character straddles the end of the
/// buffer and the file continues (`reached_eof == false`), the tail is trimmed at the character
/// boundary and `consumed_bytes` points at the start of that partial character, so the caller can
/// resume there and re-read the partial character together with its remaining bytes.
///
/// A genuinely invalid byte anywhere in the buffer, or an incomplete character at end-of-file, is
/// an error — matching the whole-file `read_to_string` behavior ("stream did not contain valid
/// UTF-8") rather than silently dropping the rest of the file.
///
/// When the *entire* chunk falls inside a single multibyte character (the file continues but the
/// chunk holds only a prefix of that character), no progress is possible with the given
/// `max_bytes`; this is reported as an error telling the caller to increase `max_bytes`, instead
/// of returning `("", 0)` which would make a paged caller retry the same offset forever.
///
/// NOTE: the tools-side fallback in `storage/file_store.rs` keeps a behavior-identical copy of
/// this function so kernel-bound and unbound reads behave the same; keep the two in sync.
fn decode_range_utf8(bytes: &[u8], reached_eof: bool) -> Result<(String, usize), VfsError> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok((s.to_string(), bytes.len())),
        // A real invalid byte (not a truncated char at the tail) or an incomplete char at EOF.
        Err(e) if e.error_len().is_some() || reached_eof => {
            Err(VfsError::Io("Failed to read file: stream did not contain valid UTF-8".to_string()))
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if valid == 0 {
                // The whole chunk is a prefix of one multibyte character and the file continues.
                // Returning ("", 0) would make a paged caller retry the same offset forever;
                // only a larger chunk can contain a complete character and make progress.
                return Err(VfsError::Io(
                    "Failed to read file: read chunk is too small to contain a complete UTF-8 \
                     character; increase max_bytes to make progress"
                        .to_string(),
                ));
            }
            let s = std::str::from_utf8(&bytes[..valid])
                .expect("prefix up to the first UTF-8 error is always valid");
            Ok((s.to_string(), valid))
        }
    }
}
