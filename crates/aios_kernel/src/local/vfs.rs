use super::*;

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
