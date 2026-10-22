//! Bounded, read-only access to this session's recovery assets.
//!
//! Included by both memory projection and overflow search so their filesystem
//! boundary is identical without exposing driver internals through the tool API.

use std::{
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

/// Text-bearing recovery directories, not arbitrary session assets such as images
/// or temporary command files. Scope-specific history/user roots are added by search.
pub(super) const RECOVERY_DIRECTORIES: &[&str] = &[
    "tool-overflow-compressed",
    "folded-tool-groups",
    "internal-note-overflow",
    "context-checkpoints",
    "tool-overflow",
];

/// Identity of one file version observed on an open descriptor: length and
/// modification time plus, on Unix, the device/inode pair and the metadata
/// change time. Length plus mtime alone cannot distinguish an equal-length
/// replacement that restores the mtime, nor an in-place rewrite of the same
/// inode; the device/inode pair catches replacement by rename, and the change
/// time catches in-place rewrites whose mtime was restored, because updating
/// the mtime itself advances the change time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct FileIdentity {
    /// File length in bytes, observed on the descriptor.
    pub len: u64,
    /// File modification time, observed on the descriptor.
    pub mtime: SystemTime,
    /// Owning device, Unix only; zero elsewhere.
    pub dev: u64,
    /// Inode number, Unix only; zero elsewhere.
    pub ino: u64,
    /// Metadata change time, seconds since the Unix epoch; zero elsewhere.
    pub ctime_sec: i64,
    /// Metadata change time, sub-second nanoseconds; zero elsewhere.
    pub ctime_nsec: i64,
}

fn identity_of(metadata: &fs::Metadata, len: u64, mtime: SystemTime) -> FileIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        FileIdentity {
            len,
            mtime,
            dev: metadata.dev(),
            ino: metadata.ino(),
            ctime_sec: metadata.ctime(),
            // ctime_nsec is i32 on some targets and i64 on others; narrow
            // explicitly so the struct shape stays stable across platforms.
            ctime_nsec: metadata.ctime_nsec() as i64,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        FileIdentity {
            len,
            mtime,
            dev: 0,
            ino: 0,
            ctime_sec: 0,
            ctime_nsec: 0,
        }
    }
}

pub(super) struct ArchiveRoot {
    path: PathBuf,
    input_path: PathBuf,
    #[cfg(unix)]
    directory: File,
}

impl ArchiveRoot {
    pub(super) fn new(path: &Path) -> Option<Self> {
        // The root comes from runtime session identity, never from an archived
        // pointer. Reject a symlink root as well as symlink descendants.
        let metadata = fs::symlink_metadata(path).ok()?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return None;
        }
        let input_path = path.to_path_buf();
        let path = fs::canonicalize(path).ok()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
            let directory = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&input_path)
                .ok()?;
            let opened = directory.metadata().ok()?;
            if opened.dev() != metadata.dev() || opened.ino() != metadata.ino() {
                return None;
            }
            Some(Self {
                path,
                input_path,
                directory,
            })
        }
        #[cfg(not(unix))]
        {
            Some(Self { path, input_path })
        }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    fn relative<'a>(&self, path: &'a Path) -> Option<&'a Path> {
        // Accept the runtime root's lexical spelling too (for example macOS
        // /var versus /private/var), without canonicalizing an untrusted pointer.
        let relative = path
            .strip_prefix(&self.path)
            .or_else(|_| path.strip_prefix(&self.input_path))
            .ok()?;
        relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
            .then_some(relative)
    }

    #[cfg(unix)]
    fn open(&self, path: &Path) -> Option<File> {
        use std::{
            ffi::CString,
            os::fd::{AsRawFd, FromRawFd},
            os::unix::ffi::OsStrExt,
        };
        let relative = self.relative(path)?;
        // dup()/try_clone() shares the directory stream offset. Reopening '.'
        // gives each traversal its own offset, including repeated root scans.
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                c".".as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return None;
        }
        let mut directory = unsafe { File::from_raw_fd(fd) };
        let mut parts = relative.components().peekable();
        while let Some(part) = parts.next() {
            let name = CString::new(part.as_os_str().as_bytes()).ok()?;
            let flags = libc::O_RDONLY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK
                | if parts.peek().is_some() {
                    libc::O_DIRECTORY
                } else {
                    0
                };
            // Every component is opened relative to an already-held directory.
            // O_NOFOLLOW on every hop prevents symlink swaps from escaping the
            // session root; O_NONBLOCK prevents a raced FIFO from blocking.
            let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
            if fd < 0 {
                return None;
            }
            directory = unsafe { File::from_raw_fd(fd) };
        }
        Some(directory)
    }

    #[cfg(not(unix))]
    fn open(&self, path: &Path) -> Option<File> {
        let relative = self.relative(path)?;
        let mut checked = self.path.clone();
        for part in relative.components() {
            checked.push(part.as_os_str());
            if fs::symlink_metadata(&checked)
                .ok()?
                .file_type()
                .is_symlink()
            {
                return None;
            }
        }
        if !fs::canonicalize(&checked).ok()?.starts_with(&self.path) {
            return None;
        }
        File::open(checked).ok()
    }

    /// Entire small UTF-8 files only. Oversize or malformed inputs are skipped,
    /// not silently treated as complete prefixes. The byte budget also bounds
    /// repeated failed/invalid reads and concurrent growth after stat().
    pub(super) fn read_text(
        &self,
        path: &Path,
        max_file_bytes: usize,
        remaining_bytes: &mut usize,
    ) -> Option<String> {
        let (file, _) = self.open_versioned(path)?;
        self.read_opened_text(file, max_file_bytes, remaining_bytes)
            .map(|(content, _)| content)
    }

    /// Identity of the held session root directory itself: the device/inode
    /// pair of the directory descriptor captured at `new` time. Callers bind
    /// cached file bodies to this pair so entries from one session directory
    /// can never be served for another path that merely reuses the same
    /// spelling after the directory was replaced.
    pub(super) fn root_identity(&self) -> (u64, u64) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            return self
                .directory
                .metadata()
                .ok()
                .map(|metadata| (metadata.dev(), metadata.ino()))
                .unwrap_or((0, 0));
        }
        #[cfg(not(unix))]
        {
            (0, 0)
        }
    }

    /// Opens `path` through the session-bound safe path (every component opened
    /// with O_NOFOLLOW relative to a held directory descriptor) and returns the
    /// open descriptor plus the file identity observed on that descriptor.
    /// The identity covers length, mtime, device/inode, and change time so an
    /// equal-length replacement that restores the mtime is still a mismatch.
    pub(super) fn open_versioned(&self, path: &Path) -> Option<(File, FileIdentity)> {
        let file = self.open(path)?;
        let metadata = file.metadata().ok()?;
        let mtime = metadata.modified().ok()?;
        let len = metadata.len();
        Some((file, identity_of(&metadata, len, mtime)))
    }

    /// Reads an already-open descriptor with the same size limit and UTF-8
    /// validation as `read_text`, and returns the content plus the file
    /// identity observed on the same descriptor after the read. Callers must
    /// compare the before/after identities and reject the read when they
    /// differ: the content may be torn by a concurrent rewrite.
    pub(super) fn read_opened_text(
        &self,
        file: File,
        max_file_bytes: usize,
        remaining_bytes: &mut usize,
    ) -> Option<(String, FileIdentity)> {
        if *remaining_bytes == 0 {
            return None;
        }
        let metadata = file.metadata().ok()?;
        let limit = max_file_bytes.min(remaining_bytes.saturating_sub(1));
        if !metadata.is_file() || metadata.len() > limit as u64 {
            return None;
        }
        let mut bytes = Vec::new();
        let result = (&file).take(limit as u64 + 1).read_to_end(&mut bytes);
        *remaining_bytes = remaining_bytes.saturating_sub(bytes.len());
        if result.is_err() || bytes.len() > limit {
            return None;
        }
        let content = String::from_utf8(bytes).ok()?;
        let after = file.metadata().ok()?;
        let after_mtime = after.modified().ok()?;
        let after_len = after.len();
        Some((content, identity_of(&after, after_len, after_mtime)))
    }

    #[cfg(unix)]
    fn children(
        &self,
        directory: File,
        path: &Path,
        remaining: &mut usize,
    ) -> Option<Vec<PathBuf>> {
        use std::{ffi::CStr, os::fd::IntoRawFd, os::unix::ffi::OsStrExt};
        if !directory.metadata().ok()?.is_dir() {
            return None;
        }
        let fd = directory.into_raw_fd();
        // fdopendir owns fd on success. Listing through this descriptor avoids
        // a second path lookup and a directory-symlink race during traversal.
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            unsafe { libc::close(fd) };
            return None;
        }
        let mut paths = Vec::new();
        while *remaining > 0 {
            let entry = unsafe { libc::readdir(stream) };
            if entry.is_null() {
                break;
            }
            *remaining -= 1;
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if name != b"." && name != b".." {
                paths.push(path.join(std::ffi::OsStr::from_bytes(name)));
            }
        }
        unsafe { libc::closedir(stream) };
        paths.sort();
        Some(paths)
    }

    #[cfg(not(unix))]
    fn children(
        &self,
        _directory: File,
        path: &Path,
        remaining: &mut usize,
    ) -> Option<Vec<PathBuf>> {
        // Non-Unix retains the existing path-based fallback; only Unix offers
        // descriptor-anchored enumeration and component-by-component no-follow.
        self.open(path)?;
        let mut paths = Vec::new();
        for entry in fs::read_dir(path).ok()?.take(*remaining) {
            *remaining = remaining.saturating_sub(1);
            if let Ok(entry) = entry {
                paths.push(entry.path());
            }
        }
        paths.sort();
        Some(paths)
    }

    /// Returns only regular files. Traversal is bounded even for directories
    /// with huge fan-out, and never follows filesystem links on Unix.
    pub(super) fn collect_files(
        &self,
        root: &Path,
        max_files: usize,
        remaining_entries: &mut usize,
    ) -> Vec<PathBuf> {
        self.collect_files_with_depth(root, max_files, remaining_entries, 8)
    }

    /// Manual search shares the filesystem boundary, not automatic recall's
    /// candidate, entry, or depth limits: late and deep paths remain searchable.
    #[cfg(unix)]
    pub(super) fn collect_files_unbounded(&self, root: &Path) -> Vec<PathBuf> {
        let mut remaining_entries = usize::MAX;
        self.collect_files_with_depth(root, usize::MAX, &mut remaining_entries, usize::MAX)
    }

    fn collect_files_with_depth(
        &self,
        root: &Path,
        max_files: usize,
        remaining_entries: &mut usize,
        max_depth: usize,
    ) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut stack = vec![(root.to_path_buf(), 0usize)];
        while let Some((path, depth)) = stack.pop() {
            if files.len() >= max_files {
                break;
            }
            let Some(file) = self.open(&path) else {
                continue;
            };
            let Ok(metadata) = file.metadata() else {
                continue;
            };
            if metadata.is_file() {
                files.push(path);
            } else if metadata.is_dir() && depth < max_depth && *remaining_entries > 0 {
                // Enumerate the same directory we inspected, not its pathname:
                // renames after open must not redirect enumeration to a link.
                if let Some(children) = self.children(file, &path, remaining_entries) {
                    stack.extend(children.into_iter().rev().map(|child| (child, depth + 1)));
                }
            }
        }
        files.sort();
        files
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn bounded_archive_reader_enumerates_held_directory_after_symlink_replacement() {
        let fixture = std::env::temp_dir().join(format!("archive_race_{}", uuid::Uuid::new_v4()));
        let assets = fixture.join("assets");
        let nested = assets.join("nested");
        let outside = fixture.join("outside");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(nested.join("inside.md"), "inside body").unwrap();
        fs::write(outside.join("external.md"), "external body").unwrap();
        let root = ArchiveRoot::new(&assets).unwrap();
        let held = root.open(&nested).unwrap();

        // Deterministically replace the checked path before fdopendir/readdir.
        // A pathname-based listing would now return external.md instead.
        fs::rename(&nested, assets.join("saved")).unwrap();
        std::os::unix::fs::symlink(&outside, &nested).unwrap();
        let paths = root.children(held, &nested, &mut 64).unwrap();
        assert_eq!(paths, vec![nested.join("inside.md")]);
        assert!(root.collect_files_unbounded(&nested).is_empty());
        assert!(root.collect_files(&nested, 4, &mut 64).is_empty());
        assert!(root.read_text(&paths[0], 128, &mut 256).is_none());
        fs::remove_dir_all(fixture).unwrap();
    }

    #[test]
    fn bounded_archive_reader_rejects_directory_replacement_after_discovery() {
        let fixture =
            std::env::temp_dir().join(format!("archive_discovery_{}", uuid::Uuid::new_v4()));
        let assets = fixture.join("assets");
        let nested = assets.join("nested");
        let outside = fixture.join("outside");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("external.md"), "external body").unwrap();
        let root = ArchiveRoot::new(&assets).unwrap();
        let discovered = root
            .children(root.open(&assets).unwrap(), &assets, &mut 64)
            .unwrap();
        assert_eq!(discovered, vec![nested.clone()]);

        // The queued child is replaced before its component-wise open.
        fs::rename(&nested, assets.join("saved")).unwrap();
        std::os::unix::fs::symlink(&outside, &nested).unwrap();
        assert!(root.collect_files_unbounded(&discovered[0]).is_empty());
        assert!(root.collect_files(&discovered[0], 4, &mut 64).is_empty());
        fs::remove_dir_all(fixture).unwrap();
    }

    #[test]
    fn bounded_archive_reader_keeps_recall_limits_and_manual_deep_traversal() {
        let assets = std::env::temp_dir().join(format!("archive_limits_{}", uuid::Uuid::new_v4()));
        let deep = assets.join("z/a/b/c/d/e/f/g/h/i");
        fs::create_dir_all(&deep).unwrap();
        let shallow_file = assets.join("a.md");
        let deep_file = deep.join("deep.md");
        fs::write(&shallow_file, "shallow").unwrap();
        fs::write(&deep_file, "deep").unwrap();
        let root = ArchiveRoot::new(&assets).unwrap();
        for _ in 0..2 {
            assert_eq!(
                root.collect_files_unbounded(&assets),
                vec![shallow_file.clone(), deep_file.clone()]
            );
            assert_eq!(
                root.collect_files(&assets, 16, &mut 128),
                vec![shallow_file.clone()]
            );
        }
        assert!(root.collect_files(&assets, 0, &mut 128).is_empty());
        assert!(root.collect_files(&assets, 16, &mut 0).is_empty());
        let mut entries = 1;
        root.collect_files(&assets, 16, &mut entries);
        assert_eq!(entries, 0);
        fs::write(assets.join("b.md"), "second").unwrap();
        assert_eq!(
            root.collect_files(&assets, 1, &mut 128),
            vec![shallow_file.clone()]
        );

        // Enumeration must not weaken the independent bounded content reader.
        let mut bytes = 32;
        assert!(root.read_text(&shallow_file, 3, &mut bytes).is_none());
        assert_eq!(bytes, 32);
        assert_eq!(
            root.read_text(&deep_file, 4, &mut bytes).as_deref(),
            Some("deep")
        );
        assert_eq!(bytes, 28);
        fs::write(assets.join("invalid.md"), [0xff]).unwrap();
        assert!(
            root.read_text(&assets.join("invalid.md"), 4, &mut bytes)
                .is_none()
        );
        assert_eq!(bytes, 27);
        fs::remove_dir_all(assets).unwrap();
    }

    #[test]
    fn bounded_archive_reader_root_listing_is_repeatable_and_accepts_runtime_spelling() {
        let path = std::env::temp_dir().join(format!("archive_reader_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("entry.md"), "repeatable archive body").unwrap();
        let root = ArchiveRoot::new(&path).unwrap();
        for _ in 0..3 {
            let files = root.collect_files(&path, 4, &mut 64);
            assert_eq!(files, vec![path.join("entry.md")]);
            assert_eq!(
                root.read_text(&files[0], 128, &mut 256).as_deref(),
                Some("repeatable archive body")
            );
        }
        assert!(
            root.read_text(&path.join("../outside.md"), 128, &mut 256)
                .is_none()
        );
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(ArchiveRoot::new(&link).is_none());
        fs::remove_file(link).unwrap();
        fs::remove_dir_all(path).unwrap();
    }
}
