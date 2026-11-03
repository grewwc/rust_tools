use std::path::{Component, Path, PathBuf};

/// Lexically normalize a path, resolving `.` and `..` without touching the
/// filesystem because the path may not exist yet.
pub(crate) fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::normalize_lexical;
    use std::path::{Path, PathBuf};

    #[test]
    fn resolves_dot_and_dotdot() {
        assert_eq!(
            normalize_lexical(Path::new("/home/user/proj/../proj/./src")),
            PathBuf::from("/home/user/proj/src")
        );
        assert_eq!(
            normalize_lexical(Path::new("a/./b/../c")),
            PathBuf::from("a/c")
        );
    }
}
