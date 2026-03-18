use std::borrow::Cow;

pub fn get_home_dir() -> Option<String> {
    if let Ok(home) = std::env::var("HOME") {
        return Some(home);
    }
    None
}

pub fn expanduser(path_str: &str) -> Cow<'_, str> {
    if let Some(home) = get_home_dir() {
        return Cow::Owned(path_str.replace("~", &home));
    }
    Cow::Borrowed(path_str)
}
