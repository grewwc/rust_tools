use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn flush_stdout() {
    let _ = io::stdout().flush();
}

pub fn input_with_editor(initial: &str, use_vscode: bool) -> io::Result<String> {
    let path = temp_edit_path();
    fs::write(&path, initial)?;

    let edit_result = launch_editor(&path, use_vscode);
    let content = fs::read_to_string(&path);
    let _ = fs::remove_file(&path);

    edit_result?;
    content.map(|text| text.trim_end_matches(['\n', '\r']).to_string())
}

fn launch_editor(path: &Path, use_vscode: bool) -> io::Result<()> {
    if use_vscode {
        match Command::new("code").arg("--wait").arg(path).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }

    let status = if let Some(editor_cmd) = configured_editor() {
        Command::new("sh")
            .arg("-c")
            .arg("exec $EDITOR_CMD \"$1\"")
            .arg("sh")
            .arg(path)
            .env("EDITOR_CMD", editor_cmd)
            .status()?
    } else {
        match Command::new("vim").arg(path).status() {
            Ok(status) => status,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Command::new("vi").arg(path).status()?
            }
            Err(e) => return Err(e),
        }
    };

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("editor command failed"))
    }
}

fn configured_editor() -> Option<String> {
    ["VISUAL", "EDITOR"].into_iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn temp_edit_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rust_tools_editor_{}_{}.txt",
        std::process::id(),
        stamp
    ))
}
