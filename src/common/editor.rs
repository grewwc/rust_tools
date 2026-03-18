use std::{io::Write, process::Command};

pub fn input_with_editor(initial: &str, use_vscode: bool) -> Result<String, String> {
    let mut path = std::env::temp_dir();
    path.push(format!("rust_tools_editor_{}.txt", uuid::Uuid::new_v4()));
    std::fs::write(&path, initial).map_err(|e| e.to_string())?;

    let editor = if use_vscode {
        "code".to_string()
    } else {
        std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string())
    };

    let status = if use_vscode {
        Command::new(editor)
            .arg("-w")
            .arg(&path)
            .status()
            .map_err(|e| e.to_string())?
    } else {
        Command::new(editor)
            .arg(&path)
            .status()
            .map_err(|e| e.to_string())?
    };
    if !status.success() {
        return Err("editor exited with non-zero status".to_string());
    }

    let s = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(s)
}

pub fn stdin_to_string() -> String {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).ok();
    buf
}

trait ReadToStringExt {
    fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize>;
}

impl ReadToStringExt for std::io::Stdin {
    fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize> {
        use std::io::Read;
        self.lock().read_to_string(buf)
    }
}

pub fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

