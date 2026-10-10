pub(crate) fn is_shell_program(program: &str) -> bool {
    matches!(program, "bash" | "sh" | "zsh" | "ksh" | "dash")
}

// Script interpreters also accept `-c` / `-e` to pass and execute a code string directly,
// which would bypass the per-segment blacklist validation.
pub(crate) fn is_interpreter_program(program: &str) -> bool {
    matches!(
        program,
        "python" | "python3" | "perl" | "ruby" | "node" | "php" | "awk" | "lua"
    )
}

pub(crate) fn is_python_program(program: &str) -> bool {
    matches!(program, "python" | "python3")
}

/// Presence of "second-interpretation" options: `-c` / `--command` (shells),
/// `-c` / `-e` (interpreters).
pub(crate) fn shell_c_option_present(program: &str, tokens: &[String]) -> bool {
    let is_shell = is_shell_program(program);
    let is_interpreter = is_interpreter_program(program);
    if !is_shell && !is_interpreter {
        return false;
    }
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return false;
        }
        if !tok.starts_with('-') || tok == "-" {
            return false;
        }
        // Clustered short options can also carry `-c` / `-e` (e.g. `bash -lc`,
        // `perl -le`, `node -pe`). Match only single-dash tokens longer than 2
        // chars: long options like `--norc`, and `-c` / `-e` already exactly
        // matched, are not hit (note tokens are lowercased, so `-C` noclobber gets
        // conflated with `-c` — a pre-existing false positive).
        let grouped = !tok.starts_with("--") && tok.len() > 2;
        if is_shell && (tok == "-c" || tok == "--command" || (grouped && tok.contains('c'))) {
            return true;
        }
        if is_interpreter
            && (tok == "-c" || tok == "-e" || (grouped && (tok.contains('c') || tok.contains('e'))))
        {
            return true;
        }
        i += 1;
    }
    false
}

/// Extract the code string of `python -c <code>` (the tokenizer already removed
/// shell quotes).
/// - `Ok(None)`: no `-c` option (e.g. `python3 script.py` / `python3 -m mod`), no
///   code string involved.
/// - `Ok(Some(code))`: a literal code string was extracted.
/// - `Err`: `-c` is present but the code string cannot be statically obtained
///   (missing / empty / from shell variable expansion), fail-closed.
pub(crate) fn python_c_argument(tokens: &[String]) -> Result<Option<String>, String> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" || !tok.starts_with('-') || tok == "-" {
            // Option region ended without `-c` -> ordinary script / module
            // execution.
            return Ok(None);
        }
        // `-W` / `-X` / `-m` consume one value argument (attached `-Wfoo` or
        // separate `-W foo`); the value itself is not `-c`, so skip and keep
        // looking.
        if matches!(tok, "-W" | "-X" | "-m")
            || tok.starts_with("-W")
            || tok.starts_with("-X")
            || tok.starts_with("-m")
        {
            if matches!(tok, "-W" | "-X" | "-m") {
                i += 1; // skip the value argument token
            }
            i += 1;
            continue;
        }
        if tok == "-c" {
            return match tokens.get(i + 1) {
                Some(code) => Ok(Some(code.clone())),
                None => Err("`-c` requires a code argument".to_string()),
            };
        }
        if let Some(code) = tok.strip_prefix("-c") {
            // Attached form `-cCODE`.
            return if code.is_empty() {
                Err("`-c` requires a non-empty code argument".to_string())
            } else {
                Ok(Some(code.to_string()))
            };
        }
        // Clustered short options may contain `-c` (e.g. `-uc` equals `-u -c`,
        // `-Oc` equals `-O -c`).
        if tok.contains('c') {
            return match tokens.get(i + 1) {
                Some(code) => Ok(Some(code.clone())),
                None => Err("`-c` requires a code argument".to_string()),
            };
        }
        i += 1;
    }
    Ok(None)
}

/// Validate the code string passed to `python -c` (static, best-effort): strip
/// whitespace, lowercase-flatten, then scan for dangerous primitives; a hit is
/// rejected (fail-closed). Occurrences inside comments/strings are matched too —
/// an accepted false positive (safety first).
///
/// Note: this is a static defense at the same level as the whole command audit,
/// not a sandbox — deliberately obfuscated code can in theory always find blind
/// spots. But compared with blanket blocking, it turns python `-c` from
/// "unauditable" into "auditable", covering direct calls and common obfuscation
/// entry points (getattr / __import__ / exec / eval / dunder escape chains, etc.).
pub(crate) fn validate_python_code(code: &str) -> Result<(), String> {
    // `$` or backticks in the code string suggest the content may come from shell
    // variable expansion (e.g. `python3 -c $CODE`); the audit cannot see the
    // expanded content -> fail-closed.
    if code.contains('$') || code.contains('`') {
        return Err(
            "python -c code must be a literal quoted string without shell expansion (`$`)"
                .to_string(),
        );
    }
    let compact: String = code
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect();
    if compact.is_empty() {
        return Err("python -c requires a non-empty code string".to_string());
    }

    const DANGEROUS_PYTHON_PATTERNS: &[&str] = &[
        // —— External command execution / process control ——
        "os.system",
        "os.popen",
        "os.spawn",
        "os.exec",
        "os.fork",
        "os.kill",
        "os.killpg",
        "subprocess.",
        "importsubprocess",
        "fromsubprocess",
        // Block every `from X import ...` form (including wildcard) and alias
        // form (`import X as ...`) of dangerous modules: otherwise `from os
        // import system; system("...")` / `import os as o; o.system(...)` bypass
        // the direct `os.system` match above. Plain `import os` / `import
        // shutil` stay allowed so read-only uses keep working (`os.getcwd()`,
        // `os.path.exists(...)`, `shutil.which(...)`); direct and
        // dunder-mediated attribute calls (`os.system(...)`, `os.__dict__[..]`,
        // `os.__getattribute__(..)`) are still caught by the patterns below.
        // Residual blind spots (same best-effort level as the whole audit):
        // copied references and `vars(...)` indirection (`import os; x = os;
        // x.system(...)`, `vars(os)["system"](...)`) are not caught statically.
        "importosas",
        "fromos",
        "importposix",
        "fromposix",
        "importshutilas",
        "fromshutil",
        "importsocket",
        "fromsocket",
        "importctypes",
        "fromctypes",
        "importpty",
        "frompty",
        "importmarshal",
        "frommarshal",
        "importpickle",
        "frompickle",
        "importtelnetlib",
        "fromtelnetlib",
        "importftplib",
        "fromftplib",
        "importsmtplib",
        "fromsmtplib",
        "importpwn",
        "frompwn",
        "importcommands",
        "fromcommands",
        "importimportlib",
        "fromimportlib",
        // Fetch the already-loaded os via `sys.modules` and call through it (e.g.
        // `import json` loads os internally).
        "sys.modules",
        "commands.getoutput",
        "signal.kill",
        "pty.",
        // —— File destruction / permissions / ownership / links / renaming ——
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "os.chmod",
        "os.chown",
        "os.chflags",
        "os.rename",
        "os.replace",
        "os.link",
        "os.symlink",
        "os.truncate",
        "os.mkfifo",
        "os.mknod",
        "os.setuid",
        "os.setgid",
        "shutil.rmtree",
        "shutil.move",
        "shutil.chown",
        // Path(...) method-call forms (`.unlink()` etc.); `).replace(` also covers
        // os.replace.
        ").unlink(",
        ").rmdir(",
        ").rename(",
        ").replace(",
        ").chmod(",
        ").chown(",
        ").symlink_to(",
        ").write_text(",
        ").write_bytes(",
        ").truncate(",
        // —— Dynamic execution / dynamic import (obfuscation and escape entry
        // points) ——
        "eval(",
        "exec(",
        "execfile(",
        "compile(",
        "__import__",
        "importlib.",
        "getattr(",
        "setattr(",
        "__builtins__",
        "__globals__",
        "__subclasses__",
        "__dict__",
        "__getattribute__",
        "ctypes.",
        "marshal.",
        "pickle.loads",
        // —— Network / listening (mirrors the shell-side nc / telnet / socat
        // blacklist) ——
        "socket.",
        "http.server",
        "baseserver",
        "socketserver",
        "telnetlib.",
        "ftplib.",
        "smtplib.",
        "asyncio.start_server",
        "pwn.",
    ];

    for pattern in DANGEROUS_PYTHON_PATTERNS {
        if compact.contains(pattern) {
            return Err(format!(
                "python -c code contains blocked primitive '{pattern}'"
            ));
        }
    }
    Ok(())
}

pub(crate) fn find_has_blocked_exec_semantics(tokens: &[String]) -> Option<&str> {
    const BLOCKED_FIND_FLAGS: &[&str] = &["-delete", "-exec", "-execdir", "-ok", "-okdir"];
    fn find_primary_arg_count(tok: &str) -> usize {
        match tok {
            "-amin" | "-anewer" | "-atime" | "-cmin" | "-cnewer" | "-context" | "-ctime"
            | "-files0-from" | "-fls" | "-fprint" | "-fprint0" | "-fstype" | "-gid" | "-group"
            | "-ilname" | "-iname" | "-inum" | "-ipath" | "-iregex" | "-iwholename" | "-links"
            | "-lname" | "-maxdepth" | "-mindepth" | "-mmin" | "-mtime" | "-name" | "-newer"
            | "-newerxy" | "-path" | "-perm" | "-printf" | "-regex" | "-samefile" | "-size"
            | "-since" | "-type" | "-uid" | "-used" | "-user" | "-wholename" | "-xtype" => 1,
            "-fprintf" => 2,
            _ => 0,
        }
    }

    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok.starts_with('-') || matches!(tok, "!" | "(" | ")" | ",") {
            break;
        }
        i += 1;
    }
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if BLOCKED_FIND_FLAGS.contains(&tok) {
            return Some(tok);
        }
        if tok == "--" || matches!(tok, "!" | "(" | ")" | "," | "-a" | "-and" | "-o" | "-or") {
            i += 1;
            continue;
        }
        let arg_count = find_primary_arg_count(tok);
        if arg_count > 0 {
            i += 1 + arg_count;
            continue;
        }
        i += 1;
    }
    None
}
