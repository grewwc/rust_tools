//! Audit rules for `osascript`, the macOS AppleScript runner.
//!
//! `osascript` is a second-interpretation wrapper like `bash -c`: the script it
//! runs never passes through the segment program blacklist, and it reaches
//! capabilities the sandbox must not hand out.
//!
//! 1. **GUI input injection.** `keystroke "q" using command down` types into the
//!    *frontmost* application, not into the process the script names; System
//!    Events adds `key code`, `click`, `set frontmost` and `delete process` on
//!    top. A cleanup attempt once quit an unrelated editor this way: the
//!    command named a background test browser, the keystroke landed on
//!    whatever the user was looking at.
//! 2. **Application control.** `tell application "X" to quit` terminates a
//!    process this session does not own; `delete` removes files the `rm` path
//!    rules would reject.
//! 3. **Shell escape.** `do shell script "..."` (and Terminal's `do script`)
//!    runs shell code that no audit rule ever sees; `run script` / `load script`
//!    run a script file with the same effect.
//!
//! The rules keep the useful half — querying an app, driving a browser over
//! `execute javascript` — and block the dangerous half. What the scanner
//! cannot see is rejected: the script has to be statically available through
//! `-e`, and raw AppleScript codes (`«class ktxt»`) are refused because they
//! would let a script spell a blocked verb without naming it.
//!
//! Two scans run over every script:
//!
//! - the **code** scan sees the script with string literals emptied, so a
//!   verb is only matched where it is actually a command. A JavaScript snippet
//!   like `execute javascript "el.click()"` therefore stays allowed, while
//!   `keystroke "q"` is caught.
//! - the **literal** scan sees the whole script, so the *target* of a command
//!   (`tell application "System Events" to ...`) is matched even though the app
//!   name lives inside a string.
//!
//! Known limitation: this rule covers inline commands. A script file run
//! through an interpreter (`bash script.sh`) is not content-audited — the same
//! exemption the sandbox already grants to script files, so a file that calls
//! `osascript` is not scanned here.

/// True when `program` (a program word, possibly a path) is `osascript`.
pub(crate) fn is_osascript_program(program: &str) -> bool {
    let base = program.rsplit('/').next().unwrap_or(program);
    base.eq_ignore_ascii_case("osascript")
}

/// Validate one `osascript` invocation.
///
/// `raw_tokens` is the token list of the command with the program at index 0.
pub(crate) fn validate_osascript(raw_tokens: &[String]) -> Result<(), String> {
    let sources = script_sources(raw_tokens)?;
    for source in &sources {
        if let Some(reason) = blocked_capability(source) {
            return Err(format!("osascript {reason}"));
        }
    }
    Ok(())
}

/// Extract the AppleScript sources of an `osascript` invocation.
///
/// `osascript` takes the script either as `-e <statement>` (repeatable) or as
/// a script file / stdin. Only the `-e` form is statically visible, so every
/// other form fails closed.
fn script_sources(raw_tokens: &[String]) -> Result<Vec<String>, String> {
    let mut sources = Vec::new();
    // Token 0 is the program itself.
    let mut index = 1usize;
    while index < raw_tokens.len() {
        let token = raw_tokens[index].as_str();
        if token == "--" {
            break;
        }
        if let Some(inline) = token.strip_prefix("-e") {
            if inline.is_empty() {
                // `-e <statement>`: the statement is the next token.
                let statement = raw_tokens
                    .get(index + 1)
                    .ok_or_else(|| "option '-e' has no script argument".to_string())?;
                sources.push(statement.clone());
                index += 2;
            } else {
                // `-e<statement>`
                sources.push(inline.to_string());
                index += 1;
            }
            continue;
        }
        match token {
            // Value-taking options: `-s <flags>` (stdout formatting) and
            // `-l <language>`.
            "-s" | "-l" => {
                index += 2;
                continue;
            }
            // Interactive mode reads the script from stdin.
            "-i" => {
                return Err(
                    "interactive mode reads the script from stdin, which cannot be audited"
                        .to_string(),
                );
            }
            _ => {}
        }
        if token.starts_with('-') && token.len() > 1 {
            return Err(format!(
                "option '{token}' is not understood by the sandbox; osascript can only be \
                 audited as `osascript -e '<script>'`"
            ));
        }
        // A non-option argument is a script file.
        return Err(
            "script file arguments cannot be audited; pass the script with `-e '<script>'`"
                .to_string(),
        );
    }
    if sources.is_empty() {
        return Err(
            "no statically visible script: without `-e` osascript reads the script from \
             stdin, which cannot be audited"
                .to_string(),
        );
    }
    Ok(sources)
}

/// Detect the capabilities the sandbox refuses to grant through AppleScript.
fn blocked_capability(source: &str) -> Option<String> {
    // Raw AppleScript codes (`«class ktxt»`) can spell any verb without naming
    // it, so a script that uses them cannot be audited.
    if source.contains('«') || source.contains('»') {
        return Some(
            "raw AppleScript codes are not audited; write the script with plain keywords"
                .to_string(),
        );
    }

    let literal = normalize_literal_scan(source);
    let code = normalize_code_scan(source);

    // Addressing System Events is the gateway to GUI input and to the process
    // table of every other application.
    if literal.contains("systemevents") {
        return Some(
            "System Events drives the GUI and the processes of other applications; it is \
             blocked — inspect processes with `ps`/`pgrep` and drive a specific app \
             through its own scripting interface"
                .to_string(),
        );
    }

    // Verbs, matched on the code only: a verb inside a string literal is data
    // (a JavaScript snippet, a file name), not a command.
    const GUI_INPUT_VERBS: &[&str] = &[
        "keystroke",
        "keycode",
        "keydown",
        "keyup",
        "click",
        "frontmost",
    ];
    for verb in GUI_INPUT_VERBS {
        if code.contains(verb) {
            return Some(format!(
                "`{verb}` injects input into the frontmost application; GUI scripting is \
                 blocked"
            ));
        }
    }

    // Shell escape and script-file execution.
    if code.contains("doshellscript") {
        return Some(
            "`do shell script` runs a shell command that no audit rule sees; run the \
             command directly instead"
                .to_string(),
        );
    }
    if code.contains("doscript") {
        return Some(
            "`do script` runs a command inside another application, unaudited; run the \
             command directly instead"
                .to_string(),
        );
    }
    if code.contains("runscript") || code.contains("loadscript") {
        return Some(
            "`run script` / `load script` execute a script file that cannot be audited".to_string(),
        );
    }

    // Application control: `tell application "X" to quit` / `quit app "X"`.
    if code.contains("toquit") || code.contains("quitapp") {
        return Some(
            "`quit` terminates applications this session does not own; use the \
             application's own shutdown path, or terminate a process this session \
             registered"
                .to_string(),
        );
    }

    // Destructive file operations through an app (Finder `delete`, System
    // Events `delete process`) bypass the `rm` path rules.
    if code.contains("delete") {
        return Some(
            "`delete` removes files or processes behind the sandbox's back; use the \
             session's own file tools instead"
                .to_string(),
        );
    }

    None
}

/// Normalize the whole script for the literal scan: lowercased and without
/// whitespace, so `tell application "System Events"` and
/// `tell application "system events"` match alike.
fn normalize_literal_scan(source: &str) -> String {
    source
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

/// Normalize the script for the code scan: lowercased, without whitespace and
/// without the content of string literals, so that a verb is only matched
/// where it is a command rather than data.
fn normalize_code_scan(source: &str) -> String {
    let mut out = String::new();
    let mut chars = source.chars().peekable();
    let mut in_string = false;
    let mut in_block_comment = false;
    while let Some(ch) = chars.next() {
        if in_block_comment {
            if ch == '(' && chars.peek() == Some(&'*') {
                chars.next();
            } else if ch == '*' && chars.peek() == Some(&')') {
                chars.next();
                in_block_comment = false;
            }
            continue;
        }
        if in_string {
            if ch == '\\' {
                // Escaped character inside a string literal (`\"` keeps the
                // string open instead of ending it).
                chars.next();
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            // `--` starts a line comment.
            '-' if chars.peek() == Some(&'-') => {
                for next in chars.by_ref() {
                    if next == '\n' {
                        break;
                    }
                }
            }
            // `(*` starts a block comment.
            '(' if chars.peek() == Some(&'*') => {
                chars.next();
                in_block_comment = true;
            }
            _ => {
                if !ch.is_whitespace() {
                    out.push(ch.to_ascii_lowercase());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| (*p).to_string()).collect()
    }

    fn blocked(script: &str) -> String {
        validate_osascript(&tokens(&["osascript", "-e", script])).unwrap_err()
    }

    #[test]
    fn keystroke_without_system_events_is_blocked() {
        let err = blocked("keystroke \"q\" using command down");
        assert!(err.contains("keystroke"), "got: {err}");
        assert!(err.contains("GUI scripting is blocked"), "got: {err}");
    }

    #[test]
    fn system_events_keystroke_is_blocked() {
        let err = blocked(
            "tell application \"System Events\" to tell (first process whose unix id is 1) to keystroke \"q\" using command down",
        );
        assert!(err.contains("System Events"), "got: {err}");
    }

    #[test]
    fn system_events_case_variants_are_blocked() {
        for script in [
            "tell application \"system events\" to key code 17",
            "tell application \"SYSTEM EVENTS\" to click button 1",
            "tell application \"System Events\" to set frontmost of process \"X\" to true",
        ] {
            let err = blocked(script);
            assert!(err.contains("System Events"), "got: {err}");
        }
    }

    #[test]
    fn deleting_a_process_or_file_is_blocked() {
        // System Events is blocked as a whole, so its `delete process` form is
        // caught before the verb scan.
        let err = blocked("tell application \"System Events\" to delete process \"Trae\"");
        assert!(err.contains("System Events"), "got: {err}");
        // A file delete through another app is caught by the verb scan.
        let err = blocked("tell application \"Finder\" to delete POSIX file \"/Users/x/y\"");
        assert!(err.contains("`delete`"), "got: {err}");
    }

    #[test]
    fn quitting_an_application_is_blocked() {
        let err = blocked("tell application \"Google Chrome\" to quit");
        assert!(err.contains("`quit` terminates applications"), "got: {err}");
        let err = blocked("quit app \"Safari\"");
        assert!(err.contains("`quit` terminates applications"), "got: {err}");
    }

    #[test]
    fn shell_escape_is_blocked() {
        let err = blocked("do shell script \"rm -rf /tmp/x\"");
        assert!(err.contains("do shell script"), "got: {err}");
        let err = blocked("tell application \"Terminal\" to do script \"ls\"");
        assert!(err.contains("do script"), "got: {err}");
        let err = blocked("run script \"/tmp/other.applescript\"");
        assert!(err.contains("run script"), "got: {err}");
    }

    #[test]
    fn raw_applescript_codes_are_blocked() {
        let err = blocked("tell application \"System Events\" to «class ktst»");
        assert!(err.contains("raw AppleScript codes"), "got: {err}");
    }

    #[test]
    fn script_file_and_stdin_forms_fail_closed() {
        let err = validate_osascript(&tokens(&["osascript", "script.applescript"])).unwrap_err();
        assert!(err.contains("script file"), "got: {err}");
        let err = validate_osascript(&tokens(&["osascript"])).unwrap_err();
        assert!(err.contains("no statically visible script"), "got: {err}");
        let err = validate_osascript(&tokens(&["osascript", "-i"])).unwrap_err();
        assert!(err.contains("stdin"), "got: {err}");
    }

    #[test]
    fn benign_scripts_are_allowed() {
        validate_osascript(&tokens(&[
            "osascript",
            "-e",
            "tell application \"Finder\" to get name of startup disk",
        ]))
        .unwrap();
        // Browser scripting stays available (the browser driver relies on it),
        // including JavaScript that mentions a click.
        validate_osascript(&tokens(&[
            "osascript",
            "-e",
            "tell application \"Google Chrome\" to execute javascript \"document.querySelector('#ok').click()\"",
        ]))
        .unwrap();
        // An escaped quote keeps the string open; the text behind it stays
        // data instead of becoming code.
        validate_osascript(&tokens(&[
            "osascript",
            "-e",
            "tell application \"Google Chrome\" to execute javascript \"var s = \\\"click\\\"; s\"",
        ]))
        .unwrap();
    }

    #[test]
    fn program_detection_accepts_paths() {
        assert!(is_osascript_program("osascript"));
        assert!(is_osascript_program("/usr/bin/osascript"));
        assert!(!is_osascript_program("osascriptx"));
        assert!(!is_osascript_program("open"));
    }
}
