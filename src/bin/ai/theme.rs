//! Terminal color theme.
//!
//! The terminal renderer used to hardcode ANSI escape strings as `const`s.
//! Since those colors were duplicated in ~13 modules, restyling the terminal
//! required editing Rust source. This module turns the palette into a data
//! model loaded from a VSCode-style JSON theme file (hot-reloaded at each
//! turn boundary), so styling changes are a config edit, not a recompile.
//!
//! # Theme JSON format (VSCode-like keys, opencode-style values)
//!
//! ```json
//! {
//!   "name": "my-theme",
//!   "type": "dark",
//!   "defs": {
//!     "brand": "#7C9CF5",
//!     "panel": "#242832"
//!   },
//!   "colors": {
//!     "accent.primary": "brand",
//!     "accent.submittedBar": "brand",
//!     "code.background": "panel",
//!     "markdown.math": "\u001b[95m"
//!   }
//! }
//! ```
//!
//! `colors` maps flat dotted keys (like VSCode's `editor.foreground`) to a
//! color value; every key falls back to the built-in default, so a custom
//! theme only needs to list what it changes. Each value accepts:
//!
//! - `#RRGGBB` (optionally `#RRGGBBAA`, alpha ignored) — truecolor.
//! - `0`-`255` (a JSON number) — 256-color palette index.
//! - `"none"` — emits no SGR sequence, so the terminal's own color shows.
//! - a `defs` key — a reusable definition holding any of the above (or a raw
//!   escape); chains resolve up to 8 deep, cycles fall back to the default.
//! - a raw ANSI escape sequence (a string starting with ESC, e.g.
//!   `"\u001b[95m"`).
//!
//! Unknown keys and unparsable values are ignored.
//!
//! `name` and `type` (`"dark"` / `"light"`) describe the terminal a theme was
//! designed for; the renderer reads the `colors` map alone, so they are
//! documentation for the person writing the file.
//!
//! # Color slots
//!
//! - `accent.*` — terminal chrome: status glyphs, tool names, file paths,
//!   commands, rules, the `❯` prompt marker (`accent.prompt`, which follows
//!   `accent.primary` when a theme omits it) and the echo of submitted input.
//! - `markdown.*` — assistant prose: body, strong, heading, links, inline code
//!   and math.
//!
//! # Theme resolution
//!
//! On first use, `current()` resolves in this order:
//! 1. `ai.theme.file` (config key) — absolute path to a theme JSON.
//! 2. `ai.theme` (config key) — theme name:
//!    - `~/.config/rust_tools/themes/<name>.json` (user override), then
//!    - a built-in theme embedded from `src/bin/ai/builtin_themes/`.
//! 3. The built-in default palette.
//!
//! The theme is resolved lazily on first use and re-resolved at every turn
//! boundary (`theme::refresh()`, fired at the top of `run_turn`), so
//! `ai.theme` / `ai.theme.file` edits — or edits to a custom theme JSON —
//! apply on the next user message without restarting the process.
//!
//! `/theme` sets a session-scoped override that takes precedence over the
//! config resolution above (and therefore survives turn-boundary refreshes)
//! without touching the config file; `/theme clear` drops it.
//!
//! # Style codes vs. colors
//!
//! `RESET`/`BOLD`/`DIM` are SGR style parameters, not colors, so they stay
//! consts. `accent.input` is also exposed as an RGB tuple
//! (`accent_input_rgb`) because the multiline editor dims/blends it
//! numerically.

use std::{
    fs,
    path::PathBuf,
    sync::Mutex,
};

use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::commonw::{configw, utils::expanduser};

pub(in crate::ai) const RESET: &str = "\x1b[0m";
pub(in crate::ai) const BOLD: &str = "\x1b[1m";
pub(in crate::ai) const DIM: &str = "\x1b[2m";

/// Resolved terminal palette. All ANSI strings are `&'static str` because each
/// theme instance is built and leaked once per resolution; `refresh()`
/// replaces the cached instance (the previous one stays leaked, see below).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Theme {
    // ── Accents: UI / status text (previously `ACCENT_*` consts) ──────────
    pub(crate) accent_primary: &'static str,
    pub(crate) accent_tool_name: &'static str,
    pub(crate) accent_secondary: &'static str,
    pub(crate) accent_command: &'static str,
    pub(crate) accent_muted: &'static str,
    pub(crate) accent_input: &'static str,
    /// RGB tuple of `accent.input`, used by the multiline editor to compute
    /// derived shades numerically.
    pub(crate) accent_input_rgb: (u8, u8, u8),
    pub(crate) accent_success: &'static str,
    pub(crate) accent_emphasized_output: &'static str,
    pub(crate) accent_warn: &'static str,
    pub(crate) accent_danger: &'static str,
    pub(crate) accent_marked: &'static str,
    pub(crate) accent_rule: &'static str,
    /// Echoed user input after submit: the theme's bright signature warm hue
    /// (amber/yellow on dark themes, deep purple on light) — deliberately NOT
    /// the white/gray markdown body colors, so the user's own text stays
    /// visible and distinguishable from assistant output. The `❯` marker in
    /// `prompt/multiline/multiline_ui.rs` keeps the submit boundary.
    pub(crate) accent_submitted: &'static str,
    /// Left emphasis bar for echoed user input after submit. A theme that sets
    /// `accent.submittedBar` uses that color; a theme that changes other colors
    /// but not the bar mirrors its resolved `accent.success`; a theme that
    /// overrides no known color keeps the compiled-in default.
    pub(crate) accent_submitted_bar: &'static str,
    /// Input affordance: the `❯` prompt marker in front of the text the user
    /// is typing. It is a separate slot from `accent.success` so green can
    /// keep meaning "done" while the prompt marker carries the theme's brand
    /// hue. A theme that does not set `accent.prompt` inherits its resolved
    /// `accent.primary`.
    pub(crate) accent_prompt: &'static str,
    // ── Markdown prose (previously `MARKDOWN_*` consts in render/mod.rs) ───
    pub(crate) markdown_body: &'static str,
    pub(crate) markdown_strong: &'static str,
    pub(crate) markdown_heading: &'static str,
    pub(crate) markdown_accent: &'static str,
    pub(crate) markdown_code_fg: &'static str,
    pub(crate) markdown_math: &'static str,
    // ── Code block syntax (previously `MONOKAI_*` consts in render/code.rs) ─
    pub(crate) code_background: &'static str,
    pub(crate) code_foreground: &'static str,
    pub(crate) code_comment: &'static str,
    pub(crate) code_keyword: &'static str,
    pub(crate) code_string: &'static str,
    pub(crate) code_number: &'static str,
    pub(crate) code_type: &'static str,
    pub(crate) code_dim: &'static str,
}

impl Theme {
    /// The built-in default palette. Values must match
    /// `src/bin/ai/builtin_themes/default.json` (a test keeps them in sync);
    /// that file is the copy-paste template for custom themes.
    pub(crate) fn default_theme() -> Theme {
        Theme {
            accent_primary: "\x1b[38;2;124;156;245m",
            accent_tool_name: "\x1b[38;2;220;227;240m",
            accent_secondary: "\x1b[38;2;169;155;240m",
            accent_command: "\x1b[38;2;116;199;217m",
            accent_muted: "\x1b[38;2;138;147;166m",
            accent_input: "\x1b[38;2;221;227;238m",
            accent_input_rgb: (221, 227, 238),
            accent_success: "\x1b[38;2;143;217;168m",
            accent_emphasized_output: "\x1b[38;2;165;214;255m",
            accent_warn: "\x1b[38;2;232;185;107m",
            accent_danger: "\x1b[38;2;232;131;139m",
            accent_marked: "\x1b[38;2;245;199;126m",
            accent_rule: "\x1b[38;2;107;120;144m",
            accent_submitted: "\x1b[38;2;224;192;136m",
            accent_submitted_bar: "\x1b[38;2;124;156;245m",
            accent_prompt: "\x1b[38;2;124;156;245m",
            markdown_body: "\x1b[38;2;211;217;228m",
            markdown_strong: "\x1b[38;2;233;237;245m",
            markdown_heading: "\x1b[38;2;245;248;252m",
            markdown_accent: "\x1b[38;2;127;179;240m",
            markdown_code_fg: "\x1b[38;2;196;181;253m",
            markdown_math: "\x1b[38;2;239;168;200m",
            code_background: "\x1b[48;2;36;40;50m",
            code_foreground: "\x1b[38;2;226;231;240m",
            code_comment: "\x1b[38;2;138;149;168m",
            code_keyword: "\x1b[38;2;224;139;180m",
            code_string: "\x1b[38;2;155;214;160m",
            code_number: "\x1b[38;2;229;192;123m",
            code_type: "\x1b[38;2;116;199;217m",
            code_dim: "\x1b[38;2;107;116;136m",
        }
    }

    /// Parse a VSCode-style theme JSON, merging over the built-in default.
    /// Returns `None` only when the document is not valid JSON (or `colors`
    /// is present but not an object). A non-object `defs` is ignored.
    pub(crate) fn from_json(json: &str) -> Option<Theme> {
        let root: Value = serde_json::from_str(json).ok()?;
        let colors = match root.get("colors") {
            Some(Value::Object(map)) => map,
            Some(_) => return None,
            None => return Some(Theme::default_theme()),
        };
        let empty_defs = serde_json::Map::new();
        let defs = match root.get("defs") {
            Some(Value::Object(map)) => map,
            _ => &empty_defs,
        };
        let mut t = Theme::default_theme();
        let mut submitted_bar_overridden = false;
        let mut prompt_overridden = false;
        for (key, value) in colors {
            match key.as_str() {
                "accent.primary" => set_color(&mut t.accent_primary, value, defs, false),
                "accent.toolName" => set_color(&mut t.accent_tool_name, value, defs, false),
                "accent.secondary" => set_color(&mut t.accent_secondary, value, defs, false),
                "accent.command" => set_color(&mut t.accent_command, value, defs, false),
                "accent.muted" => set_color(&mut t.accent_muted, value, defs, false),
                "accent.input" => {
                    set_color(&mut t.accent_input, value, defs, false);
                    if let Some(rgb) = resolve_rgb(value, defs, 0) {
                        t.accent_input_rgb = rgb;
                    }
                }
                "accent.success" => set_color(&mut t.accent_success, value, defs, false),
                "accent.emphasizedOutput" => {
                    set_color(&mut t.accent_emphasized_output, value, defs, false)
                }
                "accent.warn" => set_color(&mut t.accent_warn, value, defs, false),
                "accent.danger" => set_color(&mut t.accent_danger, value, defs, false),
                "accent.marked" => set_color(&mut t.accent_marked, value, defs, false),
                "accent.rule" => set_color(&mut t.accent_rule, value, defs, false),
                "accent.submitted" => set_color(&mut t.accent_submitted, value, defs, false),
                "accent.submittedBar" => {
                    if let Some(ansi) = resolve_color(value, defs, false, 0) {
                        t.accent_submitted_bar = leak(ansi);
                        submitted_bar_overridden = true;
                    }
                }
                "accent.prompt" => {
                    if let Some(ansi) = resolve_color(value, defs, false, 0) {
                        t.accent_prompt = leak(ansi);
                        prompt_overridden = true;
                    }
                }
                "markdown.body" => set_color(&mut t.markdown_body, value, defs, false),
                "markdown.strong" => set_color(&mut t.markdown_strong, value, defs, false),
                "markdown.heading" => set_color(&mut t.markdown_heading, value, defs, false),
                "markdown.accent" => set_color(&mut t.markdown_accent, value, defs, false),
                "markdown.codeFg" => set_color(&mut t.markdown_code_fg, value, defs, false),
                "markdown.math" => set_color(&mut t.markdown_math, value, defs, false),
                "code.background" => set_color(&mut t.code_background, value, defs, true),
                "code.foreground" => set_color(&mut t.code_foreground, value, defs, false),
                "code.comment" => set_color(&mut t.code_comment, value, defs, false),
                "code.keyword" => set_color(&mut t.code_keyword, value, defs, false),
                "code.string" => set_color(&mut t.code_string, value, defs, false),
                "code.number" => set_color(&mut t.code_number, value, defs, false),
                "code.type" => set_color(&mut t.code_type, value, defs, false),
                "code.dim" => set_color(&mut t.code_dim, value, defs, false),
                _ => {} // unknown keys are ignored so themes stay forward-compatible
            }
        }
        // A theme JSON that overrides no known color must stay identical to
        // `Theme::default_theme()` (see `from_json_unknown_keys_ignored`), so the
        // success fallback only applies once the theme actually changed a color.
        // Resolving here, after the loop, keeps the bar independent of key order.
        if !submitted_bar_overridden && t != Theme::default_theme() {
            // Resolve after all theme keys are applied, so a partial theme that
            // customizes `accent.success` automatically gives its user-input
            // bar the same color regardless of JSON object iteration order.
            t.accent_submitted_bar = t.accent_success;
        }
        // The prompt marker follows the theme's primary hue when the theme
        // does not name one: an input affordance in the brand color reads as
        // "type here" instead of "success", and every built-in theme sets
        // `accent.prompt` to that same hue.
        if !prompt_overridden {
            t.accent_prompt = t.accent_primary;
        }
        Some(t)
    }

    /// Resolve the active theme from config (see module docs for the order).
    fn load_from_config() -> Theme {
        // A live `/theme` override wins over config so it survives the
        // turn-boundary `refresh()` that re-runs this resolution.
        if let Some((_, theme)) = THEME_OVERRIDE.lock().unwrap().as_ref() {
            return theme.clone();
        }
        let cfg = configw::get_all_config();
        if let Some(path) = cfg.get_opt(AiConfig::THEME_FILE) {
            if let Ok(content) = fs::read_to_string(expanduser(&path).as_ref())
                && let Some(theme) = Theme::from_json(&content)
            {
                return theme;
            }
        }
        let name = cfg.get_opt(AiConfig::THEME).unwrap_or_default();
        if !name.is_empty() {
            if let Some(theme) = theme_from_name(&name) {
                return theme;
            }
        }
        Theme::default_theme()
    }

    /// A short sample of the agent's terminal UI painted with this palette:
    /// tool status lines, prose, a code block and the prompt marker. It exists
    /// so a palette can be judged before it is switched on.
    pub(crate) fn preview_lines(&self) -> Vec<String> {
        vec![
            format!(
                "  {}{}{} {}{}{}",
                self.accent_primary, "●", RESET, self.accent_tool_name, "read_file", RESET
            ),
            format!(
                "  {}{}{} {}{}{}",
                self.accent_success, "✓", RESET, self.accent_tool_name, "apply_patch", RESET
            ),
            format!(
                "  {}{}{} {}{}{}",
                self.accent_danger, "×", RESET, self.accent_tool_name, "execute_command", RESET
            ),
            format!(
                "  {}{}{} {}{}{} {}{}{}",
                self.accent_rule,
                "│",
                RESET,
                self.accent_muted,
                "·",
                RESET,
                self.accent_secondary,
                "src/main.rs",
                RESET
            ),
            format!(
                "  {}{}{} {}{}{}",
                self.accent_rule,
                "│",
                RESET,
                self.accent_command,
                "cargo test --lib",
                RESET
            ),
            format!(
                "  {}{}{} {}{}{}",
                self.accent_rule,
                "│",
                RESET,
                self.accent_emphasized_output,
                "important: 3 warnings",
                RESET
            ),
            String::new(),
            format!("{}{}## Heading{}", self.markdown_heading, RESET, RESET),
            [
                (self.markdown_body, "Body with "),
                (self.markdown_strong, "strong"),
                (self.markdown_body, ", "),
                (self.markdown_accent, "a link"),
                (self.markdown_body, ", and "),
                (self.markdown_code_fg, "inline_code()"),
                (self.markdown_body, "."),
            ]
            .iter()
            .map(|(color, text)| format!("{color}{text}{RESET}"))
            .collect::<String>(),
            String::new(),
            format!("{}{}╭─ rust{}", self.code_background, self.code_dim, RESET),
            [
                (self.code_dim, "  1 │"),
                (self.code_comment, "// a comment"),
                (self.code_keyword, "fn"),
                (self.code_foreground, " main() { let n = "),
                (self.code_number, "42"),
                (self.code_foreground, "; }"),
            ]
            .iter()
            .map(|(color, text)| format!("{}{}{}", self.code_background, color, text))
            .collect::<String>()
                + RESET,
            format!(
                "{}{}╰{}{}",
                self.code_background,
                self.code_dim,
                "─".repeat(20),
                RESET
            ),
            String::new(),
            format!(
                "{}{}{}❯{} your input{}",
                BOLD, self.accent_prompt, RESET, self.accent_input, RESET
            ),
            format!(
                "{}{}╭─{} {}{}❯{} {}{}{}╮{}",
                BOLD,
                self.accent_submitted_bar,
                RESET,
                BOLD,
                self.accent_prompt,
                RESET,
                BOLD,
                self.accent_submitted_bar,
                "─".repeat(18),
                RESET
            ),
            format!(
                "{}{}│{} {}{}your submitted input{} {}{}│{}",
                BOLD,
                self.accent_submitted_bar,
                RESET,
                BOLD,
                self.accent_submitted,
                RESET,
                BOLD,
                self.accent_submitted_bar,
                RESET
            ),
            format!(
                "{}{}╰{}╯{}",
                BOLD,
                self.accent_submitted_bar,
                "─".repeat(22),
                RESET
            ),
        ]
    }
}

/// The process-wide active theme. Resolved lazily on first use, then replaced
/// by `refresh()` at turn boundaries so config changes apply live. Instances
/// are leaked: fields are `&'static str` so call sites can format them without
/// borrowing, and a refresh leaks one small (~300 B) palette, which is
/// negligible for a rare re-theming event.
static CURRENT_THEME: Mutex<Option<&'static Theme>> = Mutex::new(None);

/// The active theme for the current process. Cheap after first access.
pub(crate) fn current() -> &'static Theme {
    let mut guard = CURRENT_THEME.lock().unwrap();
    *guard.get_or_insert_with(|| Box::leak(Box::new(Theme::load_from_config())))
}

/// Re-resolve the theme from the current config. Called at the start of every
/// turn (top of `run_turn`), right after `configw::refresh()` drops the cached
/// config, so `ai.theme` / `ai.theme.file` edits — or edits to a custom theme
/// JSON — apply on the next user message without restarting the process.
pub(crate) fn refresh() {
    let fresh = Box::leak(Box::new(Theme::load_from_config()));
    *CURRENT_THEME.lock().unwrap() = Some(fresh);
}

/// Session-scoped live override set by `/theme` (labelled for `/theme current`
/// display). It never touches the config file: set `ai.theme` in config to make
/// a choice permanent.
static THEME_OVERRIDE: Mutex<Option<(String, Theme)>> = Mutex::new(None);

/// Switch the live palette immediately; paint code reads `current()`, so the
/// next terminal output already uses the new theme.
pub(crate) fn set_override(label: String, theme: Theme) {
    let fresh = Box::leak(Box::new(theme.clone()));
    *CURRENT_THEME.lock().unwrap() = Some(fresh);
    *THEME_OVERRIDE.lock().unwrap() = Some((label, theme));
}

/// Drop the session override and re-resolve from config.
pub(crate) fn clear_override() {
    THEME_OVERRIDE.lock().unwrap().take();
    refresh();
}

/// Label of the live override (`/theme <name>` argument or `file:<path>`),
/// or `None` when the theme comes from config.
pub(crate) fn override_label() -> Option<String> {
    THEME_OVERRIDE
        .lock()
        .unwrap()
        .as_ref()
        .map(|(label, _)| label.clone())
}

/// Resolve a theme by name: user override dir first, then built-ins.
pub(crate) fn theme_from_name(name: &str) -> Option<Theme> {
    let user_path = user_themes_dir().join(format!("{name}.json"));
    if let Ok(content) = fs::read_to_string(&user_path)
        && let Some(theme) = Theme::from_json(&content)
    {
        return Some(theme);
    }
    let json = builtin_theme_json(name)?;
    Theme::from_json(json)
}

/// Built-in theme names: the single source for `/theme` listing, tab
/// completion, and the parse test below (a name missing from the `match` in
/// `builtin_theme_json` fails that test).
pub(crate) const BUILTIN_THEME_NAMES: &[&str] = &[
    "default",
    "monokai",
    "light",
    "dracula",
    "one-dark",
    "tokyo-night",
    "nord",
    "catppuccin",
    "everforest",
    "ayu",
    "gruvbox",
    "kanagawa",
    "matrix",
];

fn builtin_theme_json(name: &str) -> Option<&'static str> {
    match name {
        "default" => Some(include_str!("builtin_themes/default.json")),
        "monokai" => Some(include_str!("builtin_themes/monokai.json")),
        "light" => Some(include_str!("builtin_themes/light.json")),
        "dracula" => Some(include_str!("builtin_themes/dracula.json")),
        "one-dark" => Some(include_str!("builtin_themes/one-dark.json")),
        "tokyo-night" => Some(include_str!("builtin_themes/tokyo-night.json")),
        "nord" => Some(include_str!("builtin_themes/nord.json")),
        "catppuccin" => Some(include_str!("builtin_themes/catppuccin.json")),
        "everforest" => Some(include_str!("builtin_themes/everforest.json")),
        "ayu" => Some(include_str!("builtin_themes/ayu.json")),
        "gruvbox" => Some(include_str!("builtin_themes/gruvbox.json")),
        "kanagawa" => Some(include_str!("builtin_themes/kanagawa.json")),
        "matrix" => Some(include_str!("builtin_themes/matrix.json")),
        _ => None,
    }
}

/// User theme override directory: `~/.config/rust_tools/themes/`.
pub(crate) fn user_themes_dir() -> PathBuf {
    PathBuf::from(expanduser("~/.config/rust_tools/themes").as_ref())
}

/// Every theme name `/theme` can switch to: built-ins plus the `*.json` stems
/// in the user override dir (a user file shadows the built-in of the same
/// name). Sorted for stable display.
pub(crate) fn available_theme_names() -> Vec<String> {
    let mut names: Vec<String> = BUILTIN_THEME_NAMES.iter().map(|s| s.to_string()).collect();
    if let Ok(entries) = fs::read_dir(user_themes_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && !names.iter().any(|n| n == stem)
            {
                names.push(stem.to_string());
            }
        }
    }
    names.sort();
    names
}

/// Assign a resolved color to a palette slot; unparsable values keep the default.
fn set_color(
    field: &mut &'static str,
    value: &Value,
    defs: &serde_json::Map<String, Value>,
    is_bg: bool,
) {
    if let Some(ansi) = resolve_color(value, defs, is_bg, 0) {
        *field = leak(ansi);
    }
}

/// Resolve a theme color value to an ANSI SGR sequence (opencode-style):
/// `#RRGGBB`/`#RRGGBBAA` truecolor, a `0`-`255` palette index, `"none"` for
/// the terminal default (empty string), a `defs` reference, or a raw ESC
/// sequence. Returns `None` for anything else, including reference chains
/// deeper than 8 (cycles fall back to the default instead of hanging).
fn resolve_color(
    value: &Value,
    defs: &serde_json::Map<String, Value>,
    is_bg: bool,
    depth: u8,
) -> Option<String> {
    if depth > 8 {
        return None;
    }
    match value {
        Value::Number(n) => {
            let idx = n.as_u64()?;
            if idx > 255 {
                return None;
            }
            let code = if is_bg { 48 } else { 38 };
            Some(format!("\x1b[{code};5;{idx}m"))
        }
        Value::String(s) => {
            if s == "none" {
                return Some(String::new());
            }
            if let Some(target) = defs.get(s) {
                return resolve_color(target, defs, is_bg, depth + 1);
            }
            color_to_ansi(s, is_bg)
        }
        _ => None,
    }
}

/// Resolve a theme color value to an RGB tuple for numeric blending
/// (`accent.input`): hex directly or through `defs`, else the standard
/// 256-palette RGB for indices. `"none"` and unparsable values keep the
/// default blend base.
fn resolve_rgb(
    value: &Value,
    defs: &serde_json::Map<String, Value>,
    depth: u8,
) -> Option<(u8, u8, u8)> {
    if depth > 8 {
        return None;
    }
    match value {
        Value::Number(n) => {
            let idx = n.as_u64()?;
            if idx > 255 {
                return None;
            }
            Some(ansi256_rgb(idx as u8))
        }
        Value::String(s) => {
            if let Some(target) = defs.get(s) {
                return resolve_rgb(target, defs, depth + 1);
            }
            hex_rgb(s)
        }
        _ => None,
    }
}

/// RGB of a 256-color palette index (standard xterm palette: 0-15 system
/// colors, 16-231 a 6x6x6 cube, 232-255 a grayscale ramp).
fn ansi256_rgb(idx: u8) -> (u8, u8, u8) {
    const SYSTEM: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    if idx < 16 {
        SYSTEM[idx as usize]
    } else if idx < 232 {
        let i = idx - 16;
        (
            LEVELS[(i / 36) as usize],
            LEVELS[((i / 6) % 6) as usize],
            LEVELS[(i % 6) as usize],
        )
    } else {
        let v = 8 + 10 * (idx - 232);
        (v, v, v)
    }
}

/// Each resolved theme instance owns its strings and leaks them to get
/// `&'static str` fields; this keeps call-site ergonomics identical to the
/// old consts (`format!` / `push_str` without borrowing). `refresh()` leaks
/// one small instance per re-theming event.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Convert a theme string to an ANSI color sequence: `#RRGGBB` (optionally
/// `#RRGGBBAA`, alpha ignored) becomes an RGB SGR code; a string starting
/// with ESC is passed through verbatim (raw ANSI, e.g. 256-color or style
/// sequences). Numbers, `"none"` and `defs` references are resolved by
/// `resolve_color` before reaching here.
fn color_to_ansi(value: &str, is_bg: bool) -> Option<String> {
    if value.starts_with('\u{1b}') {
        return Some(value.to_string());
    }
    let (r, g, b) = hex_rgb(value)?;
    let code = if is_bg { 48 } else { 38 };
    Some(format!("\x1b[{code};2;{r};{g};{b}m"))
}

/// Parse `#RRGGBB` / `#RRGGBBAA` into an RGB tuple.
fn hex_rgb(value: &str) -> Option<(u8, u8, u8)> {
    let hex = value.strip_prefix('#')?;
    if !matches!(hex.len(), 6 | 8) {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_json_template_matches_default_theme() {
        // Keeps the user-facing template (builtin_themes/default.json) in sync
        // with the Rust fallback palette.
        let from_template = Theme::from_json(include_str!("builtin_themes/default.json"))
            .expect("default.json must be valid");
        assert_eq!(from_template, Theme::default_theme());
    }

    #[test]
    fn from_json_overrides_subset_and_merges_rest() {
        let json = r##"{"colors": {"accent.primary": "#FF0000", "code.keyword": "#00FF00"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, "\x1b[38;2;255;0;0m");
        assert_eq!(t.code_keyword, "\x1b[38;2;0;255;0m");
        // untouched keys fall back to the default palette
        assert_eq!(t.accent_success, Theme::default_theme().accent_success);
        assert_eq!(t.code_background, Theme::default_theme().code_background);
    }

    #[test]
    fn from_json_code_background_uses_bg_sequence() {
        let json = r##"{"colors": {"code.background": "#101010"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.code_background, "\x1b[48;2;16;16;16m");
    }

    #[test]
    fn from_json_invalid_hex_keeps_default() {
        let json = r##"{"colors": {"accent.primary": "not-a-color", "code.type": "#12"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, Theme::default_theme().accent_primary);
        assert_eq!(t.code_type, Theme::default_theme().code_type);
    }

    #[test]
    fn from_json_upper_case_hex_and_alpha_ignored() {
        let json = r##"{"colors": {"accent.success": "#FFAABBCC"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_success, "\x1b[38;2;255;170;187m");
    }

    #[test]
    fn from_json_raw_escape_passthrough() {
        let json = r#"{"colors": {"markdown.math": "\u001b[38;5;123m"}}"#;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.markdown_math, "\x1b[38;5;123m");
    }

    #[test]
    fn from_json_input_also_updates_rgb() {
        let json = r##"{"colors": {"accent.input": "#112233"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_input, "\x1b[38;2;17;34;51m");
        assert_eq!(t.accent_input_rgb, (17, 34, 51));
    }

    #[test]
    fn from_json_submitted_bar_defaults_to_resolved_success() {
        let json = r##"{"colors": {"accent.success": "#112233"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_success, "\x1b[38;2;17;34;51m");
        assert_eq!(t.accent_submitted_bar, t.accent_success);

        let json = r##"{"colors": {"accent.success": "#112233", "accent.submittedBar": "#445566"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_submitted_bar, "\x1b[38;2;68;85;102m");
    }

    #[test]
    fn from_json_prompt_defaults_to_resolved_primary() {
        // A theme that does not name a prompt color gets the theme's primary
        // hue, so the input marker keeps the palette's brand color.
        let json = r##"{"colors": {"accent.primary": "#112233"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_prompt, t.accent_primary);
        assert_eq!(t.accent_prompt, "\x1b[38;2;17;34;51m");

        let json = r##"{"colors": {"accent.primary": "#112233", "accent.prompt": "#445566"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_prompt, "\x1b[38;2;68;85;102m");
    }

    #[test]
    fn from_json_unknown_keys_ignored() {
        let json = r##"{"colors": {"editor.background": "#000000", "accent.bogus": "#000000"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t, Theme::default_theme());
    }

    #[test]
    fn from_json_malformed_returns_none() {
        assert!(Theme::from_json("{not json").is_none());
        assert!(Theme::from_json(r#"{"colors": 42}"#).is_none());
    }

    #[test]
    fn from_json_missing_colors_is_default() {
        assert_eq!(Theme::from_json(r#"{"name": "x"}"#).unwrap(), Theme::default_theme());
    }

    #[test]
    fn from_json_defs_reference_resolves() {
        let json = r##"{"defs": {"brand": "#112233", "panel": 235},
            "colors": {"accent.primary": "brand", "code.background": "panel"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, "\x1b[38;2;17;34;51m");
        assert_eq!(t.code_background, "\x1b[48;5;235m");
    }

    #[test]
    fn from_json_defs_chain_resolves_and_cycle_falls_back() {
        let json = r##"{"defs": {"a": "b", "b": "#AABBCC"},
            "colors": {"accent.primary": "a"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, "\x1b[38;2;170;187;204m");
        // cycles terminate and fall back to the default palette
        let json = r##"{"defs": {"a": "b", "b": "a"},
            "colors": {"accent.primary": "a"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, Theme::default_theme().accent_primary);
    }

    #[test]
    fn from_json_none_is_transparent_and_number_is_256color() {
        let json = r##"{"colors": {"accent.primary": "none", "accent.warn": 214,
            "code.background": 235}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, "");
        assert_eq!(t.accent_warn, "\x1b[38;5;214m");
        assert_eq!(t.code_background, "\x1b[48;5;235m");
    }

    #[test]
    fn from_json_bad_numbers_and_non_object_defs_ignored() {
        let json = r##"{"defs": 42, "colors": {"accent.primary": 999,
            "accent.warn": -1, "accent.danger": 1.5}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_primary, Theme::default_theme().accent_primary);
        assert_eq!(t.accent_warn, Theme::default_theme().accent_warn);
        assert_eq!(t.accent_danger, Theme::default_theme().accent_danger);
    }

    #[test]
    fn from_json_input_rgb_follows_defs_and_numbers() {
        let json = r##"{"defs": {"ink": "#112233"}, "colors": {"accent.input": "ink"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_input, "\x1b[38;2;17;34;51m");
        assert_eq!(t.accent_input_rgb, (17, 34, 51));
        let json = r##"{"colors": {"accent.input": 196}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_input, "\x1b[38;5;196m");
        assert_eq!(t.accent_input_rgb, (255, 0, 0));
        // "none" keeps the default blend base (nothing to blend toward)
        let json = r##"{"colors": {"accent.input": "none"}}"##;
        let t = Theme::from_json(json).unwrap();
        assert_eq!(t.accent_input, "");
        assert_eq!(t.accent_input_rgb, Theme::default_theme().accent_input_rgb);
    }

    #[test]
    fn ansi256_rgb_spot_checks() {
        assert_eq!(ansi256_rgb(0), (0, 0, 0));
        assert_eq!(ansi256_rgb(9), (255, 0, 0));
        assert_eq!(ansi256_rgb(15), (255, 255, 255));
        assert_eq!(ansi256_rgb(196), (255, 0, 0));
        assert_eq!(ansi256_rgb(232), (8, 8, 8));
        assert_eq!(ansi256_rgb(255), (238, 238, 238));
    }

    #[test]
    fn all_builtin_themes_parse_and_differ_from_default() {
        // Iterating the shared name table also locks it to the `match` in
        // `builtin_theme_json`: an unregistered name panics below.
        for name in BUILTIN_THEME_NAMES {
            if *name == "default" {
                continue;
            }
            let json = builtin_theme_json(name).unwrap_or_else(|| panic!("{name} must be registered"));
            let t = Theme::from_json(json).unwrap_or_else(|| panic!("{name} must parse"));
            assert_ne!(t.code_background, Theme::default_theme().code_background);
            // every key resolves to a non-empty ANSI string (catches typos)
            let all = [
                t.accent_primary, t.accent_tool_name, t.accent_secondary, t.accent_command,
                t.accent_muted, t.accent_input, t.accent_success, t.accent_emphasized_output,
                t.accent_warn, t.accent_danger, t.accent_marked, t.accent_rule, t.accent_submitted,
                t.accent_submitted_bar, t.accent_prompt,
                t.markdown_body, t.markdown_strong, t.markdown_heading, t.markdown_accent,
                t.markdown_code_fg, t.markdown_math, t.code_background, t.code_foreground,
                t.code_comment, t.code_keyword, t.code_string, t.code_number, t.code_type,
                t.code_dim,
            ];
            assert!(
                all.iter().all(|s| !s.is_empty()),
                "{name} leaves an empty color key"
            );
        }
    }

    #[test]
    fn theme_from_name_resolves_builtin_and_falls_back() {
        let monokai = theme_from_name("monokai").unwrap();
        assert_ne!(monokai.code_background, Theme::default_theme().code_background);
        let tokyo = theme_from_name("tokyo-night").unwrap();
        assert_ne!(tokyo.code_background, Theme::default_theme().code_background);
        assert_eq!(theme_from_name("no-such-theme"), None);
    }
}
