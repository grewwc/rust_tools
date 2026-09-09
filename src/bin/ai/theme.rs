//! Terminal color theme.
//!
//! The terminal renderer used to hardcode ANSI escape strings as `const`s.
//! Since those colors were duplicated in ~13 modules, restyling the terminal
//! required editing Rust source. This module turns the palette into a data
//! model loaded from a VSCode-style JSON theme file (hot-reloaded at each
//! turn boundary), so styling changes are a config edit, not a recompile.
//!
//! # Theme JSON format (VSCode-like)
//!
//! ```json
//! {
//!   "name": "my-theme",
//!   "type": "dark",
//!   "colors": {
//!     "accent.primary": "#6E82A0",
//!     "code.background": "#282828",
//!     "markdown.math": "\u001b[95m"
//!   }
//! }
//! ```
//!
//! `colors` maps flat dotted keys (like VSCode's `editor.foreground`) to
//! either `#RRGGBB` (optionally `#RRGGBBAA`, alpha ignored) or a raw ANSI
//! escape sequence (a string starting with ESC, e.g. `"\u001b[95m"` or a
//! 256-color code). Unknown keys and unparsable values are ignored; every key
//! falls back to the built-in default, so a custom theme only needs to list
//! what it changes.
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
    /// visible and distinguishable from assistant output. The bold green `❯`
    /// marker in `prompt/multiline/multiline_ui.rs` keeps the submit boundary.
    pub(crate) accent_submitted: &'static str,
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
            accent_primary: "\x1b[38;2;110;130;160m",
            accent_tool_name: "\x1b[38;2;235;140;130m",
            accent_secondary: "\x1b[38;2;196;181;253m",
            accent_command: "\x1b[38;2;165;185;225m",
            accent_muted: "\x1b[38;2;148;163;184m",
            accent_input: "\x1b[38;2;215;212;206m",
            accent_input_rgb: (215, 212, 206),
            accent_success: "\x1b[38;2;134;194;166m",
            accent_emphasized_output: "\x1b[38;2;152;195;121m",
            accent_warn: "\x1b[38;2;245;158;11m",
            accent_danger: "\x1b[38;2;251;113;133m",
            accent_marked: "\x1b[38;2;255;150;150m",
            accent_rule: "\x1b[38;2;71;85;105m",
            accent_submitted: "\x1b[38;2;245;158;11m",
            markdown_body: "\x1b[38;2;212;209;203m",
            markdown_strong: "\x1b[38;2;229;223;213m",
            markdown_heading: "\x1b[38;2;221;215;204m",
            markdown_accent: "\x1b[38;2;190;203;219m",
            markdown_code_fg: "\x1b[38;2;195;188;220m",
            markdown_math: "\x1b[38;2;229;115;158m",
            code_background: "\x1b[48;2;40;40;40m",
            code_foreground: "\x1b[38;2;248;248;242m",
            code_comment: "\x1b[38;2;117;113;94m",
            code_keyword: "\x1b[38;2;249;38;114m",
            code_string: "\x1b[38;2;230;219;116m",
            code_number: "\x1b[38;2;174;129;255m",
            code_type: "\x1b[38;2;102;217;239m",
            code_dim: "\x1b[38;2;107;107;107m",
        }
    }

    /// Parse a VSCode-style theme JSON, merging over the built-in default.
    /// Returns `None` only when the document is not valid JSON.
    pub(crate) fn from_json(json: &str) -> Option<Theme> {
        let root: Value = serde_json::from_str(json).ok()?;
        let colors = match root.get("colors") {
            Some(Value::Object(map)) => map,
            Some(_) => return None,
            None => return Some(Theme::default_theme()),
        };
        let mut t = Theme::default_theme();
        for (key, value) in colors {
            let Some(raw) = value.as_str() else { continue };
            match key.as_str() {
                "accent.primary" => set_fg(&mut t.accent_primary, raw),
                "accent.toolName" => set_fg(&mut t.accent_tool_name, raw),
                "accent.secondary" => set_fg(&mut t.accent_secondary, raw),
                "accent.command" => set_fg(&mut t.accent_command, raw),
                "accent.muted" => set_fg(&mut t.accent_muted, raw),
                "accent.input" => {
                    set_fg(&mut t.accent_input, raw);
                    if let Some(rgb) = hex_rgb(raw) {
                        t.accent_input_rgb = rgb;
                    }
                }
                "accent.success" => set_fg(&mut t.accent_success, raw),
                "accent.emphasizedOutput" => set_fg(&mut t.accent_emphasized_output, raw),
                "accent.warn" => set_fg(&mut t.accent_warn, raw),
                "accent.danger" => set_fg(&mut t.accent_danger, raw),
                "accent.marked" => set_fg(&mut t.accent_marked, raw),
                "accent.rule" => set_fg(&mut t.accent_rule, raw),
                "accent.submitted" => set_fg(&mut t.accent_submitted, raw),
                "markdown.body" => set_fg(&mut t.markdown_body, raw),
                "markdown.strong" => set_fg(&mut t.markdown_strong, raw),
                "markdown.heading" => set_fg(&mut t.markdown_heading, raw),
                "markdown.accent" => set_fg(&mut t.markdown_accent, raw),
                "markdown.codeFg" => set_fg(&mut t.markdown_code_fg, raw),
                "markdown.math" => set_fg(&mut t.markdown_math, raw),
                "code.background" => set_bg(&mut t.code_background, raw),
                "code.foreground" => set_fg(&mut t.code_foreground, raw),
                "code.comment" => set_fg(&mut t.code_comment, raw),
                "code.keyword" => set_fg(&mut t.code_keyword, raw),
                "code.string" => set_fg(&mut t.code_string, raw),
                "code.number" => set_fg(&mut t.code_number, raw),
                "code.type" => set_fg(&mut t.code_type, raw),
                "code.dim" => set_fg(&mut t.code_dim, raw),
                _ => {} // unknown keys are ignored so themes stay forward-compatible
            }
        }
        Some(t)
    }

    /// Resolve the active theme from config (see module docs for the order).
    fn load_from_config() -> Theme {
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

/// Resolve a theme by name: user override dir first, then built-ins.
fn theme_from_name(name: &str) -> Option<Theme> {
    let user_path = user_themes_dir().join(format!("{name}.json"));
    if let Ok(content) = fs::read_to_string(&user_path)
        && let Some(theme) = Theme::from_json(&content)
    {
        return Some(theme);
    }
    let json = builtin_theme_json(name)?;
    Theme::from_json(json)
}

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
        _ => None,
    }
}

/// User theme override directory: `~/.config/rust_tools/themes/`.
fn user_themes_dir() -> PathBuf {
    PathBuf::from(expanduser("~/.config/rust_tools/themes").as_ref())
}

fn set_fg(field: &mut &'static str, raw: &str) {
    if let Some(ansi) = color_to_ansi(raw, false) {
        *field = leak(ansi);
    }
}

fn set_bg(field: &mut &'static str, raw: &str) {
    if let Some(ansi) = color_to_ansi(raw, true) {
        *field = leak(ansi);
    }
}

/// Each resolved theme instance owns its strings and leaks them to get
/// `&'static str` fields; this keeps call-site ergonomics identical to the
/// old consts (`format!` / `push_str` without borrowing). `refresh()` leaks
/// one small instance per re-theming event.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Convert a theme value to an ANSI color sequence. `#RRGGBB` (optionally
/// `#RRGGBBAA`, alpha ignored) becomes an RGB SGR code; a string starting
/// with ESC is passed through verbatim (raw ANSI, e.g. 256-color or style
/// sequences).
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
    fn all_builtin_themes_parse_and_differ_from_default() {
        for name in [
            "monokai",
            "light",
            "dracula",
            "one-dark",
            "tokyo-night",
            "nord",
            "catppuccin",
        ] {
            let json = builtin_theme_json(name).unwrap_or_else(|| panic!("{name} must be registered"));
            let t = Theme::from_json(json).unwrap_or_else(|| panic!("{name} must parse"));
            assert_ne!(t.code_background, Theme::default_theme().code_background);
            // every key resolves to a non-empty ANSI string (catches typos)
            let all = [
                t.accent_primary, t.accent_tool_name, t.accent_secondary, t.accent_command,
                t.accent_muted, t.accent_input, t.accent_success, t.accent_emphasized_output,
                t.accent_warn, t.accent_danger, t.accent_marked, t.accent_rule, t.accent_submitted,
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
