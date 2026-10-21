pub(super) mod code;
pub(super) mod html;
pub(super) mod inline;
pub(super) mod markdown;
pub(super) mod math;
pub(super) mod table;

/// Explicit display width reported by the client that renders a serve turn's
/// output (see `background::SERVE_TERM_COLS_ENV`). A serve turn child has no
/// terminal of its own — its stdout is a pipe — so the width the client
/// actually draws at travels in the environment. Both the markdown renderer
/// and the table renderer consult it first, so one turn's output is laid out
/// for one width.
pub(super) fn term_cols_override() -> Option<usize> {
    // Only a serve turn child honors the override: it renders the rows its
    // client displays. A daemon, a serve client, or a nested process that
    // merely inherited the variable keeps its own width probe, so a stale
    // value can never bend the layout of a process that owns a real terminal.
    if !crate::ai::background::is_serve_child() {
        return None;
    }
    std::env::var(crate::ai::background::SERVE_TERM_COLS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|cols| *cols > 0)
}
