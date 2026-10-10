use std::path::PathBuf;
#[derive(Debug, Clone)]
pub(crate) struct UnifiedHunk {
    pub(super) old_start: usize,
    pub(super) lines: Vec<UnifiedLine>,
}

#[derive(Debug, Clone)]
pub(crate) enum UnifiedLine {
    Context(String),
    Remove(String),
    Add(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PatchEnvelopeOp {
    Update,
    Add,
    Delete,
    /// Inline substring replacement: use `anchor:` to locate the line, then exactly replace
    /// `old:` with `new:` within that line.
    /// Does not go through the unified-diff path; handled directly by `apply_inline_replace`.
    ReplaceInLine,
}

#[derive(Debug, Clone)]
pub(crate) struct PatchEnvelope {
    pub(super) op: PatchEnvelopeOp,
    pub(super) target_path: String,
    pub(super) body_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedPatchWrite {
    pub(super) path: PathBuf,
    pub(super) before: Option<String>,
    pub(super) action: PreparedPatchAction,
    /// Hints produced during application (e.g. pure-insert hunks are located by line number
    /// only), returned together with the success message to remind the model to re-read the file
    /// with read_file once it has changed.
    pub(super) hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum PreparedPatchAction {
    Write(String),
    Delete,
}
