pub(super) mod code;
pub(super) mod html;
pub(super) mod inline;
pub(super) mod markdown;
pub(super) mod math;
pub(super) mod table;

// Markdown uses a warm-neutral prose ladder with a restrained slate accent.
// Keep it separate from syntax highlighting and status colors so long answers
// remain calm while headings, emphasis, and inline code stay distinguishable.
pub(super) const MARKDOWN_BODY: &str = "\x1b[38;2;212;209;203m";
pub(super) const MARKDOWN_STRONG: &str = "\x1b[38;2;229;223;213m";
pub(super) const MARKDOWN_HEADING: &str = "\x1b[38;2;221;215;204m";
// Links stay a restrained slate blue; they are underlined, so they need less
// saturation than a plain text chip to stay identifiable.
pub(super) const MARKDOWN_ACCENT: &str = "\x1b[38;2;190;203;219m";
// Inline code chips use no background: background patches at high chip density
// (identifiers, paths, commands) read as noisy dark boxes. A soft, low-saturation
// periwinkle foreground keeps chips distinguishable from warm-neutral prose and
// cool blue links without filling the screen with colored blocks. Blue (links)
// and green (success accents) are already taken, so a gentle lavender gives code
// its own identity while staying calm rather than reading as a loud highlight.
pub(super) const MARKDOWN_CODE_FG: &str = "\x1b[38;2;195;188;220m";
