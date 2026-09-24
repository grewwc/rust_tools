//! Platform backend for the computer-use tools.
//!
//! There is no cross-platform API for OS-level input injection or screen capture,
//! so "generic computer-use" always means "one thin backend per OS behind one
//! stable tool surface". This module owns that seam:
//!
//! - `backend::macos` — CoreGraphics (`CGEvent*`, `CGWindow*`, `screencapture`).
//! - `backend::linux` — the distro CLIs for the session type in use (`grim` /
//!   `import` / `scrot` for capture, `xdotool` / `ydotool` / `wtype` for input).
//! - `backend::unsupported` — everything else; every call reports a clear error.
//!
//! **Adding a platform must not change `tools.rs`.** Each backend exposes the same
//! function set with the same signatures (listed in crates/mcp_computer/AGENTS.md)
//! and returns a model-facing text report or a model-facing error string. Anything
//! platform-specific (which helper binary exists, which permission is missing)
//! is discovered at runtime and reported through those strings, so the agent
//! sees the real reason instead of a silent no-op.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::*;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::*;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod unsupported;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) use unsupported::*;