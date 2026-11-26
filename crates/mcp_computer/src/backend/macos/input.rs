//! Mouse, keyboard and scroll injection through CoreGraphics (`CGEvent*`).
//!
//! All coordinates are global screen points with the origin at the top-left of
//! the main display — the same space `screencapture -R` and `CGWindowList*`
//! report, so a coordinate read from a screenshot maps straight into `click`.

use std::time::Duration;

use objc2_core_foundation::{CFRetained, CGPoint};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
    CGEventType, CGMouseButton, CGScrollEventUnit,
};

use super::input_sequence::{DragPhase, post_drag, post_pair, text_chunks};

/// Wait after moving the cursor before the following event. Applications that
/// read the pointer position asynchronously (Electron, Java, remote desktops)
/// deliver a click to the wrong target if it arrives in the same run-loop pass
/// as the move.
const MOVE_SETTLE_MS: u64 = 30;
/// Gap between the press/release pairs of a multi-click, short enough that the
/// system still groups them into one double/triple click.
const MULTI_CLICK_GAP_MS: u64 = 60;
/// Pause between the steps of a drag.
const DRAG_STEP_MS: u64 = 12;
const KEY_SETTLE_MS: u64 = 8;
const TEXT_CHUNK_SETTLE_MS: u64 = 8;

fn source() -> Option<CFRetained<CGEventSource>> {
    CGEventSource::new(CGEventSourceStateID::HIDSystemState)
}

fn parse_button(name: &str) -> Result<CGMouseButton, String> {
    match name.to_ascii_lowercase().as_str() {
        "left" => Ok(CGMouseButton::Left),
        "right" => Ok(CGMouseButton::Right),
        other => Err(format!(
            "unknown mouse button `{other}`; this backend supports `left` and `right`"
        )),
    }
}

fn parse_modifiers(modifiers: &[String]) -> Result<CGEventFlags, String> {
    let mut flags = CGEventFlags::empty();
    for name in modifiers {
        let flag = match name.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "super" | "meta" => CGEventFlags::MaskCommand,
            "shift" => CGEventFlags::MaskShift,
            "alt" | "option" | "opt" => CGEventFlags::MaskAlternate,
            "ctrl" | "control" => CGEventFlags::MaskControl,
            "fn" => CGEventFlags::MaskSecondaryFn,
            other => {
                return Err(format!(
                    "unknown modifier `{other}`; expected cmd / shift / alt / ctrl / fn"
                ));
            }
        };
        flags |= flag;
    }
    Ok(flags)
}

fn post_event(event: CFRetained<CGEvent>) {
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&*event));
}

fn mouse_event(
    kind: CGEventType,
    point: CGPoint,
    button: CGMouseButton,
    flags: CGEventFlags,
) -> Result<CFRetained<CGEvent>, String> {
    let event = CGEvent::new_mouse_event(source().as_deref(), kind, point, button)
        .ok_or_else(|| "CoreGraphics refused to create a pointer event".to_string())?;
    CGEvent::set_flags(Some(&*event), flags);
    Ok(event)
}

/// Post a pointer event that carries no click state (moves and drags).
fn post_mouse(
    kind: CGEventType,
    point: CGPoint,
    button: CGMouseButton,
    flags: CGEventFlags,
) -> Result<(), String> {
    post_event(mouse_event(kind, point, button, flags)?);
    Ok(())
}

fn keyboard_event(keycode: u16, down: bool) -> Result<CFRetained<CGEvent>, String> {
    CGEvent::new_keyboard_event(source().as_deref(), keycode, down)
        .ok_or_else(|| "CoreGraphics refused to create a keyboard event".to_string())
}

pub(crate) fn move_mouse(x: f64, y: f64) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    post_mouse(
        CGEventType::MouseMoved,
        CGPoint::new(x, y),
        CGMouseButton::Left,
        CGEventFlags::empty(),
    )
}

pub(crate) fn click(
    x: f64,
    y: f64,
    button: &str,
    count: u32,
    modifiers: &[String],
) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    let button = parse_button(button)?;
    let flags = parse_modifiers(modifiers)?;
    let point = CGPoint::new(x, y);

    post_mouse(CGEventType::MouseMoved, point, CGMouseButton::Left, flags)?;
    std::thread::sleep(Duration::from_millis(MOVE_SETTLE_MS));

    let (down_type, up_type) = if button == CGMouseButton::Right {
        (CGEventType::RightMouseDown, CGEventType::RightMouseUp)
    } else {
        (CGEventType::LeftMouseDown, CGEventType::LeftMouseUp)
    };

    let count = count.max(1);
    for index in 1..=count {
        post_pair(
            |down| {
                let kind = if down { down_type } else { up_type };
                let event = mouse_event(kind, point, button, flags)?;
                // The click state groups successive pairs into one multi-click.
                CGEvent::set_integer_value_field(
                    Some(&*event),
                    CGEventField::MouseEventClickState,
                    index as i64,
                );
                Ok(event)
            },
            post_event,
        )?;
        if index < count {
            std::thread::sleep(Duration::from_millis(MULTI_CLICK_GAP_MS));
        }
    }
    Ok(())
}

pub(crate) fn drag(
    from_x: f64,
    from_y: f64,
    to_x: f64,
    to_y: f64,
    button: &str,
    modifiers: &[String],
    steps: u32,
) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    let button = parse_button(button)?;
    let flags = parse_modifiers(modifiers)?;
    let (down_type, drag_type, up_type) = if button == CGMouseButton::Right {
        (
            CGEventType::RightMouseDown,
            CGEventType::RightMouseDragged,
            CGEventType::RightMouseUp,
        )
    } else {
        (
            CGEventType::LeftMouseDown,
            CGEventType::LeftMouseDragged,
            CGEventType::LeftMouseUp,
        )
    };

    let start = CGPoint::new(from_x, from_y);
    post_mouse(CGEventType::MouseMoved, start, CGMouseButton::Left, flags)?;
    std::thread::sleep(Duration::from_millis(MOVE_SETTLE_MS));

    // A drag must be delivered as a path: selection-tracking targets (text
    // editors, canvas tools, sliders) ignore a single jump from start to end.
    post_drag((from_x, from_y), (to_x, to_y), steps, |phase, (x, y)| {
        let kind = match phase {
            DragPhase::Down => down_type,
            DragPhase::Dragged => drag_type,
            DragPhase::Up => up_type,
        };
        post_mouse(kind, CGPoint::new(x, y), button, flags)?;
        if phase == DragPhase::Dragged {
            std::thread::sleep(Duration::from_millis(DRAG_STEP_MS));
        }
        Ok(())
    })
}

/// Scroll at the current pointer position, or at `(x, y)` when both are given.
/// Scroll events are delivered to whatever is under the pointer, so moving first
/// is what makes the target explicit.
pub(crate) fn scroll(
    x: Option<f64>,
    y: Option<f64>,
    dx: f64,
    dy: f64,
    unit: &str,
) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    let units = match unit.to_ascii_lowercase().as_str() {
        "pixel" | "pixels" => CGScrollEventUnit::Pixel,
        "line" | "lines" => CGScrollEventUnit::Line,
        other => {
            return Err(format!(
                "unknown scroll unit `{other}`; expected `pixel` or `line`"
            ));
        }
    };
    if let (Some(x), Some(y)) = (x, y) {
        move_mouse(x, y)?;
        std::thread::sleep(Duration::from_millis(MOVE_SETTLE_MS));
    }
    // Two axes: wheel1 is vertical, wheel2 horizontal. Positive vertical means
    // "scroll up" (the content moves down) in the system's own convention.
    let event = CGEvent::new_scroll_wheel_event2(
        source().as_deref(),
        units,
        2,
        dy.round() as i32,
        dx.round() as i32,
        0,
    )
    .ok_or_else(|| "CoreGraphics refused to create a scroll event".to_string())?;
    post_event(event);
    Ok(())
}

/// Type `text` into whatever holds keyboard focus. The text goes through
/// `CGEventKeyboardSetUnicodeString`, so it is character-agnostic: no keycode
/// table and no keyboard layout is involved, and non-ASCII text works.
pub(crate) fn type_text(text: &str) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    let units: Vec<u16> = text.encode_utf16().collect();
    for chunk in text_chunks(&units) {
        post_pair(
            |key_down| {
                let event = keyboard_event(0, key_down)?;
                // SAFETY: `chunk` is a live slice for the whole call and its length
                // is exactly the number of UTF-16 units it holds.
                unsafe {
                    CGEvent::keyboard_set_unicode_string(
                        Some(&*event),
                        // The binding uses a crate-private alias of `c_ulong`.
                        chunk.len() as core::ffi::c_ulong,
                        chunk.as_ptr(),
                    );
                }
                Ok(event)
            },
            post_event,
        )?;
        std::thread::sleep(Duration::from_millis(TEXT_CHUNK_SETTLE_MS));
    }
    Ok(())
}

/// Press a named key or chord, e.g. `escape`, `return`, `f5`, `cmd+shift+t`.
pub(crate) fn press_key(chord: &str) -> Result<(), String> {
    super::screen::ensure_accessibility_permission()?;
    let parts: Vec<&str> = chord
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    let Some((base, modifier_names)) = parts.split_last() else {
        return Err("press_key needs a key, e.g. `escape` or `cmd+shift+t`".to_string());
    };
    let modifiers: Vec<String> = modifier_names
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let flags = parse_modifiers(&modifiers)?;
    let keycode = keycode_for(base).ok_or_else(|| {
        format!(
            "unknown key `{base}`; use a letter, a digit, or one of \
             return/enter/tab/space/escape/backspace/forwarddelete/up/down/left/right/\
             home/end/pageup/pagedown/f1..f12"
        )
    })?;

    post_pair(
        |key_down| {
            let event = keyboard_event(keycode, key_down)?;
            // The flags on both base-key events describe the shortcut.
            CGEvent::set_flags(Some(&*event), flags);
            Ok(event)
        },
        |event| {
            post_event(event);
            std::thread::sleep(Duration::from_millis(KEY_SETTLE_MS));
        },
    )
}

/// ANSI virtual keycodes (`HIToolbox/Events.h`, `kVK_*`). These are fixed
/// hardware-position codes: they identify the physical key, while the character
/// actually produced depends on the user's keyboard layout — which is why
/// `type_text` does not go through this table.
fn keycode_for(name: &str) -> Option<u16> {
    let lower = name.to_ascii_lowercase();
    let code = match lower.as_str() {
        "a" => 0x00,
        "s" => 0x01,
        "d" => 0x02,
        "f" => 0x03,
        "h" => 0x04,
        "g" => 0x05,
        "z" => 0x06,
        "x" => 0x07,
        "c" => 0x08,
        "v" => 0x09,
        "b" => 0x0B,
        "q" => 0x0C,
        "w" => 0x0D,
        "e" => 0x0E,
        "r" => 0x0F,
        "y" => 0x10,
        "t" => 0x11,
        "1" => 0x12,
        "2" => 0x13,
        "3" => 0x14,
        "4" => 0x15,
        "6" => 0x16,
        "5" => 0x17,
        "=" => 0x18,
        "equal" => 0x18,
        "9" => 0x19,
        "7" => 0x1A,
        "-" => 0x1B,
        "minus" => 0x1B,
        "8" => 0x1C,
        "0" => 0x1D,
        "]" => 0x1E,
        "rightbracket" => 0x1E,
        "o" => 0x1F,
        "u" => 0x20,
        "[" => 0x21,
        "leftbracket" => 0x21,
        "i" => 0x22,
        "p" => 0x23,
        "return" | "enter" => 0x24,
        "l" => 0x25,
        "j" => 0x26,
        "'" => 0x27,
        "quote" => 0x27,
        "k" => 0x28,
        ";" | "semicolon" => 0x29,
        "\\" | "backslash" => 0x2A,
        "," | "comma" => 0x2B,
        "/" | "slash" => 0x2C,
        "n" => 0x2D,
        "m" => 0x2E,
        "." | "period" => 0x2F,
        "tab" => 0x30,
        "space" => 0x31,
        "`" | "grave" => 0x32,
        "backspace" | "delete" => 0x33,
        "escape" | "esc" => 0x35,
        "f1" => 0x7A,
        "f2" => 0x78,
        "f3" => 0x63,
        "f4" => 0x76,
        "f5" => 0x60,
        "f6" => 0x61,
        "f7" => 0x62,
        "f8" => 0x64,
        "f9" => 0x65,
        "f10" => 0x6D,
        "f11" => 0x67,
        "f12" => 0x6F,
        "home" => 0x73,
        "end" => 0x77,
        "pageup" => 0x74,
        "pagedown" => 0x79,
        "forwarddelete" | "del" => 0x75,
        "left" => 0x7B,
        "right" => 0x7C,
        "down" => 0x7D,
        "up" => 0x7E,
        _ => return None,
    };
    Some(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_aliases_and_order_preserve_the_coregraphics_flags() {
        for (name, expected) in [
            ("CMD", CGEventFlags::MaskCommand),
            ("meta", CGEventFlags::MaskCommand),
            ("ctrl", CGEventFlags::MaskControl),
            ("alt", CGEventFlags::MaskAlternate),
            ("option", CGEventFlags::MaskAlternate),
            ("shift", CGEventFlags::MaskShift),
            ("fn", CGEventFlags::MaskSecondaryFn),
        ] {
            assert_eq!(parse_modifiers(&[name.to_string()]).unwrap(), expected);
        }
        let expected = CGEventFlags::MaskCommand | CGEventFlags::MaskShift;
        for order in [["cmd", "shift"], ["shift", "cmd"]] {
            assert_eq!(
                parse_modifiers(&order.map(str::to_string)).unwrap(),
                expected
            );
        }
        assert!(parse_modifiers(&["tab".to_string()]).is_err());
    }
}
