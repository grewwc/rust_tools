//! Event sequencing without CoreGraphics so failure paths can be tested off macOS.

const TEXT_CHUNK_UNITS: usize = 20;

/// Create both events before posting either. CoreGraphics posting returns no
/// status, but event creation can fail, including for the release event.
pub(super) fn post_pair<E>(
    mut create: impl FnMut(bool) -> Result<E, String>,
    mut post: impl FnMut(E),
) -> Result<(), String> {
    let down = create(true)?;
    let up = create(false)?;
    post(down);
    post(up);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DragPhase {
    Down,
    Dragged,
    Up,
}

pub(super) fn post_drag(
    start: (f64, f64),
    end: (f64, f64),
    steps: u32,
    mut post: impl FnMut(DragPhase, (f64, f64)) -> Result<(), String>,
) -> Result<(), String> {
    post(DragPhase::Down, start)?;
    let mut last_posted = start;
    let steps = steps.clamp(2, 200);
    for step in 1..=steps {
        let t = step as f64 / steps as f64;
        let point = if step == steps {
            end
        } else {
            (
                start.0 + (end.0 - start.0) * t,
                start.1 + (end.1 - start.1) * t,
            )
        };
        if let Err(error) = post(DragPhase::Dragged, point) {
            // Release where the last event was posted, not at an unreached destination.
            let cleanup = match post(DragPhase::Up, last_posted) {
                Ok(()) => "mouse-up posted at the last posted position".to_string(),
                Err(error) => {
                    format!("mouse-up cleanup failed: {error}; the button may still be held")
                }
            };
            return Err(format!(
                "drag interrupted after mouse-down: {error}; {cleanup}"
            ));
        }
        last_posted = point;
    }
    post(DragPhase::Up, end).map_err(|error| {
        format!("drag path posted, but mouse-up failed: {error}; the button may still be held")
    })
}

/// `units` comes from `str::encode_utf16`, so a leading surrogate always has a pair.
pub(super) fn text_chunks(mut units: &[u16]) -> impl Iterator<Item = &[u16]> {
    std::iter::from_fn(move || {
        if units.is_empty() {
            return None;
        }
        let mut len = units.len().min(TEXT_CHUNK_UNITS);
        if len < units.len() && (0xD800..=0xDBFF).contains(&units[len - 1]) {
            len -= 1;
        }
        let (chunk, rest) = units.split_at(len);
        units = rest;
        Some(chunk)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_posts_neither_event_if_release_creation_fails() {
        let mut posted = Vec::new();
        let result = post_pair(
            |down| {
                down.then_some(down)
                    .ok_or_else(|| "no release event".into())
            },
            |event| posted.push(event),
        );
        assert_eq!(result.unwrap_err(), "no release event");
        assert!(posted.is_empty());

        post_pair(Ok, |event| posted.push(event)).unwrap();
        assert_eq!(posted, [true, false]);
    }

    #[test]
    fn interrupted_drag_releases_at_last_posted_point() {
        let mut attempts = Vec::new();
        let result = post_drag((0.0, 0.0), (3.0, 3.0), 3, |phase, point| {
            attempts.push((phase, point));
            if phase == DragPhase::Dragged && point == (2.0, 2.0) {
                Err("no drag event".into())
            } else {
                Ok(())
            }
        });
        assert_eq!(
            attempts,
            [
                (DragPhase::Down, (0.0, 0.0)),
                (DragPhase::Dragged, (1.0, 1.0)),
                (DragPhase::Dragged, (2.0, 2.0)),
                (DragPhase::Up, (1.0, 1.0)),
            ]
        );
        let error = result.unwrap_err();
        assert!(error.contains("no drag event"));
        assert!(error.contains("mouse-up posted"));
    }

    #[test]
    fn drag_keeps_the_first_error_if_cleanup_also_fails() {
        let mut attempts = Vec::new();
        let result = post_drag((0.0, 0.0), (2.0, 2.0), 2, |phase, _| {
            attempts.push(phase);
            match phase {
                DragPhase::Down => Ok(()),
                DragPhase::Dragged => Err("no drag event".into()),
                DragPhase::Up => Err("no release event".into()),
            }
        });
        assert_eq!(
            attempts,
            [DragPhase::Down, DragPhase::Dragged, DragPhase::Up]
        );
        let error = result.unwrap_err();
        assert!(error.contains("no drag event"));
        assert!(error.contains("no release event"));
    }

    #[test]
    fn drag_does_not_release_if_no_press_was_posted() {
        let mut attempts = Vec::new();
        let result = post_drag((0.0, 0.0), (2.0, 2.0), 2, |phase, _| {
            attempts.push(phase);
            Err("no down event".into())
        });
        assert_eq!(attempts, [DragPhase::Down]);
        assert_eq!(result.unwrap_err(), "no down event");
    }

    #[test]
    fn text_chunk_boundary_keeps_non_bmp_characters_intact() {
        let text = format!("{}😀字𐐀", "a".repeat(TEXT_CHUNK_UNITS - 1));
        let units: Vec<_> = text.encode_utf16().collect();
        let chunks: Vec<_> = text_chunks(&units).collect();
        assert_eq!(chunks[0].len(), TEXT_CHUNK_UNITS - 1);
        assert!(chunks.iter().all(|chunk| chunk.len() <= TEXT_CHUNK_UNITS));
        let decoded: String = chunks
            .into_iter()
            .map(|chunk| String::from_utf16(chunk).unwrap())
            .collect();
        assert_eq!(decoded, text);
    }
}
