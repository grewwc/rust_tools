use super::state::{SseEvent, StreamFramingState};

pub(super) fn push_chunk(state: &mut StreamFramingState, chunk: &[u8]) {
    state.pending.extend_from_slice(chunk);
}

pub(super) fn take_complete_lines(
    state: &mut StreamFramingState,
) -> Result<Vec<String>, std::str::Utf8Error> {
    let mut pending = std::mem::take(&mut state.pending);
    let mut lines = Vec::new();
    let mut consumed = 0usize;

    while let Some(line_end_rel) = pending[consumed..].iter().position(|b| *b == b'\n') {
        let line_end = consumed + line_end_rel + 1;
        let line = std::str::from_utf8(&pending[consumed..line_end])?;
        lines.push(line.to_string());
        consumed = line_end;
    }

    if consumed != 0 {
        pending.drain(..consumed);
    }
    state.pending = pending;
    Ok(lines)
}

pub(super) fn take_pending_tail(
    state: &mut StreamFramingState,
) -> Result<Option<String>, std::str::Utf8Error> {
    if state.pending.is_empty() {
        return Ok(None);
    }
    let pending = std::mem::take(&mut state.pending);
    let line = std::str::from_utf8(&pending)?.to_string();
    state.pending = pending;
    Ok(Some(line))
}

pub(super) fn consume_sse_line(state: &mut StreamFramingState, line: &str) -> Option<SseEvent> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        return flush_sse_event(state);
    }
    if trimmed.starts_with(':') {
        return None;
    }
    if let Some(event_type) = trimmed.strip_prefix("event:") {
        state.sse_event_type = Some(event_type.trim().to_string());
        return None;
    }
    let _ = append_sse_payload_line(&mut state.sse_event_data, trimmed);
    None
}

pub(super) fn flush_sse_event(state: &mut StreamFramingState) -> Option<SseEvent> {
    if state.sse_event_data.trim().is_empty() {
        state.sse_event_type = None;
        state.sse_event_data.clear();
        return None;
    }
    Some(SseEvent {
        event_type: state.sse_event_type.take(),
        payload: std::mem::take(&mut state.sse_event_data),
    })
}

fn append_sse_payload_line(sse_event_data: &mut String, line: &str) -> bool {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if let Some(data) = trimmed.strip_prefix("data:") {
        let payload = data.strip_prefix(' ').unwrap_or(data);
        if !sse_event_data.is_empty() {
            sse_event_data.push('\n');
        }
        sse_event_data.push_str(payload);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{consume_sse_line, flush_sse_event};
    use crate::ai::stream::state::{SseEvent, StreamFramingState};

    #[test]
    fn sse_data_lines_are_aggregated_until_event_boundary() {
        let mut state = StreamFramingState {
            decode_error_count: 0,
            pending: Vec::new(),
            sse_event_type: None,
            sse_event_data: String::new(),
        };

        assert_eq!(consume_sse_line(&mut state, "data: {\"choices\":"), None);
        assert_eq!(
            consume_sse_line(&mut state, "data: [{\"delta\":{\"content\":\"hi\"}}]}"),
            None
        );
        assert_eq!(
            flush_sse_event(&mut state),
            Some(SseEvent {
                event_type: None,
                payload: "{\"choices\":\n[{\"delta\":{\"content\":\"hi\"}}]}".to_string(),
            })
        );
    }

    #[test]
    fn sse_event_type_is_preserved_until_boundary() {
        let mut state = StreamFramingState {
            decode_error_count: 0,
            pending: Vec::new(),
            sse_event_type: None,
            sse_event_data: String::new(),
        };

        assert_eq!(
            consume_sse_line(&mut state, "event: response.reasoning_text.delta"),
            None
        );
        assert_eq!(consume_sse_line(&mut state, "data: {\"delta\":\"step\"}"), None);
        assert_eq!(
            flush_sse_event(&mut state),
            Some(SseEvent {
                event_type: Some("response.reasoning_text.delta".to_string()),
                payload: "{\"delta\":\"step\"}".to_string(),
            })
        );
    }
}
