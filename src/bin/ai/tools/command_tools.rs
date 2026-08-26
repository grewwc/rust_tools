use serde_json::Value;

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    super::service::audit::validate_execute_command(command)
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    super::service::command::execute_command(args)
}

pub(crate) fn execute_command_streaming<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    super::service::command::execute_command_streaming(args, on_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_command_captures_stdout() {
        let args = serde_json::json!({
            "command": "echo hello"
        });
        let result = execute_command(&args);
        assert!(result.is_ok(), "command failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("hello"),
            "stdout should contain 'hello', got: {}",
            output
        );
    }

    #[test]
    fn test_execute_command_captures_stderr() {
        // Note: `sh -c "..."` is rejected by validate_execute_command (it is a second
        // shell interpretation that could bypass the blacklist). Use a command that
        // writes to stderr on its own (`ls` on a nonexistent path) to verify capture.
        let args = serde_json::json!({
            "command": "ls /nonexistent_dir_for_test_xyz_12345"
        });
        let result = execute_command(&args);
        assert!(result.is_ok(), "command failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("nonexistent_dir_for_test_xyz_12345")
                || output.to_lowercase().contains("no such")
                || output.contains("Exit code:"),
            "stderr should describe the error, got: {}",
            output
        );
    }

    #[test]
    fn test_execute_command_timeout() {
        let args = serde_json::json!({
            "command": "sleep 10",
            "timeout": 1
        });
        let result = execute_command(&args);
        match result {
            Ok(output) => {
                assert!(
                    output.contains("timeout") || output.contains("Exit code:"),
                    "should indicate timeout or failure, got: {}",
                    output
                );
            }
            Err(err) => {
                let normalized = err.to_ascii_lowercase();
                assert!(
                    normalized.contains("timeout") || normalized.contains("timed out"),
                    "error should state that the command timed out, got: {}",
                    err
                );
            }
        }
    }

    #[test]
    fn test_execute_command_streaming_matches_final_output() {
        let args = serde_json::json!({
            "command": "printf 'hello\\nworld'"
        });
        let mut chunks = Vec::new();
        let result = execute_command_streaming(&args, |chunk| chunks.extend_from_slice(chunk));
        assert!(result.is_ok(), "command failed: {:?}", result);
        assert_eq!(String::from_utf8_lossy(&chunks), "hello\nworld");
        assert_eq!(result.unwrap(), "hello\nworld");
    }

    #[test]
    fn test_execute_command_streaming_registry_dispatch_matches_final_output() {
        let args = serde_json::json!({
            "command": "printf 'hello\\nworld'"
        });
        let mut streamed = Vec::new();
        let mut capture = |chunk: &[u8]| streamed.extend_from_slice(chunk);
        let result = crate::ai::tools::common::execute_tool_call_with_args_streaming(
            "call_execute_command_streaming",
            "execute_command",
            &args,
            &mut capture,
        )
        .expect("registry streaming dispatch should succeed");

        assert_eq!(String::from_utf8_lossy(&streamed), "hello\nworld");
        assert_eq!(result.content, "hello\nworld");
    }
}
