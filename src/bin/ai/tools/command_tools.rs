use serde_json::Value;

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    super::service::command::validate_execute_command(command)
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
        let args = serde_json::json!({
            "command": "sh -c 'echo error_msg >&2'"
        });
        let result = execute_command(&args);
        assert!(result.is_ok(), "command failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("error_msg"),
            "stderr should contain 'error_msg', got: {}",
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
                assert!(
                    err.contains("timeout"),
                    "error should mention timeout, got: {}",
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
}
