use serde_json::Value;

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    super::service::command::validate_execute_command(command)
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    super::service::command::execute_command(args)
}
