use std::fmt;

/// Error types for the `a` binary.
///
/// This enum provides structured errors that callers can match on
/// programmatically, replacing the pervasive `Result<T, String>` pattern.
#[derive(Debug)]
pub enum AiError {
    /// Tool execution failed
    ToolError { tool: String, message: String },
    /// File operation failed
    FileError { path: String, message: String },
    /// HTTP/MCP network error
    NetworkError { message: String },
    /// Configuration issue
    ConfigError { key: String, message: String },
    /// User input/validation error
    UserError { message: String },
    /// Unexpected internal error
    InternalError { message: String },
}

impl AiError {
    pub fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        AiError::ToolError {
            tool: tool.into(),
            message: message.into(),
        }
    }

    pub fn file(path: impl Into<String>, message: impl Into<String>) -> Self {
        AiError::FileError {
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn network(message: impl Into<String>) -> Self {
        AiError::NetworkError {
            message: message.into(),
        }
    }

    pub fn config(key: impl Into<String>, message: impl Into<String>) -> Self {
        AiError::ConfigError {
            key: key.into(),
            message: message.into(),
        }
    }

    pub fn user(message: impl Into<String>) -> Self {
        AiError::UserError {
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        AiError::InternalError {
            message: message.into(),
        }
    }
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::ToolError { tool, message } => {
                write!(f, "Tool error ({tool}): {message}")
            }
            AiError::FileError { path, message } => {
                write!(f, "File error ({path}): {message}")
            }
            AiError::NetworkError { message } => {
                write!(f, "Network error: {message}")
            }
            AiError::ConfigError { key, message } => {
                write!(f, "Config error ({key}): {message}")
            }
            AiError::UserError { message } => {
                write!(f, "User error: {message}")
            }
            AiError::InternalError { message } => {
                write!(f, "Internal error: {message}")
            }
        }
    }
}

impl std::error::Error for AiError {}

impl From<Box<dyn std::error::Error>> for AiError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        AiError::InternalError {
            message: err.to_string(),
        }
    }
}
