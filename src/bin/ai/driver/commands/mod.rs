pub mod agent;
pub mod feishu;
pub mod help;
pub mod model;
pub mod session;
pub mod share;

use crate::ai::{
    agents::AgentManifest,
    mcp::McpClient,
    types::App,
};

pub use agent::try_handle_agent_command;
pub use feishu::try_handle_feishu_auth_command;
pub use help::try_handle_help_command;
pub use model::try_handle_model_command;
pub use session::try_handle_session_command;
pub use share::try_handle_share_command;

pub fn try_handle_interactive_command(
    app: &mut App,
    mcp_client: &mut McpClient,
    input: &str,
    agent_manifests: &mut Vec<AgentManifest>,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_help_command(input) {
        return Ok(true);
    }
    if try_handle_model_command(app, input)? {
        return Ok(true);
    }
    if try_handle_session_command(app, input)? {
        return Ok(true);
    }
    if try_handle_agent_command(app, input, agent_manifests)? {
        return Ok(true);
    }
    if try_handle_feishu_auth_command(mcp_client, input)? {
        return Ok(true);
    }
    if try_handle_share_command(app, input)? {
        return Ok(true);
    }
    Ok(false)
}
