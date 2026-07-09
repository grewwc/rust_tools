pub mod agent;
pub mod checkpoint;
pub mod feishu;
pub mod goal;
pub mod help;
pub mod model;
pub mod persona;
pub mod proc;
pub mod session;
pub mod share;
pub mod skills;
pub mod usage;

use std::sync::Arc;

use crate::ai::{agents::AgentManifest, mcp::SharedMcpClient, skills::SkillManifest, types::App};

pub use agent::try_handle_agent_command;
pub use checkpoint::try_handle_checkpoint_command;
pub use feishu::try_handle_feishu_auth_command;
pub use goal::try_handle_goal_command;
pub use help::try_handle_help_command;
pub use model::try_handle_model_command;
pub use persona::try_handle_persona_command;
pub use proc::try_handle_proc_command;
pub use session::try_handle_session_command;
pub use share::try_handle_share_command;
pub use skills::try_handle_skills_command;
pub use usage::try_handle_usage_command;

pub fn try_handle_interactive_command(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    input: &str,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_help_command(input) {
        return Ok(true);
    }
    if try_handle_goal_command(app, input)? {
        return Ok(true);
    }
    if try_handle_model_command(app, input)? {
        return Ok(true);
    }
    if try_handle_persona_command(app, input)? {
        return Ok(true);
    }
    if try_handle_usage_command(input)? {
        return Ok(true);
    }
    if try_handle_session_command(app, input)? {
        return Ok(true);
    }
    if try_handle_proc_command(app, input)? {
        return Ok(true);
    }
    if try_handle_checkpoint_command(app, input)? {
        return Ok(true);
    }
    if try_handle_agent_command(app, input, agent_manifests)? {
        return Ok(true);
    }
    if try_handle_skills_command(app, input, skill_manifests)? {
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
