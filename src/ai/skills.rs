use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SkillManifest {
    pub(super) name: String,
    pub(super) version: String,
    #[serde(default)]
    pub(super) description: String,
    #[serde(default)]
    pub(super) author: Option<String>,
    #[serde(default)]
    pub(super) tools: Vec<String>,
    #[serde(default)]
    pub(super) mcp_servers: Vec<String>,
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) system_prompt: Option<String>,
    #[serde(default)]
    pub(super) examples: Vec<SkillExample>,
    #[serde(default)]
    pub(super) triggers: Vec<String>,
    #[serde(default)]
    pub(super) priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SkillExample {
    pub(super) user: String,
    pub(super) assistant: String,
}

impl SkillManifest {
    pub(super) fn build_system_prompt(&self) -> String {
        let mut prompt = if let Some(sys) = &self.system_prompt {
            sys.clone()
        } else {
            String::new()
        };

        if !self.prompt.is_empty() {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(&self.prompt);
        }

        prompt
    }
}

pub(super) fn create_default_skills() -> Vec<SkillManifest> {
    vec![
        SkillManifest {
            name: "code-review".to_string(),
            version: "1.0.0".to_string(),
            description: "Review code for quality, security, and best practices".to_string(),
            author: Some("system".to_string()),
            tools: vec!["read_file".to_string(), "grep_search".to_string()],
            mcp_servers: vec![],
            prompt: "You are a code reviewer. Analyze the code for:\n- Bugs and potential issues\n- Security vulnerabilities\n- Code style and best practices\n- Performance concerns\n- Maintainability".to_string(),
            system_prompt: Some("You are an expert code reviewer with deep knowledge of software engineering best practices.".to_string()),
            examples: vec![],
            triggers: vec!["review this code".to_string(), "code review".to_string()],
            priority: 10,
        },
        SkillManifest {
            name: "debugger".to_string(),
            version: "1.0.0".to_string(),
            description: "Help debug code and find issues".to_string(),
            author: Some("system".to_string()),
            tools: vec!["read_file".to_string(), "execute_command".to_string(), "grep_search".to_string()],
            mcp_servers: vec![],
            prompt: "You are a debugging assistant. Help identify and fix bugs in code. Use available tools to:\n- Read relevant source files\n- Search for related code\n- Run tests or commands to reproduce issues".to_string(),
            system_prompt: Some("You are an expert debugger with deep knowledge of common programming errors and debugging techniques.".to_string()),
            examples: vec![],
            triggers: vec!["debug".to_string(), "fix this bug".to_string(), "not working".to_string()],
            priority: 20,
        },
        SkillManifest {
            name: "refactor".to_string(),
            version: "1.0.0".to_string(),
            description: "Refactor code to improve quality".to_string(),
            author: Some("system".to_string()),
            tools: vec!["read_file".to_string(), "write_file".to_string()],
            mcp_servers: vec![],
            prompt: "You are a refactoring assistant. Improve code quality by:\n- Reducing duplication\n- Improving naming\n- Simplifying complex logic\n- Applying design patterns\n- Improving testability".to_string(),
            system_prompt: Some("You are an expert at refactoring code while preserving functionality.".to_string()),
            examples: vec![],
            triggers: vec!["refactor".to_string(), "clean up".to_string(), "improve this code".to_string()],
            priority: 5,
        },
        SkillManifest {
            name: "openclaw".to_string(),
            version: "1.0.0".to_string(),
            description: "OpenClaw-like autonomous tool-using agent".to_string(),
            author: Some("system".to_string()),
            tools: vec![
                "read_file".to_string(),
                "read_file_lines".to_string(),
                "write_file".to_string(),
                "apply_patch".to_string(),
                "search_files".to_string(),
                "list_directory".to_string(),
                "grep_search".to_string(),
                "execute_command".to_string(),
                "web_search".to_string(),
                "web_fetch".to_string(),
                "git_status".to_string(),
                "git_diff".to_string(),
                "cargo_check".to_string(),
                "cargo_test".to_string(),
            ],
            mcp_servers: vec![],
            prompt: "Work like an OpenClaw-style agent:\n- First: restate the goal and list concrete steps.\n- Then: use tools iteratively to gather evidence (read/search) before editing.\n- Prefer minimal, targeted edits (patches) and keep changes reversible.\n- After edits: run relevant checks/tests.\n- Finish: summarize changes and how to verify.\n- Preserve existing behavior unless user explicitly requests changes.".to_string(),
            system_prompt: Some("You are an autonomous coding agent optimized for safe, incremental changes and verification.".to_string()),
            examples: vec![],
            triggers: vec![
                "openclaw".to_string(),
                "openclaw模式".to_string(),
                "开启openclaw".to_string(),
            ],
            priority: 30,
        },
    ]
}
