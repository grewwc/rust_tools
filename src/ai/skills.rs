use std::{collections::HashMap, fs, path::PathBuf};

use serde::{Deserialize, Serialize};

use super::types::{FunctionDefinition, SkillDefinition, ToolDefinition};

const DEFAULT_SKILLS_DIR: &str = "~/.config/skills";

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SkillExample {
    pub(super) user: String,
    pub(super) assistant: String,
}

pub(super) struct SkillRegistry {
    skills: HashMap<String, SkillManifest>,
    skills_dir: PathBuf,
}

impl SkillRegistry {
    pub(super) fn new() -> Self {
        let skills_dir = crate::common::utils::expanduser(DEFAULT_SKILLS_DIR);
        let mut registry = Self {
            skills: HashMap::new(),
            skills_dir: PathBuf::from(skills_dir.as_ref()),
        };
        let _ = registry.load_all_skills();
        registry
    }

    pub(super) fn with_dir(dir: PathBuf) -> Self {
        let mut registry = Self {
            skills: HashMap::new(),
            skills_dir: dir,
        };
        let _ = registry.load_all_skills();
        registry
    }

    fn load_all_skills(&mut self) -> Result<(), String> {
        if !self.skills_dir.exists() {
            fs::create_dir_all(&self.skills_dir)
                .map_err(|e| format!("Failed to create skills dir: {}", e))?;
            return Ok(());
        }

        let entries = fs::read_dir(&self.skills_dir)
            .map_err(|e| format!("Failed to read skills dir: {}", e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(skill) = self.load_skill(&path) {
                    self.skills.insert(skill.name.clone(), skill);
                }
            }
        }

        Ok(())
    }

    fn load_skill(&self, path: &PathBuf) -> Result<SkillManifest, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read skill file: {}", e))?;

        let skill: SkillManifest =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse skill: {}", e))?;

        Ok(skill)
    }

    pub(super) fn get_skill(&self, name: &str) -> Option<&SkillManifest> {
        self.skills.get(name)
    }

    pub(super) fn list_skills(&self) -> Vec<&SkillManifest> {
        self.skills.values().collect()
    }

    pub(super) fn register_skill(&mut self, skill: SkillManifest) -> Result<(), String> {
        let path = self.skills_dir.join(format!("{}.json", skill.name));
        let content = serde_json::to_string_pretty(&skill)
            .map_err(|e| format!("Failed to serialize skill: {}", e))?;

        fs::write(&path, content).map_err(|e| format!("Failed to write skill file: {}", e))?;

        self.skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    pub(super) fn unregister_skill(&mut self, name: &str) -> Result<(), String> {
        let path = self.skills_dir.join(format!("{}.json", name));

        if path.exists() {
            fs::remove_file(&path).map_err(|e| format!("Failed to remove skill file: {}", e))?;
        }

        self.skills.remove(name);
        Ok(())
    }

    pub(super) fn match_skill(&self, input: &str) -> Option<&SkillManifest> {
        let input_lower = input.to_lowercase();

        for skill in self.skills.values() {
            for trigger in &skill.triggers {
                if input_lower.contains(&trigger.to_lowercase()) {
                    return Some(skill);
                }
            }
        }

        None
    }
}

impl SkillManifest {
    pub(super) fn to_skill_definition(&self) -> SkillDefinition {
        SkillDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            prompt: self.prompt.clone(),
            tools: self.tools.clone(),
            mcp_servers: self.mcp_servers.clone(),
        }
    }

    pub(super) fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter_map(|tool_name| get_skill_tool_definition(tool_name))
            .collect()
    }

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

    pub(super) fn build_examples(&self) -> Vec<(String, String)> {
        self.examples
            .iter()
            .map(|ex| (ex.user.clone(), ex.assistant.clone()))
            .collect()
    }
}

fn get_skill_tool_definition(name: &str) -> Option<ToolDefinition> {
    let definitions: HashMap<&str, ToolDefinition> = get_skill_tools_map();
    definitions.get(name).cloned()
}

fn get_skill_tools_map() -> HashMap<&'static str, ToolDefinition> {
    use serde_json::json;

    let mut map = HashMap::new();

    map.insert(
        "code_review",
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "code_review".to_string(),
                description: "Perform a code review on the given code".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to review"},
                        "language": {"type": "string", "description": "The programming language"}
                    },
                    "required": ["code"]
                }),
            },
        },
    );

    map.insert("debug", ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "debug".to_string(),
            description: "Debug code and find issues".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to debug"},
                    "error_message": {"type": "string", "description": "The error message if any"}
                },
                "required": ["code"]
            }),
        },
    });

    map.insert("refactor", ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "refactor".to_string(),
            description: "Refactor code to improve quality".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to refactor"},
                    "goals": {"type": "array", "items": {"type": "string"}, "description": "Refactoring goals"}
                },
                "required": ["code"]
            }),
        },
    });

    map.insert(
        "test_generate",
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "test_generate".to_string(),
                description: "Generate tests for the given code".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to test"},
                        "framework": {"type": "string", "description": "The test framework to use"}
                    },
                    "required": ["code"]
                }),
            },
        },
    );

    map.insert("document", ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "document".to_string(),
            description: "Generate documentation for code".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to document"},
                    "style": {"type": "string", "description": "Documentation style (rustdoc, jsdoc, etc)"}
                },
                "required": ["code"]
            }),
        },
    });

    map
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
        },
    ]
}
