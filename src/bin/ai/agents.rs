// =============================================================================
// AIOS Agents - Agent Definitions and Loading
// =============================================================================
// Agents are LLM-powered assistants with specific personalities and capabilities.
//
// Agent files (.agent) contain YAML front-matter and a prompt body:
//   - name: Agent identifier
//   - description: For agent routing selection
//   - mode: primary, subagent, or all
//   - model: Override default model
//   - temperature: Override default temperature
//   - tools/tool_groups/mcp_servers/disable_mcp_tools: Available tools
//   - routing_tags: For ML-based routing
//   - model_tier: light/standard/heavy preference
//
// Builtin agents:
//   - build: Default code-writing agent
//   - executor: Background task execution
//   - plan: Multi-step planning
//   - explore: Codebase exploration
//   - prompt-skill: Skill creation
// =============================================================================

use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use crate::commonw::{configw, utils::expanduser};

const BUILTIN_AGENTS: &[(&str, &str)] = &[
    ("build.agent", include_str!("builtin_agents/build.agent")),
    (
        "executor.agent",
        include_str!("builtin_agents/executor.agent"),
    ),
    ("plan.agent", include_str!("builtin_agents/plan.agent")),
    (
        "explore.agent",
        include_str!("builtin_agents/explore.agent"),
    ),
    (
        "prompt-skill.agent",
        include_str!("builtin_agents/prompt-skill.agent"),
    ),
];
const PROJECT_INSTRUCTION_FILENAMES: &[&str] = &[
    "AGENTS.md",
    "Agent.md",
    "agent.md",
    "CLAUDE.md",
    "Claude.md",
    "claude.md",
];
const PROJECT_ROOT_MARKERS: &[&str] = &[
    ".git",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "pom.xml",
    "Gemfile",
];
const PROJECT_INSTRUCTION_MAX_DOC_CHARS: usize = 8_000;
const PROJECT_INSTRUCTION_MAX_TOTAL_CHARS: usize = 16_000;

/// 已识别的项目语言/构建体系类型，用来在 system prompt 里给 agent 一些
/// 默认约定（构建/测试命令、惯用工具等），减少"摸索式"工具调用。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectKind {
    Rust,
    NodeJs,
    Python,
    Go,
    JavaMaven,
    JavaGradle,
    Ruby,
}

impl ProjectKind {
    /// system prompt 里展示给 LLM 的简短描述（语言 + 推荐构建/测试命令）。
    /// 注意：这里只是默认建议，仓库根的 AGENTS.md 等指令文件可覆盖。
    pub(super) fn prompt_hint(self) -> &'static str {
        match self {
            ProjectKind::Rust => {
                "Rust project (Cargo.toml). Prefer: `cargo check` for fast type-check, \
                 `cargo test` for tests, `cargo clippy` for lint. \
                 Use Rust idioms (Result, ?, Box<dyn Error>)."
            }
            ProjectKind::NodeJs => {
                "Node.js / TypeScript project (package.json). Prefer: `npm test` / \
                 `pnpm test` / `yarn test` based on lockfile. Check scripts in package.json \
                 before guessing build commands."
            }
            ProjectKind::Python => {
                "Python project (pyproject.toml). Prefer: `pytest` for tests, \
                 `python -m build` or project-defined entrypoints. Respect virtualenv if active."
            }
            ProjectKind::Go => {
                "Go project (go.mod). Prefer: `go build ./...`, `go test ./...`, \
                 `go vet ./...`. Module path is in go.mod."
            }
            ProjectKind::JavaMaven => {
                "Java project (pom.xml, Maven). Prefer: `mvn -q compile` / `mvn -q test`."
            }
            ProjectKind::JavaGradle => {
                "Java project (build.gradle / build.gradle.kts, Gradle). Prefer: \
                 `./gradlew build` / `./gradlew test`."
            }
            ProjectKind::Ruby => {
                "Ruby project (Gemfile). Prefer: `bundle exec rspec` / `bundle exec rake`."
            }
        }
    }
}

/// 从 `cwd` 起向上查找直到 root marker，根据命中的 manifest 文件推断项目类型。
/// 返回首个命中的类型；若全部不命中返回 None。
/// 与 `project_instruction_search_scope` 共用 ancestor 遍历语义。
pub(super) fn detect_project_kind(cwd: &Path) -> Option<ProjectKind> {
    let home_dir = std::env::var_os("HOME").map(PathBuf::from);
    for dir in cwd.ancestors() {
        if home_dir.as_deref() == Some(dir) && dir != cwd {
            break;
        }
        // 优先级：Cargo > go.mod > pyproject > package.json > pom > gradle > Gemfile
        if dir.join("Cargo.toml").is_file() {
            return Some(ProjectKind::Rust);
        }
        if dir.join("go.mod").is_file() {
            return Some(ProjectKind::Go);
        }
        if dir.join("pyproject.toml").is_file() {
            return Some(ProjectKind::Python);
        }
        if dir.join("package.json").is_file() {
            return Some(ProjectKind::NodeJs);
        }
        if dir.join("pom.xml").is_file() {
            return Some(ProjectKind::JavaMaven);
        }
        if dir.join("build.gradle").is_file() || dir.join("build.gradle.kts").is_file() {
            return Some(ProjectKind::JavaGradle);
        }
        if dir.join("Gemfile").is_file() {
            return Some(ProjectKind::Ruby);
        }
        // 命中 root marker（如裸 .git）但没有上述 manifest 时停止上溯。
        if has_project_root_marker(dir) {
            return None;
        }
    }
    None
}

pub(super) fn detect_project_kind_from_cwd() -> Option<ProjectKind> {
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok()?;
    detect_project_kind(&cwd)
}

/// Categorizes an agent's role: `Primary` for main conversation,
/// `Subagent` for delegated tasks, or `All` for both.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum AgentMode {
    #[serde(rename = "primary")]
    Primary,
    #[serde(rename = "subagent")]
    Subagent,
    #[serde(rename = "all")]
    All,
}

impl Default for AgentMode {
    fn default() -> Self {
        AgentMode::All
    }
}

/// Declares the preferred model strength tier for an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum AgentModelTier {
    #[serde(rename = "light")]
    Light,
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "heavy")]
    Heavy,
}

/// Parsed configuration for an agent, loaded from a `.agent` file
/// with front-matter metadata and a prompt body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AgentManifest {
    pub(super) name: String,
    pub(super) description: String,
    #[serde(default)]
    pub(super) mode: AgentMode,
    #[serde(default)]
    pub(super) model: Option<String>,
    #[serde(default)]
    pub(super) temperature: Option<f64>,
    #[serde(default)]
    pub(super) max_steps: Option<usize>,
    #[serde(default)]
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) system_prompt: Option<String>,
    #[serde(default)]
    pub(super) tools: Vec<String>,
    #[serde(default)]
    pub(super) tool_groups: Vec<String>,
    #[serde(default)]
    pub(super) mcp_servers: Vec<String>,
    #[serde(default)]
    pub(super) disable_mcp_tools: bool,
    #[serde(default)]
    pub(super) routing_tags: Vec<String>,
    #[serde(default)]
    pub(super) model_tier: Option<AgentModelTier>,
    #[serde(default)]
    pub(super) disabled: bool,
    #[serde(default)]
    pub(super) hidden: bool,
    #[serde(default)]
    pub(super) color: Option<String>,
    #[serde(skip)]
    pub(super) source_path: Option<String>,
}

impl AgentManifest {
    pub(super) fn build_system_prompt(&self) -> String {
        let mut prompt = self.system_prompt.clone().unwrap_or_default();

        if !self.prompt.is_empty() {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(self.prompt.as_str());
        }

        prompt
    }

    pub(super) fn is_primary(&self) -> bool {
        matches!(self.mode, AgentMode::Primary | AgentMode::All)
    }

    pub(super) fn is_subagent(&self) -> bool {
        matches!(self.mode, AgentMode::Subagent | AgentMode::All)
    }

    pub(super) fn routing_tags_normalized(&self) -> Vec<String> {
        self.routing_tags
            .iter()
            .map(|tag| tag.trim().to_ascii_lowercase())
            .filter(|tag| !tag.is_empty())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProjectInstructionDoc {
    pub(super) path: String,
    pub(super) content: String,
}

/// Discovery level for an agent, used to determine precedence.
/// Higher priority levels override lower ones when agents share the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum DiscoveryLevel {
    /// Built-in agents shipped with the binary
    Builtin = 0,
    /// Global user-level agents (~/.config/rust_tools/agents/)
    User = 1,
    /// Workspace-level agents (from config)
    Workspace = 2,
    /// Project-level agents (.agents/ or agents/ near cwd)
    Project = 3,
}

/// Loads all builtin and user-defined agents, merging them by name
/// with precedence: project > workspace > user > builtin.
pub(super) fn load_all_agents() -> Vec<AgentManifest> {
    let dir = agents_dir();
    let _ = ensure_seeded_agents_dir(&dir);
    let mut by_name: Box<SkipMap<String, (DiscoveryLevel, AgentManifest)>> =
        SkipMap::new(32, |a: &String, b: &String| a.cmp(b) as i32);

    // Level 0: Built-in agents (lowest precedence)
    for (filename, content) in BUILTIN_AGENTS {
        if let Ok(mut agent) = parse_agent_front_matter(content) {
            agent.source_path = Some(format!("builtin:{filename}"));
            by_name.insert(agent.name.clone(), (DiscoveryLevel::Builtin, agent));
        }
    }

    // Level 1: User-level agents from config dir
    for agent in load_agents_from_dir_with_level(&dir, DiscoveryLevel::User) {
        let should_insert = match by_name.get_ref(&agent.name) {
            Some((level, _)) => DiscoveryLevel::User > *level,
            None => true,
        };
        if should_insert {
            by_name.insert(agent.name.clone(), (DiscoveryLevel::User, agent));
        }
    }

    // Level 2: Workspace-level agents
    if let Some(ref ws_dir) = workspace_agents_dir() {
        for agent in load_agents_from_dir_with_level(ws_dir, DiscoveryLevel::Workspace) {
            let should_insert = match by_name.get_ref(&agent.name) {
                Some((level, _)) => DiscoveryLevel::Workspace > *level,
                None => true,
            };
            if should_insert {
                by_name.insert(agent.name.clone(), (DiscoveryLevel::Workspace, agent));
            }
        }
    }

    // Level 3: Project-level agents (highest precedence)
    for project_dir in discover_project_dirs() {
        for agent in load_agents_from_dir_with_level(&project_dir, DiscoveryLevel::Project) {
            let should_insert = match by_name.get_ref(&agent.name) {
                Some((level, _)) => DiscoveryLevel::Project > *level,
                None => true,
            };
            if should_insert {
                by_name.insert(agent.name.clone(), (DiscoveryLevel::Project, agent));
            }
        }
    }

    let mut out: Vec<AgentManifest> = (&*by_name)
        .into_iter()
        .map(|(_, (_, v))| v.clone())
        .collect();
    out.sort_by(|a, b| {
        let primary_a = a.is_primary() as i32;
        let primary_b = b.is_primary() as i32;
        primary_b.cmp(&primary_a).then(a.name.cmp(&b.name))
    });
    out
}

/// Returns the workspace-level agents directory if configured.
fn workspace_agents_dir() -> Option<PathBuf> {
    let cfg = configw::get_all_config();
    let raw = cfg.get_opt("ai.agents.workspace_dir")?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(expanduser(raw.trim()).as_ref()))
}

/// Discovers project-level agent directories.
/// Looks for `.agents/` or `agents/` in the current working directory.
fn discover_project_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(cwd) = crate::ai::driver::runtime_ctx::effective_cwd() {
        let dot_agents = cwd.join(".agents");
        if dot_agents.is_dir() {
            dirs.push(dot_agents.clone());
        }
        let plain_agents = cwd.join("agents");
        if plain_agents.is_dir() && plain_agents != dot_agents {
            dirs.push(plain_agents);
        }
    }

    dirs
}

pub(super) fn load_project_instruction_docs() -> Vec<ProjectInstructionDoc> {
    let Ok(cwd) = crate::ai::driver::runtime_ctx::effective_cwd() else {
        return Vec::new();
    };
    load_project_instruction_docs_from(&cwd)
}

/// 用 (path, len, mtime) 指纹缓存项目指令文档，避免每个 turn 都重新做磁盘
/// I/O + truncate。实测中 AGENTS.md / CLAUDE.md 在一个会话里几乎不会变化，
/// 但 build_system_prompt 每个 turn / 每个 iteration 都要拿一次它们，单
/// 次最大 16KB×ancestors，token 与 syscall 都不便宜。
///
/// 缓存语义保证：只要任一参与文件的 (len, mtime) 发生变化，或扫描范围里
/// 出现/消失了文件，就重新加载。命中时返回的是上次构建好的 Vec 的 clone，
/// 内容与未缓存路径完全一致。
type ProjectInstructionFingerprint = Vec<(PathBuf, u64, Option<SystemTime>)>;

struct ProjectInstructionCacheEntry {
    fingerprint: ProjectInstructionFingerprint,
    docs: Vec<ProjectInstructionDoc>,
}

fn project_instruction_cache() -> &'static Mutex<SkipMap<PathBuf, ProjectInstructionCacheEntry>> {
    static CACHE: OnceLock<Mutex<SkipMap<PathBuf, ProjectInstructionCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SkipMap::default()))
}

fn fingerprint_project_instruction_files(cwd: &Path) -> ProjectInstructionFingerprint {
    let mut entries: ProjectInstructionFingerprint = Vec::new();
    for dir in project_instruction_search_scope(cwd) {
        for name in PROJECT_INSTRUCTION_FILENAMES {
            let path = dir.join(name);
            // 注意：与 load_project_instruction_docs_uncached 保持完全一致的发现顺序；
            // 这里不做 canonicalize（原实现仅对去重做 canonicalize，发现顺序仍按
            // path），fingerprint 比较的是"扫描看到的文件序列+元数据"。
            let Ok(meta) = fs::metadata(&path) else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let len = meta.len();
            let mtime = meta.modified().ok();
            entries.push((path, len, mtime));
        }
    }
    entries
}

fn load_project_instruction_docs_from(cwd: &Path) -> Vec<ProjectInstructionDoc> {
    let fingerprint = fingerprint_project_instruction_files(cwd);

    if let Ok(mut cache) = project_instruction_cache().lock() {
        let key = cwd.to_path_buf();
        if let Some(entry) = cache.get_ref(&key) {
            if entry.fingerprint == fingerprint {
                return entry.docs.clone();
            }
        }
        let docs = load_project_instruction_docs_uncached(cwd);
        cache.insert(
            key,
            ProjectInstructionCacheEntry {
                fingerprint,
                docs: docs.clone(),
            },
        );
        return docs;
    }

    // 锁中毒走 uncached 路径，不影响正确性。
    load_project_instruction_docs_uncached(cwd)
}

fn load_project_instruction_docs_uncached(cwd: &Path) -> Vec<ProjectInstructionDoc> {
    let mut docs = Vec::new();
    let mut used = 0usize;
    let mut seen_paths = std::collections::BTreeSet::new();

    for dir in project_instruction_search_scope(cwd) {
        for name in PROJECT_INSTRUCTION_FILENAMES {
            let path = dir.join(name);
            if !path.is_file() {
                continue;
            }
            let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            let canonical_key = canonical.display().to_string();
            if !seen_paths.insert(canonical_key.clone()) {
                continue;
            }
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let trimmed = content.trim();
            if trimmed.is_empty() {
                continue;
            }
            if used >= PROJECT_INSTRUCTION_MAX_TOTAL_CHARS {
                return docs;
            }
            let budget = PROJECT_INSTRUCTION_MAX_TOTAL_CHARS.saturating_sub(used);
            let limited =
                truncate_instruction_doc(trimmed, PROJECT_INSTRUCTION_MAX_DOC_CHARS.min(budget));
            if limited.is_empty() {
                continue;
            }
            used += limited.chars().count();
            docs.push(ProjectInstructionDoc {
                path: canonical_key,
                content: limited,
            });
        }
    }

    docs
}

fn project_instruction_search_scope(cwd: &Path) -> Vec<PathBuf> {
    let home_dir = std::env::var_os("HOME").map(PathBuf::from);
    let mut ancestors = Vec::new();
    for dir in cwd.ancestors() {
        if home_dir.as_deref() == Some(dir) && dir != cwd {
            break;
        }
        ancestors.push(dir.to_path_buf());
    }

    let boundary = ancestors
        .iter()
        .rposition(|dir| has_project_root_marker(dir))
        .or_else(|| {
            ancestors
                .iter()
                .rposition(|dir| has_project_instruction_doc(dir))
        });

    match boundary {
        Some(idx) => ancestors[..=idx].iter().rev().cloned().collect(),
        None => vec![cwd.to_path_buf()],
    }
}

fn has_project_root_marker(dir: &Path) -> bool {
    PROJECT_ROOT_MARKERS
        .iter()
        .any(|name| dir.join(name).exists())
}

fn has_project_instruction_doc(dir: &Path) -> bool {
    PROJECT_INSTRUCTION_FILENAMES
        .iter()
        .any(|name| dir.join(name).is_file())
}

fn truncate_instruction_doc(content: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (idx, ch) in content.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

/// Filters agents that can serve as primary agents, excluding
/// disabled and hidden ones.
pub(super) fn get_primary_agents(agents: &[AgentManifest]) -> Vec<&AgentManifest> {
    agents
        .iter()
        .filter(|a| a.is_primary() && !a.disabled && !a.hidden)
        .collect()
}

/// Filters agents that can be spawned as subagents, excluding
/// disabled and hidden ones.
pub(super) fn get_subagents(agents: &[AgentManifest]) -> Vec<&AgentManifest> {
    agents
        .iter()
        .filter(|a| a.is_subagent() && !a.disabled && !a.hidden)
        .collect()
}

pub(super) fn find_agent_by_name<'a>(
    agents: &'a [AgentManifest],
    name: &str,
) -> Option<&'a AgentManifest> {
    let canonical = canonical_agent_name(name);
    agents
        .iter()
        .find(|a| a.name == canonical || a.name == name)
}

pub(super) fn canonical_agent_name(name: &str) -> &str {
    match name {
        "openclaw" => "executor",
        other => other,
    }
}

pub(crate) fn agents_dir() -> PathBuf {
    let cfg = configw::get_all_config();
    let raw = cfg.get_opt("ai.agents.dir").unwrap_or_default();
    let path = if raw.trim().is_empty() {
        "~/.config/rust_tools/agents".to_string()
    } else {
        raw
    };
    PathBuf::from(expanduser(&path).as_ref())
}

/// 返回 builtin agent 的文件名集合（如 "build.agent"）。供 save_agent
/// 工具校验，禁止覆盖编译进二进制的 builtin。
pub(crate) fn builtin_agent_filenames() -> &'static [&'static str] {
    const NAMES: &[&str] = &[
        "build.agent",
        "executor.agent",
        "plan.agent",
        "explore.agent",
        "prompt-skill.agent",
    ];
    NAMES
}

fn looks_like_front_matter_agent(content: &str) -> bool {
    content.lines().next().is_some_and(|l| l.trim() == "---")
}

fn parse_agent_front_matter(content: &str) -> Result<AgentManifest, String> {
    let mut lines = content.lines();
    let Some(first) = lines.next() else {
        return Err("empty agent file".to_string());
    };
    if first.trim() != "---" {
        return Err("missing front matter start".to_string());
    }

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut mode: Option<String> = None;
    let mut model: Option<String> = None;
    let mut temperature: Option<f64> = None;
    let mut max_steps: Option<usize> = None;
    let mut system_prompt: Option<String> = None;
    let mut model_tier: Option<String> = None;
    let mut disabled = false;
    let mut hidden = false;
    let mut color: Option<String> = None;
    let mut tools: Vec<String> = Vec::new();
    let mut tool_groups: Vec<String> = Vec::new();
    let mut mcp_servers: Vec<String> = Vec::new();
    let mut disable_mcp_tools = false;
    let mut routing_tags: Vec<String> = Vec::new();

    let mut body = String::new();
    let mut in_front_matter = true;
    let mut pending_list_key: Option<String> = None;

    for line in lines {
        if in_front_matter {
            if line.trim() == "---" {
                in_front_matter = false;
                pending_list_key = None;
                continue;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(key) = pending_list_key.as_deref()
                && trimmed.starts_with('-')
            {
                let v = trimmed.trim_start_matches('-').trim();
                let v = v.trim_matches('"').trim_matches('\'').to_string();
                if v.is_empty() {
                    continue;
                }
                match key {
                    "tools" => tools.push(v),
                    "tool_groups" => tool_groups.push(v),
                    "mcp_servers" => mcp_servers.push(v),
                    "routing_tags" => routing_tags.push(v),
                    _ => {}
                }
                continue;
            }

            pending_list_key = None;

            let Some((k, v)) = trimmed.split_once(':') else {
                continue;
            };
            let key = k.trim();
            let value = v.trim();

            if value.is_empty() {
                pending_list_key = Some(key.to_string());
                continue;
            }

            let unquoted = value.trim_matches('"').trim_matches('\'');
            match key {
                "name" => name = Some(unquoted.to_string()),
                "description" => description = Some(unquoted.to_string()),
                "mode" => mode = Some(unquoted.to_string()),
                "model" => model = Some(unquoted.to_string()),
                "model_tier" => model_tier = Some(unquoted.to_string()),
                "system_prompt" => system_prompt = Some(unquoted.to_string()),
                "color" => color = Some(unquoted.to_string()),
                "temperature" => {
                    temperature = unquoted.parse::<f64>().ok();
                }
                "max_steps" => {
                    max_steps = unquoted.parse::<usize>().ok();
                }
                "disabled" => {
                    disabled = unquoted.eq_ignore_ascii_case("true");
                }
                "hidden" => {
                    hidden = unquoted.eq_ignore_ascii_case("true");
                }
                "disable_mcp_tools" => {
                    disable_mcp_tools = unquoted.eq_ignore_ascii_case("true");
                }
                "tools" => tools = parse_list_value(unquoted),
                "tool_groups" => tool_groups = parse_list_value(unquoted),
                "mcp_servers" => mcp_servers = parse_list_value(unquoted),
                "routing_tags" => routing_tags = parse_list_value(unquoted),
                _ => {}
            }
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }

    if in_front_matter {
        return Err("missing front matter end".to_string());
    }

    let Some(name) = name else {
        return Err("missing name".to_string());
    };

    let Some(description) = description else {
        return Err("missing description".to_string());
    };

    let agent_mode = match mode.as_deref() {
        Some("primary") => AgentMode::Primary,
        Some("subagent") => AgentMode::Subagent,
        Some("all") => AgentMode::All,
        None => AgentMode::All,
        Some(other) => return Err(format!("invalid mode: {}", other)),
    };
    let agent_model_tier = match model_tier.as_deref() {
        Some("light") => Some(AgentModelTier::Light),
        Some("standard") => Some(AgentModelTier::Standard),
        Some("heavy") => Some(AgentModelTier::Heavy),
        None => None,
        Some(other) => return Err(format!("invalid model_tier: {}", other)),
    };

    Ok(AgentManifest {
        name,
        description,
        mode: agent_mode,
        model: model.filter(|s| !s.trim().is_empty()),
        temperature,
        max_steps,
        prompt: body.trim().to_string(),
        system_prompt: system_prompt.filter(|s| !s.trim().is_empty()),
        tools,
        tool_groups,
        mcp_servers,
        disable_mcp_tools,
        routing_tags,
        model_tier: agent_model_tier,
        disabled,
        hidden,
        color: color.filter(|s| !s.trim().is_empty()),
        source_path: None,
    })
}

fn parse_list_value(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let s = s.trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .map(|x| x.trim().trim_matches('"').trim_matches('\''))
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect()
}

fn parse_agent_front_matter_with_path(content: &str, path: &Path) -> Result<AgentManifest, String> {
    let mut agent = parse_agent_front_matter(content)?;
    agent.source_path = Some(path.display().to_string());
    Ok(agent)
}

fn ensure_seeded_agents_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("failed to create agents dir: {e}"))?;
    Ok(())
}

/// Loads agents from a directory, annotated with the given discovery level.
/// The level is used for logging and precedence resolution in `load_all_agents`.
fn load_agents_from_dir_with_level(dir: &Path, level: DiscoveryLevel) -> Vec<AgentManifest> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !looks_like_front_matter_agent(&content) {
            continue;
        }
        if let Ok(agent) = parse_agent_front_matter_with_path(&content, &path) {
            out.push(agent);
        }
    }
    if !out.is_empty() {
        let level_name = match level {
            DiscoveryLevel::Builtin => "builtin",
            DiscoveryLevel::User => "user",
            DiscoveryLevel::Workspace => "workspace",
            DiscoveryLevel::Project => "project",
        };
        eprintln!(
            "[agent discovery] loaded {} agent(s) from {} ({})",
            out.len(),
            dir.display(),
            level_name
        );
    }
    out
}

/// Legacy wrapper for backward compatibility.
fn load_agents_from_dir(dir: &Path) -> Vec<AgentManifest> {
    load_agents_from_dir_with_level(dir, DiscoveryLevel::User)
}

#[cfg(test)]
mod tests {
    use super::{
        AgentModelTier, BUILTIN_AGENTS, load_project_instruction_docs_from,
        parse_agent_front_matter,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "rust_tools_agents_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn parses_routing_tags_and_model_tier_from_front_matter() {
        let content = r#"---
name: explore
description: Fast read-only codebase exploration
mode: subagent
model_tier: light
routing_tags:
  - find
  - search
  - read-only
---

Read the codebase and summarize findings.
"#;

        let agent = parse_agent_front_matter(content).unwrap();
        assert_eq!(agent.name, "explore");
        assert_eq!(agent.model_tier, Some(AgentModelTier::Light));
        assert_eq!(
            agent.routing_tags,
            vec![
                "find".to_string(),
                "search".to_string(),
                "read-only".to_string()
            ]
        );
    }

    #[test]
    fn rejects_invalid_model_tier_in_front_matter() {
        let content = r#"---
name: bad
description: invalid tier
model_tier: giant
---

noop
"#;

        let err = parse_agent_front_matter(content).unwrap_err();
        assert!(err.contains("invalid model_tier"));
    }

    #[test]
    fn parses_disable_mcp_tools_from_front_matter() {
        let content = r#"---
name: build
description: Development agent
disable_mcp_tools: true
---

Build things.
"#;

        let agent = parse_agent_front_matter(content).unwrap();

        assert!(agent.disable_mcp_tools);
    }

    #[test]
    fn builtin_agents_do_not_mount_mcp_tools_by_default() {
        for (filename, content) in BUILTIN_AGENTS {
            let agent = parse_agent_front_matter(content).unwrap();
            assert!(
                agent.disable_mcp_tools,
                "{filename} should use progressive MCP loading instead of mounting every MCP tool"
            );
        }
    }

    #[test]
    fn prompt_skill_builtin_uses_explicit_tools_not_whole_builtin_group() {
        let prompt_skill = BUILTIN_AGENTS
            .iter()
            .find_map(|(filename, content)| {
                (*filename == "prompt-skill.agent")
                    .then(|| parse_agent_front_matter(content).unwrap())
            })
            .unwrap();

        assert!(!prompt_skill.tools.is_empty());
        assert!(prompt_skill.tool_groups.is_empty());
    }

    #[test]
    fn project_instruction_docs_include_root_and_nested_scope() {
        let root = temp_dir("project_docs");
        let nested = root.join("packages/app/src");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("AGENTS.md"), "# Root rules\nUse pnpm.\n").unwrap();
        fs::write(
            root.join("packages/app/CLAUDE.md"),
            "# App rules\nRun app tests only.\n",
        )
        .unwrap();

        let docs = load_project_instruction_docs_from(&nested);
        assert_eq!(docs.len(), 2);
        assert!(docs[0].path.ends_with("AGENTS.md"));
        assert!(docs[0].content.contains("Use pnpm."));
        assert!(docs[1].path.ends_with("CLAUDE.md"));
        assert!(docs[1].content.contains("Run app tests only."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn project_instruction_docs_fall_back_to_doc_ancestors_without_repo_markers() {
        let root = temp_dir("project_docs_nomarker");
        let nested = root.join("services/api/src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join("claude.md"),
            "# Project rules\nPrefer make targets.\n",
        )
        .unwrap();

        let docs = load_project_instruction_docs_from(&nested);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].path.ends_with("claude.md"));
        assert!(docs[0].content.contains("Prefer make targets."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn project_instruction_docs_cache_invalidates_on_content_change() {
        // 该测试锁住缓存语义：只要文件 mtime/len 变化就必须重新读盘，缓存
        // 不能让 LLM 看到旧版指令。
        let root = temp_dir("project_docs_cache");
        fs::create_dir_all(root.join(".git")).unwrap();
        let agents_md = root.join("AGENTS.md");
        fs::write(&agents_md, "v1: use pnpm.\n").unwrap();

        let first = load_project_instruction_docs_from(&root);
        assert_eq!(first.len(), 1);
        assert!(first[0].content.contains("v1: use pnpm."));

        // 同样输入再调一次，结果应等价（命中缓存或不命中都允许，只要内容一致）。
        let cached = load_project_instruction_docs_from(&root);
        assert_eq!(cached, first);

        // 改文件并睡眠确保 mtime 推进；同时显式让 len 变化，双重保险触发
        // fingerprint 失配。
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(
            &agents_md,
            "v2: use cargo and longer content for len change.\n",
        )
        .unwrap();

        let after = load_project_instruction_docs_from(&root);
        assert_eq!(after.len(), 1);
        assert!(
            after[0].content.contains("v2: use cargo"),
            "cache must invalidate on file change, got: {}",
            after[0].content
        );

        let _ = fs::remove_dir_all(root);
    }
}
