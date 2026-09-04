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
//   - model_tier: light/standard/heavy preference
//
// Builtin agents:
//   - build: Default unified development agent (planning, code-writing, execution, prompt engineering, exploration)
//   - audit: Evidence-driven code, configuration, prompt, and behavior reviewer
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
    ("audit.agent", include_str!("builtin_agents/audit.agent")),
    (
        "audit-fast.agent",
        include_str!("builtin_agents/audit-fast.agent"),
    ),
];
const PROJECT_INSTRUCTION_FILENAMES: &[&str] = &[
    "AGENTS.md",
    "Agents.md",
    "agents.md",
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
const TARGET_SCOPED_INSTRUCTION_MAX_TOTAL_CHARS: usize = 16_000;

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

/// 加载已经被本轮工具调用触达的目标文件所适用、但 cwd 基础范围尚未包含的 scoped
/// 指令。这样从仓库根启动时，读取 `src/bin/ai/driver/foo.rs` 后，下一次模型请求能
/// 同时看到 `src/bin/ai/AGENTS.md` 与 `src/bin/ai/driver/AGENTS.md`。
pub(super) fn load_scoped_project_instruction_docs_for_targets(
    targets: &[PathBuf],
) -> Vec<ProjectInstructionDoc> {
    let Ok(cwd) = crate::ai::driver::runtime_ctx::effective_cwd() else {
        return Vec::new();
    };
    load_scoped_project_instruction_docs_for_target_priority_from(&cwd, targets, &[])
}

/// 加载 scoped 指令时，先为当前必须解锁的 mutation 目标分配预算，再用本轮其它已
/// 观察目标填充剩余预算。安全 preflight 不能与普通历史读取竞争同一个无优先级池，
/// 否则长 turn 中旧目标会让新 mutation 永远拿不到规则、反复暂停。
pub(super) fn load_scoped_project_instruction_docs_for_target_priority(
    required_targets: &[PathBuf],
    observed_targets: &[PathBuf],
) -> Vec<ProjectInstructionDoc> {
    let Ok(cwd) = crate::ai::driver::runtime_ctx::effective_cwd() else {
        return Vec::new();
    };
    load_scoped_project_instruction_docs_for_target_priority_from(
        &cwd,
        required_targets,
        observed_targets,
    )
}

fn load_scoped_project_instruction_docs_for_targets_from(
    cwd: &Path,
    targets: &[PathBuf],
) -> Vec<ProjectInstructionDoc> {
    load_scoped_project_instruction_docs_for_target_priority_from(cwd, targets, &[])
}

fn load_scoped_project_instruction_docs_for_target_priority_from(
    cwd: &Path,
    required_targets: &[PathBuf],
    observed_targets: &[PathBuf],
) -> Vec<ProjectInstructionDoc> {
    if required_targets.is_empty() && observed_targets.is_empty() {
        return Vec::new();
    }
    let base_docs = load_project_instruction_docs_from(cwd);
    let base_paths = base_docs
        .iter()
        .map(|doc| doc.path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let scope = project_instruction_search_scope(cwd);
    let project_root = scope.first().map(PathBuf::as_path).unwrap_or(cwd);
    let canonical_root = fs::canonicalize(project_root).unwrap_or_else(|_| project_root.into());

    let mut seen_paths = std::collections::BTreeSet::new();
    let mut required_candidates = Vec::new();
    let mut observed_candidates = Vec::new();
    for (targets, candidates) in [
        (required_targets, &mut required_candidates),
        (observed_targets, &mut observed_candidates),
    ] {
        for target in targets {
            let absolute = if target.is_absolute() {
                target.clone()
            } else {
                cwd.join(target)
            };
            let Some(parent) = absolute.parent() else {
                continue;
            };
            let Ok(target_dir) = fs::canonicalize(parent) else {
                continue;
            };
            if !target_dir.starts_with(&canonical_root) {
                continue;
            }
            for doc in load_project_instruction_candidates_from(&target_dir) {
                if base_paths.contains(doc.path.as_str()) || !seen_paths.insert(doc.path.clone()) {
                    continue;
                }
                candidates.push(doc);
            }
        }
    }
    // 每个优先级内部先选最具体规则；required 整组始终排在 observed 之前。
    for candidates in [&mut required_candidates, &mut observed_candidates] {
        candidates.sort_by(|a, b| {
            instruction_doc_depth(b)
                .cmp(&instruction_doc_depth(a))
                .then_with(|| a.path.cmp(&b.path))
        });
    }
    let mut selected = Vec::new();
    let mut used = 0usize;
    for doc in required_candidates
        .into_iter()
        .chain(observed_candidates.into_iter())
    {
        if used >= TARGET_SCOPED_INSTRUCTION_MAX_TOTAL_CHARS {
            break;
        }
        let remaining = TARGET_SCOPED_INSTRUCTION_MAX_TOTAL_CHARS - used;
        let content = truncate_instruction_doc(&doc.content, remaining);
        if content.is_empty() {
            continue;
        }
        used += content.chars().count();
        selected.push(ProjectInstructionDoc {
            path: doc.path,
            content,
        });
    }
    selected.sort_by(|a, b| {
        instruction_doc_depth(a)
            .cmp(&instruction_doc_depth(b))
            .then_with(|| a.path.cmp(&b.path))
    });
    selected
}

fn instruction_doc_depth(doc: &ProjectInstructionDoc) -> usize {
    Path::new(&doc.path).components().count()
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
    candidates: Vec<ProjectInstructionDoc>,
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
        let candidates = discover_project_instruction_docs(cwd);
        let docs =
            limit_project_instruction_docs(candidates.clone(), PROJECT_INSTRUCTION_MAX_TOTAL_CHARS);
        cache.insert(
            key,
            ProjectInstructionCacheEntry {
                fingerprint,
                candidates,
                docs: docs.clone(),
            },
        );
        return docs;
    }

    // 锁中毒走 uncached 路径，不影响正确性。
    load_project_instruction_docs_uncached(cwd)
}

fn load_project_instruction_docs_uncached(cwd: &Path) -> Vec<ProjectInstructionDoc> {
    let candidates = discover_project_instruction_docs(cwd);
    limit_project_instruction_docs(candidates, PROJECT_INSTRUCTION_MAX_TOTAL_CHARS)
}

fn load_project_instruction_candidates_from(cwd: &Path) -> Vec<ProjectInstructionDoc> {
    let fingerprint = fingerprint_project_instruction_files(cwd);
    if let Ok(mut cache) = project_instruction_cache().lock() {
        let key = cwd.to_path_buf();
        if let Some(entry) = cache.get_ref(&key)
            && entry.fingerprint == fingerprint
        {
            return entry.candidates.clone();
        }
        let candidates = discover_project_instruction_docs(cwd);
        let docs =
            limit_project_instruction_docs(candidates.clone(), PROJECT_INSTRUCTION_MAX_TOTAL_CHARS);
        cache.insert(
            key,
            ProjectInstructionCacheEntry {
                fingerprint,
                candidates: candidates.clone(),
                docs,
            },
        );
        return candidates;
    }
    discover_project_instruction_docs(cwd)
}

fn limit_project_instruction_docs(
    candidates: Vec<ProjectInstructionDoc>,
    max_chars: usize,
) -> Vec<ProjectInstructionDoc> {
    let mut docs = Vec::new();
    let mut used = 0usize;
    for doc in candidates {
        if used >= max_chars {
            break;
        }
        let budget = max_chars - used;
        let content = truncate_instruction_doc(&doc.content, budget);
        if content.is_empty() {
            continue;
        }
        used += content.chars().count();
        docs.push(ProjectInstructionDoc {
            path: doc.path,
            content,
        });
    }
    docs
}

fn discover_project_instruction_docs(cwd: &Path) -> Vec<ProjectInstructionDoc> {
    let mut docs = Vec::new();
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
            let content = truncate_instruction_doc(trimmed, PROJECT_INSTRUCTION_MAX_DOC_CHARS);
            if !content.is_empty() {
                docs.push(ProjectInstructionDoc {
                    path: canonical_key,
                    content,
                });
            }
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

    let mut scope = match boundary {
        Some(idx) => ancestors[..=idx].iter().rev().cloned().collect(),
        None => vec![cwd.to_path_buf()],
    };

    // User-supplied per-project instructions: `~/.config/rust_tools/<project-name>/`
    // (AGENTS.md / agents.md / ...), where `<project-name>` is the leaf name of the project
    // root. Insert right after the project root so these are treated as project-level docs
    // (loaded before deeper repo-local files and not starved by the total instruction budget).
    let project_root = scope.first().map(PathBuf::as_path).unwrap_or(cwd);
    if let Some(cfg_dir) = project_config_instruction_dir(project_root)
        && !scope.iter().any(|dir| dir == &cfg_dir)
    {
        scope.insert(1, cfg_dir);
    }

    scope
}

/// User config instruction dir for a project: `~/.config/rust_tools/<project-name>/`.
/// Returns None when the leaf name of `project_root` is unusable or the config dir cannot
/// be derived (no HOME).
fn project_config_instruction_dir(project_root: &Path) -> Option<PathBuf> {
    let project_name = project_root.file_name()?.to_str()?;
    if project_name.is_empty() {
        return None;
    }
    let config_root = crate::commonw::utils::get_config_dir()?.join("rust_tools");
    Some(config_root.join(project_name))
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
    name
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

    // Same rationale as the skill-side check in skills.rs: `get_tool_definitions_by_names`
    // silently skips `tools:` entries absent from the registry, so warn here rather
    // than letting `.agent` authors chase permanently-missing tools.
    if let Some(unknown) = super::tools::manifest_unknown_tool_names_warning(&tools) {
        eprintln!(
            "[agents] agent \"{}\": `tools` lists unknown tool(s): {} \
             (must be a registered builtin name or an mcp_* server tool)",
            name, unknown
        );
    }

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
#[path = "agents_tests.rs"]
mod tests;
