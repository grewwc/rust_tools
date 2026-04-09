use std::{
    fs,
    path::PathBuf,
    sync::OnceLock,
};

use serde::{Deserialize, Serialize};

use crate::commonw::utils::expanduser;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CodeDiscoveryConfidence {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CodeDiscoveryKind {
    ErrorSite,
    RootCause,
    EntryPoint,
    CallChain,
    Symbol,
    CodePath,
    Config,
    Todo,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CodeDiscoveryRecord {
    pub(super) finding: String,
    pub(super) kind: CodeDiscoveryKind,
    pub(super) confidence: CodeDiscoveryConfidence,
}

#[derive(Debug, Clone)]
struct CodeDiscoveryPolicy {
    classification_rules: Vec<ClassificationRule>,
    recall_max_items: usize,
    persistence_max_per_turn: usize,
    min_persist_confidence: CodeDiscoveryConfidence,
    confidence_weight: ConfidenceWeights,
    kind_weight: KindWeights,
    priority_weight: ConfidenceWeights,
}

#[derive(Debug, Clone)]
struct ClassificationRule {
    enabled: bool,
    tool_names: Vec<String>,
    any_contains: Vec<String>,
    all_contains: Vec<String>,
    none_contains: Vec<String>,
    kind: CodeDiscoveryKind,
    confidence: CodeDiscoveryConfidence,
}

#[derive(Debug, Clone, Copy)]
struct ConfidenceWeights {
    low: i32,
    medium: i32,
    high: i32,
}

#[derive(Debug, Clone, Copy)]
struct KindWeights {
    error_site: i32,
    root_cause: i32,
    entry_point: i32,
    call_chain: i32,
    symbol: i32,
    code_path: i32,
    config: i32,
    todo: i32,
}

#[derive(Debug, Default, Deserialize)]
struct PolicyOverride {
    classification: Option<ClassificationSectionOverride>,
    recall: Option<RecallSectionOverride>,
    persistence: Option<PersistenceSectionOverride>,
}

#[derive(Debug, Default, Deserialize)]
struct ClassificationSectionOverride {
    rules: Option<Vec<ClassificationRuleOverride>>,
}

#[derive(Debug, Deserialize)]
struct ClassificationRuleOverride {
    enabled: Option<bool>,
    tool_names: Option<Vec<String>>,
    #[serde(rename = "match")]
    match_spec: Option<MatchSpecOverride>,
    kind: CodeDiscoveryKind,
    confidence: CodeDiscoveryConfidence,
}

#[derive(Debug, Default, Deserialize)]
struct MatchSpecOverride {
    any_contains: Option<Vec<String>>,
    all_contains: Option<Vec<String>>,
    none_contains: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
struct RecallSectionOverride {
    max_items: Option<usize>,
    confidence_weight: Option<ConfidenceWeightsOverride>,
    kind_weight: Option<KindWeightsOverride>,
}

#[derive(Debug, Default, Deserialize)]
struct PersistenceSectionOverride {
    max_persist_per_turn: Option<usize>,
    min_confidence: Option<CodeDiscoveryConfidence>,
    priority_weight: Option<ConfidenceWeightsOverride>,
}

#[derive(Debug, Default, Deserialize)]
struct ConfidenceWeightsOverride {
    low: Option<i32>,
    medium: Option<i32>,
    high: Option<i32>,
}

#[derive(Debug, Default, Deserialize)]
struct KindWeightsOverride {
    error_site: Option<i32>,
    root_cause: Option<i32>,
    entry_point: Option<i32>,
    call_chain: Option<i32>,
    symbol: Option<i32>,
    code_path: Option<i32>,
    config: Option<i32>,
    todo: Option<i32>,
}

static POLICY: OnceLock<CodeDiscoveryPolicy> = OnceLock::new();

pub(super) fn classify_finding(
    tool_name: &str,
    highlight: &str,
    rendered: &str,
) -> Option<CodeDiscoveryRecord> {
    let tool_name = tool_name.trim().to_ascii_lowercase();
    let normalized = highlight.trim().to_ascii_lowercase();
    let policy = code_discovery_policy();

    for rule in &policy.classification_rules {
        if !rule.enabled {
            continue;
        }
        if !rule.tool_names.is_empty() && !rule.tool_names.iter().any(|name| name == &tool_name) {
            continue;
        }
        if !rule.any_contains.is_empty()
            && !rule
                .any_contains
                .iter()
                .any(|needle| normalized.contains(needle))
        {
            continue;
        }
        if !rule
            .all_contains
            .iter()
            .all(|needle| normalized.contains(needle))
        {
            continue;
        }
        if rule
            .none_contains
            .iter()
            .any(|needle| normalized.contains(needle))
        {
            continue;
        }

        return Some(CodeDiscoveryRecord {
            finding: rendered.to_string(),
            kind: rule.kind,
            confidence: rule.confidence,
        });
    }

    None
}

pub(super) fn render_record(record: &CodeDiscoveryRecord) -> String {
    format!(
        "- [confidence={} kind={}] {}",
        confidence_label(record.confidence),
        kind_label(record.kind),
        record.finding
    )
}

pub(super) fn parse_record_line(line: &str) -> Option<CodeDiscoveryRecord> {
    let line = line.trim();
    let rest = line.strip_prefix("- [confidence=")?;
    let (confidence, rest) = rest.split_once(" kind=")?;
    let (kind, finding) = rest.split_once("] ")?;
    Some(CodeDiscoveryRecord {
        finding: finding.trim().to_string(),
        confidence: parse_confidence(confidence.trim())?,
        kind: parse_kind(kind.trim())?,
    })
}

pub(super) fn parse_confidence(value: &str) -> Option<CodeDiscoveryConfidence> {
    match value {
        "low" => Some(CodeDiscoveryConfidence::Low),
        "medium" => Some(CodeDiscoveryConfidence::Medium),
        "high" => Some(CodeDiscoveryConfidence::High),
        _ => None,
    }
}

pub(super) fn parse_kind(value: &str) -> Option<CodeDiscoveryKind> {
    match value {
        "error_site" => Some(CodeDiscoveryKind::ErrorSite),
        "root_cause" => Some(CodeDiscoveryKind::RootCause),
        "entry_point" => Some(CodeDiscoveryKind::EntryPoint),
        "call_chain" => Some(CodeDiscoveryKind::CallChain),
        "symbol" => Some(CodeDiscoveryKind::Symbol),
        "code_path" => Some(CodeDiscoveryKind::CodePath),
        "config" => Some(CodeDiscoveryKind::Config),
        "todo" => Some(CodeDiscoveryKind::Todo),
        _ => None,
    }
}

pub(super) fn confidence_label(confidence: CodeDiscoveryConfidence) -> &'static str {
    match confidence {
        CodeDiscoveryConfidence::Low => "low",
        CodeDiscoveryConfidence::Medium => "medium",
        CodeDiscoveryConfidence::High => "high",
    }
}

pub(super) fn kind_label(kind: CodeDiscoveryKind) -> &'static str {
    match kind {
        CodeDiscoveryKind::ErrorSite => "error_site",
        CodeDiscoveryKind::RootCause => "root_cause",
        CodeDiscoveryKind::EntryPoint => "entry_point",
        CodeDiscoveryKind::CallChain => "call_chain",
        CodeDiscoveryKind::Symbol => "symbol",
        CodeDiscoveryKind::CodePath => "code_path",
        CodeDiscoveryKind::Config => "config",
        CodeDiscoveryKind::Todo => "todo",
    }
}

pub(super) fn persistence_limit() -> usize {
    code_discovery_policy().persistence_max_per_turn
}

pub(super) fn recall_limit() -> usize {
    code_discovery_policy().recall_max_items
}

pub(super) fn should_persist(confidence: CodeDiscoveryConfidence) -> bool {
    confidence >= code_discovery_policy().min_persist_confidence
}

pub(super) fn priority_for_confidence(confidence: CodeDiscoveryConfidence) -> u8 {
    let weight = match confidence {
        CodeDiscoveryConfidence::Low => code_discovery_policy().priority_weight.low,
        CodeDiscoveryConfidence::Medium => code_discovery_policy().priority_weight.medium,
        CodeDiscoveryConfidence::High => code_discovery_policy().priority_weight.high,
    };
    weight.clamp(0, 255) as u8
}

pub(super) fn recall_rank(record: &CodeDiscoveryRecord) -> i32 {
    let policy = code_discovery_policy();
    let confidence = match record.confidence {
        CodeDiscoveryConfidence::Low => policy.confidence_weight.low,
        CodeDiscoveryConfidence::Medium => policy.confidence_weight.medium,
        CodeDiscoveryConfidence::High => policy.confidence_weight.high,
    };
    let kind = match record.kind {
        CodeDiscoveryKind::ErrorSite => policy.kind_weight.error_site,
        CodeDiscoveryKind::RootCause => policy.kind_weight.root_cause,
        CodeDiscoveryKind::EntryPoint => policy.kind_weight.entry_point,
        CodeDiscoveryKind::CallChain => policy.kind_weight.call_chain,
        CodeDiscoveryKind::Symbol => policy.kind_weight.symbol,
        CodeDiscoveryKind::CodePath => policy.kind_weight.code_path,
        CodeDiscoveryKind::Config => policy.kind_weight.config,
        CodeDiscoveryKind::Todo => policy.kind_weight.todo,
    };
    confidence + kind
}

fn code_discovery_policy() -> &'static CodeDiscoveryPolicy {
    POLICY.get_or_init(load_policy)
}

fn load_policy() -> CodeDiscoveryPolicy {
    let mut policy = default_policy();
    for path in policy_override_paths() {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        match serde_json::from_str::<PolicyOverride>(&text) {
            Ok(override_policy) => apply_override(&mut policy, override_policy),
            Err(err) => eprintln!(
                "[code_discovery_policy] failed to parse {}: {}",
                path.display(),
                err
            ),
        }
    }
    policy
}

fn policy_override_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    paths.push(PathBuf::from(
        expanduser("~/.config/rust_tools/code_discovery_policy.json").as_ref(),
    ));
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join(".rust_tools/code_discovery_policy.json"));
    }
    paths
}

fn apply_override(policy: &mut CodeDiscoveryPolicy, override_policy: PolicyOverride) {
    if let Some(classification) = override_policy.classification
        && let Some(rules) = classification.rules
    {
        let compiled = rules
            .into_iter()
            .filter_map(compile_rule_override)
            .collect::<Vec<_>>();
        if !compiled.is_empty() {
            policy.classification_rules = compiled;
        }
    }

    if let Some(recall) = override_policy.recall {
        if let Some(max_items) = recall.max_items.filter(|value| *value > 0) {
            policy.recall_max_items = max_items;
        }
        if let Some(weights) = recall.confidence_weight {
            apply_confidence_weights(&mut policy.confidence_weight, weights);
        }
        if let Some(weights) = recall.kind_weight {
            apply_kind_weights(&mut policy.kind_weight, weights);
        }
    }

    if let Some(persistence) = override_policy.persistence {
        if let Some(limit) = persistence.max_persist_per_turn.filter(|value| *value > 0) {
            policy.persistence_max_per_turn = limit;
        }
        if let Some(min_confidence) = persistence.min_confidence {
            policy.min_persist_confidence = min_confidence;
        }
        if let Some(weights) = persistence.priority_weight {
            apply_confidence_weights(&mut policy.priority_weight, weights);
        }
    }
}

fn compile_rule_override(rule: ClassificationRuleOverride) -> Option<ClassificationRule> {
    let match_spec = rule.match_spec.unwrap_or_default();
    let any_contains = normalize_needles(match_spec.any_contains.unwrap_or_default());
    let all_contains = normalize_needles(match_spec.all_contains.unwrap_or_default());
    let none_contains = normalize_needles(match_spec.none_contains.unwrap_or_default());
    if any_contains.is_empty() && all_contains.is_empty() && none_contains.is_empty() {
        return None;
    }

    Some(ClassificationRule {
        enabled: rule.enabled.unwrap_or(true),
        tool_names: normalize_needles(rule.tool_names.unwrap_or_default()),
        any_contains,
        all_contains,
        none_contains,
        kind: rule.kind,
        confidence: rule.confidence,
    })
}

fn normalize_needles(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn apply_confidence_weights(target: &mut ConfidenceWeights, weights: ConfidenceWeightsOverride) {
    if let Some(low) = weights.low {
        target.low = low;
    }
    if let Some(medium) = weights.medium {
        target.medium = medium;
    }
    if let Some(high) = weights.high {
        target.high = high;
    }
}

fn apply_kind_weights(target: &mut KindWeights, weights: KindWeightsOverride) {
    if let Some(value) = weights.error_site {
        target.error_site = value;
    }
    if let Some(value) = weights.root_cause {
        target.root_cause = value;
    }
    if let Some(value) = weights.entry_point {
        target.entry_point = value;
    }
    if let Some(value) = weights.call_chain {
        target.call_chain = value;
    }
    if let Some(value) = weights.symbol {
        target.symbol = value;
    }
    if let Some(value) = weights.code_path {
        target.code_path = value;
    }
    if let Some(value) = weights.config {
        target.config = value;
    }
    if let Some(value) = weights.todo {
        target.todo = value;
    }
}

fn default_policy() -> CodeDiscoveryPolicy {
    CodeDiscoveryPolicy {
        classification_rules: vec![
            rule(
                &["read_file", "read_file_lines", "code_search", "grep_search"],
                &["root cause", "caused by", "because ", "due to "],
                &[],
                &[],
                CodeDiscoveryKind::RootCause,
                CodeDiscoveryConfidence::High,
            ),
            rule(
                &["code_search", "read_file", "read_file_lines"],
                &["call chain", "caller", "callee", "invokes ", " -> "],
                &[],
                &[],
                CodeDiscoveryKind::CallChain,
                CodeDiscoveryConfidence::Medium,
            ),
            rule(
                &["read_file", "read_file_lines", "code_search"],
                &["entry point", "bootstrap", "startup", "fn main(", "app::run("],
                &[],
                &[],
                CodeDiscoveryKind::EntryPoint,
                CodeDiscoveryConfidence::High,
            ),
            rule(
                &["read_file", "read_file_lines", "code_search", "grep_search"],
                &["panic", "error", "failed", "missing"],
                &[],
                &[],
                CodeDiscoveryKind::ErrorSite,
                CodeDiscoveryConfidence::High,
            ),
            rule(
                &["read_file", "read_file_lines", "code_search"],
                &["fn ", "class ", "struct ", "impl "],
                &[],
                &[],
                CodeDiscoveryKind::Symbol,
                CodeDiscoveryConfidence::High,
            ),
            rule(
                &["read_file", "read_file_lines", "code_search"],
                &["todo", "fixme"],
                &[],
                &[],
                CodeDiscoveryKind::Todo,
                CodeDiscoveryConfidence::Medium,
            ),
            rule(
                &["read_file", "read_file_lines", "code_search", "grep_search"],
                &["config", "feature flag", ".toml"],
                &[],
                &[],
                CodeDiscoveryKind::Config,
                CodeDiscoveryConfidence::Medium,
            ),
            rule(
                &["code_search"],
                &[".rs:"],
                &[],
                &[],
                CodeDiscoveryKind::CodePath,
                CodeDiscoveryConfidence::Medium,
            ),
            rule(
                &["read_file", "read_file_lines", "grep_search"],
                &[".rs:"],
                &[],
                &[],
                CodeDiscoveryKind::CodePath,
                CodeDiscoveryConfidence::Low,
            ),
        ],
        recall_max_items: 8,
        persistence_max_per_turn: 3,
        min_persist_confidence: CodeDiscoveryConfidence::Medium,
        confidence_weight: ConfidenceWeights {
            low: 100,
            medium: 200,
            high: 300,
        },
        kind_weight: KindWeights {
            error_site: 60,
            root_cause: 70,
            entry_point: 50,
            call_chain: 40,
            symbol: 30,
            code_path: 10,
            config: 20,
            todo: 0,
        },
        priority_weight: ConfidenceWeights {
            low: 120,
            medium: 160,
            high: 200,
        },
    }
}

fn rule(
    tool_names: &[&str],
    any_contains: &[&str],
    all_contains: &[&str],
    none_contains: &[&str],
    kind: CodeDiscoveryKind,
    confidence: CodeDiscoveryConfidence,
) -> ClassificationRule {
    ClassificationRule {
        enabled: true,
        tool_names: tool_names.iter().map(|value| value.to_string()).collect(),
        any_contains: any_contains.iter().map(|value| value.to_string()).collect(),
        all_contains: all_contains.iter().map(|value| value.to_string()).collect(),
        none_contains: none_contains.iter().map(|value| value.to_string()).collect(),
        kind,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CodeDiscoveryConfidence, CodeDiscoveryKind, CodeDiscoveryRecord, PolicyOverride,
        apply_override, classify_finding, default_policy, parse_record_line, recall_rank,
        render_record,
    };

    #[test]
    fn classify_finding_uses_default_root_cause_rule() {
        let record = classify_finding(
            "read_file_lines",
            "root cause: config cache is empty due to missing APP_ENV",
            "- read_file_lines(file=src/main.rs, lines=1..20) => root cause: config cache is empty due to missing APP_ENV",
        )
        .expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::RootCause);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn render_and_parse_record_round_trip() {
        let record = CodeDiscoveryRecord {
            finding: "code_search(...) => fn main()".to_string(),
            kind: CodeDiscoveryKind::EntryPoint,
            confidence: CodeDiscoveryConfidence::High,
        };
        let rendered = render_record(&record);
        assert_eq!(parse_record_line(&rendered), Some(record));
    }

    #[test]
    fn override_updates_kind_weights() {
        let mut policy = default_policy();
        let override_policy: PolicyOverride = serde_json::from_str(
            r#"{
              "recall": {
                "kind_weight": {
                  "root_cause": 999
                }
              }
            }"#,
        )
        .unwrap();
        apply_override(&mut policy, override_policy);

        let high_root = CodeDiscoveryRecord {
            finding: "a".to_string(),
            kind: CodeDiscoveryKind::RootCause,
            confidence: CodeDiscoveryConfidence::High,
        };
        let high_symbol = CodeDiscoveryRecord {
            finding: "b".to_string(),
            kind: CodeDiscoveryKind::Symbol,
            confidence: CodeDiscoveryConfidence::High,
        };
        assert!(recall_rank(&high_root) > recall_rank(&high_symbol));
    }
}
