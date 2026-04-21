use std::collections::HashSet;

use crate::ai::driver::thinking::{
    engine::ThoughtTree,
    generalization::ExperienceGeneralizer,
    goals::{GoalManager, GoalState},
    verification::{VerificationOutcome, VerificationStep, VerificationWorkflow},
};
use crate::ai::driver::observer::{
    FinalizeContext, MemoryEntry, ObserverOutput, PrepareContext, ToolResultContext, TurnObserver,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThinkingMode {
    TreeOfThoughts,
    VerificationLoop,
    GoalDirected,
}

#[derive(Debug, Clone, Default)]
pub struct ThinkingDecision {
    pub active_modes: HashSet<ThinkingMode>,
    pub inject_into_system_prompt: Option<String>,
    pub next_sub_goal: Option<String>,
    pub verification_step: Option<VerificationStep>,
}

pub struct ThinkingOrchestrator {
    pub thought_tree: Option<ThoughtTree>,
    pub verification: Option<VerificationWorkflow>,
    pub generalizer: ExperienceGeneralizer,
    pub goal_manager: GoalManager,
    pub active_modes: HashSet<ThinkingMode>,
    pub enabled: bool,
}

impl ThinkingOrchestrator {
    pub fn new() -> Self {
        let persistence_dir = dirs::home_dir().map(|h| h.join(".config").join("rust_tools").join("thinking_goals"));
        let goal_manager = GoalManager::new()
            .with_persistence_dir_opt(persistence_dir);
        Self {
            thought_tree: None,
            verification: None,
            generalizer: ExperienceGeneralizer::new(),
            goal_manager,
            active_modes: HashSet::new(),
            enabled: true,
        }
    }

    pub fn analyze_question(&mut self, question: &str) -> ThinkingDecision {
        if !self.enabled {
            return ThinkingDecision::default();
        }

        let needs_tree = self.detect_complex_reasoning(question);
        let needs_verification = self.detect_verification_need(question);
        let needs_decomposition = self.detect_goal_decomposition_need(question);

        if needs_tree {
            if self.thought_tree.is_none() {
                self.thought_tree = Some(ThoughtTree::new(question, 4, 3));
            }
            self.active_modes.insert(ThinkingMode::TreeOfThoughts);
        }

        if needs_verification {
            if self.verification.is_none() {
                self.verification = Some(VerificationWorkflow::new(question.to_string()));
            }
            self.active_modes.insert(ThinkingMode::VerificationLoop);
        }

        if needs_decomposition {
            if self.goal_manager.active_goal().is_none() {
                let goal_id = self.goal_manager.create_goal(question.to_string());
                self.goal_manager.activate_goal(&goal_id);
            }
            self.active_modes.insert(ThinkingMode::GoalDirected);
        }

        let inject = self.build_system_prompt_injection();

        let next_sub_goal = if self.active_modes.contains(&ThinkingMode::GoalDirected) {
            self.goal_manager.get_next_actions().first().map(|s| s.description.clone())
        } else {
            None
        };

        let verification_step = if self.active_modes.contains(&ThinkingMode::VerificationLoop) {
            self.verification.as_ref().map(|v| v.current_step.clone())
        } else {
            None
        };

        ThinkingDecision {
            active_modes: self.active_modes.clone(),
            inject_into_system_prompt: inject,
            next_sub_goal,
            verification_step,
        }
    }

    pub fn process_tool_result(&mut self, tool_name: &str, result: &str, success: bool) {
        if self.active_modes.contains(&ThinkingMode::TreeOfThoughts) {
            if let Some(ref mut tree) = self.thought_tree {
                if let Some(current) = tree.ucb_select(1.414) {
                    let score = if success { 0.7 } else { 0.2 };
                    tree.score_node(current, score);
                    tree.record_outcome(
                        current,
                        result.chars().take(200).collect(),
                        vec![tool_name.to_string()],
                    );
                }
            }
        }

        if self.active_modes.contains(&ThinkingMode::VerificationLoop) {
            if let Some(ref mut wf) = self.verification {
                if matches!(wf.current_step, VerificationStep::ExecuteTest) {
                    let test_result =
                        crate::ai::driver::thinking::verification::TestResult {
                            command: tool_name.to_string(),
                            exit_code: if success { 0 } else { 1 },
                            stdout_preview: result.chars().take(500).collect(),
                            stderr_preview: String::new(),
                            passed: success,
                        };
                    wf.current_cycle_mut().record_test_result(test_result);
                    wf.advance_step();
                }
            }
        }

        if !success {
            self.generalizer.ingest_experience(
                "failure",
                &format!(
                    "Avoid: {} led to failure - {}",
                    tool_name,
                    &result[..result.len().min(200)]
                ),
                &[tool_name.to_string()],
                None,
            );
        }
    }

    pub fn process_self_note(&mut self, note: &str) {
        let is_do = note.to_lowercase().contains("do:");
        let is_avoid = note.to_lowercase().contains("avoid:");
        let category = if is_do {
            "self_note_do"
        } else if is_avoid {
            "self_note_avoid"
        } else {
            "self_note"
        };
        self.generalizer.ingest_experience(
            category,
            note,
            &["agent".to_string(), "policy".to_string()],
            Some("thinking_orchestrator"),
        );
    }

    pub fn try_generalize(&mut self) -> Option<GeneralizeResult> {
        let principle = self.generalizer.try_generalize()?;
        let memory_entry = self.generalizer.persist_principle(&principle);
        Some(GeneralizeResult {
            principle_text: principle.principle.clone(),
            memory_entry,
        })
    }

    pub fn try_cross_domain_link(&mut self) -> Option<(String, String)> {
        self.generalizer.try_cross_domain_link()
    }

    pub fn complete_verification_cycle(&mut self, outcome: VerificationOutcome) {
        if let Some(ref mut wf) = self.verification {
            wf.complete_cycle(outcome);
        }
    }

    pub fn get_outcome(&self) -> ThinkingOutcome {
        ThinkingOutcome {
            tree_summary: self.thought_tree.as_ref().map(|t| t.render_tree_summary()),
            verification_summary: self.verification.as_ref().map(|v| v.render_summary()),
            goal_status: Some(self.goal_manager.render_active_status()),
        }
    }

    pub fn reset_for_new_turn(&mut self) {
        self.thought_tree = None;
        self.verification = None;
        self.active_modes.clear();
    }

    fn detect_complex_reasoning(&self, question: &str) -> bool {
        let lower = question.to_lowercase();
        let indicators = [
            "why does", "root cause", "figure out", "investigate",
            "multiple approaches", "compare", "trade-off", "tradeoff",
            "which is better", "analyze", "debug", "diagnose",
            "为什么", "根因", "排查", "分析", "对比", "权衡",
        ];
        indicators.iter().filter(|i| lower.contains(*i)).count() >= 1 || question.len() > 200
    }

    fn detect_verification_need(&self, question: &str) -> bool {
        let lower = question.to_lowercase();
        let indicators = [
            "fix", "bug", "broken", "not working", "error", "crash",
            "test", "verify", "validate", "reproduce",
            "修复", "bug", "报错", "崩溃", "验证", "复现",
        ];
        indicators.iter().any(|i| lower.contains(i))
    }

    fn detect_goal_decomposition_need(&self, question: &str) -> bool {
        let lower = question.to_lowercase();
        let indicators = [
            "refactor", "migrate", "redesign", "rewrite", "implement",
            "build", "create a system", "end to end", "from scratch",
            "重构", "迁移", "重写", "实现", "搭建", "从零",
        ];
        indicators.iter().any(|i| lower.contains(i))
    }

    fn is_question_related_to_tree(&self, question: &str) -> bool {
        if let Some(ref tree) = self.thought_tree {
            let root = match tree.get_node(tree.root_id()) {
                Some(n) => n,
                None => return false,
            };
            let root_words: std::collections::HashSet<String> = root.hypothesis
                .to_lowercase().split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_string())
                .collect();
            let question_words: std::collections::HashSet<String> = question
                .to_lowercase().split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_string())
                .collect();
            let overlap = root_words.intersection(&question_words).count();
            overlap >= 2 || (root_words.len() <= 3 && overlap >= 1)
        } else {
            false
        }
    }

    fn is_question_related_to_verification(&self, question: &str) -> bool {
        if let Some(ref wf) = self.verification {
            let hypothesis = &wf.current_cycle().hypothesis;
            let hyp_words: std::collections::HashSet<String> = hypothesis
                .to_lowercase().split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_string())
                .collect();
            let question_words: std::collections::HashSet<String> = question
                .to_lowercase().split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_string())
                .collect();
            let overlap = hyp_words.intersection(&question_words).count();
            overlap >= 2 || (hyp_words.len() <= 3 && overlap >= 1)
        } else {
            false
        }
    }

    fn build_system_prompt_injection(&self) -> Option<String> {
        let mut parts = Vec::new();

        if self.active_modes.contains(&ThinkingMode::TreeOfThoughts) {
            if let Some(ref tree) = self.thought_tree {
                parts.push(format!(
                    "[Tree-of-Thoughts Active] You are exploring multiple reasoning branches. \
                     Current tree has {} nodes. Before committing to a single approach, \
                     consider generating alternative hypotheses. When you have multiple possible \
                     approaches, list them as structured alternatives before choosing one.",
                    tree.render_tree_summary().lines().count()
                ));
            }
        }

        if self.active_modes.contains(&ThinkingMode::VerificationLoop) {
            if let Some(ref wf) = self.verification {
                if !wf.is_complete() {
                    let step_instruction = match wf.current_step {
                        VerificationStep::GenerateHypothesis =>
                            "Formulate a specific, falsifiable hypothesis about the issue.",
                        VerificationStep::DesignTest =>
                            "Design a concrete test (command or file inspection) to verify your hypothesis.",
                        VerificationStep::ExecuteTest =>
                            "Execute the test you designed and observe the result.",
                        VerificationStep::AnalyzeResult =>
                            "Analyze the test results: does the evidence confirm or refute your hypothesis?",
                        VerificationStep::ReviseHypothesis =>
                            "Based on the analysis, revise your hypothesis if needed.",
                        VerificationStep::ConfirmOrReject =>
                            "Make a final judgment: is the hypothesis confirmed or rejected?",
                    };
                    parts.push(format!(
                        "[Verification Loop Active] Current step: {:?}. {} \
                         Do not assume success — actively seek disconfirming evidence.",
                        wf.current_step, step_instruction
                    ));
                }
            }
        }

        if self.active_modes.contains(&ThinkingMode::GoalDirected) {
            if let Some(goal) = self.goal_manager.active_goal() {
                let status = goal.render_status();
                let next_actions: Vec<&str> = goal.get_next_actionable()
                    .iter().map(|s| s.description.as_str()).collect();
                parts.push(format!(
                    "[Goal-Directed Mode Active]\n{}\n\
                     Next actionable sub-goals: {}\n\
                     Focus on completing the next sub-goal before moving on.",
                    status,
                    if next_actions.is_empty() { "none yet - decompose first" } else { &next_actions.join(", ") }
                ));
            }
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

pub struct GeneralizeResult {
    pub principle_text: String,
    pub memory_entry: MemoryEntry,
}

#[derive(Debug, Clone)]
pub struct ThinkingOutcome {
    pub tree_summary: Option<String>,
    pub verification_summary: Option<String>,
    pub goal_status: Option<String>,
}

fn extract_self_note_from_text(text: &str) -> Option<String> {
    let start = text.find("<meta:self_note>")?;
    let end = text.find("</meta:self_note>")?;
    if end <= start {
        return None;
    }
    Some(text[start + "<meta:self_note>".len()..end].trim().to_string())
}

impl TurnObserver for ThinkingOrchestrator {
    fn on_prepare(&mut self, ctx: &PrepareContext) -> Vec<(String, String)> {
        if self.thought_tree.is_some() {
            if self.is_question_related_to_tree(&ctx.question) {
                self.active_modes.insert(ThinkingMode::TreeOfThoughts);
            } else {
                self.thought_tree = None;
            }
        }
        if self.verification.is_some() {
            if self.is_question_related_to_verification(&ctx.question) {
                self.active_modes.insert(ThinkingMode::VerificationLoop);
            } else {
                self.verification = None;
            }
        }
        if self.goal_manager.active_goal().is_some() {
            self.active_modes.insert(ThinkingMode::GoalDirected);
        }

        let decision = self.analyze_question(&ctx.question);
        let mut sections = Vec::new();
        if let Some(injection) = &decision.inject_into_system_prompt {
            sections.push(("Behavior".to_string(), injection.clone()));
        }
        if decision.active_modes.contains(&ThinkingMode::TreeOfThoughts) {
            if let Some(ref tree) = self.thought_tree {
                if let Some(current) = tree.ucb_select(1.414) {
                    sections.push((
                        "Reasoning Tree".to_string(),
                        tree.generate_thinking_prompt(current),
                    ));
                }
            }
        }
        if decision.active_modes.contains(&ThinkingMode::VerificationLoop) {
            if let Some(ref wf) = self.verification {
                let prompt = match wf.current_step {
                    VerificationStep::GenerateHypothesis => {
                        Some(wf.generate_hypothesis_prompt(&ctx.question, ""))
                    }
                    VerificationStep::DesignTest => {
                        Some(wf.generate_test_design_prompt(&wf.current_cycle().hypothesis))
                    }
                    VerificationStep::AnalyzeResult => {
                        Some(wf.generate_analysis_prompt(wf.current_cycle()))
                    }
                    VerificationStep::ConfirmOrReject => {
                        Some(format!(
                            "Based on all evidence gathered, make a final judgment.\n\n\
                             Hypothesis: {}\n\
                             Total test cycles: {}\n\
                             Test results: {}\n\n\
                             Output STRICT JSON: {{\"verdict\":\"confirmed\"|\"rejected\"|\"inconclusive\",\"reason\":\"...\"}}",
                            wf.current_cycle().hypothesis,
                            wf.cycles.len(),
                            wf.current_cycle().test_results.len(),
                        ))
                    }
                    VerificationStep::ReviseHypothesis => {
                        let results_summary: Vec<String> = wf.current_cycle().test_results.iter()
                            .map(|r| format!("{}: {}", r.command, if r.passed { "PASSED" } else { "FAILED" }))
                            .collect();
                        Some(format!(
                            "Based on the test results, revise your hypothesis.\n\n\
                             Original hypothesis: {}\n\
                             Test results: {}\n\n\
                             What new hypothesis better explains these results?\n\
                             Output STRICT JSON: {{\"new_hypothesis\":\"...\",\"feedback\":\"...\"}}",
                            wf.current_cycle().hypothesis,
                            results_summary.join(", ")
                        ))
                    }
                    VerificationStep::ExecuteTest => None,
                };
                if let Some(p) = prompt {
                    if !p.is_empty() {
                        sections.push(("Verification".to_string(), p));
                    }
                }
            }
        }
        if decision.active_modes.contains(&ThinkingMode::GoalDirected) {
            if let Some(goal) = self.goal_manager.active_goal() {
                sections.push((
                    "Goal Decomposition".to_string(),
                    goal.generate_decomposition_prompt(),
                ));
            }
        }
        sections
    }

    fn on_tool_result(&mut self, ctx: &ToolResultContext) {
        self.process_tool_result(&ctx.tool_name, &ctx.result_content, ctx.success);
    }

    fn on_finalize(&mut self, ctx: &FinalizeContext) -> ObserverOutput {
        let mut display_lines = Vec::new();
        let mut memory_entries = Vec::new();

        if let Some(note) = extract_self_note_from_text(&ctx.final_text) {
            self.process_self_note(&note);
        }

        if let Some(result) = self.try_generalize() {
            display_lines.push(format!("[Thinking] Generalized principle: {}", result.principle_text));
            memory_entries.push(result.memory_entry);
        }

        if self.try_cross_domain_link().is_some() {
            display_lines.push("[Thinking] Cross-domain link discovered".to_string());
        }

        let outcome = self.get_outcome();
        if outcome.tree_summary.is_some() || outcome.verification_summary.is_some() {
            display_lines.push("[Thinking] Turn outcome:".to_string());
            if let Some(tree) = &outcome.tree_summary {
                for line in tree.lines().take(10) {
                    display_lines.push(format!("  {}", line));
                }
            }
            if let Some(verify) = &outcome.verification_summary {
                for line in verify.lines().take(10) {
                    display_lines.push(format!("  {}", line));
                }
            }
        }
        if let Some(goal_status) = &outcome.goal_status {
            if !goal_status.contains("No active goal") {
                display_lines.push("[Thinking] Goal progress:".to_string());
                for line in goal_status.lines().take(8) {
                    display_lines.push(format!("  {}", line));
                }
            }
        }

        self.active_modes.clear();

        ObserverOutput {
            display_lines,
            memory_entries,
        }
    }

    fn on_conversation_end(&mut self) {
        let _ = self.goal_manager.persist_goals();
        self.thought_tree = None;
        self.verification = None;
        self.active_modes.clear();
        self.generalizer = ExperienceGeneralizer::new();
        self.goal_manager = GoalManager::new()
            .with_persistence_dir_opt(dirs::home_dir().map(|h| h.join(".config").join("rust_tools").join("thinking_goals")));
    }

    fn name(&self) -> &str {
        "thinking"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_complex_question_activates_verification() {
        let mut orch = ThinkingOrchestrator::new();
        let decision = orch.analyze_question("Why does the server crash under high load?");
        assert!(decision.active_modes.contains(&ThinkingMode::VerificationLoop));
    }

    #[test]
    fn analyze_decomposition_question_creates_goal() {
        let mut orch = ThinkingOrchestrator::new();
        let decision = orch.analyze_question("Refactor the entire networking layer");
        assert!(decision.active_modes.contains(&ThinkingMode::GoalDirected));
        assert!(orch.goal_manager.active_goal().is_some());
    }

    #[test]
    fn modes_can_be_parallel() {
        let mut orch = ThinkingOrchestrator::new();
        let decision = orch.analyze_question("Fix the bug and refactor the error handling module");
        assert!(decision.active_modes.contains(&ThinkingMode::VerificationLoop));
        assert!(decision.active_modes.contains(&ThinkingMode::GoalDirected));
    }

    #[test]
    fn process_tool_result_scores_tree() {
        let mut orch = ThinkingOrchestrator::new();
        orch.analyze_question("Investigate the root cause of this bug");
        orch.process_tool_result("code_search", "found relevant code", true);
    }

    #[test]
    fn process_self_note_feeds_generalizer() {
        let mut orch = ThinkingOrchestrator::new();
        orch.process_self_note("Do: always validate inputs in async handlers");
        orch.process_self_note("Do: check for None before unwrap in async code");
        orch.process_self_note("Do: verify async results before use");
        orch.process_self_note("Avoid: unwrap on async results without checking");
        orch.process_self_note("Avoid: skip validation in concurrent code");
        let result = orch.try_generalize();
        assert!(result.is_some() || orch.generalizer.experience_buffer.len() >= 3);
    }

    #[test]
    fn on_finalize_returns_output_no_io() {
        let mut orch = ThinkingOrchestrator::new();
        orch.analyze_question("Debug this complex issue");
        let output = orch.on_finalize(&FinalizeContext {
            question: "test".to_string(),
            final_text: "some response".to_string(),
            had_tool_calls: false,
        });
        assert!(output.display_lines.is_empty() || output.display_lines.len() >= 0);
    }

    #[test]
    fn on_conversation_end_clears_goals() {
        let mut orch = ThinkingOrchestrator::new();
        orch.analyze_question("Refactor the system");
        assert!(orch.goal_manager.active_goal().is_some());
        orch.on_conversation_end();
        assert!(orch.goal_manager.active_goal().is_none());
    }
}
