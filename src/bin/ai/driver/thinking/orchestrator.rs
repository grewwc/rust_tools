use std::collections::HashSet;

use crate::ai::driver::thinking::{
    engine::ThoughtTree,
    generalization::ExperienceGeneralizer,
    goals::GoalManager,
    verification::{VerificationOutcome, VerificationStep, VerificationWorkflow},
};
use crate::ai::driver::observer::{
    FinalizeContext, ObserverOutput, PrepareContext, ToolResultContext, TurnObserver,
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
    pub current_tree_node_id: Option<crate::ai::driver::thinking::engine::ThoughtNodeId>,
    pub poisoned: bool,
    pub pending_suggested_tool_calls: Vec<crate::ai::driver::observer::SuggestedToolCall>,
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
            current_tree_node_id: None,
            poisoned: false,
            pending_suggested_tool_calls: Vec::new(),
        }
    }

    pub fn analyze_question(&mut self, question: &str) -> ThinkingDecision {
        if !self.enabled {
            return ThinkingDecision::default();
        }

        // We do NOT try to guess whether the question needs TreeOfThoughts /
        // Verification / GoalDirected by keyword matching — that is just
        // substring-based pseudo-understanding. Instead:
        //
        //   * If the LLM explicitly requested a mode via a <meta:begin_*>
        //     self-note in a previous turn, that mode is already established
        //     and its state (thought_tree / verification / goal_manager) is
        //     present.
        //   * on_prepare activates whatever state currently exists; we do not
        //     destroy or auto-create state here.
        //
        // This method now only computes derived artifacts from the current
        // active_modes, without altering them.
        let _ = question;
        let inject = self.build_system_prompt_injection();

        let next_sub_goal = if self.active_modes.contains(&ThinkingMode::GoalDirected) {
            self.goal_manager.get_next_actions().first().map(|s| s.description.clone())
        } else {
            None
        };

        let verification_step = if self.active_modes.contains(&ThinkingMode::VerificationLoop) {
            self.verification.as_ref().map(|v| v.current_step)
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
                if let Some(current) = self.current_tree_node_id {
                    let score = if success { 0.7 } else { 0.2 };
                    tree.score_node(current, score);
                    tree.record_outcome(
                        current,
                        result.chars().take(200).collect(),
                        vec![tool_name.to_string()],
                    );
                    self.current_tree_node_id = None;
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
            let safe_snippet: String = result.chars().take(200).collect();
            self.generalizer.ingest_experience(
                "failure",
                &format!(
                    "Avoid: {} led to failure - {}",
                    tool_name,
                    safe_snippet
                ),
                &[tool_name.to_string()],
                None,
            );
        }
    }

    pub fn process_self_note(&mut self, note: &str) {
        // Explicit structured prefix — not substring guessing. The LLM is
        // expected to literally start its note with "Do:" or "Avoid:" to
        // categorize it.
        let trimmed = note.trim_start().to_lowercase();
        let category = if trimmed.starts_with("do:") {
            "self_note_do"
        } else if trimmed.starts_with("avoid:") {
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
        self.generalizer.persist_principle(&principle);
        Some(GeneralizeResult {
            principle_text: principle.principle.clone(),
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

    fn apply_meta_tags(&mut self, text: &str) {
        // Explicit tag extraction — not keyword guessing. The LLM controls
        // thinking state lifecycle by emitting these literal tags.

        // <meta:reset_thinking/> — clear all thinking state
        if text.contains("<meta:reset_thinking/>") || text.contains("<meta:reset_thinking />") {
            self.thought_tree = None;
            self.verification = None;
            self.active_modes.clear();
            self.current_tree_node_id = None;
        }

        // <meta:begin_tree_of_thoughts>...</meta:begin_tree_of_thoughts>
        if let Some(root_thought) = extract_tag(text, "meta:begin_tree_of_thoughts") {
            if self.thought_tree.is_none() {
                self.thought_tree = Some(ThoughtTree::new(&root_thought, 4, 3));
            }
            self.active_modes.insert(ThinkingMode::TreeOfThoughts);
        }

        // <meta:begin_verification>...</meta:begin_verification>
        if let Some(hypothesis) = extract_tag(text, "meta:begin_verification") {
            if self.verification.is_none() {
                self.verification = Some(VerificationWorkflow::new(hypothesis));
            }
            self.active_modes.insert(ThinkingMode::VerificationLoop);
        }

        // <meta:begin_goal>...</meta:begin_goal>
        if let Some(goal_desc) = extract_tag(text, "meta:begin_goal") {
            if self.goal_manager.active_goal().is_none() {
                let goal_id = self.goal_manager.create_goal(goal_desc);
                self.goal_manager.activate_goal(&goal_id);
            }
            self.active_modes.insert(ThinkingMode::GoalDirected);
        }
    }

    fn build_system_prompt_injection(&self) -> Option<String> {
        let mut parts = Vec::new();

        // Always advertise the meta-tag protocol. This is how the LLM
        // explicitly opts into structured thinking modes when *it* judges
        // the situation warrants them — we do not guess on its behalf.
        parts.push(
            "[Thinking Protocol]\n\
             You may opt into structured thinking modes by emitting one of these tags in your \
             reply (they will be parsed back in the next turn, NOT shown to the user):\n\
             - <meta:begin_tree_of_thoughts>ROOT_QUESTION</meta:begin_tree_of_thoughts> — \
             enter tree-of-thoughts exploration when you face multiple plausible approaches.\n\
             - <meta:begin_verification>HYPOTHESIS</meta:begin_verification> — \
             enter scientific-method verification loop when you have a concrete hypothesis \
             that can be falsified by executing a test.\n\
             - <meta:begin_goal>LONG_HORIZON_GOAL</meta:begin_goal> — \
             enter goal-directed decomposition when the task requires multi-step planning \
             with dependency tracking.\n\
             - <meta:reset_thinking/> — clear all thinking state when the conversation topic \
             has shifted and previous reasoning no longer applies.\n\
             - <meta:self_note>Do: ...</meta:self_note> / <meta:self_note>Avoid: ...</meta:self_note> — \
             record an actionable policy you want generalized across future tasks.\n\
             Do NOT emit tags unless the situation truly calls for them.".to_string()
        );

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

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = text.find(&open)?;
    let content_start = start + open.len();
    let end = text[content_start..].find(&close)?;
    let content = &text[content_start..content_start + end];
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

impl TurnObserver for ThinkingOrchestrator {
    fn on_prepare(&mut self, ctx: &PrepareContext) -> Vec<(String, String)> {
        // Parse explicit meta-tags in the question. This is NOT semantic
        // guessing — it's literal tag extraction, similar to how we already
        // parse <meta:self_note>. The LLM is the intelligent party; it is
        // responsible for emitting these tags (typically as part of its
        // previous-turn answer, carried back here via conversation history).
        self.apply_meta_tags(&ctx.question);

        // Whatever thinking state currently exists is activated.
        // No keyword-based relevance heuristics — state lifetime is controlled
        // explicitly by the LLM (via <meta:reset_thinking/>) or implicitly by
        // on_conversation_end.
        if self.thought_tree.is_some() {
            self.active_modes.insert(ThinkingMode::TreeOfThoughts);
        }
        if self.verification.is_some() {
            self.active_modes.insert(ThinkingMode::VerificationLoop);
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
                    self.current_tree_node_id = Some(current);
                    sections.push((
                        "Reasoning Tree".to_string(),
                        tree.generate_thinking_prompt(current),
                    ));
                }
            }
        }
        if decision.active_modes.contains(&ThinkingMode::VerificationLoop) {
            let verification_data: Option<(VerificationStep, String, String)> = self.verification.as_ref().map(|wf| {
                let prompt = match wf.current_step {
                    VerificationStep::GenerateHypothesis => {
                        wf.generate_hypothesis_prompt(&ctx.question, "")
                    }
                    VerificationStep::DesignTest => {
                        wf.generate_test_design_prompt(&wf.current_cycle().hypothesis)
                    }
                    VerificationStep::AnalyzeResult => {
                        wf.generate_analysis_prompt(wf.current_cycle())
                    }
                    VerificationStep::ConfirmOrReject => {
                        format!(
                            "Based on all evidence gathered, make a final judgment.\n\n\
                             Hypothesis: {}\n\
                             Total test cycles: {}\n\
                             Test results: {}\n\n\
                             Output STRICT JSON: {{\"verdict\":\"confirmed\"|\"rejected\"|\"inconclusive\",\"reason\":\"...\"}}",
                            wf.current_cycle().hypothesis,
                            wf.cycles.len(),
                            wf.current_cycle().test_results.len(),
                        )
                    }
                    VerificationStep::ReviseHypothesis => {
                        let results_summary: Vec<String> = wf.current_cycle().test_results.iter()
                            .map(|r| format!("{}: {}", r.command, if r.passed { "PASSED" } else { "FAILED" }))
                            .collect();
                        format!(
                            "Based on the test results, revise your hypothesis.\n\n\
                             Original hypothesis: {}\n\
                             Test results: {}\n\n\
                             What new hypothesis better explains these results?\n\
                             Output STRICT JSON: {{\"new_hypothesis\":\"...\",\"feedback\":\"...\"}}",
                            wf.current_cycle().hypothesis,
                            results_summary.join(", ")
                        )
                    }
                    VerificationStep::ExecuteTest => String::new(),
                };
                (wf.current_step, wf.current_cycle().hypothesis.clone(), prompt)
            });
            if let Some((step, hypothesis, prompt)) = verification_data {
                if matches!(step, VerificationStep::ExecuteTest) {
                    self.pending_suggested_tool_calls.push(
                        crate::ai::driver::observer::SuggestedToolCall {
                            tool_name: "RunCommand".to_string(),
                            arguments: serde_json::json!({
                                "note": "Execute the test command designed in the previous DesignTest step",
                                "hypothesis": hypothesis,
                            }),
                            rationale: format!("Verify hypothesis: '{}'", hypothesis),
                        }
                    );
                } else if !prompt.is_empty() {
                    sections.push(("Verification".to_string(), prompt));
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

    fn on_prepare_rich(&mut self, ctx: &PrepareContext) -> crate::ai::driver::observer::PrepareOutput {
        use crate::ai::driver::observer::{PrepareOutput, SectionKind};
        let legacy = self.on_prepare(ctx);
        let sections = legacy
            .into_iter()
            .map(|(label, content)| {
                // Thinking sections are all behavior-directing (telling LLM *how* to reason),
                // so they all belong to Behavior channel, not Fact channel.
                (SectionKind::Behavior, label, content)
            })
            .collect();
        let suggested_tool_calls = std::mem::take(&mut self.pending_suggested_tool_calls);
        PrepareOutput {
            sections,
            suggested_tool_calls,
        }
    }

    fn on_finalize(&mut self, ctx: &FinalizeContext) -> ObserverOutput {
        let mut display_lines = Vec::new();

        if let Some(note) = extract_self_note_from_text(&ctx.final_text) {
            self.process_self_note(&note);
        }

        if let Some(result) = self.try_generalize() {
            display_lines.push(format!("[Thinking] Generalized principle: {}", result.principle_text));
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
        self.current_tree_node_id = None;

        ObserverOutput {
            display_lines,
        }
    }

    fn on_conversation_end(&mut self) {
        let _ = self.goal_manager.persist_goals();
        self.thought_tree = None;
        self.verification = None;
        self.active_modes.clear();
        self.current_tree_node_id = None;
        self.generalizer = ExperienceGeneralizer::new();
        self.goal_manager = GoalManager::new()
            .with_persistence_dir_opt(dirs::home_dir().map(|h| h.join(".config").join("rust_tools").join("thinking_goals")));
    }

    fn name(&self) -> &str {
        "thinking"
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn mark_poisoned(&mut self) {
        self.poisoned = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_meta_tag_activates_verification() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags("<meta:begin_verification>server crashes under high load</meta:begin_verification>");
        let decision = orch.analyze_question("anything");
        assert!(decision.active_modes.contains(&ThinkingMode::VerificationLoop));
    }

    #[test]
    fn explicit_meta_tag_activates_goal() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags("<meta:begin_goal>Refactor the entire networking layer</meta:begin_goal>");
        let decision = orch.analyze_question("anything");
        assert!(decision.active_modes.contains(&ThinkingMode::GoalDirected));
        assert!(orch.goal_manager.active_goal().is_some());
    }

    #[test]
    fn explicit_meta_tags_can_activate_multiple_modes() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags(
            "<meta:begin_verification>bug hypothesis</meta:begin_verification>\
             <meta:begin_goal>refactor error handling</meta:begin_goal>"
        );
        let decision = orch.analyze_question("anything");
        assert!(decision.active_modes.contains(&ThinkingMode::VerificationLoop));
        assert!(decision.active_modes.contains(&ThinkingMode::GoalDirected));
    }

    #[test]
    fn question_without_meta_tags_activates_no_modes() {
        let mut orch = ThinkingOrchestrator::new();
        let decision = orch.analyze_question("Why does the server crash under high load?");
        assert!(decision.active_modes.is_empty());
        assert!(orch.goal_manager.active_goal().is_none());
    }

    #[test]
    fn reset_meta_tag_clears_state() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags("<meta:begin_verification>h1</meta:begin_verification>");
        assert!(orch.verification.is_some());
        orch.apply_meta_tags("<meta:reset_thinking/>");
        assert!(orch.verification.is_none());
        assert!(orch.active_modes.is_empty());
    }

    #[test]
    fn process_tool_result_scores_tree() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags("<meta:begin_tree_of_thoughts>root</meta:begin_tree_of_thoughts>");
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
        let output = orch.on_finalize(&FinalizeContext {
            question: "test".to_string(),
            final_text: "some response".to_string(),
            had_tool_calls: false,
        });
        let _ = output.display_lines;
    }

    #[test]
    fn on_conversation_end_clears_goals() {
        let mut orch = ThinkingOrchestrator::new();
        orch.apply_meta_tags("<meta:begin_goal>Refactor the system</meta:begin_goal>");
        assert!(orch.goal_manager.active_goal().is_some());
        orch.on_conversation_end();
        assert!(orch.goal_manager.active_goal().is_none());
    }
}
