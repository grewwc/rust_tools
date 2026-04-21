use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerificationStep {
    GenerateHypothesis,
    DesignTest,
    ExecuteTest,
    AnalyzeResult,
    ReviseHypothesis,
    ConfirmOrReject,
}

impl VerificationStep {
    pub fn next(&self) -> Option<VerificationStep> {
        match self {
            VerificationStep::GenerateHypothesis => Some(VerificationStep::DesignTest),
            VerificationStep::DesignTest => Some(VerificationStep::ExecuteTest),
            VerificationStep::ExecuteTest => Some(VerificationStep::AnalyzeResult),
            VerificationStep::AnalyzeResult => Some(VerificationStep::ReviseHypothesis),
            VerificationStep::ReviseHypothesis => Some(VerificationStep::ConfirmOrReject),
            VerificationStep::ConfirmOrReject => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerificationOutcome {
    Confirmed,
    Rejected { reason: String },
    NeedsRevision { feedback: String },
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCycle {
    pub hypothesis: String,
    pub test_design: String,
    pub test_commands: Vec<String>,
    pub test_results: Vec<TestResult>,
    pub analysis: Option<String>,
    pub outcome: Option<VerificationOutcome>,
    pub revision_count: usize,
    pub max_revisions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub command: String,
    pub exit_code: i32,
    pub stdout_preview: String,
    pub stderr_preview: String,
    pub passed: bool,
}

impl VerificationCycle {
    pub fn new(hypothesis: String, max_revisions: usize) -> Self {
        Self {
            hypothesis,
            test_design: String::new(),
            test_commands: Vec::new(),
            test_results: Vec::new(),
            analysis: None,
            outcome: None,
            revision_count: 0,
            max_revisions,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn record_test_result(&mut self, result: TestResult) {
        self.test_results.push(result);
    }

    pub fn revise(&mut self, new_hypothesis: String, feedback: String) -> Result<(), String> {
        if self.revision_count >= self.max_revisions {
            self.outcome = Some(VerificationOutcome::Rejected {
                reason: format!("Max revisions ({}) reached. Last feedback: {}", self.max_revisions, feedback),
            });
            return Err("max revisions reached".to_string());
        }
        self.revision_count += 1;
        self.hypothesis = new_hypothesis;
        self.test_design.clear();
        self.test_commands.clear();
        self.test_results.clear();
        self.analysis = None;
        self.outcome = None;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationWorkflow {
    pub cycles: Vec<VerificationCycle>,
    pub current_cycle_idx: usize,
    pub current_step: VerificationStep,
    pub overall_outcome: Option<VerificationOutcome>,
    pub max_cycles: usize,
}

impl VerificationWorkflow {
    pub fn new(initial_hypothesis: String) -> Self {
        Self {
            cycles: vec![VerificationCycle::new(initial_hypothesis, 3)],
            current_cycle_idx: 0,
            current_step: VerificationStep::GenerateHypothesis,
            overall_outcome: None,
            max_cycles: 5,
        }
    }

    pub fn current_cycle(&self) -> &VerificationCycle {
        &self.cycles[self.current_cycle_idx]
    }

    pub fn current_cycle_mut(&mut self) -> &mut VerificationCycle {
        &mut self.cycles[self.current_cycle_idx]
    }

    pub fn advance_step(&mut self) {
        if let Some(next) = self.current_step.next() {
            self.current_step = next;
        }
    }

    pub fn complete_cycle(&mut self, outcome: VerificationOutcome) {
        self.cycles[self.current_cycle_idx].outcome = Some(outcome.clone());
        match &outcome {
            VerificationOutcome::Confirmed => {
                self.overall_outcome = Some(VerificationOutcome::Confirmed);
            }
            VerificationOutcome::Rejected { .. } => {
                self.overall_outcome = Some(outcome);
            }
            VerificationOutcome::NeedsRevision { feedback } => {
                if self.current_cycle_idx + 1 < self.max_cycles {
                    let old_hypothesis = self.cycles[self.current_cycle_idx].hypothesis.clone();
                    let revised_hypothesis = format!("{} (revised: {})", old_hypothesis, feedback);
                    let mut new_cycle = VerificationCycle::new(revised_hypothesis, 3);
                    new_cycle.revision_count = self.cycles[self.current_cycle_idx].revision_count + 1;
                    self.cycles.push(new_cycle);
                    self.current_cycle_idx += 1;
                    self.current_step = VerificationStep::GenerateHypothesis;
                } else {
                    self.overall_outcome = Some(VerificationOutcome::Inconclusive);
                }
            }
            VerificationOutcome::Inconclusive => {
                self.overall_outcome = Some(VerificationOutcome::Inconclusive);
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.overall_outcome.is_some()
    }

    pub fn generate_hypothesis_prompt(&self, question: &str, context: &str) -> String {
        format!(
            "Given this question and context, generate a testable hypothesis.\n\n\
             Question: {}\n\
             Context: {}\n\n\
             A good hypothesis should be:\n\
             1. Specific and falsifiable\n\
             2. Testable with available tools (file read, command execution, code search)\n\
             3. Focused on a single claim\n\n\
             Output STRICT JSON: {{\"hypothesis\":\"...\",\"confidence\":0.8}}",
            question,
            if context.len() > 2000 { &context[..2000] } else { context }
        )
    }

    pub fn generate_test_design_prompt(&self, hypothesis: &str) -> String {
        format!(
            "Design verification tests for this hypothesis.\n\n\
             Hypothesis: {}\n\n\
             Generate 1-3 specific test commands or file inspections that would confirm or refute this hypothesis.\n\
             Each test should have a clear pass/fail criteria.\n\n\
             Output STRICT JSON array: [{{\"description\":\"...\",\"command\":\"...\",\"pass_criteria\":\"...\"}}]",
            hypothesis
        )
    }

    pub fn generate_analysis_prompt(&self, cycle: &VerificationCycle) -> String {
        let mut results_str = String::new();
        for result in &cycle.test_results {
            results_str.push_str(&format!(
                "- Command: {} | Exit: {} | Passed: {}\n  stdout: {}\n  stderr: {}\n",
                result.command,
                result.exit_code,
                result.passed,
                if result.stdout_preview.len() > 300 { &result.stdout_preview[..300] } else { &result.stdout_preview },
                if result.stderr_preview.len() > 300 { &result.stderr_preview[..300] } else { &result.stderr_preview },
            ));
        }
        format!(
            "Analyze these test results against the hypothesis.\n\n\
             Hypothesis: {}\n\
             Test results:\n{}\n\n\
             Determine:\n\
             1. Is the hypothesis confirmed, rejected, or needs revision?\n\
             2. If needs revision, what should the new hypothesis be?\n\
             3. What specific feedback led to this conclusion?\n\n\
             Output STRICT JSON: {{\"verdict\":\"confirmed\"|\"rejected\"|\"needs_revision\"|\"inconclusive\",\"reason\":\"...\",\"new_hypothesis\":\"...\",\"feedback\":\"...\"}}",
            cycle.hypothesis,
            results_str
        )
    }

    pub fn render_summary(&self) -> String {
        let mut summary = String::from("Verification Workflow Summary:\n");
        for (i, cycle) in self.cycles.iter().enumerate() {
            summary.push_str(&format!(
                "\nCycle {} (revision {}):\n  Hypothesis: {}\n  Tests: {}\n  Outcome: {}\n",
                i + 1,
                cycle.revision_count,
                cycle.hypothesis,
                cycle.test_results.len(),
                match &cycle.outcome {
                    Some(VerificationOutcome::Confirmed) => "✓ Confirmed".to_string(),
                    Some(VerificationOutcome::Rejected { reason }) => format!("✗ Rejected: {}", reason),
                    Some(VerificationOutcome::NeedsRevision { feedback }) => format!("↻ Needs revision: {}", feedback),
                    Some(VerificationOutcome::Inconclusive) => "? Inconclusive".to_string(),
                    None => "... In progress".to_string(),
                }
            ));
        }
        if let Some(ref outcome) = self.overall_outcome {
            summary.push_str(&format!("\nOverall: {:?}", outcome));
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_advances_steps() {
        let mut wf = VerificationWorkflow::new("the bug is in parser".to_string());
        assert_eq!(wf.current_step, VerificationStep::GenerateHypothesis);
        wf.advance_step();
        assert_eq!(wf.current_step, VerificationStep::DesignTest);
    }

    #[test]
    fn cycle_revision() {
        let mut cycle = VerificationCycle::new("h1".to_string(), 3);
        cycle.revise("h2".to_string(), "evidence against h1".to_string()).unwrap();
        assert_eq!(cycle.revision_count, 1);
        assert_eq!(cycle.hypothesis, "h2");
    }

    #[test]
    fn workflow_complete_on_confirm() {
        let mut wf = VerificationWorkflow::new("h".to_string());
        wf.complete_cycle(VerificationOutcome::Confirmed);
        assert!(wf.is_complete());
        assert_eq!(wf.overall_outcome, Some(VerificationOutcome::Confirmed));
    }

    #[test]
    fn workflow_revises_and_creates_new_cycle() {
        let mut wf = VerificationWorkflow::new("h1".to_string());
        wf.complete_cycle(VerificationOutcome::NeedsRevision {
            feedback: "try again".to_string(),
        });
        assert_eq!(wf.cycles.len(), 2);
        assert_eq!(wf.current_cycle_idx, 1);
    }
}
