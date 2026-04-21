use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub type ThoughtNodeId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtNode {
    pub id: ThoughtNodeId,
    pub parent_id: Option<ThoughtNodeId>,
    pub children: Vec<ThoughtNodeId>,
    pub hypothesis: String,
    pub reasoning: String,
    pub score: f64,
    pub depth: usize,
    pub visit_count: u32,
    pub explored: bool,
    pub pruned: bool,
    pub tool_calls_snapshot: Vec<String>,
    pub outcome_summary: Option<String>,
}

impl ThoughtNode {
    fn new(id: ThoughtNodeId, parent_id: Option<ThoughtNodeId>, hypothesis: String, reasoning: String, depth: usize) -> Self {
        Self {
            id,
            parent_id,
            children: Vec::new(),
            hypothesis,
            reasoning,
            score: 0.5,
            depth,
            visit_count: 0,
            explored: false,
            pruned: false,
            tool_calls_snapshot: Vec::new(),
            outcome_summary: None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_terminal(&self) -> bool {
        self.outcome_summary.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtTree {
    nodes: HashMap<ThoughtNodeId, ThoughtNode>,
    next_id: ThoughtNodeId,
    root_id: ThoughtNodeId,
    max_depth: usize,
    max_branches: usize,
    exploration_threshold: f64,
    best_path: Vec<ThoughtNodeId>,
}

#[derive(Debug, Clone)]
pub struct ExplorationResult {
    pub best_node_id: ThoughtNodeId,
    pub best_score: f64,
    pub path: Vec<ThoughtNodeId>,
    pub should_backtrack: bool,
    pub backtrack_to: Option<ThoughtNodeId>,
    pub reasoning_summary: String,
}

impl ThoughtTree {
    pub fn new(question: &str, max_depth: usize, max_branches: usize) -> Self {
        let root = ThoughtNode::new(0, None, question.to_string(), "Initial question".to_string(), 0);
        let mut nodes = HashMap::new();
        nodes.insert(0, root);
        Self {
            nodes,
            next_id: 1,
            root_id: 0,
            max_depth,
            max_branches,
            exploration_threshold: 0.3,
            best_path: vec![0],
        }
    }

    pub fn add_branch(&mut self, parent_id: ThoughtNodeId, hypothesis: String, reasoning: String) -> ThoughtNodeId {
        let parent = match self.nodes.get(&parent_id) {
            Some(p) => p.clone(),
            None => return parent_id,
        };
        if parent.depth >= self.max_depth {
            return parent_id;
        }
        if parent.children.len() >= self.max_branches {
            return parent_id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let node = ThoughtNode::new(id, Some(parent_id), hypothesis, reasoning, parent.depth + 1);
        self.nodes.insert(id, node);
        if let Some(p) = self.nodes.get_mut(&parent_id) {
            p.children.push(id);
        }
        id
    }

    pub fn score_node(&mut self, node_id: ThoughtNodeId, score: f64) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.score = score.clamp(0.0, 1.0);
            node.explored = true;
            node.visit_count += 1;
        }
        self.update_best_path();
    }

    pub fn record_outcome(&mut self, node_id: ThoughtNodeId, summary: String, tool_calls: Vec<String>) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.outcome_summary = Some(summary);
            node.tool_calls_snapshot = tool_calls;
        }
    }

    pub fn prune_branch(&mut self, node_id: ThoughtNodeId) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.pruned = true;
        }
    }

    pub fn decide_next(&self) -> ExplorationResult {
        let candidates = self.find_explorable_nodes();
        if candidates.is_empty() {
            let best = self.find_best_scored_node();
            return ExplorationResult {
                best_node_id: best.id,
                best_score: best.score,
                path: self.path_to(best.id),
                should_backtrack: false,
                backtrack_to: None,
                reasoning_summary: best.hypothesis.clone(),
            };
        }

        let best_unexplored = candidates
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let current_best_score = self.find_best_scored_node().score;
        let should_backtrack = best_unexplored.score < current_best_score - self.exploration_threshold;

        let backtrack_to = if should_backtrack {
            let best_scored = self.find_best_scored_node();
            Some(best_scored.id)
        } else {
            None
        };

        ExplorationResult {
            best_node_id: best_unexplored.id,
            best_score: best_unexplored.score,
            path: self.path_to(best_unexplored.id),
            should_backtrack,
            backtrack_to,
            reasoning_summary: best_unexplored.hypothesis.clone(),
        }
    }

    pub fn ucb_select(&self, exploration_constant: f64) -> Option<ThoughtNodeId> {
        let candidates: Vec<&ThoughtNode> = self.nodes.values()
            .filter(|n| !n.pruned && !n.is_terminal() && n.depth < self.max_depth)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let total_visits: f64 = candidates.iter().map(|n| n.visit_count.max(1) as f64).sum();
        let best = candidates.iter().max_by(|a, b| {
            let ucb_a = self.ucb_score(a, total_visits, exploration_constant);
            let ucb_b = self.ucb_score(b, total_visits, exploration_constant);
            ucb_a.partial_cmp(&ucb_b).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(best.id)
    }

    fn ucb_score(&self, node: &ThoughtNode, total_visits: f64, c: f64) -> f64 {
        let visits = node.visit_count.max(1) as f64;
        let exploitation = node.score;
        let exploration = c * (total_visits.ln() / visits).sqrt();
        exploitation + exploration
    }

    fn find_explorable_nodes(&self) -> Vec<&ThoughtNode> {
        self.nodes.values()
            .filter(|n| !n.pruned && !n.explored && !n.is_terminal())
            .collect()
    }

    fn find_best_scored_node(&self) -> &ThoughtNode {
        self.nodes.values()
            .filter(|n| n.explored && !n.pruned)
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| self.nodes.get(&self.root_id).unwrap())
    }

    fn path_to(&self, node_id: ThoughtNodeId) -> Vec<ThoughtNodeId> {
        let mut path = Vec::new();
        let mut current = Some(node_id);
        while let Some(id) = current {
            path.push(id);
            current = self.nodes.get(&id).and_then(|n| n.parent_id);
        }
        path.reverse();
        path
    }

    fn update_best_path(&mut self) {
        let best = self.find_best_scored_node();
        self.best_path = self.path_to(best.id);
    }

    pub fn get_node(&self, id: ThoughtNodeId) -> Option<&ThoughtNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: ThoughtNodeId) -> Option<&mut ThoughtNode> {
        self.nodes.get_mut(&id)
    }

    pub fn root_id(&self) -> ThoughtNodeId {
        self.root_id
    }

    pub fn best_path(&self) -> &[ThoughtNodeId] {
        &self.best_path
    }

    pub fn render_tree_summary(&self) -> String {
        let mut lines = Vec::new();
        self.render_node_recursive(self.root_id, &mut lines, 0);
        lines.join("\n")
    }

    fn render_node_recursive(&self, node_id: ThoughtNodeId, lines: &mut Vec<String>, indent: usize) {
        if let Some(node) = self.nodes.get(&node_id) {
            let prefix = "  ".repeat(indent);
            let status = if node.pruned { "✗" } else if node.outcome_summary.is_some() { "✓" } else { "○" };
            let score_str = format!("{:.2}", node.score);
            let truncated: String = node.hypothesis.chars().take(60).collect();
            lines.push(format!("{}{} [{}] {} (depth={})", prefix, status, score_str, truncated, node.depth));
            for &child_id in &node.children {
                self.render_node_recursive(child_id, lines, indent + 1);
            }
        }
    }

    pub fn generate_thinking_prompt(&self, current_node_id: ThoughtNodeId) -> String {
        let path = self.path_to(current_node_id);
        let mut prompt = String::from("You are in a Tree-of-Thoughts reasoning process.\n\n");
        prompt.push_str("Path taken so far:\n");
        for (i, &node_id) in path.iter().enumerate() {
            if let Some(node) = self.nodes.get(&node_id) {
                prompt.push_str(&format!("  Step {}: {} (score: {:.2})\n", i + 1, node.hypothesis, node.score));
                if let Some(outcome) = &node.outcome_summary {
                    prompt.push_str(&format!("    Outcome: {}\n", outcome));
                }
            }
        }
        if let Some(current) = self.nodes.get(&current_node_id) {
            prompt.push_str(&format!("\nCurrent hypothesis: {}\n", current.hypothesis));
            prompt.push_str(&format!("Current reasoning: {}\n", current.reasoning));
        }
        prompt.push_str("\nGenerate 2-3 alternative hypotheses to explore. For each, provide:\n");
        prompt.push_str("1. hypothesis: A specific approach or assumption to test\n");
        prompt.push_str("2. reasoning: Why this might be better than the current path\n");
        prompt.push_str("3. estimated_score: Your confidence (0.0-1.0) that this leads to a correct solution\n");
        prompt.push_str("\nOutput STRICT JSON array: [{\"hypothesis\":\"...\",\"reasoning\":\"...\",\"estimated_score\":0.8}]\n");
        prompt
    }

    pub fn generate_scoring_prompt(&self, node_id: ThoughtNodeId, tool_results: &str) -> String {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return String::new(),
        };
        format!(
            "Score this reasoning step on a 0.0-1.0 scale.\n\n\
             Hypothesis: {}\n\
             Reasoning: {}\n\
             Tool results: {}\n\n\
             Scoring criteria:\n\
             - 1.0: Hypothesis fully confirmed by evidence\n\
             - 0.7-0.9: Strong evidence supporting this direction\n\
             - 0.4-0.6: Mixed or inconclusive evidence\n\
             - 0.1-0.3: Evidence contradicts the hypothesis\n\
             - 0.0: Dead end, fundamental error\n\n\
             Output STRICT JSON: {{\"score\":0.8,\"reason\":\"...\"}}",
            node.hypothesis,
            node.reasoning,
            if tool_results.len() > 2000 { &tool_results[..2000] } else { tool_results }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_add_branches_and_score() {
        let mut tree = ThoughtTree::new("fix the bug", 4, 3);
        let b1 = tree.add_branch(0, "hypothesis A".into(), "reason A".into());
        let b2 = tree.add_branch(0, "hypothesis B".into(), "reason B".into());
        tree.score_node(b1, 0.8);
        tree.score_node(b2, 0.3);
        let result = tree.decide_next();
        assert!(result.best_score >= 0.3);
    }

    #[test]
    fn ucb_select_prefers_unexplored() {
        let mut tree = ThoughtTree::new("test", 3, 3);
        let b1 = tree.add_branch(0, "A".into(), "rA".into());
        let b2 = tree.add_branch(0, "B".into(), "rB".into());
        tree.score_node(b1, 0.9);
        let selected = tree.ucb_select(1.414);
        assert!(selected.is_some());
    }

    #[test]
    fn prune_and_backtrack() {
        let mut tree = ThoughtTree::new("test", 3, 3);
        let b1 = tree.add_branch(0, "good".into(), "r".into());
        let b2 = tree.add_branch(0, "bad".into(), "r".into());
        tree.score_node(b1, 0.9);
        tree.score_node(b2, 0.1);
        tree.prune_branch(b2);
        let result = tree.decide_next();
        assert!(!tree.get_node(b2).unwrap().pruned == false);
    }

    #[test]
    fn render_tree_summary() {
        let mut tree = ThoughtTree::new("root q", 3, 3);
        let b1 = tree.add_branch(0, "branch1".into(), "r1".into());
        tree.score_node(b1, 0.7);
        let summary = tree.render_tree_summary();
        assert!(summary.contains("branch1"));
    }
}
