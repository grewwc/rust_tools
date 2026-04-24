use std::collections::HashMap;
use std::path::PathBuf;

use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

pub type GoalId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalState {
    Proposed,
    Active,
    InProgress,
    Blocked { reason: String },
    Completed,
    Abandoned { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGoal {
    pub id: GoalId,
    pub description: String,
    pub state: GoalState,
    pub depends_on: Vec<GoalId>,
    pub parent_id: Option<GoalId>,
    pub priority: u8,
    pub progress: f64,
    pub context_snapshot: Option<String>,
    pub result: Option<String>,
}

impl SubGoal {
    pub fn new(id: GoalId, description: String, parent_id: Option<GoalId>, priority: u8) -> Self {
        Self {
            id,
            description,
            state: GoalState::Proposed,
            depends_on: Vec::new(),
            parent_id,
            priority,
            progress: 0.0,
            context_snapshot: None,
            result: None,
        }
    }

    pub fn is_ready(&self, completed_ids: &std::collections::HashSet<&GoalId>) -> bool {
        self.depends_on.iter().all(|dep| completed_ids.contains(dep))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub state: GoalState,
    pub sub_goals: Vec<SubGoal>,
    pub created_at: String,
    pub updated_at: String,
    pub max_depth: usize,
    pub overall_progress: f64,
    pub context: String,
    pub strategy: Option<String>,
    pub persistence_path: Option<PathBuf>,
}

impl Goal {
    pub fn new(description: String) -> Self {
        let now = chrono::Local::now().to_rfc3339();
        Self {
            id: format!("goal_{}", uuid::Uuid::new_v4().simple()),
            description,
            state: GoalState::Proposed,
            sub_goals: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
            max_depth: 5,
            overall_progress: 0.0,
            context: String::new(),
            strategy: None,
            persistence_path: None,
        }
    }

    pub fn add_sub_goal(&mut self, description: String, depends_on_indices: Vec<usize>, priority: u8) -> GoalId {
        let sub_id = format!("{}_{}", self.id, self.sub_goals.len());
        let depends_on: Vec<GoalId> = depends_on_indices.iter()
            .filter_map(|&idx| self.sub_goals.get(idx).map(|s| s.id.clone()))
            .collect();
        let mut sub = SubGoal::new(sub_id.clone(), description, Some(self.id.clone()), priority);
        sub.depends_on = depends_on;
        self.sub_goals.push(sub);
        sub_id
    }

    pub fn update_sub_goal_state(&mut self, sub_id: &GoalId, new_state: GoalState) {
        if let Some(sub) = self.sub_goals.iter_mut().find(|s| &s.id == sub_id) {
            sub.state = new_state;
        }
        self.recalculate_progress();
        self.updated_at = chrono::Local::now().to_rfc3339();
    }

    pub fn update_sub_goal_progress(&mut self, sub_id: &GoalId, progress: f64, context: Option<String>) {
        if let Some(sub) = self.sub_goals.iter_mut().find(|s| &s.id == sub_id) {
            sub.progress = progress.clamp(0.0, 1.0);
            if let Some(ctx) = context {
                sub.context_snapshot = Some(ctx);
            }
        }
        self.recalculate_progress();
    }

    pub fn complete_sub_goal(&mut self, sub_id: &GoalId, result: String) {
        if let Some(sub) = self.sub_goals.iter_mut().find(|s| &s.id == sub_id) {
            sub.state = GoalState::Completed;
            sub.progress = 1.0;
            sub.result = Some(result);
        }
        self.recalculate_progress();
        if self.sub_goals.iter().all(|s| s.state == GoalState::Completed) {
            self.state = GoalState::Completed;
        }
        self.updated_at = chrono::Local::now().to_rfc3339();
    }

    pub fn get_next_actionable(&self) -> Vec<&SubGoal> {
        let completed: std::collections::HashSet<&GoalId> = self.sub_goals.iter()
            .filter(|s| s.state == GoalState::Completed)
            .map(|s| &s.id)
            .collect();
        self.sub_goals.iter()
            .filter(|s| matches!(s.state, GoalState::Proposed | GoalState::InProgress))
            .filter(|s| s.is_ready(&completed))
            .collect()
    }

    pub fn get_blocked(&self) -> Vec<&SubGoal> {
        self.sub_goals.iter()
            .filter(|s| matches!(s.state, GoalState::Blocked { .. }))
            .collect()
    }

    pub fn generate_decomposition_prompt(&self) -> String {
        format!(
            "You are a goal decomposition engine. Break down this goal into actionable sub-goals.\n\n\
             Goal: {}\n\
             Context: {}\n\
             Current strategy: {}\n\n\
             Rules:\n\
             - Each sub-goal should be independently achievable\n\
             - Specify dependencies using 0-based indices of previously listed sub-goals\n\
             - Assign priority (1-10, 10 = highest)\n\
             - Sub-goals should be concrete and testable\n\
             - Maximum {} levels of decomposition\n\n\
             Output STRICT JSON array: [{{\"description\":\"...\",\"depends_on_indices\":[0],\"priority\":8}}]\n\
             Use empty array for depends_on_indices if no dependencies. \
             depends_on_indices uses 0-based index of previously listed sub-goals.",
            self.description,
            if self.context.len() > 1000 { &self.context[..1000] } else { &self.context },
            self.strategy.as_deref().unwrap_or("none"),
            self.max_depth
        )
    }

    pub fn generate_strategy_prompt(&self) -> String {
        let sub_goals_summary: Vec<String> = self.sub_goals.iter()
            .map(|s| format!("- [{}] {} (progress: {:.0}%)", 
                match &s.state {
                    GoalState::Completed => "✓",
                    GoalState::InProgress => "→",
                    GoalState::Blocked { .. } => "✗",
                    _ => "○",
                },
                s.description,
                s.progress * 100.0
            ))
            .collect();
        format!(
            "You are a strategic planner. Given the current goal state, propose a strategy.\n\n\
             Goal: {}\n\
             Sub-goals:\n{}\n\
             Overall progress: {:.0}%\n\n\
             Rules:\n\
             - Identify the critical path\n\
             - Suggest which blocked goals to unblock first\n\
             - Propose any new sub-goals if needed\n\
             - Identify risks and mitigation strategies\n\n\
             Output STRICT JSON: {{\"strategy\":\"...\",\"critical_path\":[\"id1\",\"id2\"],\"risks\":[\"...\"],\"new_sub_goals\":[{{\"description\":\"...\",\"priority\":8}}]}}",
            self.description,
            sub_goals_summary.join("\n"),
            self.overall_progress * 100.0
        )
    }

    fn recalculate_progress(&mut self) {
        if self.sub_goals.is_empty() {
            self.overall_progress = 0.0;
            return;
        }
        let total: f64 = self.sub_goals.iter().map(|s| s.progress).sum();
        self.overall_progress = total / self.sub_goals.len() as f64;
        if self.overall_progress > 0.0 && self.state == GoalState::Proposed {
            self.state = GoalState::InProgress;
        }
    }

    pub fn render_status(&self) -> String {
        let mut status = format!("Goal: {}\n", self.description);
        status.push_str(&format!("State: {:?} | Progress: {:.0}%\n", self.state, self.overall_progress * 100.0));
        status.push_str("Sub-goals:\n");
        for sub in &self.sub_goals {
            let icon = match &sub.state {
                GoalState::Completed => "✓",
                GoalState::InProgress => "→",
                GoalState::Blocked { reason } => &format!("✗({})", reason.chars().take(20).collect::<String>()),
                GoalState::Proposed => "○",
                GoalState::Active => "●",
                GoalState::Abandoned { reason } => &format!("⊘({})", reason.chars().take(20).collect::<String>()),
            };
            let deps = if sub.depends_on.is_empty() {
                String::new()
            } else {
                format!(" [after: {}]", sub.depends_on.join(","))
            };
            status.push_str(&format!("  {} {} ({}%){}\n", icon, sub.description, (sub.progress * 100.0) as u8, deps));
        }
        status
    }
}

pub struct GoalManager {
    goals: FastMap<GoalId, Goal>,
    active_goal_id: Option<GoalId>,
    persistence_dir: Option<PathBuf>,
}

impl GoalManager {
    pub fn new() -> Self {
        Self {
            goals: FastMap::default(),
            active_goal_id: None,
            persistence_dir: None,
        }
    }

    pub fn with_persistence_dir(mut self, dir: PathBuf) -> Self {
        self.persistence_dir = Some(dir);
        let _ = self.load_goals();
        self
    }

    pub fn with_persistence_dir_opt(mut self, dir: Option<PathBuf>) -> Self {
        self.persistence_dir = dir;
        let _ = self.load_goals();
        self
    }

    pub fn create_goal(&mut self, description: String) -> GoalId {
        let goal = Goal::new(description);
        let id = goal.id.clone();
        self.goals.insert(id.clone(), goal);
        if self.active_goal_id.is_none() {
            self.active_goal_id = Some(id.clone());
        }
        id
    }

    pub fn activate_goal(&mut self, goal_id: &GoalId) {
        if self.goals.contains_key(goal_id) {
            self.active_goal_id = Some(goal_id.clone());
            if let Some(goal) = self.goals.get_mut(goal_id) {
                if goal.state == GoalState::Proposed {
                    goal.state = GoalState::Active;
                }
            }
        }
    }

    pub fn active_goal(&self) -> Option<&Goal> {
        self.active_goal_id.as_ref().and_then(|id| self.goals.get(id))
    }

    pub fn deactivate_active_goal(&mut self) {
        self.active_goal_id = None;
    }

    pub fn active_goal_mut(&mut self) -> Option<&mut Goal> {
        self.active_goal_id.as_ref().and_then(|id| self.goals.get_mut(id))
    }

    pub fn decompose_active(&mut self, sub_goals: Vec<(String, Vec<usize>, u8)>) {
        if let Some(goal) = self.active_goal_mut() {
            for (desc, deps, priority) in sub_goals {
                goal.add_sub_goal(desc, deps, priority);
            }
        }
    }

    pub fn get_next_actions(&self) -> Vec<&SubGoal> {
        self.active_goal()
            .map(|g| g.get_next_actionable())
            .unwrap_or_default()
    }

    pub fn persist_goals(&self) -> Result<(), String> {
        let dir = match &self.persistence_dir {
            Some(d) => d.clone(),
            None => return Ok(()),
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}", e))?;
        let path = dir.join("goals.json");
        let data = serde_json::to_string_pretty(&self.goals).map_err(|e| format!("{}", e))?;
        std::fs::write(&path, data).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load_goals(&mut self) -> Result<(), String> {
        let dir = match &self.persistence_dir {
            Some(d) => d.clone(),
            None => return Ok(()),
        };
        let path = dir.join("goals.json");
        if !path.exists() {
            return Ok(());
        }
        let data = std::fs::read_to_string(&path).map_err(|e| format!("{}", e))?;
        self.goals = serde_json::from_str(&data).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn render_active_status(&self) -> String {
        match self.active_goal() {
            Some(goal) => goal.render_status(),
            None => "No active goal.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_decompose_goal() {
        let mut mgr = GoalManager::new();
        let id = mgr.create_goal("refactor the network layer".to_string());
        mgr.activate_goal(&id);
        mgr.decompose_active(vec![
            ("define new API surface".to_string(), vec![], 10),
            ("implement new client".to_string(), vec![0], 8),
            ("migrate existing callers".to_string(), vec![1], 6),
        ]);
        let actions = mgr.get_next_actions();
        assert!(!actions.is_empty());
    }

    #[test]
    fn complete_sub_goal_updates_progress() {
        let mut mgr = GoalManager::new();
        let id = mgr.create_goal("test goal".to_string());
        mgr.activate_goal(&id);
        mgr.decompose_active(vec![
            ("step 1".to_string(), vec![], 5),
            ("step 2".to_string(), vec![], 5),
        ]);
        let goal = mgr.active_goal_mut().unwrap();
        let sub_id = goal.sub_goals[0].id.clone();
        goal.complete_sub_goal(&sub_id, "done".to_string());
        assert_eq!(goal.overall_progress, 0.5);
    }

    #[test]
    fn dependency_blocks_execution() {
        let mut mgr = GoalManager::new();
        let id = mgr.create_goal("goal".to_string());
        mgr.activate_goal(&id);
        mgr.decompose_active(vec![
            ("first".to_string(), vec![], 10),
            ("second".to_string(), vec![0], 5),
        ]);
        let goal = mgr.active_goal().unwrap();
        let ready = goal.get_next_actionable();
        let ready_descs: Vec<&str> = ready.iter().map(|s| s.description.as_str()).collect();
        assert!(ready_descs.contains(&"first"));
        assert!(!ready_descs.contains(&"second"));
    }
}
