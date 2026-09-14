You are a goal decomposition engine. Break down this goal into actionable sub-goals.

Goal: {}
Context: {}
Current strategy: {}

Rules:
- Each sub-goal should be independently achievable
- Specify dependencies using 0-based indices of previously listed sub-goals
- Assign priority (1-10, 10 = highest)
- Sub-goals should be concrete and testable
- Maximum {} levels of decomposition

Output STRICT JSON array: [{{"description":"...","depends_on_indices":[0],"priority":8}}]
Use empty array for depends_on_indices if no dependencies. depends_on_indices uses 0-based index of previously listed sub-goals.