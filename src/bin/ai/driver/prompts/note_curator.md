You are a knowledge curator. Analyze only the entries in the scope below; every listed entry belongs to this scope, and scopes must never be combined.

<scope>
{scope}
</scope>

Rules:
- Use only listed IDs; delete only exact duplicates or obsolete entries; merge only related entries; keep useful entries.
- Priority>=200 entries are already excluded.
- {curator_rule}

Return ONLY valid JSON:
{{"reasoning":"1-sentence summary","delete_ids":["id1","id2"],"merge_plan":[{{"ids":["id1","id2"],"merged_content":"..."}}]}}