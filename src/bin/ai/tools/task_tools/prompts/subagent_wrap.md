<subagent_task>
{task_description}
</subagent_task>

Runtime constraints:
- Treat this as a bounded leaf task for the parent agent. Do not expand scope beyond the task.
- Reuse observed evidence and avoid equivalent read/search/list/command variants unless omitted text is needed; prefer one targeted broad call over many small ones.
- Ground factual claims in observed evidence. For review or diagnosis, trace the relevant path and check likely counter-evidence before reporting a finding.
- If evidence is incomplete, return a concise partial result separating confirmed conclusions, unresolved hypotheses, missing evidence, and the next verification step.

{response_contract}<parent_task_prompt>
{parent_prompt}
</parent_task_prompt>