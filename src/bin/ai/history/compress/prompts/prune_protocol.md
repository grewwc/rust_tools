
## Context Management Protocol
Mark a candidate only when you have finished using it and it is fully superseded;
age, compression, an edited file, or a shorter replacement alone does not qualify. Put the
candidate's exact id — a `tool_call_id`, or a `fold_...` id for a folded group — on its own
`prune:` line holding nothing but comma-separated ids, inside a hidden self-note written as
plain text in your reply (never as a tool call):

<meta:self_note>
prune:call_abc,call_xyz
</meta:self_note>

The note is not shown to the user; its non-directive lines stay in your context as your
self-note, so keep any normal self_note content in the same note.
Only ids present in this request can be marked — the list below shows the largest eligible
ones, and a mark that cannot be applied is reported back with its reason. Never mark
user/system/assistant messages, checkpoints, plans, unknown-source evidence, or recent tool
results; a fold holding any protected tool is ineligible; a subagent result appears only
after its task is integrated.
Do not mark findings, decisions, constraints, or verification evidence you still need:
offloading is lossless (exact text archived, recallable stub, canonical history unchanged),
but hiding evidence can still affect reasoning.
Marks accumulate across turns (need not be consecutive); offloading requires the listed
threshold (`marks n/threshold`); a folded group always needs two distinct responses.