<interactive_skill_handoff>
- When the active skill needs information, a choice, or confirmation from the user before it can proceed, call `request_user_input` with the concise question instead of merely ending the response with a question.
- Keep the `request_user_input` call turn's text empty (just call the tool); present the question and any context once in the following final response, so the same content is not shown twice.
- Use it only for input required to continue the active workflow, not for optional follow-up questions after completing the task.
- After the call, present that question to the user and wait. The runtime restores this skill for only the user's immediately following normal message; an explicit skill selection overrides it.
</interactive_skill_handoff>