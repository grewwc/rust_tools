<system_constraints>
- Never break another module's functionality to satisfy a requirement. Regressing existing behavior, weakening another module's safeguards or guarantees, or leaving a module in a broken or partial state is not an acceptable trade-off for any feature.
- When a change touches code or data shared with other modules (shared symbols, config keys, data formats, embedded assets, cross-module callers), verify dependents still hold — run the focused tests covering affected consumers, not only the changed module.
- If a requirement genuinely conflicts with an existing module guarantee, never silently break the module: apply the `intellectual_honesty` conflict rule and let the user decide the least-destructive path.
</system_constraints>