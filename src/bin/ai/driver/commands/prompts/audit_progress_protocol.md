<audit_progress_protocol>
After finishing each independent audit branch, and before calling any further tools, output one short progress record starting with `AUDIT_CHECKPOINT:`, containing: the scope already checked, findings so far with file:line, and the questions still to verify.
Do not wait until the final answer to report findings for the first time; checkpoints are recoverable stage-by-stage evidence.
Once the wrap-up signal arrives, stop expanding the investigation immediately and produce the final audit conclusion from the checkpoints and tool evidence gathered so far.
</audit_progress_protocol>