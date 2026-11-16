# NextCode Webhook fixtures

These payloads use the CamelCase field names from `webhook_event.thrift` and
the identifiers of the retained verification MR !9. They intentionally omit
unconsumed optional fields and contain no credentials.

They can be replayed against a running local service with
`scripts/replay_nextcode_webhook.py`. Inject the Webhook Secret token through
the process environment; never add a literal secret to these files or a shell
command. The fixtures exercise:

- `comment/created`: both `Comment` and `Comments`, including sparse
  `CreatedBy` data;
- `merge_request_review/commented`: overall review text in `Review.Content`;
- `merge_request/edited`: MR fields used before querying mergeability;
- `push`: a target-branch update used before listing affected open MRs and
  querying their mergeability.

The fixtures are receiver-focused, not complete snapshots of every optional
IDL field. For a new live run, replace the retained repository/MR/branch IDs
with the output from `scripts/verify_nextcode_contracts.py`.
