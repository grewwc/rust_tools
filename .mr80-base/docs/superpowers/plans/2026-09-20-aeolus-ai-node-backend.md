# Aeolus AI Node Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-memory FastAPI backend for the Aeolus AI node that validates a configured Feishu requirement document, asks Coco to implement it in `dp/aeolus:master`, reuses one branch/MR for feedback rounds, and completes the Meegle node after user confirmation.

**Architecture:** The webhook acknowledges within one second and schedules an `AeolusService` background task. A locked in-memory registry owns task state and round history; narrow integration clients handle Feishu document reads, Meegle AI-node OpenAPI, Coco sessions, and existing NextCode helpers.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, httpx, pytest/pytest-asyncio, existing Coco and NextCode clients.

## Global Constraints

- Keep repository and base branch fixed to `dp/aeolus` and `master`.
- Read the AI-node input property with key `design_doc`.
- Start only for event `8001` whose `change_ai_node_info.after.status` is `running`.
- Verify webhook signatures as lowercase hex SHA-256 of `source_plugin_id + request_time + webhook_token`.
- Return webhook HTTP 200 before running document, Coco, or NextCode work.
- Keep state in memory; do not add a database migration or dependency.
- Reuse the same task branch, Coco session when usable, and MR for feedback rounds.
- Do not modify existing unrelated dirty files.

---

### Task 1: Event contract and in-memory state

**Files:**
- Create: `src/conciliator/domain/aeolus_task.py`
- Create: `src/conciliator/services/aeolus_registry.py`
- Test: `tests/unit/test_aeolus_task.py`
- Test: `tests/unit/test_aeolus_registry.py`

**Interfaces:**
- Produces `parse_start_event(payload, expected_plugin_id, webhook_token) -> AeolusStartEvent | None`.
- Produces `extract_requirement_url(event) -> str`.
- Produces async `AeolusTaskRegistry.create_from_event`, `get`, `find_by_node`, and guarded mutation helpers.

- [ ] Write parsing/signature/state-transition tests and verify they fail because the modules do not exist.
- [ ] Implement immutable event parsing plus task/round snapshots and a lock-protected registry.
- [ ] Run `pytest tests/unit/test_aeolus_task.py tests/unit/test_aeolus_registry.py -q` and require PASS.

### Task 2: Platform clients

**Files:**
- Create: `src/conciliator/integrations/meegle/ai_node.py`
- Create: `src/conciliator/integrations/meegle/__init__.py`
- Modify: `src/conciliator/integrations/lark/docs.py`
- Test: `tests/unit/test_meegle_ai_node.py`
- Modify: `tests/unit/test_lark_docs.py`

**Interfaces:**
- Produces `MeegleAINodeClient.edit_status(ref, status, idem_key)` and `confirm(ref, idem_key)`.
- Produces `extract_document_ref(url) -> DocumentRef` and `read_document_text(client, url) -> str` supporting `/docx/` and `/wiki/` links.

- [ ] Write exact HTTP-contract tests for plugin-token caching, headers, edit/confirm bodies, docx raw content, and wiki resolution; verify RED.
- [ ] Implement the minimal clients using documented endpoints and source comments next to hand-written Meegle calls.
- [ ] Run the two targeted test files and require PASS.

### Task 3: Coco evaluation and coding adapter

**Files:**
- Create: `src/conciliator/integrations/coco/aeolus.py`
- Test: `tests/unit/test_coco_aeolus.py`

**Interfaces:**
- Produces `parse_evaluation_result`, `parse_coding_result`, evaluation/coding/feedback prompts, and `CocoAeolusRunner`.
- `CocoAeolusRunner.evaluate(text)` returns structured validity, missing items, and risks.
- `CocoAeolusRunner.code(...)` and `.recode(...)` return session/cursor/commit metadata.

- [ ] Write strict marker/schema parsing and prompt contract tests; verify RED.
- [ ] Implement parsing, prompts, session creation, message dispatch, and bounded result polling.
- [ ] Run `pytest tests/unit/test_coco_aeolus.py -q` and require PASS.

### Task 4: Orchestration and HTTP API

**Files:**
- Create: `src/conciliator/services/aeolus_service.py`
- Create: `src/conciliator/api/aeolus.py`
- Modify: `src/conciliator/api/__init__.py`
- Modify: `src/conciliator/config.py`
- Modify: `src/conciliator/main.py`
- Modify: `src/conciliator/utils/secrets.py`
- Test: `tests/unit/test_aeolus_service.py`
- Test: `tests/unit/test_aeolus_api.py`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Webhook `POST /hooks/meegle/ai-node` returns an acknowledgement after scheduling work.
- Card APIs expose lookup, retry validation, feedback, and confirmation.
- `AeolusService` reads/evaluates the document, creates the branch/MR, records rounds, and completes the platform node.

- [ ] Write service tests with fakes for validation failure, successful coding/MR creation, feedback reuse, and idempotent confirmation; verify RED.
- [ ] Write API tests for signature rejection, fast scheduling, task lookup, action-token checks, and state guards; verify RED.
- [ ] Implement the smallest service/API/config/lifespan changes that pass those tests.
- [ ] Run all Aeolus tests, then `pytest tests/unit -q`, `ruff check`, and `mypy src/conciliator`.

### Task 5: Final verification

**Files:**
- Modify only files introduced or required by Tasks 1-4.

- [ ] Review `git diff --check` and `git status --short`; confirm unrelated dirty files remain untouched.
- [ ] Run the narrow frontend tests to confirm no published-card regression: `npm test --prefix aeolus-larkoffice`.
- [ ] Record any environment-only integration gaps without claiming a live Meegle/Coco run.
