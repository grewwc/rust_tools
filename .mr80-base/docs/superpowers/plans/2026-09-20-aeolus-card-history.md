# Aeolus Card History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make validation and Coding each wait 10 seconds, improve card contrast, and show the complete in-session Coding round history.

**Architecture:** Keep the existing React component and front-end Mock flow. Move round transitions into one CommonJS-compatible pure reducer in `workflow.js`; `App.tsx` owns async waits and renders the reducer state, while CSS owns the timeline and contrast changes.

**Tech Stack:** React, TypeScript/TSX, CSS, Node.js built-in test runner, LPM/Webpack.

## Global Constraints

- Do not add dependencies, network requests, persistence, or backend behavior.
- Use `PROCESS_DELAY_MS = 10_000` for both validation and every Coding run.
- Keep history only for the current component session; refresh clears it.
- Latest round is expanded; older rounds remain expandable.
- Keep tests focused on public state transitions rather than markup details.

---

### Task 1: Coding-round state model

**Files:**
- Modify: `aeolus-larkoffice/tests/workflow.test.cjs`
- Modify: `aeolus-larkoffice/src/features/ai_node_web_aeolus_card/workflow.js`

**Interfaces:**
- Produces: `PROCESS_DELAY_MS` number and `reduceRounds(rounds, event)` pure function.
- `event.type` is one of `START`, `CODED`, `FEEDBACK`, `CONFIRM`, `FAIL`, or `RETRY`.

- [x] **Step 1: Write concise failing tests**

Add one delay assertion, one multi-round happy-path test, and one failure/retry test. The multi-round test drives `START → CODED → FEEDBACK` three times and asserts four ordered rounds before confirming the latest. The retry test asserts `FAIL → RETRY` keeps the same round ID and array length; it also asserts blank feedback returns the unchanged history.

- [x] **Step 2: Run the tests and verify RED**

Run: `npm test`

Expected: FAIL because `PROCESS_DELAY_MS` and `reduceRounds` are not exported.

- [x] **Step 3: Implement the minimum reducer**

Add:

```js
const PROCESS_DELAY_MS = 10_000;

function reduceRounds(rounds, event) {
  const current = rounds[rounds.length - 1];
  const replaceCurrent = (updates) => [
    ...rounds.slice(0, -1),
    { ...current, ...updates },
  ];

  switch (event.type) {
    case 'START':
      return [...rounds, { id: (current?.id || 0) + 1, status: 'coding', feedback: '', codeUrl: '' }];
    case 'CODED':
      return current ? replaceCurrent({ status: 'review', codeUrl: event.codeUrl }) : rounds;
    case 'FEEDBACK':
      if (!current || !canSubmitFeedback(event.feedback)) return rounds;
      return [
        ...replaceCurrent({ status: 'feedback', feedback: event.feedback.trim() }),
        { id: current.id + 1, status: 'coding', feedback: '', codeUrl: '' },
      ];
    case 'CONFIRM':
      return current ? replaceCurrent({ status: 'completed' }) : rounds;
    case 'FAIL':
      return current ? replaceCurrent({ status: 'error' }) : rounds;
    case 'RETRY':
      return current ? replaceCurrent({ status: 'coding' }) : rounds;
    default:
      return rounds;
  }
}
```

Export both new symbols without changing existing helpers.

- [x] **Step 4: Run tests and verify GREEN**

Run: `npm test`

Expected: all workflow tests pass.

### Task 2: Card flow, history timeline, and contrast

**Files:**
- Modify: `aeolus-larkoffice/src/features/ai_node_web_aeolus_card/App.tsx`
- Modify: `aeolus-larkoffice/src/features/ai_node_web_aeolus_card/index.css`

**Interfaces:**
- Consumes: `PROCESS_DELAY_MS` and `reduceRounds` from Task 1.
- Keeps existing `window.JSSDK.navigation.open(codeUrl)` behavior unchanged.

- [x] **Step 1: Integrate round state into the component**

Define the local `CodingRound` type matching the design spec and replace standalone `codeUrl` state with `rounds`. On validation success dispatch `START`; on Coding completion dispatch `CODED`; feedback dispatches `FEEDBACK` before calling `runCoding`; confirmation dispatches `CONFIRM`. Record whether the last failed operation was validation or Coding so the error retry executes the correct operation; Coding retry dispatches `RETRY` without appending a round.

- [x] **Step 2: Render the timeline**

Below the active operation, render `rounds` as semantic `<details>` items. Each summary contains `第 N 轮` and its localized status. The latest item receives `open`; expanded content shows saved feedback when present and a code-link button when the round has a URL.

- [x] **Step 3: Apply the approved contrast and responsive styles**

Use `#1F2329` for the title, `#5B3CC4` on `#F3F0FF` for the AI-node badge, `#176B3A` on `#EAF8F0` with `#A9D9BD` for success, and `#1456F0` for code links. Add a vertical timeline with blue/purple/orange/green/red state chips and retain the existing single-column narrow-screen layout.

- [x] **Step 4: Run focused verification**

Run: `npm test`

Expected: all tests pass with no warnings or errors.

Run: `lpm build` using the installed absolute CLI path if `lpm` is absent from `PATH`.

Expected: Webpack build succeeds. The known standalone bundled-TypeScript incompatibility is not treated as a card regression when the LPM build passes.

### Task 3: Audit and commit

**Files:**
- Inspect only the files listed in Tasks 1 and 2 plus this plan.

- [x] **Step 1: Audit SDK usage and backend boundaries**

Run the LPM JSSDK audit against the card files. Then scan for `fetch('/api/proxy/` and confirm no new proxy route or backend requirement was introduced.

- [x] **Step 2: Review the diff**

Run `git diff --check` and verify every changed production line maps to delay, history, retry correctness, or approved visual contrast. Leave unrelated working-tree files untouched.

- [x] **Step 3: Commit the implementation**

Stage only the plan, `workflow.js`, its test, `App.tsx`, and `index.css`, then commit with:

```bash
git commit -m "feat: add aeolus coding round history"
```
