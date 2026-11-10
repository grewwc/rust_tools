// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_task_results.test.cjs
// Execute the page's tool dispatch, replay and canonical rendering with the
// terminal payload and the actual task tool result envelopes.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const task = "Verify subagent lifetime/wait semantics in runtime";
const command = "completed: rg --files -g AGENTS.md -g '*task*' -g '*agent-team*' src/...";
const terminalPayload = [
  "✓ task_spawn",
  "[deepseek-v4-flash-volcano (effort: max) · task_spawn与task_wait之间模型空等]",
  "✓ thinking · 1 lines",
  "↳ cache · 34.8k/41.6k tokens · 84% hit",
  "↳ speed · reasoning 50 tok @ 67.5 tok/s · output 119 tok @ 107 tok/s",
  "✓ task_wait",
  "subagents · 1 active",
  "└─ [backend=traecli] build (GPT-6.1-Sol) (effort: max) · running 29s · task: " + task,
  "└─ phase: processing output · updates 23 · latest: " + command + " · 15s ago",
].join("\n");

function statusTable(rows) {
  const widths = [37, 8, 14, Math.max(14, ...rows.map((row) => Buffer.byteLength(row[3]))), 17, 11];
  return [["TaskID", "PID", "Agent", "Model", "Effort", "State", "Description"], ...rows]
    .map((row) => row.map((value, index) => String(value).padEnd(widths[index] || 0)).join(" ")).join("\n");
}

function source(first, last) {
  const start = script.indexOf(first), end = script.indexOf(last, start);
  assert.ok(start >= 0 && end > start, "missing page source: " + first);
  return script.slice(start, end);
}

function harness({ phone = false, storage = new Map() } = {}) {
  function element(tag = "div") {
    let text = "";
    const listeners = new Map();
    const el = {
      tagName: tag.toUpperCase(), children: [], parentNode: null, className: "", dataset: {}, hidden: false, open: false,
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        this.children.forEach((child) => { child.parentNode = null; });
        this.children = [];
      },
      get lastElementChild() { return this.children.at(-1) || null; },
      get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; },
      appendChild(child) { child.remove(); this.children.push(child); child.parentNode = this; return child; },
      insertBefore(child, anchor) {
        if (child === anchor) return child;
        assert.ok(!anchor || anchor.parentNode === this, "the insertion anchor belongs to this card");
        child.remove();
        const index = anchor ? this.children.indexOf(anchor) : this.children.length;
        this.children.splice(index, 0, child); child.parentNode = this;
        return child;
      },
      remove() {
        if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
        this.parentNode = null;
      },
      matches(selector) {
        const classes = [...selector.matchAll(/\.([\w-]+)/g)].map((match) => match[1]);
        const tag = /^[\w-]+/.exec(selector)?.[0];
        return (!tag || this.tagName === tag.toUpperCase()) && classes.every((name) => this.classList.contains(name));
      },
      querySelectorAll(selector) {
        return this.children.flatMap((child) => [
          ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
      },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
      closest(selector) {
        for (let node = this; node; node = node.parentNode) if (node.matches(selector)) return node;
        return null;
      },
      addEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).concat(fn)); },
      fire(type) { for (const fn of listeners.get(type) || []) fn({ target: this }); },
    };
    el.classList = {
      contains: (name) => el.className.split(/\s+/).includes(name),
      add: (name) => { if (!el.classList.contains(name)) el.className = (el.className + " " + name).trim(); },
      remove: (name) => { el.className = el.className.split(/\s+/).filter((part) => part !== name).join(" "); },
    };
    return el;
  }
  const root = element(), context = vm.createContext({
    document: { createElement: element, hidden: false, body: root, addEventListener() {} },
    window: { addEventListener() {} }, MutationObserver: class { observe() {} },
    $: () => root, sessionId: "s", isPhoneLayout: () => phone,
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    textOf: (text) => typeof text === "string" ? text : "", isRuntimeSyntheticUser: () => false,
    Date: { now: () => 1000 }, setInterval: () => 1, clearInterval() {},
    createTurnRateLine: () => () => {},
    createOutputRateCounter: () => ({ text: () => "", counts: () => "", reset() {} }),
    createThinkingRateRecorder: () => ({ start() {}, close() {} }),
    statusRateText: (text) => text,
    closeThinkingPhase() {}, sealAnswerSegment() {}, mutateTranscript: (fn) => fn(), schedulePaint() {},
  });
  for (const [first, last] of [
    ["function toolArgsOf(", "const RATE_STORAGE ="],
    ["function addFold(", "// The three override picks"],
    ["function historyRowsOf(", "// A session the server reports as running"],
    ["function recoveredProgressView(", "function showEventWarning("],
  ]) vm.runInContext(source(first, last), context);

  const send = script.slice(script.indexOf("async function send()"));
  const liveBranch = /else if \(kind === "tool"\) \{([\s\S]*?)\n        \}\n        else if \(kind === "subagent_progress"\)/.exec(send);
  assert.ok(liveBranch, "missing live SSE tool dispatch");
  vm.runInContext("function liveTool(payload) {" + liveBranch[1] + "\n}", context);
  const emit = (mode, output, name = "task_wait", extra = {}) => {
    const event = { id: "wait", name, phase: "finished", ok: true, output, ...extra };
    if (mode === "live") {
      context.activity ||= context.createLiveActivity(root, null);
      context.liveTool(JSON.stringify(event));
    } else if (mode === "replay") {
      context.view ||= context.recoveredProgressView("s");
      context.handleReplayEvent("s", "tool", JSON.stringify(event), context.view);
    } else {
      const messages = [
        { role: "assistant", content: "", tool_calls: [{ id: "wait", function: { name, arguments: extra.arguments || "{}" } }] },
        { role: "tool", tool_call_id: "wait", content: output },
      ];
      // A fresh document receives canonical JSON only, with no replay or rate cache.
      const canonical = JSON.parse(JSON.stringify(messages));
      for (const row of context.historyRowsOf(canonical, canonical.length, true).rows) context.renderHistoryRow(row, root);
    }
    return root.querySelector(".task-output");
  };
  return { root, context, storage, emit };
}

function assertTerminalDetails(panel) {
  assert.ok(panel, "the task result has no visible structured preview");
  assert.equal(panel.closest("details"), null, "task details are hidden by the tool fold");
  const rows = panel.children.map((row) => row.textContent);
  for (const value of [
    "✓ task_spawn", "deepseek-v4-flash-volcano", "effort: max", "task_spawn与task_wait之间模型空等",
    "thinking · 1 lines", "cache · 34.8k/41.6k tokens · 84% hit",
    "reasoning 50 tok @ 67.5 tok/s · output 119 tok @ 107 tok/s",
    "subagents · 1 active", "backend=traecli", "build (GPT-6.1-Sol)", "running 29s", task,
    "phase: processing output · updates 23 · latest: " + command + " · 15s ago",
  ]) assert.ok(rows.some((row) => row.includes(value)), "missing task detail: " + value);
  assert.equal(rows.filter((row) => row.includes("subagents · 1 active")).length, 1);
  assert.equal(rows.filter((row) => row.includes(command)).length, 1);
  assert.ok(!rows.includes("✓ task_wait"), "the tool title is repeated in its result");
}

for (const phone of [false, true]) {
  for (const mode of ["live", "replay", "history", "cold"]) {
    test(`terminal task payload: ${mode} ${phone ? "phone" : "desktop"} retains every supplied detail`, () => {
      const h = harness({ phone });
      const panel = h.emit(mode, terminalPayload);
      assertTerminalDetails(panel);
      const card = panel.parentNode, details = card.querySelector("details");
      assert.equal(details.open, false, "structured rendering changed the existing fold state");
      assert.doesNotMatch(card.querySelector(".prog").textContent, /34\.8k|GPT-6\.1-Sol|updates 23/,
        "structured rows are duplicated in the folded raw result");
      assert.equal(panel.hidden, false);
      assert.equal(card.hidden, false);
    });
  }
}

test("repeated live completion replaces the preview without duplicate rows or changing an open fold", () => {
  const h = harness();
  const panel = h.emit("live", terminalPayload);
  assertTerminalDetails(panel);
  const details = panel.parentNode.querySelector("details");
  details.open = true;
  const updated = terminalPayload.replace("updates 23", "updates 24").replace("15s ago", "1s ago");
  h.emit("live", updated);
  h.emit("live", updated);
  assert.equal(h.root.querySelectorAll(".task-output").length, 1);
  assert.equal(details.open, true);
  assert.equal(panel.children.filter((row) => row.textContent.includes(command)).length, 1);
  assert.match(panel.textContent, /updates 24/);
  assert.doesNotMatch(panel.textContent, /updates 23|15s ago/);
});

test("task_spawn projects its real JSON fields and the matching input prompt into readable rows", () => {
  const h = harness();
  const output = JSON.stringify({
    spawned: 1, failed: 0,
    tasks: [{ index: 0, task_id: "task_build", pid: 42, status: "spawned", agent: "build", model: "GPT-6.1-Sol",
      reasoning_effort: "max", backend: "traecli", progress: "phase: processing output · updates 23 · latest: " + command + " · 15s ago" }],
    next: "Use task_status for non-blocking progress.",
  });
  const panel = h.emit("live", output, "task_spawn", { arguments: JSON.stringify({ tasks: [{ prompt: task }] }) });
  assert.ok(panel);
  assert.match(panel.textContent, /subagents · 1 spawned · 0 failed/);
  assert.match(panel.textContent, /\[backend=traecli\] build \(GPT-6\.1-Sol\) \(effort: max\) · spawned/);
  assert.ok(panel.textContent.includes(task));
  assert.match(panel.textContent, /updates 23/);
  assert.match(panel.textContent, /task_id: task_build/);
  assert.doesNotMatch(panel.textContent, /"tasks"|"reasoning_effort"|\{|\}/);
  const raw = panel.parentNode.querySelector(".prog").textContent;
  assert.match(raw, /Arguments/);
  assert.match(raw, /Use task_status/);
  assert.ok(!raw.includes(output), "the original JSON duplicates the structured result");
});

test("task_status renders the real table and associates its progress with the task id", () => {
  const h = harness();
  const output = [
    statusTable([["task_build", 42, "build", "GPT-6.1-Sol", "max", "running", task]]),
    "  progress[task_build]: processing output · updates 23 · last activity 15s ago",
  ].join("\n");
  const panel = h.emit("history", output, "task_status");
  assert.ok(panel);
  assert.match(panel.textContent, /build \(GPT-6\.1-Sol\) \(effort: max\) · running/);
  assert.ok(panel.textContent.includes(task));
  assert.match(panel.textContent, /progress\[task_build\]: processing output · updates 23/);
  assert.doesNotMatch(panel.textContent, /backend=|running \d+s|speed ·|cache ·/);
});

test("ordinary task_wait history keeps task metadata, its last progress and the full delivered body", () => {
  const h = harness();
  const body = "Delivered result: " + "evidence ".repeat(400) + "end";
  const output = "[task_id=task_build]\n[Task: " + task + " via build @ GPT-6.1-Sol (effort: max)] COMPLETED after 29.0s\n" +
    "Last known progress: processing output · updates 23 · last activity 15s ago\n" + body;
  const before = output;
  const panel = h.emit("cold", output);
  assert.ok(panel);
  assert.match(panel.textContent, /task_id: task_build/);
  assert.match(panel.textContent, /COMPLETED after 29\.0s/);
  assert.match(panel.textContent, /Last known progress: processing output · updates 23/);
  assert.equal(panel.parentNode.querySelector(".prog").textContent,
    "Last known progress: processing output · updates 23 · last activity 15s ago\n" + body);
  assert.equal(output, before, "the display parser mutated canonical evidence");
});

test("task parser regression: delivered metadata-looking lines remain byte-for-byte in the body", () => {
  const selection = "agent_reason=explicit agent override\nmodel_reason=explicit model override";
  const body = [
    "Delivered config:", "```toml", "[features]", "rate = true", "```",
    "[diagnostic]", "Last known progress: text supplied by the subagent",
    "Failure reason: text supplied by the subagent", "subagents · 9 active",
    "  indented evidence  ", "", "\u001b[32mcolored delivered text\u001b[0m", "",
  ].join("\n");
  const output = "[task_id=task_build]\n[Task: " + task + " via build @ GPT-6.1-Sol (effort: max)] COMPLETED after 29.0s\n" +
    selection + "\nLast known progress: processing output · updates 23\n" + body;
  for (const mode of ["live", "replay", "cold"]) {
    const h = harness(), panel = h.emit(mode, output);
    assert.ok(panel);
    assert.equal(panel.parentNode.querySelector(".prog").textContent,
      selection + "\nLast known progress: processing output · updates 23\n" + body);
    assert.doesNotMatch(panel.textContent, /\[features\]|\[diagnostic\]|text supplied by the subagent|9 active/);
  }
});

test("task parser regression: an unclosed Task marker retains the plain-text fallback", () => {
  const output = "[Task: truncated without closing bracket";
  for (const mode of ["live", "replay", "cold"]) {
    const h = harness();
    assert.equal(h.emit(mode, output), null);
    assert.equal(h.root.querySelector(".prog").textContent, output);
  }
});

test("task parser regression: incidental task and bracket markers retain the plain-text fallback", () => {
  for (const output of [
    "Cannot parse output\ntask_wait is unavailable\n[diagnostic]",
    "✓ task_wait\n[features]",
    "[task_id=task_build]\n[Task: truncated without closing bracket",
  ]) {
    const h = harness();
    assert.equal(h.emit("live", output), null);
    assert.equal(h.root.querySelector(".prog").textContent, output);
  }
});

test("task parser regression: padded table columns preserve models with spaces and dynamic widths", () => {
  for (const model of ["GPT 6.1 Sol", "Vendor GPT 6.1 Sol Premium"]) {
    const output = statusTable([["task_build", 42, "build", model, "max", "running", task]]);
    const h = harness(), panel = h.emit("history", output, "task_status");
    assert.ok(panel);
    assert.equal(panel.querySelector(".task-identity").textContent, "build (" + model + ") (effort: max) · running");
    assert.equal(panel.querySelector(".task-description").textContent, "task: " + task);
    assert.equal(panel.querySelector(".task-meta").textContent, "task_id: task_build · pid: 42");
  }
});

test("task parser regression: textual failure reasons render visibly beside the tool fold", () => {
  const reason = "The subagent exceeded its wall-clock limit.";
  const output = "[task_id=task_build]\n[Task: " + task + " via build @ GPT-6.1-Sol (effort: max)] FAILED after 29.0s\n" +
    "agent_reason=explicit agent override\nmodel_reason=explicit model override\nFailure reason: " + reason +
    "\nLast known progress: processing output · updates 23\nPartial output:\nSaved evidence";
  for (const mode of ["live", "replay", "cold"]) {
    const h = harness(), panel = h.emit(mode, output, "task_wait", { ok: false });
    assert.ok(panel);
    const error = panel.querySelector(".task-error");
    assert.ok(error, "the textual failure reason is hidden in the raw fold");
    assert.equal(error.textContent, "Failure reason: " + reason);
    assert.equal(error.closest("details"), null);
    assert.match(panel.parentNode.querySelector(".prog").textContent, /Partial output:\nSaved evidence/);
  }
});

test("quiet task wait and status digests show their supplied state without fabricated metrics", () => {
  for (const [name, output] of [
    ["task_wait", "[task-wait] still running: reason=budget_elapsed waited=60s task_build(idle=15s,phase=processing output)"],
    ["task_status", "[task-status] running=1 done=0 failed=0\ntask_build status=running phase=processing output idle=15s rev=2 checkpoint=saved finding"],
  ]) {
    const h = harness(), panel = h.emit("history", output, name);
    assert.ok(panel);
    assert.ok(panel.textContent.includes("processing output"));
    assert.doesNotMatch(panel.textContent, /backend=|\(effort:|cache ·|speed ·|tok\/s|latest:/);
  }
});

test("pending task_wait exposes the real latest-progress snapshot below its parked status", () => {
  const h = harness();
  const output = "[task_wait PARKED] Yielded CPU so 1 pending subagent task(s) can run. Pending task_ids: [task_build].\n" +
    "Latest progress snapshots:\n- task_build: processing output · updates 23 · last activity 15s ago";
  const panel = h.emit("history", output);
  assert.ok(panel);
  assert.match(panel.textContent, /wait · parked · 1 pending/);
  assert.match(panel.textContent, /task_build: processing output · updates 23 · last activity 15s ago/);
  assert.doesNotMatch(panel.parentNode.querySelector(".prog").textContent, /updates 23/);
});

test("missing JSON task fields omit identity, phase, elapsed time and throughput rather than inventing them", () => {
  const h = harness();
  const panel = h.emit("live", '{"tasks":[{"task_id":"task_build","status":"running"}]}');
  assert.ok(panel);
  assert.match(panel.textContent, /task_id: task_build/);
  assert.match(panel.textContent, /running/);
  assert.doesNotMatch(panel.textContent, /backend=|model|effort|phase:|latest:|updates|\d+s|cache|speed|tok/);
});

test("malformed task JSON and unrelated tools keep the raw fallback and truncation notices", () => {
  for (const [name, output] of [["task_wait", '{"tasks":[{'], ["task_spawn", '{"unknown":42}'], ["read_file", terminalPayload]]) {
    const h = harness();
    assert.equal(h.emit("live", output, name, { arguments: "{", arguments_truncated: true, output_truncated: true, ok: false }), null);
    assert.equal(h.root.querySelector("details").open, false);
    const raw = h.root.querySelector(".prog").textContent;
    assert.ok(raw.includes(output));
    assert.match(raw, /Arguments\n\{\n\[preview truncated\]/);
    assert.match(raw, /full output in history/);
    assert.match(h.root.querySelector("summary").textContent, /^! /);
  }
});

test("truncated recognized previews keep the warning and parsed-to-raw updates remove stale task rows", () => {
  const h = harness();
  const panel = h.emit("live", terminalPayload, "task_wait", { output_truncated: true });
  assert.ok(panel);
  assert.match(panel.parentNode.querySelector(".prog").textContent, /full output in history/);
  h.emit("live", "Error: result unavailable", "task_wait", { output_truncated: false, ok: false });
  assert.equal(h.root.querySelector(".task-output"), null);
  assert.equal(h.root.querySelectorAll("summary").length, 1);
  assert.equal(h.root.querySelector(".prog").textContent, "Error: result unavailable");
});

test("history refresh and a fresh document retain fold expansion while structured details stay visible", () => {
  const storage = new Map(), h = harness({ storage });
  const panel = h.emit("history", terminalPayload), details = panel.parentNode.querySelector("details");
  details.open = true;
  details.fire("toggle");
  h.root.textContent = "";
  const refreshed = h.emit("history", terminalPayload);
  assert.equal(refreshed.parentNode.querySelector("details").open, true);
  const cold = harness({ storage: new Map(storage) }), reopened = cold.emit("cold", terminalPayload);
  assert.equal(reopened.parentNode.querySelector("details").open, true);
  assertTerminalDetails(reopened);
  const closed = reopened.parentNode.querySelector("details");
  closed.open = false;
  closed.fire("toggle");
  const next = harness({ storage: new Map(cold.storage) }), collapsed = next.emit("cold", terminalPayload);
  assert.equal(collapsed.parentNode.querySelector("details").open, false);
  assertTerminalDetails(collapsed);
});

test("terminal control sequences are removed from parsed rows and command text stays selectable plain text", () => {
  const h = harness();
  const injection = '<img src=x onerror="alert(1)">';
  const panel = h.emit("live", "\u001b[32m" + terminalPayload.replace(command, injection) + "\u001b[0m");
  assert.ok(panel);
  assert.ok(panel.textContent.includes(injection));
  assert.doesNotMatch(panel.textContent, /\u001b/);
  assert.equal(panel.querySelectorAll("img").length, 0);
  assert.match(css, /\.msg \.task-output\s*\{[^}]*overflow-wrap:\s*anywhere/);
  assert.match(css, /\.msg \.task-output\s*\{[^}]*white-space:\s*pre-wrap/);
  assert.doesNotMatch(css, /\.task-output[^}]*user-select:\s*none/);
});
