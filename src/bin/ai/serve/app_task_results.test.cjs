// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_task_results.test.cjs
// Execute the page's live, replay, and canonical paths with versioned reports.
// Historical prose is displayed verbatim without deriving task state.
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

function harness({ phone = false, width = phone ? 390 : 1280, storage = new Map() } = {}) {
  const clock = { now: 1000 };
  function checkPlacement(owner, child) {
    for (let ancestor = owner; ancestor; ancestor = ancestor.parentNode) {
      assert.notEqual(ancestor, child, "DOM placement cannot insert an ancestor into its descendant");
    }
  }
  function element(tag = "div") {
    let text = "";
    const listeners = new Map();
    const el = {
      tagName: tag.toUpperCase(), children: [], parentNode: null, className: "", dataset: {}, hidden: false, open: false,
      get isConnected() {
        for (let node = this; node; node = node.parentNode) if (node === root) return true;
        return false;
      },
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        this.children.forEach((child) => { child.parentNode = null; });
        this.children = [];
      },
      get lastElementChild() { return this.children.at(-1) || null; },
      get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; },
      appendChild(child) { checkPlacement(this, child); child.remove(); this.children.push(child); child.parentNode = this; return child; },
      insertBefore(child, anchor) {
        if (child === anchor) return child;
        checkPlacement(this, child);
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
        if (selector.includes(",")) return selector.split(",").some((part) => this.matches(part.trim()));
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
    window: {
      addEventListener() {},
      matchMedia(query) {
        assert.equal(query, "(min-width:701px)");
        return { matches: width >= 701 };
      },
    }, MutationObserver: class { observe() {} },
    $: () => root, sessionId: "s", isPhoneLayout: () => phone,
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    textOf: (text) => typeof text === "string" ? text : "", isRuntimeSyntheticUser: () => false,
    Date: { now: () => clock.now }, setInterval: () => 1, clearInterval() {},
    createTurnRateLine: () => () => {},
    createOutputRateCounter: () => ({ text: () => "", counts: () => "", reset() {} }),
    createThinkingRateRecorder: () => ({ start() {}, close() {} }),
    statusRateText: (text) => text,
    closeThinkingPhase() {}, sealAnswerSegment() {}, mutateTranscript: (fn) => fn(), schedulePaint() {},
  });
  for (const [first, last] of [
    ["let lastFoldTap", "// The agent sends a phone action"],
    ["function toolArgsOf(", "const RATE_STORAGE ="],
    ["function addFold(", "// The three override picks"],
    ["function historyRowsOf(", "// A session the server reports as running"],
    ["function recoveredProgressView(", "function showEventWarning("],
  ]) vm.runInContext(source(first, last), context);

  const send = script.slice(script.indexOf("async function send()"));
  const liveBranch = /else if \(kind === "tool"\) \{([\s\S]*?)\n        \}\n        else if \(kind === "subagent_progress"\)/.exec(send);
  assert.ok(liveBranch, "missing live SSE tool dispatch");
  vm.runInContext("function liveTool(payload) {" + liveBranch[1] + "\n}", context);
  const historyMessages = [];
  const emit = (mode, output, name = "task_wait", extra = {}) => {
    const event = { id: "wait", name, phase: "finished", ok: true, output, ...extra };
    if (mode === "live") {
      context.activity ||= context.createLiveActivity(root, null);
      context.liveTool(JSON.stringify(event));
    } else if (mode === "replay") {
      context.view ||= context.recoveredProgressView("s");
      context.handleReplayEvent("s", "tool", JSON.stringify(event), context.view);
    } else {
      historyMessages.push(
        { role: "assistant", content: "", tool_calls: [{ id: event.id, function: { name, arguments: extra.arguments || "{}" } }] },
        { role: "tool", tool_call_id: event.id, content: output, task_report: extra.task_report },
      );
      // A fresh document receives canonical JSON only, with no replay or rate cache.
      const canonical = JSON.parse(JSON.stringify(historyMessages));
      root.textContent = "";
      for (const row of context.historyRowsOf(canonical, canonical.length, true).rows) context.renderHistoryRow(row, root);
    }
    return root.querySelector(".task-output");
  };
  return { root, context, storage, clock, emit };
}


function report(tasks, extra = {}) {
  return { schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: false, tasks, ...extra };
}
function reportedTask(task_id, overrides = {}) {
  return {
    task_id, pid: 42, agent: "build", model: "Vendor GPT 6.1 Sol Premium", backend: "traecli",
    reasoning_effort: "max", description: task, status: "completed", active: false, elapsed_ms: 29000,
    progress: {
      phase: "processing output", latest_activity: command, sequence: 23, elapsed_ms: 29000, stale_for_ms: 15000,
      checkpoint_summary: "Saved findings", checkpoint_excerpt: "Evidence line one\n  Evidence line two  \n",
      checkpoint_revision: 2, checkpoint_due: false,
      timeline: [{ kind: "activity", sequence: 23, elapsed_ms: 14000, summary: command }],
    },
    output: "  Delivered result for " + task_id + "\n", ...overrides,
  };
}
function taskCard(h) {
  return h.root.querySelectorAll(".task-output").find((panel) => panel.parentNode.dataset.subagents === "true");
}
function displayFold(node) {
  return node.parentNode.classList.contains("execution-items") ? node.parentNode.parentNode.parentNode : node;
}
function assertTask(panel, expected) {
  assert.ok(panel, "a task result alone must create the dedicated subagent detail card");
  const rows = panel.children.filter((row) => row.dataset.taskId === expected.task_id);
  const all = rows.map((row) => row.textContent).join("\n");
  for (const value of [expected.task_id, expected.description, expected.agent, expected.model, expected.status]) {
    assert.ok(all.includes(value), "missing explicit task field: " + value);
  }
  if (expected.progress) {
    for (const field of ["phase", "latest_activity", "checkpoint_summary", "checkpoint_excerpt"]) {
      assert.ok(all.includes(expected.progress[field]), "missing structured progress: " + field);
    }
  }
  if (expected.error) assert.ok(rows.some((row) => row.className === "task-error" && row.textContent.includes(expected.error)));
  if (typeof expected.output === "string" && expected.output) {
    const body = rows.find((row) => row.className === "task-output-body");
    assert.ok(body, "missing delivered output for " + expected.task_id);
    assert.equal(body.textContent, expected.output, "delivered output was altered");
  }
}

for (const phone of [false, true]) for (const mode of ["live", "replay", "history", "cold"]) {
  const label = mode + " " + (phone ? "phone" : "desktop");
  for (const width of phone ? [390, 700] : [701, 1280]) test("structured subagent row double tap respects the desktop breakpoint, " + label + " " + width + "px", () => {
    const h = harness({ phone, width }), expected = reportedTask("task_collapse");
    h.emit(mode, "raw task result", "task_wait", { task_report: report([expected]) });
    const panel = taskCard(h), details = panel.parentNode.querySelector("details");
    const rawDetails = h.root.querySelector("details");
    const body = panel.querySelector(".task-output-body");
    assert.equal(body.closest("details"), null, "the delivered output is outside the disclosure");
    assert.equal(details.parentNode, panel.parentNode, "the disclosure and task rows share one owner");
    rawDetails.open = true; rawDetails.fire("toggle");
    details.open = true; details.fire("toggle");
    const click = (target, detail) => {
      const event = {
        target, detail, clientX: 100, clientY: 200, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; },
      };
      h.context.collapseFoldForTap(event);
      return event;
    };
    assert.equal(click(body, 1).defaultPrevented, false);
    assert.equal(details.open, true, "a single click leaves the report open");
    h.clock.now += 100;
    assert.equal(click(body, 2).defaultPrevented, false, "task text retains native selection events");
    assert.equal(details.open, phone, "only desktop closes the sibling disclosure; mobile retains its open state");
    if (!phone) details.fire("toggle");
    assert.equal(rawDetails.open, true, "the separate tool disclosure must stay open");
    if (mode === "live" || mode === "replay") {
      assert.equal(vm.runInContext('foldOpenByKey.get("subagents")', h.context), phone);
    } else {
      const cold = harness({ phone, width, storage: new Map(h.storage) });
      cold.emit("cold", "raw task result", "task_wait", { task_report: report([expected]) });
      assert.equal(taskCard(cold).parentNode.querySelector("details").open, phone,
        "canonical reconstruction must retain the desktop collapse or the untouched mobile expansion");
    }
  });
  test("structured task placement: distinct spawn and wait put one detail card after wait, " + label, () => {
    const h = harness({ phone });
    const spawned = reportedTask("task_placement", { status: "spawned", active: true, output: undefined });
    const completed = { ...spawned, status: "completed", active: false,
      output: "  exact delivered bytes\n[Task: literal] FAILED after 999s\n\u001b[32mraw ANSI\u001b[0m\n\n" };
    const spawnRaw = "\n  raw spawn response  \n";
    const waitRaw = "\n  raw wait response\n" + completed.output + "  trailing wait bytes  \n";
    h.emit(mode, spawnRaw, "task_spawn", { id: "spawn-call", task_report: report([spawned], { operation: "task_spawn" }) });
    const originalCard = taskCard(h).parentNode;
    h.emit(mode, waitRaw, "task_wait", { id: "wait-call", task_report: report([completed]) });
    const cards = h.root.querySelectorAll(".task-output").filter((panel) => panel.parentNode.dataset.subagents === "true");
    assert.equal(cards.length, 1, "an earlier spawn must not keep a separate subagent detail card");
    const card = cards[0].parentNode;
    const rawBodies = h.root.querySelectorAll(".prog");
    const waitFold = rawBodies.find((body) => body.textContent === waitRaw)?.closest(".msg.asst");
    assert.ok(waitFold, "the raw wait response must remain verbatim");
    assert.ok(displayFold(waitFold).nextElementSibling === card, "the detail card must follow the later wait result's display group");
    assert.ok(rawBodies.some((body) => body.textContent === spawnRaw), "moving the detail card changed raw spawn bytes");
    assertTask(cards[0], completed);
    if (mode === "live" || mode === "replay") {
      assert.equal(card, originalCard, "placement must reuse the shared card");
      const progressText = "  late raw backend progress  \n";
      const progress = { text: progressText, task_report: report([{ ...spawned, status: "running" }], { snapshot: true }) };
      if (mode === "live") h.context.activity.subagents(progress);
      else h.context.view.subagents(progress);
      assert.ok(displayFold(waitFold).nextElementSibling === card, "later progress must stay beside the latest result's display group");
      assert.equal(card.querySelector(".prog").textContent, progressText);
      assertTask(cards[0], completed);
    }
  });
  test("structured task placement: separate waits retain earlier delivered bodies in the final card, " + label, () => {
    const h = harness({ phone });
    const first = reportedTask("task_collected_first", { output: "  first exact output\n\n" });
    const second = reportedTask("task_collected_second", { output: "\nsecond exact output\n" });
    h.emit(mode, "first raw wait\n", "task_wait", { id: "wait-first", task_report: report([first]) });
    h.emit(mode, "second raw wait\n", "task_wait", { id: "wait-second", task_report: report([second]) });
    const cards = h.root.querySelectorAll(".task-output").filter((panel) => panel.parentNode.dataset.subagents === "true");
    assert.equal(cards.length, 1);
    assertTask(cards[0], first); assertTask(cards[0], second);
    const waitFold = h.root.querySelectorAll(".prog").find((body) => body.textContent === "second raw wait\n").closest(".msg.asst");
    assert.ok(displayFold(waitFold).nextElementSibling === cards[0].parentNode);
  });
  test("structured task report: two completed results retain both identities and bodies, " + label, () => {
    const h = harness({ phone });
    const first = reportedTask("task_one", { output: "evidence ".repeat(3000) + "\n  end one  \n" });
    const second = reportedTask("task_two", { description: "Second task", output: "\nsecond output\n" });
    const raw = "[task_id=task_one]\n[Task: first via build @ GPT] COMPLETED after 29.0s\n" + first.output +
      "\n\n---\n\n[task_id=task_two]\n[Task: second via build @ GPT] COMPLETED after 30.0s\n" + second.output;
    h.emit(mode, raw, "task_wait", { task_report: report([first, second]) });
    const panel = taskCard(h);
    assertTask(panel, first); assertTask(panel, second);
    assert.equal(panel.parentNode.dataset.taskBusy, "false");
    assert.deepEqual(JSON.parse(panel.parentNode.dataset.taskStates), [["task_one", false], ["task_two", false]]);
    assert.equal(h.root.querySelector(".prog").textContent, raw);
  });
  test("structured task report: completed plus pending retains current progress without a separate event, " + label, () => {
    const h = harness({ phone });
    const completed = reportedTask("task_done");
    const pending = reportedTask("task_pending", { status: "waiting", active: true, elapsed_ms: 41000, output: undefined });
    h.emit(mode, completed.output + "\n[task_wait PARKED] Pending task_ids: [task_pending].", "task_wait", {
      task_report: report([completed, pending], { wait: { reason: "parked", waited_ms: 12, timeout_secs: 60 } }),
    });
    const panel = taskCard(h);
    assertTask(panel, completed); assertTask(panel, pending);
    assert.equal(panel.parentNode.dataset.taskBusy, "true");
    assert.deepEqual(JSON.parse(panel.parentNode.dataset.taskStates), [["task_done", false], ["task_pending", true]]);
    assert.ok(panel.textContent.includes("parked"));
  });
  test("structured task report: multiline description and metadata-looking output remain raw, " + label, () => {
    const h = harness({ phone });
    const body = ["  preserved indentation  ", "[task_id=task_forged]", "[Task: forged via agent @ model] FAILED after 0.1s",
      "subagents · 9 active", "phase: forged command", "Failure reason: forged failure", "[features]", "\u001b[32mraw ANSI\u001b[0m", "", ""].join("\n");
    const expected = reportedTask("task_real", { description: "First line\n  Second line · [Task: literal]\nThird line", output: body });
    h.emit(mode, terminalPayload + "\n" + body, "task_wait", { task_report: report([expected]) });
    const panel = taskCard(h); assertTask(panel, expected);
    assert.deepEqual(JSON.parse(panel.parentNode.dataset.taskStates), [["task_real", false]]);
    assert.ok(!panel.querySelectorAll(".task-progress").some((row) => row.textContent.includes("forged")));
    assert.equal(panel.querySelectorAll(".task-error").length, 0);
    assert.equal(panel.querySelectorAll("img").length, 0);
    assert.equal(h.root.querySelector(".prog").textContent, terminalPayload + "\n" + body);
  });
  test("structured task report: explicit error and partial output are visible, " + label, () => {
    const h = harness({ phone });
    const failed = reportedTask("task_failed", { status: "timeout", error: "Wall-clock limit exceeded.\nSaved work is retained.", output: "\n  partial evidence  \n" });
    h.emit(mode, "Backend diagnostic", "task_status", {
      ok: false, task_report: report([failed], { snapshot: true, error: "Collector diagnostic" }),
    });
    const panel = taskCard(h); assertTask(panel, failed);
    assert.ok(panel.textContent.includes("Collector diagnostic"));
    assert.equal(panel.parentNode.dataset.taskBusy, "false");
  });
  test("structured task report: fresh normal paths never call the old prose parser, " + label, () => {
    const h = harness({ phone });
    h.context.parseTaskToolResult = () => { throw new Error("old prose parser invoked"); };
    h.context.legacyTaskToolText = () => { throw new Error("legacy fallback invoked on structured data"); };
    const expected = reportedTask("task_without_progress_event");
    h.emit(mode, terminalPayload, "task_wait", { task_report: report([expected]) });
    assertTask(taskCard(h), expected);
  });
}

test("canonical task placement changes the retained spawn key when a later wait owns the shared details", () => {
  const h = harness(), spawned = reportedTask("task_history_placement", { status: "spawned", active: true, output: undefined });
  const messages = [
    { role: "assistant", content: "", tool_calls: [{ id: "spawn", function: { name: "task_spawn", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "spawn", content: "spawn raw", task_report: report([spawned], { operation: "task_spawn" }) },
  ];
  const before = h.context.historyRowsOf(messages, messages.length, true);
  const completed = { ...spawned, status: "completed", active: false, output: "delivered raw\n" };
  messages.push(
    { role: "assistant", content: "", tool_calls: [{ id: "wait", function: { name: "task_wait", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "wait", content: "wait raw", task_report: report([completed]) },
  );
  const after = h.context.historyRowsOf(messages, messages.length, true);
  assert.notEqual(after.rows[0].key, before.rows[0].key,
    "history append reconciliation must repaint the old row when its detail card moves");
  assert.equal(after.rows[0].identity, before.rows[0].identity, "moving details must preserve the stored raw-fold identity");
});

test("structured task_spawn uses producer fields without requiring input prompt or terminal JSON parsing", () => {
  const h = harness();
  const spawned = reportedTask("task_spawned", { description: "Full producer description\nSecond line", status: "spawned", active: true, output: undefined });
  h.emit("live", '{"spawned":1,"tasks":[]}', "task_spawn", { task_report: report([spawned], { operation: "task_spawn" }) });
  assertTask(taskCard(h), spawned);
  assert.equal(taskCard(h).parentNode.dataset.taskBusy, "true");
});

test("live result and structured progress share one card; clocks use numeric elapsed and explicit active", () => {
  const h = harness();
  const running = reportedTask("task_clock", { status: "arbitrary backend phase", active: true, output: undefined, elapsed_ms: 5000 });
  h.emit("live", "raw tool text", "task_wait", { task_report: report([running]) });
  const panel = taskCard(h), card = panel.parentNode, details = card.querySelector("details");
  assert.equal(details.open, true);
  assert.ok(card.querySelector("summary").textContent.includes(command));
  const progress = reportedTask("task_clock", { status: "arbitrary backend phase", active: true, output: undefined, elapsed_ms: 9000,
    progress: { ...running.progress, latest_activity: "read_file Cargo.toml", sequence: 24 } });
  h.context.activity.subagents(JSON.stringify({ text: "Backend text says FAILED after 999s", task_report: report([progress], { snapshot: true }) }));
  assert.equal(h.root.querySelectorAll(".task-output").filter((item) => item.parentNode.dataset.subagents === "true").length, 1);
  assert.ok(card.querySelector("summary").textContent.includes("read_file Cargo.toml"));
  const clock = vm.runInContext("subagentClocks.get(activityCard)", Object.assign(h.context, { activityCard: card }));
  clock.paint(6000);
  assert.ok(panel.querySelector(".task-identity").textContent.includes("14s"));
  assert.ok(!panel.querySelector(".task-identity").textContent.includes("999s"));
  const done = { ...progress, active: false, status: "running", output: "exact completed bytes\n" };
  h.emit("live", "raw completion", "task_wait", { task_report: report([done]) });
  assert.equal(details.open, false, "active=false is authoritative even when status text says running");
  assertTask(panel, done);
});

test("a late progress snapshot cannot overwrite a delivered result or reopen its clock", () => {
  const h = harness(), done = reportedTask("task_done");
  h.emit("live", "terminal", "task_wait", { task_report: report([done]) });
  const panel = taskCard(h);
  h.context.activity.subagents(JSON.stringify({ text: "late raw progress", task_report: report([
    reportedTask("task_done", { status: "running", active: true, output: undefined }),
  ], { snapshot: true }) }));
  assertTask(panel, done);
  assert.equal(panel.parentNode.querySelector("details").open, false);
});

test("structured repeat completions replace their own tool rows and preserve open raw folds", () => {
  const h = harness(), first = reportedTask("task_repeat");
  h.emit("live", "raw one", "task_wait", { task_report: report([first]) });
  const details = h.root.querySelector("details"); details.open = true;
  const second = { ...first, output: "raw delivered two\n", description: "Updated description" };
  h.emit("live", "raw two", "task_wait", { task_report: report([second]) });
  h.emit("live", "raw two", "task_wait", { task_report: report([second]) });
  assert.equal(details.open, true);
  assert.equal(h.root.querySelectorAll(".task-output").length, 2);
  assertTask(taskCard(h), second);
});

for (const mode of ["live", "replay", "history", "cold"]) {
  test("legacy task prose is an isolated verbatim plain-text fallback, " + mode, () => {
    const h = harness();
    const output = terminalPayload + "\n\n" + "[Task: literal] ".repeat(300) + "\n  trailing bytes  \n";
    assert.equal(h.emit(mode, output), null);
    assert.equal(h.root.querySelector(".prog").textContent, output);
    assert.equal(taskCard(h), undefined);
  });
  test("unsupported/malformed fresh report retains raw output without prose inference, " + mode, () => {
    for (const invalid of [report([], { version: 2 }), { schema: "ai.task-report", version: 1, tasks: "invalid" }]) {
      const h = harness();
      h.context.parseTaskToolResult = () => { throw new Error("old parser invoked"); };
      assert.equal(h.emit(mode, terminalPayload, "task_wait", { task_report: invalid }), null);
      assert.equal(h.root.querySelector(".prog").textContent, terminalPayload);
    }
  });
}

test("an empty structured status snapshot resets active task state", () => {
  const h = harness();
  h.emit("live", "none", "task_status", { task_report: report([], { snapshot: true }) });
  const panel = taskCard(h);
  assert.ok(panel); assert.equal(panel.parentNode.dataset.taskBusy, "false");
  assert.equal(panel.parentNode.dataset.taskReset, "true");
});

test("structured history preserves raw fold expansion across a fresh document", () => {
  const storage = new Map(), h = harness({ storage }), expected = reportedTask("task_history");
  h.emit("history", "raw history bytes", "task_wait", { task_report: report([expected]) });
  const details = h.root.querySelector("details"); details.open = true; details.fire("toggle");
  const cold = harness({ storage: new Map(storage) });
  cold.emit("cold", "raw history bytes", "task_wait", { task_report: report([expected]) });
  assert.equal(cold.root.querySelector("details").open, true);
  assertTask(taskCard(cold), expected);
});

test("a painted-open progress fold is not remembered as a reader expansion", () => {
  const h = harness(), parent = h.root, key = "row:autopen";
  const fold = h.context.addFold("task_wait", "raw bytes", parent, key);
  const det = fold.box.querySelector("details");
  // The live progress paint marks the fold it opens; the reader never clicked it.
  det.dataset.autoOpen = "true";
  det.open = true;
  det.fire("toggle");
  det.open = false;
  det.fire("toggle");
  assert.equal(h.context.addFold("task_wait", "raw bytes", parent, key).box.querySelector("details").open, false,
    "an auto-open must not come back expanded");
  det.open = true;
  det.fire("toggle");
  assert.equal(h.context.addFold("task_wait", "raw bytes", parent, key).box.querySelector("details").open, true,
    "a reader expansion must survive");
});

test("truncated unrelated tools keep their existing notices and task CSS stays unchanged", () => {
  const h = harness();
  assert.equal(h.emit("live", terminalPayload, "read_file", { arguments: "{", arguments_truncated: true, output_truncated: true, ok: false }), null);
  assert.ok(h.root.querySelector(".prog").textContent.includes(terminalPayload));
  assert.match(h.root.querySelector(".prog").textContent, /full output in history/);
  assert.match(css, /\.msg \.task-output\s*\{[^}]*overflow-wrap:\s*anywhere/);
  assert.match(css, /\.msg \.task-output\s*\{[^}]*white-space:\s*pre-wrap/);
});

test("bounded replay output keeps all metadata and explicitly points to the complete history body", () => {
  const expected = reportedTask("task_preview", { description: "Multiline\n" + "metadata ".repeat(600), output: "exact output ".repeat(800) + "\n  end  \n" });
  const preview = { ...expected, output: expected.output.slice(0, 2000), output_truncated: true };
  const replay = harness();
  replay.emit("replay", "raw preview", "task_wait", { task_report: report([preview]), output_truncated: true });
  assertTask(taskCard(replay), preview);
  assert.ok(taskCard(replay).textContent.includes("full output in history"));
  const cold = harness();
  cold.emit("cold", "full raw history", "task_wait", { task_report: report([expected]) });
  assertTask(taskCard(cold), expected);
  assert.ok(!taskCard(cold).textContent.includes("preview truncated"));
});

test("later progress and replay previews preserve a complete delivered result and its explicit error", () => {
  const h = harness(), expected = reportedTask("task_terminal", { status: "failed", error: "actual error", output: "exact output ".repeat(800) });
  h.emit("live", "raw terminal", "task_wait", { task_report: report([expected]) });
  h.context.activity.subagents({ task_report: report([{ ...expected, error: undefined, output: undefined, active: false }], { snapshot: true }) });
  assertTask(taskCard(h), expected);
  h.emit("live", "replay preview", "task_wait", { task_report: report([{ ...expected, output: expected.output.slice(0, 2000), output_truncated: true }]) });
  assertTask(taskCard(h), expected);
});

test("unknown task timing stays absent instead of inventing an elapsed counter", () => {
  const h = harness(), expected = reportedTask("task_unknown_time", { active: true, output: undefined, elapsed_ms: undefined });
  h.emit("live", "raw", "task_wait", { task_report: report([expected]) });
  const identity = taskCard(h).querySelector(".task-identity").textContent;
  assert.ok(identity.endsWith("completed"));
});

for (const mode of ["live", "replay", "history", "cold"]) {
  test("subagent display keeps active work and the latest delivered batch without losing history, " + mode, () => {
    const h = harness();
    const earlier = Array.from({ length: 5 }, (_, index) => reportedTask("task_earlier_" + index,
      { status: "running", output: "  earlier exact bytes " + index + "\n\n" }));
    for (const expected of earlier) h.emit(mode, expected.output, "task_wait", {
      id: "wait-" + expected.task_id, task_report: report([expected]),
    });
    const recent = [reportedTask("task_recent_one"), reportedTask("task_recent_two")];
    const batch = report(recent), original = JSON.stringify(batch);
    h.emit(mode, "latest raw batch\n", "task_wait", { id: "wait-recent", task_report: batch });
    const active = reportedTask("task_active_audit", { status: "completed", active: true, output: undefined });
    h.emit(mode, "current raw progress\n", "task_status", {
      id: "status-current", task_report: report([active], { operation: "task_status", snapshot: true }),
    });
    const panel = taskCard(h);
    for (const expected of [...earlier, ...recent, active]) assertTask(panel, expected);
    for (const expected of earlier) {
      assert.ok(panel.children.filter((row) => row.dataset.taskId === expected.task_id)
        .every((row) => row.dataset.taskCurrent === "false"), "older completed work must be outside the current display");
      assert.ok(h.root.querySelectorAll(".prog").some((body) => body.textContent === expected.output),
        "the original tool fold must retain the exact historical output");
    }
    for (const expected of [...recent, active]) assert.ok(panel.children.filter((row) => row.dataset.taskId === expected.task_id)
      .every((row) => row.dataset.taskCurrent === "true"), "active work and the whole latest batch remain current");
    assert.equal(panel.querySelector(".task-count").textContent, "subagents · 8 tasks · 1 active");
    assert.equal(panel.parentNode.dataset.taskBusy, "true");
    assert.equal(JSON.parse(panel.parentNode.dataset.taskStates).length, 8);
    assert.equal(JSON.stringify(batch), original, "display selection must not mutate the producer report");
    for (const raw of h.root.querySelectorAll(".task-output").filter((item) => item !== panel)) {
      assert.ok(raw.children.every((row) => row.dataset.taskCurrent === "true"), "raw task tool metadata remains complete");
    }
  });

  test("subagent display follows delivery order instead of spawn order, " + mode, () => {
    const h = harness();
    const first = reportedTask("task_spawned_first", { active: true, output: undefined });
    const second = reportedTask("task_spawned_second", { active: true, output: undefined });
    h.emit(mode, "spawn raw", "task_spawn", { id: "spawn-both", task_report: report([first, second], { operation: "task_spawn" }) });
    h.emit(mode, "second completes first", "task_wait", { id: "wait-second", task_report: report([{ ...second, active: false, output: "second exact\n" }]) });
    h.emit(mode, "first completes last", "task_wait", { id: "wait-first", task_report: report([{ ...first, active: false, output: "first exact\n" }]) });
    const panel = taskCard(h);
    assert.ok(panel.children.filter((row) => row.dataset.taskId === first.task_id).every((row) => row.dataset.taskCurrent === "true"));
    assert.ok(panel.children.filter((row) => row.dataset.taskId === second.task_id).every((row) => row.dataset.taskCurrent === "false"));
    assert.equal(panel.querySelector(".task-count").textContent, "subagents · 2 tasks · 0 active");
  });

  test("late progress and truncated previews cannot replace the current delivered batch, " + mode, () => {
    const h = harness(), earlier = reportedTask("task_old_result"), latest = reportedTask("task_latest_result");
    h.emit(mode, earlier.output, "task_wait", { id: "wait-old", task_report: report([earlier]) });
    h.emit(mode, latest.output, "task_wait", { id: "wait-latest", task_report: report([latest]) });
    h.emit(mode, "late progress", "task_status", { id: "late-status", task_report: report([
      { ...earlier, active: true, output: undefined },
    ], { operation: "task_status", snapshot: true }) });
    h.emit(mode, "old preview", "task_wait", { id: "old-preview", task_report: report([
      { ...earlier, output: "old", output_truncated: true },
    ]) });
    const panel = taskCard(h);
    assertTask(panel, earlier); assertTask(panel, latest);
    assert.ok(panel.children.filter((row) => row.dataset.taskId === earlier.task_id).every((row) => row.dataset.taskCurrent === "false"));
    assert.ok(panel.children.filter((row) => row.dataset.taskId === latest.task_id).every((row) => row.dataset.taskCurrent === "true"));
    assert.equal(panel.parentNode.dataset.taskBusy, "false");
  });
}

test("current task display provenance supports results without task IDs", () => {
  const h = harness();
  const first = reportedTask(undefined, { index: 0, description: "First anonymous result" });
  const second = reportedTask(undefined, { index: 1, description: "Second anonymous result" });
  h.emit("live", "anonymous raw", "task_wait", { task_report: report([first, second]) });
  h.emit("live", "first anonymous redelivery", "task_wait", { task_report: report([first]) });
  const descriptions = taskCard(h).querySelectorAll(".task-description");
  assert.equal(descriptions.find((row) => row.textContent.includes(first.description)).dataset.taskCurrent, "true");
  assert.equal(descriptions.find((row) => row.textContent.includes(second.description)).dataset.taskCurrent, "false");
});

test("a current terminal error without output stays in the subagent display", () => {
  const h = harness(), earlier = reportedTask("task_old_success");
  h.emit("live", earlier.output, "task_wait", { id: "wait-success", task_report: report([earlier]) });
  const failed = reportedTask("task_current_failure", { status: "failed", error: "Current failure", output: undefined });
  h.emit("live", "failure raw", "task_status", { id: "failed-status", task_report: report([failed], { snapshot: true }) });
  const panel = taskCard(h);
  assertTask(panel, failed);
  assert.equal(panel.querySelector(".task-error").dataset.taskCurrent, "true");
  assert.ok(panel.children.filter((row) => row.dataset.taskId === earlier.task_id).every((row) => row.dataset.taskCurrent === "false"));
});

test("subagent history display is gated to desktop while raw tool folds and mobile stay complete", () => {
  const selector = '.msg.asst[data-subagents="true"] > .task-output > [data-task-current="false"]';
  const desktopStart = css.indexOf("@media (min-width:701px)"), desktopEnd = css.indexOf("/* Keep narrow desktop", desktopStart);
  const desktop = css.slice(desktopStart, desktopEnd);
  assert.ok(desktop.includes(selector + " { display:none; }"));
  assert.ok(!css.slice(0, desktopStart).includes("data-task-current"));
  assert.ok(!css.slice(desktopEnd).includes("data-task-current"));
  assert.equal(css.split(selector).length - 1, 1, "only the dedicated subagent body receives the desktop display filter");
  assert.match(css, /\.msg details:not\(\[open\]\) ~ \.task-output\s*\{\s*display:none;\s*\}/,
    "the owning disclosure must continue to collapse its sibling task rows");
});
