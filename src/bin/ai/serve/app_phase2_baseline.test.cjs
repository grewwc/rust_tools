// Run with: env NODE_OPTIONS= node --test --test-reporter=tap src/bin/ai/serve/app_phase2_baseline.test.cjs
// Characterize the current worktree. Quirk cases document observations, not desired contracts.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");

function loadFixtures(file, names) {
  const filename = path.join(__dirname, file);
  // Reuse the handwritten DOM and VM helpers without registering their tests again.
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : require(name),
    __dirname, __filename: filename, console, Buffer,
    process: { env: { ...process.env, ASERVE_TEST_APP_HTML: process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html") } },
    TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
    setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
  });
}

const stream = loadFixtures("app_stream.test.cjs", [
  "script", "streamHarness", "replayHarness", "retainedHistoryHarness", "descendants",
  "nodeList", "structuredTask", "structuredSubagents", "rateNodes",
]);
const desktop = loadFixtures("app_desktop.test.cjs", ["harness", "loadChat"]);
const errors = loadFixtures("app_error_rows.test.cjs", ["harness"]);
const team = loadFixtures("app_team_mode.test.cjs", ["harness", "NORMAL", "TRAECLI", "response"]);

test("owner regression: adoptComposerSession preserves the exact HEAD function contract", () => {
  const start = stream.script.indexOf("function adoptComposerSession(");
  const end = stream.script.indexOf("\n}", start) + 2;
  assert.ok(start >= 0 && end > start);
  const source = stream.script.slice(start, end), body = source.slice(source.indexOf("{"));
  // HEAD 18e9fbc3, measured by the independent contracts.cjs checker.
  assert.equal(source.slice(0, source.indexOf("{")), "function adoptComposerSession(newId, previousId, base) ");
  assert.equal(Buffer.byteLength(body), 425);
  assert.equal(crypto.createHash("sha256").update(body).digest("hex"),
    "b17cffd76ff46aa23f5e3675485a4e06d445c6812bef9580fcece60abb76f006");
});

function pageSlice(start, end) {
  const from = stream.script.indexOf(start), to = stream.script.indexOf(end, from);
  assert.ok(from >= 0 && to > from, "missing page slice: " + start);
  return stream.script.slice(from, to);
}
function runSlice(context, start, end) {
  vm.runInContext(pageSlice(start, end), context);
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
async function settle() {
  for (let i = 0; i < 32; i++) await Promise.resolve();
}
function answerText(node) {
  if (node.innerHTML) return node.innerHTML;
  let text = node.textContent;
  for (const rate of Array.from(stream.rateNodes(node))) {
    if (text.endsWith(rate.textContent)) text = text.slice(0, -rate.textContent.length);
  }
  return text;
}
function records(root) {
  return Array.from(stream.descendants(root)).filter((node) => node.classList?.contains("msg")
    && !node.hidden && !node.classList.contains("pending") && !node.classList.contains("execution-group"));
}
function foldParts(node) {
  return { summary: node.summary || node.querySelector?.("summary"),
    body: node.body || node.querySelector?.(".prog") };
}
function projection(root) {
  return records(root).flatMap((node) => {
    const { summary, body } = foldParts(node);
    if (!summary) {
      const text = answerText(node).trim();
      return text ? [{ role: "assistant", text }] : [];
    }
    const label = summary.textContent.replace(/^[◌!✓•]\s*/, "");
    const state = node.dataset.executionState;
    const error = node.dataset.executionException === "true";
    if (node.dataset.subagents === "true") return [{ role: "subagents", state, error,
      tasks: JSON.parse(node.dataset.taskStates || "[]") }];
    if (node.dataset.thinkingPhase === "true" || /^thinking(?: ·|$)/.test(label)) {
      // Compare the common visible tail; assert each source's complete body separately.
      return [{ role: "thinking", text: body.textContent.slice(-6000), state, error }];
    }
    return [{ role: "tool", label,
      text: body.textContent.replace(/^Arguments\n[^\n]*\n\n/, "").trim(), state, error }];
  });
}
function toolNode(root, name) {
  const node = records(root).find((item) => foldParts(item).summary?.textContent.includes(name));
  assert.ok(node, "missing real tool record: " + name);
  return node;
}
function replayAndLive() {
  return [stream.streamHarness({ realThinking: true }), stream.replayHarness({ realThinking: true })];
}

const prose = "Planning segment one. Planning segment two.";
const thought = "Opening thought.\n" + "reason ".repeat(950) + "\nClosing thought.";
const finalAnswer = "Final answer after replacement.";
const rawTask = "\n  task bytes\r\nresult line\n\n";
const calls = [
  { id: "read-1", type: "function", function: { name: "read_file", arguments: JSON.stringify({ path: "README.md" }) } },
  { id: "shell-1", type: "function", function: { name: "shell", arguments: JSON.stringify({ command: "rg baseline" }) } },
  { id: "wait-1", type: "function", function: { name: "task_wait", arguments: JSON.stringify({ task_id: "worker" }) } },
];
const activeTasks = [stream.structuredTask("worker", 5000), stream.structuredTask("reviewer", 6000)];
const completedTasks = activeTasks.map((task) => ({ ...task, status: "completed", active: false, delivered: true,
  output: task.task_id + " result" }));
const completedReport = { ...stream.structuredSubagents(completedTasks).task_report, operation: "task_wait" };
const trace = [
  ["delta", { delta: "Planning segment one. " }],
  ["delta", { delta: "Planning segment two." }],
  ["thinking_start", {}],
  ["thinking", { text: thought.slice(0, 3100) }],
  ["thinking", { text: thought.slice(3100) }],
  ["thinking_done", {}],
  ...calls.map((call) => ["tool", { id: call.id, name: call.function.name, phase: "started", arguments: call.function.arguments }]),
  ["subagent_progress", stream.structuredSubagents(activeTasks, "Workers are running.")],
  // Parallel completions update their original start-order cards.
  ["tool", { id: "shell-1", name: "shell", phase: "finished", ok: true, output: "Found baseline." }],
  ["tool", { id: "read-1", name: "read_file", phase: "finished", ok: true, output: "README contents." }],
  ["tool", { id: "wait-1", name: "task_wait", phase: "finished", ok: true, output: rawTask, task_report: completedReport }],
  ["subagent_progress", stream.structuredSubagents(completedTasks, "Workers are complete.")],
  ["delta", { delta: "Draft answer." }],
  ["delta", { delta: " Draft tail." }],
  ["replace", { text: finalAnswer }],
];
const historyMessages = [
  { role: "assistant", content: prose },
  { role: "assistant", content: "", reasoning_content: thought, tool_calls: calls },
  { role: "tool", name: "read_file", tool_call_id: "read-1", content: "README contents." },
  { role: "tool", name: "shell", tool_call_id: "shell-1", content: "Found baseline." },
  { role: "tool", name: "task_wait", tool_call_id: "wait-1", content: rawTask, task_report: completedReport },
  { role: "assistant", content: finalAnswer },
];

test("three paths: prose, thinking, parallel tools, task snapshots, replace and completion keep logical order", async () => {
  const [live, replay] = replayAndLive();
  for (const [kind, value] of trace) { live.emit(kind, value); replay.emit(kind, value); }
  live.emit("done", {});
  replay.view.finish(); // attachEvents invokes this exact hook on its terminal frame.
  const history = stream.retainedHistoryHarness(historyMessages);
  await history.c.loadHistory();
  const expected = [
    { role: "assistant", text: prose },
    { role: "thinking", text: thought.slice(-6000), state: "completed", error: false },
    { role: "tool", label: "read_file README.md", text: "README contents.", state: "completed", error: false },
    { role: "tool", label: "shell rg baseline", text: "Found baseline.", state: "completed", error: false },
    { role: "tool", label: "task_wait · raw", text: rawTask.trim(), state: "completed", error: false },
    { role: "subagents", state: "completed", error: false, tasks: [["worker", false], ["reviewer", false]] },
    { role: "assistant", text: finalAnswer },
  ];
  for (const [source, root, context] of [
    ["live", live.box, live.context], ["replay", replay.view.box, replay.context], ["history", history.root, history.c],
  ]) {
    assert.deepEqual(projection(root), expected, source + " logical transcript");
    assert.ok(Array.from(stream.descendants(root)).some((node) => node.dataset.taskReset === "true"), source + " terminal marker");
    assert.deepEqual(Array.from(context.executionState().taskStates), [], source + " final answer settles the ledger");
  }
  assert.equal(live.context.finished, true);
  assert.equal(history.c.canonicalHistoryTotal, historyMessages.length);
  assert.equal(history.root._histWin, historyMessages.length);
});

test("three paths: history omits Arguments and replay caps thinking while live retains the complete raw body", async () => {
  const [live, replay] = replayAndLive();
  for (const [kind, value] of trace) { live.emit(kind, value); replay.emit(kind, value); }
  live.emit("done", {}); replay.view.finish();
  const history = stream.retainedHistoryHarness(historyMessages);
  await history.c.loadHistory();
  for (const [call, output] of [[calls[0], "README contents."], [calls[1], "Found baseline."], [calls[2], rawTask]]) {
    assert.equal(foldParts(toolNode(live.box, call.function.name)).body.textContent, "Arguments\n" + call.function.arguments + "\n\n" + output);
    assert.equal(foldParts(toolNode(replay.view.box, call.function.name)).body.textContent, "Arguments\n" + call.function.arguments + "\n\n" + output);
    assert.equal(foldParts(toolNode(history.root, call.function.name)).body.textContent, call.function.name === "task_wait" ? output : output.trim());
  }
  const thinkingBody = (root) => foldParts(records(root).find((node) => foldParts(node).summary?.textContent.startsWith("thinking"))).body.textContent;
  assert.equal(thinkingBody(live.box), thought);
  assert.equal(thinkingBody(history.root), thought);
  assert.equal(thinkingBody(replay.view.box), "…" + thought.slice(-6000));
  assert.equal(thinkingBody(replay.view.box).length, 6001);
});

function renderingHarness(options) {
  const h = desktop.harness(options);
  desktop.loadChat(h);
  runSlice(h.context, "function toolArgsOf(", "// Which stored folds");
  runSlice(h.context, "function formatSubagentElapsed(", "function createSubagentClock(");
  return h;
}
function mountRows(h, messages) {
  const rows = h.context.historyRowsOf(messages, messages.length, true).rows;
  for (const row of rows) h.context.renderHistoryRow(row, h.elements.msgs);
  return rows;
}

test("quirk: ordinary history output is trimmed and silently cut at 2000 characters; live/replay honor the preview hint", () => {
  const raw = "\n  " + "x".repeat(2100) + "TAIL\r\n ";
  const h = renderingHarness(), call = calls[0];
  mountRows(h, [{ role: "assistant", content: "", tool_calls: [call] },
    { role: "tool", content: raw, tool_call_id: call.id }]);
  const body = foldParts(toolNode(h.elements.msgs, "read_file")).body.textContent;
  assert.equal(body, raw.trim().slice(0, 2000));
  assert.equal(body.length, 2000);
  // Quirk: the history renderer drops the tail without a truncation notice.
  assert.doesNotMatch(h.elements.msgs.textContent, /truncated|full output in history|TAIL/);
  for (const path of replayAndLive()) {
    path.emit("tool", { id: call.id, name: call.function.name, arguments: call.function.arguments, phase: "started" });
    path.emit("tool", { id: call.id, name: call.function.name, phase: "finished", output: "preview", ok: true, output_truncated: true });
    const root = path.box || path.view.box;
    assert.equal(foldParts(toolNode(root, "read_file")).body.textContent,
      "Arguments\n" + call.function.arguments + "\n\npreview\n[preview truncated; full output in history]");
  }
});

test("history task tools preserve raw bytes beyond the ordinary output limit with and without a typed report", () => {
  const raw = "\n  task raw\r\n" + "z".repeat(2100) + "\n END \n\n";
  for (const report of [undefined, completedReport]) {
    const h = renderingHarness(), call = calls[2];
    mountRows(h, [{ role: "assistant", content: "", tool_calls: [call] },
      { role: "tool", content: raw, tool_call_id: call.id, task_report: report }]);
    assert.equal(foldParts(toolNode(h.elements.msgs, "task_wait")).body.textContent, raw);
  }
});

test("snapshot history routes canonical rows through the shared projection without stream events or live activity", async () => {
  const messages = [{ role: "user", content: "question" }, ...historyMessages,
    { role: "internal_note", content: "stored note" }];
  const h = stream.retainedHistoryHarness(messages), c = h.c;
  assert.equal(typeof c.dispatchExecutionSnapshot, "function", "history must expose its materialized snapshot projection");
  assert.equal(typeof c.dispatchExecutionProjection, "function", "events and snapshots must share the projection entry point");
  const table = vm.runInContext("EXECUTION_SNAPSHOT_HANDLERS", c);
  const snapshots = [], projected = [], updates = [];
  const snapshot = c.dispatchExecutionSnapshot, project = c.dispatchExecutionProjection, update = c.updateExecutionModel;
  c.dispatchExecutionSnapshot = (row, view, source) => {
    snapshots.push({ row, source });
    return snapshot(row, view, source);
  };
  c.dispatchExecutionProjection = (handlers, kind, value, view, source) => {
    projected.push({ handlers, kind, value, source });
    return project(handlers, kind, value, view, source);
  };
  c.updateExecutionModel = (event, ...args) => { updates.push(event); return update(event, ...args); };
  c.decodeExecutionEvent = () => assert.fail("materialized history must not decode a fabricated wire event");
  c.dispatchExecutionEvent = () => assert.fail("materialized history must not dispatch started/delta/done events");
  c.createLiveActivity = () => assert.fail("materialized history must not acquire live activity ownership");
  await c.loadHistory();
  assert.equal(snapshots.length, messages.length);
  assert.deepEqual(projected.map((item) => item.kind), ["user", "assistant", "assistant", "fold", "fold", "fold", "assistant", "fold"]);
  for (const [index, item] of snapshots.entries()) {
    assert.equal(item.source, "history");
    assert.equal(item.row.message, messages[index], "the canonical message remains the snapshot source");
    assert.equal(item.row.historyIndex, index);
    assert.equal(projected[index].handlers, table);
    assert.equal(projected[index].value, item.row, "projection retains canonical identity and merged task fields");
    assert.equal(projected[index].source, "history");
  }
  assert.deepEqual(updates.map((event) => event.kind), ["history", "display"]);
  assert.deepEqual(JSON.parse(JSON.stringify(updates[0].summaries)),
    JSON.parse(JSON.stringify(c.historyExecutionOf(snapshots.map((item) => item.row)))));
  const nodes = [h.root, ...stream.descendants(h.root)];
  await c.loadHistory();
  assert.equal(snapshots.length, messages.length, "unchanged canonical rows refresh without another materialization");
  assert.deepEqual([h.root, ...stream.descendants(h.root)], nodes);
  assert.deepEqual(updates.map((event) => event.kind), ["history", "display", "history", "display"], "each load commits canonical summaries once");
  assert.equal(c.canonicalHistoryTotal, messages.length);

  for (const source of ["live", "replay"]) {
    const live = source === "live" ? stream.streamHarness() : stream.replayHarness();
    const context = live.context, calls = [], dispatch = context.dispatchExecutionProjection;
    const events = vm.runInContext("EXECUTION_EVENT_HANDLERS", context);
    context.dispatchExecutionProjection = (...args) => { calls.push(args); return dispatch(...args); };
    live.emit("delta", { delta: "wire answer" });
    assert.equal(calls.length, 1);
    assert.equal(calls[0][0], events);
    assert.equal(calls[0][1], "delta");
    assert.equal(calls[0][2].parsed, true);
    assert.equal(calls[0][4], source);
    const root = live.box || live.view.box, before = projection(root), original = Array.from(stream.descendants(root));
    live.emit("snapshot", { role: "assistant", text: "not a wire event" });
    assert.equal(calls.length, 1, "snapshot projection does not add a server event kind");
    assert.deepEqual(projection(root), before);
    assert.deepEqual(Array.from(stream.descendants(root)), original);
  }
});

for (const phone of [false, true]) test("snapshot history preserves " + (phone ? "phone" : "desktop") + " source text, reasoning, rates and frozen task observations", () => {
  const h = renderingHarness({ phone }), c = h.context;
  assert.equal(typeof c.dispatchExecutionSnapshot, "function", "history must use the snapshot projection");
  c.Date = { now() { assert.fail("history must not age a materialized task or reasoning observation"); } };
  c.setInterval = () => assert.fail("history must not start an elapsed clock");
  c.setTimeout = () => assert.fail("history snapshots must not schedule live paint");
  c.createLiveActivity = () => assert.fail("history snapshots must not create activity slots");
  const prefix = "\u001eaios:reasoning-encrypted-replay:v1\u001f";
  const typed = "typed one\n\ntyped two", plain = "ordinary reasoning";
  const ordinary = "\n  " + "o".repeat(2100) + "ORDINARY TAIL\r\n ";
  const taskRaw = "\n  " + "t".repeat(2100) + "TASK TAIL\r\n\n";
  const taskOutput = "\n  retained task output\r\n\n";
  const report = { ...stream.structuredSubagents([{ ...stream.structuredTask("snapshot-worker", 5000),
    output: taskOutput, output_truncated: true }]).task_report, operation: "task_wait" };
  const messages = [
    { role: "assistant", content: "", tool_calls: calls },
    { role: "assistant", content: "", tool_calls: calls, reasoning_content: prefix + JSON.stringify({ items: [
      { type: "reasoning", encrypted_content: "ENCRYPTED BYTES", summary: [
        { type: "summary_text", text: "  typed one \n" }, { type: "text", text: "untyped secret" },
        { type: "summary_text", text: "\n typed two  " },
      ] }, { type: "message", summary: [{ type: "summary_text", text: "wrong item" }] },
    ] }) },
    { role: "assistant", content: "visible metadata holder", tool_calls: calls, reasoning_content: " runtime-origin:internal metadata " },
    { role: "assistant", content: "", tool_calls: calls, reasoning_content: "\n  " + plain + " \r\n" },
    { role: "assistant", content: "", reasoning_content: prefix + "{broken" },
    { role: "tool", content: ordinary, tool_call_id: calls[0].id },
    { role: "tool", content: taskRaw, tool_call_id: calls[2].id, task_report: report },
    { role: "assistant", content: "\n final snapshot answer\n\n" },
  ];
  c.storedRate = (kind, sid, text) => {
    assert.equal(sid, "open-session");
    return kind === "thinking" && text === typed ? "~12 tok @ 6 tok/s"
      : kind === "turn" && text === "final snapshot answer" ? "output 20 tok @ 10 tok/s" : "";
  };
  const rows = c.historyRowsOf(messages, 50, true).rows;
  for (const row of rows) c.renderHistoryRow(row, h.elements.msgs);
  assert.deepEqual(Array.from(rows, (row) => row.historyIndex), [43, 44, 45, 47, 48, 49]);
  assert.deepEqual(Array.from(rows, (row) => row.reasoning), [typed, "", plain, "", "", ""]);
  assert.equal(rows[3].call, calls[0], "the filtered call message still supplies the canonical lookup");
  assert.equal(foldParts(toolNode(h.elements.msgs, "read_file")).body.textContent, ordinary.trim().slice(0, 2000));
  assert.equal(foldParts(toolNode(h.elements.msgs, "task_wait")).body.textContent, taskRaw);
  assert.doesNotMatch(h.elements.msgs.textContent, /Arguments\n|ENCRYPTED BYTES|untyped secret|wrong item|ORDINARY TAIL|runtime-origin:/);
  const thoughts = records(h.elements.msgs).filter((node) => foldParts(node).summary?.textContent.startsWith("thinking"));
  assert.deepEqual(thoughts.map((node) => foldParts(node).body.textContent), [typed, plain]);
  assert.equal(foldParts(thoughts[0]).summary.textContent, "thinking · ~12 tok @ 6 tok/s");
  assert.equal(foldParts(thoughts[1]).summary.textContent, "thinking · ~" + c.estimateTextTokens(plain) + " tok");
  const card = records(h.elements.msgs).find((node) => node.dataset.subagents === "true");
  assert.equal(card.dataset.executionState, "completed", "a busy history report is still a frozen card");
  assert.equal(card.dataset.taskBusy, "true");
  assert.equal(card.dataset.executionSettled, "true");
  assert.equal(card.querySelector(".task-output-body").textContent, taskOutput);
  assert.match(card.textContent, /5s|output preview truncated; full output in history/);
  assert.match(card.textContent, /\[output preview truncated; full output in history\]/);
  const terminal = h.elements.msgs.lastElementChild;
  assert.equal(terminal.dataset.taskReset, "true");
  assert.equal(terminal.dataset.canonicalIndex, "49");
  assert.equal(terminal.dataset.canonicalKey, rows.at(-1).key);
  assert.equal(terminal.dataset.canonicalIdentity, rows.at(-1).identity);
  assert.match(terminal.textContent, /output 20 tok @ 10 tok\/s/);
});

for (const kind of ["delta", "replace"]) test("malformed " + kind + " JSON: live falls back to raw payload and replay ignores it", () => {
  const [live, replay] = replayAndLive();
  live.emit("delta", { delta: "seed" }); replay.emit("delta", { delta: "seed" });
  const malformed = kind === "delta" ? '{"delta":' : '{"text":';
  live.emit(kind, malformed); replay.emit(kind, malformed);
  assert.equal(live.reply(), kind === "delta" ? "seed" + malformed : malformed);
  assert.equal(replay.reply(), "seed");
  assert.equal(projection(live.box).length, 1);
  assert.equal(projection(replay.view.box).length, 1);
});

test("unknown kind and kindless stdout leave the current live/replay transcript layout unchanged", () => {
  for (const path of replayAndLive()) {
    path.emit("delta", { delta: "visible answer" });
    const root = path.box || path.view.box, before = projection(root), nodes = Array.from(stream.descendants(root));
    for (const [kind, payload] of [["future_kind", "unknown payload"], ["", "plain child stdout"], ["", ""]]) {
      path.emit(kind, payload);
      assert.deepEqual(projection(root), before);
      assert.deepEqual(Array.from(stream.descendants(root)), nodes, "ignored events do not create a fold or answer");
    }
  }
});

test("duplicate tool completions upsert the same card and update output, status and preview flags", () => {
  for (const path of replayAndLive()) {
    const root = path.box || path.view.box;
    path.emit("tool", { id: "duplicate", name: "read_file", phase: "started", arguments: calls[0].function.arguments });
    const original = toolNode(root, "read_file");
    path.emit("tool", { id: "duplicate", name: "read_file", phase: "finished", output: "first completion", ok: true });
    path.emit("tool", { id: "duplicate", name: "read_file", phase: "finished", output: "second completion", ok: false, output_truncated: true });
    assert.equal(toolNode(root, "read_file"), original);
    assert.equal(records(root).filter((node) => node.classList.contains("tool-activity")).length, 1);
    assert.equal(original.dataset.executionState, "completed");
    assert.equal(original.dataset.executionException, "true");
    assert.match(foldParts(original).summary.textContent, /^! read_file README\.md$/);
    assert.match(foldParts(original).body.textContent, /second completion\n\[preview truncated; full output in history\]$/);
    path.emit("tool", { id: "duplicate", name: "read_file", phase: "finished", output: "third completion", ok: true, output_truncated: false });
    assert.equal(toolNode(root, "read_file"), original);
    assert.equal(original.dataset.executionException, "false");
    assert.doesNotMatch(foldParts(original).body.textContent, /truncated|second completion/);
  }
});

test("two identical delta texts remain two deltas in live and replay", () => {
  for (const path of replayAndLive()) {
    path.emit("delta", { delta: "same" }); path.emit("delta", { delta: "same" });
    assert.equal(path.reply(), "samesame");
    const root = path.box || path.view.box;
    assert.deepEqual(projection(root), [{ role: "assistant", text: "samesame" }]);
  }
});

function controlledReader() {
  let pending = null, reads = 0, cancellations = 0;
  const queue = [];
  const deliver = (entry) => {
    if (!pending) { queue.push(entry); return; }
    const next = pending; pending = null;
    if (entry.error) next.reject(entry.error); else next.resolve(entry.result);
  };
  return {
    read() {
      reads++;
      if (queue.length) {
        const entry = queue.shift();
        return entry.error ? Promise.reject(entry.error) : Promise.resolve(entry.result);
      }
      assert.equal(pending, null, "the transport has one outstanding read");
      pending = deferred(); return pending.promise;
    },
    cancel() { cancellations++; deliver({ result: { done: true } }); return Promise.resolve(); },
    frame(text) { assert.ok(pending, "the real transport is awaiting its next frame"); deliver({ result: { done: false, value: Buffer.from(text) } }); },
    eof() { deliver({ result: { done: true } }); },
    reject(message) { deliver({ error: new Error(message) }); },
    pending: () => pending !== null, reads: () => reads, cancellations: () => cancellations,
  };
}
function sse(kind, value) {
  const payload = typeof value === "string" ? value : JSON.stringify(value ?? {});
  return (kind ? "event: " + kind + "\n" : "") + "data: " + payload + "\n\n";
}

function sendHarness(options = {}) {
  const h = desktop.harness(options), c = h.context, timers = new Map(), replays = [], requests = [];
  let timerId = 0, now = 1000;
  const headers = deferred(), requested = deferred(), reader = controlledReader();
  const create = h.document.createElement;
  const decorate = (node) => {
    Object.defineProperty(node, "isConnected", { configurable: true, get() {
      for (let owner = node; owner; owner = owner.parentNode) if (owner === h.document.body) return true;
      return false;
    } });
    return node;
  };
  for (const node of Object.values(h.elements)) { decorate(node); for (const child of stream.descendants(node)) decorate(child); }
  h.document.createElement = (tag) => decorate(create(tag));
  h.document.body = h.document.createElement("body");
  h.document.body.appendChild(h.elements.msgs); h.document.body.appendChild(h.elements.composer);
  h.document.hidden = false; h.document.visibilityState = "visible";
  h.document.addEventListener = () => {};
  c.window.addEventListener = () => {};
  Object.assign(c, {
    Date: { now: () => now }, AbortController, TextDecoder,
    MutationObserver: class { observe() {} },
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, delay, at: now + delay, type: "timeout" }); return id; },
    clearTimeout(id) { timers.delete(id); },
    setInterval(fn, delay) { const id = ++timerId; timers.set(id, { fn, delay, at: now + delay, type: "interval" }); return id; },
    clearInterval(id) { timers.delete(id); },
    staleTimer: null, staleBox: null, staleTries: 0, staleBusy: false,
    STALE_FOLLOW_MS: Number(/const STALE_FOLLOW_MS = (\d+)/.exec(stream.script)[1]),
    STALE_FOLLOW_TRIES: Number(/const STALE_FOLLOW_TRIES = (\d+)/.exec(stream.script)[1]),
    REMOTE_RUNNING_TTL_MS: 30000, confirmPollTimer: null, confirmPollTick: 0,
    eventAbort: null, recoveredView: null, eventFailures: 0, confirmShown: null,
    NATIVE_PREFIX: JSON.parse(/const NATIVE_PREFIX = ("[^"\n]*");/.exec(stream.script)[1]),
    addAnswerCopy() {},
  });
  desktop.loadChat(h);
  runSlice(c, "function toolArgsOf(", "// Which stored folds");
  runSlice(c, "function createThinkingFold(", "// The three override picks");
  runSlice(c, "function addUserBubble(", "function askRewind(");
  runSlice(c, "function attachLiveBox(", "// A turn the user has left");
  runSlice(c, "function dropBackgroundTurn(", "// BEGIN history retry");
  runSlice(c, "function isRemoteTurn()", "// Transient note through");
  runSlice(c, "async function reloadStale()", "// --- turn confirmation");
  runSlice(c, "function ensureConfirmPoll()", "// A pick is the moment");
  runSlice(c, "function recoveredProgressView(", "function handleReplayEvent(");
  runSlice(c, "function showEventWarning(", "async function pollConfirm(");
  const fetch = c.fetch;
  c.fetch = async (url, options = {}) => {
    const pathname = new URL(url).pathname;
    if (pathname.endsWith("/turns/stream")) {
      requests.push({ url, method: options.method, body: JSON.parse(options.body) });
      requested.resolve();
      return headers.promise;
    }
    if (pathname.endsWith("/events")) {
      const replay = controlledReader(); replays.push(replay);
      options.signal.addEventListener("abort", () => replay.reject("replay aborted"), { once: true });
      return { ok: true, body: { getReader: () => replay } };
    }
    return fetch(url, options);
  };
  const flush = async () => {
    await settle();
    for (const [id, timer] of [...timers]) if (timer.type === "timeout" && timer.at <= now) {
      timers.delete(id); timer.fn();
    }
    await settle();
  };
  let pending, entry;
  return {
    ...h, c, timers, reader, replays, requests,
    async start() {
      h.elements.prompt.value = "run characterization";
      pending = c.send(); await requested.promise; await flush();
      entry = c.executionState().liveEntry;
      assert.ok(entry, "real send registers its live turn before headers arrive");
      assert.equal(h.elements.prompt.value, "");
      h.elements.prompt.value = "follow-up draft";
      return entry;
    },
    async accept() { headers.resolve({ ok: true, body: { getReader: () => reader } }); await flush(); assert.ok(reader.pending()); },
    async httpFailure() { headers.resolve({ ok: false, status: 503, statusText: "Unavailable", json: async () => ({ error: "try later" }) }); await pending; await flush(); },
    async headersFailure() { headers.reject(new Error("fetch failed before headers")); await pending; await flush(); },
    async frame(kind, value) { reader.frame(sse(kind, value)); await flush(); },
    async finish(action) { action(); await pending; await flush(); },
    liveBox: () => entry.box,
    entry: () => entry,
    snapshot() {
      const state = c.executionState(), b = h.elements.send;
      return { prompt: h.elements.prompt.value, sendBusy: state.sendBusy, canSend: c.canSend(), canStop: c.canStop(),
        send: b.textContent, title: b.title, stopClass: b.classList.contains("stop"),
        composerBusy: h.elements.composer.getAttribute("aria-busy"), sideNote: h.elements.sidenote.classList.contains("show"),
        live: state.live, attached: state.attached, recovering: state.recovering, ended: state.ended, failed: state.failed,
        tasks: Array.from(state.taskStates, ([id, active]) => [id, active]), taskBusy: state.taskBusy,
        settled: entry.userInput.settled,
        errors: records(entry.box).filter((node) => node.classList.contains("err")).map(answerText),
        timeouts: [...timers.values()].filter((timer) => timer.type === "timeout").map((timer) => timer.delay).sort((a, b) => a - b),
        intervals: [...timers.values()].filter((timer) => timer.type === "interval").map((timer) => timer.delay).sort((a, b) => a - b),
      };
    },
    async dispose() { c.abortEvents(); c.stopConfirmPoll(); c.stopStaleFollow(); await flush(); assert.equal(timers.size, 0); },
  };
}

async function activeTurn(h) {
  await h.frame("thinking_start", {});
  await h.frame("thinking", { text: "Working on the request." });
  await h.frame("tool", { id: "pending-tool", name: "read_file", phase: "started", arguments: calls[0].function.arguments });
  await h.frame("subagent_progress", stream.structuredSubagents([activeTasks[0]]));
  await h.frame("delta", { delta: "Provisional answer." });
  assert.deepEqual(h.snapshot().tasks, [["worker", true]]);
  assert.equal(h.snapshot().taskBusy, true);
  assert.equal(h.snapshot().canStop, true);
  assert.equal(h.snapshot().settled, false, "pending-input settlement currently happens at send cleanup");
}

const cleanState = {
  prompt: "follow-up draft", sendBusy: false, canSend: true, canStop: false, send: "➤", title: "send", stopClass: false,
  composerBusy: "false", sideNote: false, live: false, attached: false, recovering: false,
  ended: true, failed: false, tasks: [], taskBusy: null, settled: true, errors: [], timeouts: [], intervals: [],
};

test("real send: done followed by EOF preserves the new composer draft, settles tasks and clears timers", async () => {
  const h = sendHarness();
  await h.start();
  assert.equal(h.snapshot().canSend, false); assert.equal(h.snapshot().canStop, true);
  assert.deepEqual(h.snapshot().timeouts, []); assert.deepEqual(h.snapshot().intervals, [1000]);
  await h.accept();
  assert.deepEqual(h.snapshot().timeouts, [60000, 300000]);
  await activeTurn(h);
  await h.frame("tool", { id: "pending-tool", name: "read_file", phase: "finished", ok: true, output: "complete" });
  await h.frame("replace", { text: "Final answer." });
  await h.finish(() => { h.reader.frame(sse("done", {})); h.reader.eof(); });
  assert.deepEqual(h.snapshot(), cleanState);
  assert.equal(h.liveBox().parentNode, h.elements.msgs);
  assert.equal(h.replays.length, 0);
  assert.equal(h.requests[0].body.prompt, "run characterization");
  await h.dispose();
});

for (const phone of [false, true]) for (const scenario of [
  { name: "EOF without done", active: true, end: (h) => h.reader.eof(),
    error: "stream ended before completion — checking server history" },
  { name: "reader rejection immediately after acceptance", active: false, end: (h) => h.reader.reject("reader rejected"),
    error: "connection lost: reader rejected — the turn keeps running on the server (following history)" },
  { name: "transport loss after a partial answer and active task snapshot", active: true, end: (h) => h.reader.reject("transport lost"),
    error: "connection lost: transport lost — the turn keeps running on the server (following history)" },
]) test("real send recovery: " + scenario.name + " keeps stop while replay is attached (phone=" + phone + ")", async () => {
  const h = sendHarness({ phone }); await h.start(); await h.accept();
  if (scenario.active) await activeTurn(h);
  await h.finish(() => scenario.end(h));
  assert.equal(h.snapshot().send, "⏹", "transport loss must keep the running turn stoppable");
  assert.deepEqual(h.snapshot(), { ...cleanState, canSend: false, canStop: true, send: "⏹", title: "stop this turn",
    stopClass: true, ended: false, recovering: true, attached: true, sideNote: true,
    errors: [scenario.error], timeouts: [25000, 300000], intervals: [1000, 1500, 2000] });
  assert.equal(h.c.staleSession, h.c.sessionId); assert.notEqual(h.c.staleTimer, null);
  assert.equal(h.liveBox().parentNode, null, "the recorded error belongs to the retired live box");
  assert.equal(h.replays.length, 1);
  assert.equal(records(h.elements.msgs).filter((node) => node.classList.contains("err")).length, 0,
    "the retired live error is not visible in the replacement transcript");
  if (scenario.active) {
    const interrupted = toolNode(h.liveBox(), "read_file");
    assert.equal(interrupted.dataset.executionState, "completed");
    assert.equal(interrupted.dataset.executionException, "true");
  }
  h.replays[0].frame(sse("thinking", { text: "Still working after reconnect." }));
  await settle();
  assert.equal(h.snapshot().send, "⏹");
  assert.equal(h.snapshot().canStop, true);
  assert.equal(h.snapshot().failed, false);
  h.c.turnRunning = false; h.c.turnRunningAt = 0;
  h.replays[0].eof();
  await settle();
  assert.equal(h.snapshot().attached, false);
  assert.equal(h.snapshot().recovering, true);
  assert.equal(h.snapshot().send, "⏹", "unresolved recovery survives another lost stream without a running sample");
  await h.dispose();
});

for (const phone of [false, true]) test("real send recovery: only a confirmed replay failure returns to send (phone=" + phone + ")", async () => {
  const h = sendHarness({ phone }); await h.start(); await h.accept();
  await h.frame("thinking", { text: "Working before transport loss." });
  await h.finish(() => h.reader.reject("network changed"));
  assert.equal(h.snapshot().send, "⏹");
  h.replays[0].frame(sse("thinking", { text: "Working after transport recovery." }));
  await settle();
  assert.equal(h.snapshot().send, "⏹");
  h.replays[0].frame(sse("error", { error: "confirmed turn failure after reconnect" }));
  await settle();
  assert.equal(h.snapshot().failed, true);
  assert.equal(h.snapshot().ended, true);
  assert.equal(h.snapshot().recovering, false);
  assert.equal(h.snapshot().attached, false);
  assert.equal(h.snapshot().send, "➤");
  assert.equal(h.snapshot().title, "send");
  assert.equal(h.snapshot().canSend, true);
  assert.equal(h.snapshot().canStop, false);
  assert.match(h.elements.msgs.textContent, /confirmed turn failure after reconnect/);
  assert.deepEqual(h.snapshot().timeouts, []);
  await h.dispose();
});

test("real send recovery: the phone stop button still posts an interrupt for the running turn", async () => {
  const h = sendHarness({ phone: true }); await h.start(); await h.accept();
  await h.finish(() => h.reader.eof());
  assert.equal(h.snapshot().send, "⏹");
  runSlice(h.c, "async function stopTurn()", "// Parse one SSE block");
  const interruptions = [];
  h.c.api = async (pathname, options) => {
    interruptions.push({ pathname, method: options.method });
    return { json: async () => ({ interrupted: true }) };
  };
  h.elements.send.onclick();
  await settle();
  assert.deepEqual(interruptions, [{ pathname: "/sessions/" + h.c.sessionId + "/interrupt", method: "POST" }]);
  assert.equal(h.snapshot().send, "⏹", "the turn stays stoppable until the server confirms termination");
  h.replays[0].frame(sse("error", { error: "turn interrupted" }));
  await settle();
  assert.equal(h.snapshot().send, "➤");
  assert.equal(h.snapshot().canStop, false);
  await h.dispose();
});

test("real send: a server error block interrupts task state and retains one error row without reconnecting", async () => {
  const h = sendHarness(); await h.start(); await h.accept(); await activeTurn(h);
  await h.finish(() => h.reader.frame(sse("error", { error: "server rejected the turn" })));
  assert.deepEqual(h.snapshot(), { ...cleanState, failed: true, taskBusy: false, errors: ["turn failed: server rejected the turn"] });
  assert.equal(h.liveBox().parentNode, h.elements.msgs);
  assert.equal(h.replays.length, 0);
  assert.equal(records(h.liveBox()).filter((node) => node.classList.contains("err")).length, 1);
  await h.dispose();
});

for (const outcome of ["HTTP failure", "fetch rejection"]) test("real send: " + outcome + " before acceptance keeps the input unsettled and clears all turn timers", async () => {
  const h = sendHarness(); await h.start();
  assert.equal(h.snapshot().live, true); assert.equal(h.snapshot().settled, false);
  assert.equal(h.snapshot().composerBusy, "false");
  if (outcome === "HTTP failure") await h.httpFailure(); else await h.headersFailure();
  const message = outcome === "HTTP failure" ? "turn failed: 503 try later" : "turn failed: fetch failed before headers";
  assert.deepEqual(h.snapshot(), { ...cleanState, failed: true, settled: false, errors: [message] });
  assert.equal(h.replays.length, 0);
  assert.equal(h.liveBox().parentNode, h.elements.msgs);
  await h.dispose();
});

test("real send recovery: a server-confirmed end follows history without reattaching", async () => {
  const h = sendHarness(); await h.start(); await h.accept(); await activeTurn(h);
  const api = h.c.api;
  Object.assign(h.c, {
    pollConfirm: async () => false,
    api: (pathname, ...args) => pathname.includes("/history?")
      ? Promise.resolve({ headers: { get: () => "0" }, json: async () => [] }) : api(pathname, ...args),
    histKey: (messages) => JSON.stringify(messages), staleKey: "", historyLoadStates: new Map(),
  });
  await h.finish(() => h.entry().recover("server says the turn ended", true));
  assert.deepEqual(h.snapshot(), { ...cleanState, ended: false,
    errors: ["server says the turn ended — checking server history"] });
  assert.equal(h.reader.cancellations(), 1);
  assert.equal(h.replays.length, 0, "recoverEnded suppresses retained-event reattachment");
  assert.equal(h.c.staleSession, ""); assert.equal(h.c.staleTimer, null);
  assert.equal(h.liveBox().parentNode, null);
  await h.dispose();
});

function viewSnapshot(h) {
  const c = h.context;
  return JSON.stringify({ text: h.root.textContent, selected: c.sessionId, mounted: c.historySession,
    keys: c.historyKeys, sequence: c.historyLoadSeq, total: c.canonicalHistoryTotal, count: c.lastMsgCount,
    window: h.root._histWin, windowSession: h.root._histWinSession,
    nodes: [h.root, ...stream.descendants(h.root)].map((node) => ({ text: node.textContent, className: node.className, dataset: node.dataset })) });
}
function historyResponse(messages, total) {
  return { ok: true, headers: { get: () => String(total) }, json: async () => messages };
}

for (const stage of ["history body", "HTTP failure body"]) for (const navigation of ["A→B", "A→B→A"]) {
  test("view ownership: delayed " + stage + " after " + navigation + " cannot mutate transcript identity, text or pager metadata", async () => {
    const h = errors.harness({ realApi: true }), c = h.context, body = deferred(), reading = deferred(), requests = [];
    const storage = new Map();
    Object.assign(c, {
      persistDraft() {}, restoreDraft() {}, resetQuestionNavigator() {}, restoreSkillPick() {}, restoreTeamPick() {},
      localStorage: { setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key) },
    });
    runSlice(c, "// BEGIN session switch", "// END session switch");
    c.fetch = async (url) => {
      const pathname = new URL(url).pathname; requests.push(pathname);
      if (requests.length === 1) return { ok: stage === "history body", status: 503, statusText: "Unavailable",
        headers: { get: () => "999" }, json() { reading.resolve(); return body.promise; } };
      const isA = pathname === "/sessions/s/history";
      assert.ok(isA || pathname === "/sessions/session-b/history");
      return historyResponse([{ role: "user", content: isA ? "Current A prompt." : "Current B prompt." },
        { role: "assistant", content: isA ? "Current A answer." : "Current B answer." }], isA ? 73 : 42);
    };
    const pendingA = c.loadHistory(); await reading.promise;
    c.switchSession("session-b", false); await c.loadHistory();
    if (navigation === "A→B→A") { c.switchSession("s", false); await c.loadHistory(); }
    const owner = navigation === "A→B" ? "session-b" : "s", total = navigation === "A→B" ? 42 : 73;
    assert.equal(c.historySession, owner); assert.equal(h.root._histWinSession, owner);
    assert.equal(c.canonicalHistoryTotal, total); assert.equal(h.root._histWin, 2);
    const before = viewSnapshot(h), nodes = [h.root, ...stream.descendants(h.root)];
    body.resolve(stage === "history body" ? [{ role: "assistant", content: "Obsolete A answer." }] : { error: "obsolete A failure" });
    await pendingA;
    assert.equal(viewSnapshot(h), before);
    const after = [h.root, ...stream.descendants(h.root)];
    assert.equal(after.length, nodes.length); nodes.forEach((node, index) => assert.equal(after[index], node));
    assert.doesNotMatch(h.root.textContent, /Obsolete|obsolete|history failed/);
    assert.equal(c.historyLoadState("s").failure, null);
    assert.equal(h.timers.size, 0, "obsolete bodies cancel their deadlines without scheduling retries");
    assert.equal(h.renderErrors.length, 0);
    assert.deepEqual(requests, navigation === "A→B" ? ["/sessions/s/history", "/sessions/session-b/history"]
      : ["/sessions/s/history", "/sessions/session-b/history", "/sessions/s/history"]);
  });
}

function installOwnershipNavigation(c) {
  Object.assign(c, {
    persistDraft() {}, restoreDraft() {}, resetQuestionNavigator() {}, restoreSkillPick() {}, restoreTeamPick() {},
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  });
  runSlice(c, "// BEGIN session switch", "// END session switch");
}

function installServerSave(c) {
  const originalQuery = c.$, originalBase = c.store.base, storage = new Map([["serve.base", originalBase]]);
  const nodes = {
    save: {}, base: { value: originalBase }, token: { value: "" },
    conn: { classList: { remove() {} } }, connstat: { classList: { add() {} } },
  };
  c.$ = (id) => nodes[id] || originalQuery(id);
  c.localStorage = { getItem: (key) => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key) };
  Object.defineProperty(c.store, "base", { configurable: true, get: () => storage.get("serve.base") || originalBase });
  Object.assign(c, { URL, location: { origin: originalBase }, showStored() {}, boot: async () => {} });
  runSlice(c, "function normalizeServeBase(", "const store =");
  runSlice(c, '  $("save").onclick = async () => {', '  $("newbtn").onclick =');
  return async (base) => { nodes.base.value = base; await nodes.save.onclick(); };
}

for (const outcome of ["body", "error body"]) for (const navigation of ["session ABA", "server ABA"]) {
  test("a late history " + outcome + " after " + navigation + " cannot commit without a replacement load", async () => {
    const h = errors.harness({ realApi: true }), c = h.context, body = deferred(), reading = deferred();
    installOwnershipNavigation(c);
    const saveServer = installServerSave(c), originalBase = c.store.base;
    h.root.textContent = "Retained transcript.";
    c.fetch = async () => ({ ok: outcome === "body", status: 503, statusText: "Unavailable",
      headers: { get: () => "999" }, json() { reading.resolve(); return body.promise; } });
    const pending = c.loadHistory(); await reading.promise;
    if (navigation === "session ABA") { c.switchSession("session-b", false); c.switchSession("s", false); }
    else { await saveServer("https://other.test"); await saveServer(originalBase); }
    const before = viewSnapshot(h);
    body.resolve(outcome === "body" ? [{ role: "assistant", content: "Obsolete response." }] : { error: "obsolete error" });
    await pending;
    assert.equal(viewSnapshot(h), before, "the complete selected view remains unchanged by an obsolete result");
    assert.equal(h.root.textContent, "");
    assert.equal(c.historySession, "");
    assert.equal(JSON.stringify(c.historyKeys), "[]");
    assert.equal(c.historyLoadSeq, 3);
    assert.equal(c.canonicalHistoryTotal, 0);
    assert.equal(c.lastMsgCount, null);
    assert.equal(h.root._histWinSession, undefined);
    assert.equal(h.root._histWin, undefined);
    assert.equal(c.historyLoadState("s").failure, null);
    assert.equal(c.historyLoadState("s").loading, 0);
    assert.equal(h.timers.size, 0);
  });
}

test("a retained retry state cannot display its failure row after a server switch", async () => {
  const h = errors.harness(), c = h.context;
  installOwnershipNavigation(c);
  const saveServer = installServerSave(c), originalBase = c.store.base;
  await h.failHistory("server A is unavailable");
  const originalState = c.historyLoadState("s");
  assert.ok(originalState.failure);
  await saveServer("https://other.test");
  assert.equal(c.historyLoadState("s"), originalState);
  assert.equal(c.historyLoadState("s").failure, originalState.failure);
  assert.equal(c.historyLoadState("s", originalBase), originalState);
  assert.equal(h.historyRows().length, 0);
  assert.equal(originalState.failure.row.parentNode, null);
  assert.equal(originalState.failure.timer, null);
});

for (const navigation of ["session ABA", "server ABA"]) {
  const title = navigation === "session ABA"
    ? "HEAD parity: a late confirm poll after session ABA still samples the turn and shows its question"
    : "HEAD parity: a late confirm poll after server ABA still samples the turn and shows its question";
  test(title, async () => {
    for (const running of [false, true]) {
      const h = errors.harness(), c = h.context, response = deferred(), shown = [];
      let attached = 0;
      installOwnershipNavigation(c);
      const saveServer = installServerSave(c), originalBase = c.store.base;
      runSlice(c, "async function pollConfirm()", "// Live streams, replay streams");
      Object.assign(c, { confirmBusy: false, turnRunning: false, isRemoteTurn: () => false,
        attachEvents() { attached++; }, refreshSideNotes() {},
        showConfirm(...args) { shown.push(args); },
        api: async () => ({ json: () => response.promise }) });
      const pending = c.pollConfirm(); await settle();
      const epoch = c.runEpoch, seq = c.confirmPollSeq;
      if (navigation === "session ABA") {
        c.switchSession("session-b", false); c.switchSession("s", false);
        assert.equal(c.runEpoch, epoch); assert.equal(c.confirmPollSeq, seq);
      } else { await saveServer("https://other.test"); await saveServer(originalBase); }
      const before = JSON.stringify(c.executionState());
      response.resolve({ running, turn: 42,
        pending: { id: 7, prompt: "Confirm navigation question", token: "navigation-question", input: false } });
      {
        assert.equal(await pending, running);
        if (!running) assert.equal(JSON.stringify(c.executionState()), before);
        else assert.equal(c.turnRunningPid, 42);
        assert.equal(c.turnRunning, running);
        assert.equal(attached, Number(running));
        assert.deepEqual(shown, [[7, "Confirm navigation question", "navigation-question", false]]);
      }
    }
  });
}

test("HEAD parity: a queued cancelled history retry can restart after session ABA", async () => {
  const h = errors.harness(), c = h.context;
  installOwnershipNavigation(c);
  await h.failHistory("temporary history failure");
  const retry = [...h.timers.values()].find((timer) => timer.delay === 1000);
  assert.ok(retry);
  c.switchSession("session-b", false); c.switchSession("s", false);
  let calls = 0;
  c.api = async () => { calls++; return historyResponse(h.messages, 2); };
  retry.callback(); await settle();
  assert.equal(calls, 1, "HEAD checks the current sid and failure identity when a queued callback runs");
  assert.equal(h.root.textContent, "promptreply");
  assert.equal(h.timers.size, 0);
});

test("owner regression: a late history finally leaves the replacement load lock intact", async () => {
  const h = errors.harness(), c = h.context, first = deferred(), second = deferred();
  installOwnershipNavigation(c);
  let calls = 0;
  c.api = async () => ({ headers: { get: () => "2" }, json: () => (++calls === 1 ? first.promise : second.promise) });
  const old = c.loadHistory(); await settle();
  c.switchSession("session-b", false); c.switchSession("s", false);
  const replacement = c.loadHistory(); await settle();
  const state = c.historyLoadState("s"), lock = state.loading;
  first.resolve([{ role: "assistant", content: "Old answer." }]); await old;
  assert.equal(state.loading, lock);
  assert.ok(state.loading);
  assert.equal(h.root.textContent, "");
  second.resolve(h.messages); await replacement;
  assert.equal(state.loading, 0);
  assert.equal(h.root.textContent, "promptreply");
  assert.equal(h.timers.size, 0);
});

test("HEAD parity: a superseded pager skips desktop scrolling but clears the shared paging flag", async () => {
  const h = errors.harness(), c = h.context, first = deferred(), second = deferred();
  installOwnershipNavigation(c);
  Object.assign(c, { lastMsgCount: 2, canonicalHistoryTotal: 20 });
  let scrollOwner = {};
  Object.assign(c, { isDesktopTranscriptScroll: () => true, ownedTranscriptScroll: () => scrollOwner });
  Object.assign(h.root, { _histWinSession: "s", _histWin: 2 });
  let calls = 0;
  c.loadHistory = () => ++calls === 1 ? first.promise : second.promise;
  const old = c.loadOlderHistory(); await settle();
  c.switchSession("session-b", false);
  scrollOwner = {};
  h.root._histWinSession = "session-b";
  Object.assign(c, { lastMsgCount: 2, canonicalHistoryTotal: 20 });
  h.root._histOlder = false; // The replacement transcript starts its own pager.
  h.root.scrollTop = 77;
  const replacement = c.loadOlderHistory(); await settle();
  h.root.scrollHeight = 350;
  first.resolve(); await old;
  assert.equal(h.root._histOlder, false, "HEAD finally releases the shared flag even during the replacement request");
  assert.equal(h.root.scrollTop, 77);
  second.resolve(); await replacement;
  assert.equal(h.root._histOlder, false);
  assert.equal(h.root.scrollTop, 227);
});

for (const kind of ["confirm_request", "confirm_done", "input_request", "input_done", "native input", "delta"]) {
  test("HEAD parity: a live " + kind + " from the same sid on another server still affects this page", async () => {
    const h = sendHarness(); await h.start(); await h.accept();
    const c = h.c, originalBase = c.store.base, calls = [];
    c.store.base = "https://other.test";
    Object.assign(c, { showConfirm() { calls.push("show"); }, hideConfirm() { calls.push("hide"); },
      handleNativeRequest() { calls.push("native"); }, shouldFollowOutput: () => true,
      followOutput() { calls.push("follow"); }, confirmShown: { id: 7, token: "current-server" } });
    const event = { id: 7, token: "old-server", prompt: kind === "native input" ? "\x1enative:action" : "Old question." };
    if (kind === "native input") event.prompt = vm.runInContext("NATIVE_PREFIX", c) + "action";
    await h.frame(kind === "native input" ? "input_request" : kind, kind === "delta" ? { delta: "Background answer." } : event);
    const notification = kind === "delta" ? [] : [kind === "native input" ? "native"
      : kind.endsWith("_done") ? "hide" : "show"];
    assert.deepEqual(calls, [...notification, "follow"]);
    if (kind === "delta") assert.match(h.liveBox().textContent, /Background answer\./);
    assert.equal(c.executionState(c.sessionId, originalBase).liveEntry, h.entry());
    c.store.base = originalBase;
    await h.finish(() => h.reader.frame(sse("done", {})));
    await h.dispose();
  });
}

test("owner regression: valid background live feeds update their own box and model across session ABA", async () => {
  const h = sendHarness(); await h.start(); await h.accept();
  const c = h.c, sid = c.sessionId, base = c.store.base, entry = h.entry(), calls = [];
  c.switchSession("session-b", false); entry.box.remove();
  Object.assign(c, { showConfirm() { calls.push("show"); }, shouldFollowOutput: () => true,
    followOutput() { calls.push("follow"); } });
  await h.frame("delta", { delta: "While A was hidden." });
  await h.frame("subagent_progress", stream.structuredSubagents([stream.structuredTask("background-worker", 5000)]));
  await h.frame("confirm_request", { id: 7, token: "A-question", prompt: "Continue A?" });
  assert.deepEqual(calls, []);
  assert.match(entry.box.textContent, /While A was hidden/);
  assert.equal(c.executionState(sid, base).liveEntry, entry);
  assert.deepEqual(Array.from(c.executionState(sid, base).taskStates, ([id, active]) => [id, active]), [["background-worker", true]]);
  c.switchSession(sid, false); c.attachLiveBox();
  assert.equal(entry.box.parentNode, h.elements.msgs);
  calls.length = 0;
  await h.frame("confirm_request", { id: 7, token: "A-question", prompt: "Continue A?" });
  assert.ok(calls.includes("show"), "the feed remains valid after its original view generation expired");
  await h.frame("delta", { delta: " Back on A." });
  assert.match(entry.box.textContent, /While A was hidden\. Back on A\./);
  await h.finish(() => h.reader.frame(sse("done", {})));
  assert.equal(c.executionState(sid, base).live, false);
  assert.equal(c.canSend(), true);
  await h.dispose();
});

for (const newerTurn of [false, true]) {
  test("HEAD parity: old live cleanup deletes a replacement feed" + (newerTurn ? " in a newer turn" : " in the same turn"), async () => {
    const h = sendHarness(); await h.start(); await h.accept();
    const c = h.c, sid = c.sessionId, base = c.store.base;
    if (newerTurn) c.updateExecutionModel({ kind: "begin" }, sid, base);
    const replacement = { box: c.document.createElement("div"), lastEventAt: 2000 };
    c.updateExecutionModel({ kind: "live", entry: replacement }, sid, base);
    c.turnRunning = true;
    const pageEffects = [];
    Object.assign(c, { loadInfo() { pageEffects.push("info"); }, loadSessions() { pageEffects.push("sessions"); },
      loadSessionConfig() { pageEffects.push("config"); }, followOutput() { pageEffects.push("follow"); } });
    await h.finish(() => h.reader.frame(sse("done", {})));
    assert.equal(c.executionState(sid, base).liveEntry, undefined);
    assert.equal(c.executionState(sid, base).running, false);
    assert.equal(c.executionState(sid, base).ended, true);
    assert.equal(c.executionState(sid, base).failed, false);
    assert.deepEqual(pageEffects, ["follow", "follow", "follow", "info", "sessions", "config"]);
    assert.equal(c.canStop(), false);
    assert.equal(c.canSend(), true);
    c.updateExecutionModel({ kind: "live", entry: null }, sid, base);
    await h.dispose();
  });
}

test("owner regression: old replay finally preserves the replacement controller and view", async () => {
  const h = errors.harness(), c = h.context, headers = [deferred(), deferred()];
  let calls = 0;
  c.fetch = () => headers[calls++].promise;
  const old = c.attachEvents();
  const oldController = c.eventAbort;
  c.abortEvents();
  const replacement = c.attachEvents(), controller = c.eventAbort, view = c.recoveredView;
  assert.notEqual(controller, oldController);
  headers[0].resolve({ ok: true, body: {} }); await old;
  assert.equal(c.eventAbort, controller);
  assert.equal(c.recoveredView, view);
  assert.equal(c.eventActive, true);
  assert.equal(view.box.parentNode, h.root);
  c.abortEvents(); headers[1].resolve({ ok: true, body: {} }); await replacement;
  assert.equal(h.timers.size, 0);
});

test("HEAD parity: a queued replay retry can attach after session ABA", async () => {
  const h = errors.harness(), c = h.context;
  installOwnershipNavigation(c);
  let calls = 0;
  c.fetch = async () => { calls++; throw new Error("wire unavailable"); };
  await c.attachEvents();
  const retry = [...h.timers.values()].find((timer) => timer.delay === 2000);
  assert.ok(retry);
  c.switchSession("session-b", false); c.switchSession("s", false);
  c.turnRunning = true; c.turnRunningAt = Date.now();
  retry.callback(); await settle();
  assert.equal(calls, 2);
  assert.equal(c.eventActive, false);
  assert.equal(c.recoveredView, null);
  assert.deepEqual([...h.timers.values()].map((timer) => timer.delay), [2000, 2000]);
});

test("HEAD parity: composer preparation adopts a late created session after session ABA", async () => {
  const h = desktop.harness({ sid: "" }), c = h.context, created = deferred();
  c.api = async () => ({ json: () => created.promise });
  const pending = c.prepareTurnSession("", c.store.base); await settle();
  c.switchSession("session-b", false); c.switchSession("", false);
  created.resolve({ id: "obsolete-created-session", cwd: "/old" });
  assert.equal(await pending, "obsolete-created-session");
  assert.equal(c.sessionId, "obsolete-created-session");
  assert.equal(c.pendingNew, "obsolete-created-session");
  assert.equal(c.pendingCwd, "/old");
});

test("HEAD parity: composer config waits resume after session or server ABA", async () => {
  for (const navigation of ["session ABA", "server ABA"]) {
    const h = desktop.harness(), c = h.context, config = deferred(), sid = c.sessionId;
    const saveServer = installServerSave(c), originalBase = c.store.base;
    c.ensureTeamConfig = () => config.promise;
    const pending = c.prepareTurnSession(sid, c.store.base); await settle();
    if (navigation === "session ABA") { c.switchSession("session-b", false); c.switchSession(sid, false); }
    else { await saveServer("https://other.test"); await saveServer(originalBase); }
    config.resolve(true);
    assert.equal(await pending, sid);
    assert.equal(c.sessionId, sid); assert.equal(c.store.base, originalBase);
  }
});

test("HEAD parity: an active composer preparation adopts the created session", async () => {
  const h = desktop.harness({ sid: "" }), c = h.context, config = deferred();
  c.api = async () => ({ json: async () => ({ id: "fresh-owned-session", cwd: "/fresh" }) });
  c.ensureTeamConfig = (sid, base) => {
    assert.equal(sid, "fresh-owned-session"); assert.equal(base, c.store.base); return config.promise;
  };
  const pending = c.prepareTurnSession("", c.store.base); await settle();
  assert.equal(c.sessionId, "fresh-owned-session");
  assert.equal(c.pendingNew, "fresh-owned-session");
  config.resolve(true);
  assert.equal(await pending, "fresh-owned-session");
  assert.equal(c.pendingCwd, "/fresh");
});

for (const navigation of ["session ABA", "server ABA"]) {
  test("HEAD parity: a late session-list page after " + navigation + " commits list and pager state", async () => {
    const h = desktop.harness(), c = h.context, body = deferred(), sid = c.sessionId;
    const saveServer = installServerSave(c), originalBase = c.store.base;
    c.fetch = async () => ({ ok: true, headers: { get: (name) => name === "x-sessions-total" ? "71" : "1" },
      json: () => body.promise });
    const pending = c.loadSessions(); await settle();
    if (navigation === "session ABA") { c.switchSession("session-b", false); c.switchSession(sid, false); }
    else { await saveServer("https://other.test"); await saveServer(originalBase); }
    const snapshot = () => vm.runInContext("JSON.stringify([lastItems,sessionsTotal,sessionsPageOffset,sessionsHasMore])", c);
    const before = snapshot();
    body.resolve([{ id: sid, summary: "Obsolete session list.", size_bytes: 1 }]); await pending;
    assert.notEqual(snapshot(), before);
    assert.equal(snapshot(), JSON.stringify([[{ id: sid, summary: "Obsolete session list.", size_bytes: 1 }], 71, 1, true]));
  });
}

test("HEAD parity: an obsolete legacy session page issues its fallback request on the current server", async () => {
  const h = desktop.harness(), c = h.context, body = deferred(), requests = [];
  const saveServer = installServerSave(c);
  c.fetch = async (url) => {
    requests.push(url);
    return { ok: true, headers: { get: () => null }, json: () => body.promise };
  };
  const pending = c.loadSessions(); await settle();
  await saveServer("https://other.test");
  body.resolve([{ id: c.sessionId, summary: "Old server." }]); await pending;
  assert.equal(requests.length, 2);
  assert.match(requests[0], /^https:\/\/serve\.test\/sessions\?/);
  assert.equal(requests[1], "https://other.test/sessions");
});

for (const teamOnly of [false, true]) {
  test("HEAD parity: a late " + (teamOnly ? "team-only" : "full") + " config read restores the current equal sid/base", async () => {
    const body = deferred(), h = team.harness({ cfg: team.NORMAL, configGet: () => body.promise }), c = h.context;
    const pending = c.loadSessionConfig({ teamOnly }); await settle();
    c.switchSession("beta"); c.switchSession("alpha");
    const before = JSON.stringify(h.state()), label = h.elements.teamlabel.textContent;
    body.resolve(team.response(team.TRAECLI)); await pending;
    assert.notEqual(JSON.stringify(h.state()), before);
    assert.deepEqual(JSON.parse(JSON.stringify(h.state())), {
      choice: { mode: "team", backend: "traecli", model: "worker-model" },
      confirmed: { mode: "team", backend: "traecli", model: "worker-model" },
      pending: false, saving: null, error: "", errorSource: "", revision: 0,
      configRetryTimer: null, configRetries: 0, loadEpoch: 1,
      fullLoadEpoch: teamOnly ? 0 : 1, appliedTeamLoadEpoch: 1,
    });
    assert.equal(h.elements.teamlabel.textContent, "subagent");
  });
}

test("owner regression: valid background completion preserves the page's info and session-list refresh", async () => {
  const h = sendHarness(); await h.start(); await h.accept();
  const calls = [], c = h.c;
  c.switchSession("session-b", false);
  Object.assign(c, { loadInfo() { calls.push("info"); }, loadSessions() { calls.push("sessions"); },
    loadSessionConfig() { calls.push("config"); } });
  await h.finish(() => h.reader.frame(sse("done", {})));
  assert.deepEqual(calls, ["info", "sessions"]);
  assert.equal(h.entry().userInput.settled, true);
  assert.equal(h.liveBox().parentNode, null);
  await h.dispose();
});

test("HEAD parity: a stale live feed from another server publishes page notifications and refreshes", async () => {
  const h = sendHarness(); await h.start(); await h.accept();
  const c = h.c, originalBase = c.store.base, sid = c.sessionId, calls = [];
  c.store.base = "https://other.test";
  Object.assign(c, { note() { calls.push("note"); }, loadInfo() { calls.push("info"); },
    loadSessions() { calls.push("sessions"); }, loadSessionConfig() { calls.push("config"); },
    followOutput() { calls.push("follow"); } });
  await h.finish(() => h.reader.eof());
  assert.deepEqual(calls, ["note", "follow", "info", "sessions"]);
  assert.equal(c.executionState(sid, originalBase).live, false);
  assert.equal(c.executionState(sid).recovering, false);
  assert.equal(h.liveBox().parentNode, null);
  await h.dispose();
});

test("dispatcher decode retains wire bytes, decoded values and parse success independently", () => {
  const h = stream.streamHarness(), decode = h.context.decodeExecutionEvent;
  const payload = ' {"delta":"same"} ';
  const valid = decode("delta", payload), malformed = decode("replace", '{"text":'), empty = decode("delta", "null");
  assert.equal(valid.kind, "delta");
  assert.equal(valid.payload, payload);
  assert.deepEqual(JSON.parse(JSON.stringify(valid.value)), { delta: "same" });
  assert.equal(valid.parsed, true);
  assert.equal(malformed.payload, '{"text":');
  assert.equal(malformed.value, null);
  assert.equal(malformed.parsed, false);
  assert.equal(empty.value, null);
  assert.equal(empty.parsed, true, "decoded null remains distinct from malformed JSON");
});

test("dispatcher live deltas and replace update synchronously without advancing their debounce paint", () => {
  const h = stream.streamHarness(), c = h.context;
  let renders = 0, follows = 0;
  const render = c.renderMarkdown;
  c.renderMarkdown = (...args) => { renders++; return render(...args); };
  c.shouldFollowOutput = () => true;
  c.followOutput = () => { follows++; };
  h.emit("delta", { delta: "draft " }, { paint: false });
  h.emit("delta", { delta: "tail" }, { paint: false });
  assert.equal(vm.runInContext("raw", c), "draft tail");
  assert.equal(h.reply(), "");
  assert.equal(renders, 0);
  assert.equal(follows, 0);
  assert.equal(h.timers.filter((delay) => delay === 50).length, 1);
  h.emit("replace", { text: "canonical" }, { paint: false });
  assert.equal(vm.runInContext("raw", c), "canonical");
  assert.equal(h.reply(), "");
  h.flush();
  assert.equal(h.reply(), "canonical");
  assert.equal(renders, 1);
  assert.equal(follows, 1);
  h.emit("replace", { text: "" }, { paint: false });
  assert.equal(vm.runInContext("raw", c), "");
  assert.equal(h.reply(), "canonical", "empty replacement waits for the same source paint boundary");
  h.flush();
  assert.equal(h.reply(), "");
  assert.equal(vm.runInContext("bubble.hidden", c), true);
});

test("dispatcher live prose seals before a tool and the next delta opens a separate unpainted segment", () => {
  const h = stream.streamHarness(), c = h.context;
  h.emit("delta", { delta: "before tool" }, { paint: false });
  h.emit("tool", { id: "ordered", name: "read_file", phase: "started", arguments: calls[0].function.arguments }, { paint: false });
  assert.deepEqual(projection(h.box), [
    { role: "assistant", text: "before tool" },
    { role: "tool", label: "read_file README.md", text: "", state: "running", error: false },
  ]);
  assert.equal(vm.runInContext("bubble", c), null);
  assert.equal(vm.runInContext("answerBubbles[0].dataset.segmentSealed", c), "1");
  h.emit("delta", { delta: "after tool" }, { paint: false });
  assert.equal(vm.runInContext("raw", c), "after tool");
  assert.equal(vm.runInContext("answerBubbles.length", c), 2);
  assert.equal(vm.runInContext("bubble.hidden", c), true);
  h.flush();
  assert.deepEqual(projection(h.box).map((row) => row.role), ["assistant", "tool", "assistant"]);
  assert.equal(h.reply(), "after tool");
});

for (const source of ["live", "replay"]) {
  test("dispatcher " + source + " decodes subagent payload once through the real clock and preserves raw legacy text", () => {
    const h = source === "live" ? stream.streamHarness({ phaseClock: true }) : stream.replayHarness({ phaseClock: true });
    const c = h.context, root = h.box || h.view.box;
    const payload = JSON.stringify(stream.structuredSubagents([stream.structuredTask("decoded-worker", 5000)], "producer text"));
    let reads = 0;
    c.JSON = { ...JSON, parse(text, ...args) { if (text === payload) reads++; return JSON.parse(text, ...args); }, stringify: JSON.stringify };
    h.emit("subagent_progress", payload);
    assert.equal(reads, 1);
    const card = records(root).find((node) => node.dataset.subagents === "true");
    assert.ok(card);
    assert.equal(foldParts(card).body.textContent, "producer text");
    assert.deepEqual(Array.from(c.executionState().taskStates, ([id, active]) => [id, active]), [["decoded-worker", true]]);
    h.emit("subagent_progress", payload);
    assert.equal(reads, 2, "each event is interpreted even when its wire text repeats");
    assert.equal(records(root).find((node) => node.dataset.subagents === "true"), card);
    for (const raw of ['"legacy string"', '{"untyped":"legacy object"}', '{"broken":']) {
      h.emit("subagent_progress", raw);
      assert.equal(foldParts(card).body.textContent, raw);
      assert.deepEqual(Array.from(c.executionState().taskStates), []);
    }
  });
}

test("dispatcher preserves source request-done ordering and malformed input phase behavior", () => {
  for (const source of ["live", "replay"]) {
    const h = source === "live" ? stream.streamHarness() : stream.replayHarness(), c = h.context, events = [];
    c.recordPhase = (text) => events.push(text);
    if (source === "live") vm.runInContext('const originalPhase = activity.setPhase; activity.setPhase = (...args) => { recordPhase(args[0]); return originalPhase(...args); };', c);
    else {
      const phase = h.view.phase;
      h.view.phase = (...args) => { events.push(args[0]); return phase(...args); };
    }
    c.confirmShown = { id: 7 };
    c.hideConfirm = () => events.push("hide");
    h.emit("confirm_done", { id: 7 });
    assert.deepEqual(events, source === "live" ? ["Processing…", "hide"] : ["hide", "Processing…"]);
    events.length = 0;
    h.emit("input_request", '{"prompt":');
    assert.deepEqual(events, source === "live" ? ["Waiting for your input…"] : []);
  }
});
