// Run with: env NODE_OPTIONS= node --test --test-reporter=tap src/bin/ai/serve/app_phase2_baseline.test.cjs
// Characterize the current worktree. Quirk cases document observations, not desired contracts.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function loadFixtures(file, names) {
  const filename = path.join(__dirname, file);
  // Reuse the handwritten DOM and VM helpers without registering their tests again.
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : require(name),
    __dirname, __filename: filename, console, Buffer,
    process: { env: { ...process.env, ASERVE_TEST_APP_HTML: path.join(__dirname, "app.html") } },
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

function renderingHarness() {
  const h = desktop.harness();
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

function sendHarness() {
  const h = desktop.harness(), c = h.context, timers = new Map(), replays = [], requests = [];
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
    addAnswerCopy() {},
  });
  desktop.loadChat(h);
  runSlice(c, "function toolArgsOf(", "// Which stored folds");
  runSlice(c, "function createThinkingFold(", "// The three override picks");
  runSlice(c, "function addUserBubble(", "function askRewind(");
  runSlice(c, "function isRemoteTurn()", "// Transient note through");
  runSlice(c, "async function reloadStale()", "// --- turn confirmation");
  runSlice(c, "function ensureConfirmPoll()", "// A pick is the moment");
  runSlice(c, "function recoveredProgressView(", "async function pollConfirm(");
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

for (const scenario of [
  { name: "EOF without done", active: true, end: (h) => h.reader.eof(),
    error: "stream ended before completion — checking server history" },
  { name: "reader rejection immediately after acceptance", active: false, end: (h) => h.reader.reject("reader rejected"),
    error: "connection lost: reader rejected — the turn keeps running on the server (following history)" },
  { name: "transport loss after a partial answer and active task snapshot", active: true, end: (h) => h.reader.reject("transport lost"),
    error: "connection lost: transport lost — the turn keeps running on the server (following history)" },
]) test("quirk: real send " + scenario.name + " reconnects but clears recovery ownership and exposes send while replay is attached", async () => {
  const h = sendHarness(); await h.start(); await h.accept();
  if (scenario.active) await activeTurn(h);
  await h.finish(() => scenario.end(h));
  // Quirk: goStale calls fail, whose real stopStaleFollow clears the owner just set.
  // The failed execution flag then outranks the newly attached replay in the gates.
  assert.deepEqual(h.snapshot(), { ...cleanState, failed: true, attached: true, sideNote: true,
    errors: [scenario.error], timeouts: [25000, 300000], intervals: [1000, 2000] });
  assert.equal(h.c.staleSession, ""); assert.equal(h.c.staleTimer, null);
  assert.equal(h.liveBox().parentNode, null, "the recorded error belongs to the retired live box");
  assert.equal(h.replays.length, 1);
  assert.equal(records(h.elements.msgs).filter((node) => node.classList.contains("err")).length, 0,
    "the retired live error is not visible in the replacement transcript");
  if (scenario.active) {
    const interrupted = toolNode(h.liveBox(), "read_file");
    assert.equal(interrupted.dataset.executionState, "completed");
    assert.equal(interrupted.dataset.executionException, "true");
  }
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

test("quirk: real send recoverEnded cancels its reader and retires the error box without arming history follow", async () => {
  const h = sendHarness(); await h.start(); await h.accept(); await activeTurn(h);
  await h.finish(() => h.entry().recover("server says the turn ended", true));
  assert.deepEqual(h.snapshot(), { ...cleanState, failed: true,
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
