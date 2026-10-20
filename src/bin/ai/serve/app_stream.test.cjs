// Run with: node --test src/bin/ai/serve/app_stream.test.cjs
// Execute the embedded stream state and dispatch, rather than a second renderer.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "app.html"), "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];

function streamHarness() {
  const box = {
    children: [],
    insertBefore(node, before) {
      this.children = this.children.filter((child) => child !== node);
      this.children.splice(this.children.indexOf(before), 0, node);
    },
  };
  function node() {
    const classes = new Set();
    const item = { dataset: {}, textContent: "", innerHTML: "", classList: {
      add: (name) => classes.add(name), remove: (name) => classes.delete(name),
    }, remove() { box.children = box.children.filter((child) => child !== item); } };
    return item;
  }
  const context = vm.createContext({
    box, sid: "s", sessionId: "s", finished: false, turnError: false,
    setTimeout: () => 1, clearTimeout() {},
    shouldFollowOutput: () => false, followOutput() {},
    renderMarkdown: (text) => text, decorateMath() {},
    showConfirm() {}, hideConfirm() {}, confirmShown: null,
    addMsgMd() { const item = node(); box.children.push(item); return item; },
    addFold(title, text) {
      const item = node(), summary = { textContent: title }, details = { open: false };
      const body = { textContent: text };
      item.summary = summary; item.body = body;
      item.querySelector = (selector) => selector === "summary" ? summary : details;
      box.children.push(item);
      return { box: item, body, set(label, value) { summary.textContent = label; body.textContent = value; } };
    },
  });
  vm.runInContext(script.slice(script.indexOf("function toolArgsOf("), script.indexOf("const FOLD_ROLE_LABELS")), context);
  vm.runInContext(script.slice(script.indexOf("function createThinkingFold("), script.indexOf("// The three override picks")), context);
  const start = script.indexOf('  const bubble = addMsgMd("", box);');
  vm.runInContext(script.slice(start, script.indexOf("  let turnStarted = false;", start)), context);
  const dispatchStart = script.indexOf('        if (kind === "done")', start);
  const dispatchEnd = script.indexOf("      if (finished) break;", dispatchStart);
  // Remove the SSE block loop's closing brace and supply a loop for done's break.
  const dispatch = script.slice(dispatchStart, dispatchEnd).replace(/\s*}\s*$/, "");
  return {
    context,
    box,
    connect() {
      const begin = script.indexOf("  let turnStarted = false;", start);
      const end = script.indexOf("  try {\n    const res = await fetch", begin);
      assert.ok(end > begin);
      vm.runInContext(script.slice(begin, end), context);
    },
    emit(kind, value) {
      context.kind = kind;
      context.payload = typeof value === "string" ? value : JSON.stringify(value ?? {});
      vm.runInContext("for (;;) { " + dispatch + "\nbreak; } flushPaint();", context);
    },
    titles: () => box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
    bodies: () => box.children.filter((item) => item.body).map((item) => item.body.textContent),
    pending: () => vm.runInContext("bubble.textContent", context),
    fail: () => vm.runInContext('fail("turn failed")', context),
    reply: () => vm.runInContext("bubble.innerHTML", context),
  };
}

function replayHarness() {
  const { context } = streamHarness();
  function node() {
    return {
      children: [], textContent: "", dataset: {}, classList: { add() {}, remove() {} },
      appendChild(child) { this.children.push(child); child.parent = this; },
      insertBefore(child, before) {
        if (child.parent) child.remove();
        this.children.splice(this.children.indexOf(before), 0, child); child.parent = this;
      },
      remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); },
      querySelector(selector) { return selector === "summary" ? this.summary : this.details; },
    };
  }
  const msgs = node();
  context.$ = () => msgs;
  context.document = { createElement: node };
  context.addFold = (title, text, parent) => {
    const box = node(); box.summary = { textContent: title }; box.details = { open: false };
    box.body = { textContent: text }; parent.appendChild(box);
    return { box, set(label, value) { box.summary.textContent = label; box.body.textContent = value; } };
  };
  context.NATIVE_PREFIX = "@@aserve-native ";
  const requests = [];
  context.handleNativeRequest = (...args) => requests.push(args);
  vm.runInContext(script.slice(script.indexOf("function recoveredProgressView("), script.indexOf("async function pollConfirm(")), context);
  const view = context.recoveredProgressView();
  return {
    context, msgs, view, requests,
    emit(kind, payload) { context.handleReplayEvent("s", kind, JSON.stringify(payload ?? {}), view); },
    status: () => view.box.children.at(-1).textContent,
    titles: () => view.box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
  };
}

test("replayed phases paint visible status and separate thinking across tools", () => {
  const h = replayHarness();
  assert.equal(h.status(), "Connecting…");
  h.emit("thinking", { text: "first phase" });
  assert.equal(h.status(), "Thinking…");
  h.emit("thinking_done");
  h.emit("tool", { id: "alarm", name: "phone_action", phase: "started", arguments: "{}" });
  assert.equal(h.status(), "Running phone_action…");
  h.emit("input_request", { id: 7, token: "t", prompt: '@@aserve-native {"action":"alarm","hour":9,"minute":30}' });
  assert.equal(h.requests.length, 1);
  assert.equal(h.status(), "正在手机上执行…");
  h.emit("input_done", { id: 7 });
  assert.equal(h.status(), "Running phone_action…");
  h.emit("tool", { id: "alarm", name: "phone_action", phase: "finished", ok: true });
  assert.equal(h.status(), "Processing…");
  h.emit("thinking", { text: "second phase" });
  assert.equal(h.titles().filter((title) => title === "thinking").length, 2);
  h.emit("delta", { delta: "answer" });
  assert.equal(h.status(), "answer");
});

test("a reattach rebuilds fresh DOM from the retained log on every connection", async () => {
  const h = replayHarness();
  h.view.remove();
  const urls = [], statuses = [];
  const c = h.context;
  Object.assign(c, {
    eventActive: false, eventAbort: null, eventSession: "", recoveredView: null,
    turns: new Map(), store: { base: "https://server" },
    turnRunning: true, turnRunningAt: Date.now(), RUNNING_SAMPLE_TTL_MS: 60000,
    AbortController, TextDecoder, headers: () => ({}),
    syncStopButton() {}, stopConfirmPoll() {}, ensureConfirmPoll() {},
    setTimeout() {}, addMsg() {}, loadHistory() { assert.fail("must keep following the active turn"); },
    async fetch(url) {
      urls.push(url);
      let read = false;
      return { ok: true, body: { getReader: () => ({ async read() {
        if (!read) {
          read = true;
          const events = url.endsWith("after=0") ? 'id: 41\nevent: thinking\ndata: {"text":"restored"}\n\n' : "";
          return { value: new TextEncoder().encode(events), done: false };
        }
        statuses.push(h.msgs.children.at(-1).children.at(-1).textContent);
        return { done: true };
      } }) } };
    },
  });
  vm.runInContext(script.slice(script.indexOf("function parseBlock("), script.indexOf("async function send()")), c);
  await c.attachEvents();
  await c.attachEvents();
  assert.deepEqual(statuses, ["Thinking…", "Thinking…"]);
  assert.equal(urls.length, 2);
  assert.ok(urls.every((url) => url.endsWith("after=0")));
});

test("pending HTTP request paints a connection indicator before any SSE event", () => {
  const h = streamHarness();
  h.connect();
  assert.equal(h.pending(), "◌ Connecting…");
  assert.deepEqual(h.titles(), []);
  h.emit("thinking", { text: "reasoning" });
  assert.match(h.pending(), /Thinking/);
});

test("restored running sessions distinguish missing live updates from active streams", () => {
  const elements = Object.fromEntries(["send", "sidenote", "composer", "turnstatus"].map((id) => [id, {
    textContent: "", hidden: true, classList: { toggle() {} }, setAttribute() {},
  }]));
  const context = vm.createContext({
    $: (id) => elements[id], sessionId: "a", turns: new Map(),
    eventActive: false, eventSession: "",
    staleSession: "", staleDone: false, sideNoteBusy: false,
    turnRunning: true, turnRunningAt: 100000, RUNNING_SAMPLE_TTL_MS: 60000,
    Date: { now: () => 100001 },
  });
  vm.runInContext(script.slice(script.indexOf("function isTurnRunning()"), script.indexOf("function note(")), context);
  context.syncStopButton();
  assert.equal(elements.turnstatus.hidden, false);
  assert.match(elements.turnstatus.textContent, /Turn running on server.*live updates unavailable/);
  // A reattached event stream is a live source even though it is not in `turns`.
  context.eventActive = true; context.eventSession = "a";
  context.syncStopButton();
  assert.equal(elements.turnstatus.hidden, true);
  context.eventActive = false; context.eventSession = "";
  assert.equal(elements.send.title, "stop this turn");
  context.turns.set("a", {});
  context.syncStopButton();
  assert.equal(elements.turnstatus.hidden, true);
  context.turns.clear(); context.turnRunningAt = 0; context.staleSession = "a";
  context.syncStopButton();
  assert.match(elements.turnstatus.textContent, /Live connection lost/);
  context.turnRunning = false; context.staleSession = "";
  context.syncStopButton();
  assert.equal(elements.turnstatus.hidden, true);
  assert.equal(elements.send.title, "send");
  context.sessionId = "b"; context.staleSession = "a";
  context.syncStopButton();
  assert.equal(elements.turnstatus.hidden, true);
});

test("live tools separate real thinking phases, while logs cannot split tokens", () => {
  const h = streamHarness();
  h.emit("thinking_start");
  h.emit("thinking", { text: "The" });
  h.emit("progress", "request diagnostic");
  h.emit("thinking", { text: " user asks" });
  h.emit("thinking_done");
  h.emit("answer_pending");
  h.emit("tool", { id: "t1", name: "web_search", phase: "started", arguments: '{"query":"DeepSeek"}' });
  assert.match(h.pending(), /Running web_search/);
  assert.match(h.titles().join("\n"), /web_search DeepSeek/);
  h.emit("answer_pending");
  assert.match(h.pending(), /Running web_search/);
  h.emit("tool", { id: "t1", name: "web_search", phase: "finished", output: "search results" });
  h.emit("thinking_start");
  h.emit("thinking", { text: "Next thought" });
  h.emit("progress", "another diagnostic");
  assert.deepEqual(h.titles().filter((title) => title.startsWith("thinking")), ["thinking", "thinking…"]);
  assert.equal(h.titles().filter((title) => title.startsWith("activity log")).length, 1);
  assert.ok(h.bodies().includes("The user asks"));
  assert.ok(h.bodies().some((text) => text.includes("search results")));
  assert.ok(h.titles().findIndex((title) => title.includes("web_search")) < h.titles().lastIndexOf("thinking…"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("done");
  assert.equal(h.reply(), "Final answer");
});

test("parallel tool completion and confirmation preserve the remaining active status", () => {
  const h = streamHarness();
  for (const id of ["a", "b"]) h.emit("tool", { id, name: "read_file", phase: "started", arguments: JSON.stringify({ file_path: id }) });
  h.emit("confirm_request", { id: "confirm", prompt: "approve?" });
  assert.match(h.pending(), /Waiting for confirmation/);
  h.emit("confirm_done", { id: "confirm" });
  h.emit("tool", { id: "b", name: "read_file", phase: "finished", output: "B" });
  assert.match(h.pending(), /Running read_file…/);
  h.emit("tool", { id: "b", name: "read_file", phase: "finished", output: "B" });
  assert.equal(h.titles().length, 2);
  h.fail();
  assert.ok(h.titles().some((title) => title.includes("interrupted")));
  assert.ok(h.bodies().some((text) => text.includes("B")));
});

test("subagent snapshots stay distinct from logs and empty thinking markers", () => {
  const h = streamHarness();
  h.emit("thinking_start"); h.emit("thinking_done");
  h.emit("subagent_progress", "worker: running");
  h.emit("subagent_progress", "worker: finished");
  assert.equal(h.titles().length, 1);
  assert.match(h.titles()[0], /subagents/);
  assert.equal(h.bodies()[0], "worker: finished");
});

test("EOF without the done event follows server history instead of declaring completion", () => {
  assert.match(script, /if \(!finished && !turnError\) \{[\s\S]*?stream ended before completion[\s\S]*?goStale/);
});

test("a finished event stream is not re-attached on every poll, but a new turn is", async () => {
  let attachCalls = 0;
  const context = vm.createContext({
    sessionId: "s", confirmBusy: false, runEpoch: 0,
    turnRunning: false, turnRunningAt: 0, turns: new Map(), staleSession: "",
    confirmShown: null, eventStreamEnded: false, NATIVE_PREFIX: "@@",
    api: async () => ({ json: async () => ({ running: true, pending: null }) }),
    armHistoryFollow() {}, ensureConfirmPoll() {}, stopConfirmPoll() {},
    attachEvents() { attachCalls++; },
    syncStopButton() {}, showConfirm() {}, hideConfirm() {}, handleNativeRequest() {},
  });
  const start = script.indexOf("async function pollConfirm()");
  const end = script.indexOf("function isTurnRunning()", start);
  assert.ok(start > 0 && end > start);
  vm.runInContext(script.slice(start, end), context);
  // The server reports a running turn: the page attaches to the event stream.
  await vm.runInContext("pollConfirm()", context);
  assert.equal(attachCalls, 1);
  // The stream closed with the server's terminal marker, so the turn it
  // belonged to is over. Polling again must not re-attach: that would rebuild
  // the progress view and re-render the transcript on every poll.
  vm.runInContext("eventStreamEnded = true", context);
  await vm.runInContext("pollConfirm()", context);
  assert.equal(attachCalls, 1);
  // The turn ends server-side and a new one starts: attach to the new stream.
  vm.runInContext("turnRunning = false; eventStreamEnded = false", context);
  await vm.runInContext("pollConfirm()", context);
  assert.equal(attachCalls, 2);
});

test("the terminal marker ends the reattach loop instead of rescheduling it", () => {
  assert.match(script, /if \(block\.kind === "done"\) \{ finished = true; eventStreamEnded = true; break; \}/);
  assert.match(script, /if \(!eventStreamEnded\) void attachEvents\(\);/);
});

test("reconnect does not bind input listeners twice and diagnostics never contain typed text", () => {
  const listeners = [];
  const prompt = { value: "中文 secret", style: { height: "44px" }, addEventListener: (type, fn) => listeners.push({ type, fn }) };
  const elements = { prompt, base: {}, token: {}, inputdiagnostics: {}, inputtrace: { focus() {}, select() {} } };
  const context = vm.createContext({
    $: (id) => elements[id], localStorage: { getItem: () => "" }, showStored() {}, syncStopButton() {},
    document: { activeElement: prompt, addEventListener() {} },
    window: { innerHeight: 600, addEventListener() {} }, navigator: { userAgent: "WebView-test" },
  });
  const start = script.indexOf("function createInputTrace(");
  const end = script.indexOf('  $("send").onclick', start);
  vm.runInContext(script.slice(start, end) + "}\nwire(); wire();", context);
  assert.equal(listeners.filter((item) => item.type === "input").length, 1);
  const input = listeners.find((item) => item.type === "input").fn;
  for (let i = 0; i < 100; i++) input({ type: "input", data: "secret", key: "中", isComposing: true, inputType: "insertCompositionText" });
  elements.inputdiagnostics.onclick();
  const report = JSON.parse(elements.inputtrace.value);
  assert.equal(report.events.length, 80);
  assert.equal(report.events[0].composing, true);
  assert.equal(report.events[0].focused, true);
  assert.equal(report.events[0].length, prompt.value.length);
  assert.doesNotMatch(elements.inputtrace.value, /secret|中文/);
});

test("failed cached tool result is not displayed as success and truncated previews are labeled", () => {
  const h = streamHarness();
  h.emit("tool", { id: "t", name: "read_file", phase: "started", arguments: "{", arguments_truncated: true });
  h.emit("tool", { id: "t", name: "read_file", phase: "finished", ok: false, cached: true, output: "failed", output_truncated: true });
  assert.match(h.titles()[0], /^! read_file · cached$/);
  assert.match(h.bodies()[0], /full output in history/);
  assert.doesNotMatch(h.pending(), /Running/);
});
