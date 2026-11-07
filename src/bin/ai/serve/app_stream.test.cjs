// Run with: node --test src/bin/ai/serve/app_stream.test.cjs
// Execute the embedded stream state and dispatch, rather than a second renderer.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "app.html"), "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];

function nodeList(nodes = []) {
  const list = {
    ...nodes, length: nodes.length,
    item(index) { return this[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach((node, index) => callback.call(thisArg, node, index, list)); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
  return list;
}

function streamHarness({ activityOnly = false } = {}) {
  const timers = [];
  const box = {
    children: [], isConnected: true,
    appendChild(node) { if (!this.children.includes(node)) this.children.push(node); },
    insertBefore(node, before) {
      this.children = this.children.filter((child) => child !== node);
      this.children.splice(this.children.indexOf(before), 0, node);
    },
    querySelectorAll() { return nodeList(this.children.filter((child) => child.summary)); },
  };
  function node() {
    const classes = new Set();
    const item = { dataset: {}, textContent: "", innerHTML: "", classList: {
      add: (name) => classes.add(name), remove: (name) => classes.delete(name),
    }, get isConnected() { return box.isConnected && box.children.includes(item); },
    remove() { box.children = box.children.filter((child) => child !== item); } };
    return item;
  }
  const clock = { now: 1000 }, intervals = new Map(), listeners = new Map(), observers = [];
  let intervalId = 0;
  const listen = (type, fn) => listeners.set(type, (listeners.get(type) || []).concat(fn));
  const document = { hidden: false, body: box, addEventListener: listen };
  const context = vm.createContext({
    box, sid: "s", sessionId: "s", finished: false, turnError: false,
    runEpoch: 0, eventFailures: 0, confirmPollSeq: 0,
    setTimeout: (fn, delay) => { timers.push(delay); return timers.length; }, clearTimeout() {},
    shouldFollowOutput: () => false, followOutput() {},
    renderMarkdown: (text) => text, decorateMath() {}, decorateAndLoad() {},
    showConfirm() {}, hideConfirm() {}, confirmShown: null,
    addMsgMd() { const item = node(); box.children.push(item); return item; },
    addFold(title, text) {
      const item = node(), summary = { textContent: title }, details = { open: false };
      const body = { textContent: text };
      item.className = "msg asst";
      item.summary = summary; item.body = body;
      item.querySelector = (selector) => selector === "summary" ? summary : details;
      box.children.push(item);
      return { box: item, body, set(label, value) { summary.textContent = label; body.textContent = value; } };
    },
    ...(activityOnly ? {
      Date: { now: () => clock.now }, document, window: { addEventListener: listen },
      MutationObserver: class { constructor(fn) { observers.push(fn); } observe() {} },
      setInterval(fn, delay) { const id = ++intervalId; intervals.set(id, { fn, delay }); return id; },
      clearInterval(id) { intervals.delete(id); },
    } : {}),
  });
  vm.runInContext(script.slice(script.indexOf("function toolArgsOf("), script.indexOf("const FOLD_ROLE_LABELS")), context);
  vm.runInContext(script.slice(script.indexOf("function createThinkingFold("), script.indexOf("// The three override picks")), context);
  // Activity tests reuse the fold DOM without executing the transport slices.
  if (activityOnly) {
    const activity = context.createLiveActivity(box, null);
    const mutate = (addedNodes = []) => observers.forEach((fn) => fn([{ addedNodes }]));
    return {
      box, intervals,
      progress: (text) => activity.subagents(text), finish: (reason) => activity.finish(reason),
      newActivity: () => context.createLiveActivity(box, null),
      step(ms = 1000) { clock.now += ms; for (const { fn } of [...intervals.values()]) fn(); },
      visible(value) { document.hidden = !value; this.fire("visibilitychange"); },
      fire(type) { for (const fn of listeners.get(type) || []) fn(); },
      detach() { box.isConnected = false; mutate(); },
      attach() { box.isConnected = true; mutate([box]); },
      remove(item) { item.remove(); mutate(); },
      titles: () => box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
      bodies: () => box.children.filter((item) => item.body).map((item) => item.body.textContent),
    };
  }
  vm.runInContext(script.slice(script.indexOf("function replyPaintDelay("), script.indexOf("function sessionTitle(")), context);
  const start = script.indexOf('  let bubble = addMsgMd("", box);');
  assert.ok(start > 0);
  vm.runInContext(script.slice(start, script.indexOf("  let turnStarted = false;", start)), context);
  // The watchdogs are `const` in the real script, so they are script-scoped
  // there. As globals here they are reachable from the block loop, which is
  // what arms them for every event, keep-alive comments included.
  const wdStart = script.indexOf("  const STREAM_STALL_MS");
  const wdEnd = script.indexOf("  const goStale", wdStart);
  assert.ok(wdStart > 0 && wdEnd > wdStart);
  vm.runInContext(
    script.slice(wdStart, wdEnd)
      .replace(/\b(const|let) (STREAM_STALL_MS|STREAM_SILENCE_MS|armStall|armProgress|stalled|stallTimer|silenceTimer|reader|wentStale)\b/g, "var $2"),
    context,
  );
  // The block loop arms the watchdogs before it dispatches the event, so the
  // slice has to start there: a keep-alive comment must reach the watchdog.
  const dispatchStart = script.indexOf("        armStall();", start);
  const dispatchEnd = script.indexOf("      if (finished) break;", dispatchStart);
  // Remove the SSE block loop's closing brace and supply a loop for done's break.
  const dispatch = script.slice(dispatchStart, dispatchEnd).replace(/\s*}\s*$/, "");
  return {
    context,
    box,
    timers,
    connect() {
      const begin = script.indexOf("  let turnStarted = false;", start);
      const end = script.indexOf("  try {\n    const res = await fetch", begin);
      assert.ok(end > begin);
      vm.runInContext(
        script.slice(begin, end)
          .replace(/\b(const|let) (STREAM_STALL_MS|STREAM_SILENCE_MS|armStall|armProgress|stalled|stallTimer|silenceTimer|reader|wentStale)\b/g, "var $2"),
        context,
      );
    },
    emit(kind, value) {
      context.kind = kind;
      context.payload = typeof value === "string" ? value : JSON.stringify(value ?? {});
      vm.runInContext("for (;;) { " + dispatch + "\nbreak; } flushPaint();", context);
    },
    titles: () => box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
    bodies: () => box.children.filter((item) => item.body).map((item) => item.body.textContent),
    pending: () => vm.runInContext("phaseBubble.textContent", context),
    pendingVisible: () => vm.runInContext("box.children.includes(phaseBubble)", context),
    fail: () => vm.runInContext('fail("turn failed")', context),
    reply: () => vm.runInContext("(bubble || answerBubbles.at(-1)).innerHTML", context),
    answerCount: () => vm.runInContext("answerBubbles.filter((b) => b.innerHTML).length", context),
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
    reply: () => view.box.children.at(-2).textContent,
    titles: () => view.box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
  };
}

function attachmentHarness(headersOnly = false) {
  const h = replayHarness(), c = h.context, timers = new Map(), requests = [];
  h.view.remove();
  let timerId = 0, polls = 0, loads = 0;
  Object.assign(c, {
    eventActive: false, eventAbort: null, eventSession: "", recoveredView: null,
    eventStreamEnded: false, turns: new Map(), store: { base: "https://server" },
    turnRunning: true, turnRunningAt: Date.now(), RUNNING_SAMPLE_TTL_MS: 60000,
    AbortController, TextDecoder, headers: () => ({}), syncStopButton() {},
    ensureConfirmPoll() { polls++; }, stopConfirmPoll() {}, addMsg() {},
    loadHistory() { loads++; },
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(url, { signal }) {
      const request = { url, signal }; requests.push(request);
      if (headersOnly) return new Promise((resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      });
      return { ok: true, body: { getReader: () => ({ read: () => new Promise((resolve, reject) => {
        request.frame = (frame) => resolve({ value: new TextEncoder().encode(frame), done: false });
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      }) }) } };
    },
  });
  vm.runInContext(script.slice(script.indexOf("function parseBlock("), script.indexOf("async function send()")), c);
  const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  return {
    c, requests, timers, settle, polls: () => polls, loads: () => loads,
    fire(ms) {
      const timer = [...timers].find(([, item]) => item.ms === ms);
      assert.ok(timer, `missing ${ms}ms timer`);
      timers.delete(timer[0]); timer[1].fn();
    },
    async frame(text) { requests.at(-1).frame(text); await settle(); },
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
  assert.equal(h.reply(), "answer");
  assert.equal(h.status(), "Responding…");
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

for (const headersOnly of [true, false]) test(`a stalled event stream reconnects without switching screens (headers=${headersOnly})`, async () => {
  const h = attachmentHarness(headersOnly), pending = h.c.attachEvents();
  await h.settle();
  assert.ok(h.polls() > 0, "state fallback stays enabled while events are attached");
  h.fire(25000);
  await pending;
  assert.equal(h.requests[0].signal.aborted, true);
  h.fire(2000);
  await h.settle();
  assert.equal(h.requests.length, 2);
  assert.ok(h.requests.every((r) => r.url.endsWith("/events?after=0")), "only GET replay is retried, never the user POST");
  h.c.abortEvents();
  await h.settle();
  assert.equal(h.timers.size, 0);
});

test("replay heartbeats renew liveness without timing out a pending confirmation", async () => {
  const h = attachmentHarness(), pending = h.c.attachEvents();
  await h.settle();
  const first = [...h.timers].find(([, t]) => t.ms === 25000)[0];
  await h.frame(': keepalive\n\n');
  assert.equal(h.timers.has(first), false);
  await h.frame('event: confirm_request\ndata: {"id":1,"token":"wait","prompt":"continue?"}\n\n');
  h.fire(300000);
  assert.equal(h.requests[0].signal.aborted, false);
  await h.frame('event: confirm_done\ndata: {}\n\n');
  h.fire(300000);
  await pending;
  assert.equal(h.requests[0].signal.aborted, true);
});

test("replay transport failures fall back to history after a bounded retry budget", async () => {
  const h = attachmentHarness(true);
  h.c.eventFailures = 4;
  const pending = h.c.attachEvents();
  await h.settle(); h.fire(25000); await pending;
  assert.equal(h.c.eventFailures, 5);
  await h.c.attachEvents();
  assert.equal(h.requests.length, 1);
  assert.equal(h.loads(), 1);
  assert.ok(h.polls() > 0);
});

test("live phases remain visible after an early assistant answer", () => {
  const h = streamHarness();
  h.connect();
  h.emit("delta", { delta: "I will check that." });
  h.emit("tool", { id: "read", name: "read_file", phase: "started" });
  assert.match(h.pending(), /Running read_file/);
  assert.equal(h.reply(), "I will check that.");
  h.emit("tool", { id: "read", name: "read_file", phase: "finished", ok: true });
  assert.match(h.pending(), /Processing/);
  h.emit("thinking", { text: "next step" });
  assert.match(h.pending(), /Thinking/);
  assert.equal(h.reply(), "I will check that.");
});

test("prose split by a tool round renders as separate answer bubbles", () => {
  const h = streamHarness();
  h.connect();
  h.emit("delta", { delta: "first segment." });
  h.emit("answer_pending");
  h.emit("tool", { id: "read", name: "read_file", phase: "started" });
  h.emit("tool", { id: "read", name: "read_file", phase: "finished", ok: true });
  h.emit("delta", { delta: "second segment." });
  h.flush && h.flush();
  assert.equal(h.answerCount(), 2);
  const answers = h.box.children.filter((item) => item.classList && !item.summary && !item.body);
  const texts = answers.map((item) => item.innerHTML).filter(Boolean);
  assert.deepEqual(texts, ["first segment.", "second segment."]);
  const shape = h.box.children.map((item) => (item.summary ? "fold" : item.innerHTML || "status"));
  assert.deepEqual(shape, ["first segment.", "fold", "second segment.", "status"]);
});

test("replayed phases remain visible after an early assistant answer", () => {
  const h = replayHarness();
  h.emit("delta", { delta: "I will check that." });
  h.emit("tool", { id: "read", name: "read_file", phase: "started" });
  assert.equal(h.status(), "Running read_file…");
  h.emit("tool", { id: "read", name: "read_file", phase: "finished", ok: true });
  assert.equal(h.status(), "Processing…");
  h.emit("thinking", { text: "next step" });
  assert.equal(h.status(), "Thinking…");
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
  vm.runInContext(script.slice(script.indexOf("let promptComposing = false;"), script.indexOf("function composerBox()")), context);
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
  assert.equal(h.pendingVisible(), false, "a completed turn must not retain a running indicator");
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
  const scheduled = h.timers.length;
  h.emit("subagent_progress", "worker: running");
  assert.ok(h.timers.length > scheduled, "subagent progress must schedule a paint");
  h.emit("subagent_progress", "worker: finished");
  assert.equal(h.titles().length, 1);
  assert.match(h.titles()[0], /subagents/);
  assert.equal(h.bodies()[0], "worker: finished");
});

test("subagent elapsed ticker advances task rows and reanchors to server snapshots", () => {
  const h = streamHarness({ activityOnly: true });
  const snapshot = "subagents · 2 active\n├─ [backend=traecli] alpha · running 59s · task: audit\n│  ↳ progress: running 3s · last activity 2s ago\n├─ beta · active · 1m59s · task: implement\n└─ gamma · completed 7m3s · task: running 5s";
  h.progress(snapshot);
  const card = h.box.children[0];
  h.step();
  assert.equal(h.bodies()[0], snapshot.replace("running 59s", "running 1m0s").replace("active · 1m59s", "active · 2m0s"));
  assert.equal(h.intervals.size, 1);
  assert.equal([...h.intervals.values()][0].delay, 1000);
  assert.equal(h.box.children[0], card);
  assert.equal(card.className, "msg asst");
  assert.deepEqual(card.dataset, {});
  h.step(5000);
  assert.match(h.bodies()[0], /alpha · running 1m5s/);
  h.progress("subagents · 1 active\n└─ [backend=traecli] alpha · running 8s · task: audit");
  assert.match(h.bodies()[0], /running 8s/);
  h.step(2000);
  assert.match(h.bodies()[0], /running 10s/);
  h.newActivity().subagents("other: running 42s");
  h.step();
  assert.equal(h.intervals.size, 1, "all activity cards share one interval");
  assert.ok(h.bodies().some((text) => text.includes("alpha · running 11s")));
  assert.ok(h.bodies().includes("other: running 43s"));
});

test("subagent elapsed ticker preserves first-seen fallback across row and progress changes", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress("subagents · 2 active\n├─ worker · ready · task: audit\n└─ peer · running 3s · task: other");
  h.step(4000);
  h.progress("subagents · 1 active\n└─ worker · running · task: audit\n   ↳ progress: reading files");
  h.step(2000);
  assert.match(h.bodies()[0], /worker · running 6s · task: audit/);
  h.progress("subagents · 1 active\n└─ worker · running · task: audit\n   ↳ progress: editing files");
  h.step(2000);
  assert.match(h.bodies()[0], /worker · running 8s · task: audit/);
  assert.match(h.bodies()[0], /progress: editing files/);
  h.progress("subagents · 0 active");
  assert.equal(h.intervals.size, 0);
  h.progress("worker: running");
  h.step();
  assert.equal(h.bodies()[0], "worker: running 1s");
});

test("subagent elapsed ticker preserves terminal text and stops when activity finishes", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress("worker: running 1h1m");
  h.step();
  assert.equal(h.bodies()[0], "worker: running 1h1m1s");
  const terminal = "worker: completed · 1h1m7s\n   ↳ final: running 42s\n";
  h.progress(terminal);
  const title = h.titles()[0];
  assert.equal(h.intervals.size, 0);
  h.step(30000);
  assert.equal(h.bodies()[0], terminal);
  assert.equal(h.titles()[0], title);
  h.progress("worker: running 9s");
  h.step();
  h.finish("ended");
  const frozen = h.bodies()[0];
  assert.equal(h.intervals.size, 0);
  h.step(3000);
  assert.equal(h.bodies()[0], frozen);
});

test("subagent elapsed ticker pauses while hidden or detached and resumes one interval", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress("worker: running 42s");
  h.step();
  h.visible(false);
  assert.equal(h.intervals.size, 0);
  h.step(5000);
  assert.equal(h.bodies()[0], "worker: running 43s");
  h.visible(true);
  assert.equal(h.bodies()[0], "worker: running 48s");
  h.visible(true);
  assert.equal(h.intervals.size, 1);
  h.fire("pagehide");
  assert.equal(h.intervals.size, 0);
  h.step(3000);
  assert.equal(h.bodies()[0], "worker: running 48s");
  h.fire("pageshow");
  assert.equal(h.bodies()[0], "worker: running 51s");
  h.detach();
  assert.equal(h.intervals.size, 0);
  h.step(4000);
  assert.equal(h.bodies()[0], "worker: running 51s");
  h.attach();
  assert.equal(h.bodies()[0], "worker: running 55s");
  assert.equal(h.intervals.size, 1);
  h.remove(h.box.children[0]);
  assert.equal(h.intervals.size, 0);
  assert.deepEqual(h.bodies(), []);
});

test("a final replace supersedes provisional streaming text", () => {
  const h = streamHarness();
  h.emit("answer_pending");
  h.emit("delta", { delta: "provisional" });
  h.emit("replace", { text: "authoritative" });
  assert.equal(h.reply(), "authoritative");
});

test("replayed final text replaces provisional deltas", () => {
  const h = replayHarness();
  h.emit("delta", { delta: "provisional" });
  h.emit("replace", { text: "authoritative" });
  assert.equal(h.reply(), "authoritative");
  assert.equal(h.status(), "Responding…");
  h.emit("replace", { text: "" });
  assert.equal(h.reply(), "");
  assert.equal(h.status(), "Processing…");
});

test("EOF without the done event follows server history instead of declaring completion", () => {
  assert.match(script, /if \(!finished && !turnError\) \{[\s\S]*?stream ended before completion[\s\S]*?goStale/);
});

test("a finished event stream is not re-attached on every poll, but a new turn is", async () => {
  let attachCalls = 0;
  const context = vm.createContext({
    sessionId: "s", confirmBusy: false, runEpoch: 0, confirmPollSeq: 0,
    eventActive: false, eventSession: "", eventFailures: 0,
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
  assert.match(script, /if \(block\.kind === "done"\) \{\s+finished = true; eventStreamEnded = true; eventFailures = 0; confirmPollSeq\+\+;\s+break;\s+\}/);
  assert.match(script, /if \(!eventStreamEnded\) void attachEvents\(\);/);
});

test("a terminal replay leaves no running view or retry timers", async () => {
  const h = attachmentHarness(), pending = h.c.attachEvents();
  await h.settle();
  await h.frame('event: done\ndata: {}\n\n');
  await pending;
  assert.equal(h.c.eventStreamEnded, true);
  assert.equal(h.c.eventActive, false);
  assert.equal(h.c.recoveredView, null);
  assert.equal(h.timers.size, 0);
  assert.equal(h.loads(), 1);
});

test("a stale running sample cannot undo a newer idle sample", async () => {
  const h = attachmentHarness(), c = h.c, responses = [];
  Object.assign(c, { confirmBusy: false, staleSession: "", confirmShown: null,
    api: () => new Promise((resolve) => responses.push((body) => resolve({ json: async () => body }))),
  });
  vm.runInContext(script.slice(script.indexOf("async function pollConfirm()"), script.indexOf("function isTurnRunning()")), c);
  const old = c.pollConfirm(), fresh = c.pollConfirm();
  responses[1]({ running: false, pending: null });
  await fresh;
  responses[0]({ running: true, pending: null });
  assert.equal(await old, null);
  assert.equal(c.turnRunning, false);
  assert.equal(h.requests.length, 0);
});

test("reconnect does not bind input listeners twice and diagnostics never contain typed text", () => {
  const listeners = [];
  const prompt = { value: "中文 secret", style: { height: "44px" }, addEventListener: (type, fn) => listeners.push({ type, fn }) };
  const elements = { prompt, base: {}, token: {}, inputdiagnostics: {}, inputtrace: { focus() {}, select() {} } };
  elements.inputtraceclear = {};
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
  assert.equal(report.events.length, 100);
  assert.equal(report.ime.compositionInputs, 100);
  assert.equal(report.ime.hint, "");
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

test("a keep-alive comment proves the connection is alive, and only a real event counts as progress", () => {
  const h = streamHarness();
  h.connect();
  const stall = Number(/const STREAM_STALL_MS = (\d+)/.exec(script)[1]);
  const silence = Number(/const STREAM_SILENCE_MS = (\d+)/.exec(script)[1]);
  assert.ok(silence > stall, "a quiet turn must get more room than a dead connection");
  h.timers.length = 0;
  h.emit("", ""); // the keep-alive comment the server sends every 15s
  assert.ok(h.timers.includes(stall), "a keep-alive must re-arm the transport watchdog");
  assert.ok(!h.timers.includes(silence), "a keep-alive is not progress");
  h.timers.length = 0;
  h.emit("delta", { text: "hi" });
  assert.ok(h.timers.includes(stall));
  assert.ok(h.timers.includes(silence), "a real event is progress");
});

test("a feed that dies mid-turn reattaches to the retained event stream", () => {
  const at = script.indexOf("turns.delete(sid);");
  assert.ok(at > 0);
  const tail = script.slice(at, at + 900);
  assert.match(tail, /wentStale/, "the stale path must be marked");
  assert.match(tail, /attachEvents\(\)/, "the stale path must reattach to the retained stream");
  assert.match(tail, /box\.remove\(\)/, "the dead live box must be replaced by the reattached view");
});
