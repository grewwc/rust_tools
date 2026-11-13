// Run with: env NODE_OPTIONS= node --test --test-reporter=tap src/bin/ai/serve/app_phase3_stream_framer.test.cjs
// Drive the real send and attachEvents functions through controlled readers.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function loadFixtures(file, names) {
  const directory = process.env.ASERVE_FRAMER_FIXTURE_DIR || __dirname;
  const filename = path.join(directory, file);
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : require(name),
    __dirname: directory, __filename: filename, console, Buffer,
    process: { env: { ...process.env, ASERVE_TEST_APP_HTML: process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html") } },
    TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
    setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
  });
}
const liveFixtures = loadFixtures("app_phase2_baseline.test.cjs", ["sendHarness", "records", "answerText"]);
const replayFixtures = loadFixtures("app_stream.test.cjs", ["attachmentHarness"]);
const plain = (value) => JSON.parse(JSON.stringify(value));
const bytes = (value) => Buffer.isBuffer(value) ? value : Buffer.from(value);
const frame = (kind, value) => (kind ? "event: " + kind + "\n" : "") +
  "data: " + (typeof value === "string" ? value : JSON.stringify(value)) + "\n\n";
const delta = (value) => frame("delta", { delta: value });
const doneFrame = frame("done", {});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
async function settle() {
  for (let i = 0; i < 64; i++) await Promise.resolve();
}
function controlledReader(trace, name) {
  let pending = null, reads = 0, cancellations = 0;
  return {
    read() {
      assert.equal(pending, null, "one outstanding read per real stream");
      reads++;
      trace.push(["read", name, reads]);
      pending = deferred();
      return pending.promise;
    },
    deliver(result) {
      assert.ok(pending, "the real page must still be awaiting this read");
      trace.push(["deliver", name, Boolean(result.done), result.value === undefined ? null : bytes(result.value).toString("hex")]);
      const next = pending; pending = null; next.resolve(result);
    },
    reject(message) {
      assert.ok(pending, "the real page must still be awaiting the rejected read");
      trace.push(["reject", name, message]);
      const next = pending; pending = null; next.reject(new Error(message));
    },
    cancel() {
      cancellations++;
      trace.push(["cancel", name, cancellations]);
      if (pending) this.deliver({ done: true });
      return Promise.resolve();
    },
    pending: () => pending !== null,
    reads: () => reads,
    cancellations: () => cancellations,
  };
}

function observeDom(document, elements, trace) {
  const ids = new WeakMap(), nodes = [];
  const scalar = (value) => {
    if (value && typeof value === "object") {
      if (ids.has(value)) return { node: ids.get(value) };
      if (Array.isArray(value)) return value.map(scalar);
      if ("tagName" in value || "children" in value) return { node: ids.get(decorate(value)) };
      return "object";
    }
    if (typeof value === "function") return "callback";
    return value === undefined ? null : value;
  };
  function decorate(node) {
    if (!node || ids.has(node)) return node;
    const id = nodes.length + 1;
    ids.set(node, id); nodes.push(node);
    for (const key of ["dataset", "style", "attrs"]) {
      node[key] = new Proxy(node[key] || {}, {
        set(target, name, value) {
          trace.push(["dom", id, key, String(name), scalar(value)]);
          target[name] = value; return true;
        },
        deleteProperty(target, name) {
          trace.push(["dom-delete", id, key, String(name)]);
          return delete target[name];
        },
      });
    }
    for (const name of ["className", "textContent", "innerHTML", "value", "hidden", "disabled", "title", "scrollTop",
      "parentNode", "open", "selectionStart", "selectionEnd"]) {
      const descriptor = Object.getOwnPropertyDescriptor(node, name);
      if (descriptor && !descriptor.configurable) continue;
      let stored = descriptor?.value;
      Object.defineProperty(node, name, {
        configurable: true, enumerable: descriptor?.enumerable ?? true,
        get() { return descriptor?.get ? descriptor.get.call(node) : stored; },
        set(value) {
          trace.push(["dom", id, name, scalar(value)]);
          if (descriptor?.set) descriptor.set.call(node, value); else stored = value;
        },
      });
    }
    for (const name of ["add", "remove", "toggle"]) {
      const original = node.classList?.[name];
      if (original) node.classList[name] = (...args) => {
        trace.push(["dom", id, "classList." + name, ...args.map(scalar)]);
        return original.apply(node.classList, args);
      };
    }
    for (const name of ["appendChild", "insertBefore", "replaceChild", "remove", "setAttribute", "removeAttribute", "focus"]) {
      const original = node[name];
      if (original) node[name] = (...args) => {
        args.forEach((arg) => { if (arg && typeof arg === "object" && "children" in arg) decorate(arg); });
        trace.push(["dom", id, name, ...args.map(scalar)]);
        return original.apply(node, args);
      };
    }
    for (const child of node.children || []) decorate(child);
    return node;
  }
  for (const node of Object.values(elements)) decorate(node);
  decorate(document.body);
  const create = document.createElement;
  document.createElement = (tag) => {
    const node = decorate(create(tag));
    trace.push(["dom", ids.get(node), "createElement", tag]);
    return node;
  };
  return {
    scalar,
    snapshot() {
      return plain({
        nodes: nodes.map((node) => ({ id: ids.get(node), tag: node.tagName || null, children: (node.children || []).map((child) => ids.get(child)),
          parent: ids.get(node.parentNode) || null, text: node.textContent, html: node.innerHTML || "", className: node.className,
          hidden: node.hidden, disabled: node.disabled, value: node.value, title: node.title, open: node.open,
          scrollTop: node.scrollTop, dataset: { ...node.dataset }, attrs: { ...node.attrs }, style: { ...node.style } })),
        input: elements.prompt ? { value: elements.prompt.value, selectionStart: elements.prompt.selectionStart,
          selectionEnd: elements.prompt.selectionEnd, style: { ...elements.prompt.style } } : null,
        focus: ids.get(document.activeElement) || null,
        roots: Object.entries(elements).map(([key, node]) => [key, ids.get(node)]),
      });
    },
  };
}

function streamFramerHarness(source, { phone = false, follow = false } = {}) {
  const live = source === "live";
  const base = live ? liveFixtures.sendHarness() : replayFixtures.attachmentHarness(false, { phone });
  const c = base.c, document = live ? base.document : c.document;
  const root = live ? base.elements.msgs : document.body;
  const elements = live ? base.elements : { msgs: root };
  const trace = [], parsed = [], timers = new Map(), controllers = [], requests = [], secondary = [];
  const clock = { now: 1000 };
  let timerId = 0, entry, view, returned = false, returnValue, failure = null, accepted = false, pending;
  const headers = deferred(), reader = controlledReader(trace, "primary");
  c.Date = { now: () => clock.now };
  c.turnRunningAt = clock.now;
  c.localStorage.setItem("serve.client", "c-framer");
  if (elements.prompt) {
    elements.prompt.value = "framer request";
    elements.prompt.selectionStart = 2; elements.prompt.selectionEnd = 2;
    document.activeElement = elements.prompt;
  } else document.activeElement = root;
  root.scrollTop = 137;
  const dom = observeDom(document, elements, trace);
  for (const [name, type] of [["setTimeout", "timeout"], ["setInterval", "interval"]]) c[name] = (fn, delay) => {
    const id = ++timerId;
    timers.set(id, { fn, delay, at: clock.now + delay, type });
    trace.push([name, id, delay, clock.now]); return id;
  };
  for (const name of ["clearTimeout", "clearInterval"]) c[name] = (id) => {
    trace.push([name, id ?? null, clock.now]); timers.delete(id);
  };
  c.AbortController = class extends AbortController {
    constructor() { super(); this.testId = controllers.length + 1; controllers.push(this); }
    abort(...args) { trace.push(["abort", this.testId, clock.now]); return super.abort(...args); }
  };
  c.TextDecoder = class extends TextDecoder {
    constructor(...args) { super(...args); trace.push(["decoder"]); }
    decode(value, options) {
      trace.push(["decode", value === undefined ? null : bytes(value).toString("hex"), options?.stream === true]);
      return super.decode(value, options);
    }
  };
  const parseBlock = c.parseBlock;
  c.parseBlock = (raw) => {
    const block = parseBlock(raw);
    parsed.push(plain(block)); trace.push(["block", block.id, block.kind, block.payload]);
    return block;
  };
  c.shouldFollowOutput = () => follow;
  for (const name of ["followOutput", "scrollToLatest", "decorateAndLoad", "addAnswerCopy", "showConfirm", "hideConfirm",
    "handleNativeRequest", "syncStopButton", "loadInfo", "loadSessions", "loadSessionConfig", "startStaleFollow",
    "stopStaleFollow", "loadHistory", "ensureConfirmPoll"]) {
    const original = c[name];
    if (typeof original === "function") c[name] = (...args) => {
      trace.push(["effect", name, ...args.map(dom.scalar)]);
      return original(...args);
    };
  }
  const originalFetch = c.fetch;
  c.fetch = (url, options = {}) => {
    const pathname = new URL(url, "https://server").pathname;
    const stream = pathname.endsWith("/turns/stream") || pathname.endsWith("/events");
    if (!stream) return originalFetch(url, options);
    requests.push({ url, method: options.method || "GET", body: options.body ? JSON.parse(options.body) : null });
    trace.push(["fetch", pathname, options.method || "GET"]);
    if (!live || pathname.endsWith("/turns/stream")) {
      options.signal?.addEventListener("abort", () => {
        if (!accepted) headers.reject(new Error("aborted"));
        else if (reader.pending()) reader.reject("aborted");
      }, { once: true });
      return headers.promise;
    }
    const next = controlledReader(trace, "replay-" + (secondary.length + 1));
    secondary.push(next);
    options.signal.addEventListener("abort", () => { if (next.pending()) next.reject("replay aborted"); }, { once: true });
    return Promise.resolve({ ok: true, body: { getReader: () => next } });
  };
  const snapshot = () => {
    const state = c.executionState();
    return plain({
      dom: dom.snapshot(),
      state: { sid: c.sessionId, base: c.store.base, runEpoch: c.runEpoch, confirmPollSeq: c.confirmPollSeq,
        live: state.live, attached: state.attached, recovering: state.recovering, ended: state.ended, failed: state.failed,
        running: state.running, sendBusy: state.sendBusy, canSend: c.canSend?.() ?? null, canStop: c.canStop?.() ?? null,
        tasks: Array.from(state.taskStates), eventActive: c.eventActive, eventSession: c.eventSession,
        eventFailures: c.eventFailures, eventStreamEnded: c.eventStreamEnded, eventStreamFailed: c.eventStreamFailed,
        lastEventAt: entry?.lastEventAt ?? null, settled: entry?.userInput?.settled ?? null,
        controller: c.eventAbort?.testId ?? null, recovered: Boolean(c.recoveredView) },
      timers: [...timers].map(([id, timer]) => [id, timer.type, timer.delay, timer.at]),
      reader: { reads: reader.reads(), cancellations: reader.cancellations(), pending: reader.pending() },
      aborts: controllers.map((controller) => [controller.testId, controller.signal.aborted]),
      parsed: parsed.slice(), trace: trace.slice(), requests: requests.slice(),
      returned, returnValue: returned ? typeof returnValue : null, failure,
    });
  };
  return {
    c, root, elements, document, reader, parsed, trace, timers, controllers, requests, secondary, snapshot,
    entry: () => entry, view: () => view, returned: () => returned,
    async start() {
      pending = (live ? c.send() : c.attachEvents()).then((value) => { returned = true; returnValue = value; }, (error) => {
        returned = true; failure = error.message;
      });
      await settle();
      assert.equal(requests.length, 1, "the real page opens the primary stream");
      entry = c.executionState().liveEntry; view = c.recoveredView;
      if (live) assert.ok(entry, "real send owns a live entry before headers");
    },
    async accept(response) {
      accepted = true;
      headers.resolve(response || { ok: true, body: { getReader() { trace.push(["getReader"]); return reader; } } });
      await settle();
    },
    async rejectHeaders() { headers.reject(new Error("header failure")); await settle(); },
    async chunk(value, done = false) { reader.deliver({ value: bytes(value), done }); await settle(); },
    async eof() { reader.deliver({ done: true }); await settle(); },
    async rejectRead() { reader.reject("reader failure"); await settle(); },
    async paint() {
      for (const [id, timer] of [...timers]) if (timer.type === "timeout" && timer.delay < 1000) {
        timers.delete(id); clock.now = Math.max(clock.now, timer.at);
        trace.push(["fire", id, timer.delay, clock.now]); timer.fn(); await settle();
      }
    },
    async fire(delay) {
      const timer = [...timers].find(([, timer]) => timer.type === "timeout" && timer.delay === delay);
      assert.ok(timer, "the original caller arms the requested watchdog");
      timers.delete(timer[0]); clock.now = timer[1].at;
      trace.push(["fire", timer[0], delay, clock.now]); timer[1].fn(); await settle();
    },
    tick(ms) { clock.now += ms; },
    async finish() {
      if (reader.pending()) await this.eof();
      await pending; await settle();
      assert.equal(failure, null, "the page handles its own transport failures");
      assert.equal(returnValue, undefined, "both page functions retain their undefined return value");
    },
    async dispose() {
      c.abortEvents(); c.stopConfirmPoll?.(); c.stopStaleFollow?.(); c.stopEventRetry?.();
      await settle();
      timers.clear();
    },
  };
}

const utf8Frame = bytes(delta("中é🙂尾"));
const utf8Start = utf8Frame.indexOf(Buffer.from("中"));
const framingCases = [
  { name: "wire order and multiple blocks in one read", steps: [delta("one") + delta("two") + doneFrame + delta("ignored")],
    kinds: ["delta", "delta", "done"], payloads: [JSON.stringify({ delta: "one" }), JSON.stringify({ delta: "two" }), "{}"], ended: true, answer: "onetwo" },
  { name: "delimiter split across reads", steps: [delta("boundary").slice(0, -1), "\n", doneFrame], kinds: ["delta", "done"], ended: true, answer: "boundary", emptyAfter: 1 },
  { name: "every byte including UTF-8 continuation", steps: [...utf8Frame].map((byte) => Buffer.from([byte])).concat(doneFrame),
    kinds: ["delta", "done"], ended: true, answer: "中é🙂尾", oneDecoder: true },
  { name: "UTF-8 decoder retains a partial code point", steps: [utf8Frame.subarray(0, utf8Start + 1), utf8Frame.subarray(utf8Start + 1, utf8Start + 2), utf8Frame.subarray(utf8Start + 2), doneFrame],
    kinds: ["delta", "done"], ended: true, answer: "中é🙂尾", emptyAfter: 2, oneDecoder: true },
  { name: "multiline data and last id/event fields", steps: ["id: first\nid: last\nevent: ignored\nevent: delta\ndata:   {\ndata: \"delta\":\ndata: \"multiline\"}\n\n" + doneFrame],
    kinds: ["delta", "done"], payloads: ["{\n\"delta\":\n\"multiline\"}", "{}"], id: "last", ended: true, answer: "multiline" },
  { name: "empty blocks comments and plain stdout", steps: ["\n\n: keepalive\n\nstdout\n\n" + delta("visible") + doneFrame],
    kinds: ["", "", "", "delta", "done"], payloads: ["", "", "stdout", JSON.stringify({ delta: "visible" }), "{}"], ended: true, answer: "visible" },
  { name: "unknown typed events still count as progress", steps: [frame("future_event", "future") + delta("known") + doneFrame],
    kinds: ["future_event", "delta", "done"], ended: true, answer: "known" },
  { name: "kindless DONE has no terminal semantics", steps: [frame("", "[DONE]"), delta("after marker"), doneFrame],
    kinds: ["", "delta", "done"], payloads: ["[DONE]", JSON.stringify({ delta: "after marker" }), "{}"], ended: true, answer: "after marker" },
  { name: "typed done ignores its payload and later blocks", steps: [frame("done", "[DONE]") + delta("ignored")], kinds: ["done"], payloads: ["[DONE]"], ended: true },
  { name: "unterminated data tail is discarded at EOF", steps: [delta("kept") + delta("discarded").slice(0, -2)], kinds: ["delta"], ended: false, answer: "kept", noAnswer: "discarded" },
  { name: "unterminated terminal block is discarded at EOF", steps: [delta("kept") + doneFrame.slice(0, -1)], kinds: ["delta"], ended: false, answer: "kept" },
  { name: "unfinished UTF-8 at EOF is never flushed", steps: [utf8Frame.subarray(0, utf8Start + 1)], kinds: [], ended: false, noFlush: true },
  { name: "CRLF frames are not normalized", steps: [delta("discarded").replaceAll("\n", "\r\n")], kinds: [], ended: false, noAnswer: "discarded" },
  { name: "complete blocks in an EOF read are consumed", steps: [{ value: delta("EOF answer") + doneFrame + delta("ignored"), done: true }], kinds: ["delta", "done"], ended: true, answer: "EOF answer" },
  { name: "EOF read with an empty byte array", steps: [delta("kept"), { value: Buffer.alloc(0), done: true }], kinds: ["delta"], ended: false, answer: "kept" },
  { name: "server error stops before following frames", steps: [delta("before error") + frame("error", { error: "server failure" }) + delta("ignored") + doneFrame],
    kinds: ["delta", "error"], ended: true, failed: true, error: "server failure", noAnswer: "ignored" },
  { name: "malformed error uses the callers existing failure behavior", steps: [frame("error", "not json") + doneFrame], kinds: ["error"], ended: true, failed: true },
  { name: "a done block before a later rejected read never reads again", steps: [doneFrame], kinds: ["done"], ended: true },
];

function visibleAnswer(h, source) {
  if (source === "replay") return h.view()?.box.textContent || "";
  return Array.from(liveFixtures.records(h.entry().box)).filter((node) => !node.classList.contains("err"))
    .map(liveFixtures.answerText).join("");
}
async function runFramingCase(source, scenario) {
  const h = streamFramerHarness(source, scenario), checkpoints = [];
  const record = (name) => checkpoints.push({ name, snapshot: h.snapshot() });
  await h.start(); record("headers pending");
  await h.accept(); record("headers accepted");
  const answers = [];
  for (let i = 0; i < scenario.steps.length; i++) {
    const step = scenario.steps[i];
    h.tick(11);
    await h.chunk(step?.value === undefined ? step : step.value, Boolean(step?.done));
    await h.paint();
    answers.push(visibleAnswer(h, source));
    record("read " + (i + 1));
  }
  const answer = visibleAnswer(h, source);
  await h.finish(); record("finished");
  const final = h.snapshot();
  await h.dispose();
  return { source, name: scenario.name, checkpoints, final, answer, answers };
}

for (const source of ["live", "replay"]) for (const scenario of framingCases) {
  test(source + " real stream: " + scenario.name, { timeout: 5000 }, async () => {
    const result = await runFramingCase(source, scenario);
    assert.deepEqual(result.final.parsed.map((block) => block.kind), scenario.kinds);
    if (scenario.payloads) assert.deepEqual(result.final.parsed.map((block) => block.payload), scenario.payloads);
    if (scenario.id) assert.equal(result.final.parsed[0].id, scenario.id);
    const terminal = scenario.ended;
    assert.equal(result.final.state.eventStreamEnded, terminal);
    assert.equal(result.final.state.eventStreamFailed, Boolean(scenario.failed));
    if (source === "live" && !terminal) {
      assert.equal(result.final.state.recovering, true, "EOF without a terminal event keeps the turn in recovery");
      assert.equal(result.final.state.canStop, true, "transport EOF must not enable another send");
    }
    if (scenario.answer) assert.ok(result.answers.some((answer) => answer.includes(scenario.answer)), "real view renders the framed answer");
    if (scenario.noAnswer) assert.ok(result.answers.every((answer) => !answer.includes(scenario.noAnswer)), "ignored bytes never reach the real view");
    if (scenario.emptyAfter) for (const checkpoint of result.checkpoints.slice(2, 2 + scenario.emptyAfter)) assert.deepEqual(checkpoint.snapshot.parsed, []);
    const decodes = result.final.trace.filter(([kind]) => kind === "decode");
    assert.equal(decodes.length, scenario.steps.length);
    assert.ok(decodes.every((entry) => entry[1] !== null && entry[2]), "every decode retains stream:true and EOF adds no flush");
    if (scenario.oneDecoder) assert.equal(result.final.trace.filter(([kind]) => kind === "decoder").length, 1);
    if (scenario.error) assert.ok(result.final.dom.nodes.some((node) => node.text.includes(scenario.error)));
    assert.equal(result.final.returnValue, "undefined");
  });
}

for (const source of ["live", "replay"]) {
  test(source + " real stream: heartbeat registration preserves byte versus block timing", { timeout: 5000 }, async () => {
    const h = streamFramerHarness(source);
    await h.start(); await h.accept();
    const registrations = (delay) => h.trace.filter(([kind, , ms]) => kind === "setTimeout" && ms === delay).length;
    const heartbeat = source === "live" ? 60000 : 25000;
    const beforeHeartbeat = registrations(heartbeat), beforeProgress = registrations(300000);
    h.tick(11); await h.chunk("event: delta\ndata: ");
    assert.equal(registrations(heartbeat), beforeHeartbeat + (source === "replay" ? 1 : 0));
    assert.equal(registrations(300000), beforeProgress);
    const beforeEmpty = registrations(heartbeat), emptyAt = h.entry()?.lastEventAt;
    h.tick(11); await h.chunk(Buffer.alloc(0));
    assert.equal(registrations(heartbeat), beforeEmpty);
    if (source === "live") assert.equal(h.entry().lastEventAt, emptyAt + 11);
    await h.chunk('{"delta":"completed"}\n\n\n\n: comment\n\n');
    assert.equal(registrations(heartbeat), beforeEmpty + (source === "live" ? 3 : 1));
    assert.equal(registrations(300000), beforeProgress + 1);
    await h.chunk(doneFrame); await h.finish();
    assert.equal(registrations(heartbeat), beforeEmpty + (source === "live" ? 4 : 2));
    assert.equal(registrations(300000), beforeProgress + (source === "live" ? 2 : 1));
    await h.dispose();
  });
  test(source + " real stream: reader rejection follows the existing recovery path", { timeout: 5000 }, async () => {
    const h = streamFramerHarness(source);
    await h.start(); await h.accept(); await h.chunk(delta("partial")); await h.rejectRead(); await h.finish();
    assert.equal(h.reader.cancellations(), 0);
    assert.equal(h.snapshot().parsed.length, 1);
    assert.ok(h.snapshot().dom.nodes.some((node) => node.text.includes(source === "live" ? "connection lost: reader failure" : "live progress unavailable: reader failure")));
    assert.equal(h.controllers[0]?.signal.aborted ?? false, false);
    await h.dispose();
  });
  test(source + " real stream: response header rejection creates no decoder or reader", { timeout: 5000 }, async () => {
    const h = streamFramerHarness(source);
    await h.start(); await h.rejectHeaders(); await h.finish();
    assert.equal(h.reader.reads(), 0);
    assert.equal(h.trace.filter(([kind]) => kind === "decoder").length, 0);
    assert.ok(h.snapshot().dom.nodes.some((node) => node.text.includes("header failure")));
    await h.dispose();
  });
}

for (const reason of ["session", "base", "epoch", "controller", "view", "abort"]) {
  test("replay real stream: " + reason + (reason === "view" ? " replacement retains the HEAD reader" : " ownership loss discards the pending read before decoding"), { timeout: 5000 }, async () => {
    const h = streamFramerHarness("replay");
    await h.start(); await h.accept();
    if (reason === "session") h.c.sessionId = "other-session";
    if (reason === "base") h.c.store.base = "https://other-server";
    if (reason === "epoch") h.c.runEpoch++;
    if (reason === "controller") h.c.eventAbort = new h.c.AbortController();
    if (reason === "view") h.c.recoveredView = { replacement: true, remove() {} };
    if (reason === "abort") h.controllers[0].abort();
    else await h.chunk(delta("obsolete") + doneFrame);
    await h.finish();
    assert.deepEqual(h.parsed, reason === "view" ? [
      { id: "", kind: "delta", payload: '{"delta":"obsolete"}' },
      { id: "", kind: "done", payload: "{}" },
    ] : []);
    assert.equal(h.trace.filter(([kind]) => kind === "decode").length, reason === "view" ? 1 : 0);
    assert.equal(h.c.eventFailures, 0);
    if (reason === "controller") assert.equal(h.c.eventAbort, h.controllers[1]);
    if (reason === "view") assert.equal(h.c.recoveredView.replacement, true);
    await h.dispose();
  });
}

test("replay real stream: abort before headers never acquires a reader", { timeout: 5000 }, async () => {
  const h = streamFramerHarness("replay");
  await h.start(); h.controllers[0].abort(); await h.finish();
  assert.equal(h.reader.reads(), 0);
  assert.deepEqual(h.parsed, []);
  assert.equal(h.trace.filter(([kind]) => kind === "decoder").length, 0);
  assert.equal(h.c.eventFailures, 0);
  await h.dispose();
});

for (const waiting of [false, true]) test("replay real stream: progress timeout waiting=" + waiting, { timeout: 5000 }, async () => {
  const h = streamFramerHarness("replay");
  await h.start(); await h.accept();
  if (waiting) await h.chunk(frame("confirm_request", { id: 1, prompt: "Continue?", token: "fixture" }));
  await h.fire(300000);
  assert.equal(h.controllers[0].signal.aborted, !waiting);
  if (waiting) {
    assert.ok(h.timers.size);
    await h.chunk(frame("confirm_done", { id: 1 }));
    await h.fire(300000);
    assert.equal(h.controllers[0].signal.aborted, true);
  }
  await h.finish();
  assert.equal(h.c.eventFailures, 1);
  assert.ok(h.trace.some(([kind, name]) => kind === "effect" && name === "loadHistory"), "an expired running sample uses history instead of retrying");
  await h.dispose();
});

test("replay real stream: silence timeout aborts the pending reader", { timeout: 5000 }, async () => {
  const h = streamFramerHarness("replay");
  await h.start(); await h.accept(); await h.fire(25000); await h.finish();
  assert.equal(h.controllers[0].signal.aborted, true);
  assert.equal(h.reader.cancellations(), 0);
  assert.equal(h.c.eventFailures, 1);
  await h.dispose();
});

for (const delay of [60000, 300000]) test("live real stream: watchdog " + delay + " cancels its reader", { timeout: 5000 }, async () => {
  const h = streamFramerHarness("live");
  await h.start(); await h.accept(); await h.fire(delay); await h.finish();
  assert.equal(h.reader.cancellations(), 1);
  assert.equal(h.trace.filter(([kind]) => kind === "abort").length, 0);
  assert.ok(h.snapshot().dom.nodes.some((node) => node.text.includes("stream stalled")));
  await h.dispose();
});

test("live real stream: recovery cancels after an incomplete block without flushing it", { timeout: 5000 }, async () => {
  const h = streamFramerHarness("live");
  await h.start(); await h.accept(); await h.chunk(delta("discarded").slice(0, -1));
  h.entry().recover("foreground recovery", true); await h.finish();
  assert.equal(h.reader.cancellations(), 1);
  assert.deepEqual(h.parsed, []);
  assert.equal(h.secondary.length, 0);
  assert.ok(h.snapshot().dom.nodes.some((node) => node.text.includes("foreground recovery")));
  await h.dispose();
});

test("live real stream: a replacement live entry still decodes the HEAD pending read", { timeout: 5000 }, async () => {
  const h = streamFramerHarness("live");
  await h.start(); await h.accept();
  const replacement = { box: h.document.createElement("div"), lastEventAt: 2000 };
  h.c.updateExecutionModel({ kind: "live", entry: replacement });
  await h.chunk(delta("obsolete") + doneFrame); await h.finish();
  assert.deepEqual(h.parsed, [
    { id: "", kind: "delta", payload: '{"delta":"obsolete"}' },
    { id: "", kind: "done", payload: "{}" },
  ]);
  assert.equal(h.trace.filter(([kind]) => kind === "decode").length, 1);
  assert.equal(h.c.executionState().liveEntry, undefined);
  assert.ok(h.snapshot().dom.nodes.some((node) => node.text.includes("obsolete")));
  h.c.updateExecutionModel({ kind: "live", entry: null });
  await h.dispose();
});
