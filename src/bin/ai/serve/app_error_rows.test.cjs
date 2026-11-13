// Run with: node --test src/bin/ai/serve/app_error_rows.test.cjs
// Exercise the embedded client and DOM ownership, using the existing VM harness pattern.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const script = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];
function source(start, end) {
  const first = script.indexOf(start), last = script.indexOf(end, first);
  assert.ok(first >= 0 && last > first, "missing client functions: " + start);
  return script.slice(first, last);
}

function nodeList(nodes) {
  return {
    ...nodes, length: nodes.length,
    item(index) { return nodes[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach(callback, thisArg); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
}

function harness({ realApi = false, phone = false } = {}) {
  function element() {
    let text = "", markup = "";
    const node = {
      children: [], dataset: {}, className: "", parentNode: null,
      scrollTop: 0, scrollHeight: 200, clientHeight: 100,
      get lastElementChild() { return this.children.at(-1) ?? null; },
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        for (const child of this.children) child.parentNode = null;
        this.children = [];
        text = String(value);
        markup = "";
      },
      get innerHTML() { return markup; },
      set innerHTML(value) {
        this.textContent = String(value).replace(/&(amp|lt|gt|quot);/g,
          (_, key) => ({ amp: "&", lt: "<", gt: ">", quot: '"' })[key]);
        markup = String(value);
      },
      insertBefore(child, anchor) {
        child.remove();
        const index = anchor === null ? this.children.length : this.children.indexOf(anchor);
        assert.ok(index >= 0, "the insertion anchor must be mounted");
        this.children.splice(index, 0, child);
        child.parentNode = this;
        return child;
      },
      appendChild(child) { return this.insertBefore(child, null); },
      remove() {
        if (!this.parentNode) return;
        const siblings = this.parentNode.children;
        siblings.splice(siblings.indexOf(this), 1);
        this.parentNode = null;
      },
      querySelectorAll(selector) {
        const matches = [];
        for (const child of this.children) {
          if (child.classList.contains(selector.slice(1))) matches.push(child);
          matches.push(...child.querySelectorAll(selector));
        }
        return nodeList(matches);
      },
      querySelector(selector) { return this.querySelectorAll(selector).item(0); },
    };
    node.classList = { contains: (name) => node.className.split(/\s+/).includes(name) };
    return node;
  }

  const root = element(), prompt = { value: "unsent draft" }, promptlist = element();
  const timers = new Map(), renderErrors = [];
  let timerId = 0, historyError = null;
  const messages = [
    { role: "user", content: "prompt" },
    { role: "assistant", content: "reply" },
  ];
  const context = vm.createContext({
    document: { createElement: element, visibilityState: "visible" },
    window: { matchMedia: () => ({ matches: !phone }) },
    navigator: { onLine: true },
    console: { error: (...args) => renderErrors.push(args) },
    $: (id) => {
      assert.ok(["msgs", "prompt", "promptlist"].includes(id), id);
      return { msgs: root, prompt, promptlist }[id];
    },
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    sessionId: "s", historySession: "", historyKeys: [], historyLoadSeq: 0,
    canonicalHistoryTotal: 0, lastMsgCount: null,
    staleKey: "", staleDone: false, staleBox: null,
    pendingUserInputs: new Map(), turns: new Map(),
    eventActive: false, eventAbort: null, eventSession: "", eventFailures: 0,
    eventStreamEnded: false, recoveredView: null, runEpoch: 0, confirmPollSeq: 0,
    turnRunning: true, turnRunningAt: Date.now(), RUNNING_SAMPLE_TTL_MS: 60000,
    confirmShown: null, store: { base: "https://server.test" },
    AbortController, TextDecoder,
    headers: () => ({}),
    api: async (url) => {
      assert.equal(url, "/sessions/s/history?limit=801");
      if (historyError) throw historyError;
      return { headers: { get: () => String(messages.length) }, json: async () => messages };
    },
    shouldFollowOutput: () => false,
    followOutput() {}, updateQuestionButton() {}, syncStopButton() {}, renderSessionBar() {},
    ensureConfirmPoll() {}, armHistoryFollow() {}, revokePreviews() {}, decorateAndLoad() {}, addAnswerCopy() {},
    createTurnRateLine: () => () => {},
    textOf: (content) => content || "", isRuntimeSyntheticUser: () => false,
  });
  for (const [start, end] of [
    ["// BEGIN execution model", "// END execution model"],
    ["const esc =", "// --- markdown"],
    ["function addMsg(", "// --- server-side media"],
    ["function histKey(", "// A session the server reports"],
    ["function attachLiveBox(", "// A turn the user has left"],
    ["// BEGIN history retry", "// Historical user inputs"],
    ["function handleReplayEvent(", "async function pollConfirm("],
  ]) vm.runInContext(source(start, end), context);
  if (script.includes("function createSseFramer()")) {
    vm.runInContext(source("function createSseFramer()", "// `/cd` changes"), context);
  }
  if (realApi) vm.runInContext(source("async function api(", "const esc ="), context);
  Object.assign(context, {
    renderMarkdown: vm.runInContext("esc", context),
    historyEndsUnanswered: (rows) => rows.at(-1)?.role === "user",
    addUserBubble: (text, index, parent) => context.addMsg("user", text, "user", parent),
    recoveredProgressView() {
      const box = element();
      box.className = "turn recovered";
      root.appendChild(box);
      return { box, phase() {}, remove: () => box.remove() };
    },
  });

  const rows = (prefix) => root.children.filter((row) => row.dataset.role === "err"
    && row.textContent.startsWith(prefix));
  return {
    context, root, prompt, messages, timers, renderErrors, rows,
    deadlineMs: vm.runInContext("HISTORY_FETCH_TIMEOUT_MS", context),
    fireTimer(delay) {
      const entry = [...timers].find(([, timer]) => timer.delay === delay);
      assert.ok(entry, "missing timer with delay " + delay);
      timers.delete(entry[0]);
      entry[1].callback();
    },
    liveRows: () => rows("live progress unavailable:"),
    historyRows: () => rows("history failed:"),
    async failLive(message) {
      context.fetch = async () => { throw new Error(message); };
      await context.attachEvents();
    },
    async connectLive() {
      let finishRead, markReading;
      const reading = new Promise((resolve) => { markReading = resolve; });
      context.fetch = async () => ({
        ok: true,
        body: { getReader: () => ({ read: () => new Promise((resolve) => {
          finishRead = resolve;
          markReading();
        }) }) },
      });
      const connection = context.attachEvents();
      await reading;
      return { close: async () => { finishRead({ done: true }); await connection; } };
    },
    async failHistory(message) { historyError = new Error(message); await context.loadHistory(); },
    async recoverHistory() {
      historyError = null;
      await context.loadHistory();
      assert.deepEqual(renderErrors, [], "history recovery must actually render without an exception");
    },
    setHistoryFailure(message) { historyError = new Error(message); },
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((finish) => { resolve = finish; });
  return { promise, resolve };
}

function historyResponse(messages, total = String(messages.length)) {
  return { ok: true, headers: { get: () => total }, json: async () => messages };
}

function hangHistory(h, stage) {
  const pending = deferred(), signals = [];
  let calls = 0, bodyReads = 0;
  h.context.fetch = async (url, options) => {
    assert.equal(url, "https://server.test/sessions/s/history?limit=801");
    signals.push(options.signal);
    calls++;
    if (calls > 1) return historyResponse(h.messages);
    // Deliberately ignore abort to prove the deadline itself releases loading.
    if (stage === "headers") return pending.promise;
    return {
      ...historyResponse([]), ok: stage !== "error body", status: 503, statusText: "Unavailable",
      json() { bodyReads++; return pending.promise; },
    };
  };
  return {
    signals, calls: () => calls, bodyReads: () => bodyReads,
    settleLate() {
      const stale = [{ role: "assistant", content: "obsolete response" }];
      pending.resolve(stage === "headers" ? historyResponse(stale)
        : stage === "body" ? stale : { error: "obsolete error" });
    },
  };
}

for (const stage of ["headers", "body", "error body"]) for (const mounted of [false, true]) {
  test(`history ${stage} timeout automatically recovers without navigation (mounted=${mounted})`, async () => {
    const h = harness({ realApi: true });
    if (mounted) await h.context.loadHistory({ sessionId: "s", messages: h.messages, histTotal: 2 });
    const retained = h.root.children.slice(), loadKey = h.context.historyLoadState("s").key;
    const request = hangHistory(h, stage);
    const loading = h.context.loadHistory();
    await new Promise(setImmediate);
    assert.ok(h.context.historyLoadState("s").loading);
    assert.deepEqual(h.root.children, retained, "do not clear mounted content while fetching");
    assert.equal(request.bodyReads(), stage === "headers" ? 0 : 1, "reach the intended waiting stage");
    assert.equal(request.signals[0].aborted, false);
    h.fireTimer(h.deadlineMs);
    await loading;
    const state = h.context.historyLoadState("s");
    assert.equal(request.signals[0].aborted, true, "abort the underlying request");
    assert.equal(state.loading, 0, "timeout must release the loading lock");
    assert.equal(state.key, loadKey);
    assert.equal(h.historyRows().length, 1, "record exactly one failure");
    assert.equal(h.historyRows()[0].textContent, "history failed: history request timed out");
    assert.deepEqual(h.root.children.slice(0, retained.length), retained);
    assert.equal(h.context.sessionId, "s");
    assert.equal(h.prompt.value, "unsent draft");
    assert.deepEqual([...h.timers.values()].map((timer) => timer.delay), [1000]);

    h.fireTimer(1000);
    await new Promise(setImmediate);
    assert.equal(request.calls(), 2, "the scheduled retry fetches the same session");
    assert.equal(state.loading, 0);
    assert.equal(state.failure, null);
    assert.equal(h.historyRows().length, 0);
    assert.deepEqual(h.root.children.map((node) => node.textContent), ["prompt", "reply"]);
    assert.deepEqual(h.root.children.slice(0, retained.length), retained, "reuse mounted transcript nodes on recovery");
    assert.equal(h.context.historySession, "s");
    assert.equal(h.context.canonicalHistoryTotal, 2);
    assert.equal(h.context.sessionId, "s");
    assert.equal(h.prompt.value, "unsent draft");
    assert.equal(h.timers.size, 0, "success clears the deadline and retry");
    assert.equal(request.signals[1].aborted, false);
    assert.deepEqual(h.renderErrors, []);

    const recovered = h.root.children.slice();
    request.settleLate();
    await new Promise(setImmediate);
    assert.deepEqual(h.root.children, recovered, "a late timed-out response cannot overwrite the retry");
    assert.equal(state.failure, null, "a late error body cannot recreate the failure");
    assert.equal(h.timers.size, 0);
  });
}

for (const stage of ["headers", "body", "error body"]) {
  test(`obsolete history ${stage} timeout cannot affect a newly selected session`, async () => {
    const h = harness({ realApi: true }), request = hangHistory(h, stage);
    const loading = h.context.loadHistory();
    await new Promise(setImmediate);
    h.context.sessionId = "other";
    await h.context.loadHistory({ sessionId: "other", messages: [{ role: "assistant", content: "current session" }], histTotal: 1 });
    const current = h.root.children.slice();
    h.fireTimer(h.deadlineMs);
    await loading;
    assert.equal(request.signals[0].aborted, true);
    assert.equal(h.context.historyLoadState("s").loading, 0);
    assert.equal(h.context.historyLoadState("s").failure, null, "superseded failures are not recorded");
    request.settleLate();
    await new Promise(setImmediate);
    assert.deepEqual(h.root.children, current);
    assert.equal(h.context.historySession, "other");
    assert.equal(h.context.sessionId, "other");
    assert.equal(h.prompt.value, "unsent draft");
    assert.equal(h.context.historyLoadState("other").failure, null);
    assert.equal(h.timers.size, 0, "obsolete loads cannot schedule a retry in the selected session");
    assert.deepEqual(h.renderErrors, []);
  });
}

for (const outcome of ["success", "missing total", "invalid total", "API error", "invalid body"]) {
  test(`settled history ${outcome} clears its deadline`, async () => {
    const h = harness({ realApi: true });
    let signal;
    h.context.fetch = async (_, options) => {
      signal = options.signal;
      if (outcome === "API error") return { ok: false, status: 503, json: async () => ({ error: "try later" }) };
      const response = historyResponse(h.messages, outcome === "missing total" ? null : outcome === "invalid total" ? "invalid" : "2");
      if (outcome === "invalid body") response.json = async () => { throw new Error("bad JSON"); };
      return response;
    };
    await h.context.loadHistory();
    assert.equal(signal.aborted, false, "normal settlement must not abort a completed request");
    assert.equal(h.context.historyLoadState("s").loading, 0);
    assert.ok(![...h.timers.values()].some((timer) => timer.delay === h.deadlineMs));
    if (outcome === "API error" || outcome === "invalid body") {
      assert.equal(h.historyRows().length, 1);
      assert.equal(h.historyRows()[0].textContent, "history failed: " + (outcome === "API error" ? "503 try later" : "bad JSON"));
      assert.deepEqual([...h.timers.values()].map((timer) => timer.delay), [1000]);
    } else {
      assert.deepEqual(h.root.children.map((node) => node.textContent), ["prompt", "reply"]);
      assert.equal(h.context.canonicalHistoryTotal, 2, "preserve the missing/invalid-header fallback to fetched length");
      assert.equal(h.timers.size, 0);
    }
    assert.deepEqual(h.renderErrors, []);
  });
}

test("consecutive live-progress request failures update one transient row", async () => {
  const h = harness();
  await h.failLive("network error");
  const row = h.liveRows()[0];
  assert.equal(h.liveRows().length, 1);
  await h.failLive("Failed to fetch");
  const rows = h.liveRows();
  assert.equal(rows.length, 1);
  assert.equal(rows[0], row, "update the mounted warning without moving it");
  assert.equal(row.textContent, "live progress unavailable: Failed to fetch (2 failures)");
  assert.ok(row.classList.contains("live-progress-warning"));
});

for (const phone of [false, true]) for (const failure of ["throw", "500"]) test(`live-progress reconnect clears the failed request warning (phone=${phone}, failure=${failure})`, async () => {
  const h = harness({ phone });
  if (failure === "throw") await h.failLive("network error");
  else {
    h.context.fetch = async () => ({ ok: false, status: 500 });
    await h.context.attachEvents();
  }
  assert.equal(h.liveRows().length, 1, "the failed request visibly reports its cause");
  const row = h.liveRows()[0];
  assert.match(row.textContent, failure === "throw" ? /network error/ : /event stream 500/);
  const connection = await h.connectLive();
  try {
    assert.equal(h.liveRows().length, 0, "successful response headers clear the warning while the turn still runs");
    assert.equal(row.parentNode, null);
    assert.equal(h.context.eventFailures, 0, "a recovered connection starts a new failure budget");
  } finally { await connection.close(); }
});

test("repeated history failures update one DOM row and keep the automatic retry", async () => {
  const h = harness();
  await h.failHistory("network error");
  const row = h.historyRows()[0], retryEntry = [...h.timers].find(([, timer]) => timer.delay === 1000);
  assert.equal(h.historyRows().length, 1);
  assert.ok(retryEntry, "the existing history retry must remain scheduled");
  h.setHistoryFailure("Failed to fetch");
  h.timers.delete(retryEntry[0]);
  retryEntry[1].callback();
  await new Promise(setImmediate);
  assert.equal(h.historyRows().length, 1);
  assert.equal(h.historyRows()[0], row);
  assert.equal(row.textContent, "history failed: Failed to fetch");
  assert.ok([...h.timers.values()].some((timer) => timer.delay === 3000), "the retry backoff continues");
});

test("a successful history render removes its error row", async () => {
  const h = harness();
  await h.failHistory("Failed to fetch");
  const row = h.historyRows()[0];
  await h.recoverHistory();
  assert.equal(h.historyRows().length, 0);
  assert.equal(row.parentNode, null);
  assert.deepEqual(h.root.children.map((node) => node.textContent), ["prompt", "reply"]);
  assert.equal(h.context.historyLoadState("s").failure, null);
  assert.equal(h.timers.size, 0, "successful rendering cancels the history retry");
});

test("live recovery preserves history errors and real conversation nodes", async () => {
  const h = harness();
  await h.recoverHistory();
  const content = h.root.children.slice();
  const realError = h.context.addMsg("err", "live progress unavailable: a recorded conversation error", "asst err");
  await h.failHistory("Failed to fetch");
  const historyError = h.historyRows().find((row) => row !== realError);
  await h.failLive("network error");
  // The recorded error deliberately has the same prefix as the transient row.
  const warning = h.liveRows().find((row) => row !== realError);
  h.root.scrollTop = 45;
  const connection = await h.connectLive();
  try {
    assert.equal(h.liveRows().length, 1);
    assert.equal(h.liveRows()[0], realError, "a recorded error with identical wording is still history");
    assert.equal(warning.parentNode, null);
    assert.deepEqual(h.historyRows(), [historyError]);
    assert.equal(historyError.textContent, "history failed: Failed to fetch");
    assert.deepEqual(h.root.children.slice(0, content.length), content);
    assert.equal(h.root.children[content.length], realError);
    assert.equal(h.root.scrollTop, 45, "warning cleanup does not reset the reader's position");
  } finally { await connection.close(); }
});

test("ending recovery removes the warning without removing canonical rows", async () => {
  const h = harness();
  await h.recoverHistory();
  const content = h.root.children.slice();
  await h.failLive("Failed to fetch");
  const warning = h.liveRows()[0];
  h.context.abortEvents();
  assert.equal(h.liveRows().length, 0);
  assert.equal(warning.parentNode, null);
  assert.deepEqual(h.root.children, content);
});

for (const rendered of [false, true]) test(`history recovery ${rendered ? "preserves" : "replaces"} the live error (history already rendered=${rendered})`, async () => {
  const h = harness();
  if (rendered) await h.recoverHistory();
  const content = h.root.children.slice();
  await h.failLive("Failed to fetch");
  const liveError = h.liveRows()[0];
  await h.failHistory("network error");
  assert.equal(h.liveRows()[0], liveError, "a history failure cannot replace the live status");
  await h.recoverHistory();
  assert.equal(h.historyRows().length, 0);
  assert.equal(h.liveRows().length, rendered ? 1 : 0);
  if (rendered) {
    assert.equal(h.liveRows()[0], liveError, "history rendering must preserve the other error node");
  } else {
    assert.equal(liveError.parentNode, null, "the first canonical render replaces the local error row");
    assert.ok(!h.root.children.includes(liveError));
  }
  assert.equal(liveError.textContent, "live progress unavailable: Failed to fetch");
  assert.deepEqual(h.root.children.slice(0, 2).map((node) => node.textContent), ["prompt", "reply"]);
  if (rendered) assert.deepEqual(h.root.children.slice(0, content.length), content);
});
