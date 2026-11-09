// Run with: node --test src/bin/ai/serve/app_error_rows.test.cjs
// Exercise the embedded client and DOM ownership, using the existing VM harness pattern.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const script = fs.readFileSync(path.join(__dirname, "app.html"), "utf8")
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

function harness() {
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

  const root = element(), timers = new Map(), renderErrors = [];
  let timerId = 0, historyError = null;
  const messages = [
    { role: "user", content: "prompt" },
    { role: "assistant", content: "reply" },
  ];
  const context = vm.createContext({
    document: { createElement: element, visibilityState: "visible" },
    navigator: { onLine: true },
    console: { error: (...args) => renderErrors.push(args) },
    $: (id) => { assert.equal(id, "msgs"); return root; },
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
    textOf: (content) => content || "", isRuntimeSyntheticUser: () => false,
  });
  for (const [start, end] of [
    ["const esc =", "// --- markdown"],
    ["function addMsg(", "// --- server-side media"],
    ["function histKey(", "// A session the server reports"],
    ["function attachLiveBox(", "// A turn the user has left"],
    ["// BEGIN history retry", "// Historical user inputs"],
    ["function handleReplayEvent(", "async function pollConfirm("],
  ]) vm.runInContext(source(start, end), context);
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
    context, root, timers, renderErrors, rows,
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

test("two consecutive live-progress request failures update one DOM row", async () => {
  const h = harness();
  await h.failLive("network error");
  const row = h.liveRows()[0];
  assert.equal(h.liveRows().length, 1);
  await h.failLive("Failed to fetch");
  assert.equal(h.liveRows().length, 1);
  assert.equal(h.liveRows()[0], row, "reuse the original DOM node");
  assert.equal(row.textContent, "live progress unavailable: Failed to fetch");
});

test("a successful live-progress reconnect immediately removes its error row", async () => {
  const h = harness();
  await h.failLive("network error");
  const connection = await h.connectLive();
  try {
    assert.equal(h.liveRows().length, 0, "clear as soon as the successful stream is attached");
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
  const realError = h.context.addMsg("err", "history failed: a recorded conversation error", "asst err");
  await h.failHistory("Failed to fetch");
  const historyError = h.historyRows().find((row) => row !== realError);
  await h.failLive("network error");
  const connection = await h.connectLive();
  try {
    assert.equal(h.liveRows().length, 0);
    assert.deepEqual(h.historyRows(), [realError, historyError]);
    assert.equal(historyError.textContent, "history failed: Failed to fetch");
    assert.deepEqual(h.root.children.slice(0, content.length), content);
  } finally { await connection.close(); }
});

for (const rendered of [false, true]) test(`history recovery preserves the live error (history already rendered=${rendered})`, async () => {
  const h = harness();
  if (rendered) await h.recoverHistory();
  const content = h.root.children.slice();
  await h.failLive("Failed to fetch");
  const liveError = h.liveRows()[0];
  await h.failHistory("network error");
  assert.equal(h.liveRows()[0], liveError, "a history failure cannot replace the live status");
  await h.recoverHistory();
  assert.equal(h.historyRows().length, 0);
  assert.equal(h.liveRows().length, 1);
  assert.equal(h.liveRows()[0], liveError, "history rendering must preserve the other error node");
  assert.equal(liveError.textContent, "live progress unavailable: Failed to fetch");
  assert.deepEqual(h.root.children.slice(0, 2).map((node) => node.textContent), ["prompt", "reply"]);
  if (rendered) assert.deepEqual(h.root.children.slice(0, content.length), content);
});
