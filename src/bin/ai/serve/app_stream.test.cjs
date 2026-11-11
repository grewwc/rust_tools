// Run with: node --test src/bin/ai/serve/app_stream.test.cjs
// Execute the embedded stream state and dispatch, rather than a second renderer.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function rateNodes(root) {
  const found = [];
  const visit = (parent) => {
    for (const child of parent.children || []) {
      if (child.dataset?.turnRate && !child.hidden) found.push(child);
      visit(child);
    }
  };
  visit(root);
  return found;
}

function nodeList(nodes = []) {
  const list = {
    ...nodes, length: nodes.length,
    item(index) { return this[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach((node, index) => callback.call(thisArg, node, index, list)); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
  return list;
}

function structuredSubagents(tasks, text = "raw backend progress") {
  return { text, task_report: { schema: "ai.task-report", version: 1, operation: "subagent_progress", snapshot: true, tasks } };
}
function structuredTask(task_id, elapsed_ms, overrides = {}) {
  return { task_id, agent: task_id, model: "model", description: "task " + task_id, status: "running", active: true,
    elapsed_ms, progress: { phase: "working", latest_activity: "reading files", stale_for_ms: 0, sequence: 1 }, ...overrides };
}
function structuredIdentityRows(h) {
  return h.box.children.flatMap((card) => card.children || []).flatMap((panel) => panel.children || [])
    .filter((row) => row.className === "task-identity").map((row) => row.textContent);
}

// Most fixtures predate the typed rate frame. Translate only their fixture
// input at the test boundary so the renderer is exercised through the current
// structured protocol, never through the retired terminal-footer parser.
function legacyRateNumber(text) {
  const value = String(text).trim().replace(/^~/, "");
  const suffix = value.at(-1)?.toLowerCase();
  const multiplier = suffix === "k" ? 1_000 : suffix === "m" ? 1_000_000 : 1;
  const number = Number(multiplier === 1 ? value : value.slice(0, -1));
  return Number.isFinite(number) && number > 0 ? number * multiplier : null;
}
function legacyRateSamples(value) {
  const clean = String(value).replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "").trim();
  const prefix = "↳ speed";
  if (!clean.startsWith(prefix)) return [];
  const payload = clean.slice(prefix.length).trim().replace(/^·\s*/, "");
  return payload.split(" · ").flatMap((segment) => {
    const first = segment.indexOf(" ");
    const marker = segment.indexOf(" tok @ ");
    if (first < 1 || marker < first) return [];
    const phase = segment.slice(0, first);
    if (phase !== "reasoning" && phase !== "output" && phase !== "generation") return [];
    const tokensText = segment.slice(first + 1, marker);
    const tokens = legacyRateNumber(tokensText);
    if (!tokens) return [];
    const rate = legacyRateNumber(segment.slice(marker + " tok @ ".length).replace(" tok/s", ""));
    return [{ phase, tokens, elapsed_ms: rate ? Math.round(tokens * 1000 / rate) : null,
      approximate: tokensText.startsWith("~") }];
  });
}
function structuredFixtureEvents(kind, value) {
  if ((kind === "" || kind === "progress") && typeof value === "string") {
    const samples = legacyRateSamples(value);
    return samples.length ? samples.map((sample) => ["rate", sample]) : [[kind, value]];
  }
  if (kind === "thinking" && value && typeof value === "object" && typeof value.rate === "string") {
    const sample = legacyRateSamples("↳ speed · reasoning " + value.rate)[0];
    return [[kind, { ...value, rate: sample || null }]];
  }
  return [[kind, value]];
}
function structuredFixtureFrame(frame) {
  return String(frame).split("\n\n").flatMap((block) => {
    const lines = block.split("\n");
    const event = lines.find((line) => line.startsWith("event: "))?.slice(7) || "";
    const dataIndex = lines.findIndex((line) => line.startsWith("data: "));
    if (dataIndex < 0) return [block];
    const data = lines[dataIndex].slice(6);
    const rates = event ? [] : legacyRateSamples(data);
    if (rates.length) return rates.map((rate) => "event: rate\ndata: " + JSON.stringify(rate));
    if (event === "thinking") {
      try {
        const value = JSON.parse(data);
        if (typeof value.rate === "string") {
          const sample = legacyRateSamples("↳ speed · reasoning " + value.rate)[0];
          lines[dataIndex] = "data: " + JSON.stringify({ ...value, rate: sample || null });
        }
      } catch (_) {}
    }
    return [lines.join("\n")];
  }).join("\n\n");
}

function streamHarness({ activityOnly = false, phone = false, storage = new Map(), realThinking = false, phaseClock = false, clockStart = 1000 } = {}) {
  const timers = [];
  const box = {
    children: [], isConnected: true,
    appendChild(node) { if (!this.children.includes(node)) this.children.push(node); node.parentNode = this; return node; },
    insertBefore(node, before) {
      if (before && before.parentNode !== this) throw new Error("foreign insertion anchor");
      if (node.parentNode && node.parentNode !== this) node.remove();
      this.children = this.children.filter((child) => child !== node);
      const index = this.children.indexOf(before);
      this.children.splice(index < 0 ? this.children.length : index, 0, node);
      node.parentNode = this;
    },
    querySelectorAll() { return nodeList(this.children.filter((child) => child.summary)); },
  };
  function node() {
    const classes = new Set();
    let text = "", markup = "";
    const clearChildren = () => {
      for (const child of item.children) child.parentNode = null;
      item.children = [];
    };
    const item = { children: [], dataset: {}, hidden: false,
      get className() { return [...classes].join(" "); },
      set className(value) { classes.clear(); value.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name)); },
      get textContent() { return text + markup + item.children.map((child) => child.textContent).join(""); },
      set textContent(value) { clearChildren(); text = String(value); markup = ""; },
      get innerHTML() { return markup; },
      set innerHTML(value) { clearChildren(); markup = String(value); text = ""; },
      get lastElementChild() { return item.children.at(-1) || null; },
      get nextElementSibling() { return item.parentNode?.children[item.parentNode.children.indexOf(item) + 1] || null; },
      appendChild(child) { child.remove(); item.children.push(child); child.parentNode = item; return child; },
      insertBefore(child, before) {
        if (child === before) return child;
        if (before && before.parentNode !== item) throw new Error("foreign insertion anchor");
        child.remove();
        const index = before ? item.children.indexOf(before) : item.children.length;
        item.children.splice(index, 0, child); child.parentNode = item;
        return child;
      },
      querySelector() { return null; }, querySelectorAll() { return []; }, classList: {
      add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name),
    }, get isConnected() { return box.isConnected && box.children.includes(item); },
    remove() {
      if (item.parentNode) item.parentNode.children = item.parentNode.children.filter((child) => child !== item);
      item.parentNode = null;
    } };
    return item;
  }
  const clock = { now: clockStart }, intervals = new Map(), listeners = new Map(), observers = [];
  let intervalId = 0;
  const listen = (type, fn) => listeners.set(type, (listeners.get(type) || []).concat(fn));
  const document = { hidden: false, body: box, addEventListener: listen, createElement: () => node() };
  const context = vm.createContext({
    box, sid: "s", sessionId: "s", finished: false, turnError: false, document,
    turns: new Map(), liveEntry: {},
    Date: { now: () => clock.now }, isPhoneLayout: () => phone,
    localStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
    },
    LIVE_STALL_MS: Number(/const LIVE_STALL_MS = (\d+)/.exec(script)[1]),
    runEpoch: 0, eventFailures: 0, confirmPollSeq: 0, eventStreamEnded: false, eventStreamFailed: false,
    setTimeout: (fn, delay) => { timers.push(delay); return timers.length; }, clearTimeout() {},
    // Extracted slices may arm timers; the assertions drive them directly.
    setInterval(fn, delay) {
      if (!phaseClock) return 0;
      const id = ++intervalId; intervals.set(id, { fn, delay }); return id;
    },
    clearInterval(id) { intervals.delete(id); },
    shouldFollowOutput: () => false, followOutput() {}, closeProgressPhase() {},
    mutateTranscript(mutation) { mutation(); },
    renderMarkdown: (text) => text, decorateMath() {}, decorateAndLoad() {}, addAnswerCopy() {},
    showConfirm() {}, hideConfirm() {}, confirmShown: null, stopStaleFollow() {},
    addMsgMd() { const item = node(); item.className = "msg asst md"; box.appendChild(item); return item; },
    addFold(title, text) {
      const item = node(), summary = { textContent: title }, details = { open: false };
      const body = { textContent: text };
      item.className = "msg asst";
      item.summary = summary; item.body = body;
      item.querySelector = (selector) => selector === "summary" ? summary : selector === "details" ? details : null;
      box.appendChild(item);
      return { box: item, body, set(label, value) { summary.textContent = label; body.textContent = value; } };
    },
    window: { addEventListener: listen },
    MutationObserver: class { constructor(fn) { observers.push(fn); } observe() {} },
    ...(activityOnly ? {
      Date: { now: () => clock.now }, document,
      setInterval(fn, delay) { const id = ++intervalId; intervals.set(id, { fn, delay }); return id; },
      clearInterval(id) { intervals.delete(id); },
    } : {}),
  });
  context.turns.set("s", context.liveEntry);
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
  vm.runInContext(script.slice(script.indexOf("const RATE_STORAGE ="), script.indexOf("function addFold(")), context);
  // Phase-only fixtures leave unspecified reasoning rates blank. Rate tests
  // execute the real estimator, storage and output counter from the page.
  if (!realThinking) {
    context.estimateTextTokens = () => 1;
    context.measuredRateText = () => "";
  }
  vm.runInContext(script.slice(script.indexOf("function setAnswerRate("), script.indexOf("// Only completed answer bubbles")), context);
  const start = script.indexOf('  let bubble = addMsgMd("", box);');
  assert.ok(start > 0);
  vm.runInContext(script.slice(start, script.indexOf("  let turnStarted = false;", start)), context);
  // The watchdogs are `const` in the real script, so they are script-scoped
  // there. As globals here they are reachable from the block loop, which is
  // what arms them for every event, keep-alive comments included.
  const wdStart = script.indexOf("  const STREAM_SILENCE_MS");
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
    storage,
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
      for (const [eventKind, eventValue] of structuredFixtureEvents(kind, value)) {
        context.kind = eventKind;
        context.payload = typeof eventValue === "string" ? eventValue : JSON.stringify(eventValue ?? {});
        vm.runInContext("for (;;) { " + dispatch + "\nbreak; } flushPaint();", context);
      }
    },
    titles: () => box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
    bodies: () => box.children.filter((item) => item.body).map((item) => item.body.textContent),
    pending: () => vm.runInContext("phaseBubble.textContent", context),
    pendingVisible: () => vm.runInContext("box.children.includes(phaseBubble)", context),
    answerRate: () => vm.runInContext("(bubble || answerBubbles.at(-1)).dataset.tokenRate || ''", context),
    mainRate: () => rateNodes(box)[0]?.textContent || "",
    ageOutput: (ms) => { clock.now += ms; },
    step(ms = 1000) { clock.now += ms; for (const { fn } of [...intervals.values()]) fn(); },
    fail: () => vm.runInContext('fail("turn failed")', context),
    reply: () => vm.runInContext("(bubble || answerBubbles.at(-1)).innerHTML", context),
    // Throughput belongs inside a styled card, never at the turn's root.
    metrics: () => box.children.filter((item) => item.dataset.turnRate).map((item) => item.textContent),
    answerCount: () => vm.runInContext("answerBubbles.filter((b) => b.innerHTML).length", context),
  };
}

function replayHarness(options) {
  const live = streamHarness(options), { context, storage } = live;
  function node() {
    let text = "";
    const item = {
      children: [], dataset: {}, hidden: false, scrollTop: 0, className: "", classList: {
        add(name) { if (!this.contains(name)) item.className = (item.className + " " + name).trim(); },
        remove(name) { item.className = item.className.split(/\s+/).filter((part) => part !== name).join(" "); },
        contains(name) { return item.className.split(/\s+/).includes(name); },
      },
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        this.children.forEach((child) => { child.parentNode = null; });
        this.children = [];
      },
      get lastElementChild() { return this.children.at(-1) || null; },
      get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; },
      appendChild(child) { child.remove(); this.children.push(child); child.parentNode = this; return child; },
      insertBefore(child, before) {
        if (child === before) return child;
        if (before && before.parentNode !== this) throw new Error("foreign insertion anchor");
        child.remove();
        const index = this.children.indexOf(before);
        this.children.splice(index < 0 ? this.children.length : index, 0, child); child.parentNode = this;
        return child;
      },
      remove() {
        if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
        this.parentNode = null;
      },
      querySelector(selector) { return selector === "summary" ? this.summary : selector === "details" ? this.details : null; },
      querySelectorAll(selector) {
        const found = [];
        const visit = (parent) => parent.children.forEach((child) => {
          if (selector.startsWith(".") && String(child.className || "").split(/\s+/).includes(selector.slice(1))) found.push(child);
          visit(child);
        });
        visit(this);
        return found;
      },
    };
    return item;
  }
  const msgs = node();
  context.$ = () => msgs;
  context.document = { ...context.document, body: msgs, createElement: node };
  context.addFold = (title, text, parent) => {
    const box = node(); box.summary = { textContent: title }; box.details = { open: false };
    box.className = "msg asst";
    box.body = { textContent: text }; parent.appendChild(box);
    return { box, set(label, value) { box.summary.textContent = label; box.body.textContent = value; } };
  };
  context.NATIVE_PREFIX = "@@aserve-native ";
  const requests = [];
  context.handleNativeRequest = (...args) => requests.push(args);
  vm.runInContext(script.slice(script.indexOf("function recoveredProgressView("), script.indexOf("async function pollConfirm(")), context);
  const view = context.recoveredProgressView();
  return {
    context, msgs, view, requests, storage, ageOutput: live.ageOutput, step: live.step,
    emit(kind, payload) {
      for (const [eventKind, eventValue] of structuredFixtureEvents(kind, payload)) {
        context.handleReplayEvent("s", eventKind, typeof eventValue === "string" ? eventValue : JSON.stringify(eventValue ?? {}), view);
      }
    },
    status: () => view.box.children.at(-1).textContent,
    reply: () => view.box.children.filter((item) => !item.summary && item.className === "msg asst" && !item.hidden).at(-1)?.textContent || "",
    mainRate: () => rateNodes(view.box)[0]?.textContent || "",
    titles: () => view.box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
  };
}

function attachmentHarness(headersOnly = false, options) {
  const h = replayHarness(options), c = h.context, timers = new Map(), requests = [];
  h.view.remove();
  let timerId = 0, polls = 0, loads = 0;
  Object.assign(c, {
    eventActive: false, eventAbort: null, eventSession: "", recoveredView: null,
    eventStreamEnded: false, eventStreamFailed: false, turns: new Map(), store: { base: "https://server" },
    staleSession: "", staleDone: false, staleTimer: null, staleBox: null, staleTries: 0,
    turnRunning: true, turnRunningAt: Date.now(), RUNNING_SAMPLE_TTL_MS: 60000,
    AbortController, TextDecoder, headers: () => ({}), syncStopButton() {},
    ensureConfirmPoll() { polls++; }, stopConfirmPoll() {},
    addMsg(role, text, cls, parent) {
      const row = c.document.createElement("div");
      row.className = "msg " + cls; row.dataset.role = role; row.textContent = text;
      return (parent || h.msgs).appendChild(row);
    },
    loadHistory() { loads++; },
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms, at: c.Date.now() + ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(url, { signal }) {
      const request = { url, signal }; requests.push(request);
      if (headersOnly) return new Promise((resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      });
      return { ok: true, body: { getReader: () => ({ read: () => new Promise((resolve, reject) => {
        request.frame = (frame) => resolve({ value: new TextEncoder().encode(structuredFixtureFrame(frame)), done: false });
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      }) }) } };
    },
  });
  vm.runInContext(script.slice(script.indexOf("function parseBlock("), script.indexOf("async function send()")), c);
  vm.runInContext(script.slice(script.indexOf("function stopStaleFollow()"), script.indexOf("function startStaleFollow()")), c);
  const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  return {
    c, requests, timers, settle, storage: h.storage, polls: () => polls, loads: () => loads,
    advance(ms) {
      h.ageOutput(ms);
      for (const [id, timer] of [...timers]) if (timer.at <= c.Date.now()) { timers.delete(id); timer.fn(); }
    },
    fire(ms) {
      const timer = [...timers].find(([, item]) => item.ms === ms);
      assert.ok(timer, `missing ${ms}ms timer`);
      timers.delete(timer[0]); timer[1].fn();
    },
    async frame(text) { requests.at(-1).frame(text); await settle(); },
  };
}

function runningComposer(c) {
  const lookup = c.$;
  const elements = Object.fromEntries(["send", "sidenote", "composer", "turnstatus"].map((id) => [id, {
    textContent: "", hidden: true, classList: { toggle() {} }, setAttribute() {},
  }]));
  Object.assign(c, {
    $: (id) => elements[id] || lookup(id), sideNoteBusy: false,
    holdLayout: (_, apply) => apply(),
  });
  vm.runInContext(script.slice(script.indexOf("function isTurnRunning()"), script.indexOf("function note(")), c);
  return elements;
}

function historyRateHarness(storage, content, reasoning = "", phone = false) {
  const h = replayHarness({ storage, phone });
  h.view.remove();
  h.context.textOf = (text) => text;
  h.context.addMsgMd = (text, parent) => {
    const bubble = h.context.document.createElement("div");
    bubble.className = "msg asst md";
    bubble.textContent = text;
    return parent.appendChild(bubble);
  };
  const start = script.indexOf("function renderHistoryRow(");
  vm.runInContext(script.slice(start, script.indexOf("// A session the server reports", start)), h.context);
  h.context.renderHistoryRow({
    message: { role: "assistant", content, reasoning_content: reasoning },
    text: content.trim(), reasoning, identity: "answer", key: "canonical-answer", historyIndex: 1,
  }, h.msgs);
  const group = h.msgs.lastElementChild;
  return {
    ...h, group,
    restoredRate: () => rateNodes(group)[0]?.textContent || "",
    restoredThinking: () => group.children.find((item) => item.summary)?.summary.textContent || "",
  };
}

function coldHistoryHarness(savedRates, { phone = true } = {}) {
  function element(tag = "div") {
    let text = "";
    const item = {
      tagName: tag.toUpperCase(), className: "", dataset: {}, children: [], hidden: false,
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        for (const child of this.children) child.parentNode = null;
        this.children = []; text = String(value);
      },
      set innerHTML(value) { this.textContent = String(value).replace(/<[^>]*>/g, ""); },
      get lastElementChild() { return this.children.at(-1) || null; },
      get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; },
      appendChild(child) { child.remove(); this.children.push(child); child.parentNode = this; return child; },
      insertBefore(child, before) {
        if (child === before) return child;
        assert.ok(!before || before.parentNode === this, "the insertion anchor must belong to its card");
        child.remove();
        const index = before ? this.children.indexOf(before) : this.children.length;
        this.children.splice(index, 0, child); child.parentNode = this;
        return child;
      },
      remove() {
        if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
        this.parentNode = null;
      },
      addEventListener() {},
      matches(selector) {
        return selector.startsWith(".")
          ? selector.slice(1).split(".").every((name) => this.classList.contains(name))
          : this.tagName === selector.toUpperCase();
      },
      querySelectorAll(selector) {
        return this.children.flatMap((child) => [
          ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
      },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
    };
    item.classList = { contains: (name) => item.className.split(/\s+/).includes(name) };
    return item;
  }
  const root = element(), storage = new Map([["serve.rates", savedRates]]);
  // A new document receives only serialized storage. It never constructs a
  // live/recovered view or measures the arrival time of history bodies.
  const context = vm.createContext({
    sessionId: "s", transcriptRoot: root, isPhoneLayout: () => phone,
    document: { createElement: element }, $: (id) => id === "msgs" ? root : null,
    localStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem() { assert.fail("cold history must not write a live rate sample"); },
    },
    Date: { now() { assert.fail("cold history must not measure replay timing"); } },
    setInterval() { assert.fail("cold history must not create a running clock"); },
    setTimeout() { assert.fail("cold history must not schedule live paint"); },
    esc: (text) => String(text), textOf: (text) => String(text), isRuntimeSyntheticUser: () => false,
    renderMarkdown: (text) => text, decorateAndLoad() {}, addAnswerCopy() {}, attachUserRewind: (bubble) => bubble,
  });
  for (const [start, end] of [
    ["const RATE_STORAGE =", "function addFold("],
    ["function addMsg(", "// --- server-side media"],
    ["function addFold(", "function createThinkingFold("],
    ["function addUserBubble(", "function findUserHistoryIndex("],
    ["function historyRowsOf(", "// A session the server reports as running"],
  ]) {
    const from = script.indexOf(start), to = script.indexOf(end, from);
    assert.ok(from >= 0 && to > from, "missing cold history source: " + start);
    vm.runInContext(script.slice(from, to), context);
  }
  return {
    context, root, storage,
    mount(historyPayload, showAll = true) {
      context.historyPayload = historyPayload;
      context.historyShowAll = showAll;
      vm.runInContext("const coldMessages = JSON.parse(historyPayload);" +
        "const coldRows = historyRowsOf(coldMessages, coldMessages.length, historyShowAll);" +
        "for (const row of coldRows.rows) renderHistoryRow(row, transcriptRoot);", context);
    },
  };
}

function retainedHistoryHarness(messages) {
  const h = attachmentHarness(false, { realThinking: true }), c = h.c;
  const root = c.$("msgs"), historyCalls = [];
  Object.assign(c, {
    $: (id) => id === "msgs" ? root : null,
    navigator: { onLine: true },
    historySession: "", historyKeys: [], historyLoadSeq: 0, lastMsgCount: null,
    canonicalHistoryTotal: 0, pendingUserInputs: new Map(),
    staleSession: "", staleDone: false, staleTries: 0,
    isRuntimeSyntheticUser: () => false, isTurnRunning: () => false,
    textOf: (text) => text, FOLD_ROLE_LABELS: { tool: "tool" },
    histKey: (rows) => JSON.stringify(rows), histEndsTurn: () => true, historyEndsUnanswered: () => false,
    revokePreviews() {}, renderSessionBar() {}, updateQuestionButton() {}, startStaleFollow() {},
    addUserBubble(text, index, parent) {
      const bubble = c.document.createElement("div");
      bubble.className = "msg user"; bubble.textContent = text;
      return parent.appendChild(bubble);
    },
    addMsgMd(text, parent) {
      const bubble = c.document.createElement("div");
      bubble.className = "msg asst md"; bubble.textContent = text;
      return parent.appendChild(bubble);
    },
    addMsg(role, text) { assert.fail(text); },
    async api(url) {
      assert.match(url, /^\/sessions\/s\/history\?limit=/);
      historyCalls.push(url);
      return { headers: { get: () => String(messages.length) }, json: async () => messages };
    },
    console: { error(...args) { assert.fail(args.map(String).join(" ")); } },
  });
  c.document.visibilityState = "visible";
  const start = script.indexOf("function historyRowsOf(");
  vm.runInContext(script.slice(start, script.indexOf("// Scrolling to the top of the transcript", start)), c);
  return {
    ...h, root, historyCalls,
    visibleRates() {
      const found = [];
      const visit = (parent) => parent.children.forEach((child) => {
        if (child.dataset.turnRate && !child.hidden) found.push(child);
        visit(child);
      });
      visit(root);
      return found;
    },
  };
}

for (const reasoning of ["", "saved thought"]) {
  test("retained canonical history gains replay " + (reasoning ? "thinking and output rates" : "output rate") + " without remounting", async () => {
    const h = retainedHistoryHarness([
      { role: "user", content: "question" },
      { role: "tool", content: "tool result" },
      { role: "assistant", content: "saved answer", reasoning_content: reasoning },
    ]);
    await h.c.loadHistory();
    const original = [...h.root.children], canonical = original.at(-1);
  assert.equal(canonical.dataset.taskReset, "true", "a canonical final answer must terminate earlier task state");
    const answer = canonical.children.find((child) => child.className === "msg asst md");
    const thinking = canonical.children.find((child) => child.summary);
    if (thinking) thinking.details.open = true;
    assert.equal(h.visibleRates().length, 0);
    const rate = reasoning ? "reasoning 20 tok @ 10 tok/s · output 40 tok @ 20 tok/s" : "output 40 tok @ 20 tok/s";
    const pending = h.c.attachEvents();
    await h.settle();
    if (reasoning) {
      await h.frame('event: thinking\ndata: {"text":"saved thought","rate":"~10 tok @ 5 tok/s"}\n\n');
      await h.frame("event: thinking_done\ndata: \n\n");
    }
    await h.frame('event: delta\ndata: {"delta":"saved answer"}\n\n');
    await h.frame("data: ↳ speed · " + rate + "\n\n");
    await h.frame("event: done\ndata: \n\n");
    await pending;
    assert.equal(h.c.eventFailures, 0);
    assert.equal(h.c.eventStreamEnded, true);
    assert.equal(h.historyCalls.length, 2, "completion must run the real loader again");
    assert.deepEqual(h.root.children, original, "answers and tool rows must retain their nodes and order");
    assert.equal(canonical.children.find((child) => child.className === "msg asst md"), answer);
    assert.equal(h.c.storedRate("turn", "s", "saved answer"), rate);
    const visible = h.visibleRates();
    assert.equal(visible.length, 1, "retained canonical history did not display the newly cached rate");
    assert.equal(visible[0].textContent, "↳ speed · " + rate);
    assert.equal(visible[0].parentNode, answer);
    if (thinking) {
      assert.equal(thinking.summary.textContent, "thinking · ~20 tok @ 10 tok/s");
      assert.equal(thinking.details.open, true, "rate refresh collapsed a retained thinking fold");
    }
    await h.c.loadHistory();
    assert.deepEqual(h.root.children, original);
    assert.deepEqual(h.visibleRates(), visible, "another retained load duplicated or replaced the rate row");
    const updated = rate.replace("40 tok @ 20 tok/s", "60 tok @ 30 tok/s");
    h.c.rememberRate("turn", "s", "saved answer", updated);
    if (reasoning) h.c.rememberRate("thinking", "s", reasoning, "~20 tok @ 10 tok/s");
    await h.c.loadHistory();
    assert.deepEqual(h.root.children, original);
    assert.deepEqual(h.visibleRates(), visible);
    assert.equal(visible[0].textContent, "↳ speed · " + updated);
    if (thinking) assert.equal(thinking.summary.textContent, "thinking · ~20 tok @ 10 tok/s");
    h.storage.delete("serve.rates");
    await h.c.loadHistory();
    assert.deepEqual(h.root.children, original);
    assert.equal(h.visibleRates().length, 0, "retained history displayed throughput after the cached metric was removed");
    if (thinking) assert.equal(thinking.summary.textContent, "thinking · ~4 tok");
  });
}

test("retained canonical history does not create rates without cached metrics", async () => {
  const h = retainedHistoryHarness([{ role: "assistant", content: "old answer", reasoning_content: "old thought" }]);
  await h.c.loadHistory();
  const canonical = h.root.lastElementChild;
  await h.c.loadHistory();
  assert.equal(h.root.lastElementChild, canonical);
  assert.equal(h.visibleRates().length, 0);
  assert.equal(canonical.children.find((child) => child.summary).summary.textContent, "thinking · ~3 tok");
});

test("replayed phases paint visible status and separate thinking across tools", () => {
  const h = replayHarness();
  assert.equal(h.status(), "Connecting… · 0:00");
  h.emit("thinking", { text: "first phase" });
  assert.equal(h.status(), "Thinking… · 0:00");
  h.emit("thinking_done");
  h.emit("tool", { id: "alarm", name: "phone_action", phase: "started", arguments: "{}" });
  assert.equal(h.status(), "Running phone_action… · 0:00");
  h.emit("input_request", { id: 7, token: "t", prompt: '@@aserve-native {"action":"alarm","hour":9,"minute":30}' });
  assert.equal(h.requests.length, 1);
  assert.equal(h.status(), "正在手机上执行… · 0:00");
  h.emit("input_done", { id: 7 });
  assert.equal(h.status(), "Running phone_action… · 0:00");
  h.emit("tool", { id: "alarm", name: "phone_action", phase: "finished", ok: true });
  assert.equal(h.status(), "Processing… · 0:00");
  h.emit("thinking", { text: "second phase" });
  assert.equal(h.titles().filter((title) => title === "thinking").length, 2);
  h.emit("delta", { delta: "answer" });
  assert.equal(h.reply(), "answer");
  assert.equal(h.status(), "Responding… · 0:00");
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
          const events = url.includes("/events?after=0&client=") ? 'id: 41\nevent: thinking\ndata: {"text":"restored"}\n\n' : "";
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
  assert.deepEqual(statuses, ["Thinking… · 0:00", "Thinking… · 0:00"]);
  assert.equal(urls.length, 2);
  assert.ok(urls.every((url) => url.includes("/events?after=0&client=")), "each connection replays from the start under this client");
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
  assert.ok(h.requests.every((r) => r.url.includes("/events?after=0&client=")), "only GET replay is retried, never the user POST");
  assert.equal(
    new URL(h.requests[0].url).searchParams.get("client"),
    new URL(h.requests[1].url).searchParams.get("client"),
    "a reconnect reattaches as the same client instead of a new anonymous one",
  );
  h.c.abortEvents();
  await h.settle();
  assert.equal(h.timers.size, 0);
});

test("a reattached page names the same client so the stream is not refused", async () => {
  const h = attachmentHarness();
  const pending = h.c.attachEvents();
  await h.settle();
  const first = new URL(h.requests.at(-1).url).searchParams.get("client");
  assert.ok(first && first.length > 3, "the attach names its client");
  h.c.abortEvents();
  await h.settle();
  h.c.attachEvents();
  await h.settle();
  assert.equal(
    new URL(h.requests.at(-1).url).searchParams.get("client"),
    first,
    "the second attach keeps the identity the server already granted",
  );
  h.c.abortEvents();
  await pending;
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

for (const phone of [false, true]) test(`event attach conflicts reuse one warning and clear it on recovery (phone=${phone})`, async () => {
  const h = attachmentHarness(false, { phone }), c = h.c, fetch = c.fetch;
  const root = c.$("msgs"), warnings = () => root.querySelectorAll(".err");
  let attempts = 0;
  c.fetch = (url, options) => ++attempts <= 2
    ? Promise.resolve({ ok: false, status: 409 }) : fetch(url, options);
  await c.attachEvents();
  assert.equal(warnings().length, 1);
  const row = warnings()[0];
  assert.equal(row.textContent, "live progress unavailable: event stream 409");
  h.fire(2000);
  await h.settle();
  assert.equal(attempts, 2);
  assert.equal(warnings().length, 1, "a retry must update the existing warning");
  assert.equal(warnings()[0], row);
  assert.equal(row.textContent, "live progress unavailable: event stream 409 (2 failures)");
  h.fire(2000);
  await h.settle();
  assert.equal(attempts, 3);
  assert.equal(warnings().length, 0, "successful response headers clear the warning before the stream ends");
  assert.equal(row.parentNode, null);
  await h.frame('event: done\ndata: {}\n\n');
  assert.equal(c.eventStreamEnded, true);
  assert.equal(h.timers.size, 0);
});

test("event attach warnings distinguish sessions, epochs and error kinds", async () => {
  const h = attachmentHarness(), c = h.c, root = c.$("msgs");
  c.fetch = async () => ({ ok: false, status: 409 });
  await c.attachEvents();
  c.runEpoch++;
  await c.attachEvents();
  c.sessionId = "other";
  await c.attachEvents();
  c.fetch = async () => { throw new Error("event stream 409: lease expired"); };
  await c.attachEvents();
  const rows = root.querySelectorAll(".live-progress-warning");
  assert.equal(rows.length, 4);
  assert.equal(new Set(rows.map((row) => row.dataset.eventWarningKey)).size, 4);
  assert.ok(rows.every((row) => row.dataset.eventWarningFailures === "1"));
  assert.equal(rows.at(-1).textContent, "live progress unavailable: event stream 409: lease expired");
  c.fetch = async () => ({ ok: true, body: { getReader: () => ({ read: async () => ({ done: true }) }) } });
  await c.attachEvents();
  assert.deepEqual(root.querySelectorAll(".live-progress-warning"), rows.slice(0, 2),
    "recovery clears only warnings belonging to the attached session and epoch");
});

test("event attach errors other than conflicts keep separate warning rows", async () => {
  const h = attachmentHarness(), c = h.c, root = c.$("msgs");
  let status = 409;
  c.fetch = async () => ({ ok: false, status });
  await c.attachEvents();
  status = 500;
  await c.attachEvents();
  await c.attachEvents();
  const rows = root.querySelectorAll(".err");
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((row) => row.textContent), [
    "live progress unavailable: event stream 409",
    "live progress unavailable: event stream 500",
    "live progress unavailable: event stream 500",
  ]);
  assert.equal(root.querySelectorAll(".live-progress-warning").length, 1);
});

test("event attach warnings stay silent for missing, aborted and superseded streams", async () => {
  for (const reason of ["404", "aborted", "session", "epoch", "controller"]) {
    const h = attachmentHarness(), c = h.c, root = c.$("msgs");
    let respond;
    c.fetch = () => new Promise((resolve) => { respond = resolve; });
    const pending = c.attachEvents();
    await h.settle();
    if (reason === "aborted") c.abortEvents();
    else if (reason === "session") c.sessionId = "other";
    else if (reason === "epoch") c.runEpoch++;
    else if (reason === "controller") c.eventAbort = new AbortController();
    respond({ ok: false, status: reason === "404" ? 404 : 409 });
    await pending;
    assert.equal(root.querySelectorAll(".err").length, 0, reason + " must remain silent");
  }
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
  const rates = rateNodes(h.box);
  assert.equal(rates.length, 1);
  assert.equal(rates[0].hidden, false);
  assert.equal(rates[0].parentNode, h.box.children.at(-2));
  assert.equal(rates[0].textContent, "↳ speed · output ~4 tok");
  const shape = h.box.children.map((item) => (item.summary ? "fold" : item.dataset.turnRate ? "rate" : item.innerHTML || "status"));
  assert.deepEqual(shape, ["first segment.", "fold", "second segment.", "status"]);
});

test("replayed phases remain visible after an early assistant answer", () => {
  const h = replayHarness();
  h.emit("delta", { delta: "I will check that." });
  h.emit("tool", { id: "read", name: "read_file", phase: "started" });
  assert.equal(h.status(), "Running read_file… · 0:00");
  h.emit("tool", { id: "read", name: "read_file", phase: "finished", ok: true });
  assert.equal(h.status(), "Processing… · 0:00");
  h.emit("thinking", { text: "next step" });
  assert.equal(h.status(), "Thinking… · 0:00");
});

test("turn throughput never duplicates the rate inside a thinking-only card", () => {
  const h = streamHarness({ realThinking: true });
  h.connect();
  h.emit("thinking", { text: "reasoning" });
  h.emit("", "↳ speed · reasoning 22 tok · output 82 tok @ 151 tok/s");
  assert.equal(rateNodes(h.box).length, 0,
    "turn throughput must not attach below a thinking fold");
  h.emit("delta", { delta: "answer" });
  h.emit("", "↳ speed · reasoning 22 tok · output 82 tok @ 151 tok/s");
  const line = rateNodes(h.box)[0];
  assert.ok(line);
  assert.ok(line.parentNode.classList.contains("asst"));
  assert.ok(line.parentNode.classList.contains("md"));
});

test("pending HTTP request paints a connection indicator before any SSE event", () => {
  const h = streamHarness();
  h.connect();
  assert.equal(h.pending(), "◌ Connecting… · 0:00");
  assert.deepEqual(h.titles(), []);
  h.emit("thinking", { text: "reasoning" });
  assert.match(h.pending(), /Thinking/);
});

test("status strip regression: restored sessions keep stop and busy state without a hint row", () => {
  const elements = Object.fromEntries(["send", "sidenote", "composer"].map((id) => [id, {
    textContent: "", hidden: true, attrs: {}, classList: { toggle() {} },
    setAttribute(name, value) { this.attrs[name] = value; },
  }]));
  const context = vm.createContext({
    $: (id) => { assert.notEqual(id, "turnstatus", "the removed strip is still accessed"); return elements[id]; },
    sessionId: "a", turns: new Map(),
    eventActive: false, eventSession: "", eventStreamEnded: false, eventStreamFailed: false,
    staleSession: "", staleDone: false, sideNoteBusy: false,
    turnRunning: true, turnRunningAt: 100000, RUNNING_SAMPLE_TTL_MS: 60000,
    Date: { now: () => 100001 },
  });
  vm.runInContext(script.slice(script.indexOf("let promptComposing = false;"), script.indexOf("function composerBox()")), context);
  vm.runInContext(script.slice(script.indexOf("function isTurnRunning()"), script.indexOf("function note(")), context);
  context.syncStopButton();
  assert.equal(elements.send.title, "stop this turn");
  assert.equal(elements.composer.attrs["aria-busy"], "true");
  // A reattached event stream is a live source even though it is not in `turns`.
  context.eventActive = true; context.eventSession = "a";
  context.syncStopButton();
  assert.equal(elements.composer.attrs["aria-busy"], "false");
  context.eventActive = false; context.eventSession = "";
  assert.equal(elements.send.title, "stop this turn");
  context.turns.set("a", {});
  context.syncStopButton();
  assert.equal(elements.composer.attrs["aria-busy"], "false");
  context.turns.clear(); context.turnRunningAt = 100000; context.staleSession = "a";
  context.syncStopButton();
  assert.equal(elements.composer.attrs["aria-busy"], "true");
  context.turnRunningAt = 0;
  context.syncStopButton();
  assert.equal(context.isTurnRunning(), true, "an unresolved stale follow remains active after its running sample expires");
  assert.equal(elements.composer.attrs["aria-busy"], "true");
  assert.equal(elements.send.title, "stop this turn");
  context.turnRunning = false; context.staleSession = "";
  context.syncStopButton();
  assert.equal(elements.composer.attrs["aria-busy"], "false");
  assert.equal(elements.send.title, "send");
  context.sessionId = "b"; context.staleSession = "a";
  context.syncStopButton();
  assert.equal(elements.composer.attrs["aria-busy"], "false");
});

test("status strip regression: phone transcript directly meets model controls without dot, hint, or divider row", () => {
  const markup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, "");
  const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  assert.doesNotMatch(markup, /\bid="turnstatus"/);
  assert.doesNotMatch(css, /#turnstatus\b|\bturnpulse\b/);
  assert.doesNotMatch(script, /\$\("turnstatus"\)|server is still running the previous turn|Live connection lost ·/);
  const transcriptEnd = markup.indexOf('<section id="questionpanel"');
  const settingsStart = markup.indexOf('<details id="ovrwrap"');
  assert.ok(transcriptEnd > 0 && settingsStart > transcriptEnd);
  assert.match(markup.slice(transcriptEnd, settingsStart), /<\/section>\s*<\/div>\s*$/,
    "a separate empty row remains between the transcript and model controls");
  assert.match(markup, /id="ovrsummary"/);
  assert.match(markup, /id="teamtrigger"/);
  assert.match(markup, /id="composer"/);
  const phoneStart = css.indexOf("@media (max-width:700px)");
  const phone = css.slice(phoneStart, css.indexOf("@media (prefers-reduced-motion:reduce)", phoneStart));
  assert.match(phone, /#ovrwrap\s*\{[^}]*display:block/);
  assert.match(phone, /#ovrwrap > summary\s*\{[^}]*min-height:48px/);
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
  // Plain child stdout (diagnostics included) is not transcript content; tool
  // work is rendered from structured `tool` events instead.
  assert.equal(h.titles().filter((title) => title.startsWith("activity log")).length, 0);
  assert.ok(h.bodies().includes("The user asks"));
  assert.ok(h.bodies().some((text) => text.includes("search results")));
  assert.ok(h.titles().findIndex((title) => title.includes("web_search")) < h.titles().lastIndexOf("thinking…"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("done");
  assert.equal(h.reply(), "Final answer");
  assert.equal(h.pendingVisible(), false, "a completed turn must not retain a running indicator");
});

test("desktop status keeps only phase and elapsed while main throughput stays visible", () => {
  const h = streamHarness({ phaseClock: true });
  h.emit("progress", "request diagnostic");
  h.emit("thinking_done");
  h.emit("progress", "↳ speed · reasoning 18 tok @ — tok/s · output 122 tok @ — tok/s");
  assert.deepEqual(h.metrics(), []);
  assert.equal(h.pending(), "◌ Processing… · 0:00");
  assert.doesNotMatch(h.pending(), /tok/);
  assert.equal(h.mainRate(), "↳ speed · reasoning 18 tok · output 122 tok");
  h.emit("thinking", { text: "brief thought", rate: "~10 tok @ — tok/s" });
  assert.ok(h.titles().includes("thinking… · ~10 tok"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("progress", "  \x1b[2m↳ speed · reasoning 800 tok @ 400 tok/s · output 400 tok @ 400 tok/s\x1b[0m");
  assert.deepEqual(h.metrics(), []);
  assert.equal(h.pending(), "◌ Responding… · 0:00");
  assert.doesNotMatch(h.pending(), /tok/);
  assert.match(h.mainRate(), /reasoning 800 tok @ 400 tok\/s · output 400 tok @ 400 tok\/s/);
  h.step(88000);
  assert.equal(h.pending(), "◌ Responding… · 1:28");
  assert.match(h.mainRate(), /reasoning 800 tok @ 400 tok\/s · output 400 tok @ 400 tok\/s/);
  assert.equal(h.titles().filter((title) => title.startsWith("activity log")).length, 0);
  assert.doesNotMatch(h.bodies().join("\n"), /400 tok\/s/);
  h.emit("done");
  assert.equal(h.metrics().length, 0, "throughput never creates its own message");
  assert.equal(h.answerRate(), "reasoning 800 tok @ 400 tok/s · output 400 tok @ 400 tok/s");
  assert.ok(h.titles().includes("thinking · ~800 tok @ 400 tok/s"));
  assert.doesNotMatch(h.mainRate() + h.titles().join("\n"), /— tok\/s/);
});

test("typed rate events render producer metrics without parsing terminal output", () => {
  const h = streamHarness({ phone: true, realThinking: true });
  h.emit("thinking_start");
  h.emit("thinking", { text: "brief thought", rate: {
    phase: "reasoning", tokens: 4, elapsed_ms: 500, approximate: true,
  } });
  assert.ok(h.titles().includes("thinking… · ~4 tok @ 8 tok/s"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("rate", { phase: "reasoning", tokens: 8, elapsed_ms: 1_000, approximate: false });
  h.emit("rate", { phase: "output", tokens: 40, elapsed_ms: 2_000, approximate: false });
  assert.equal(h.mainRate(), "↳ speed · reasoning 8 tok @ 8 tok/s · output 40 tok @ 20 tok/s");
  // Plain terminal stdout is not a rate event and cannot change this display.
  h.context.kind = "";
  h.context.payload = "↳ speed · output 999 tok @ 999 tok/s";
  assert.equal(h.mainRate(), "↳ speed · reasoning 8 tok @ 8 tok/s · output 40 tok @ 20 tok/s");
});

test("ordinary streamed output shows a live rate and keeps a final estimate", () => {
  const h = streamHarness();
  h.emit("delta", { delta: "abcdefgh" });
  h.ageOutput(1000);
  h.emit("delta", { delta: "ijkl" });
  assert.equal(h.pending(), "◌ Responding… · 0:01");
  assert.doesNotMatch(h.pending(), /tok/);
  assert.match(h.mainRate(), /output ~3 tok @ [\d.]+ tok\/s/);
  h.emit("done");
  assert.equal(h.metrics().length, 0);
  assert.match(h.answerRate(), /output ~3 tok @ [\d.]+ tok\/s/);
});

for (const phone of [true, false]) {
  test((phone ? "phone" : "desktop") + " main rate is visible during output and restores after a page refresh", () => {
    const h = streamHarness({ phone });
    h.emit("delta", { delta: "abcdefgh" });
    h.ageOutput(1000);
    h.emit("delta", { delta: "ijkl" });
    if (phone) {
      assert.doesNotMatch(h.pending(), /tok/, "the phone status bar must stay compact");
      assert.match(h.mainRate(), /output ~3 tok @ 3.00 tok\/s/);
      assert.match(rateNodes(h.box)[0].parentNode.className, /\bmsg asst\b/);
    } else {
      assert.match(h.mainRate(), /output ~3 tok @ 3.00 tok\/s/);
      assert.equal(rateNodes(h.box)[0].parentNode, h.box.children.at(-2));
      assert.doesNotMatch(h.pending(), /tok/);
    }
    h.emit("done");
    const line = rateNodes(h.box)[0];
    assert.equal(line.hidden, false);
    assert.match(line.parentNode.className, /\bmsg asst\b/);
    const refreshed = historyRateHarness(h.storage, "abcdefghijkl", "", phone);
    assert.equal(refreshed.restoredRate(), h.mainRate());
    assert.equal(refreshed.msgs.children.length, 1, "the rate must not add a canonical reconciliation row");
    assert.equal(refreshed.group.dataset.canonicalKey, "canonical-answer");
  });
}

test("completion after a tool round caches the sealed answer instead of empty raw text", () => {
  const h = streamHarness();
  h.emit("delta", { delta: "answer before a tool" });
  h.ageOutput(1000);
  h.emit("tool", { phase: "start", name: "read_file", call_id: "r" });
  h.emit("", "↳ speed · output 20 tok @ 10 tok/s");
  h.emit("done");
  assert.equal(h.context.storedRate("turn", "s", "answer before a tool"), "output 20 tok @ 10 tok/s");
  assert.equal(historyRateHarness(h.storage, "answer before a tool").restoredRate(), "↳ speed · output 20 tok @ 10 tok/s");
});

test("desktop throughput follows new answer and tool cards at the end of the live turn", () => {
  const h = streamHarness();
  h.emit("delta", { delta: "first answer" });
  h.emit("", "↳ speed · output 40 tok @ 20 tok/s");
  h.emit("tool", { phase: "started", name: "read_file", id: "r" });
  h.emit("delta", { delta: "next answer" });
  h.emit("", "↳ speed · output 60 tok @ 30 tok/s");
  const line = rateNodes(h.box)[0];
  assert.equal(line.dataset.turnRate, "1", "the main rate was left above newer transcript nodes");
  assert.equal(line.textContent, "↳ speed · output 60 tok @ 30 tok/s");
  assert.equal(line.parentNode, h.box.children.at(-2));
  h.emit("done");
  assert.equal(line.parentNode, h.box.children.at(-1));
});

test("a display replacement does not attach a rate to a different canonical body", () => {
  const notice = "[audit report · machine-checked JSON hidden from the terminal; full text stays in session history]";
  const canonical = 'answer<<<IMAGE_DIGEST>>>image metadata<<<END_IMAGE_DIGEST>>>\n<audit_report>{"findings":[]}</audit_report>\n\n[Runtime warning]\ninternal warning';
  const displayed = "answer\n" + notice;
  const h = streamHarness();
  h.emit("delta", { delta: canonical });
  h.ageOutput(1000);
  h.emit("replace", { text: displayed });
  h.emit("", "↳ speed · output 40 tok @ 20 tok/s");
  h.emit("done");
  const refreshed = historyRateHarness(h.storage, canonical);
  assert.equal(refreshed.restoredRate(), "");
  assert.equal(refreshed.group.children.find((item) => item.className === "msg asst md").textContent.slice(0, canonical.length), canonical,
    "rate normalization must not rewrite canonical history rendering");
});

test("uncached history stays empty while a replacement starts a local timing window", () => {
  assert.equal(historyRateHarness(new Map(), "old answer").restoredRate(), "");
  const h = streamHarness();
  h.emit("replace", { text: "buffered answer" });
  h.ageOutput(1000);
  h.emit("done");
  assert.match(h.mainRate(), /^↳ speed · output ~4 tok @ [\d.]+ tok\/s$/);
  assert.match(h.context.storedRate("turn", "s", "buffered answer"), /^output ~4 tok @ [\d.]+ tok\/s$/);
});

test("typed replay rate events are dispatched while comment heartbeats stay separate", async () => {
  const h = attachmentHarness();
  const seen = [];
  const handle = h.c.handleReplayEvent;
  h.c.handleReplayEvent = (...args) => { seen.push(args[1]); return handle(...args); };
  const pending = h.c.attachEvents();
  await h.settle();
  await h.frame(": keep-alive\n\n");
  assert.deepEqual(seen, []);
  await h.frame("data: ↳ speed · reasoning 6.7k tok @ 86.2 tok/s · output 415 tok @ 167 tok/s\n\n");
  assert.deepEqual(seen, ["rate", "rate"]);
  const box = h.c.recoveredView.box;
  const main = rateNodes(box)[0];
  assert.match(main.textContent, /86.2 tok\/s.*167 tok\/s/);
  assert.equal(box.lastElementChild.textContent, "Reconnecting… · 0:00");
  assert.doesNotMatch(box.lastElementChild.textContent, /tok/);
  await h.frame("event: done\ndata: \n\n");
  await pending;
});

test("replayed completion preserves reasoning and output caches before teardown", async () => {
  const h = attachmentHarness(false, { phone: true, realThinking: true });
  const pending = h.c.attachEvents();
  await h.settle();
  await h.frame('event: thinking\ndata: {"text":"saved thought","rate":"~10 tok @ 5 tok/s"}\n\n');
  await h.frame("event: thinking_done\ndata: \n\n");
  await h.frame('event: delta\ndata: {"delta":"saved answer"}\n\n');
  await h.frame("data: ↳ speed · output 40 tok @ 20 tok/s\n\n");
  await h.frame("event: done\ndata: \n\n");
  await pending;
  assert.equal(h.c.recoveredView, null);
  assert.equal(h.loads(), 1);
  const refreshed = historyRateHarness(h.storage, "saved answer", "saved thought", true);
  assert.equal(refreshed.restoredThinking(), "thinking · ~10 tok @ 5 tok/s");
  assert.equal(refreshed.restoredRate(), "↳ speed · output 40 tok @ 20 tok/s");
  assert.equal(refreshed.msgs.children.length, 1);
});

test("recovered output rates share live measurement while desktop status stays compact", () => {
  const h = replayHarness();
  h.emit("delta", { delta: "abcdefgh" });
  h.ageOutput(1000);
  h.emit("delta", { delta: "ijkl" });
  assert.match(h.mainRate(), /output ~3 tok @ 3.00 tok\/s/);
  assert.equal(h.status(), "Responding… · 0:01");
  assert.doesNotMatch(h.status(), /tok/);
  h.view.finish();
  assert.equal(historyRateHarness(h.storage, "abcdefghijkl").restoredRate(), h.mainRate());
});

for (const recovered of [false, true]) {
  test((recovered ? "replayed" : "live") + " phone rates use one styled status surface during generation", () => {
    const h = recovered ? replayHarness({ phone: true }) : streamHarness({ phone: true });
    h.emit("delta", { delta: "answer" });
    h.emit("progress", "↳ speed · reasoning 67 tok @ 69.6 tok/s · output 148 tok @ — tok/s");
    const status = recovered ? h.status() : h.pending();
    assert.doesNotMatch(status, /tok/, "the phone status bar must not own throughput");
    assert.match(h.mainRate(), /reasoning 67 tok @ 69\.6 tok\/s · output 148 tok/);
    assert.doesNotMatch(h.mainRate(), /— tok\/s/);
    const box = recovered ? h.view.box : h.box;
    assert.equal(rateNodes(box).length, 1);
    if (recovered) h.view.finish();
    else h.emit("done");
    const rates = rateNodes(box);
    assert.equal(rates.length, 1, "completed phone rates must remain visible");
    assert.match(rates[0].parentNode.className, /\bmsg\b/);
    assert.match(rates[0].parentNode.className, /\basst\b/);
    assert.ok(!box.children.some((child) => child.dataset.turnRate), "a bare rate escaped its card");
    const refreshed = historyRateHarness(h.storage, "answer", "", true);
    assert.equal(refreshed.restoredRate(), rates[0].textContent);
    assert.match(rateNodes(refreshed.group)[0].parentNode.className, /\bmsg\b/);
  });

  test((recovered ? "replayed" : "live") + " first thinking content and its completed timing survive later phases", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    h.emit("thinking_start");
    h.emit("thinking", { text: "first thought" });
    h.ageOutput(1000);
    h.emit("thinking_done");
    assert.ok(h.titles().includes("thinking · ~4 tok @ 4.00 tok/s"), "the first timing sample was left unmeasured");
    h.emit("tool", { id: "spawn", name: "task_spawn", phase: "started" });
    h.emit("tool", { id: "spawn", name: "task_spawn", phase: "finished", ok: true });
    h.emit("thinking_start");
    h.emit("thinking", { text: "second thought", rate: "~4 tok @ 8 tok/s" });
    h.emit("thinking_done");
    const box = recovered ? h.view.box : h.box;
    const thinking = box.children.filter((child) => child.summary?.textContent.startsWith("thinking"));
    assert.deepEqual(thinking.map((child) => child.body.textContent), ["first thought", "second thought"]);
    assert.ok(thinking.every((child) => !child.hidden && child.parentNode === box));
    if (recovered) h.view.finish();
    else h.emit("done");
    const refreshed = historyRateHarness(h.storage, "", "first thought", true);
    assert.equal(refreshed.restoredThinking(), "thinking · ~4 tok @ 4.00 tok/s");
    assert.equal(refreshed.group.children.find((child) => child.summary).body.textContent, "first thought");
  });
}

for (const recovered of [false, true]) {
  test((recovered ? "replayed" : "live") + " phone status shows the current thinking rate before output exists", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    h.emit("thinking", { text: "current thought", rate: "~1647 tok @ 86.2 tok/s" });
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /tok/);
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~1647 tok @ 86.2 tok/s"));
    h.emit("thinking_done");
    assert.ok(h.titles().includes("thinking · ~1647 tok @ 86.2 tok/s"));
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /reasoning ~1647/,
      "a closed thinking phase must not remain the active status rate");
    h.emit("delta", { delta: "early answer" });
    h.emit("progress", "↳ speed · output 40 tok @ 20 tok/s");
    h.emit("thinking", { text: "next thought", rate: "~431 tok @ 61 tok/s" });
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /tok/);
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~431 tok @ 61 tok/s"));
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /output 40/,
      "the previous answer's rate hid the current thinking measurement");
    h.emit("thinking_done");
    if (recovered) h.view.finish();
    else h.emit("done");
    const cold = coldHistoryHarness(h.storage.get("serve.rates"));
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: "current thought" }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~1647 tok @ 86.2 tok/s");
    assert.equal(cold.root.querySelector(".prog").textContent, "current thought");
  });

  test((recovered ? "replayed" : "live") + " phone clock persists measured thinking before refresh and instant replay", () => {
    const options = { phone: true, realThinking: true, phaseClock: true };
    const h = recovered ? replayHarness(options) : streamHarness(options);
    h.emit("thinking", { text: "first thought" });
    h.step();
    assert.equal(h.context.storedRate("thinking", "s", "first thought"), "~4 tok @ 4.00 tok/s",
      "the clock displayed a measurement without persisting it for refresh");
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /tok/);
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~4 tok @ 4.00 tok/s"));
    const refreshed = replayHarness({ phone: true, realThinking: true, storage: h.storage });
    refreshed.emit("thinking", { text: "first thought" });
    refreshed.emit("thinking_done");
    refreshed.view.finish();
    const cold = coldHistoryHarness(refreshed.storage.get("serve.rates"));
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: "first thought" }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~4 tok @ 4.00 tok/s");
    assert.equal(cold.root.querySelector(".prog").textContent, "first thought");
  });
}

for (const recovered of [false, true]) {
  test((recovered ? "replayed" : "live") + " desktop thinking throughput appears once in its fold while status retains elapsed time", () => {
    const options = { realThinking: true, phaseClock: true };
    const h = recovered ? replayHarness(options) : streamHarness(options);
    const status = () => recovered ? h.status() : h.pending();
    const prefix = recovered ? "" : "◌ ";
    h.emit("thinking", { text: "current thought", rate: "~1647 tok @ 86.2 tok/s" });
    h.emit("progress", "↳ speed · reasoning 1647 tok @ 86.2 tok/s");
    assert.equal(status(), prefix + "Thinking… · 0:00");
    assert.doesNotMatch(status(), /tok/);
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~1647 tok @ 86.2 tok/s"));
    h.step(88000);
    assert.equal(status(), prefix + "Thinking… · 1:28");
    assert.equal(h.mainRate(), "", "turn throughput duplicated the measured thinking rate");
    h.emit("thinking_done");
    assert.equal(status(), prefix + "Processing… · 1:28");
    assert.ok(h.titles().includes("thinking · ~1647 tok @ 86.2 tok/s"));
    if (recovered) h.view.finish();
    else h.emit("done");
    const refreshed = historyRateHarness(h.storage, "", "current thought");
    assert.equal(refreshed.restoredThinking(), "thinking · ~1647 tok @ 86.2 tok/s");
    assert.equal(refreshed.restoredRate(), "", "cold history duplicated the measured thinking rate");
  });
}

for (const recovered of [false, true]) {
  test((recovered ? "replayed" : "live") + " phone thinking folds receive late server samples across model rounds and cold reopen", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    h.emit("thinking_start");
    h.emit("thinking", { text: "first thought", rate: "~4 tok @ — tok/s" });
    h.emit("thinking_done");
    h.emit("delta", { delta: "first answer" });
    h.emit("progress", "↳ speed · reasoning 4 tok @ 8 tok/s · output 3 tok @ 6 tok/s");
    assert.ok(h.titles().includes("thinking · ~4 tok @ 8 tok/s"),
      "the final server reasoning sample never reached the closed thinking fold");
    h.emit("tool", { id: "lookup", name: "web_search", phase: "finished", ok: true });
    h.emit("thinking_start");
    h.emit("thinking", { text: "second thought", rate: "~4 tok @ — tok/s" });
    h.emit("thinking_done");
    h.emit("progress", "↳ speed · reasoning 4 tok @ 16 tok/s");
    // The last thinking phase finishes in the same read, without a later chunk
    // or a clock tick to repair its label and persisted measurement.
    if (recovered) h.view.finish();
    else h.emit("done");
    assert.deepEqual(h.titles().filter((title) => title.startsWith("thinking")), [
      "thinking · ~4 tok @ 8 tok/s", "thinking · ~4 tok",
    ]);
    const cold = coldHistoryHarness(h.storage.get("serve.rates"));
    cold.mount(JSON.stringify([
      { role: "assistant", content: "first answer", reasoning_content: "first thought" },
      { role: "assistant", content: "", reasoning_content: "second thought" },
    ]));
    assert.deepEqual(cold.root.querySelectorAll("summary").map((summary) => summary.textContent), [
      "thinking · ~4 tok @ 8 tok/s", "thinking · ~4 tok",
    ]);
  });

  test((recovered ? "replayed" : "live") + " phone persists the first measured thinking sample in every phase without another event", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    h.ageOutput(100);
    h.emit("thinking", { text: "first thought", rate: "~4 tok @ 8 tok/s" });
    h.emit("thinking_done");
    h.ageOutput(100);
    h.emit("thinking_start");
    h.emit("thinking", { text: "second thought", rate: "~4 tok @ 16 tok/s" });
    assert.doesNotMatch(recovered ? h.status() : h.pending(), /tok/);
    // The fixture's 16 tok/s implies a 250 ms window, below the 500 ms display
    // gate, so the timed sample renders tokens only and stays authoritative.
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~4 tok"));
    assert.equal(h.context.storedRate("thinking", "s", "second thought"), "~4 tok",
      "the current phase's valid sample was throttled behind an earlier phase");
    const refreshed = replayHarness({ phone: true, realThinking: true, storage: h.storage });
    refreshed.emit("thinking", { text: "second thought" });
    refreshed.emit("thinking_done");
    refreshed.view.finish();
    const cold = coldHistoryHarness(refreshed.storage.get("serve.rates"));
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: "second thought" }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~4 tok");
  });

  test((recovered ? "replayed" : "live") + " phone thinking uses generation throughput when reasoning usage is unavailable", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    const thought = "中".repeat(129);
    h.emit("thinking_start");
    h.emit("thinking", { text: thought });
    h.emit("thinking_done");
    h.emit("progress", "↳ speed · generation 180 tok @ 86.2 tok/s");
    if (recovered) h.view.finish();
    else h.emit("done");
    assert.ok(h.titles().includes("thinking · ~129 tok @ 86.2 tok/s"));
    assert.equal(h.context.storedRate("thinking", "s", thought), "~129 tok @ 86.2 tok/s");

    const cold = coldHistoryHarness(h.storage.get("serve.rates"), { phone: true });
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: thought }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~129 tok @ 86.2 tok/s");
  });

  test((recovered ? "replayed" : "live") + " phone thinking excludes unrelated and unmeasured terminal speeds", () => {
    const h = recovered ? replayHarness({ phone: true, realThinking: true }) : streamHarness({ phone: true, realThinking: true });
    h.emit("thinking_start");
    h.emit("thinking", { text: "first thought", rate: "~4 tok @ — tok/s" });
    h.emit("thinking_done");
    for (const speed of ["output 4 tok @ 8 tok/s", "reasoning 4 tok @ — tok/s", "reasoning 4 tok @ Infinity tok/s", "reasoning 4 tok @ -1 tok/s"]) {
      h.emit("progress", "↳ speed · " + speed);
      assert.ok(h.titles().some((title) => title.includes("~4 tok")));
      assert.equal(h.context.storedRate("thinking", "s", "first thought"), "~4 tok @ 16.0 tok/s",
        "unrelated or invalid server rates must preserve the bounded arrival estimate");
    }
    h.emit("thinking_start");
    // A new response without displayed reasoning cannot lend its terminal
    // reasoning measurement to the preceding response's fold.
    h.emit("progress", "↳ speed · reasoning 4 tok @ 8 tok/s");
    if (recovered) h.view.finish();
    else h.emit("done");
    assert.ok(h.titles().some((title) => title.includes("~4 tok")));
    const cold = coldHistoryHarness(h.storage.get("serve.rates"));
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: "first thought" }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~4 tok @ 16.0 tok/s");
  });
}

test("phone screenshot-sized history distinguishes persisted throughput from missing timing", () => {
  const first = "x".repeat(1647 * 4), second = "y".repeat(431 * 4);
  const h = coldHistoryHarness("{}");
  const key = "thinking|s|phase:fixture";
  h.storage.set("serve.rates", JSON.stringify({ [key]: {
    structured: true, textKey: h.context.rateTextKey(first, true), rate: "~1647 tok @ 86.2 tok/s",
  } }));
  const messages = Array.from({ length: 198 }, (_, i) => ({ role: "user", content: "question " + i }));
  messages.push({ role: "assistant", content: "", reasoning_content: first },
    { role: "assistant", content: "", reasoning_content: second });
  h.mount(JSON.stringify(messages), false);
  assert.equal(h.root.children.length, 200);
  const summaries = h.root.querySelectorAll("summary");
  assert.deepEqual(summaries.map((item) => item.textContent), [
    "thinking · ~1647 tok @ 86.2 tok/s", "thinking · ~431 tok",
  ]);
  assert.deepEqual(h.root.querySelectorAll(".prog").map((item) => item.textContent), [first, second]);
});

const screenshotThought = "The status bar rate removal worker (task_03504bdf97f1481fba03b62252ab9ef4) is completed. Let me collect it with task_wait.";

test("thinking rate floor covers zero and short elapsed time without changing output debounce", () => {
  const h = streamHarness({ realThinking: true });
  for (const elapsed of [0, 1, 50, 249, 250]) {
    assert.equal(h.context.measuredRateText(31, 0, elapsed), "~31 tok @ 124 tok/s");
    assert.equal(h.context.formatRateText(31, 0, elapsed), "~31 tok");
  }
  assert.equal(h.context.measuredRateText(31, 0, 1000), "~31 tok @ 31.0 tok/s");
  assert.equal(h.context.formatRateText(31, 0, 500), "~31 tok @ 62.0 tok/s");
  assert.equal(h.context.measuredRateText(31, null, 0), "");
  assert.equal(h.context.measuredRateText(0, 0, 0), "");
  assert.equal(h.context.rateTextFromSample({ phase: "output", tokens: 8, elapsed_ms: 6_667 }), "8 tok @ 1.2 tok/s");
  assert.equal(h.context.rateTextFromSample({ phase: "output", tokens: 8, elapsed_ms: null }), "8 tok");
});

for (const recovered of [false, true]) {
  for (const phone of [false, true]) {
    test(`same-tick thinking rate: ${recovered ? "recovered" : "live"} ${phone ? "phone" : "desktop"} survives canonical and cold history`, () => {
      const options = { phone, realThinking: true, clockStart: 0 };
      const h = recovered ? replayHarness(options) : streamHarness(options);
      h.emit("thinking_start");
      h.emit("thinking", { text: screenshotThought });
      h.emit("thinking_done");
      const title = h.titles().find((value) => value.startsWith("thinking"));
      assert.equal(title, "thinking · ~31 tok @ 124 tok/s");
      const rate = /@\s+(\S+)\s+tok\/s/.exec(title);
      assert.ok(rate && Number.isFinite(Number(rate[1])), "same-tick throughput must be finite");
      assert.equal(h.context.storedRate("thinking", "s", screenshotThought), "~31 tok @ 124 tok/s");
      if (recovered) h.view.finish();
      else h.emit("done");
      assert.equal(historyRateHarness(h.storage, "", screenshotThought, phone).restoredThinking(), title);
      const cold = coldHistoryHarness(h.storage.get("serve.rates"), { phone });
      cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: screenshotThought }]));
      assert.equal(cold.root.querySelector("summary").textContent, title);
    });
  }
}

for (const recovered of [false, true]) {
  for (const completionDelay of [0, 10]) {
    test(`thinking phase-start timestamp: ${recovered ? "replay" : "live"} single chunk completes at ${50 + completionDelay}ms`, () => {
      const options = { phone: true, realThinking: true, clockStart: 0 };
      const h = recovered ? replayHarness(options) : streamHarness(options);
      h.emit("thinking_start");
      h.ageOutput(50);
      h.emit("thinking", { text: screenshotThought });
      assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~31 tok @ 124 tok/s"),
        "the first sample must use the floor for a phase shorter than 250ms");
      h.ageOutput(completionDelay);
      h.emit("thinking_done");
      const title = h.titles().find((value) => value.startsWith("thinking"));
      assert.equal(title, "thinking · ~31 tok @ 124 tok/s",
        "completion must retain the minimum thinking window");
      if (recovered) h.view.finish();
      else h.emit("done");
    });
  }

  for (const emptyPhase of [false, true]) {
    test(`thinking phase-start timestamp: ${recovered ? "replay" : "live"} single chunk ${emptyPhase ? "after an empty phase" : "without a start"} starts an arrival window`, () => {
      const options = { phone: true, realThinking: true, clockStart: 0 };
      const h = recovered ? replayHarness(options) : streamHarness(options);
      if (emptyPhase) {
        h.emit("thinking_start");
        h.ageOutput(40);
        h.emit("thinking_done");
      }
      h.ageOutput(emptyPhase ? 10 : 50);
      h.emit("thinking", { text: screenshotThought });
      h.emit("thinking_done");
      const title = h.titles().find((value) => value.startsWith("thinking"));
      assert.equal(title, "thinking · ~31 tok @ 124 tok/s");
      assert.equal(h.context.storedRate("thinking", "s", screenshotThought), "~31 tok @ 124 tok/s");
      if (recovered) h.view.finish();
      else h.emit("done");
    });
  }
}

test("thinking rate regression: a short measured phase survives completion, refresh, and a 200-row cold reopen", () => {
  const h = streamHarness({ phone: true, realThinking: true });
  h.emit("thinking_start");
  h.emit("thinking", { text: screenshotThought });
  h.ageOutput(250);
  h.emit("thinking_done");
  const title = h.titles().find((value) => value.startsWith("thinking"));
  assert.equal(title, "thinking · ~31 tok @ 124 tok/s");
  h.emit("done");
  const refreshed = historyRateHarness(h.storage, "", screenshotThought, true);
  assert.equal(refreshed.restoredThinking(), title);
  const cold = coldHistoryHarness(h.storage.get("serve.rates"));
  const messages = Array.from({ length: 198 }, (_, i) => ({ role: "user", content: "question " + i }));
  messages.push({ role: "assistant", content: "", reasoning_content: "\n" + screenshotThought + "\n" },
    { role: "assistant", content: "", reasoning_content: "Legacy thinking with no timing sample." });
  cold.mount(JSON.stringify(messages), false);
  assert.equal(cold.root.children.length, 200);
  assert.deepEqual(cold.root.querySelectorAll("summary").map((summary) => summary.textContent), [
    title, "thinking · ~10 tok",
  ]);
  assert.equal(cold.root.querySelector(".prog").textContent, screenshotThought);
  const replay = replayHarness({ phone: true, realThinking: true });
  replay.emit("thinking", { text: screenshotThought, rate: "~31 tok" });
  replay.emit("thinking_done");
  assert.ok(replay.titles().includes("thinking · ~31 tok @ 124 tok/s"), "same-tick replay must retain its bounded arrival estimate");
  replay.view.finish();
});

test("short-thinking arrival floor: recovered SSE saves a rate immediately before phase completion", async () => {
  const h = attachmentHarness(false, { phone: true, realThinking: true });
  const pending = h.c.attachEvents();
  await h.settle();
  await h.frame('event: thinking_start\ndata: \n\nevent: thinking\ndata: ' + JSON.stringify({ text: screenshotThought }) + '\n\n');
  const view = h.c.recoveredView;
  const caption = () => view.box.children.find((child) => child.summary).summary.textContent;
  assert.equal(caption(), "thinking · ~31 tok @ 124 tok/s");
  h.advance(250);
  assert.equal(caption(), "thinking · ~31 tok @ 124 tok/s", "the running recovered fold waited for thinking_done");
  assert.equal(h.c.eventActive, true);
  const serialized = h.storage.get("serve.rates");
  const cold = coldHistoryHarness(serialized);
  cold.mount(JSON.stringify([
    { role: "assistant", content: "", reasoning_content: screenshotThought },
    { role: "assistant", content: "", reasoning_content: "Legacy thinking with no timing sample." },
  ]));
  assert.deepEqual(cold.root.querySelectorAll("summary").map((summary) => summary.textContent), [
    "thinking · ~31 tok @ 124 tok/s", "thinking · ~10 tok",
  ]);
  await h.frame('event: thinking_done\ndata: \n\n');
  assert.equal(caption(), "thinking · ~31 tok @ 124 tok/s");
  assert.equal(h.storage.get("serve.rates"), serialized);
  await h.frame('event: done\ndata: \n\n');
  await pending;
  assert.equal(historyRateHarness(h.storage, "", screenshotThought, true).restoredThinking(), "thinking · ~31 tok @ 124 tok/s");
  assert.equal(h.timers.size, 0, "completion left an early sampling timer alive");
  h.advance(1000);
  assert.equal(caption(), "thinking · ~31 tok @ 124 tok/s");
});

for (const recovered of [false, true]) {
  test(`short-thinking early repaint: ${recovered ? "recovered" : "live"} zero-start clock survives phase end and cold restore`, () => {
    const options = { phone: true, realThinking: true, clockStart: 0 };
    const h = recovered ? replayHarness(options) : streamHarness(options);
    h.emit("thinking_start");
    h.emit("thinking", { text: screenshotThought });
    assert.ok(h.titles().includes((recovered ? "thinking" : "thinking…") + " · ~31 tok @ 124 tok/s"));
    h.ageOutput(250);
    h.emit("thinking_done");
    assert.ok(h.titles().includes("thinking · ~31 tok @ 124 tok/s"));
    if (recovered) h.view.finish();
    else h.emit("done");
    const cold = coldHistoryHarness(h.storage.get("serve.rates"));
    cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: screenshotThought }]));
    assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~31 tok @ 124 tok/s");
  });
}

for (const recovered of [false, true]) {
  test(`thinking rate regression: ${recovered ? "replay" : "live"} keeps a real sample with the complete body during the write throttle`, () => {
    const options = { phone: true, realThinking: true };
    const h = recovered ? replayHarness(options) : streamHarness(options);
    h.emit("thinking_start");
    h.emit("thinking", { text: screenshotThought.slice(0, 70), rate: "~31 tok @ 62 tok/s" });
    h.ageOutput(100);
    h.emit("thinking", { text: screenshotThought.slice(70), rate: "~31 tok" });
    const reopen = () => {
      const cold = coldHistoryHarness(h.storage.get("serve.rates"));
      cold.mount(JSON.stringify([
        { role: "assistant", content: "", reasoning_content: screenshotThought },
        { role: "assistant", content: "", reasoning_content: screenshotThought + " No sample for this separate phase." },
      ]));
      const summaries = cold.root.querySelectorAll("summary");
      assert.equal(summaries[0].textContent, "thinking · ~31 tok @ 62 tok/s");
      assert.doesNotMatch(summaries[1].textContent, /tok\/s/);
      assert.equal(cold.root.querySelector(".prog").textContent, screenshotThought);
    };
    reopen(); // Refresh may happen before the phase's closing event is read.
    h.emit("thinking_done");
    if (recovered) h.view.finish();
    else h.emit("done");
    reopen();
    assert.equal(Object.keys(JSON.parse(h.storage.get("serve.rates"))).filter((key) => key.startsWith("thinking|")).length, 1,
      "intermediate bodies must not consume historical cache entries");
  });
}

test("thinking rate regression: growing phases do not evict completed samples at the 1000-entry cap", () => {
  const h = streamHarness({ phone: true, realThinking: true });
  h.emit("thinking", { text: screenshotThought, rate: "~31 tok @ 62 tok/s" });
  h.emit("thinking_done");
  h.emit("thinking_start");
  h.emit("thinking", { text: " \n", rate: "~1 tok @ 2 tok/s" });
  for (let i = 0; i < 1001; i++) {
    h.ageOutput(1000);
    h.emit("thinking", { text: "abcd", rate: "~1001 tok @ 42 tok/s" });
  }
  h.emit("thinking_done");
  const cold = coldHistoryHarness(h.storage.get("serve.rates"));
  cold.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: screenshotThought }]));
  assert.equal(cold.root.querySelector("summary").textContent, "thinking · ~31 tok @ 62 tok/s");
  assert.equal(Object.keys(JSON.parse(h.storage.get("serve.rates"))).length, 2);
});

test("thinking rate regression: full-body keys distinguish sample-less rows with identical heads and tails", () => {
  const h = streamHarness({ realThinking: true });
  const head = "a".repeat(512), tail = "z".repeat(512);
  const measured = head + "measured" + tail, unmeasured = head + "untimed!" + tail;
  h.context.rememberRate("thinking", "s", measured, "~258 tok @ 86 tok/s");
  const cold = coldHistoryHarness(h.storage.get("serve.rates"));
  cold.mount(JSON.stringify([
    { role: "assistant", content: "", reasoning_content: measured },
    { role: "assistant", content: "", reasoning_content: unmeasured },
  ]));
  assert.deepEqual(cold.root.querySelectorAll("summary").map((summary) => summary.textContent), [
    "thinking · ~258 tok @ 86 tok/s", "thinking · ~258 tok",
  ]);
  const legacyKey = "thinking|s|" + h.context.rateTextKey(measured);
  const legacy = coldHistoryHarness(JSON.stringify({ [legacyKey]: "~258 tok @ 86 tok/s" }));
  legacy.mount(JSON.stringify([{ role: "assistant", content: "", reasoning_content: measured }]));
  assert.equal(legacy.root.querySelector("summary").textContent, "thinking · ~258 tok");
});

test("thinking rate regression: a new phase cannot replace a completed sample merely by sharing its prefix", () => {
  const h = streamHarness({ realThinking: true });
  h.emit("thinking", { text: "first thought", rate: "~4 tok @ 8 tok/s" });
  h.emit("thinking_done");
  h.emit("thinking_start");
  h.emit("thinking", { text: "first thought", rate: "~4 tok @ 16 tok/s" });
  h.ageOutput(100);
  h.emit("thinking", { text: " continues in another phase", rate: "~10 tok @ 20 tok/s" });
  h.emit("thinking_done");
  const cold = coldHistoryHarness(h.storage.get("serve.rates"));
  cold.mount(JSON.stringify([
    { role: "assistant", content: "", reasoning_content: "first thought" },
    { role: "assistant", content: "", reasoning_content: "first thought continues in another phase" },
  ]));
  assert.deepEqual(cold.root.querySelectorAll("summary").map((summary) => summary.textContent), [
    "thinking · ~4 tok @ 8 tok/s", "thinking · ~10 tok @ 20 tok/s",
  ]);
  assert.equal(Object.keys(JSON.parse(h.storage.get("serve.rates"))).length, 2);
});

test("instant thinking replay preserves an existing measured rate and fold body", () => {
  const cached = streamHarness({ realThinking: true });
  cached.context.rememberRate("thinking", "s", "first thought", "~4 tok @ 8 tok/s");
  const h = replayHarness({ phone: true, realThinking: true, storage: cached.storage });
  h.emit("thinking", { text: "first thought" });
  h.emit("thinking_done");
  assert.ok(h.titles().includes("thinking · ~4 tok @ 8 tok/s"));
  assert.equal(h.context.storedRate("thinking", "s", "first thought"), "~4 tok @ 8 tok/s");
  assert.equal(h.view.box.children.find((child) => child.summary).body.textContent, "first thought");
  h.view.finish();
  assert.equal(historyRateHarness(h.storage, "", "first thought", true).restoredThinking(), "thinking · ~4 tok @ 8 tok/s");
});

test("unmeasured structured samples never render a dash rate", () => {
  const h = streamHarness({ realThinking: true });
  assert.equal(h.context.formatRateText(10, 1000, 1100), "~10 tok");
  const thoughtKey = "thinking|s|phase:fixture";
  const answerKey = "turn|s|" + h.context.rateTextKey("old answer");
  h.storage.set("serve.rates", JSON.stringify({
    [thoughtKey]: { structured: true, textKey: h.context.rateTextKey("old thought", true), rate: "~10 tok" },
    [answerKey]: { structured: true, rate: "reasoning 67 tok @ 69.6 tok/s · output 148 tok" },
  }));
  for (const phone of [false, true]) {
    const refreshed = historyRateHarness(h.storage, "old answer", "old thought", phone);
    assert.equal(refreshed.restoredThinking(), "thinking · ~10 tok");
    assert.equal(refreshed.restoredRate(), "↳ speed · reasoning 67 tok @ 69.6 tok/s · output 148 tok");
    assert.doesNotMatch(refreshed.restoredRate() + refreshed.restoredThinking(), /— tok\/s/);
  }
});

const encryptedHistoryPrefix = "\u001eaios:reasoning-encrypted-replay:v1\u001f";
function encryptedHistoryReasoning(summary = []) {
  return encryptedHistoryPrefix + JSON.stringify({
    model: "synthetic-responses-model", endpoint: "non-secret-resource",
    items: [{ type: "reasoning", encrypted_content: "OPAQUE_PAYLOAD".repeat(500), summary }],
  });
}

for (const phone of [false, true]) {
  test(`history reasoning display preserves ordinary plaintext and canonical bytes (phone=${phone})`, () => {
    const h = coldHistoryHarness("{}", { phone });
    const payload = JSON.stringify([{ role: "assistant", content: "saved answer", reasoning_content: "\n readable thought \n" }]);
    h.mount(payload);
    assert.equal(h.root.querySelector(".prog").textContent, "readable thought");
    assert.match(h.root.querySelector("summary").textContent, /^thinking · ~\d+ tok$/);
    assert.equal(vm.runInContext("JSON.stringify(coldMessages)", h.context), payload);
  });

  test(`history reasoning display hides encrypted replay without counting it (phone=${phone})`, () => {
    const h = coldHistoryHarness("{}", { phone }), opaque = encryptedHistoryReasoning();
    const payload = JSON.stringify([
      { role: "assistant", content: "", reasoning_content: opaque },
      { role: "assistant", content: "saved answer", reasoning_content: opaque },
    ]);
    h.context.estimateTextTokens = () => assert.fail("opaque replay must not be token-counted");
    const storedRate = h.context.storedRate;
    h.context.storedRate = (kind, sid, text) => {
      assert.notEqual(text, opaque, "opaque replay must not supply thinking metadata");
      return storedRate(kind, sid, text);
    };
    h.mount(payload);
    assert.equal(h.root.children.length, 1, "an empty replay-only message has no display row");
    assert.equal(h.root.querySelector("summary"), null);
    assert.equal(h.root.textContent, "saved answer");
    assert.equal(rateNodes(h.root).length, 0);
    assert.equal(vm.runInContext("coldRows.rows[0].historyIndex", h.context), 1);
    assert.equal(vm.runInContext("coldRows.shownCount", h.context), 2);
    assert.equal(vm.runInContext("JSON.stringify(coldMessages)", h.context), payload);
  });

  test(`history reasoning display shows only typed plaintext summaries from encrypted replay (phone=${phone})`, () => {
    const h = coldHistoryHarness("{}", { phone }), counted = [];
    const opaque = encryptedHistoryReasoning([
      { type: "summary_text", text: " First step. " },
      { type: "summary_text", text: "" },
      { type: "encrypted_content", text: "OPAQUE_SUMMARY" },
      { type: "summary_text", text: 42 },
      { type: "summary_text", text: "Next step." },
    ]);
    const payload = JSON.stringify([{ role: "assistant", content: "saved answer", reasoning_content: opaque }]);
    const estimate = h.context.estimateTextTokens;
    h.context.estimateTextTokens = (text) => { counted.push(text); return estimate(text); };
    h.mount(payload);
    assert.equal(h.root.querySelector(".prog").textContent, "First step.\n\nNext step.");
    assert.deepEqual(counted, ["First step.\n\nNext step."]);
    assert.equal(h.root.querySelector("summary").textContent, "thinking · ~6 tok");
    assert.doesNotMatch(h.root.textContent, /OPAQUE|aios:|non-secret-resource/);
    assert.equal(vm.runInContext("JSON.stringify(coldMessages)", h.context), payload);
  });
}

test("history reasoning display hides malformed and unsupported encrypted replay summaries", () => {
  const wrappers = [
    encryptedHistoryPrefix + "{", encryptedHistoryPrefix + "null",
    encryptedHistoryPrefix + JSON.stringify({ items: {} }),
    encryptedHistoryPrefix + JSON.stringify({ items: [null, { type: "message", summary: [{ type: "summary_text", text: "wrong item" }] }] }),
    encryptedHistoryReasoning(["untyped summary", { text: "missing type" }, { type: "summary_text", text: "  " }]),
  ];
  for (const reasoning of wrappers) {
    const h = coldHistoryHarness("{}");
    h.context.estimateTextTokens = () => assert.fail("invalid replay must not be token-counted");
    h.mount(JSON.stringify([{ role: "assistant", content: "saved answer", reasoning_content: reasoning }]));
    assert.equal(h.root.querySelector("summary"), null);
    assert.equal(h.root.textContent, "saved answer");
  }
});

test("history reasoning display filtering leaves live SSE thinking unchanged", () => {
  const h = streamHarness({ realThinking: true });
  h.emit("thinking_start");
  h.emit("thinking", { text: "Live ", rate: "~3 tok @ 6 tok/s" });
  h.emit("thinking", { text: "summary.", rate: "~3 tok @ 6 tok/s" });
  h.emit("thinking_done");
  assert.ok(h.bodies().includes("Live summary."));
  assert.ok(h.titles().includes("thinking · ~3 tok @ 6 tok/s"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("done");
  assert.equal(h.reply(), "Final answer");
});

test("desktop cold reload restores thinking rates and counts across a 316-message history", () => {
  const uncachedThought = "x".repeat(312);
  const messages = Array.from({ length: 158 }, (_, i) => [
    { role: "user", content: "question " + i },
    { role: "assistant", content: i === 0 ? "saved answer" : "answer " + i,
      reasoning_content: i === 0 ? "\n saved  thought \n" : i === 1 ? uncachedThought : "thought " + i },
  ]).flat();
  const seed = coldHistoryHarness("{}");
  const savedRates = JSON.stringify({
    "thinking|s|phase:fixture": { structured: true, textKey: seed.context.rateTextKey(messages[1].reasoning_content, true), rate: "~4 tok @ 8 tok/s" },
    ["turn|s|" + seed.context.rateTextKey("saved answer")]: { structured: true, rate: "output 40 tok @ 20 tok/s" },
  });
  const h = coldHistoryHarness(savedRates, { phone: false });
  assert.equal(h.context.storedRate("thinking", "s", messages[1].reasoning_content), "~4 tok @ 8 tok/s");
  assert.equal(h.context.storedRate("thinking", "s", uncachedThought), "");
  h.mount(JSON.stringify(messages), false);
  assert.equal(h.root.children.length, 316, "desktop history lost rows beyond the phone window");
  const summaries = h.root.querySelectorAll("summary");
  assert.equal(summaries.length, 158);
  assert.equal(summaries[0].textContent, "thinking · ~4 tok @ 8 tok/s");
  assert.equal(summaries[1].textContent, "thinking · ~78 tok");
  assert.ok(summaries.every((summary) => summary.textContent.startsWith("thinking · ")));
  assert.equal(h.root.children[3].querySelector(".prog").textContent, uncachedThought);
  const rates = rateNodes(h.root);
  assert.equal(rates.length, 1);
  assert.equal(rates[0].textContent, "↳ speed · output 40 tok @ 20 tok/s");
  assert.equal(rates[0].parentNode, h.root.children[1].querySelector(".msg.asst.md"));
  assert.doesNotMatch(h.root.textContent, /— tok\/s/);
  assert.equal(h.root.querySelector(".pending"), null);
  assert.equal(h.storage.get("serve.rates"), savedRates);
});

test("phone cold reopen restores persisted rates inside assistant cards without live turn state", () => {
  const seed = coldHistoryHarness("{}");
  const savedRates = JSON.stringify({
    "thinking|s|phase:fixture": { structured: true, textKey: seed.context.rateTextKey("saved thought", true), rate: "~4 tok @ 8 tok/s" },
    ["turn|s|" + seed.context.rateTextKey("saved answer")]: { structured: true, rate: "reasoning 67 tok @ 69.6 tok/s · output 148 tok @ 20 tok/s" },
  });
  const h = coldHistoryHarness(savedRates);
  assert.equal(h.root.children.length, 0, "a cold document already contains a turn");
  assert.equal(h.context.$("turnstatus"), null);
  assert.equal(vm.runInContext("[typeof turns, typeof phaseBubble, typeof tokenRate, typeof recoveredView].join(',')", h.context),
    "undefined,undefined,undefined,undefined");
  h.mount(JSON.stringify([{ role: "assistant", content: "saved answer", reasoning_content: "saved thought" }]));
  assert.equal(h.root.children.length, 1, "rate restoration added a canonical history row");
  const group = h.root.lastElementChild, rates = rateNodes(h.root);
  assert.equal(rates.length, 1, "the reopened phone lost its cached rate");
  assert.equal(rates[0].textContent, "↳ speed · reasoning 67 tok @ 69.6 tok/s · output 148 tok @ 20 tok/s");
  assert.equal(rates[0].parentNode, group.querySelector(".msg.asst.md"));
  assert.ok(!group.children.some((child) => child.dataset.turnRate), "a bare rate escaped its card");
  for (let node = rates[0]; node; node = node.parentNode) assert.equal(node.hidden, false);
  assert.equal(group.querySelector("summary").textContent, "thinking · ~4 tok @ 8 tok/s");
  assert.equal(group.querySelector(".prog").textContent, "saved thought");
  assert.equal(h.root.querySelector(".pending"), null, "cold history depends on a live status bubble");
  assert.equal(h.storage.get("serve.rates"), savedRates, "history replay rewrote persisted timing");
});

test("phone cold reopen omits rate text when structured samples lack timing", () => {
  const seed = coldHistoryHarness("{}");
  const savedRates = JSON.stringify({
    "thinking|s|phase:fixture": { structured: true, textKey: seed.context.rateTextKey("saved thought", true), rate: "~4 tok" },
    ["turn|s|" + seed.context.rateTextKey("saved answer")]: { structured: true, rate: "reasoning 67 tok @ 69.6 tok/s · output 148 tok" },
  });
  const h = coldHistoryHarness(savedRates);
  h.mount(JSON.stringify([{ role: "assistant", content: "saved answer", reasoning_content: "saved thought" }]));
  const rates = rateNodes(h.root);
  assert.equal(rates.length, 1);
  assert.equal(rates[0].textContent, "↳ speed · reasoning 67 tok @ 69.6 tok/s · output 148 tok");
  assert.ok(rates[0].parentNode.matches(".msg.asst.md"));
  assert.equal(h.root.querySelector("summary").textContent, "thinking · ~4 tok");
  assert.equal(h.root.querySelector(".prog").textContent, "saved thought");
  assert.doesNotMatch(h.root.textContent, /— tok\/s/);
  assert.equal(h.root.querySelector(".pending"), null);
  assert.equal(h.storage.get("serve.rates"), savedRates);
});

test("a thinking-only turn restores one rate inside its fold", () => {
  const h = streamHarness({ realThinking: true });
  h.emit("thinking", { text: "tool reasoning", rate: "~10 tok @ 5 tok/s" });
  h.emit("tool", { phase: "start", name: "read_file", call_id: "r" });
  h.emit("", "↳ speed · reasoning 20 tok @ 10 tok/s");
  h.emit("done");
  const refreshed = historyRateHarness(h.storage, "", "tool reasoning");
  assert.equal(refreshed.restoredThinking(), "thinking · ~20 tok @ 10 tok/s");
  assert.equal(refreshed.restoredRate(), "");
});

test("a recovered single output chunk gains a main rate on the clock and stops ticking at completion", () => {
  const h = replayHarness(), intervals = new Map();
  h.view.remove();
  h.context.setInterval = (fn, ms) => { intervals.set(1, { fn, ms }); return 1; };
  h.context.clearInterval = (id) => intervals.delete(id);
  const view = h.context.recoveredProgressView();
  view.delta("abcdefgh");
  assert.equal(rateNodes(view.box).length, 0);
  const tick = intervals.get(1);
  assert.equal(tick.ms, 1000);
  h.ageOutput(1000);
  tick.fn();
  const line = rateNodes(view.box)[0];
  assert.equal(line.textContent, "↳ speed · output ~2 tok @ 2.00 tok/s");
  assert.equal(view.box.lastElementChild.textContent, "Responding… · 0:01");
  assert.doesNotMatch(view.box.lastElementChild.textContent, /tok/);
  view.finish();
  assert.equal(intervals.size, 0);
  h.ageOutput(1000);
  tick.fn();
  assert.equal(line.textContent, "↳ speed · output ~2 tok @ 2.00 tok/s", "a queued clock tick changed completed throughput");
});

test("untyped legacy rate cache entries are ignored", () => {
  const h = streamHarness(), canonical = "answer\n\n[Runtime warning]\nold warning";
  const key = "turn|s|" + h.context.rateTextKey(canonical);
  h.storage.set("serve.rates", JSON.stringify({ [key]: "output 40 tok @ 20 tok/s" }));
  assert.equal(historyRateHarness(h.storage, canonical).restoredRate(), "");
});

test("an invalid cache container is replaced by newly recorded throughput", () => {
  for (const cache of ["true", "[]"]) {
    const h = streamHarness({ storage: new Map([["serve.rates", cache]]) });
    h.emit("delta", { delta: "answer" });
    h.emit("", "↳ speed · output 40 tok @ 20 tok/s");
    h.emit("done");
    assert.equal(historyRateHarness(h.storage, "answer").restoredRate(), "↳ speed · output 40 tok @ 20 tok/s");
  }
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
  h.emit("", "↳ speed · output 40 tok @ 20 tok/s");
  h.fail();
  assert.ok(h.titles().some((title) => title.includes("interrupted")));
  assert.ok(h.bodies().some((text) => text.includes("B")));
  assert.equal(h.mainRate(), "↳ speed · output 40 tok @ 20 tok/s");
});

for (const recovered of [false, true]) {
  for (const phone of [false, true]) {
    test(`active subagent progress: ${recovered ? "recovered" : "live"} ${phone ? "phone" : "desktop"} opens details and summarizes the latest command`, () => {
      const h = recovered ? replayHarness({ phone }) : streamHarness({ phone });
      h.emit("tool", { id: "wait", name: "task_wait", phase: "started" });
      const header = "subagents · 1 active\n└─ worker · running 5s · task: inspect thinking rates\n";
      const command = "execute_command rg -n thinking app.html";
      h.emit("subagent_progress", structuredSubagents([structuredTask("worker", 5000, { progress: { phase: "working", latest_activity: command } })], header));
      const box = recovered ? h.view.box : h.box;
      const card = box.children.find((item) => item.summary?.textContent.startsWith("subagents"));
      const details = card.querySelector("details");
      assert.equal(details.open, true, "active child progress must be visible without opening task_wait");
      assert.ok(card.summary.textContent.includes(command), "the summary omitted the current command");
      const latest = "read_file Cargo.toml";
      h.emit("subagent_progress", structuredSubagents([structuredTask("worker", 5000, { progress: { phase: "working", latest_activity: latest } })], header));
      assert.equal(details.open, true);
      assert.ok(card.summary.textContent.includes(latest));
      assert.ok(!card.summary.textContent.includes(command));
      h.emit("subagent_progress", structuredSubagents([]));
      assert.equal(details.open, false, "completed subagents must leave a collapsed card");
      h.emit("subagent_progress", structuredSubagents([structuredTask("worker", 5000, { progress: { phase: "working", latest_activity: latest } })], header));
      assert.equal(details.open, true);
      if (recovered) h.view.finish();
      else h.emit("done");
      assert.equal(details.open, false, "turn completion must collapse the progress card");
    });
  }
}

for (const phone of [false, true]) {
  test(`reader collapse survives streaming subagent progress ${phone ? "phone" : "desktop"}`, () => {
    const h = streamHarness({ phone });
    const header = "subagents · 1 active\n└─ worker · running 5s · task: inspect\n";
    const active = (command) => structuredSubagents(
      [structuredTask("worker", 5000, { progress: { phase: "working", latest_activity: command } })], header);
    h.emit("subagent_progress", active("execute_command ls"));
    const card = h.box.children.find((item) => item.summary?.textContent.startsWith("subagents"));
    const details = card.querySelector("details");
    assert.equal(details.open, true, "a run opens the card once");
    details.open = false;
    h.emit("subagent_progress", active("read_file a"));
    assert.equal(details.open, false, "streaming progress must not re-open a card the reader collapsed");
    h.emit("subagent_progress", active("read_file b"));
    assert.equal(details.open, false);
  });
}

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

test("subagent elapsed ticker advances explicit numeric fields and reanchors to server snapshots", () => {
  const h = streamHarness({ activityOnly: true });
  const snapshot = structuredSubagents([
    structuredTask("alpha", 59000, { description: "audit" }),
    structuredTask("beta", 119000, { description: "implement" }),
    structuredTask("gamma", 423000, { status: "completed", active: false, description: "running 5s", output: "exact output\n" }),
  ], "raw terminal text containing running 999s");
  h.progress(snapshot); const card = h.box.children[0]; h.step();
  let identities = structuredIdentityRows(h);
  assert.ok(identities.some((text) => text.includes("alpha") && text.includes("running 1m0s")));
  assert.ok(identities.some((text) => text.includes("beta") && text.includes("running 2m0s")));
  assert.ok(identities.some((text) => text.includes("gamma") && text.includes("completed 7m3s")));
  assert.equal(h.bodies()[0], snapshot.text, "raw backend text is never rewritten by the clock");
  assert.equal(h.intervals.size, 1); assert.equal([...h.intervals.values()][0].delay, 1000);
  assert.equal(h.box.children[0], card); assert.equal(card.dataset.subagents, "true");
  h.step(5000);
  assert.ok(structuredIdentityRows(h).some((text) => text.includes("alpha") && text.includes("running 1m5s")));
  h.progress(structuredSubagents([structuredTask("alpha", 8000)]));
  h.step(2000);
  assert.ok(structuredIdentityRows(h).some((text) => text.includes("alpha") && text.includes("running 10s")));
  h.newActivity().subagents(structuredSubagents([structuredTask("other", 42000)])); h.step();
  assert.equal(h.intervals.size, 1, "all activity cards share one interval");
  assert.ok(structuredIdentityRows(h).some((text) => text.includes("other") && text.includes("running 43s")));
});

test("subagent elapsed ticker uses stable task ids across row and progress updates", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress(structuredSubagents([structuredTask("worker", 0), structuredTask("peer", 3000)]));
  h.step(4000);
  h.progress(structuredSubagents([structuredTask("worker", 4000, { progress: { phase: "working", latest_activity: "reading files" } })]));
  h.step(2000);
  assert.ok(structuredIdentityRows(h).some((text) => text.includes("worker") && text.includes("running 6s")));
  h.progress(structuredSubagents([structuredTask("worker", 6000, { progress: { phase: "working", latest_activity: "editing files" } })]));
  h.step(2000);
  assert.ok(structuredIdentityRows(h).some((text) => text.includes("worker") && text.includes("running 8s")));
  assert.ok(h.titles()[0].includes("editing files"));
  h.progress(structuredSubagents([])); assert.equal(h.intervals.size, 0);
  h.progress("worker: running"); h.step();
  assert.equal(h.bodies()[0], "worker: running", "legacy text does not invent an elapsed clock");
});

test("subagent elapsed ticker preserves terminal text and stops on explicit completion", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress(structuredSubagents([structuredTask("worker", 3660000)])); h.step();
  assert.ok(structuredIdentityRows(h)[0].includes("running 1h1m1s"));
  const terminal = "worker: completed · 1h1m7s\n   ↳ final: running 42s\n";
  h.progress(structuredSubagents([structuredTask("worker", 3667000, { status: "completed", active: false, output: "raw output\n" })], terminal));
  const title = h.titles()[0]; const identities = structuredIdentityRows(h);
  assert.equal(h.intervals.size, 0); h.step(30000);
  assert.equal(h.bodies()[0], terminal); assert.equal(h.titles()[0], title);
  assert.deepEqual(structuredIdentityRows(h), identities);
  h.progress(structuredSubagents([structuredTask("new-worker", 9000)])); h.step(); h.finish("ended");
  const frozen = structuredIdentityRows(h);
  assert.equal(h.intervals.size, 0); h.step(3000); assert.deepEqual(structuredIdentityRows(h), frozen);
});

test("subagent elapsed ticker pauses while hidden or detached and resumes one interval", () => {
  const h = streamHarness({ activityOnly: true });
  h.progress(structuredSubagents([structuredTask("worker", 42000)])); h.step(); h.visible(false);
  assert.equal(h.intervals.size, 0); h.step(5000);
  assert.ok(structuredIdentityRows(h)[0].includes("running 43s")); h.visible(true);
  assert.ok(structuredIdentityRows(h)[0].includes("running 48s")); h.visible(true); assert.equal(h.intervals.size, 1);
  h.fire("pagehide"); assert.equal(h.intervals.size, 0); h.step(3000);
  assert.ok(structuredIdentityRows(h)[0].includes("running 48s")); h.fire("pageshow");
  assert.ok(structuredIdentityRows(h)[0].includes("running 51s")); h.detach(); assert.equal(h.intervals.size, 0); h.step(4000);
  assert.ok(structuredIdentityRows(h)[0].includes("running 51s")); h.attach();
  assert.ok(structuredIdentityRows(h)[0].includes("running 55s")); assert.equal(h.intervals.size, 1);
  h.remove(h.box.children[0]); assert.equal(h.intervals.size, 0); assert.deepEqual(h.bodies(), []);
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
  assert.equal(h.status(), "Responding… · 0:00");
  h.emit("replace", { text: "" });
  assert.equal(h.reply(), "");
  assert.equal(h.status(), "Processing… · 0:00");
});

test("EOF without the done event follows server history instead of declaring completion", () => {
  assert.match(script, /if \(!finished && !turnError\) \{[\s\S]*?stream ended before completion[\s\S]*?goStale/);
  assert.match(script, /const goStale = \(msg\) => \{[\s\S]*?staleSession = sessionId;[\s\S]*?staleDone = false;[\s\S]*?startStaleFollow\(\);/,
    "a lost accepted stream must follow unresolved server history");
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
  assert.match(script, /if \(block\.kind === "done"\) \{\s+view\.finish\(\);\s+finished = true; eventStreamEnded = true; eventFailures = 0; confirmPollSeq\+\+;\s+stopStaleFollow\(\);\s+break;\s+\}/);
  assert.match(script, /if \(!eventStreamEnded\) void attachEvents\(\);/);
});

test("a terminal replay leaves no running view or retry timers", async () => {
  const h = attachmentHarness(), elements = runningComposer(h.c);
  h.c.staleSession = "s";
  const pending = h.c.attachEvents();
  await h.settle();
  await h.frame('event: done\ndata: {}\n\n');
  await pending;
  assert.equal(h.c.eventStreamEnded, true);
  assert.equal(h.c.eventActive, false);
  assert.equal(h.c.recoveredView, null);
  assert.equal(h.c.staleSession, "");
  assert.equal(h.c.isTurnRunning(), false);
  assert.equal(elements.send.title, "send");
  assert.equal(h.timers.size, 0);
  assert.equal(h.loads(), 1);
});

for (const phone of [false, true]) {
  test(`confirmed live failure clears structured task activity and composer busy state (phone=${phone})`, () => {
    const h = streamHarness({ phone, phaseClock: true }), c = h.context;
    Object.assign(c, { eventActive: false, eventSession: "", staleSession: "s", turnRunning: true,
      turnRunningAt: c.Date.now(), RUNNING_SAMPLE_TTL_MS: 60000 });
    c.document.querySelectorAll = () => h.box.children.filter((card) =>
      card.dataset.taskStates || card.dataset.taskBusy || card.dataset.taskReset);
    const elements = runningComposer(c);
    elements.composer.setAttribute = (name, value) => { elements.composer[name] = value; };
    const active = structuredTask("task_failure", 5000);
    h.emit("tool", { id: "spawn", name: "task_spawn", phase: "finished", output: "raw spawn\n",
      task_report: { ...structuredSubagents([active]).task_report, operation: "task_spawn", snapshot: false } });
    h.emit("subagent_progress", structuredSubagents([active], "raw backend says completed, but active is true\n"));
    const card = h.box.children.find((node) => node.dataset.subagents === "true");
    assert.equal(card.dataset.taskBusy, "true");
    assert.throws(() => h.emit("error", { error: "confirmed turn failure" }), /confirmed turn failure/);
    h.fail();
    c.turns.clear();
    c.syncStopButton();
    assert.equal(c.isTurnRunning(), false);
    assert.equal(elements.send.title, "send");
    assert.equal(elements.composer["aria-busy"], "false");
    assert.notEqual(card.dataset.taskBusy, "true", "a terminal interruption must clear active task metadata");
    assert.ok(JSON.parse(card.dataset.taskStates || "[]").every(([, active]) => active !== true));
    assert.equal(c.eventStreamEnded, true);
    assert.equal(c.eventStreamFailed, true);
    assert.equal(h.bodies().find((text) => text.startsWith("raw backend")), "raw backend says completed, but active is true\n");
    assert.equal(h.box.children.find((node) => node.summary?.textContent.includes("task_spawn")).body.textContent, "raw spawn\n");
  });

  test(`confirmed replay error overrides active canonical task state after history refresh (phone=${phone})`, async () => {
    const h = attachmentHarness(false, { phone }), c = h.c;
    const oldCard = { dataset: { taskBusy: "true", taskStates: JSON.stringify([["task_replay_failure", true]]) } };
    c.document.querySelectorAll = () => [oldCard];
    const elements = runningComposer(c);
    elements.composer.setAttribute = (name, value) => { elements.composer[name] = value; };
    c.staleSession = "s";
    const pending = c.attachEvents();
    await h.settle();
    await h.frame("event: subagent_progress\ndata: " + JSON.stringify(structuredSubagents([structuredTask("task_replay_failure", 5000)]))
      + '\n\nevent: error\ndata: {"error":"confirmed interruption"}\n\nevent: done\ndata: {}\n\n');
    await pending;
    assert.equal(c.eventStreamFailed, true);
    assert.equal(c.eventStreamEnded, true);
    assert.equal(c.eventActive, false);
    assert.equal(c.staleSession, "");
    assert.equal(c.isTurnRunning(), false, "confirmed failure must take precedence over a stale canonical active report");
    assert.equal(elements.send.title, "send");
    assert.equal(elements.composer["aria-busy"], "false");
    assert.equal(h.timers.size, 0);
    assert.equal(h.loads(), 1);
    c.abortEvents();
    assert.equal(c.eventStreamFailed, false, "a new attachment/session lifecycle must clear the old failure marker");
  });
}

test("a parked active task keeps the composer stopped after output complete", () => {
  const h = attachmentHarness(), c = h.c, elements = runningComposer(c);
  Object.assign(c, {
    sessionId: "s", turns: new Map(), eventStreamEnded: true,
    eventActive: false, eventSession: "", staleSession: "",
    turnRunning: false, turnRunningAt: 0,
  });
  const card = { dataset: {
    taskStates: JSON.stringify([["task_demo", true]]), taskBusy: "true",
  } };
  c.document.querySelectorAll = () => [card];
  c.syncStopButton();
  assert.equal(elements.send.textContent, "⏹");
  assert.equal(elements.send.title, "stop this turn");

  const finalAnswer = { dataset: { taskReset: "true" } };
  c.document.querySelectorAll = () => [card, finalAnswer];
  c.syncStopButton();
  assert.equal(c.isTurnRunning(), false, "a final answer after the task must clear stale running metadata");
  assert.equal(elements.send.textContent, "➤");
  assert.equal(elements.send.title, "send");

  card.dataset.taskStates = JSON.stringify([["task_demo", false]]);
  card.dataset.taskBusy = "false";
  c.syncStopButton();
  assert.equal(elements.send.textContent, "➤");
  assert.equal(elements.send.title, "send");
});

test("a confirmed failure stays authoritative through an idle poll and resets for the next remote turn", async () => {
  const h = attachmentHarness(), c = h.c;
  const oldCard = { dataset: { taskBusy: "true", taskStates: JSON.stringify([["old_task", true]]) } };
  c.document.querySelectorAll = () => [oldCard];
  const elements = runningComposer(c);
  Object.assign(c, { confirmBusy: false, confirmShown: null, armHistoryFollow() {},
    api: async () => ({ json: async () => ({ running: false, pending: null }) }),
  });
  vm.runInContext(script.slice(script.indexOf("async function pollConfirm()"), script.indexOf("function isTurnRunning()")), c);
  const pending = c.attachEvents();
  await h.settle();
  await h.frame('event: error\ndata: {"error":"confirmed failure"}\n\nevent: done\ndata: {}\n\n');
  await pending;
  await c.pollConfirm();
  assert.equal(c.eventStreamEnded, false, "an idle poll permits the next remote turn to attach");
  assert.equal(c.eventStreamFailed, true, "idle history may still contain a superseded active task report");
  assert.equal(elements.send.title, "send");
  c.api = async () => ({ json: async () => ({ running: true, pending: null }) });
  await c.pollConfirm();
  await h.settle();
  assert.equal(c.eventStreamFailed, false);
  assert.equal(h.requests.length, 2);
  assert.equal(elements.send.title, "stop this turn");
  c.abortEvents();
  await h.settle();
});

test("retained canonical task history moves merged details when a later wait is appended", async () => {
  const spawned = structuredTask("task_retained", 5000);
  const spawnReport = { ...structuredSubagents([spawned]).task_report, operation: "task_spawn", snapshot: false };
  const messages = [
    { role: "assistant", content: "", tool_calls: [{ id: "spawn", function: { name: "task_spawn", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "spawn", content: "raw spawn\n", task_report: spawnReport },
  ];
  const h = retainedHistoryHarness(messages);
  await h.c.loadHistory();
  assert.equal(h.root.querySelectorAll(".task-output").length, 2);
  const done = { ...spawned, status: "completed", active: false, output: "  delivered exact bytes\n" };
  messages.push(
    { role: "assistant", content: "", tool_calls: [{ id: "wait", function: { name: "task_wait", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "wait", content: "raw wait\n", task_report: { ...spawnReport, operation: "task_wait", tasks: [done] } },
  );
  await h.c.loadHistory();
  const cards = h.root.querySelectorAll(".task-output").filter((panel) => panel.parentNode.dataset.subagents === "true");
  assert.equal(cards.length, 1);
  assert.equal(h.root.children.length, 2, "task display must retain one root child per canonical display row");
  assert.equal(h.root.children[0].children.filter((child) => child.dataset.subagents === "true").length, 0);
  const waitRow = h.root.children[1], [waitFold, subagents] = waitRow.children;
  assert.equal(waitFold.body.textContent, "raw wait\n");
  assert.equal(waitFold.nextElementSibling, subagents);
  assert.equal(subagents.dataset.subagents, "true");
  assert.equal(subagents.querySelectorAll(".task-output-body")[0].textContent, done.output);
  assert.equal(h.root.children[0].dataset.canonicalIndex, "1");
  assert.equal(waitRow.dataset.canonicalIndex, "3");
});

test("active replay remains stopped after assistant-ending history and an expired running sample", async () => {
  const h = retainedHistoryHarness([{ role: "assistant", content: "[Request failed this round; please retry or rephrase]" }]);
  const c = h.c, elements = runningComposer(c);
  c.document.addEventListener = () => {};
  vm.runInContext(script.slice(script.indexOf("function histEndsTurn("), script.indexOf("// The prompt list mirrors")), c);
  const pending = c.attachEvents();
  await h.settle();
  await h.frame('event: subagent_progress\ndata: worker: running\n\n');
  c.turnRunningAt = c.Date.now() - c.RUNNING_SAMPLE_TTL_MS - 1;
  await c.loadHistory();
  assert.equal(c.staleDone, true, "history still records its assistant-ending snapshot");
  assert.equal(c.eventActive, true);
  assert.equal(c.isTurnRunning(), true);
  assert.equal(elements.send.textContent, "⏹");
  assert.equal(elements.send.title, "stop this turn");
  assert.equal(elements.turnstatus.hidden, true, "attached replay needs no disconnected status");
  c.eventSession = "other"; c.staleSession = "";
  c.syncStopButton();
  assert.equal(elements.send.title, "send", "another session's replay must not block this composer");
  c.eventSession = "s";
  await h.frame('event: done\ndata: {}\n\n');
  await pending;
  assert.equal(elements.send.title, "send");
});

test("terminal replay ignores child reap lag until an idle poll acknowledges completion", async () => {
  const h = attachmentHarness(), c = h.c, elements = runningComposer(c);
  Object.assign(c, { confirmBusy: false, confirmShown: null, armHistoryFollow() {},
    api: async () => ({ json: async () => ({ running: true, pending: null }) }),
  });
  vm.runInContext(script.slice(script.indexOf("async function pollConfirm()"), script.indexOf("function isTurnRunning()")), c);
  const pending = c.attachEvents();
  await h.settle();
  await h.frame('event: done\ndata: {}\n\n');
  await pending;
  c.turnRunning = false; // Live-feed completion also clears the previous sample.
  await c.pollConfirm();
  assert.equal(c.turnRunning, true, "the child can still be registered after output completion");
  assert.equal(c.eventStreamEnded, true);
  assert.equal(elements.send.title, "send");
  assert.equal(h.requests.length, 1, "reap lag must not restart the finished replay");
  c.api = async () => ({ json: async () => ({ running: false, pending: null }) });
  await c.pollConfirm();
  assert.equal(c.eventStreamEnded, false);
  c.api = async () => ({ json: async () => ({ running: true, pending: null }) });
  await c.pollConfirm();
  await h.settle();
  assert.equal(h.requests.length, 2);
  assert.equal(elements.send.title, "stop this turn");
  await h.frame('event: done\ndata: {}\n\n');
  await h.settle();
});

for (const running of [true, null, false]) test(`assistant-ending stale history resolves only on an idle server poll (running=${running})`, async () => {
  const h = attachmentHarness(), c = h.c, elements = runningComposer(c);
  Object.assign(c, {
    staleSession: "s", staleDone: true, staleBusy: false, staleKey: "same",
    STALE_FOLLOW_TRIES: 300, historyLoadSeq: 0, historyLoadStates: new Map(),
    turnRunning: false, pollConfirm: async () => running, histKey: () => "same",
    api: async () => ({ headers: { get: () => "1" }, json: async () => [{ role: "assistant", content: "early answer" }] }),
  });
  c.document.visibilityState = "visible";
  vm.runInContext(script.slice(script.indexOf("async function reloadStale()"), script.indexOf("// A restart can end a turn")), c);
  await c.reloadStale();
  c.syncStopButton();
  assert.equal(c.staleSession, running === false ? "" : "s");
  assert.equal(elements.send.title, running === false ? "send" : "stop this turn");
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
  const stall = h.context.LIVE_STALL_MS;
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

test("a collapsed fold hides its task rows even though they set their own display", () => {
  // The task row list carries `display:flex` so rows stay readable while the
  // fold is open. An author display also beats the engine's collapsed-content
  // default, which left the subagent card looking uncollapsible.
  assert.match(html, /\.msg details:not\(\[open\]\) > \.task-output \{ display:none; \}/);
  // The turn rate footer lives inside the same fold on restored turns, so the
  // narrowing above must not be widened into a blanket child rule.
  assert.doesNotMatch(html, /details:not\(\[open\]\) > \*:not\(summary\)/);
});

test("the desktop settings row wraps instead of collapsing its inputs", () => {
  // Both inputs carry min-width:0 and every button refuses to shrink, so on a
  // narrow desktop window the token field used to collapse to a sliver and
  // could not be filled in; wrapping keeps the pair usable.
  assert.match(html, /#conn\.open \{ flex-wrap:wrap; \}/);
  assert.match(html, /#conn input \{ flex:1 1 260px; \}/);
  // Phones keep the single-line row: the override lives in the desktop query.
  const desktopStart = html.indexOf("@media (min-width:701px) {");
  assert.ok(desktopStart > 0, "desktop media query is present");
  assert.ok(html.indexOf("#conn.open { flex-wrap:wrap; }") > desktopStart,
    "the wrap rule sits inside the desktop media query");
});
