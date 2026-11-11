// Run with: node --test src/bin/ai/serve/app_questions.test.cjs
// Execute the question click path with the real history loader and user rows.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(process.env.SERVE_QUESTIONS_APP || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function between(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page functions: " + start);
  return script.slice(a, b);
}
const user = (content) => ({ role: "user", content });
const synthetic = (content) => ({ ...user(content), reasoning_content: "runtime-origin:synthetic-user:v1" });
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};

function harness({ messages, phone = true, requestHook } = {}) {
  const document = { visibilityState: "visible", activeElement: null };
  const scrolls = [], focuses = [], requests = [], notes = [], failures = [], follows = [];
  function node(tag = "div") {
    const classes = new Set(), listeners = new Map();
    let content = "";
    const el = {
      tag, dataset: {}, children: [], parentNode: null, attributes: {},
      scrollTop: 0, clientHeight: 100, offsetWidth: 1,
      get scrollHeight() { return this.children.length * 100; },
      get childElementCount() { return this.children.length; },
      get lastElementChild() { return this.children.at(-1) || null; },
      get previousElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) - 1] || null; },
      set className(value) { classes.clear(); value.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name)); },
      get className() { return [...classes].join(" "); },
      classList: {
        add: (...names) => names.forEach((name) => classes.add(name)),
        remove: (...names) => names.forEach((name) => classes.delete(name)),
        contains: (name) => classes.has(name),
        toggle(name, force) { if (force ?? !classes.has(name)) classes.add(name); else classes.delete(name); },
      },
      set textContent(value) { this.children.forEach((child) => { child.parentNode = null; }); this.children = []; content = value; },
      get textContent() { return content + this.children.map((child) => child.textContent).join(""); },
      set innerHTML(value) { this.textContent = value; },
      get innerHTML() { return content; },
      appendChild(child) { child.remove(); child.parentNode = this; this.children.push(child); return child; },
      append(...children) { children.forEach((child) => this.appendChild(child)); },
      replaceChild(next, old) { const index = this.children.indexOf(old); assert.ok(index >= 0); old.parentNode = null; next.parentNode = this; this.children[index] = next; },
      remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this); this.parentNode = null; },
      setAttribute(name, value) { this.attributes[name] = String(value); },
      matches(selector) { return selector.startsWith(".") && selector.slice(1).split(".").every((name) => classes.has(name)); },
      querySelectorAll(selector) {
        const result = [];
        const visit = (parent) => parent.children.forEach((child) => { if (child.matches(selector)) result.push(child); visit(child); });
        visit(this);
        return result;
      },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
      addEventListener(type, listener) { if (!listeners.has(type)) listeners.set(type, []); listeners.get(type).push(listener); },
      async dispatch(type, event = {}) {
        event.target ||= this;
        event.stopPropagation ||= () => { event.stopped = true; };
        if (this["on" + type]) await this["on" + type](event);
        for (const listener of listeners.get(type) || []) await listener(event);
        if (!event.stopped && this.parentNode) await this.parentNode.dispatch(type, event);
        return event;
      },
      focus(options) { document.activeElement = this; focuses.push({ node: this, options }); },
      scrollIntoView(options) { scrolls.push({ node: this, options }); },
    };
    return el;
  }
  document.createElement = node;
  const elements = Object.fromEntries(["msgs", "questionpanel", "questionlist", "questionbtn", "questioncount"].map((id) => [id, node()]));
  elements.questionpanel.appendChild(elements.questionlist);
  let current = messages;
  const response = (body, total = current.length) => ({ headers: { get: () => String(total) }, json: async () => body });
  const context = vm.createContext({
    document, $: (id) => elements[id], sessionId: "session-a", questionCache: null,
    window: { matchMedia: () => ({ matches: !phone }) },
    historySession: "", historyKeys: [], historyLoadSeq: 0, canonicalHistoryTotal: 0, lastMsgCount: 0,
    pendingUserInputs: new Map(), turns: new Map(), eventSession: "", recoveredView: null,
    outputFollowSuspended: false, navigator: { onLine: true },
    AbortController, setTimeout, clearTimeout,
    isPhoneLayout: () => phone, esc: (value) => String(value), requestAnimationFrame: (fn) => fn(),
    note: (text) => notes.push(text), console: { error: (...args) => failures.push(args) },
    api: async (request) => {
      requests.push(request);
      if (requestHook) await requestHook(request);
      assert.ok(request.startsWith("/sessions/session-a/history"), request);
      if (request.endsWith("?meta=roles")) return response(current.map((message) => context.isRuntimeSyntheticUser(message) ? "runtime_synthetic_user" : message.role));
      const count = request.match(/\?role=user&count=(\d+)$/);
      if (count) {
        let ordinal = 0;
        return response(current.filter((message) => message.role === "user" && !context.isRuntimeSyntheticUser(message))
          .map((message) => ({ ...message, user_ordinal: ++ordinal })).slice(-Number(count[1])));
      }
      const limit = request.match(/\?limit=(\d+)$/);
      return response(limit ? current.slice(-Number(limit[1])) : current.slice());
    },
    shouldFollowOutput: () => !context.outputFollowSuspended,
    followOutput: (follow) => { follows.push(follow); if (follow) elements.msgs.scrollTop = elements.msgs.scrollHeight; },
    syncStopButton() {}, renderSessionBar() {}, revokePreviews() {}, attachLiveBox() {}, abortEvents() {},
    histKey: (rows) => JSON.stringify(rows), histEndsTurn: () => false,
    historyLoadState: () => ({ loading: 0, key: "", failure: null }),
    stopHistoryRetry() {}, scheduleHistoryRetry() {},
    recordHistoryFailure: (...args) => failures.push(args),
    createTurnRateLine: () => () => {},
  });
  vm.runInContext(between("function textOf(", "const PHONE_LAYOUT_QUERY"), context);
  vm.runInContext(between("function closeQuestionPanel(", "// BEGIN session switch"), context);
  context.updateQuestionButton = (count) => { if (count !== undefined) elements.questioncount.textContent = String(count); };
  vm.runInContext(between("function questionRowsOf(", "// Assistant replies render"), context);
  vm.runInContext(between("function historyRowsOf(", "// A session the server reports as running"), context);
  vm.runInContext(between("// BEGIN history retry", "const HISTORY_RETRY_DELAYS_MS"), context);
  vm.runInContext(between("async function loadHistory(", "// Scrolling to the top of the transcript"), context);
  vm.runInContext(between("function attachUserRewind(", "function findUserHistoryIndex("), context);
  context.addMsgMd = (text, parent) => context.addMsg("assistant", text, "asst md", parent);
  const links = () => elements.questionlist.children.filter((child) => child.className === "question-link");
  return { context, document, elements, links, requests, notes, failures, scrolls, focuses, follows,
    setMessages(value) { current = value; },
    async open() { await context.loadHistory(); await context.openQuestionPanel(); },
    async tap(index, child = true) {
      const target = child ? links()[index].children[1] : links()[index];
      await target.dispatch("pointerdown", { pointerType: "touch" });
      await target.dispatch("pointerup", { pointerType: "touch" });
      return target.dispatch("click");
    },
  };
}

for (const phone of [true, false]) test((phone ? "phone" : "desktop") + " question tap resolves multiline text and focuses the canonical user bubble", async () => {
  const h = harness({ phone, messages: [user("  first\nsecond  part\tend  ")] });
  await h.open();
  assert.equal(h.links()[0].children[1].textContent, "first second part end");
  let outsideClicks = 0;
  h.elements.questionpanel.addEventListener("click", () => outsideClicks++);
  const click = await h.tap(0);
  assert.equal(h.elements.questionpanel.classList.contains("open"), false);
  assert.equal(h.elements.questionbtn.attributes["aria-expanded"], "false");
  assert.equal(h.scrolls.length, 1);
  assert.equal(h.scrolls[0].node.dataset.historyIndex, "0");
  assert.equal(h.scrolls[0].node.dataset.question, "first\nsecond  part\tend");
  assert.equal(h.focuses[0].node, h.scrolls[0].node);
  assert.equal(h.focuses[0].options.preventScroll, true);
  assert.equal(h.context.outputFollowSuspended, true);
  assert.equal(click.stopped, true);
  assert.equal(outsideClicks, 0);
  assert.deepEqual(h.failures, []);
});

test("question tap loads a target outside the phone history window using the real loader", async () => {
  const h = harness({ messages: [user("old prompt"), ...Array.from({ length: 250 }, (_, i) => ({ role: "assistant", content: "reply " + i }))] });
  await h.open();
  assert.equal(h.elements.msgs.querySelectorAll(".msg.user").length, 0);
  await h.tap(0);
  assert.equal(h.scrolls.length, 1);
  assert.equal(h.scrolls[0].node.dataset.historyIndex, "0");
  assert.equal(h.context.historySession, "session-a");
  assert.equal(h.context.canonicalHistoryTotal, 251);
  assert.equal(h.context.lastMsgCount, 251);
  assert.equal(h.requests.filter((request) => !request.includes("role=user") && !request.includes("meta=roles")).length, 2);
  assert.equal(h.follows.at(-1), false, "loading an older target must not resume following live output");
  assert.deepEqual(h.failures, []);
});

for (const messages of [[], [user("prefetched row")]]) test("explicit prefetched history stays a data load: " + messages.length, async () => {
  const h = harness({ messages: [user("server row")] });
  await h.context.loadHistory({ sessionId: "session-a", messages, histTotal: messages.length ? 3 : 0, showAll: true });
  assert.deepEqual(h.requests, []);
  assert.equal(h.context.historySession, "session-a");
  const bubbles = h.elements.msgs.querySelectorAll(".msg.user");
  assert.equal(bubbles.length, messages.length);
  if (messages.length) {
    assert.equal(bubbles[0].dataset.question, "prefetched row");
    assert.equal(bubbles[0].dataset.historyIndex, "2");
  }
  assert.deepEqual(h.failures, []);
});

for (const sessionId of ["session-a", ""]) test("a cancelled history load leaves data and loading state untouched: " + (sessionId || "no session"), async () => {
  const h = harness({ messages: [user("server row")] });
  h.context.sessionId = sessionId;
  const rows = h.elements.msgs.children;
  await h.context.loadHistory({ sessionId, messages: [user("cancelled row")], histTotal: 1, showAll: true }, () => false);
  assert.deepEqual(h.requests, []);
  assert.equal(h.context.historyLoadSeq, 0);
  assert.equal(h.context.historySession, "");
  assert.equal(h.context.canonicalHistoryTotal, 0);
  assert.equal(h.elements.msgs.children, rows);
  assert.equal(h.elements.msgs._histWin, undefined);
  assert.deepEqual(h.failures, []);
});

test("identical questions use their real user ordinal, excluding empty and synthetic display rows", async () => {
  const h = harness({ messages: [user("repeat"), synthetic("repeat"), { role: "tool", content: "" }, user(""), user("repeat")] });
  await h.open();
  assert.deepEqual(h.links().map((link) => link.children[0].textContent), ["1", "3"]);
  await h.tap(0);
  assert.equal(h.scrolls[0].node.dataset.historyIndex, "0");
  await h.context.openQuestionPanel();
  await h.tap(1, false);
  assert.equal(h.scrolls[1].node.dataset.historyIndex, "4");
  assert.deepEqual(h.failures, []);
});

test("image placeholders participate in the canonical target identity", async () => {
  const h = harness({ messages: [user([{ type: "text", text: "look\n here " }, { type: "image_url", image_url: { url: "(redacted)" } }])] });
  await h.open();
  assert.equal(h.links()[0].children[1].textContent, "look here");
  await h.tap(0);
  assert.equal(h.scrolls.length, 1);
  assert.equal(h.scrolls[0].node.dataset.question, "look\n here \n[image]");
});

for (const replacement of [[], [user("different question")]]) test("a stale question fails safely after history changes: " + replacement.length, async () => {
  const h = harness({ messages: [user("original question")] });
  await h.open();
  h.setMessages(replacement);
  await h.context.loadHistory();
  await h.tap(0);
  assert.equal(h.scrolls.length, 0);
  assert.equal(h.focuses.length, 0);
  assert.deepEqual(h.failures, []);
});

test("an old card cannot navigate another session", async () => {
  const h = harness({ messages: [user("shared text")] });
  await h.open();
  const before = h.requests.length;
  h.context.sessionId = "session-b";
  await h.tap(0);
  assert.equal(h.requests.length, before);
  assert.equal(h.scrolls.length, 0);
});

for (const blockedRequest of ["?meta=roles", "/history"]) test("switching sessions while navigation awaits " + blockedRequest + " cancels the jump", async () => {
  const wait = deferred();
  let block = false;
  const h = harness({ messages: [user("old prompt"), ...Array.from({ length: 250 }, () => ({ role: "assistant", content: "" }))],
    requestHook: async (request) => { if (block && request.endsWith(blockedRequest)) { block = false; await wait.promise; } },
  });
  await h.open();
  block = true;
  const mountedWindow = h.elements.msgs._histWin;
  const tap = h.tap(0).then(() => null, (error) => error);
  await new Promise((resolve) => setImmediate(resolve));
  h.context.sessionId = "session-b";
  wait.resolve();
  assert.equal(await tap, null);
  assert.equal(h.scrolls.length, 0);
  assert.equal(h.focuses.length, 0);
  assert.equal(h.elements.msgs._histWin, mountedWindow);
  assert.deepEqual(h.failures, []);
});

test("the latest tap wins when earlier role metadata arrives late", async () => {
  const wait = deferred();
  let block = false;
  const h = harness({ messages: [user("first"), user("second")], requestHook: async (request) => {
    if (block && request.endsWith("?meta=roles")) { block = false; await wait.promise; }
  } });
  await h.open();
  const first = h.links()[0], second = h.links()[1];
  block = true;
  const firstTap = first.dispatch("click");
  await new Promise((resolve) => setImmediate(resolve));
  await second.dispatch("click");
  wait.resolve();
  await firstTap;
  assert.deepEqual(h.scrolls.map(({ node }) => node.dataset.historyIndex), ["1"]);
  assert.deepEqual(h.failures, []);
});

for (const fail of [false, true]) test("reopening the overlay and tapping a mounted question cancels the old full-history " + (fail ? "failure" : "commit"), async () => {
  const wait = deferred(), requested = deferred();
  let block = false;
  const h = harness({
    messages: [user("older question"), ...Array.from({ length: 250 }, (_, i) => ({ role: "assistant", content: "reply " + i })), user("newer question")],
    requestHook: async (request) => {
      if (block && request.endsWith("/history")) {
        block = false;
        requested.resolve();
        await wait.promise;
        if (fail) throw new Error("late old history failure");
      }
    },
  });
  await h.open();
  const root = h.elements.msgs;
  assert.deepEqual(root.querySelectorAll(".msg.user").map((bubble) => bubble.dataset.question), ["newer question"]);
  block = true;
  const oldTap = h.tap(0);
  await requested.promise;
  await h.context.openQuestionPanel();
  await h.tap(1);
  const latest = h.scrolls.at(-1).node;
  assert.equal(latest.dataset.historyIndex, "251");
  assert.equal(latest.classList.contains("question-target"), true);
  root.scrollTop = 123;
  const mounted = root.children.slice(), keys = h.context.historyKeys;
  const mountedWindow = root._histWin, windowSession = root._histWinSession, shown = h.context.lastMsgCount, follows = h.follows.length;
  wait.resolve();
  await oldTap;
  assert.ok(root.querySelectorAll(".msg.user").includes(latest), "the latest target must remain the mounted canonical bubble");
  assert.equal(latest.classList.contains("question-target"), true);
  assert.equal(h.document.activeElement, latest);
  assert.deepEqual(root.children, mounted, "an obsolete response must not replace transcript nodes");
  assert.equal(root._histWin, mountedWindow, "an obsolete response must not expand the pager");
  assert.equal(root._histWinSession, windowSession);
  assert.equal(h.context.lastMsgCount, shown);
  assert.equal(h.context.historyKeys, keys);
  assert.equal(root.scrollTop, 123);
  assert.equal(h.follows.length, follows);
  assert.deepEqual(h.scrolls.map(({ node }) => node.dataset.historyIndex), ["251"]);
  assert.deepEqual(h.failures, []);
  assert.deepEqual(h.notes, []);
});

test("leaving and returning to the session invalidates the pending question jump", async () => {
  const wait = deferred();
  let block = false;
  const h = harness({ messages: [user("question")], requestHook: async (request) => {
    if (block && request.endsWith("?meta=roles")) { block = false; await wait.promise; }
  } });
  await h.open();
  block = true;
  const tap = h.tap(0);
  await new Promise((resolve) => setImmediate(resolve));
  h.context.sessionId = "session-b";
  h.context.resetQuestionNavigator();
  h.context.sessionId = "session-a";
  wait.resolve();
  await tap;
  assert.deepEqual(h.scrolls, []);
});

test("a failed navigation request is reported without an unhandled rejection or a wrong jump", async () => {
  let fail = false;
  const h = harness({ messages: [user("question")], requestHook: async () => { if (fail) throw new Error("offline"); } });
  await h.open();
  fail = true;
  await h.tap(0);
  assert.equal(h.scrolls.length, 0);
  assert.match(h.notes.at(-1), /question.*offline/);
  assert.deepEqual(h.failures, []);
});
