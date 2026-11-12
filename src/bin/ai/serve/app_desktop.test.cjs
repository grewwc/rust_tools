// Run with: node --test src/bin/ai/serve/app_desktop.test.cjs
// Exercise the actual sidebar, draft lifecycle and send functions in a small
// DOM. Network responses are controlled; page code owns navigation and payloads.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const baseCss = css.slice(0, css.indexOf("@media (min-width:701px)"));
const desktopCss = css.slice(css.indexOf("@media (min-width:701px)"), css.indexOf("/* Keep narrow desktop"));
const phoneStart = css.indexOf("@media (max-width:700px)");
const phoneCss = css.slice(phoneStart, css.indexOf("@media (prefers-reduced-motion:reduce)", phoneStart));
const plain = (value) => JSON.parse(JSON.stringify(value));
function cssRule(source, selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rules = [...source.matchAll(new RegExp("(?:^|\\n)\\s*" + escaped + "\\s*\\{([^}]*)\\}", "g"))];
  return Object.fromEntries(rules.flatMap((rule) => rule[1].split(";").filter((entry) => entry.includes(":"))
    .map((entry) => {
      const colon = entry.indexOf(":");
      return [entry.slice(0, colon).trim(), entry.slice(colon + 1).trim()];
    })));
}
function responsiveRule(selector, phone = false) {
  return { ...cssRule(baseCss, selector), ...cssRule(phone ? phoneCss : desktopCss, selector) };
}
// This reads transcript ancestry from the source markup; it is not a layout
// engine. The live DOM fixture below separately executes the real send path.
function shellPathTo(id) {
  const source = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, "").replace(/<!--[\s\S]*?-->/g, "");
  const stack = [], voidTags = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);
  for (const match of source.matchAll(/<(\/?)([a-z][\w:-]*)([^>]*)>/gi)) {
    const tag = match[2].toLowerCase(), attrs = match[3];
    if (match[1]) {
      const index = stack.findLastIndex((item) => item.tag === tag);
      if (index >= 0) stack.length = index;
      continue;
    }
    const item = {
      tag, id: /\bid="([^"]*)"/.exec(attrs)?.[1] || "",
      className: /\bclass="([^"]*)"/.exec(attrs)?.[1] || "",
      hidden: /(?:^|\s)hidden(?:\s|=|$)/.test(attrs),
      style: /\bstyle="([^"]*)"/.exec(attrs)?.[1] || "",
    };
    if (item.id === id) return [...stack, item];
    if (!voidTags.has(tag) && !/\/\s*$/.test(attrs)) stack.push(item);
  }
  assert.fail("missing source element " + id);
}
function between(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page functions: " + start);
  return script.slice(a, b);
}
function marked(name) {
  return between("// BEGIN " + name, "// END " + name);
}
const response = (body, status = 200, headers = {}) => ({
  ok: status === 200, status, statusText: "Unavailable",
  headers: { get: (name) => headers[name] ?? null }, json: async () => plain(body),
});
const sessions = (marked = false) => [
  { id: "open-session", summary: "Open session", marked: false, size_bytes: 1024, cwd: "/srv/open-session" },
  { id: "target-session", summary: "Target session", marked, size_bytes: 2048, cwd: "/srv/target-session" },
];

function harness({ phone = false, sid = "open-session", marked: isMarked = false, confirmDelete = true, deleteError = "", forkError = "" } = {}) {
  const document = { activeElement: null };
  function node(tag = "div") {
    let text = "", markup = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, style: {},
      hidden: false, disabled: false, value: "", title: "", scrollTop: 0,
      get lastElementChild() { return el.children.at(-1) || null; },
      get nextElementSibling() { return el.parentNode?.children[el.parentNode.children.indexOf(el) + 1] || null; },
      get className() { return [...classes].join(" "); },
      set className(value) { classes.clear(); value.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name)); },
      classList: {
        add: (...names) => names.forEach((name) => classes.add(name)),
        remove: (...names) => names.forEach((name) => classes.delete(name)),
        contains: (name) => classes.has(name),
        toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
      },
      get textContent() { return text + el.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        markup = "";
        el.children.forEach((child) => { child.parentNode = null; });
        el.children = [];
      },
      get innerHTML() { return markup; },
      set innerHTML(value) {
        markup = String(value);
        text = markup.replace(/<[^>]*>/g, "");
        el.children.forEach((child) => { child.parentNode = null; });
        el.children = [];
      },
      addEventListener(type, listener) { el["on" + type] = listener; },
      setAttribute(key, value) { el.attrs[key] = String(value); },
      getAttribute(key) { return el.attrs[key] ?? null; },
      appendChild(child) { child.remove(); child.parentNode = el; el.children.push(child); return child; },
      insertBefore(child, before) {
        if (child === before) return child;
        child.remove();
        const index = before ? el.children.indexOf(before) : el.children.length;
        assert.ok(index >= 0, "the insertion anchor must belong to its parent");
        child.parentNode = el;
        el.children.splice(index, 0, child);
        return child;
      },
      replaceChild(next, current) {
        const index = el.children.indexOf(current);
        assert.ok(index >= 0, "replace an existing transcript child");
        next.remove();
        el.children[index] = next;
        current.parentNode = null;
        next.parentNode = el;
        return current;
      },
      remove() {
        if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((child) => child !== el);
        el.parentNode = null;
      },
      matches(selector) {
        return selector.split(",").some((part) => part.trim().startsWith(".") ? classes.has(part.trim().slice(1)) : tag === part.trim());
      },
      closest(selector) {
        for (let current = el; current; current = current.parentNode) if (current.matches(selector)) return current;
        return null;
      },
      querySelectorAll(selector) {
        const found = [];
        const walk = (parent) => parent.children.forEach((child) => { if (child.matches(selector)) found.push(child); walk(child); });
        walk(el);
        return found;
      },
      querySelector(selector) { return el.querySelectorAll(selector)[0] || null; },
      focus() { if (!el.disabled) document.activeElement = el; },
    };
    return el;
  }
  document.createElement = node;
  const elements = {};
  for (const [id, tag] of [
    ["sesslist", "ul"], ["sesscount", "div"], ["sesstitle", "div"], ["sesscwd", "span"], ["sessmeta", "div"],
    ["prompt", "textarea"], ["msgs", "div"], ["promptlist", "ol"], ["promptcount", "span"], ["newbtn", "button"], ["fork", "button"],
    ["refresh", "button"], ["del", "button"], ["omodel", "select"], ["oagent", "select"], ["oeffort", "select"],
    ["composer", "div"], ["img", "button"], ["sidenote", "button"], ["send", "button"], ["turnstatus", "div"],
    ["sidequeue", "div"], ["sidequeuecount", "span"], ["sidequeuelist", "ul"],
  ]) elements[id] = node(tag);
  for (const id of ["img", "prompt", "sidenote", "send"]) elements.composer.appendChild(elements[id]);
  elements.prompt.value = "draft in A";
  for (const match of html.matchAll(/<(div|section|button)[^>]*id="(sessionaction[^"]*)"[^>]*>([^<]*)/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    el.textContent = match[3];
  }
  for (const [id, tag] of [
    ["sessdel", "div"], ["sessdelscrim", "div"], ["sessdelpanel", "section"],
    ["sessdeltitle", "div"], ["sessdeltext", "div"], ["sessdelbtns", "div"],
    ["sessdelcancel", "button"], ["sessdelconfirm", "button"],
  ]) elements[id] = node(tag);
  elements.sessdel.hidden = true;
  for (const id of ["sessionactionstitle", "sessionactionmark", "sessionactionfork", "sessionactiondelete", "sessionactioncancel"]) {
    elements.sessionactionspanel.appendChild(elements[id]);
  }
  elements.sessionactions.appendChild(elements.sessionactionsscrim);
  elements.sessionactions.appendChild(elements.sessionactionspanel);
  document.querySelectorAll = (selector) => {
    if (selector === "#msgs .turn-interrupted") return elements.msgs.querySelectorAll(".turn-interrupted");
    assert.equal(selector, "#msgs [data-task-states], #msgs [data-task-busy], #msgs [data-task-reset]");
    return [];
  };
  const serverItems = plain(sessions(isMarked)), stored = new Map(), requests = [], confirmations = [], messages = [], notes = [], drawers = [], sideNotes = [];
  const media = { matches: phone, listeners: [], addEventListener: (_, listener) => media.listeners.push(listener) };
  const timers = new Map();
  let nextTimer = 1, now = 1000, historyLoads = 0, configLoads = 0, grows = 0;
  const context = vm.createContext({
    document, $: (id) => elements[id], window: { matchMedia: () => media }, promptHistory: null,
    store: { base: "https://serve.test", token: "test-token" },
    sessionId: sid, lastItems: plain(serverItems), pendingNew: null, pendingCwd: "", lastMsgCount: 5, PAGE_ID: "test-page",
    localStorage: { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    Date: { now: () => now }, TextDecoder,
    setTimeout: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, at: now + ms }); return id; },
    clearTimeout: (id) => timers.delete(id), setInterval: () => 1, clearInterval() {},
    fetch: async (url, options = {}) => {
      const request = {
        path: new URL(url).pathname, method: options.method || "GET",
        body: options.body ? JSON.parse(options.body) : null,
      };
      requests.push(request);
      if (request.path.endsWith("/side-notes")) {
        if (request.method === "GET") return response({ notes: sideNotes });
        return response({});
      }
      if (request.path.endsWith("/turns/stream")) {
        let read = false;
        return { ...response({}), body: { getReader: () => ({
          read: async () => read ? { done: true } : (read = true, { done: false, value: Buffer.from("event: done\n\n") }),
          cancel() {},
        }) } };
      }
      if (request.path.endsWith("/fork")) {
        if (forkError) return response({ error: forkError }, 503);
        serverItems.push({ id: "forked-session", summary: "Forked session", marked: false, size_bytes: 1024 });
        return response({ id: "forked-session" });
      }
      if (request.path.endsWith("/mark")) {
        const target = serverItems.find((item) => request.path === "/sessions/" + item.id + "/mark");
        assert.ok(target, "unexpected mark target");
        target.marked = request.body.marked;
        return response({ marked: target.marked });
      }
      if (request.method === "DELETE") {
        if (deleteError) return response({ error: deleteError }, 503);
        const index = serverItems.findIndex((item) => request.path === "/sessions/" + item.id);
        assert.ok(index >= 0, "unexpected delete target");
        serverItems.splice(index, 1);
        return response({});
      }
      assert.equal(request.path, "/sessions");
      if (request.method === "POST") return response({ id: "fresh-session" });
      return response(serverItems, 200, {
        "x-sessions-total": String(serverItems.length),
        "x-sessions-page-size": String(serverItems.length),
      });
    },
    confirm: (text) => { confirmations.push(text); return confirmDelete; },
    addMsg: (role, text) => { messages.push({ role, text }); return node(); }, note: (text) => notes.push(text),
    loadHistory: async () => { historyLoads++; }, loadSessionConfig: async () => { configLoads++; },
    setDrawer: (open) => drawers.push(open), growPrompt: () => { grows++; },
    promptComposing: false, traceMark() {}, composerInUse: () => false,
    syncStopButton() {}, resetQuestionNavigator() {}, restoreSkillPick() {}, restoreTeamPick() {},
    stopStaleFollow() {}, hideConfirm() {}, pollConfirm() {}, resetPicks() {}, abortEvents() {},
    carrySkillPick() {}, carryTeamPick() {}, persistPicks() {}, ensureTeamConfig: async () => true,
    runCdCommand: async () => false, validateSkillInput: () => true,
    captureSkillPick: () => ({ skills: [] }), consumeSkillPick() {},
    sendBusy: false, pendingImages: [], turns: new Map(), MAX_LIVE_TURNS: 4,
    historySession: sid, canonicalHistoryTotal: 0, pendingUserInputs: new Map(),
    questionCache: null, runEpoch: 0, turnRunning: false, confirmPollSeq: 0,
    turnRunningAt: 0, RUNNING_SAMPLE_TTL_MS: 60000,
    turnRunningPid: 0, endedTurnPid: 0,
    eventActive: false, eventSession: "", eventStreamEnded: false, eventStreamFailed: false, staleSession: "", staleDone: false,
    lastHistoryEndsUnanswered: false, LIVE_STALL_MS: 60000, STREAM_SILENCE_MS: 60000,
    addLiveUserBubble: () => { const row = node(), bubble = node(); row.appendChild(bubble); elements.msgs.appendChild(row); return bubble; },
    addMsgMd: (text, parent) => {
      const bubble = node(); bubble.className = "msg asst md";
      bubble.dataset.role = "assistant"; bubble.textContent = text;
      return parent.appendChild(bubble);
    },
    createLiveActivity: () => ({ label: () => "Connecting", setPhase() {}, finish() {}, anchor() {} }),
    scrollToLatest() {}, shouldFollowOutput: () => false, followOutput() {}, decorateAndLoad() {},
    formatElapsed: () => "0s", replyPaintDelay: () => 0, renderMarkdown: (text) => text,
    loadInfo: async () => {},
  });
  vm.runInContext(between("function headers(", "const esc ="), context);
  vm.runInContext(between("function sessionTitle(s)", "// Phone drawer floats"), context);
  vm.runInContext(between("function setPromptValue(", "// True while the composer"), context);
  vm.runInContext(between("const DRAFT_PREFIX =", "let questionCache ="), context);
  vm.runInContext(between("const composerDrafts = new Map();", "function closeQuestionPanel("), context);
  vm.runInContext(marked("history retry"), context);
  vm.runInContext(marked("session switch"), context);
  vm.runInContext(between("function renderSessionBar()", "// History is append-only"), context);
  vm.runInContext(marked("composer session adoption"), context);
  vm.runInContext(marked("composer git command"), context);
  vm.runInContext(marked("turn session preparation"), context);
  vm.runInContext(between("function isTurnRunning()", "// The one send/stop button"), context);
  vm.runInContext(between("let switchPollTimer = null;", "// Rebuild a compact progress card"), context);
  vm.runInContext(between("function parseBlock(raw)", "// \u0060/cd\u0060 changes"), context);
  vm.runInContext(between("const RATE_STORAGE =", "function addFold("), context);
  vm.runInContext(between("function setAnswerRate(", "// Only completed answer bubbles"), context);
  vm.runInContext(marked("composer send"), context);
  vm.runInContext(between('$("newbtn").onclick =', '$("img").onclick ='), context);
  context.wireSessionActions();
  context.renderSessionList(context.lastItems);
  context.renderSessionBar();
  const drafts = vm.runInContext("composerDrafts", context);
  function dispatch(target, type, extra = {}) {
    const event = { target, detail: type === "click" ? 1 : 0, ...extra, stopped: false, prevented: false,
      stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; } };
    const pending = [];
    if (!(type === "click" && target.disabled)) for (let current = target; current; current = current.parentNode) {
      const result = current["on" + type]?.(event);
      if (result && typeof result.then === "function") pending.push(result);
      if (event.stopped) break;
    }
    return { event, done: Promise.all(pending) };
  }
  const row = (id = "target-session") => elements.sesslist.children.find((li) => li.querySelector("small")?.textContent === id.slice(0, 8));
  const title = (id) => row(id).querySelector("span");
  const open = (id) => { const target = title(id); target.focus(); return dispatch(target, "keydown", { key: "F10", shiftKey: true }); };
  const select = (id) => dispatch(phone ? title(id) : row(id), "click").done;
  const advance = (ms) => {
    now += ms;
    for (const [id, timer] of [...timers]) if (timer.at <= now) { timers.delete(id); timer.fn(); }
  };
  return { context, document, elements, drafts, stored, requests, confirmations, messages, notes, drawers, sideNotes, media,
    dispatch, row, title, open, select, advance, historyLoads: () => historyLoads, configLoads: () => configLoads, grows: () => grows };
}

test("desktop double-click opens the shared mark/fork/delete sheet without marking", async () => {
  const h = harness(), title = h.title();
  const click = h.dispatch(title, "click", { detail: 1 });
  assert.equal(h.context.sessionId, "target-session");
  const second = h.dispatch(title, "click", { detail: 2 });
  assert.equal(second.event.stopped, true);
  const double = h.dispatch(title, "dblclick", { detail: 2 });
  assert.equal(double.event.prevented, true);
  assert.equal(double.event.stopped, true);
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(h.elements.sessionactionstitle.textContent, "Target session");
  assert.equal(h.elements.sessionactionmark.textContent, "标记会话");
  assert.equal(h.elements.sessionactionfork.hidden, false);
  assert.deepEqual(h.elements.sessionactionspanel.querySelectorAll("button").filter((button) => !button.hidden).map((button) => button.textContent),
    ["标记会话", "fork", "删除会话", "取消"]);
  assert.equal(h.document.activeElement, h.elements.sessionactionmark);
  assert.equal(h.historyLoads(), 1, "the first click selects, the second must not select twice");
  h.dispatch(h.document.activeElement, "keydown", { key: "Escape" });
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.document.activeElement, title);
  await Promise.all([click.done, second.done, double.done]);
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.configLoads(), 1);
  assert.deepEqual(h.requests, []);
});

test("desktop right-click opens the shared sheet without switching sessions", async () => {
  const h = harness(), row = h.row();
  const menu = h.dispatch(row, "contextmenu", {});
  assert.equal(menu.event.prevented, true);
  assert.equal(menu.event.stopped, true);
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(h.elements.sessionactionstitle.textContent, "Target session");
  assert.equal(h.elements.sessionactionfork.hidden, false);
  assert.deepEqual(h.elements.sessionactionspanel.querySelectorAll("button").filter((button) => !button.hidden).map((button) => button.textContent),
    ["标记会话", "fork", "删除会话", "取消"]);
  await menu.done;
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.historyLoads(), 0);
  assert.deepEqual(h.requests, []);
});

test("desktop single title click selects the session and leaves the mark alone", async () => {
  for (const marked of [false, true]) {
    const h = harness({ marked });
    const click = h.dispatch(h.title(), "click");
    await click.done;
    assert.equal(click.event.stopped, true);
    assert.equal(h.context.sessionId, "target-session");
    assert.equal(h.row().classList.contains("cur"), true);
    assert.equal(h.title().getAttribute("aria-current"), "true");
    assert.equal(h.elements.sesstitle.textContent, marked ? "★ Target session" : "Target session");
    assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(marked));
    assert.equal(h.elements.sessionactions.hidden, true);
    assert.deepEqual(h.requests, []);
  }
});

test("desktop single row body click selects the session", async () => {
  const h = harness();
  const click = h.dispatch(h.row(), "click");
  await click.done;
  assert.equal(click.event.stopped, false);
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.row().classList.contains("cur"), true);
  assert.equal(h.title().getAttribute("aria-current"), "true");
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.historyLoads(), 1);
  assert.equal(h.configLoads(), 1);
  assert.deepEqual(h.drawers, []);
  assert.deepEqual(h.requests, []);
});

for (const key of ["Enter", " "]) test("desktop focused title " + JSON.stringify(key) + " opens the session without marking", async () => {
  for (const marked of [false, true]) {
    const h = harness({ marked }), title = h.title();
    title.focus();
    const press = h.dispatch(title, "keydown", { key });
    await press.done;
    assert.equal(press.event.prevented, true);
    assert.equal(press.event.stopped, true);
    assert.equal(title.getAttribute("role"), "button");
    assert.equal(title.getAttribute("tabindex"), "0");
    assert.equal(h.context.sessionId, "target-session");
    assert.equal(h.title().getAttribute("aria-current"), "true");
    assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(marked));
    assert.equal(h.historyLoads(), 1);
    assert.deepEqual(h.requests.filter((request) => request.path.endsWith("/mark")), []);
    assert.equal(h.elements.sessionactions.hidden, true);
    h.dispatch(title, "dblclick", { detail: 0 });
    assert.equal(h.elements.sessionactions.hidden, true, "a keyboard activation cannot become a double-click action");
  }
});

for (const extra of [{ key: "F10", shiftKey: true }, { key: "ContextMenu" }]) test("desktop title keyboard menu " + extra.key + " opens and Escape restores focus", () => {
  const h = harness({ marked: true }), title = h.title();
  title.focus();
  const press = h.dispatch(title, "keydown", extra);
  assert.equal(press.event.prevented, true);
  assert.equal(press.event.stopped, true);
  assert.equal(title.getAttribute("aria-haspopup"), "dialog");
  assert.equal(title.getAttribute("aria-controls"), "sessionactionspanel");
  assert.equal(title.getAttribute("aria-keyshortcuts"), "Shift+F10");
  assert.equal(h.elements.sessionactionmark.textContent, "取消标记");
  assert.equal(h.context.sessionId, "open-session");
  const escape = h.dispatch(h.document.activeElement, "keydown", { key: "Escape" });
  assert.equal(escape.event.prevented, true);
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.document.activeElement, title);
  assert.deepEqual(h.requests, []);
});

test("desktop Tab includes Fork, wraps in both directions, and Cancel or scrim returns focus", () => {
  for (const cancel of ["sessionactioncancel", "sessionactionsscrim"]) {
    const h = harness(), title = h.title();
    h.open();
    for (const id of ["sessionactionfork", "sessionactiondelete", "sessionactioncancel", "sessionactionmark"]) {
      h.dispatch(h.document.activeElement, "keydown", { key: "Tab" });
      assert.equal(h.document.activeElement, h.elements[id]);
    }
    h.dispatch(h.document.activeElement, "keydown", { key: "Tab", shiftKey: true });
    assert.equal(h.document.activeElement, h.elements.sessionactioncancel);
    h.dispatch(h.elements[cancel], "click");
    assert.equal(h.elements.sessionactions.hidden, true);
    assert.equal(h.document.activeElement, title);
  }
});

for (const initiallyMarked of [false, true]) test("desktop sheet and star " + (initiallyMarked ? "unmark" : "mark") + " without switching sessions", async () => {
  for (const sheet of [true, false]) {
    const h = harness({ marked: initiallyMarked });
    if (sheet) h.open();
    await h.dispatch(sheet ? h.elements.sessionactionmark : h.row().querySelector(".mk"), "click").done;
    assert.deepEqual(h.requests.map((request) => ({ path: request.path, method: request.method, body: request.body })), [
      { path: "/sessions/target-session/mark", method: "POST", body: { marked: !initiallyMarked } },
    ]);
    assert.equal(h.context.sessionId, "open-session");
    assert.equal(h.elements.prompt.value, "draft in A");
    assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(!initiallyMarked));
  }
});

test("desktop Fork uses the exact sheet target and the session-bar helper, then restores inherited config", async () => {
  const h = harness();
  h.elements.prompt.value = "stale fork draft";
  h.context.persistDraft("forked-session");
  h.elements.prompt.value = "keep the open session draft";
  h.open();
  await h.dispatch(h.elements.sessionactionfork, "click").done;
  assert.deepEqual(h.requests.map((request) => [request.method, request.path]), [
    ["POST", "/sessions/target-session/fork"], ["GET", "/sessions"],
  ]);
  assert.equal(h.context.sessionId, "forked-session");
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.historyLoads(), 1);
  assert.equal(h.configLoads(), 1);
  assert.equal(h.drafts.get(h.context.draftKey("open-session")), "keep the open session draft");
  assert.match(between('$("fork").onclick =', '$("del").onclick ='), /forkSession\(sessionId\)/);
});

test("a failed desktop fork keeps selection and the draft", async () => {
  const h = harness({ forkError: "fork unavailable" });
  h.open();
  await h.dispatch(h.elements.sessionactionfork, "click").done;
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.prompt.value, "draft in A");
  assert.match(h.messages.at(-1).text, /fork failed: 503 fork unavailable/);
});

for (const acceptDelete of [false, true]) test("deleting a marked desktop session asks in-app (accept=" + acceptDelete + ")", async () => {
  const h = harness({ marked: true });
  h.elements.prompt.value = "target draft";
  h.context.persistDraft("target-session");
  h.elements.prompt.value = "keep the open draft";
  h.open();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.elements.sessdel.hidden, false);
  assert.match(h.elements.sessdeltext.textContent, /Target session/);
  await h.dispatch(acceptDelete ? h.elements.sessdelconfirm : h.elements.sessdelcancel, "click").done;
  assert.equal(h.elements.sessdel.hidden, true);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.prompt.value, "keep the open draft");
  assert.equal(h.requests.some((request) => request.method === "DELETE"), acceptDelete);
  assert.equal(h.drafts.has(h.context.draftKey("target-session")), !acceptDelete);
  assert.equal(h.stored.has(h.context.draftKey("target-session")), !acceptDelete);
  if (acceptDelete) {
    h.context.restoreDraft("target-session");
    assert.equal(h.elements.prompt.value, "");
  } else assert.deepEqual(h.requests, []);
});

test("deleting the selected desktop session cannot re-stash its removed draft", async () => {
  const h = harness({ marked: true });
  h.elements.prompt.value = "A survives deletion";
  await h.select();
  h.elements.prompt.value = "B is deleted";
  h.context.persistDraft();
  h.open();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  await h.dispatch(h.elements.sessdelconfirm, "click").done;
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.prompt.value, "A survives deletion");
  assert.equal(h.drafts.has(h.context.draftKey("target-session")), false);
  assert.equal(h.stored.has(h.context.draftKey("target-session")), false);
});

test("a failed deletion retains its session draft", async () => {
  const h = harness({ marked: true, deleteError: "delete unavailable" });
  h.context.persistDraft("target-session");
  h.open();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  await h.dispatch(h.elements.sessdelconfirm, "click").done;
  assert.equal(h.drafts.get(h.context.draftKey("target-session")), "draft in A");
  assert.equal(h.context.sessionId, "open-session");
  assert.match(h.messages.at(-1).text, /delete failed: 503 delete unavailable/);
});

test("desktop navigation stashes exact drafts and restores each session from memory", async () => {
  const h = harness(), a = "  half typed A\nsecond line  ", b = "unfinished B";
  h.elements.prompt.value = a;
  await h.select();
  assert.equal(h.elements.prompt.value, "");
  h.elements.prompt.value = b;
  h.stored.set(h.context.draftKey("open-session"), "stale storage value");
  await h.select("open-session");
  assert.equal(h.elements.prompt.value, a);
  await h.select();
  assert.equal(h.elements.prompt.value, b);
  assert.equal(h.drafts.get(h.context.draftKey("open-session")), a);
  assert.equal(h.grows(), 3);
  assert.deepEqual(h.requests, []);
});

test("an empty desktop draft suppresses old stored text and sends no request", async () => {
  const h = harness();
  h.elements.prompt.value = "";
  await h.select();
  h.stored.set(h.context.draftKey("open-session"), "old text");
  await h.select("open-session");
  assert.equal(h.elements.prompt.value, "");
  await h.context.send();
  assert.deepEqual(h.requests, []);
});

for (const text of ["  actual message\nwith whitespace  ", "/reset"]) test("desktop successful send clears its draft and preserves payload " + JSON.stringify(text), async () => {
  const h = harness();
  h.elements.prompt.value = text;
  h.context.persistDraft();
  await h.context.send();
  const turn = h.requests.find((request) => request.path.endsWith("/turns/stream"));
  assert.ok(turn, "the actual send must POST a turn");
  assert.equal(turn.path, "/sessions/open-session/turns/stream");
  assert.equal(turn.method, "POST");
  assert.equal(turn.body.prompt, text);
  assert.equal(turn.body.confirm, true);
  assert.equal(Object.hasOwn(turn.body, "skills"), false);
  assert.equal(Object.hasOwn(turn.body, "model"), false);
  assert.equal(h.context.pendingUserInputs.get("open-session")[0].settled, true, "the mock stream must be accepted");
  assert.equal(h.messages.some((message) => message.role === "err"), false);
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.drafts.has(h.context.draftKey("open-session")), false);
  assert.equal(h.stored.has(h.context.draftKey("open-session")), false);
  await h.select();
  h.elements.prompt.value = "B still owns its draft";
  await h.select("open-session");
  assert.equal(h.elements.prompt.value, "");
  await h.select();
  assert.equal(h.elements.prompt.value, "B still owns its draft");
  assert.equal(h.requests.filter((request) => request.path.endsWith("/turns/stream")).length, 1);
});

test("desktop new-session creation clears a stale returned-id draft and preserves the session left behind", async () => {
  const h = harness();
  h.elements.prompt.value = "stale returned-id draft";
  h.context.persistDraft("fresh-session");
  h.elements.prompt.value = "A should survive a new session";
  await h.elements.newbtn.onclick();
  assert.equal(h.context.sessionId, "fresh-session");
  assert.equal(h.context.pendingNew, "fresh-session");
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.drafts.has(h.context.draftKey("fresh-session")), false);
  assert.equal(h.stored.has(h.context.draftKey("fresh-session")), false);
  h.context.switchSession("open-session");
  assert.equal(h.elements.prompt.value, "A should survive a new session");
  h.context.switchSession("fresh-session");
  assert.equal(h.elements.prompt.value, "");
});

test("switching from a completed session lets the selected running session control the stop state", () => {
  const h = harness();
  vm.runInContext("eventStreamEnded = true", h.context);
  vm.runInContext("abortEvents = () => { eventStreamEnded = false; }", h.context);

  h.context.switchSession("fresh-session");

  assert.equal(h.context.eventStreamEnded, false);
  vm.runInContext("turnRunning = true; turnRunningAt = Date.now()", h.context);
  assert.equal(h.context.isTurnRunning(), true);
});

test("a picked session retries a dropped running-state sample", async () => {
  const h = harness();
  vm.runInContext("turnRunning = false; turnRunningAt = 0; eventStreamEnded = false; eventStreamFailed = false", h.context);
  let calls = 0;
  h.context.pollConfirm = async () => {
    calls += 1;
    if (calls === 1) return null; // dropped while a newer marker bumped the sequence
    vm.runInContext("turnRunning = true; turnRunningAt = Date.now()", h.context);
    return true;
  };
  h.context.pollSwitchRunningState();
  for (let i = 0; i < 3; i++) await Promise.resolve();
  assert.equal(calls, 1, "the first pick poll is sent");
  h.advance(400);
  for (let i = 0; i < 3; i++) await Promise.resolve();
  assert.equal(calls, 2, "a dropped sample is retried once");
  assert.equal(h.context.isTurnRunning(), true);
});

test("a first desktop send clears both the anonymous and adopted session drafts", async () => {
  const h = harness({ sid: "" });
  h.elements.prompt.value = "stale fresh draft";
  h.context.persistDraft("fresh-session");
  h.elements.prompt.value = "first actual message";
  h.context.persistDraft();
  await h.context.send();
  assert.equal(h.context.sessionId, "fresh-session");
  assert.equal(h.requests.find((request) => request.path === "/sessions/fresh-session/turns/stream").body.prompt, "first actual message");
  for (const id of ["", "fresh-session"]) {
    assert.equal(h.drafts.has(h.context.draftKey(id)), false);
    assert.equal(h.stored.has(h.context.draftKey(id)), false);
  }
  h.context.switchSession("open-session");
  h.context.switchSession("fresh-session");
  assert.equal(h.elements.prompt.value, "");
});

test("phone drafts continue to restore from storage and the phone hold keeps its original action set", async () => {
  const h = harness({ phone: true });
  await h.select();
  h.drafts.set(h.context.draftKey("open-session"), "desktop cache");
  h.stored.set(h.context.draftKey("open-session"), "phone stored draft");
  await h.select("open-session");
  assert.equal(h.elements.prompt.value, "phone stored draft");
  const title = h.title(), star = h.row().querySelector(".mk");
  assert.equal(title.getAttribute("aria-haspopup"), null);
  assert.equal(title.getAttribute("aria-keyshortcuts"), null);
  h.dispatch(title, "dblclick", { detail: 2 });
  assert.equal(h.elements.sessionactions.hidden, true);
  const touch = { pointerType: "touch", isPrimary: true, button: 0, clientX: 10, clientY: 20 };
  h.dispatch(h.row(), "pointerdown", touch);
  h.advance(650);
  h.dispatch(h.row(), "pointerup", touch);
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(h.elements.sessionactionfork.hidden, true);
  h.dispatch(h.document.activeElement, "keydown", { key: "Tab" });
  assert.equal(h.document.activeElement, h.elements.sessionactiondelete);
  h.dispatch(h.document.activeElement, "keydown", { key: "Escape" });
  assert.equal(h.document.activeElement, star);
});

test("the sheet has desktop placement, modal attributes and unchanged theme-based touch targets", () => {
  const desktop = css.slice(css.indexOf("@media (min-width:701px)"), css.indexOf("/* Keep narrow desktop"));
  assert.match(desktop, /#sessionactionspanel \{[\s\S]*?top:50%; bottom:auto; transform:translateY\(-50%\)/);
  assert.match(desktop, /border-bottom:1px solid var\(--line\); border-radius:var\(--radius-card\)/);
  assert.match(css, /#sessionactionspanel button \{ min-width:44px; min-height:44px;/);
  assert.match(html, /id="sessionactionspanel" role="dialog" aria-modal="true" aria-labelledby="sessionactionstitle"/);
});

for (const phone of [true, false]) test((phone ? "phone" : "desktop") + " session header title stays readable alongside long metadata", () => {
  const h = harness({ phone });
  assert.equal(h.context.isPhoneLayout(), phone);
  const summary = "Investigate mobile session title readability and " + "a_long_unbroken_title_".repeat(8);
  Object.assign(h.context.lastItems[0], { summary, marked: true, size_bytes: 12800000 });
  h.context.lastMsgCount = 123456;
  h.context.renderSessionBar();
  assert.equal(h.elements.sesstitle.textContent, "★ " + summary);
  assert.equal(h.elements.sessmeta.textContent, "open-ses · 12500 KB · 123456 msgs · test-page");
  assert.equal(h.elements.sesscwd.textContent, "📁 /srv/open-session");
  assert.equal(h.elements.sesscwd.title, "当前路径: /srv/open-session");
  assert.equal(h.elements.sesscwd.hidden, false);
  assert.equal(h.elements.sesstitle.hidden, false);
  const titlePath = shellPathTo("sesstitle");
  assert.deepEqual(titlePath.map((item) => item.id).filter(Boolean), ["serve-app", "sessbar", "sesstitle"]);
  assert.ok(titlePath.every((item) => !item.hidden));
  const bar = responsiveRule("#sessbar", phone), title = responsiveRule("#sesstitle", phone);
  const meta = responsiveRule("#sessmeta", phone);
  const cwd = responsiveRule("#sesscwd", phone);
  assert.notEqual(title.display, "none");
  assert.equal(title.overflow, "hidden");
  assert.equal(meta["white-space"], "nowrap");
  if (phone) {
    assert.equal(bar.display, "grid");
    assert.equal(bar["grid-template-columns"], "minmax(0,1fr) auto");
    assert.equal(bar["flex-shrink"], "0");
    assert.equal(title["grid-column"], "1 / -1");
    assert.equal(title["min-width"], "0");
    assert.equal(title["white-space"], "normal");
    assert.equal(title["overflow-wrap"], "anywhere");
    assert.equal(title.display, "-webkit-box");
    assert.equal(title["-webkit-box-orient"], "vertical");
    assert.equal(title["-webkit-line-clamp"], "2");
    assert.equal(title["line-height"], "1.4");
    assert.equal(title["max-height"], "2.8em");
    assert.equal(cwd["grid-column"], "1 / -1");
    assert.equal(cwd["max-width"], "none");
    assert.equal(meta["min-width"], "0");
    assert.equal(meta.overflow, "hidden");
    assert.equal(meta["text-overflow"], "ellipsis");
    assert.equal(responsiveRule("#questionbtn.show", true).display, "grid");
  } else {
    assert.equal(bar.display, "flex");
    assert.equal(bar["min-height"], "32px");
    assert.equal(bar.padding, "6px 16px");
    assert.equal(title.flex, "1");
    assert.equal(title["white-space"], "nowrap");
    assert.equal(title["text-overflow"], "ellipsis");
    assert.deepEqual(cssRule(desktopCss, "#sesstitle"), { "min-width": "0", "font-size": "15px" });
    assert.equal(responsiveRule("#questionbtn").display, "none");
  }
});

test("desktop Markdown tables use readable spacing without stretching or clipping wide tables", () => {
  const table = responsiveRule(".msg.md table");
  const wrap = responsiveRule(".msg.md .tablewrap");
  assert.equal(table["font-size"], "15px");
  assert.equal(table["border-collapse"], "collapse");
  assert.equal(wrap.margin, "12px 0");
  assert.equal(wrap["overflow-x"], "auto", "wide tables keep the existing horizontal scroll region");
  for (const rule of [table, wrap]) {
    for (const property of ["width", "min-width", "max-width", "table-layout", "white-space", "overflow", "display"]) {
      assert.equal(rule[property], undefined, property + " must not replace intrinsic sizing or scrolling");
    }
  }
  assert.equal(responsiveRule(".msg")["font-size"], "17px", "body copy is not resized");
});

test("desktop Markdown tables use horizontal separators and a theme-aware header", () => {
  const cells = responsiveRule(".msg.md th, .msg.md td");
  assert.equal(cells.border, "0", "remove the shared full cell grid");
  assert.equal(cells["border-bottom"], "1px solid var(--card-edge)");
  assert.equal(cells.padding, "8px 12px");
  assert.equal(cells["vertical-align"], "top", "multiline cells align at their first line");
  assert.equal(cells["text-align"], "left");
  assert.deepEqual(cssRule(desktopCss, ".msg.md th"), {
    background: "var(--surface-2)", color: "var(--fg)", "font-weight": "600",
    "border-bottom-color": "var(--line)",
  });
});

test("desktop table code is quieter without changing code outside tables", () => {
  assert.deepEqual(cssRule(desktopCss, ".msg.md table code"), {
    "font-size": "14px", background: "var(--tint-2)", border: "0",
    "border-radius": "3px", padding: "1px 3px",
  });
  const code = responsiveRule(".msg.md code");
  assert.equal(code["font-size"], "14px");
  assert.equal(code.background, "var(--code-bg)");
  assert.equal(code.border, "1px solid var(--line)");
  assert.equal(code.padding, "0 4px");
  assert.equal(code["overflow-wrap"], "anywhere");
  assert.deepEqual(cssRule(desktopCss, ".msg.md pre code"), {}, "code blocks keep their existing rules");
});

test("phone Markdown tables retain compact cells and the original code capsules", () => {
  assert.deepEqual(responsiveRule(".msg.md table", true), { "border-collapse": "collapse", "font-size": "13px" });
  assert.deepEqual(responsiveRule(".msg.md th, .msg.md td", true), {
    border: "1px solid var(--line)", padding: "2px 6px", "text-align": "left",
  });
  assert.deepEqual(responsiveRule(".msg.md .tablewrap", true), { "overflow-x": "auto", margin: "6px 0" });
  const code = responsiveRule(".msg.md code", true);
  assert.equal(code["font-size"], "13px");
  assert.equal(code.background, "var(--code-bg)");
  assert.equal(code.border, "1px solid var(--line)");
  assert.equal(code["border-radius"], "5px");
  assert.equal(code.padding, "0 4px");
  assert.deepEqual(responsiveRule(".msg.md table code", true), {});
});

test("desktop user bubbles stay neutral but carry a border and accent rule", () => {
  const desktopBubble = responsiveRule(".msg.user");
  assert.equal(desktopBubble.background, "var(--surface-2)");
  assert.equal(desktopBubble.color, "var(--fg)");
  assert.equal(desktopBubble.border, "1px solid var(--line)");
  assert.equal(responsiveRule(".user-msg-row > .msg.user")["border-left"], "3px solid var(--acc)");
  for (const [selector, background] of [
    [":root", "#2e2e2e"],
    [':root[data-theme="light"]', "#e5e5e5"],
    [':root[data-theme="warm"]', "#e8e5de"],
  ]) {
    assert.equal(responsiveRule(selector)["--surface"], background, selector + " user bubble fill");
  }
  assert.equal(responsiveRule("#sessions li.cur")["box-shadow"], "inset 3px 0 0 var(--acc)");
  assert.equal(responsiveRule("#sessions li.cur").background, "var(--tint-3)");
  assert.equal(responsiveRule('#sessions li.cur span[role="button"]')["font-weight"], "600");
  assert.equal(responsiveRule("button.primary").background, "var(--acc)");
  assert.equal(responsiveRule("button.primary").color, "var(--on-acc)");
  const phoneBubble = responsiveRule(".msg.user", true);
  assert.equal(phoneBubble.background, "var(--acc-solid)");
  assert.equal(phoneBubble.color, "var(--on-user)");
  assert.equal(responsiveRule("#sessions li.cur", true)["box-shadow"], "inset 3px 0 0 var(--acc)");
});

test("desktop prompt index starts at the newest question and upward wheel loads older prompts", async () => {
  const h = harness(), list = h.elements.promptlist;
  const originalQueryAll = list.querySelectorAll.bind(list);
  list.querySelectorAll = (selector) => {
    if (selector === "li[data-history-index]") return list.children.filter((item) => item.dataset.historyIndex !== undefined);
    if (selector === "li.cur") return list.children.filter((item) => item.classList.contains("cur"));
    return originalQueryAll(selector);
  };
  Object.defineProperty(list, "scrollHeight", { get: () => list.children.length * 30 });
  list.clientHeight = 90;
  const rows = [];
  const makeRow = (index) => {
    const row = h.document.createElement("div"), bubble = h.document.createElement("div");
    row.className = "user-msg-row";
    bubble.className = "msg user";
    bubble.dataset.historyIndex = String(index);
    bubble.textContent = "question " + index;
    row.appendChild(bubble);
    row.querySelector = (selector) => selector === ".msg.user" ? bubble : null;
    return row;
  };
  rows.push(...[4, 5, 6, 7, 8, 9].map(makeRow));
  h.elements.msgs.querySelectorAll = (selector) => selector === ".user-msg-row" ? rows : [];
  Object.assign(h.context, { canonicalHistoryTotal: 10, lastMsgCount: 6 });
  vm.runInContext(between("function promptNavHasOlder()", 'if ($("promptlist") && typeof $("promptlist").addEventListener'), h.context);
  h.context.syncPromptNav();
  assert.equal(list.children.at(-1).classList.contains("cur"), true, "the newest prompt is selected by default");
  assert.equal(list.scrollTop, list.scrollHeight, "the initial prompt window opens at its newest end");

  const stableCount = h.elements.promptcount.textContent;
  const stableItems = list.children.slice();
  h.elements.msgs.querySelectorAll = () => [];
  h.context.historyLoadState = () => ({ loading: 1 });
  h.context.syncPromptNav();
  assert.equal(h.elements.promptcount.textContent, stableCount, "a transient empty history frame keeps its count badge");
  assert.deepEqual(list.children, stableItems, "a transient empty history frame keeps the prompt rail mounted");
  h.context.historyLoadState = () => ({ loading: 0 });
  h.elements.msgs.querySelectorAll = (selector) => selector === ".user-msg-row" ? rows : [];

  list.scrollTop = 0;
  rows.unshift(...[0, 1, 2, 3].map(makeRow));
  h.context.lastMsgCount = 10;
  h.context.syncPromptNav();
  assert.equal(list.scrollTop, 120, "prepending older prompts keeps the previous first prompt in place");
  assert.equal(list.children.at(-1).classList.contains("cur"), true, "the newest prompt remains selected");

  const listeners = {}, olderLoads = [];
  list.addEventListener = (type, listener) => { listeners[type] = listener; };
  h.context.lastMsgCount = 6;
  h.context.loadOlderHistory = async () => { olderLoads.push(true); };
  vm.runInContext(between('if ($("promptlist") && typeof $("promptlist").addEventListener', "if (typeof MutationObserver"), h.context);
  list.scrollTop = 0;
  listeners.wheel({ deltaY: -1 });
  await Promise.resolve();
  assert.equal(olderLoads.length, 1, "an upward gesture works even before the list has a scroll range");
});

function loadChat(h) {
  Object.assign(h.context, {
    esc: (text) => String(text), updateQuestionButton() {},
    sideNoteBusy: false, turnRunningAt: 1000, RUNNING_SAMPLE_TTL_MS: 30000,
    staleSession: "", staleDone: false, eventActive: false, eventSession: "",
    holdLayout: (_, apply) => apply(),
    foldOpenByKey: new Map(),
  });
  for (const [start, end] of [
    ["function textOf(", "const PHONE_LAYOUT_QUERY"],
    ["function addMsg(", "// Assistant replies render as markdown"],
    ["function historyRowsOf(", "// A session the server reports as running"],
    ["function addFold(", "// Detached cards are kept weakly"],
    ["function attachUserRewind(", "async function rewindTo("],
    ["function isTurnRunning(", "// Transient note through"],
    ["async function sendSideNote(", "async function stopTurn("],
    ['$("send").onclick =', "// The dialog is the only way"],
  ]) vm.runInContext(between(start, end), h.context);
}

function sessionFoldHarness(width) {
  const h = harness({ phone: width <= 700 }), pendingToggles = new Set(), historyLoads = [];
  const createElement = h.document.createElement;
  h.document.createElement = (tag) => {
    const el = createElement(tag);
    if (tag === "details") {
      let open = false;
      Object.defineProperty(el, "isConnected", {
        get() {
          for (let node = el; node; node = node.parentNode) if (node === h.elements.msgs) return true;
          return false;
        },
      });
      Object.defineProperty(el, "open", {
        get: () => open,
        set(value) {
          value = !!value;
          if (value === open) return;
          open = value;
          // Browsers queue and coalesce toggle events independently of clicks.
          pendingToggles.add(el);
        },
      });
    }
    return el;
  };
  const matchMedia = h.context.window.matchMedia;
  h.context.window.matchMedia = (query) => query === "(min-width:701px)"
    ? { matches: width >= 701 } : matchMedia(query);
  loadChat(h);
  vm.runInContext(between("function toolArgsOf(", "const RATE_STORAGE ="), h.context);
  Object.assign(h.context, {
    historySession: "", historyKeys: [], historyLoadSeq: 0,
    AbortController,
    navigator: { onLine: true }, staleKey: "", staleBox: null, recoveredView: null, staleTries: 0,
    histKey: (messages) => JSON.stringify(messages), histEndsTurn: () => true,
    revokePreviews() {},
    console: { error(...args) { assert.fail(args.map(String).join(" ")); } },
  });
  h.document.visibilityState = "visible";
  vm.runInContext(between("function armHistoryFollow()", "// BEGIN history retry"), h.context);
  vm.runInContext(between("async function loadHistory(", "// Scrolling to the top of the transcript"), h.context);
  const messages = [
    { role: "assistant", content: "Finished answer", reasoning_content: "Stored reasoning" },
    { role: "tool", content: "Plain tool output" },
    { role: "tool", content: "Raw task result", task_report: {
      schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: false,
      tasks: [{ task_id: "fold-task", agent: "build", status: "completed", active: false, output: "Delivered output" }],
    } },
    { role: "internal_note", content: "Explicitly expanded control" },
  ];
  const fetch = h.context.fetch;
  h.context.fetch = async (url, options = {}) => {
    const path = new URL(url).pathname;
    if (/^\/sessions\/(open-session|target-session)\/history$/.test(path)) {
      h.requests.push({ path, method: options.method || "GET", body: null });
      return response(plain(messages), 200, { "x-history-total": String(messages.length) });
    }
    return fetch(url, options);
  };
  const loadHistory = h.context.loadHistory;
  h.context.loadHistory = (...args) => {
    const load = loadHistory(...args);
    historyLoads.push(load);
    return load;
  };
  return {
    ...h,
    // Existing fold-state tests still exercise every original disclosure. The
    // outer execution disclosures have their own default/state tests below.
    folds: () => h.elements.msgs.querySelectorAll("details").filter((details) => !details.parentNode.classList.contains("execution-group")),
    async activateSummary(details, detail = 1) {
      const click = h.dispatch(details.querySelector("summary"), "click", { detail });
      if (!click.event.prevented) details.open = !details.open;
      await click.done;
    },
    flushToggles() {
      const queued = [...pendingToggles]; pendingToggles.clear();
      for (const details of queued) h.dispatch(details, "toggle");
    },
    deliverToggle(details) {
      assert.equal(pendingToggles.delete(details), true, "the selected element must have a queued toggle");
      h.dispatch(details, "toggle");
    },
    async selectHistory(id) {
      await h.select(id);
      await Promise.all(historyLoads.splice(0));
      assert.equal(h.context.sessionId, id);
      assert.equal(h.context.historySession, id, "the real sidebar selection must finish the history render");
    },
  };
}

test("completed execution groups add vertical breathing room without changing the shared gutter on both layouts", () => {
  for (const phone of [false, true]) {
    const items = responsiveRule(".execution-group > details > .execution-items", phone);
    assert.equal(items.padding, "16px", "the same inset applies on every side");
    assert.deepEqual([items.background, items.border, items["border-radius"]], [undefined, undefined, undefined],
      "the merged rows keep the shared card surface instead of a nested well");
    const summary = responsiveRule(".msg.asst.execution-group > details > summary", phone);
    assert.deepEqual(summary, {}, "the group header keeps the shared summary inset");
  }
  // The stylesheet has multiple media blocks, so enumerate every gap rule
  // rather than relying on the fixture's slices around the first media block.
  const turnRules = [...css.matchAll(/(?:^|\n)\s*\.turn\s*\{[^}]*\}/g)];
  assert.deepEqual(turnRules.map(([rule]) => cssRule(rule, ".turn").gap), ["8px", "12px", "8px"],
    "base, shared visual and phone card spacing declarations stay unchanged");
});

for (const width of [390, 1280]) {
  for (const stored of [false, true]) test(`completed execution groups collapse only on blank double clicks (${stored ? "history" : "live"}) at ${width}px`, async () => {
    const h = sessionFoldHarness(width), parent = h.document.createElement("div");
    parent.className = "turn"; h.elements.msgs.appendChild(parent);
    const first = h.context.addFold("read_file", "selectable output", parent, "blank:first");
    const second = h.context.addFold("thinking", "reasoning", parent, "blank:second");
    first.box.dataset.executionState = second.box.dataset.executionState = "completed";
    h.context.addFold("current execution", "running", parent, "blank:current");
    const groups = h.context.executionGroupsFor(parent, stored);
    groups.sync();
    const box = parent.querySelector(".execution-group"), details = box.querySelector("details");
    const items = box.querySelector(".execution-items"), inner = first.box.querySelector("details");
    const key = details.dataset.foldKey;
    const rememberedOpen = () => vm.runInContext(stored ? "foldOpenState" : "foldOpenByKey", h.context).get(key) === true;
    const button = h.document.createElement("button"), text = h.document.createElement("span");
    text.textContent = "copy this text"; first.body.appendChild(text); first.body.appendChild(button);
    let activations = 0;
    button.ondblclick = () => { activations++; };
    await h.activateSummary(inner); h.flushToggles();
    await h.activateSummary(details); h.flushToggles();
    assert.equal(rememberedOpen(), true);
    for (const target of [box, details, items]) {
      h.dispatch(target, "click");
      assert.equal(details.open, true, "single clicks in blank space do not collapse the group");
    }
    for (const target of [details.querySelector("summary"), first.box, inner, inner.querySelector("summary"), first.body, text, button]) {
      const { event } = h.dispatch(target, "dblclick", { detail: 2 });
      assert.equal(details.open, true, "nested cards and summaries do not invoke the outer shortcut");
      assert.equal(event.prevented, false, "text selection and control defaults remain available");
    }
    assert.equal(activations, 1, "an inner control still receives its double click");
    for (const target of [items, details, box]) {
      const { event } = h.dispatch(target, "dblclick", { detail: 2 });
      assert.equal(details.open, false, "double clicking outer padding or a gap closes only this group");
      assert.equal(event.prevented, true);
      assert.equal(event.stopped, true);
      assert.equal(inner.open, true, "closing the outer group retains the inner disclosure");
      h.flushToggles();
      assert.equal(rememberedOpen(), false, "the existing fold cache remembers the collapse");
      // Restoring takes cards out of the group; close the inner disclosure so
      // the existing expanded-card safeguard allows it to be grouped again.
      await h.activateSummary(inner); h.flushToggles();
      groups.restore(); groups.sync(); h.flushToggles();
      assert.equal(parent.querySelector(".execution-group"), box, "regrouping reuses the original group");
      assert.equal(details.open, false, "regrouping does not reopen a collapsed group");
      assert.equal(first.box.parentNode, items);
      await h.activateSummary(inner); h.flushToggles();
      await h.activateSummary(details); h.flushToggles();
    }
    await h.activateSummary(details); h.flushToggles();
    const { event } = h.dispatch(box, "dblclick", { detail: 2 });
    assert.equal(details.open, false, "double clicking a closed group never reopens it");
    assert.equal(event.prevented, false);
  });

  test(`completed execution groups keep current execution outside and retain both disclosures at ${width}px`, async () => {
    const h = sessionFoldHarness(width), root = h.elements.msgs;
    vm.runInContext(between("function createLiveActivity(", "// The three override picks"), h.context);
    const box = h.document.createElement("div"); box.className = "turn"; root.appendChild(box);
    const pending = h.document.createElement("div"); pending.className = "msg asst pending";
    pending.textContent = "Processing…"; box.appendChild(pending);
    const activity = h.context.createLiveActivity(box, pending);
    const first = h.context.createThinkingFold(box, pending);
    first.body.textContent = "first reasoning";
    const inner = first.box.querySelector("details");
    h.context.completeExecution(first, box);
    assert.equal(first.box.parentNode, box, "the latest completed item stays visible until another execution appears");
    assert.equal(box.querySelector(".execution-group"), null);
    activity.tool({ id: "run", name: "execute_command", phase: "started" });
    const outerBox = box.querySelector(".execution-group"), outer = outerBox.querySelector("details");
    assert.equal(outer.open, false, "completed records default closed on both layouts");
    assert.equal(inner.open, false);
    assert.equal(first.box.parentNode, outerBox.querySelector(".execution-items"));
    await h.activateSummary(inner); h.flushToggles();
    assert.equal(inner.open, true);
    h.context.executionGroupsFor(box).sync();
    assert.equal(inner.open, true, "merging leaves an expanded inner disclosure open");
    assert.equal(first.box.parentNode, outerBox.querySelector(".execution-items"), "an expanded record inside a group stays there");
    await h.activateSummary(outer); h.flushToggles(); assert.equal(outer.open, true);
    await h.activateSummary(outer); h.flushToggles(); assert.equal(outer.open, false);

    const activeTool = box.children.find((child) => child.classList.contains("tool-activity"));
    assert.ok(activeTool, "a started tool must be a direct, visible sibling of the completed group");
    assert.equal(activeTool.dataset.running, "true");
    assert.equal(activeTool.dataset.executionState, "running");
    assert.equal(pending.parentNode, box);
    assert.equal(box.classList.contains("execution-empty"), false);
    activity.tool({ id: "run", name: "execute_command", phase: "finished", ok: true, output: "done" });
    assert.equal(activeTool.parentNode, box, "finishing does not hide the latest progress");
    assert.equal(outer.open, false);
    activity.tool({ id: "next", name: "read_file", phase: "started" });
    assert.equal(activeTool.parentNode, outerBox.querySelector(".execution-items"), "a third item merges previous progress with the old group");
    assert.equal(outerBox.querySelector(".execution-items").children.length, 2);
    activity.tool({ id: "run", name: "execute_command", phase: "started" });
    assert.equal(activeTool.parentNode, box, "reactivation extracts the same node from the closed group");
    assert.equal(activeTool.dataset.running, "true");

    const second = h.context.createThinkingFold(box, pending);
    second.body.textContent = "current reasoning";
    h.context.executionGroupsFor(box).sync();
    assert.equal(second.box.parentNode, box, "current reasoning remains outside old execution");
    assert.equal(second.box.dataset.executionState, "running");
    assert.equal(second.body.textContent, "current reasoning");
    assert.equal(pending.parentNode, box);
    assert.equal(outerBox.querySelector(".execution-items").children.length, 1);
    const body = outerBox.querySelector(".execution-items");
    let moves = 0;
    const append = body.appendChild.bind(body), insert = box.insertBefore.bind(box);
    body.appendChild = (...args) => { moves++; return append(...args); };
    box.insertBefore = (...args) => { moves++; return insert(...args); };
    root.scrollTop = 75;
    h.context.executionGroupsFor(box).sync();
    h.context.executionGroupsFor(box).sync();
    assert.equal(moves, 0, "unchanged progress does not detach or remount completed records");
    assert.equal(root.scrollTop, 75, "unchanged progress does not move the reading position");
    activity.finish("interrupted");
    assert.equal(activeTool.parentNode, box, "an interrupted tool stays visible rather than becoming hidden success");
  });

  test(`stored task_wait rows merge into one completed group at ${width}px`, async () => {
    const h = sessionFoldHarness(width), parent = h.document.createElement("div");
    h.elements.msgs.appendChild(parent);
    const report = (id, output) => ({ schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: false,
      tasks: [{ task_id: id, agent: "build", status: "running", active: true, output }] });
    const rows = [
      { role: "assistant", content: "", reasoning_content: "first wait", tool_calls: [{ id: "c1", type: "function", function: { name: "task_wait", arguments: "{}" } }] },
      { role: "tool", content: "wait one", task_report: report("t1", "one") },
      { role: "assistant", content: "", reasoning_content: "second wait", tool_calls: [{ id: "c2", type: "function", function: { name: "task_wait", arguments: "{}" } }] },
      { role: "tool", content: "wait two", task_report: report("t2", "two") },
      { role: "assistant", content: "Final answer" },
    ];
    for (const row of h.context.historyRowsOf(rows, rows.length, true).rows) h.context.renderHistoryRow(row, parent);
    for (const turn of parent.querySelectorAll(".turn")) {
      const rate = h.document.createElement("div");
      rate.className = "rate"; rate.dataset.turnRate = "1";
      turn.appendChild(rate);
    }
    h.context.executionGroupsFor(parent, true).sync();
    // One stored turn ends each assistant row with a rate line; it must not split
    // the run, or every task card stays in its own "Completed execution · 1" group.
    assert.deepEqual(parent.querySelectorAll(".execution-group").map((group) => group.querySelector("summary").textContent),
      ["Completed execution · 5"], "consecutive stored task records share one group");
    assert.equal(parent.querySelectorAll(".execution-items .rate").length, 0, "rate text stays outside the wrapped records");
  });

  test(`stored records keep their place when a live row arrives at ${width}px`, () => {
    const h = sessionFoldHarness(width), parent = h.document.createElement("div");
    h.elements.msgs.appendChild(parent);
    const call = (id, name) => ({ role: "assistant", content: "",
      tool_calls: [{ id, type: "function", function: { name, arguments: "{}" } }] });
    const rows = [
      call("c1", "read_file"), { role: "tool", content: "one" },
      call("c2", "execute_command"), { role: "tool", content: "two" },
      call("c3", "task_wait"), { role: "tool", content: "three" },
      { role: "assistant", content: "Older answer" },
    ];
    for (const row of h.context.historyRowsOf(rows, rows.length, true).rows) h.context.renderHistoryRow(row, parent);
    const groups = h.context.executionGroupsFor(parent, true);
    groups.sync();
    const group = parent.querySelectorAll(".execution-group")[0];
    const items = group.querySelectorAll(".execution-items")[0];
    const order = Array.from(items.children).map((node) => node.textContent.trim().slice(0, 16));
    const live = h.document.createElement("div");
    live.className = "turn";
    live.dataset.canonicalKey = "live";
    const pending = h.document.createElement("div");
    pending.className = "msg asst pending";
    pending.textContent = "Processing…";
    live.appendChild(pending);
    parent.appendChild(live);
    groups.sync();
    // Re-inserting the old members must not fall back to "append at the end":
    // that moved the earliest record below newer rows and dragged the group with it.
    assert.equal(parent.lastElementChild, live, "a new live row stays last");
    assert.deepEqual(Array.from(items.children).map((node) => node.textContent.trim().slice(0, 16)), order,
      "merged records keep their order");
    assert.equal(parent.querySelectorAll(".execution-group")[0], group, "the group keeps its place");
  });

  test(`expanded progress keeps its place until the reader collapses it at ${width}px`, async () => {
    const h = sessionFoldHarness(width), root = h.elements.msgs;
    vm.runInContext(between("function createLiveActivity(", "// The three override picks"), h.context);
    const box = h.document.createElement("div"); box.className = "turn"; root.appendChild(box);
    const pending = h.document.createElement("div"); pending.className = "msg asst pending";
    pending.textContent = "Processing…"; box.appendChild(pending);
    const activity = h.context.createLiveActivity(box, pending);
    const first = h.context.createThinkingFold(box, pending);
    first.body.textContent = "watching";
    h.context.completeExecution(first, box);
    const inner = first.box.querySelector("details");
    await h.activateSummary(inner); h.flushToggles();
    assert.equal(inner.open, true);
    activity.tool({ id: "run", name: "execute_command", phase: "started" });
    assert.equal(first.box.parentNode, box, "an expanded record is not absorbed by the next execution");
    assert.equal(box.querySelector(".execution-group"), null);
    await h.activateSummary(inner); h.flushToggles();
    const outerBox = box.querySelector(".execution-group");
    assert.ok(outerBox, "closing the record lets the deferred merge run");
    assert.equal(first.box.parentNode, outerBox.querySelector(".execution-items"));
  });

  test(`an expanded subagent card is not re-anchored by later progress at ${width}px`, async () => {
    const h = sessionFoldHarness(width), root = h.elements.msgs;
    Object.defineProperty(h.document, "hidden", { value: true, configurable: true });
    h.document.addEventListener = () => {};
    h.context.window.addEventListener = () => {};
    h.context.MutationObserver = class { observe() {} disconnect() {} };
    vm.runInContext(between("const subagentClocks = new WeakMap();", "// The three override picks"), h.context);
    const box = h.document.createElement("div"); box.className = "turn"; root.appendChild(box);
    const pending = h.document.createElement("div"); pending.className = "msg asst pending";
    pending.textContent = "Processing…"; box.appendChild(pending);
    const activity = h.context.createLiveActivity(box, pending);
    const report = { schema: "ai.task-report", version: 1, tasks: [{ task_id: "worker", status: "running", active: true, agent: "build", elapsed_ms: 4000 }] };
    activity.tool({ id: "spawn", name: "task_spawn", phase: "started", task_report: report });
    const card = box.children.find((child) => child.dataset?.subagents === "true"), details = card.querySelector("details");
    assert.equal(details.open, true, "a running report opens the card");
    const index = () => box.children.indexOf(card), before = index();
    activity.tool({ id: "wait", name: "task_wait", phase: "started", task_report: report });
    assert.equal(index(), before, "a later task row does not move an open subagent card");
    await h.activateSummary(details); h.flushToggles();
    assert.equal(details.open, false);
    activity.tool({ id: "wait2", name: "task_wait", phase: "started", task_report: report });
    assert.ok(index() > before, "a collapsed card follows the newest task row again");
  });

  test(`execution merges compensate the reading anchor and ignore detached turns at ${width}px`, () => {
    const h = sessionFoldHarness(width), root = h.elements.msgs;
    const createElement = h.document.createElement;
    h.document.createElement = (tag) => {
      const node = createElement(tag);
      node.getBoundingClientRect = () => node.classList.contains("execution-group")
        ? { top: -30, bottom: 10 } : { top: 20, bottom: 60 };
      return node;
    };
    root.getBoundingClientRect = () => ({ top: 0, bottom: 300 });
    h.context.shouldFollowOutput = () => false;
    const box = h.document.createElement("div"); box.className = "turn"; root.appendChild(box);
    const first = h.context.createThinkingFold(box);
    h.context.completeExecution(first, box);
    root.scrollTop = 100;
    const second = h.context.createThinkingFold(box);
    assert.equal(root.scrollTop, 50, "a 50px upward anchor shift is compensated in the same mutation");
    h.context.completeExecution(second, box);
    h.context.shouldFollowOutput = () => true;
    const frames = [];
    h.context.followOutput = (follow, target) => { if (follow) frames.push(() => { target.scrollTop = target.scrollHeight; }); };
    root.scrollHeight = 1234;
    const third = h.context.createThinkingFold(box);
    assert.equal(root.scrollTop, 50, "a merge does not move the reader before the cancellable follow frame");
    frames.shift()();
    assert.equal(root.scrollTop, 1234, "the follow frame retains the live edge");
    box.remove();
    root.scrollTop = 77;
    h.context.completeExecution(third, box);
    h.context.createThinkingFold(box);
    assert.equal(root.scrollTop, 77, "background updates cannot scroll the mounted session");
    assert.equal(frames.length, 0, "detached updates cannot queue a follow for the mounted session");
  });

  test(`history execution groups preserve boundaries and settle retained task rows at ${width}px`, async () => {
    const h = sessionFoldHarness(width), root = h.elements.msgs;
    let messages = [
      { role: "user", content: "request" },
      { role: "assistant", content: "", reasoning_content: "old reasoning" },
      { role: "tool", content: "old tool output" },
      { role: "assistant", content: "Visible answer" },
      { role: "tool", content: "active task", task_report: {
        schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: false,
        tasks: [{ task_id: "active", agent: "build", status: "running", active: true, output: "still running" }],
      } },
    ];
    const fetch = h.context.fetch;
    h.context.fetch = async (url, options) => new URL(url).pathname.endsWith("/history")
      ? response(plain(messages), 200, { "x-history-total": String(messages.length) }) : fetch(url, options);
    await h.context.loadHistory();
    const outer = root.querySelector(".execution-group"), detail = outer.querySelector("details");
    const records = outer.querySelector(".execution-items").children.slice();
    assert.equal(detail.open, false);
    assert.equal(records.length, 2, "only contiguous old reasoning and tool rows share an outer group");
    const active = root.children.find((node) => node.dataset.canonicalIndex === "4");
    const task = active.querySelector(".task-output-body");
    assert.ok(task);
    assert.equal(active.parentNode, root);
    assert.equal(active.querySelector(".execution-group"), null, "active subagent progress stays directly visible");
    assert.equal(root.children.find((node) => node.dataset.canonicalIndex === "0").textContent.includes("request"), true);
    assert.equal(root.children.find((node) => node.dataset.canonicalIndex === "3").textContent.includes("Visible answer"), true);
    await h.activateSummary(detail); h.flushToggles(); assert.equal(detail.open, true);
    messages = [...messages, { role: "assistant", content: "Final answer" }];
    await h.context.loadHistory();
    assert.equal(root.querySelector(".execution-group"), outer, "unchanged rows keep the original outer node");
    assert.equal(detail.open, true, "an explicit expansion survives an incremental history load");
    assert.deepEqual(outer.querySelector(".execution-items").children, records);
    assert.equal(active.querySelector(".task-output-body"), task, "settlement never remounts the retained attachment node");
    assert.equal(active.querySelector(".execution-group").querySelector("details").open, false);
    assert.ok(root.textContent.includes("Final answer"));
    let removals = 0;
    const remove = outer.remove.bind(outer);
    const present = new Set();
    const collect = (owner) => {
      for (const node of owner.children) {
        if (node.dataset.canonicalKey !== undefined) present.add(node.dataset.canonicalKey);
        collect(node);
      }
    };
    collect(root);
    assert.equal(vm.runInContext("historyKeys", h.context).every((key) => present.has(key)), true,
      "nested groups retain every canonical row identity");
    outer.remove = () => { removals++; remove(); };
    await h.context.loadHistory();
    assert.equal(removals, 0, "an unchanged history refresh never dismantles the outer disclosure");
    assert.equal(root.querySelector(".execution-group"), outer);
    assert.equal(detail.open, true);
  });
}

function sessionScrollHarness(width, withTasks) {
  const h = sessionFoldHarness(width), root = h.elements.msgs;
  const frames = new Map(), growth = new Map(), historyGates = new Map();
  const resizeObservers = [], mutationObservers = [], positionSaves = [];
  let nextFrame = 1, top = 0, scrollQueued = false;
  // Both real transcript scroll listeners run; addEventListener does not replace
  // the direction listener when wire() installs position persistence.
  const addListener = root.addEventListener;
  root.addEventListener = (type, listener) => {
    const previous = root["on" + type];
    addListener(type, (event) => {
      const results = [previous?.(event), listener(event)];
      if (results.some((result) => result?.then)) return Promise.all(results);
    });
  };
  const setTimeout = h.context.setTimeout;
  h.context.setTimeout = (fn, ms) => {
    if (ms === 200) positionSaves.push(fn);
    return setTimeout(fn, ms);
  };
  h.elements.jumpbottom = h.document.createElement("button");
  h.context.requestAnimationFrame = (fn) => {
    const id = nextFrame++;
    frames.set(id, fn);
    return id;
  };
  h.context.cancelAnimationFrame = (id) => frames.delete(id);
  h.context.window.requestAnimationFrame = h.context.requestAnimationFrame;
  const heightOf = (node) => {
    if (node.classList.contains("task-output") && !node.parentNode.querySelector("details").open) return 0;
    if (width >= 701 && node.dataset.taskCurrent === "false" && node.parentNode?.parentNode?.dataset.subagents === "true") return 0;
    let height;
    if (node.tagName === "details") height = heightOf(node.children[0]) +
      (node.open ? node.children.slice(1).reduce((sum, child) => sum + heightOf(child), 0) : 0);
    else if (node.children.length) height = node.children.reduce((sum, child) => sum + heightOf(child), 0) + 12;
    else height = Math.max(24, Math.ceil(node.textContent.length / Math.max(20, Math.floor(width / 12))) * 20);
    return height + (growth.get(node) || 0);
  };
  h.context.ResizeObserver = class {
    constructor(callback) { this.callback = callback; this.targets = new Map(); resizeObservers.push(this); }
    observe(target) { if (!this.targets.has(target)) this.targets.set(target, null); }
    unobserve(target) { this.targets.delete(target); }
    disconnect() { this.targets.clear(); }
  };
  h.context.MutationObserver = class {
    constructor(callback) { this.callback = callback; mutationObservers.push(this); }
    observe(target) { assert.equal(target, root); this.children = root.children.slice(); }
    disconnect() { this.children = null; }
  };
  root.clientHeight = 240;
  Object.defineProperty(root, "scrollHeight", {
    get: () => Math.max(root.clientHeight, 40 + root.children.reduce((sum, child) => sum + heightOf(child), 0)),
  });
  Object.defineProperty(root, "scrollTop", {
    get: () => top,
    set(value) {
      const next = Math.max(0, Math.min(Number(value), root.scrollHeight - root.clientHeight));
      if (top !== next) { top = next; scrollQueued = true; }
    },
  });
  const content = Object.getOwnPropertyDescriptor(root, "textContent");
  Object.defineProperty(root, "textContent", {
    get: content.get,
    set(value) {
      content.set(value);
      growth.clear();
      // Clearing a transcript clamps its old offset and queues a scroll event.
      root.scrollTop = top;
    },
  });
  vm.runInContext(between("const PHONE_LAYOUT_QUERY", "// The desktop composer"), h.context);
  // Jump-button drawing is synchronous here; only transcript-follow callbacks
  // use the frame queue. Persistence still uses the real wire() registration.
  h.context.scheduleJumpButtonUpdate = () => h.context.updateJumpButton();
  vm.runInContext(between("async function loadOlderHistory()", "// Historical user inputs"), h.context);
  vm.runInContext(between('$("msgs").addEventListener("scroll", () => {\n    scheduleJumpButtonUpdate();',
    'window.addEventListener("pagehide"'), h.context);
  const histories = new Map(["open-session", "target-session"].map((id, sessionIndex) => {
    const messages = [];
    for (let i = 0; i < 8 + sessionIndex * 3; i++) {
      messages.push({ role: "user", content: id + " prompt " + i });
      messages.push({ role: "assistant", content: id + " answer " + i });
    }
    if (withTasks) messages.splice(messages.length - 1, 0, {
      role: "tool", content: "Raw task result", task_report: {
        schema: "ai.task-report", version: 1, operation: "task_wait", snapshot: false,
        tasks: [{ task_id: id + "-task", agent: "build", status: "completed", active: false,
          output: ("Delivered output for " + id + "\n").repeat(12) }],
      },
    });
    return [id, messages];
  }));
  const fetch = h.context.fetch;
  h.context.fetch = async (url, options = {}) => {
    const path = new URL(url).pathname;
    const id = /^\/sessions\/(open-session|target-session)\/history$/.exec(path)?.[1];
    if (!id) return fetch(url, options);
    h.requests.push({ path, method: options.method || "GET", body: null });
    if (historyGates.has(id)) await historyGates.get(id).promise;
    const messages = histories.get(id);
    return response(messages, 200, { "x-history-total": String(messages.length) });
  };
  const flushMutations = () => {
    for (const observer of mutationObservers) {
      if (!observer.children || observer.children.length === root.children.length &&
          observer.children.every((child, index) => child === root.children[index])) continue;
      observer.children = root.children.slice();
      observer.callback([{ target: root }]);
    }
  };
  const flushResizes = () => {
    for (const observer of resizeObservers) {
      const entries = [];
      for (const [target, previous] of observer.targets) {
        const height = target === root ? root.clientHeight : heightOf(target);
        if (height === previous) continue;
        observer.targets.set(target, height);
        entries.push({ target, contentRect: { width, height } });
      }
      if (entries.length) observer.callback(entries);
    }
  };
  const flushScroll = () => {
    if (!scrollQueued) return;
    scrollQueued = false;
    h.dispatch(root, "scroll");
  };
  const flushFrame = () => {
    flushMutations();
    // Browser scroll events run before the frame's animation callbacks.
    flushScroll();
    const batch = [...frames];
    for (const [id, fn] of batch) if (frames.delete(id)) fn();
    // ResizeObserver delivers after animation callbacks. Following requested
    // by a resize therefore runs in the next frame, not this one.
    flushResizes();
  };
  return {
    ...h, root, histories, flushFrame, flushScroll,
    pendingFrames: () => frames.size,
    growLayout(height, node = root.lastElementChild) { growth.set(node, (growth.get(node) || 0) + height); },
    resizeViewport(height) { root.clientHeight = height; root.scrollTop = top; },
    resizeComposer(height) { h.elements.composer.offsetHeight = height; },
    resizeObservers: () => resizeObservers,
    queuedPositionSave: () => positionSaves.at(-1),
    deferHistory(id) {
      let resolve;
      const promise = new Promise((done) => { resolve = done; });
      historyGates.set(id, { promise });
      const release = () => { historyGates.delete(id); resolve(); };
      release.allowNewRequests = () => historyGates.delete(id);
      return release;
    },
    async selectWhileHistoryPending(id) {
      const loadHistory = h.context.loadHistory;
      let selectedLoad;
      h.context.loadHistory = (...args) => { selectedLoad = loadHistory(...args); return selectedLoad; };
      try { await h.select(id); }
      finally { h.context.loadHistory = loadHistory; }
      assert.ok(selectedLoad, "the sidebar starts the selected session's history load");
      await selectedLoad;
      assert.equal(h.context.sessionId, id);
      assert.equal(h.context.historySession, id, "the selected render finishes independently of an outgoing pager");
    },
    settleLayout() { flushFrame(); flushFrame(); flushScroll(); },
    async openHistory() {
      await h.context.loadHistory();
      flushFrame();
      flushScroll();
      assert.equal(root.scrollTop, root.scrollHeight - root.clientHeight, "the initial history opens at the live edge");
    },
  };
}

for (const width of [390, 700, 701, 1280]) for (const withTasks of [false, true]) {
  const suffix = " at " + width + "px " + (withTasks ? "with task-output bubbles" : "without task-output bubbles");
  test("session-switch scroll A to B to A handles queued layout scrolls" + suffix, async () => {
    const h = sessionScrollHarness(width, withTasks);
    await h.openHistory();
    const oldRows = h.root.children.slice();
    await h.selectHistory("target-session");
    if (withTasks) assert.equal(h.root.querySelectorAll(".task-output").length, 2,
      "the real history renderer mounts the raw task and merged subagent projections");
    assert.equal(h.root.children.includes(oldRows[0]), false);
    // Intrinsic layout can settle after loadHistory, before its follow frame.
    h.growLayout(180);
    h.flushFrame();
    h.flushScroll();
    assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight,
      "every layout follows the fully rendered history");
    await h.selectHistory("open-session");
    h.growLayout(240);
    h.flushFrame();
    h.flushScroll();
    assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight,
      "desktop returns to A's current bottom; mobile keeps its existing return-to-A behavior");
    assert.deepEqual(h.requests.filter((request) => request.path.endsWith("/history")).map((request) => request.path), [
      "/sessions/open-session/history", "/sessions/target-session/history", "/sessions/open-session/history",
    ]);
  });
  test("session-switch scroll preserves a saved reading position" + suffix, async () => {
    const h = sessionScrollHarness(width, withTasks);
    await h.openHistory();
    h.root.scrollTop = 120;
    h.flushScroll();
    await h.selectHistory("target-session");
    h.flushFrame();
    h.flushScroll();
    const saved = h.stored.get("serve.scroll.open-session");
    assert.deepEqual(JSON.parse(saved), { top: 120, atBottom: false });
    await h.selectHistory("open-session");
    h.flushFrame();
    h.flushScroll();
    assert.equal(h.root.scrollTop, 120, "an intentionally scrolled-up session keeps the existing restoration policy");
    assert.equal(h.stored.get("serve.scroll.open-session"), saved, "restoration does not rewrite the saved preference");
    h.histories.get("open-session").push({ role: "assistant", content: "An ordinary in-session update" });
    await h.context.loadHistory();
    h.flushFrame();
    h.flushScroll();
    assert.equal(h.root.scrollTop, 120, "an in-session refresh does not pull a reader to the bottom");
  });
}

for (const width of [390, 700, 701, 1280]) {
  test("session-switch scroll respects upward input before the follow frame at " + width + "px", async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await h.selectHistory("target-session");
    h.dispatch(h.root, "wheel", { deltaY: -1 });
    h.flushFrame();
    assert.equal(h.root.scrollTop, 0, "fresh upward reader intent still cancels the pending follow");
  });
  test("session-switch scroll cancels the outgoing desktop follow frame at " + width + "px", async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    h.context.followOutput(true);
    assert.equal(h.pendingFrames(), 1);
    const switched = h.selectHistory("target-session");
    assert.equal(h.pendingFrames(), 0,
      "the outgoing frame is cancelled while the new history is still fetching");
    await switched;
    h.flushFrame();
    h.flushScroll();
    assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight);
    assert.equal(h.pendingFrames(), 0, "no follow frame remains after the landing");
  });
}

async function openOtherTranscriptAtBottom(h, width) {
  await h.selectHistory("target-session");
  h.settleLayout();
  if (width <= 700) { h.context.scrollToLatest(false); h.flushScroll(); }
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight);
}

for (const width of [390, 700, 701, 1280]) {
  const desktop = width >= 701, suffix = " at " + width + "px";
  test("desktop transcript ownership ignores outgoing clamps during the history fetch" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await openOtherTranscriptAtBottom(h, width);
    const release = h.deferHistory("open-session"), selecting = h.selectHistory("open-session");
    assert.equal(h.context.sessionId, "open-session");
    assert.equal(h.context.historySession, "target-session", "B stays mounted while A is fetching");
    const before = h.root.scrollTop;
    h.resizeViewport(h.root.clientHeight + 80);
    h.flushScroll();
    assert.equal(h.root.scrollTop, before - 80, "the outgoing layout really clamps and dispatches a scroll");
    release(); await selecting; h.settleLayout();
    assert.equal(h.root.querySelectorAll(".task-output").length, 2);
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : 0,
      "desktop ignores the outgoing clamp; mobile retains its existing suspension");
  });
  test("desktop transcript ownership invalidates pending position saves on navigation" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await openOtherTranscriptAtBottom(h, width);
    h.advance(200);
    h.root.scrollTop = 120; h.flushScroll();
    const pendingSave = h.queuedPositionSave();
    assert.equal(typeof pendingSave, "function", "the real wire listener queues the position save");
    const savedA = h.stored.get("serve.scroll.open-session");
    assert.equal(JSON.parse(savedA).atBottom, true);
    const release = h.deferHistory("open-session"), selecting = h.selectHistory("open-session");
    h.advance(200);
    if (desktop) pendingSave(); // A callback already queued before cancellation must also be harmless.
    assert.equal(h.stored.get("serve.scroll.open-session"), desktop ? savedA : JSON.stringify({ top: 120, atBottom: false }),
      "only desktop rejects B's pending offset under A's storage key");
    release(); await selecting; h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : 120);
    assert.deepEqual(JSON.parse(h.stored.get("serve.scroll.target-session")), { top: 120, atBottom: false },
      "navigation still saves B's intentional reading position under B");
  });
  test("desktop transcript ownership follows child growth after the landing frame" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await openOtherTranscriptAtBottom(h, width);
    await h.selectHistory("open-session"); h.settleLayout();
    assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight);
    const before = h.root.scrollTop, child = h.root.lastElementChild;
    h.growLayout(240, child); h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : before,
      "content grows after the first follow callback, while the scrollport keeps the same size");
    assert.equal(h.resizeObservers().some((observer) => observer.targets.has(child)), desktop,
      "desktop observes content children; mobile installs no observer");
    h.context.rememberTranscriptPosition();
    assert.equal(JSON.parse(h.stored.get("serve.scroll.open-session")).atBottom, desktop,
      "late layout does not turn a desktop bottom goal into a saved reading offset");
  });
  test("desktop transcript ownership follows a viewport shrink after landing" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    const before = h.root.scrollTop;
    h.resizeViewport(h.root.clientHeight - 60); h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : before);
    assert.equal(h.resizeObservers().some((observer) => observer.targets.has(h.root)), desktop,
      "the scrollport observer handles viewport changes independently of its children");
  });
  test("a taller composer keeps the reader's row instead of following" + suffix, async () => {
    if (!desktop) return;
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    const before = h.root.scrollTop;
    h.resizeComposer(140); // a pasted screenshot adds a marker line to the prompt
    h.resizeViewport(h.root.clientHeight - 40); h.settleLayout();
    assert.equal(h.root.scrollTop, before, "the composer's own growth never moves the transcript");
  });
  test("desktop transcript ownership respects upward input during the fetch" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await openOtherTranscriptAtBottom(h, width);
    const release = h.deferHistory("open-session"), selecting = h.selectHistory("open-session");
    h.resizeViewport(h.root.clientHeight + 80); h.flushScroll();
    h.dispatch(h.root, "wheel", { deltaY: -1 });
    release(); await selecting; h.settleLayout();
    assert.equal(h.root.scrollTop, 0, "explicit upward intent still cancels A's landing");
    assert.equal(h.resizeObservers().every((observer) => observer.targets.size === 0), true);
    assert.equal(h.pendingFrames(), 0);
  });
  test("desktop transcript ownership releases following when the reader scrolls away" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    h.root.scrollTop -= 40; h.flushScroll();
    const readingTop = h.root.scrollTop;
    h.growLayout(180); h.settleLayout();
    for (const observer of h.resizeObservers()) observer.callback([]);
    h.settleLayout();
    assert.equal(h.root.scrollTop, readingTop, "late resize callbacks never move an active reader");
    assert.equal(h.resizeObservers().every((observer) => observer.targets.size === 0), true);
    h.context.rememberTranscriptPosition();
    assert.deepEqual(JSON.parse(h.stored.get("serve.scroll.open-session")), { top: readingTop, atBottom: false });
    await h.selectHistory("target-session"); h.settleLayout();
    await h.selectHistory("open-session"); h.settleLayout();
    h.growLayout(100); h.settleLayout();
    assert.equal(h.root.scrollTop, readingTop, "a restored reading position also stays free of resize following");
    assert.equal(h.resizeObservers().every((observer) => observer.targets.size === 0), true);
    h.context.scrollToLatest(false); h.flushScroll();
    const resumedTop = h.root.scrollTop;
    h.growLayout(60); h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : resumedTop,
      "the explicit jump resumes desktop following while mobile keeps its current jump behavior");
  });
  test("desktop transcript ownership observes new children and rejects obsolete resize callbacks" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    const child = h.document.createElement("div"); child.className = "msg asst"; child.textContent = "Late task output";
    const before = h.root.scrollTop;
    h.root.appendChild(child); h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : before);
    assert.equal(h.resizeObservers().some((observer) => observer.targets.has(child)), desktop);
    const obsolete = h.resizeObservers().slice();
    const release = h.deferHistory("target-session"), selecting = h.selectHistory("target-session");
    h.growLayout(120, child);
    for (const observer of obsolete) observer.callback([]);
    assert.equal(h.pendingFrames(), 0, "the outgoing generation cannot enqueue another follow callback");
    assert.equal(obsolete.every((observer) => observer.targets.size === 0), true);
    release(); await selecting; h.settleLayout();
    assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight);
  });
  test("desktop transcript ownership survives A to B to A before B finishes fetching" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory(); h.advance(200);
    h.root.scrollTop = 120; h.flushScroll();
    const pendingSave = h.queuedPositionSave(), firstRow = h.root.children[0];
    const release = h.deferHistory("target-session");
    h.context.switchSession("target-session");
    const obsoleteLoad = h.context.loadHistory();
    h.context.switchSession("open-session");
    await h.context.loadHistory(); h.settleLayout();
    if (desktop) pendingSave();
    assert.equal(h.root.children[0], firstRow, "the retained A transcript can acquire a new scroll generation");
    assert.equal(h.root.scrollTop, 120);
    assert.equal(h.stored.get("serve.scroll.target-session"), desktop ? undefined : JSON.stringify({ top: 120, atBottom: false }),
      "desktop never saves A's mounted DOM under the unmounted B session");
    release(); await obsoleteLoad; h.settleLayout();
    assert.equal(h.context.historySession, "open-session");
    assert.equal(h.root.children[0], firstRow, "B's superseded response cannot replace A");
    assert.deepEqual(JSON.parse(h.stored.get("serve.scroll.open-session")), { top: 120, atBottom: false });
  });
  test("desktop transcript ownership carries lazy session adoption into following" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    h.context.switchSession(""); await h.context.loadHistory(); h.settleLayout();
    assert.equal(h.context.adoptComposerSession("created-session", "", "https://serve.test"), true);
    h.context.scrollToLatest(false); h.flushScroll();
    const before = h.root.scrollTop;
    h.growLayout(400); h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? h.root.scrollHeight - h.root.clientHeight : before,
      "first-turn session creation keeps the same desktop following lifecycle");
    h.context.rememberTranscriptPosition();
    assert.equal(JSON.parse(h.stored.get("serve.scroll.created-session")).atBottom, desktop);
  });
  test("desktop transcript ownership ignores stale older-history pagination after returning to A" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory(); await openOtherTranscriptAtBottom(h, width);
    // Supply the count of a partial server window so the real upward-scroll
    // listener starts pagination rather than stopping at the loaded tail.
    h.context.canonicalHistoryTotal = h.histories.get("target-session").length + 50;
    const release = h.deferHistory("target-session"), loadOlderHistory = h.context.loadOlderHistory;
    let paging;
    h.context.loadOlderHistory = (...args) => {
      const completion = loadOlderHistory(...args); paging ||= completion; return completion;
    };
    h.root.scrollTop = 0; h.flushScroll();
    assert.ok(paging?.then, "scrolling up starts B's deferred older-history request");
    assert.equal(h.root._histOlder, true);
    await h.selectWhileHistoryPending("open-session"); h.settleLayout();
    const bottom = h.root.scrollHeight - h.root.clientHeight;
    assert.equal(h.root.scrollTop, bottom, "A reaches its bottom while B's pager is still pending");
    assert.equal(h.root.querySelectorAll(".task-output").length, 2);
    release(); await paging; h.settleLayout();
    assert.equal(h.context.historySession, "open-session");
    assert.equal(h.root.scrollTop, desktop ? bottom : 0,
      "only desktop rejects B's stale compensation; mobile keeps its current compensation policy");
    assert.equal(h.context.shouldFollowOutput(), desktop);
    assert.equal(vm.runInContext("outputFollowSuspended", h.context), !desktop);
    if (desktop) assert.equal(h.context.ownedTranscriptScroll(h.root).following, true);
    assert.equal(h.root._histOlder, false);
  });
  test("desktop transcript ownership keeps current older-history pagination compensation" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory(); await openOtherTranscriptAtBottom(h, width);
    h.root.scrollTop = 120; h.flushScroll();
    const beforeTop = h.root.scrollTop, beforeHeight = h.root.scrollHeight;
    h.context.canonicalHistoryTotal = h.histories.get("target-session").length + 50;
    const release = h.deferHistory("target-session"), paging = h.context.loadOlderHistory();
    assert.equal(h.root._histOlder, true);
    h.histories.get("target-session").unshift(
      { role: "user", content: "Older prompt" }, { role: "assistant", content: "Older answer" },
    );
    release(); await paging; h.settleLayout();
    assert.ok(h.root.scrollHeight > beforeHeight, "older rows add height above the existing transcript");
    const readingTop = beforeTop + h.root.scrollHeight - beforeHeight;
    assert.equal(h.root.scrollTop, readingTop, "the current owner still compensates for prepended rows");
    assert.equal(h.context.shouldFollowOutput(), false);
    h.context.rememberTranscriptPosition();
    assert.deepEqual(JSON.parse(h.stored.get("serve.scroll.target-session")), { top: readingTop, atBottom: false });
    assert.equal(h.root._histOlder, false);
  });
  test("desktop transcript ownership rejects older-history pagination from an earlier B generation" + suffix, async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory(); await openOtherTranscriptAtBottom(h, width);
    h.context.canonicalHistoryTotal = h.histories.get("target-session").length + 50;
    const firstOwner = h.context.ownedTranscriptScroll(h.root);
    const release = h.deferHistory("target-session"), loadOlderHistory = h.context.loadOlderHistory;
    let paging;
    h.context.loadOlderHistory = (...args) => {
      const completion = loadOlderHistory(...args); paging ||= completion; return completion;
    };
    h.root.scrollTop = 0; h.flushScroll();
    assert.ok(paging?.then);
    await h.selectWhileHistoryPending("open-session"); h.settleLayout();
    // The outstanding request is already waiting on its gate. Let a fresh B
    // history request run while keeping the original pager promise deferred.
    release.allowNewRequests();
    await h.selectWhileHistoryPending("target-session"); h.settleLayout();
    h.context.scrollToLatest(false); h.flushScroll();
    const bottom = h.root.scrollHeight - h.root.clientHeight;
    if (desktop) assert.notEqual(h.context.ownedTranscriptScroll(h.root).generation, firstOwner.generation);
    release(); await paging; h.settleLayout();
    assert.equal(h.root.scrollTop, desktop ? bottom : 0);
    assert.equal(h.context.shouldFollowOutput(), desktop);
  });
}

for (const width of [701, 1280]) for (const input of ["touch", "keyboard", "scrollbar"]) {
  test("desktop transcript ownership respects " + input + " input before the landing frame at " + width + "px", async () => {
    const h = sessionScrollHarness(width, true);
    await h.openHistory();
    await h.selectHistory("target-session");
    let expected = 0;
    if (input === "touch") {
      h.dispatch(h.root, "touchstart", { touches: [{ clientY: 100 }] });
      h.dispatch(h.root, "touchmove", { touches: [{ clientY: 120 }] });
    } else if (input === "keyboard") h.dispatch(h.root, "keydown", { key: "PageUp" });
    else {
      h.root.scrollTop = 100; h.flushScroll();
      h.dispatch(h.root, "pointerdown");
      h.root.scrollTop = expected = 60; h.flushScroll();
    }
    h.settleLayout();
    assert.equal(h.root.scrollTop, expected, "reader input wins over the pending desktop landing");
    assert.equal(h.resizeObservers().every((observer) => observer.targets.size === 0), true);
  });
}

for (const width of [390, 700, 701, 1280]) {
  for (const queued of [false, true]) test("session A to B to A preserves fold choices with " +
    (queued ? "queued" : "settled") + " toggles at " + width + "px", async () => {
    const h = sessionFoldHarness(width);
    await h.context.loadHistory();
    const original = h.folds();
    assert.equal(original.length, 5, "history must render thinking, tool, raw task, subagent and control folds");
    assert.deepEqual(original.map((details) => details.open), [false, false, false, false, false]);
    for (const details of original) await h.activateSummary(details);
    h.flushToggles();
    for (const details of original.slice(0, -1)) await h.activateSummary(details);
    assert.deepEqual(original.map((details) => details.open), [false, false, false, false, true],
      "A is collapsed before the sidebar switch, with one explicitly expanded control");
    if (!queued) h.flushToggles();
    await h.selectHistory("target-session");
    assert.deepEqual(h.folds().map((details) => details.open), [false, false, false, false, false],
      "B must not inherit A's fold state even with identical message identities");
    await h.activateSummary(h.folds()[0]);
    await h.selectHistory("open-session");
    const returned = h.folds();
    assert.notEqual(returned[0], original[0], "returning to A must rebuild its canonical rows");
    assert.deepEqual(returned.map((details) => details.dataset.foldKey), original.map((details) => details.dataset.foldKey));
    const expected = queued && width <= 700 ? [true, true, true, true, true] : [false, false, false, false, true];
    assert.deepEqual(returned.map((details) => details.open), expected,
      "desktop must preserve the collapsed DOM state; mobile keeps its existing queued-toggle behavior");
    assert.equal(returned.at(-1).open, true, "the explicitly expanded control must remain expanded");
    h.flushToggles();
    await h.selectHistory("target-session");
    await h.selectHistory("open-session");
    assert.deepEqual(h.folds().map((details) => details.open), expected,
      "late toggle events from detached rows must not corrupt the next desktop restore");
    assert.deepEqual(h.requests.filter((request) => request.path.endsWith("/history")).map((request) => request.path), [
      "/sessions/open-session/history", "/sessions/target-session/history", "/sessions/open-session/history",
      "/sessions/target-session/history", "/sessions/open-session/history",
    ]);
  });
  test("session A to B to A preserves a queued explicit expansion on desktop at " + width + "px", async () => {
    const h = sessionFoldHarness(width);
    await h.context.loadHistory();
    const control = h.folds().at(-1);
    await h.activateSummary(control, 0);
    assert.equal(control.open, true, "keyboard activation expands A's control before navigation");
    await h.selectHistory("target-session");
    await h.selectHistory("open-session");
    assert.deepEqual(h.folds().map((details) => details.open), [false, false, false, false, width >= 701],
      "desktop captures a pending expansion; mobile retains its prior restoration timing");
  });
}

for (const width of [701, 1280]) test("session A to B to A rejects a detached toggle after a conflicting newer collapse at " + width + "px", async () => {
  const h = sessionFoldHarness(width);
  await h.context.loadHistory();
  const obsolete = h.folds();
  for (const details of obsolete) await h.activateSummary(details);
  await h.selectHistory("target-session");
  await h.selectHistory("open-session");
  const current = h.folds();
  assert.deepEqual(current.map((details) => details.open), [true, true, true, true, true]);
  assert.notEqual(current[0], obsolete[0], "returning to A creates new fold owners");
  for (const details of current) await h.activateSummary(details);
  assert.deepEqual(current.map((details) => details.open), [false, false, false, false, false]);
  await h.selectHistory("target-session");
  const collapsedStorage = h.stored.get("serve.foldopen");
  assert.equal(collapsedStorage, "[]", "navigation snapshots A's newer collapsed choices");
  for (const details of obsolete) {
    assert.equal(details.isConnected, false);
    assert.equal(details.open, true, "the old queued event conflicts with the newer collapsed owner");
    h.deliverToggle(details);
  }
  const afterObsoleteToggle = h.stored.get("serve.foldopen");
  await h.selectHistory("open-session");
  assert.deepEqual(h.folds().map((details) => details.open), [false, false, false, false, false],
    "obsolete expanded events must not reopen any of A's newly collapsed folds");
  assert.equal(afterObsoleteToggle, collapsedStorage, "obsolete fold events must not write storage");
});

for (const width of [390, 700, 701, 1280]) test("session A to B to A keeps connected expansion and collapse toggles persistent at " + width + "px", async () => {
  const h = sessionFoldHarness(width);
  await h.context.loadHistory();
  const details = h.folds()[0], key = details.dataset.foldKey;
  assert.equal(details.isConnected, true);
  await h.activateSummary(details);
  h.deliverToggle(details);
  assert.equal(JSON.parse(h.stored.get("serve.foldopen")).includes(key), true,
    "a connected expansion must persist before any navigation snapshot");
  await h.activateSummary(details);
  h.deliverToggle(details);
  assert.equal(JSON.parse(h.stored.get("serve.foldopen")).includes(key), false,
    "a connected collapse must persist before any navigation snapshot");
  await h.selectHistory("target-session");
  await h.selectHistory("open-session");
  assert.equal(h.folds()[0].open, false);
});

for (const width of [390, 700]) test("session A to B to A preserves legacy detached toggle persistence on mobile at " + width + "px", async () => {
  const h = sessionFoldHarness(width);
  await h.context.loadHistory();
  const old = h.folds()[0], key = old.dataset.foldKey;
  await h.activateSummary(old);
  await h.selectHistory("target-session");
  assert.equal(old.isConnected, false);
  h.deliverToggle(old);
  assert.equal(JSON.parse(h.stored.get("serve.foldopen")).includes(key), true,
    "mobile must retain its existing handling of detached toggles");
  await h.selectHistory("open-session");
  assert.equal(h.folds()[0].open, true);
});

test("desktop historical tool folds use canonical roles and preserve report surfaces", () => {
  const selector = String.raw`.msg.asst[data-canonical-identity^='["tool",']:not(.git-output):not([data-subagents]):not(:has(> .task-output)):has(> details[data-fold-key*='\a ["tool",'])`;
  assert.deepEqual(cssRule(desktopCss, selector), {
    background: "transparent", "border-width": "0 0 0 2px", "border-radius": "6px", padding: "4px 12px",
  });
  assert.deepEqual(cssRule(baseCss, selector), {});
  assert.deepEqual(cssRule(phoneCss, selector), {});

  // Opaque session IDs and payload text must not supply the canonical role.
  const h = harness({ sid: 'session\n["tool",prefix' });
  loadChat(h);
  vm.runInContext(between("function taskReportOf(", "function mergeTaskReport("), h.context);
  vm.runInContext(between("const FOLD_ROLE_LABELS", "// Live rates are display-only"), h.context);
  const messages = [
    { role: "tool", content: 'output\n["internal_note",text' },
    { role: "internal_note", content: 'note\n["tool",text' },
  ];
  const { rows } = h.context.historyRowsOf(messages, messages.length, true);
  for (const row of rows) h.context.renderHistoryRow(row, h.elements.msgs);
  const [tool, note] = h.elements.msgs.children;
  assert.equal(tool.classList.contains("tool-activity"), false);
  assert.equal(tool.dataset.canonicalIdentity, rows[0].identity);
  assert.equal(note.dataset.canonicalIdentity, rows[1].identity);
  for (const [index, card] of [tool, note].entries()) {
    const key = card.querySelector("details").dataset.foldKey;
    assert.equal(key, h.context.sessionId + "\n" + rows[index].identity);
    assert.ok(key.includes('\n["tool",'), "the key substring alone cannot prove a tool role");
  }
  assert.ok(tool.dataset.canonicalIdentity.startsWith('["tool",'));
  assert.equal(note.dataset.canonicalIdentity.startsWith('["tool",'), false);
});

test("desktop user rows own a nonshrinking layout box for stored and live inputs", () => {
  const h = harness();
  loadChat(h);
  const history = h.context.historyRowsOf([{ role: "user", content: "stored question" }], 12, false);
  h.context.renderHistoryRow(history.rows[0], h.elements.msgs);
  const live = h.context.addLiveUserBubble("live question", "live question", "open-session", 12);
  const rows = h.elements.msgs.children;
  assert.equal(rows.length, 2, "each stored or pending input remains one transcript child");
  assert.ok(rows.every((row) => row.classList.contains("user-msg-row") && !row.hidden));
  assert.equal(rows[0].dataset.canonicalIndex, "11");
  assert.equal(rows[0].children[1].dataset.historyIndex, "11");
  assert.equal(rows[0].children[1].innerHTML, "stored question");
  assert.equal(live.innerHTML, "live question");
  assert.equal(live.parentNode, rows[1]);
  assert.ok([rows[0].children[1], live].every((bubble) => bubble.classList.contains("user") && !bubble.hidden));
  const row = responsiveRule(".user-msg-row");
  assert.equal(row.display, "flex", "display:contents gives the history row no layout box");
  assert.equal(row["flex-shrink"], "0", "long transcripts must scroll instead of shrinking the user row");
  assert.equal(row.width, "100%");
  assert.equal(row["min-width"], "0");
  assert.equal(row["justify-content"], "flex-end");
  const bubble = responsiveRule(".user-msg-row > .msg.user");
  assert.equal(bubble["min-width"], "0");
  assert.equal(bubble["align-self"], "auto");
  assert.equal(responsiveRule(".msg.user")["max-width"], "85%");
});

test("user rewind targets retain desktop bubble clicks and phone action clicks", async () => {
  for (const phone of [false, true]) {
    const h = harness({ phone }), rewinds = [];
    loadChat(h);
    h.context.rewindTo = (...args) => rewinds.push(args);
    const bubble = h.context.addUserBubble("rewind preview", 17);
    await h.dispatch(bubble, "click").done;
    assert.deepEqual(rewinds, [], "a bubble tap never rewinds on either layout");
    {
      const action = bubble.parentNode.children[0];
      assert.equal(action.getAttribute("aria-label"), "rewind to before this message");
      const click = h.dispatch(action, "click");
      await click.done;
      assert.equal(click.event.stopped, true);
      assert.deepEqual(rewinds, [[17, "rewind preview"]]);
    }
  }
  assert.equal(responsiveRule(".rewind-action").display, "inline-flex");
});

test("desktop side-note visibility reserves a grid column only while the turn runs", () => {
  const h = harness();
  loadChat(h);
  const button = h.elements.sidenote, composer = h.elements.composer;
  const buttonStyle = () => ({ ...responsiveRule("#sidenote"), ...(button.classList.contains("show") ? responsiveRule("#sidenote.show") : {}) });
  const composerStyle = () => ({ ...responsiveRule("#composer"), ...(composer.classList.contains("side-note-ready") ? responsiveRule("#composer.side-note-ready") : {}) });
  h.context.syncStopButton();
  assert.equal(buttonStyle().display, "none");
  assert.equal(button.getAttribute("aria-hidden"), "true");
  assert.equal(button.tabIndex, -1);
  assert.equal(composerStyle()["grid-template-columns"], "36px minmax(0,1fr) 36px");
  h.context.turns.set("open-session", {});
  h.context.syncStopButton();
  assert.equal(button.classList.contains("show"), true);
  assert.equal(composer.classList.contains("side-note-ready"), true);
  assert.equal(buttonStyle().display, "inline-flex", "a running desktop turn must expose the side-note action");
  assert.equal(button.getAttribute("aria-hidden"), "false");
  assert.equal(button.tabIndex, 0);
  assert.equal(buttonStyle()["grid-area"], "sidenote");
  assert.equal(buttonStyle().height, "36px");
  assert.equal(composerStyle()["grid-template-columns"], "36px minmax(0,1fr) 58px 36px");
  assert.equal(composerStyle()["grid-template-areas"], '"image prompt sidenote send"');
  assert.equal(responsiveRule("#prompt")["grid-area"], "prompt");
  assert.equal(responsiveRule("#prompt").width, "100%");
  assert.equal(responsiveRule("#prompt")["min-width"], "0");
  assert.equal(responsiveRule("#img")["grid-area"], "image");
  assert.equal(responsiveRule("#send")["grid-area"], "send");
  assert.equal(responsiveRule("#sidenote svg").width, "15px");
  h.context.turns.clear();
  h.context.syncStopButton();
  assert.equal(buttonStyle().display, "none");
  assert.equal(button.getAttribute("aria-hidden"), "true");
  assert.equal(button.tabIndex, -1);
  assert.equal(composerStyle()["grid-template-columns"], "36px minmax(0,1fr) 36px");
  assert.equal(composerStyle()["grid-template-areas"], '"image prompt send"');
});

test("desktop side-note clicks queue the composer text without interrupting the turn", async () => {
  const h = harness();
  loadChat(h);
  h.context.turns.set("open-session", {});
  h.context.syncStopButton();
  h.elements.prompt.value = "  steer left  ";
  h.context.persistDraft();
  const click = h.dispatch(h.elements.sidenote, "click");
  assert.equal(h.elements.sidenote.disabled, true);
  await click.done;
  assert.deepEqual(h.requests, [
    { path: "/sessions/open-session/side-notes", method: "POST", body: { text: "steer left" } },
    // The queued note is read back so the waiting list reflects server state.
    { path: "/sessions/open-session/side-notes", method: "GET", body: null },
  ]);
  assert.equal(h.context.turns.has("open-session"), true);
  assert.equal(h.elements.send.textContent, "⏹");
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.elements.sidenote.disabled, false);
  assert.equal(h.drafts.has(h.context.draftKey("open-session")), false);
  assert.equal(h.notes.at(-1), "side note queued — applies at the next model request");
});

test("desktop side-note queue lists waiting notes and clears after drain", async () => {
  const h = harness();
  loadChat(h);
  h.sideNotes.push({ from: "user", content: "steer left", ts: 1 });
  h.sideNotes.push({ from: "lead", content: "check tests", ts: 2 });
  await h.context.refreshSideNotes();
  assert.equal(h.elements.sidequeue.hidden, false);
  assert.equal(h.elements.sidequeuecount.textContent, "2 条 side note 等待发送");
  const rows = h.elements.sidequeuelist.children;
  assert.equal(rows.length, 2);
  assert.equal(rows[0].children[0].textContent, "你");
  assert.equal(rows[0].children[1].textContent, "steer left");
  assert.equal(rows[1].children[0].textContent, "lead");
  assert.equal(rows[1].children[1].textContent, "check tests");
  // The turn child drained the queue: the next sample hides the panel instead
  // of leaving stale "waiting" rows on screen.
  h.sideNotes.length = 0;
  await h.context.refreshSideNotes();
  assert.equal(h.elements.sidequeue.hidden, true);
  assert.equal(h.elements.sidequeuelist.children.length, 0);
});

test("phone user rows and side-note CSS retain their geometry and state rules", () => {
  const row = responsiveRule(".user-msg-row", true);
  assert.equal(row.display, "flex");
  assert.equal(row["align-self"], "flex-end");
  assert.equal(row["max-width"], "94%");
  assert.equal(row.gap, "7px");
  const bubble = responsiveRule(".user-msg-row > .msg.rewindable", true);
  assert.equal(bubble["max-width"], "none");
  assert.equal(bubble["min-width"], "0");
  assert.equal(bubble.cursor, "default");
  const rewind = responsiveRule(".user-msg-row > .rewind-action", true);
  assert.equal(rewind.display, "inline-flex");
  assert.equal(rewind.width, "34px");
  assert.equal(rewind.height, "34px");
  assert.equal(responsiveRule("#composer", true)["grid-template-columns"], "42px minmax(0,1fr) 42px");
  assert.equal(responsiveRule("#composer", true)["grid-template-areas"], '"image prompt send"');
  const ready = responsiveRule("#composer.side-note-ready", true);
  assert.equal(ready["grid-template-columns"], "42px minmax(0,1fr) 58px 42px");
  assert.equal(ready["grid-template-areas"], '"image prompt sidenote send"');
  assert.equal(responsiveRule("#sidenote", true).display, "none");
  const note = responsiveRule("#sidenote.show", true);
  assert.equal(note.display, "inline-flex");
  assert.equal(note["grid-area"], "sidenote");
  assert.equal(note.height, "42px");
  assert.equal(note["border-radius"], "999px");
  const prompt = responsiveRule("#prompt", true);
  assert.equal(prompt["min-height"], "40px");
  assert.equal(prompt["font-size"], "16px");
  assert.equal(prompt["max-height"], "calc(var(--app-height, 100vh) * .38)");
});

test("throughput footers wrap inside styled cards at phone and desktop widths", () => {
  for (const phone of [true, false]) {
    const rate = responsiveRule(".turn-rate", phone);
    assert.equal(rate.flex, "0 0 auto");
    assert.equal(rate["max-width"], "100%");
    assert.equal(rate["white-space"], "normal");
    assert.equal(rate["overflow-wrap"], "anywhere");
    assert.equal(rate["border-top"], undefined);
    assert.notEqual(rate.display, "none");
  }
});

test("short-thinking early repaint: real send paints and saves its first chunk's 250ms rate before thinking_done", async () => {
  const thought = "The status bar rate removal worker (task_03504bdf97f1481fba03b62252ab9ef4) is completed. Let me collect it with task_wait.";
  const h = harness({ phone: true });
  loadChat(h);
  vm.runInContext(between("function toolArgsOf(", "const FOLD_ROLE_LABELS"), h.context);
  vm.runInContext(between("function createLiveActivity(", "// The three override picks"), h.context);
  let pendingRead = null;
  const fetch = h.context.fetch;
  h.context.fetch = async (url, options) => {
    if (!new URL(url).pathname.endsWith("/turns/stream")) return fetch(url, options);
    return { ...response({}), body: { getReader: () => ({
      read: () => new Promise((resolve) => { pendingRead = resolve; }), cancel() {},
    }) } };
  };
  h.context.addAnswerCopy = () => {};
  const settle = async () => { for (let i = 0; i < 24; i++) await Promise.resolve(); };
  const frame = async (text) => {
    assert.ok(pendingRead, "the real send path did not reach its SSE reader");
    const resolve = pendingRead;
    pendingRead = null;
    resolve({ done: false, value: Buffer.from(text) });
    await settle();
  };
  assert.equal(h.stored.get("serve.rates"), undefined);
  h.elements.prompt.value = "show thinking rate";
  const pending = h.context.send();
  await settle();
  await frame('event: thinking_start\ndata: \n\nevent: thinking\ndata: ' + JSON.stringify({ text: thought }) + '\n\n');
  h.advance(50);
  const summary = h.elements.msgs.querySelector("summary");
  assert.equal(summary.textContent, "thinking… · ~31 tok @ 124 tok/s");
  h.advance(200);
  assert.equal(summary.textContent, "thinking… · ~31 tok @ 124 tok/s", "the first chunk waited for phase completion or another chunk");
  assert.equal(h.context.turns.has(h.context.sessionId), true);
  assert.doesNotMatch(h.elements.msgs.querySelector(".pending").textContent, /tok/);
  const serialized = h.stored.get("serve.rates");
  assert.ok(serialized);
  await frame('event: thinking_done\ndata: \n\n');
  assert.equal(summary.textContent, "thinking · ~31 tok @ 124 tok/s");
  assert.equal(h.stored.get("serve.rates"), serialized);
  const legacyThought = "Legacy thinking with no timing sample.";
  await frame('event: thinking_start\ndata: \n\nevent: thinking\ndata: ' + JSON.stringify({ text: legacyThought }) + '\n\nevent: thinking_done\ndata: \n\n');
  h.advance(250);
  assert.deepEqual(h.elements.msgs.querySelectorAll("summary").filter((item) => item.textContent.startsWith("thinking")).map((item) => item.textContent), [
    "thinking · ~31 tok @ 124 tok/s", "thinking · ~10 tok @ 40.0 tok/s",
  ], "a completed same-tick phase lost its 250ms rate");
  await frame('event: done\ndata: \n\n');
  await pending;
  const phase2Key = Object.keys(JSON.parse(h.stored.get("serve.rates")))
    .find((key) => !Object.hasOwn(JSON.parse(serialized), key));
  const serializedPhases = JSON.stringify({
    ...JSON.parse(serialized),
    [phase2Key]: { structured: true, textKey: h.context.rateTextKey(legacyThought), rate: "~10 tok @ 40.0 tok/s" },
  });
  assert.equal(h.stored.get("serve.rates"), serializedPhases);
  const messages = [
    { role: "assistant", content: "", reasoning_content: thought },
    { role: "assistant", content: "", reasoning_content: legacyThought },
  ];
  const refreshed = h.document.createElement("div");
  for (const row of h.context.historyRowsOf(messages, messages.length, false).rows) h.context.renderHistoryRow(row, refreshed);
  const labels = ["thinking · ~31 tok @ 124 tok/s", "thinking · ~10 tok @ 40.0 tok/s"];
  assert.deepEqual(refreshed.querySelectorAll("summary").map((item) => item.textContent), labels);
  const cold = harness({ phone: true });
  loadChat(cold);
  cold.stored.set("serve.rates", serializedPhases);
  cold.context.Date = { now() { assert.fail("cold history must not invent a timing sample"); } };
  for (const row of cold.context.historyRowsOf(messages, messages.length, false).rows) cold.context.renderHistoryRow(row, cold.elements.msgs);
  assert.deepEqual(cold.elements.msgs.querySelectorAll("summary").map((item) => item.textContent), labels);
  assert.equal(cold.stored.get("serve.rates"), serializedPhases);
});

for (const [phone, width] of [[true, 320], [true, 375], [true, 700], [false, 1280]]) {
  test((phone ? "phone" : "desktop") + " width " + width + " keeps rates on its live surface and restores card footers", async () => {
    const h = harness({ phone });
    loadChat(h);
    vm.runInContext(between("function toolArgsOf(", "const FOLD_ROLE_LABELS"), h.context);
    vm.runInContext(between("function createLiveActivity(", "// The three override picks"), h.context);
    h.context.window.innerWidth = width;
    let parent = null;
    for (const info of shellPathTo("msgs")) {
      const element = info.id === "msgs" ? h.elements.msgs : h.document.createElement(info.tag);
      element.id = info.id; element.className = info.className; element.hidden = info.hidden;
      for (const entry of info.style.split(";")) {
        const index = entry.indexOf(":");
        if (index >= 0) element.style[entry.slice(0, index).trim()] = entry.slice(index + 1).trim();
      }
      if (parent) parent.appendChild(element);
      if (info.tag === "body") h.document.body = element;
      parent = element;
    }
    let pendingRead = null;
    const fetch = h.context.fetch;
    h.context.fetch = async (url, options) => {
      if (!new URL(url).pathname.endsWith("/turns/stream")) return fetch(url, options);
      return { ...response({}), body: { getReader: () => ({
        read: () => new Promise((resolve) => { pendingRead = resolve; }),
        cancel() {},
      }) } };
    };
    h.context.addAnswerCopy = () => {};
    h.context.startStaleFollow = () => assert.fail("unexpected stale stream: " +
      JSON.stringify(h.elements.msgs.querySelectorAll(".err").map((element) => element.textContent)));
    const settle = async () => { for (let i = 0; i < 24; i++) await Promise.resolve(); };
    const frame = async (text) => {
      assert.ok(pendingRead, "the real send path did not reach its SSE reader");
      const resolve = pendingRead;
      pendingRead = null;
      resolve({ done: false, value: Buffer.from(text) });
      await settle();
    };
    h.elements.prompt.value = "phone prompt";
    const pending = h.context.send();
    await settle();
    // Several phases can share one network read before a browser paint.
    await frame('event: thinking_start\ndata: \n\nevent: thinking\ndata: {"text":"first thought"}\n\nevent: thinking_done\ndata: \n\nevent: thinking_start\ndata: \n\nevent: thinking\ndata: {"text":"second thought"}\n\nevent: thinking_done\ndata: \n\nevent: delta\ndata: {"delta":"live answer"}\n\n');
    await frame('event: rate\ndata: {"phase":"output","tokens":40,"elapsed_ms":2000}\n\n');
    // The fixture queues browser paint timers until its clock is advanced.
    h.advance(1000);
    const lines = h.elements.msgs.querySelectorAll(".turn-rate");
    assert.equal(lines.length, 1, "both layouts keep throughput in the assistant card");
    assert.equal(lines[0].textContent, "↳ speed · output 40 tok @ 20 tok/s");
    const thoughts = h.elements.msgs.querySelectorAll(".prog");
    assert.deepEqual(thoughts.map((thought) => thought.textContent), ["first thought", "second thought"]);
    h.elements.turnstatus.hidden = true;
    const status = h.elements.msgs.querySelector(".pending");
    assert.ok(status);
    assert.equal(status.textContent, "◌ Responding… · 0:01");
    assert.doesNotMatch(status.textContent, /tok/);
    assert.ok(status.classList.contains("msg") && status.classList.contains("asst"));
    const ancestors = [];
    for (let element = status; element; element = element.parentNode) {
      ancestors.push(element);
      assert.equal(element.hidden, false, "a hidden ancestor contains the main rate");
      assert.notEqual(element.id, "turnstatus");
      const selectors = [element.tagName, element.id ? "#" + element.id : "",
        ...element.className.split(/\s+/).filter(Boolean).map((name) => "." + name)].filter(Boolean);
      for (const selector of selectors) {
        const style = responsiveRule(selector, phone);
        assert.notEqual(style.display, "none", selector + " hides main throughput");
        assert.notEqual(style.visibility, "hidden", selector + " hides main throughput");
      }
      assert.notEqual(element.style.display, "none");
      assert.notEqual(element.style.visibility, "hidden");
    }
    assert.ok(ancestors.includes(h.elements.msgs));
    assert.ok(ancestors.includes(h.document.body));
    assert.equal(responsiveRule(".msg.asst", phone).background, "var(--panel)");
    assert.equal(responsiveRule(".msg.asst", phone).border, "1px solid var(--line)");
    const rate = responsiveRule(".turn-rate", phone);
    assert.equal(rate.flex, "0 0 auto");
    assert.equal(rate["max-width"], "100%");
    assert.equal(rate["white-space"], "normal");
    assert.equal(rate["overflow-wrap"], "anywhere");
    assert.equal(responsiveRule(".turn", phone)["min-width"], "0");
    assert.equal(responsiveRule("#msgs", phone)["overflow-y"], "auto");
    await frame("event: done\ndata: \n\n");
    await pending;
    const completed = h.elements.msgs.querySelectorAll(".turn-rate");
    assert.equal(completed.length, 1);
    const line = completed[0];
    assert.equal(line.textContent, "↳ speed · output 40 tok @ 20 tok/s");
    assert.ok(line.parentNode.classList.contains("msg") && line.parentNode.classList.contains("asst"));
    assert.ok(!line.parentNode.classList.contains("pending"));
    assert.equal(h.elements.msgs.querySelectorAll(".pending").length, 0);
    assert.deepEqual(h.elements.msgs.querySelectorAll(".prog"), thoughts, "completion lost a thinking body");

    // Rebuild canonical history from the same cache a refreshed page reads.
    const history = h.document.createElement("div");
    for (const [index, content, reasoning] of [[0, "", "first thought"], [1, "live answer", "second thought"]]) {
      h.context.renderHistoryRow({
        message: { role: "assistant", content }, text: content, reasoning,
        historyIndex: index, identity: "answer-" + index, key: "canonical-" + index,
      }, history);
    }
    const restored = history.querySelectorAll(".turn-rate");
    assert.equal(restored.length, 1);
    assert.equal(restored[0].textContent, line.textContent);
    assert.ok(restored[0].parentNode.classList.contains("msg"));
    assert.deepEqual(history.querySelectorAll(".prog").map((thought) => thought.textContent), ["first thought", "second thought"]);
    assert.doesNotMatch(history.textContent, /— tok\/s/);
  });
}

for (const [phone, width] of [[true, 320], [true, 375], [true, 700], [false, 1280]]) {
  test((phone ? "phone" : "desktop") + " width " + width + " displays thinking samples through the real send path and final SSE read", async () => {
    const h = harness({ phone });
    loadChat(h);
    vm.runInContext(between("function toolArgsOf(", "const FOLD_ROLE_LABELS"), h.context);
    vm.runInContext(between("function createLiveActivity(", "// The three override picks"), h.context);
    h.context.window.innerWidth = width;
    let pendingRead = null;
    const fetch = h.context.fetch;
    h.context.fetch = async (url, options) => {
      if (!new URL(url).pathname.endsWith("/turns/stream")) return fetch(url, options);
      return { ...response({}), body: { getReader: () => ({
        read: () => new Promise((resolve) => { pendingRead = resolve; }), cancel() {},
      }) } };
    };
    h.context.addAnswerCopy = () => {};
    const settle = async () => { for (let i = 0; i < 24; i++) await Promise.resolve(); };
    const frame = async (text) => {
      assert.ok(pendingRead, "the real send path did not reach its SSE reader");
      const resolve = pendingRead;
      pendingRead = null;
      resolve({ done: false, value: Buffer.from(text) });
      await settle();
    };
    h.elements.prompt.value = "phone thinking prompt";
    const pending = h.context.send();
    await settle();
    await frame('event: thinking_start\ndata: \n\nevent: thinking\ndata: {"text":"first thought","rate":{"phase":"reasoning","tokens":1647,"elapsed_ms":19107,"approximate":true}}\n\n');
    h.advance(100);
    const status = h.elements.msgs.querySelector(".pending");
    assert.equal(status.hidden, false);
    assert.equal(status.textContent, "◌ Thinking… · 0:00");
    assert.doesNotMatch(status.textContent, /tok/);
    assert.ok(status.classList.contains("msg") && status.classList.contains("asst"));
    assert.equal(h.elements.msgs.querySelector("summary").textContent, "thinking… · ~1647 tok @ 86.2 tok/s");
    // The next round ends in one network read with a terminal server sample.
    // No later thinking event or phase-clock tick can rescue its header.
    await frame('event: thinking_done\ndata: \n\nevent: tool\ndata: {"id":"lookup","name":"web_search","phase":"finished","ok":true}\n\nevent: thinking_start\ndata: \n\nevent: thinking\ndata: {"text":"second thought","rate":{"phase":"reasoning","tokens":431,"elapsed_ms":null,"approximate":true}}\n\nevent: thinking_done\ndata: \n\nevent: rate\ndata: {"phase":"reasoning","tokens":431,"elapsed_ms":7066}\n\nevent: done\ndata: \n\n');
    await pending;
    const labels = ["thinking · ~1647 tok @ 86.2 tok/s", "thinking · ~431 tok @ 61 tok/s"];
    const summaries = h.elements.msgs.querySelectorAll("summary").filter((summary) => summary.textContent.startsWith("thinking"));
    assert.deepEqual(summaries.map((summary) => summary.textContent), labels);
    for (const summary of summaries) {
      assert.equal(summary.hidden, false);
      assert.equal(summary.parentNode.parentNode.hidden, false);
      assert.notEqual(responsiveRule("summary", phone).display, "none");
      assert.ok(summary.parentNode.parentNode.classList.contains("msg"));
    }
    const history = h.document.createElement("div");
    for (const [index, reasoning] of ["first thought", "second thought"].entries()) {
      h.context.renderHistoryRow({
        message: { role: "assistant", content: "" }, text: "", reasoning,
        historyIndex: index, identity: "thought-" + index, key: "canonical-" + index,
      }, history);
    }
    assert.deepEqual(history.querySelectorAll("summary").map((summary) => summary.textContent), labels);
    assert.deepEqual(history.querySelectorAll(".prog").map((thought) => thought.textContent), ["first thought", "second thought"]);
    assert.doesNotMatch(history.textContent, /— tok\/s/);
  });
}
