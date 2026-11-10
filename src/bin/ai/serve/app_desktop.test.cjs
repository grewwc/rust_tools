// Run with: node --test src/bin/ai/serve/app_desktop.test.cjs
// Exercise the actual sidebar, draft lifecycle and send functions in a small
// DOM. Network responses are controlled; page code owns navigation and payloads.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
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
const response = (body, status = 200) => ({
  ok: status === 200, status, statusText: "Unavailable",
  headers: { get: () => null }, json: async () => plain(body),
});
const sessions = (marked = false) => [
  { id: "open-session", summary: "Open session", marked: false, size_bytes: 1024 },
  { id: "target-session", summary: "Target session", marked, size_bytes: 2048 },
];

function harness({ phone = false, sid = "open-session", marked: isMarked = false, confirmDelete = true, deleteError = "", forkError = "" } = {}) {
  const document = { activeElement: null };
  function node(tag = "div") {
    let text = "";
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
        el.children.forEach((child) => { child.parentNode = null; });
        el.children = [];
      },
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
    ["sesslist", "ul"], ["sesscount", "div"], ["sesstitle", "div"], ["sessmeta", "div"],
    ["prompt", "textarea"], ["msgs", "div"], ["promptlist", "ol"], ["promptcount", "span"], ["newbtn", "button"], ["fork", "button"],
    ["refresh", "button"], ["del", "button"], ["omodel", "select"], ["oagent", "select"], ["oeffort", "select"],
    ["composer", "div"], ["img", "button"], ["sidenote", "button"], ["send", "button"], ["turnstatus", "div"],
  ]) elements[id] = node(tag);
  for (const id of ["img", "prompt", "sidenote", "send"]) elements.composer.appendChild(elements[id]);
  elements.prompt.value = "draft in A";
  for (const match of html.matchAll(/<(div|section|button)[^>]*id="(sessionaction[^"]*)"[^>]*>([^<]*)/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    el.textContent = match[3];
  }
  for (const id of ["sessionactionstitle", "sessionactionmark", "sessionactionfork", "sessionactiondelete", "sessionactioncancel"]) {
    elements.sessionactionspanel.appendChild(elements[id]);
  }
  elements.sessionactions.appendChild(elements.sessionactionsscrim);
  elements.sessionactions.appendChild(elements.sessionactionspanel);
  document.querySelectorAll = (selector) => {
    assert.equal(selector, "#msgs .turn-interrupted");
    return elements.msgs.querySelectorAll(".turn-interrupted");
  };
  const serverItems = plain(sessions(isMarked)), stored = new Map(), requests = [], confirmations = [], messages = [], notes = [], drawers = [];
  const media = { matches: phone, listeners: [], addEventListener: (_, listener) => media.listeners.push(listener) };
  const timers = new Map();
  let nextTimer = 1, now = 1000, historyLoads = 0, configLoads = 0, grows = 0;
  const context = vm.createContext({
    document, $: (id) => elements[id], window: { matchMedia: () => media },
    store: { base: "https://serve.test", token: "test-token" },
    sessionId: sid, lastItems: plain(serverItems), pendingNew: null, lastMsgCount: 5, PAGE_ID: "test-page",
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
      if (request.path.endsWith("/side-notes")) return response({});
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
      return response(request.method === "POST" ? { id: "fresh-session" } : serverItems);
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
    lastHistoryEndsUnanswered: false, LIVE_STALL_MS: 60000, STREAM_SILENCE_MS: 60000,
    addLiveUserBubble: () => { const row = node(), bubble = node(); row.appendChild(bubble); elements.msgs.appendChild(row); return bubble; },
    addMsgMd: (_, parent) => parent.appendChild(node()),
    createLiveActivity: () => ({ label: () => "Connecting", setPhase() {}, finish() {}, anchor() {} }),
    scrollToLatest() {}, shouldFollowOutput: () => false, followOutput() {}, decorateAndLoad() {},
    formatElapsed: () => "0s", replyPaintDelay: () => 0, renderMarkdown: (text) => text,
    loadInfo: async () => {},
  });
  vm.runInContext(between("function headers(", "const esc ="), context);
  vm.runInContext(between("function sessionTitle(s)", "// Phone drawer floats"), context);
  vm.runInContext(between("function setPromptValue(value, force)", "// True while the composer"), context);
  vm.runInContext(between("const DRAFT_PREFIX =", "let questionCache ="), context);
  vm.runInContext(between("const composerDrafts = new Map();", "function closeQuestionPanel("), context);
  vm.runInContext(marked("history retry"), context);
  vm.runInContext(marked("session switch"), context);
  vm.runInContext(between("function renderSessionBar()", "// History is append-only"), context);
  vm.runInContext(marked("composer session adoption"), context);
  vm.runInContext(marked("turn session preparation"), context);
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
  return { context, document, elements, drafts, stored, requests, confirmations, messages, notes, drawers, media,
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

for (const confirmDelete of [false, true]) test("deleting a marked desktop session preserves confirmation (accept=" + confirmDelete + ")", async () => {
  const h = harness({ marked: true, confirmDelete });
  h.elements.prompt.value = "target draft";
  h.context.persistDraft("target-session");
  h.elements.prompt.value = "keep the open draft";
  h.open();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  assert.deepEqual(h.confirmations, ['delete "Target session"?']);
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.prompt.value, "keep the open draft");
  assert.equal(h.requests.some((request) => request.method === "DELETE"), confirmDelete);
  assert.equal(h.drafts.has(h.context.draftKey("target-session")), !confirmDelete);
  assert.equal(h.stored.has(h.context.draftKey("target-session")), !confirmDelete);
  if (confirmDelete) {
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

test("desktop user bubbles stay neutral but carry a border and accent rule", () => {
  const desktopBubble = responsiveRule(".msg.user");
  assert.equal(desktopBubble.background, "var(--surface-2)");
  assert.equal(desktopBubble.color, "var(--fg)");
  assert.equal(desktopBubble.border, "1px solid var(--line)");
  assert.equal(responsiveRule(".user-msg-row > .msg.user")["border-left"], "3px solid var(--acc)");
  for (const [selector, background] of [
    [":root", "#2e2e2e"],
    [':root[data-theme="light"]', "#dfdfdf"],
    [':root[data-theme="warm"]', "#e0dede"],
  ]) {
    assert.equal(responsiveRule(selector)["--surface"], background, selector + " user bubble fill");
  }
  assert.equal(responsiveRule("#sessions li.cur")["box-shadow"], "inset 3px 0 0 var(--acc), inset 0 0 0 1px var(--line)");
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
  });
  for (const [start, end] of [
    ["function textOf(", "const PHONE_LAYOUT_QUERY"],
    ["function addMsg(", "// Assistant replies render as markdown"],
    ["function historyRowsOf(", "// A session the server reports as running"],
    ["function attachUserRewind(", "async function rewindTo("],
    ["function isTurnRunning(", "// Transient note through"],
    ["async function sendSideNote(", "async function stopTurn("],
    ['$("send").onclick =', "// The dialog is the only way"],
  ]) vm.runInContext(between(start, end), h.context);
}

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
  assert.deepEqual(h.requests, [{ path: "/sessions/open-session/side-notes", method: "POST", body: { text: "steer left" } }]);
  assert.equal(h.context.turns.has("open-session"), true);
  assert.equal(h.elements.send.textContent, "⏹");
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.elements.sidenote.disabled, false);
  assert.equal(h.drafts.has(h.context.draftKey("open-session")), false);
  assert.equal(h.notes.at(-1), "side note queued — applies at the next model request");
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

test("main throughput rows wrap and retain their height at phone and desktop widths", () => {
  for (const phone of [true, false]) {
    const rate = responsiveRule(".turn-rate", phone);
    assert.equal(rate.flex, "0 0 auto");
    assert.equal(rate["max-width"], "100%");
    assert.equal(rate["white-space"], "normal");
    assert.equal(rate["overflow-wrap"], "anywhere");
    assert.notEqual(rate.display, "none");
  }
});

for (const width of [320, 375, 700]) {
  test("live main throughput at phone width " + width + " stays outside hidden status ancestors and wraps", async () => {
    const h = harness({ phone: true });
    loadChat(h);
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
    await frame('event: delta\ndata: {"delta":"live answer"}\n\n');
    await frame("data: ↳ speed · output 40 tok @ 20 tok/s\n\n");
    // The fixture queues browser paint timers until its clock is advanced.
    h.advance(1000);
    const lines = h.elements.msgs.querySelectorAll(".turn-rate");
    assert.equal(lines.length, 1);
    const line = lines[0];
    assert.equal(line.textContent, "↳ speed · output 40 tok @ 20 tok/s");
    assert.ok(line.parentNode.classList.contains("turn"));
    h.elements.turnstatus.hidden = true;
    const status = h.elements.msgs.querySelector(".pending");
    assert.ok(status);
    status.hidden = true;
    const ancestors = [];
    for (let element = line; element; element = element.parentNode) {
      ancestors.push(element);
      assert.equal(element.hidden, false, "a hidden ancestor contains the main rate");
      assert.notEqual(element.id, "turnstatus");
      assert.ok(!element.classList.contains("pending"), "the main rate was placed in the hidden phase bubble");
      const selectors = [element.tagName, element.id ? "#" + element.id : "",
        ...element.className.split(/\s+/).filter(Boolean).map((name) => "." + name)].filter(Boolean);
      for (const selector of selectors) {
        const style = responsiveRule(selector, true);
        assert.notEqual(style.display, "none", selector + " hides main throughput");
        assert.notEqual(style.visibility, "hidden", selector + " hides main throughput");
      }
      assert.notEqual(element.style.display, "none");
      assert.notEqual(element.style.visibility, "hidden");
    }
    assert.ok(ancestors.includes(h.elements.msgs));
    assert.ok(ancestors.includes(h.document.body));
    const rate = responsiveRule(".turn-rate", true);
    assert.equal(rate.flex, "0 0 auto");
    assert.equal(rate["max-width"], "100%");
    assert.equal(rate["white-space"], "normal");
    assert.equal(rate["overflow-wrap"], "anywhere");
    assert.equal(responsiveRule(".turn", true)["min-width"], "0");
    assert.equal(responsiveRule("#msgs", true)["overflow-y"], "auto");
    await frame("event: done\ndata: \n\n");
    await pending;
    assert.deepEqual(h.elements.msgs.querySelectorAll(".turn-rate"), lines);
  });
}
