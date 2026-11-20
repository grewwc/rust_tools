// Run with: node --test src/bin/ai/serve/app_mark.test.cjs
// Exercise sidebar marks and phone row actions with the real API and refresh
// functions against a small DOM and a controllable session list.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const plain = (value) => JSON.parse(JSON.stringify(value));
const sessionRows = (list) => list.querySelectorAll("[data-session-id]");
function assertUnassignedGroup(list, loadedCount) {
  const groups = list.querySelectorAll(".session-group-head");
  assert.equal(groups.length, 1);
  assert.equal(groups[0].querySelector(".session-group-name").textContent, "Unassigned");
  assert.equal(groups[0].querySelector(".session-group-toggle").title, "Unassigned · " + loadedCount + " loaded sessions");
}
function between(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page functions: " + start);
  return script.slice(a, b);
}
const response = (body, status = 200, headers = {}) => ({
  ok: status === 200, status, statusText: "Not Found",
  headers: { get: (name) => headers[String(name).toLowerCase()] ?? null },
  json: async () => plain(body),
});
function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}
const settle = () => new Promise((resolve) => setImmediate(resolve));
const sessions = (marked = false) => [
  { id: "open-session", summary: "Open session", marked: false, size_bytes: 1024 },
  { id: "target-session", summary: "Target session", marked, size_bytes: 2048 },
];
const longSessions = () => [sessions()[0], ...Array.from({ length: 100 }, (_, index) => ({
  id: "older-session-" + index, summary: "Older session " + index, marked: false, size_bytes: 1024,
})), sessions()[1]];

function harness({ phone = false, sid = "open-session", items = sessions(), cachedItems = items, groups = [], assignments = {}, storage, groupPatch, markError = "", confirmDelete = true, beforeMark, sessionsGet, activityGet, groupActivityGet, titlePost, historyGet, configGet, realNavigation = false } = {}) {
  const documentListeners = new Map(), windowListeners = new Map();
  const listen = (listeners, name, fn) => {
    if (!listeners.has(name)) listeners.set(name, []);
    listeners.get(name).push(fn);
  };
  const document = { activeElement: null, visibilityState: "hidden", addEventListener: (name, fn) => listen(documentListeners, name, fn) };
  function node(tag = "div") {
    let text = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, hidden: false, disabled: false, title: "", scrollTop: 0, eventListeners: new Map(),
      addEventListener(name, fn) { listen(el.eventListeners, name, fn); },
      get className() { return [...classes].join(" "); },
      set className(value) { classes.clear(); for (const name of value.split(/\s+/).filter(Boolean)) classes.add(name); },
      classList: {
        add: (name) => classes.add(name),
        remove: (name) => classes.delete(name),
        contains: (name) => classes.has(name),
        // The real DOM's two-argument form is what the session badges use to
        // set a state instead of flipping it.
        toggle: (name, force) => {
          const on = force === undefined ? !classes.has(name) : Boolean(force);
          if (on) classes.add(name); else classes.delete(name);
          return on;
        },
      },
      get textContent() { return text + el.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        for (const child of el.children) child.parentNode = null;
        el.children = [];
        el.scrollTop = 0;
      },
      setAttribute(key, value) { el.attrs[key] = String(value); },
      getAttribute(key) { return el.attrs[key] ?? null; },
      get lastElementChild() { return el.children.at(-1) || null; },
      appendChild(child) { child.remove(); child.parentNode = el; el.children.push(child); return child; },
      replaceChild(child, old) {
        const index = el.children.indexOf(old);
        assert.ok(index >= 0, "replace an existing child");
        child.remove(); old.parentNode = null; child.parentNode = el;
        el.children[index] = child; return old;
      },
      remove() {
        if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((child) => child !== el);
        el.parentNode = null;
      },
      matches(selector) {
        return selector.split(",").some((part) => {
          const s = part.trim();
          if (s === "[data-session-id]") return typeof el.dataset.sessionId === "string";
          return s.startsWith(".") ? classes.has(s.slice(1)) : tag === s;
        });
      },
      closest(selector) {
        for (let current = el; current; current = current.parentNode) if (current.matches(selector)) return current;
        return null;
      },
      contains(target) {
        for (let current = target; current; current = current.parentNode) if (current === el) return true;
        return false;
      },
      querySelectorAll(selector) {
        const found = [];
        const walk = (parent) => parent.children.forEach((child) => { if (child.matches(selector)) found.push(child); walk(child); });
        walk(el);
        return found;
      },
      querySelector(selector) { return el.querySelectorAll(selector)[0] || null; },
      focus() { if (!el.disabled) document.activeElement = el; },
      select() { el.selectionStart = 0; el.selectionEnd = el.value?.length || 0; },
    };
    return el;
  }
  document.createElement = node;
  const elements = { sesslist: node("ul"), sessfilter: node("input"), sesscount: node(), sesstitle: node(), sessrename: node("button"), fork: node("button"), sessmeta: node(), sesscwd: node(), msgs: node(), prompt: node("textarea") };
  elements.sessfilter.value = "";
  elements.sessfilter.hidden = /<input\b[^>]*id="sessfilter"[^>]*\shidden(?:\s|>)/.test(html);
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
  for (const id of ["sessionactionstitle", "sessionactionmeta", "sessionactionarchive", "sessionactiongroup", "sessionactionrename", "sessionactiondelete", "sessionactioncancel"]) elements.sessionactionspanel.appendChild(elements[id]);
  elements.sessionactions.appendChild(elements.sessionactionsscrim);
  elements.sessionactions.appendChild(elements.sessionactionspanel);

  const serverItems = plain(items), requests = [], switches = [], confirmations = [], draftsCleared = [], messages = [], drawerChanges = [], stored = storage || new Map();
  const media = { matches: phone, listeners: [], addEventListener(_, listener) { this.listeners.push(listener); } };
  const timers = new Map();
  let now = 1000, nextTimer = 1, historyLoads = 0, configLoads = 0;
  const context = vm.createContext({
    document, $: (id) => elements[id], AbortController,
    window: { matchMedia: () => media, addEventListener: (name, fn) => listen(windowListeners, name, fn) },
    store: { base: "https://serve.test", token: "test-token" },
    lastItems: plain(cachedItems), sessionId: sid, pendingNew: null, lastMsgCount: 5, turnRunning: true, PAGE_ID: "test-page",
    localStorage: { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    Date: { now: () => now },
    setTimeout: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, at: now + ms }); return id; },
    clearTimeout: (id) => timers.delete(id),
    fetch: async (url, options) => {
      const parsed = new URL(url);
      const request = { url, path: parsed.pathname, searchParams: parsed.searchParams, method: options.method || "GET", headers: plain(options.headers), body: options.body ? JSON.parse(options.body) : null };
      requests.push(request);
      if (request.path === "/session-groups") return response({ groups, assignments });
      if (request.path.startsWith("/session-groups/") && request.method === "PATCH") {
        if (groupPatch) return groupPatch(request);
        const group = groups.find((group) => request.path === "/session-groups/" + encodeURIComponent(group.id));
        assert.ok(group, "unexpected group target");
        Object.assign(group, request.body);
        return response(group);
      }
      if (request.path === "/session-group-activity") return groupActivityGet ? groupActivityGet(request) : response({ error: "not supported" }, 404);
      if (request.path.endsWith("/title") && request.method === "POST") {
        if (titlePost) return titlePost(request);
        const target = serverItems.find((item) => request.path === "/sessions/" + encodeURIComponent(item.id) + "/title");
        assert.ok(target, "unexpected rename target");
        target.summary = request.body.title;
        return response({});
      }
      if (request.path === "/session-activity") {
        if (activityGet) return activityGet(request);
        return response(request.body.ids.map((id) => ({ id, activity: serverItems.find((item) => item.id === id)?.activity ?? "unknown" })));
      }
      if (request.path.endsWith("/history")) return historyGet ? historyGet(request) : response([]);
      if (request.path.endsWith("/config")) return configGet ? configGet(request) : response({});
      if (request.path.endsWith("/mark")) {
        if (markError) return response({ error: markError }, 404);
        if (beforeMark) await beforeMark(request);
        const target = serverItems.find((s) => request.path === "/sessions/" + s.id + "/mark");
        assert.ok(target, "unexpected mark target");
        target.marked = request.body.marked;
        return response({ marked: target.marked });
      }
      if (request.method === "DELETE") {
        const index = serverItems.findIndex((s) => request.path === "/sessions/" + s.id);
        assert.ok(index >= 0, "unexpected delete target");
        serverItems.splice(index, 1);
        return response({});
      }
      assert.equal(request.path, "/sessions");
      if (sessionsGet) return sessionsGet(request);
      const keywords = (request.searchParams.get("query") || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
      const matching = serverItems.filter((item) => (request.searchParams.get("archived") === "include" || (request.searchParams.get("archived") === "only" ? !!item.archived : !item.archived))
        && (!request.searchParams.has("group") || (item.group_id || assignments[item.id] || "unassigned") === request.searchParams.get("group")) && (request.searchParams.get("marked") !== "true" || item.marked) && keywords.every((keyword) =>
        ((item.summary || item.first_user_prompt || "") + " " + item.id).toLowerCase().includes(keyword)));
      const ordered = request.searchParams.get("newest") === "true" ? matching.slice().reverse() : matching.slice();
      const offset = Number(request.searchParams.get("offset") || 0);
      const limit = Number(request.searchParams.get("limit") || ordered.length);
      const page = ordered.slice(offset, offset + limit);
      const include = request.searchParams.get("include");
      const selected = include && ordered.find((item) => item.id === include);
      if (offset === 0 && selected && !page.some((item) => item.id === include)) page.push(selected);
      return response(page, 200, {
        "x-sessions-total": String(ordered.length),
        "x-sessions-page-size": String(Math.min(limit, Math.max(0, ordered.length - offset))),
      });
    },
    confirm: (text) => { confirmations.push(text); return confirmDelete; },
    clearDraft: (id) => draftsCleared.push(id),
    switchSession: (id) => { switches.push(id); context.sessionId = id; },
    loadHistory: async () => { historyLoads += 1; },
    loadSessionConfig: async () => {
      configLoads += 1;
      if (realNavigation) await (await context.api("/sessions/" + context.sessionId + "/config")).json();
    },
    stopStaleFollow() {}, hideConfirm() {}, pollConfirm() {}, setDrawer: (open) => drawerChanges.push(open), resetPicks() {}, composerInUse: () => false,
    addMsg: (...args) => messages.push(args), note: (text) => messages.push(text),
  });
  vm.runInContext(between("// BEGIN execution model", "const executionModel ="), context);
  vm.runInContext(between("function headers(", "const esc ="), context);
  vm.runInContext(between("function sessionTitle(s)", "// Phone drawer floats"), context);
  vm.runInContext(between("function renderSessionBar()", "// History is append-only"), context);
  if (!realNavigation) vm.runInContext(between("function executionBase()", "function executionState("), context);
  vm.runInContext(between("let switchPollTimer = null;", "// Rebuild a compact progress card"), context);
  if (realNavigation) {
    Object.assign(context, {
      AbortController,
      historyLoadSeq: 0, historySession: sid, historyKeys: [], canonicalHistoryTotal: 0,
      pendingUserInputs: new Map(), turns: new Map(), eventSession: "", staleSession: "",
      staleBox: null, recoveredView: null, FOLD_ROLE_LABELS: {},
      persistDraft: () => stored.set("draft." + context.sessionId, elements.prompt.value),
      restoreDraft: () => { elements.prompt.value = stored.get("draft." + context.sessionId) || ""; },
      syncStopButton() {}, resetQuestionNavigator() {}, restoreSkillPick() {}, restoreTeamPick() {},
      abortEvents() {}, revokePreviews() {}, updateQuestionButton() {},
      shouldFollowOutput: () => false, followOutput() {}, isTurnRunning: () => false,
      textOf: (content) => typeof content === "string" ? content : "",
      isRuntimeSyntheticUser: () => false, historyEndsUnanswered: () => false,
      addMsgMd: (text, parent) => { const row = node(); row.textContent = text; parent.appendChild(row); },
      createTurnRateLine: () => () => {}, estimateTextTokens: () => 0,
    });
    vm.runInContext(between("const executionModel =", "// END execution model"), context);
    vm.runInContext(between("// BEGIN execution event dispatcher", "// END execution event dispatcher"), context);
    vm.runInContext(between("// BEGIN session switch", "// END session switch"), context);
    vm.runInContext(between("// History is append-only", "// Historical user inputs"), context);
    const switchSession = context.switchSession, loadHistory = context.loadHistory;
    context.switchSession = (...args) => { switches.push(args[0]); return switchSession(...args); };
    context.loadHistory = (...args) => { historyLoads += 1; return loadHistory(...args); };
  }
  elements.sessall = node("button");
  elements.sessmarked = node("button");
  for (const [id, tag] of [
    ["sessmore", "details"], ["sessmoretoggle", "summary"], ["sessmorepanel", "div"], ["sesssearch", "button"], ["sessviewmode", "button"],
    ["sessfilterstatus", "span"], ["sessmarkfilter", "div"], ["sessactive", "button"], ["sessarchived", "button"],
    ["sessiongroupnew", "button"], ["sessiongroupmanage", "button"], ["sessiongrouphidden", "button"], ["sessiongrouphiddenactivity", "button"], ["sessiongroupdialog", "div"],
    ["sessiongroupscrim", "div"], ["sessiongrouppanel", "section"], ["sessiongrouptitle", "div"],
    ["sessiongroupbody", "div"], ["sessiongrouperror", "div"], ["sessiongroupcancel", "button"],
  ]) elements[id] = node(tag);
  elements.sessmore.open = false;
  elements.sessiongroupdialog.hidden = true;
  elements.sessmore.appendChild(elements.sessmoretoggle);
  elements.sessmore.appendChild(elements.sessmorepanel);
  for (const id of ["sesssearch", "sessviewmode", "sessactive", "sessarchived", "sessall", "sessmarked", "sessiongroupnew", "sessiongroupmanage", "sessiongrouphidden"]) elements.sessmorepanel.appendChild(elements[id]);
  for (const id of ["sessiongrouptitle", "sessiongroupbody", "sessiongrouperror", "sessiongroupcancel"]) elements.sessiongrouppanel.appendChild(elements[id]);
  vm.runInContext(between('  wireSessionSearch();', '$("connbtn").onclick ='), context);
  context.wireSessionActions();
  context.renderSessionList(context.lastItems);
  context.renderSessionBar();

  function dispatch(target, type, extra = {}) {
    const event = { target, ...extra, stopped: false, prevented: false, stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; } };
    const pending = [];
    if (!(type === "click" && target.disabled)) {
      for (let current = target; current; current = current.parentNode) {
        const result = current["on" + type]?.(event);
        if (result && typeof result.then === "function") pending.push(result);
        for (const listener of current.eventListeners.get(type) || []) {
          const result = listener(event);
          if (result && typeof result.then === "function") pending.push(result);
        }
        if (event.stopped) break;
      }
    }
    return { event, done: Promise.all(pending) };
  }
  const row = (id = "target-session") => elements.sesslist.children.find((li) => li.dataset.sessionId === id);
  function advance(ms) {
    now += ms;
    for (const [id, timer] of [...timers]) if (timer.at <= now) { timers.delete(id); timer.fn(); }
  }
  const touch = (extra = {}) => ({ pointerType: "touch", isPrimary: true, button: 0, clientX: 10, clientY: 20, ...extra });
  function hold(id) { const li = row(id); dispatch(li, "pointerdown", touch()); advance(650); dispatch(li, "pointerup", touch()); return li; }
  return { context, document, documentListeners, windowListeners, elements, requests, switches, confirmations, draftsCleared, messages, drawerChanges, stored, media, timers, row, advance, touch, hold, dispatch, historyLoads: () => historyLoads, configLoads: () => configLoads };
}

function resizeHarness(savedWidth) {
  const h = harness(), properties = new Map(), layoutCalls = [];
  h.document.documentElement = h.document.createElement();
  h.document.documentElement.style = { setProperty: (key, value) => properties.set(key, value) };
  h.document.body = h.document.createElement();
  h.elements.views = h.document.createElement();
  h.elements.views.clientWidth = 1200;
  h.elements.sidebarresize = h.document.createElement();
  const handle = h.elements.sidebarresize, captured = new Set();
  handle.setPointerCapture = (id) => captured.add(id);
  handle.hasPointerCapture = (id) => captured.has(id);
  handle.releasePointerCapture = (id) => { captured.delete(id); h.dispatch(handle, "lostpointercapture", { pointerId: id }); };
  Object.assign(h.context, {
    requestAnimationFrame: (fn) => fn(),
    growPrompt: () => layoutCalls.push("composer"),
    updateJumpButton: () => layoutCalls.push("jump"),
    updateQuestionButton: () => layoutCalls.push("questions"),
  });
  h.context.window.innerWidth = 1200;
  if (savedWidth !== undefined) h.stored.set("serve.sidebarWidth", savedWidth);
  vm.runInContext(between("// BEGIN sidebar resize", "// END sidebar resize"), h.context);
  h.context.wireSidebarResize();
  const pointer = (clientX, pointerId = 7) => ({ pointerId, clientX, pointerType: "mouse", isPrimary: true, button: 0 });
  const width = () => Number(handle.getAttribute("aria-valuenow"));
  const windowEvent = (type) => { for (const listener of h.windowListeners.get(type) || []) listener(); };
  return { ...h, handle, captured, properties, layoutCalls, pointer, width, windowEvent };
}

test("desktop splitter controls only desktop width and exposes an accessible separator", () => {
  assert.match(html, /<\/nav>\s*<div id="sidebarresize" role="separator" tabindex="0"[^>]*aria-orientation="vertical" aria-controls="sessions"[^>]*>\s*<\/div>\s*<main id="chat">/);
  assert.match(css, /#sidebarresize \{ display:none; \}/);
  const desktop = css.slice(css.indexOf("@media (min-width:701px)"), css.indexOf("@media (max-width:700px)"));
  assert.match(desktop, /#sessions \{[^}]*flex:0 0 var\(--sidebar-width,264px\); width:var\(--sidebar-width,264px\);/);
  assert.match(desktop, /#sidebarresize \{ display:block;[^}]*flex:0 0 8px; cursor:col-resize; touch-action:none;/);
  assert.match(desktop, /#sidebarresize:hover::after, #sidebarresize:focus-visible::after, body\.sidebar-resizing #sidebarresize::after/);
  assert.match(desktop, /body\.sidebar-resizing \* \{[^}]*user-select:none !important;/);
  assert.match(css, /#sessions \{ position:absolute;[^}]*width:min\(86vw,320px\);/);
  assert.match(between("function wire()", "function registerAppServiceWorker()"), /wireSidebarResize\(\);/);
});

for (const [saved, expected] of [[undefined, 264], ["312", 312], ["100", 220], ["900", 480], ["garbage", 264], ["Infinity", 264], ["", 264]]) {
  test("sidebar restores and clamps stored width: " + String(saved), () => {
    const h = resizeHarness(saved);
    assert.equal(h.width(), expected);
    assert.equal(h.properties.get("--sidebar-width"), expected + "px");
    assert.equal(h.handle.getAttribute("aria-valuemin"), "220");
    assert.equal(h.handle.getAttribute("aria-valuemax"), "480");
    assert.equal(h.handle.getAttribute("aria-valuetext"), expected + " pixels");
    assert.match(h.handle.title, new RegExp("^" + expected + "px"));
  });
}

test("pointer drag clamps both bounds, captures the pointer and persists only after release", () => {
  const h = resizeHarness(), down = h.dispatch(h.handle, "pointerdown", h.pointer(264));
  assert.equal(down.event.prevented, true);
  assert.equal(down.event.stopped, true);
  assert.equal(h.captured.has(7), true);
  assert.equal(h.document.body.classList.contains("sidebar-resizing"), true);
  h.dispatch(h.handle, "pointermove", h.pointer(1000, 8));
  assert.equal(h.width(), 264, "an unrelated pointer cannot change width");
  h.dispatch(h.handle, "pointermove", h.pointer(1000));
  assert.equal(h.width(), 480);
  h.dispatch(h.handle, "pointermove", h.pointer(-100));
  assert.equal(h.width(), 220);
  assert.equal(h.stored.has("serve.sidebarWidth"), false);
  const up = h.dispatch(h.handle, "pointerup", h.pointer(350));
  assert.equal(up.event.prevented, true);
  assert.equal(up.event.stopped, true);
  assert.equal(h.width(), 350, "the final pointer position is applied before release");
  assert.equal(h.stored.get("serve.sidebarWidth"), "350");
  assert.equal(h.captured.size, 0);
  assert.equal(h.document.body.classList.contains("sidebar-resizing"), false);
  assert.deepEqual(h.layoutCalls, ["composer", "jump", "questions"]);
  assert.equal(h.dispatch(h.handle, "click").event.stopped, true);
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.requests, []);
});

for (const ending of ["pointercancel", "lostpointercapture", "blur", "resize", "phone"]) test("sidebar drag cleans up when interrupted by " + ending, () => {
  const h = resizeHarness();
  h.dispatch(h.handle, "pointerdown", h.pointer(264));
  h.dispatch(h.handle, "pointermove", h.pointer(320));
  if (ending === "phone") { h.media.matches = true; for (const listener of h.media.listeners) listener(); }
  else if (["blur", "resize"].includes(ending)) h.windowEvent(ending);
  else h.dispatch(h.handle, ending, h.pointer(320));
  assert.equal(h.captured.size, 0);
  assert.equal(h.document.body.classList.contains("sidebar-resizing"), false);
  assert.equal(h.stored.get("serve.sidebarWidth"), "320");
  h.dispatch(h.handle, "pointermove", h.pointer(400));
  assert.equal(h.width(), 320);
  if (ending === "phone") assert.deepEqual(h.layoutCalls, []);
});

test("narrow desktops reserve chat space and restore the preferred width after expanding", () => {
  const h = resizeHarness("480");
  h.elements.views.clientWidth = 701;
  h.windowEvent("resize");
  assert.equal(h.width(), 333);
  assert.equal(h.handle.getAttribute("aria-valuemax"), "333");
  assert.equal(h.stored.get("serve.sidebarWidth"), "480", "viewport clamping preserves the saved preference");
  h.dispatch(h.handle, "pointerdown", h.pointer(333));
  h.dispatch(h.handle, "pointerup", h.pointer(900));
  assert.equal(h.width(), 333);
  assert.equal(h.stored.get("serve.sidebarWidth"), "333");
  h.elements.views.clientWidth = 1200;
  h.windowEvent("resize");
  assert.equal(h.width(), 333);
  const restored = resizeHarness("480");
  restored.elements.views.clientWidth = 701;
  restored.windowEvent("resize");
  restored.elements.views.clientWidth = 1200;
  restored.windowEvent("resize");
  assert.equal(restored.width(), 480);
});

test("keyboard adjustment, endpoints and both reset actions persist their effective width", () => {
  const h = resizeHarness("350");
  for (const [key, shiftKey, expected] of [["ArrowLeft", false, 340], ["ArrowRight", true, 380], ["Home", false, 220], ["End", false, 480], ["Enter", false, 264]]) {
    const event = h.dispatch(h.handle, "keydown", { key, shiftKey }).event;
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    assert.equal(h.width(), expected);
    assert.equal(h.stored.get("serve.sidebarWidth"), String(expected));
  }
  h.dispatch(h.handle, "keydown", { key: "End" });
  h.dispatch(h.handle, "dblclick");
  assert.equal(h.width(), 264);
  assert.equal(resizeHarness(h.stored.get("serve.sidebarWidth")).width(), 264);
  assert.equal(h.dispatch(h.handle, "keydown", { key: "Tab" }).event.prevented, false);
  assert.deepEqual(h.switches, []);
});

test("phone and non-primary pointer interactions cannot resize, and wiring is idempotent", () => {
  const h = resizeHarness("350");
  for (const extra of [{ button: 2 }, { isPrimary: false }]) h.dispatch(h.handle, "pointerdown", { ...h.pointer(350), ...extra });
  assert.equal(h.captured.size, 0);
  h.media.matches = true;
  h.dispatch(h.handle, "pointerdown", h.pointer(350));
  h.dispatch(h.handle, "pointermove", h.pointer(450));
  h.dispatch(h.handle, "keydown", { key: "Home" });
  h.dispatch(h.handle, "dblclick");
  assert.equal(h.width(), 350);
  assert.equal(h.captured.size, 0);
  assert.deepEqual(h.layoutCalls, []);
  const listeners = h.windowListeners.get("resize").length;
  h.context.wireSidebarResize();
  assert.equal(h.windowListeners.get("resize").length, listeners);
  assert.deepEqual(h.drawerChanges, []);
});

test("sidebar stars reflect marked state and expose the matching accessible action", () => {
  const h = harness({ items: sessions(true) });
  for (const [id, marked] of [["open-session", false], ["target-session", true]]) {
    const li = h.row(id), star = li.querySelector(".mk");
    assert.equal(li.querySelectorAll(".mk").length, 1);
    assert.equal(star.type, "button");
    assert.equal(star.textContent, marked ? "★" : "☆");
    assert.equal(star.getAttribute("aria-pressed"), String(marked));
    assert.equal(star.getAttribute("aria-label"), marked ? "取消标记" : "标记会话");
    assert.equal(star.title, star.getAttribute("aria-label"));
    assert.equal(li.children[0], li.querySelector(".session-controls"));
    assert.deepEqual(li.children[0].children, [star, li.querySelector(".session-menu")]);
    assert.equal(li.children[1], li.querySelector(".session-title"));
    assert.equal(li.querySelector(".rn"), null);
  }
});

test("a star marks its session through the authenticated API and updates the cached rail without switching", async () => {
  const h = harness(), star = h.row().querySelector(".mk");
  const click = h.dispatch(star, "click");
  assert.equal(click.event.stopped, true);
  await click.done;
  assert.deepEqual(h.requests.map(({ path, method, body }) => ({ path, method, body })), [
    { path: "/sessions/target-session/mark", method: "POST", body: { marked: true } },
  ]);
  assert.equal(h.requests[0].headers.Authorization, "Bearer test-token");
  assert.equal(h.requests[0].headers["Content-Type"], "application/json");
  assert.equal(h.row().querySelector(".mk").textContent, "★");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
});

for (const marked of [false, true]) test("desktop " + (marked ? "unmarking" : "marking") + " a middle session keeps the rendered rail order", async () => {
  const items = [...sessions(marked), { id: "older-session", summary: "Older session", marked: false, size_bytes: 1024 }];
  const h = harness({
    phone: false, items: [...items].reverse(), cachedItems: items,
    // Persisting a mark makes the target the newest item in store order.
    sessionsGet: () => response([items[2], items[0], { ...items[1], marked: !marked }]),
  });
  const rowOrder = () => sessionRows(h.elements.sesslist).map((li) => li.querySelector(".session-title").textContent);
  const expectedOrder = ["Open session", "Target session", "Older session"];
  assert.deepEqual(rowOrder(), expectedOrder);
  assertUnassignedGroup(h.elements.sesslist, 3);

  await h.dispatch(h.row().querySelector(".mk"), "click").done;

  assert.deepEqual(rowOrder(), expectedOrder, "mark changes must keep every rendered row in place");
  assertUnassignedGroup(h.elements.sesslist, 3);
  const star = h.row().querySelector(".mk");
  assert.equal(star.textContent, marked ? "☆" : "★");
  assert.equal(star.getAttribute("aria-pressed"), String(!marked));
  assert.equal(star.getAttribute("aria-label"), marked ? "标记会话" : "取消标记");
  assert.equal(h.row("open-session").querySelector(".mk").getAttribute("aria-pressed"), "false");
  assert.equal(h.row("older-session").querySelector(".mk").getAttribute("aria-pressed"), "false");
  assert.deepEqual(h.requests.map(({ path, method, body }) => ({ path, method, body })), [
    { path: "/sessions/target-session/mark", method: "POST", body: { marked: !marked } },
  ]);
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
});

test("unmarking the current session refreshes the existing session-bar star", async () => {
  const items = sessions();
  items[0].marked = true;
  const h = harness({ items });
  assert.equal(h.elements.sesstitle.textContent, "★ Open session");
  await h.dispatch(h.row("open-session").querySelector(".mk"), "click").done;
  assert.deepEqual(h.requests[0].body, { marked: false });
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.equal(h.row("open-session").querySelector(".mk").getAttribute("aria-pressed"), "false");
  assert.deepEqual(h.switches, []);
});

test("unmarking the selected star refreshes the filtered total while preserving its chat and metadata", async () => {
  const items = sessions();
  items[0].marked = true;
  items[0].cwd = "/tmp/selected-project";
  const h = harness({ items });
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(h.elements.sesscount.textContent, "1 sessions · showing 1");
  assert.equal(h.elements.sesstitle.textContent, "★ Open session");
  await h.dispatch(h.row("open-session").querySelector(".mk"), "click").done;
  assert.equal(h.elements.sesscount.textContent, "0 sessions · showing 0");
  assert.equal(h.elements.sesslist.textContent, "暂无星标会话");
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.equal(h.elements.sesscwd.textContent, "📁 /tmp/selected-project");
  assert.equal(h.context.sessionListItem("open-session").marked, false);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.context.turnRunning, true);
  assert.equal(h.context.lastMsgCount, 5);
  assert.equal(h.elements.prompt.value, "draft in A");
  assert.deepEqual(h.switches, []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
  assert.ok(h.requests.filter((request) => request.path === "/sessions")
    .every((request) => request.searchParams.get("marked") === "true" && !request.searchParams.has("include")));
});

for (const marked of [false, true]) test("a failed " + (marked ? "unmark" : "mark") + " keeps the prior state and displays a sidebar error", async () => {
  const h = harness({ items: sessions(marked), markError: "session not found" });
  await h.dispatch(h.row().querySelector(".mk"), "click").done;
  assert.equal(h.requests.length, 1, "a failed mark must not refresh away the error");
  assert.deepEqual(h.requests[0].body, { marked: !marked });
  assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(marked));
  assert.equal(h.row().querySelector(".mk").textContent, marked ? "★" : "☆");
  assert.match(h.elements.sesslist.children.at(-1).textContent, /mark failed: 404 session not found/);
  assert.equal(h.context.lastItems.find((s) => s.id === "target-session").marked, marked);
  assert.deepEqual(h.switches, []);
});

test("a pending mark leaves the old star visible until the server response", async () => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  const h = harness({ beforeMark: () => pending }), star = h.row().querySelector(".mk");
  const click = h.dispatch(star, "click");
  assert.equal(star.disabled, true);
  assert.equal(star.textContent, "☆");
  assert.equal(star.getAttribute("aria-pressed"), "false");
  assert.equal(h.requests.length, 1);
  release();
  await click.done;
  assert.equal(h.row().querySelector(".mk").textContent, "★");
  assert.equal(h.row().querySelector(".mk").disabled, false);
});

test("phone star pointer and click events stop propagation without starting a hold or switching", async () => {
  const h = harness({ phone: true }), li = h.row(), star = li.querySelector(".mk");
  const down = h.dispatch(star, "pointerdown", h.touch());
  assert.equal(down.event.stopped, true);
  assert.equal(h.timers.size, 0);
  h.advance(650);
  assert.equal(li.classList.contains("holding"), false);
  assert.equal(h.elements.sessionactions.hidden, true);
  const click = h.dispatch(star, "click");
  assert.equal(click.event.stopped, true);
  await click.done;
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.confirmations, []);
  assert.equal(h.requests[0].path, "/sessions/target-session/mark");
});

test("a phone hold opens the action sheet at 650ms and suppresses its row click", async () => {
  const h = harness({ phone: true }), li = h.row();
  h.dispatch(li, "pointerdown", h.touch());
  assert.equal(li.classList.contains("holding"), true);
  h.advance(649);
  assert.equal(h.elements.sessionactions.hidden, true);
  h.advance(1);
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(li.classList.contains("holding"), false);
  assert.equal(h.elements.sessionactionstitle.textContent, "Target session");
  assert.equal(h.elements.sessionactionmark, undefined);
  assert.equal(h.elements.sessionactiondelete.textContent, "删除会话");
  assert.equal(h.elements.sessionactioncancel.textContent, "取消");
  assert.equal(h.context.sessionHoldConsumed(li), true);
  await h.dispatch(li, "click").done;
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.confirmations, []);
});

for (const marked of [false, true]) test("the phone card offers " + (marked ? "unmark" : "mark") + " and toggles only the held session", async () => {
  const h = harness({ phone: true, items: sessions(marked) });
  h.hold();
  h.dispatch(h.elements.sessionactioncancel, "click");
  await h.dispatch(h.row().querySelector(".mk"), "click").done;
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.deepEqual(h.requests[0].body, { marked: !marked });
  assert.equal(h.requests.length, 1);
  assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(!marked));
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
});

test("the sheet delete entry confirms and deletes its target while keeping the open session", async () => {
  const h = harness({ phone: true });
  h.hold();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  assert.equal(h.elements.sessdel.hidden, false);
  assert.match(h.elements.sessdeltext.textContent, /Target session/);
  await h.dispatch(h.elements.sessdelconfirm, "click").done;
  assert.deepEqual(h.requests.map(({ path, method }) => ({ path, method })), [
    { path: "/sessions/target-session", method: "DELETE" }, { path: "/sessions", method: "GET" },
  ]);
  assert.deepEqual(h.draftsCleared, ["target-session"]);
  assert.equal(h.row(), undefined);
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
  assert.equal(h.historyLoads(), 0);
});

test("declining the sheet delete confirmation preserves the session", async () => {
  const h = harness({ phone: true });
  h.hold();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  await h.dispatch(h.elements.sessdelcancel, "click").done;
  assert.equal(h.elements.sessdel.hidden, true);
  assert.ok(h.row());
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.switches, []);
});

test("cancel and the backdrop dismiss the sheet without marking, deleting or switching", async () => {
  for (const id of ["sessionactioncancel", "sessionactionsscrim"]) {
    const h = harness({ phone: true }), star = h.row().querySelector(".mk");
    h.hold();
    await h.dispatch(h.elements[id], "click").done;
    assert.equal(h.elements.sessionactions.hidden, true);
    assert.equal(h.document.activeElement, star);
    assert.deepEqual(h.requests, []);
    assert.deepEqual(h.switches, []);
  }
});

test("movement, pointer cancellation and short touches cancel the phone hold", () => {
  for (const [type, extra] of [["pointermove", { clientY: 33 }], ["pointercancel", {}], ["pointerup", {}]]) {
    const h = harness({ phone: true }), li = h.row();
    h.dispatch(li, "pointerdown", h.touch());
    h.dispatch(li, type, h.touch(extra));
    h.advance(650);
    assert.equal(h.timers.size, 0);
    assert.equal(li.classList.contains("holding"), false);
    assert.equal(h.elements.sessionactions.hidden, true);
    assert.deepEqual(h.requests, []);
    assert.deepEqual(h.switches, []);
  }
});

test("desktop, mouse and row controls do not open the phone action sheet", () => {
  for (const [phone, extra] of [[false, {}], [true, { pointerType: "mouse" }], [true, { isPrimary: false }], [true, { button: 1 }]]) {
    const h = harness({ phone });
    h.dispatch(h.row(), "pointerdown", h.touch(extra));
    h.advance(650);
    assert.equal(h.timers.size, 0);
    assert.equal(h.elements.sessionactions.hidden, true);
  }
  const h = harness({ phone: true });
  h.dispatch(h.row().querySelector(".mk"), "pointerdown", h.touch());
  h.advance(650);
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.timers.size, 0);
});

for (const phone of [false, true]) test((phone ? "phone" : "desktop") + " title and row clicks select the session without marking", async () => {
  for (const titleClick of [true, false]) {
    const h = harness({ phone });
    const click = h.dispatch(titleClick ? h.row().querySelector("span") : h.row(), "click");
    assert.equal(click.event.stopped, titleClick);
    assert.equal(h.context.sessionId, "target-session");
    assert.equal(h.row().classList.contains("cur"), true, "selection is highlighted before any fetch completes");
    assert.equal(h.row("open-session").classList.contains("cur"), false);
    assert.equal(h.elements.sesstitle.textContent, "Target session");
    assert.doesNotMatch(h.elements.sessmeta.textContent, /5 msgs/);
    await click.done;
    assert.deepEqual(h.switches, ["target-session"]);
    assert.equal(h.historyLoads(), 1);
    assert.equal(h.configLoads(), 1);
    assert.equal(h.requests.some((request) => request.path === "/sessions"), false);
    assert.equal(h.requests.some((request) => request.path.endsWith("/mark")), false);
    assert.deepEqual(h.drawerChanges, phone ? [false] : []);
  }
});

for (const phone of [false, true]) test((phone ? "phone" : "desktop") + " session titles support Enter and Space navigation", async () => {
  for (const key of ["Enter", " "]) {
    const h = harness({ phone }), title = h.row().querySelector("span");
    assert.equal(title.getAttribute("role"), "button");
    assert.equal(title.getAttribute("tabindex"), "0");
    assert.equal(title.getAttribute("aria-label"), "switch to Target session");
    assert.equal(title.getAttribute("aria-pressed"), null);
    assert.equal(title.getAttribute("aria-current"), "false");
    const press = h.dispatch(title, "keydown", { key });
    assert.equal(press.event.prevented, true);
    assert.equal(press.event.stopped, true);
    assert.equal(title.getAttribute("aria-current"), "true");
    assert.equal(h.row("open-session").querySelector("span").getAttribute("aria-current"), "false");
    await press.done;
    assert.equal(h.context.sessionId, "target-session");
    assert.equal(h.requests.some((request) => request.path === "/sessions"), false);
    assert.equal(h.requests.some((request) => request.path.endsWith("/mark")), false);
  }
});

for (const phone of [false, true]) test((phone ? "phone" : "desktop") + " switching from a deep session rail keeps its rows and scroll without a list request", async () => {
  for (const titleClick of phone ? [true, false] : [false]) {
    const h = harness({ phone, realNavigation: true, items: longSessions(), historyGet: () => response([{ role: "assistant", content: "Target history" }]) });
    const rail = h.elements.sesslist, rows = rail.children, target = h.row();
    rail.scrollTop = 1600;
    await h.dispatch(titleClick ? target.querySelector("span") : target, "click").done;
    await settle();
    assert.deepEqual(h.requests.map((request) => request.path), [
      "/sessions/target-session/history", "/sessions/target-session/config",
    ]);
    assert.equal(rail.children, rows, "switching must preserve the mounted list");
    assert.equal(h.row(), target);
    assert.equal(rail.scrollTop, 1600);
    assert.equal(target.classList.contains("cur"), true);
    assert.equal(target.querySelector("span").getAttribute("aria-current"), "true");
    assert.equal(h.row("open-session").classList.contains("cur"), false);
    assert.equal(h.elements.sesstitle.textContent, "Target session");
    assert.equal(h.context.historySession, "target-session");
    assert.equal(h.elements.msgs.textContent, "Target history");
    assert.equal(h.historyLoads(), 1);
    assert.equal(h.configLoads(), 1);
    assert.deepEqual(h.drawerChanges, phone ? [false] : []);
  }
});

test("session-list re-renders preserve the reader's rail scroll position", () => {
  const h = harness({ items: longSessions() }), rail = h.elements.sesslist;
  for (const errText of [undefined, "sessions refresh failed"]) {
    const rows = rail.children;
    rail.scrollTop = 1600;
    h.context.renderSessionList(h.context.lastItems, errText);
    assert.notEqual(rail.children, rows, "the regression must exercise a list rebuild");
    assert.equal(rail.scrollTop, 1600);
    assert.ok(h.row());
    assert.equal(h.row("open-session").classList.contains("cur"), true);
    if (errText) assert.equal(rail.children.at(-1).textContent, errText);
  }
});

test("filter input resets the session rail while normal re-renders preserve scroll", async () => {
  const h = harness({ items: longSessions() }), rail = h.elements.sesslist, filter = h.elements.sessfilter;
  rail.scrollTop = 1600;
  filter.value = "Older session";
  await h.dispatch(filter, "input").done;
  assert.equal(h.requests.length, 0, "search is debounced");
  h.advance(250);
  await settle();
  assert.equal(rail.scrollTop, 0);
  assert.equal(rail.children.length, 13);
  assert.equal(h.elements.sesscount.textContent, "100 matches · showing 12 loaded");
  assert.equal(h.row(), undefined);

  rail.scrollTop = 800;
  h.context.renderSessionList(h.context.lastItems);
  assert.equal(rail.scrollTop, 800, "a normal refresh must preserve scroll even with an active filter");
  assert.equal(rail.children.length, 13);

  filter.value = "target";
  await h.dispatch(filter, "input").done;
  h.advance(250);
  await settle();
  assert.equal(rail.scrollTop, 0);
  assert.equal(rail.children.length, 1);
  assert.ok(h.row());
  assert.equal(h.elements.sesscount.textContent, "1 matches · showing 1");

  filter.value = "";
  await h.dispatch(filter, "input").done;
  h.advance(250);
  await settle();
  assert.equal(rail.scrollTop, 0);
  assert.equal(sessionRows(rail).length, 13, "normal list includes the selected session outside the first page");
  assertUnassignedGroup(rail, 13);
  assert.equal(rail.querySelectorAll(".session-more").length, 1);
  assert.equal(h.elements.sesscount.textContent, "102 sessions");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.requests.filter((request) => request.path === "/sessions").map((request) => request.searchParams.get("query")), ["Older session", "target", null]);
});

test("a fresh session-list render starts at the top", () => {
  const h = harness({ cachedItems: [] }), rail = h.elements.sesslist;
  assert.equal(sessionRows(rail).length, 0);
  assertUnassignedGroup(rail, 0);
  assert.equal(rail.scrollTop, 0);
  h.context.renderSessionList(longSessions());
  assert.equal(sessionRows(rail).length, 102);
  assertUnassignedGroup(rail, 102);
  assert.equal(rail.scrollTop, 0);
});

test("a phone title click after a hold cannot navigate away from the action sheet", async () => {
  const h = harness({ phone: true }), li = h.hold();
  await h.dispatch(li.querySelector("span"), "click").done;
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.requests, []);
});

test("desktop row body navigation requests B history and rejects a late A response", async () => {
  const a = deferred(), b = deferred();
  const h = harness({ realNavigation: true, historyGet: (request) => request.path === "/sessions/open-session/history" ? a.promise : b.promise });
  const oldHistory = h.context.loadHistory();
  const click = h.dispatch(h.row(), "click");
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.row().classList.contains("cur"), true);
  assert.equal(h.elements.sesstitle.textContent, "Target session");
  assert.equal(h.stored.get("draft.open-session"), "draft in A");
  assert.deepEqual(h.requests.filter((request) => request.path.endsWith("/history")).map((request) => request.path), [
    "/sessions/open-session/history", "/sessions/target-session/history",
  ]);
  b.resolve(response([{ role: "assistant", content: "B transcript" }]));
  await settle();
  assert.equal(h.context.historySession, "target-session");
  assert.equal(h.elements.msgs.textContent, "B transcript");
  a.resolve(response([{ role: "assistant", content: "late A transcript" }]));
  await oldHistory;
  await click.done;
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.context.historySession, "target-session");
  assert.equal(h.elements.msgs.textContent, "B transcript");
  assert.equal(h.row().classList.contains("cur"), true);
});

test("boot renders every fetched session before history and config hydration finish", async () => {
  const history = deferred(), config = deferred();
  const h = harness({ realNavigation: true, sid: "", cachedItems: [], historyGet: () => history.promise, configGet: () => config.promise });
  const boot = h.context.loadSessions();
  await settle();
  assert.equal(sessionRows(h.elements.sesslist).length, 2);
  assertUnassignedGroup(h.elements.sesslist, 2);
  assert.equal(h.elements.sesscount.textContent, "2 sessions");
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.row().classList.contains("cur"), true);
  assert.equal(h.elements.sesstitle.textContent, "Target session");
  assert.equal(h.historyLoads(), 1);
  assert.equal(h.configLoads(), 0);
  await h.context.loadSessions();
  assert.equal(h.historyLoads(), 1, "another list refresh shares the selected session's pending hydration");
  history.resolve(response([]));
  await settle();
  assert.equal(sessionRows(h.elements.sesslist).length, 2);
  assertUnassignedGroup(h.elements.sesslist, 2);
  assert.equal(h.configLoads(), 1, "rows stay visible while config is still pending");
  config.resolve(response({}));
  await boot;
});

test("session rail fetches one small newest page and loads older sessions on demand", async () => {
  const h = harness({ sid: "", items: longSessions(), cachedItems: [] });
  await h.context.loadSessions();
  const listRequests = h.requests.filter((request) => request.path === "/sessions");
  assert.equal(listRequests.length, 1);
  assert.equal(listRequests[0].searchParams.get("limit"), "12");
  assert.equal(listRequests[0].searchParams.get("offset"), "0");
  assert.equal(listRequests[0].searchParams.get("newest"), "true");
  assert.equal(h.context.lastItems.length, 12);
  assert.equal(h.elements.sesscount.textContent, "102 sessions");

  await h.context.loadOlderSessions();
  assert.equal(h.requests.filter((request) => request.path === "/sessions").length, 2);
  assert.equal(h.requests.at(-1).searchParams.get("offset"), "12");
  assert.equal(h.context.lastItems.length, 24);
  assert.equal(h.elements.sesscount.textContent, "102 sessions");
});

for (const refreshAfterSwitch of [false, true]) test("a stale session list omitting B cannot steal selection (new refresh=" + refreshAfterSwitch + ")", async () => {
  const stale = deferred();
  let reads = 0;
  const h = harness({ realNavigation: true, sessionsGet: () => ++reads === 1 ? stale.promise : response(sessions(), 200, { "x-sessions-total": "2", "x-sessions-page-size": "2" }) });
  const oldList = h.context.loadSessions();
  if (refreshAfterSwitch) {
    await h.dispatch(h.row(), "click").done;
    await h.context.loadSessions();
  }
  else h.context.switchSession("target-session");
  await settle();
  assert.equal(reads, refreshAfterSwitch ? 2 : 1);
  stale.resolve(response([sessions()[0]]));
  await oldList;
  assert.equal(h.context.sessionId, "target-session");
  assert.ok(h.row(), "a stale payload cannot remove the selected row from the cached list");
  assert.equal(h.context.lastItems.length, 2);
  assert.deepEqual(h.switches, ["target-session"]);
});

test("an older session refresh cannot overwrite the newest list or its title", async () => {
  const stale = deferred();
  let reads = 0;
  const latest = sessions();
  latest[0].summary = "Latest open title";
  const h = harness({ sessionsGet: () => ++reads === 1 ? stale.promise : response(latest) });
  const oldList = h.context.loadSessions();
  await h.context.loadSessions();
  stale.resolve(response(sessions()));
  await oldList;
  assert.equal(h.elements.sesstitle.textContent, "Latest open title");
  assert.equal(h.row("open-session").querySelector("span").textContent, "Latest open title");
});

test("a session list from the previous server is discarded", async () => {
  const stale = deferred();
  const h = harness({ sessionsGet: () => stale.promise });
  const oldList = h.context.loadSessions();
  h.context.store.base = "https://other.test";
  stale.resolve(response([sessions()[1]]));
  await oldList;
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.context.lastItems.length, 2);
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.deepEqual(h.switches, []);
});

test("a failed session refresh keeps the last good list and appends its error", async () => {
  const h = harness({ sessionsGet: () => response({ error: "temporarily offline" }, 503) });
  await h.context.loadSessions();
  assert.ok(h.row("open-session"));
  assert.ok(h.row());
  assert.equal(h.context.lastItems.length, 2);
  assert.match(h.elements.sesslist.children.at(-1).textContent, /sessions refresh failed: 503 temporarily offline/);
  assert.equal(h.context.sessionId, "open-session");
});

test("sheet keyboard focus cycles through its actions and Escape restores the star focus", () => {
  const h = harness({ phone: true }), star = h.row().querySelector(".mk");
  h.hold();
  assert.equal(h.document.activeElement, h.elements.sessionactionarchive);
  for (const id of ["sessionactiongroup", "sessionactionrename", "sessionactiondelete", "sessionactioncancel", "sessionactionarchive"]) {
    const key = h.dispatch(h.document.activeElement, "keydown", { key: "Tab" });
    assert.equal(key.event.prevented, true);
    assert.equal(h.document.activeElement, h.elements[id]);
  }
  h.dispatch(h.document.activeElement, "keydown", { key: "Tab", shiftKey: true });
  assert.equal(h.document.activeElement, h.elements.sessionactioncancel);
  h.dispatch(h.document.activeElement, "keydown", { key: "Escape" });
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.equal(h.document.activeElement, star);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.switches, []);
});

test("leaving the phone layout closes the action sheet", () => {
  const h = harness({ phone: true });
  h.hold();
  h.media.matches = false;
  for (const listener of h.media.listeners) listener();
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.switches, []);
});

test("mark styles use theme gold and phone actions have 44px touch targets", () => {
  assert.match(css, /#sessions li \.mk\[aria-pressed="true"\] \{ color:var\(--mark\); \}/);
  assert.equal((css.match(/--mark:#[a-f0-9]+/gi) || []).length, 3);
  const phoneCSS = css.slice(css.indexOf("@media (max-width:700px)"));
  assert.match(phoneCSS, /#sessions li \.mk \{ display:inline-flex; align-items:center; justify-content:center; width:44px; height:44px; padding:0; font-size:15px; background:none; \}/);
  assert.match(css, /#sessmorepanel button \{[^}]*min-height:44px;/);
  assert.match(phoneCSS, /#sessions li \.mk\[aria-pressed="true"\] \{ color:var\(--mark\); \}/);
  assert.match(phoneCSS, /#sessions li \.session-menu \{ width:44px; height:44px; \}/);
  assert.match(css, /#sessionactionspanel button \{ min-width:44px; min-height:44px;/);
  assert.match(html, /id="sessionactions" hidden/);
  assert.match(html, /id="sessionactionspanel" role="dialog" aria-modal="true" aria-labelledby="sessionactionstitle"/);
  assert.match(between("function wire()", "function registerAppServiceWorker()"), /wireSessionActions\(\);/);
});

test("More contains the filters and group tools without another toolbar row", () => {
  const menu = html.match(/<details id="sessmore">([\s\S]*?)<\/details>/)[1];
  for (const id of ["sessactive", "sessarchived", "sessall", "sessmarked", "sessiongroupnew", "sessiongroupmanage"]) assert.match(menu, new RegExp('id="' + id + '"'));
  assert.match(menu, /<summary id="sessmoretoggle" aria-label="More session options">More<\/summary>/);
  assert.doesNotMatch(html, /id="sessviewrow"/);
  assert.match(css, /#sessfilterrow \{ display:flex; align-items:center;[^}]*flex-shrink:0;/);
  assert.match(css, /#sessfilterrow > #sesscount \{[^}]*min-width:0;[^}]*white-space:nowrap; overflow:hidden; text-overflow:ellipsis;/);
  const desktop = css.slice(css.indexOf("@media (min-width:701px)"), css.indexOf("@media (max-width:700px)"));
  assert.match(desktop, /#sessfilterrow \{ padding:0 8px; \}/);
  assert.match(desktop, /#sessmore > summary \{ min-height:28px; padding:2px 6px; \}/);
});

test("choosing a More filter closes the menu and leaves its active state visible", async () => {
  const h = harness({ items: sessions(true) });
  assert.equal(h.elements.sessfilterstatus.hidden, true);
  h.elements.sessmore.open = true;
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.document.activeElement, h.elements.sessmoretoggle);
  assert.equal(h.elements.sessfilterstatus.hidden, false);
  assert.equal(h.elements.sessfilterstatus.textContent, "Starred");
  assert.deepEqual(sessionRows(h.elements.sesslist).map((li) => li.dataset.sessionId), ["target-session"]);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.row().hidden, false);
  assert.equal(h.elements.sesslist.querySelector(".session-group-toggle").getAttribute("aria-expanded"), "true");
  assert.ok(sessionRows(h.elements.sesslist).filter((li) => !li.hidden).length > 0);
  await h.dispatch(h.elements.sessall, "click").done;
  assert.equal(h.elements.sessfilterstatus.hidden, true);
  assert.equal(sessionRows(h.elements.sesslist).length, 2);
});

test("archive and star filters combine without hiding their state or changing selection", async () => {
  const archived = { id: "archived-session", summary: "Archived title", marked: true, archived: true, size_bytes: 1024 };
  const h = harness({ items: [...sessions(), archived] });
  await h.dispatch(h.elements.sessarchived, "click").done;
  assert.equal(h.elements.sessfilterstatus.textContent, "Archived");
  assert.equal(h.elements.sessarchived.getAttribute("aria-pressed"), "true");
  assert.deepEqual(sessionRows(h.elements.sesslist).map((li) => li.dataset.sessionId), [archived.id]);
  assert.equal(h.row(archived.id).hidden, false);
  assert.equal(h.elements.sesslist.querySelector(".session-group-toggle").getAttribute("aria-expanded"), "true");
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(h.elements.sessfilterstatus.textContent, "Archived ★");
  assert.equal(h.elements.sessfilterstatus.getAttribute("aria-label"), "Archived · Starred");
  assert.deepEqual(sessionRows(h.elements.sesslist).map((li) => li.dataset.sessionId), [archived.id]);
  assert.equal(h.row(archived.id).hidden, false);
  assert.deepEqual(h.switches, []);
  assert.ok(h.requests.filter((request) => request.path === "/sessions").every((request) => request.searchParams.get("archived") === "only"));
});

test("All, Starred and Archived keep independent group collapse choices", async () => {
  const archived = { id: "archived-session", summary: "Archived title", marked: true, archived: true };
  const h = harness({ items: [...sessions(true), archived] });
  const toggle = () => h.elements.sesslist.querySelector(".session-group-toggle");
  h.dispatch(toggle(), "click");
  assert.equal(h.row().hidden, true);
  const allKey = h.context.sessionGroupCollapseKey(null);
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.notEqual(h.context.sessionGroupCollapseKey(null), allKey);
  assert.equal(h.row().hidden, false);
  assert.equal(toggle().getAttribute("aria-expanded"), "true");
  h.dispatch(toggle(), "click");
  const starredKey = h.context.sessionGroupCollapseKey(null);
  await h.dispatch(h.elements.sessarchived, "click").done;
  assert.notEqual(h.context.sessionGroupCollapseKey(null), starredKey);
  assert.equal(h.row(archived.id).hidden, false);
  await h.dispatch(h.elements.sessactive, "click").done;
  assert.equal(h.row().hidden, true);
  await h.dispatch(h.elements.sessall, "click").done;
  assert.equal(h.row().hidden, true);
});

for (const collapsed of [true, false]) {
  test("Starred preserves explicit " + (collapsed ? "collapse" : "expansion") + " through refresh, polling and paging", async () => {
    const stars = Array.from({ length: 15 }, (_, index) => ({ id: "star-" + index, summary: "Star " + index, marked: true, activity: "idle" }));
    const h = harness({ items: [sessions()[0], ...stars], activityGet: (request) => response(request.body.ids.map((id) => ({ id, activity: "running", waiting: "confirm" }))) });
    await h.dispatch(h.elements.sessmarked, "click").done;
    const toggle = () => h.elements.sesslist.querySelector(".session-group-toggle");
    h.dispatch(toggle(), "click");
    if (!collapsed) h.dispatch(toggle(), "click");
    const assertChoice = (count) => {
      assert.equal(toggle().getAttribute("aria-expanded"), String(!collapsed));
      const rows = sessionRows(h.elements.sesslist);
      assert.equal(rows.length, count);
      for (const row of rows) assert.equal(row.hidden, collapsed);
    };
    assertChoice(12);
    h.context.renderSessionList(h.context.lastItems);
    assertChoice(12);
    await h.context.loadSessions({ preserveSelection: true });
    assertChoice(12);
    h.document.visibilityState = "visible";
    await h.context.pollSessionActivity();
    assert.ok(h.requests.some((request) => request.path === "/session-activity"));
    assertChoice(12);
    h.document.visibilityState = "hidden";
    await h.dispatch(h.elements.sesslist.querySelector(".session-more").querySelector("button"), "click").done;
    assertChoice(15);
    const pages = h.requests.filter((request) => request.path === "/sessions");
    assert.ok(pages.every((request) => request.searchParams.get("marked") === "true"));
    assert.equal(pages.at(-1).searchParams.get("offset"), "12");
    assert.equal(h.elements.sesscount.textContent, "15 sessions · showing 15");
    const saved = JSON.parse(h.stored.get("serve.groups.collapsed.v1"));
    assert.equal(saved[h.context.sessionGroupCollapseKey(null)], collapsed);
    vm.runInContext("sessionGroupCollapsed = JSON.parse(localStorage.getItem(SESSION_GROUP_COLLAPSED_KEY));", h.context);
    h.context.renderSessionList(h.context.lastItems);
    assertChoice(15);
  });
}

test("Starred with zero results shows its empty state without a group header", async () => {
  const h = harness({ groups: [{ id: "empty", name: "Empty group", order: 0, pinned: false }] });
  await h.context.loadSessionGroups();
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(h.elements.sesslist.querySelector(".session-empty").textContent, "暂无星标会话");
  assert.equal(h.elements.sesslist.querySelectorAll(".session-group-head").length, 0);
  assert.equal(sessionRows(h.elements.sesslist).length, 0);
  assert.equal(h.elements.sesscount.textContent, "0 sessions · showing 0");
  assert.equal(h.context.sessionId, "open-session");
});

test("Starred expands every populated group when the selected session is unstarred", async () => {
  const groups = [{ id: "work", name: "Work", order: 0, pinned: false }, { id: "other", name: "Other", order: 1, pinned: false }];
  const h = harness({ groups, items: [sessions()[0], ...groups.map((group) => ({ id: group.id + "-star", summary: group.name, marked: true, group_id: group.id })), sessions(true)[1]] });
  await h.context.loadSessionGroups();
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(h.context.sessionId, "open-session");
  const heads = h.elements.sesslist.querySelectorAll(".session-group-head");
  assert.equal(heads.length, 3);
  for (const head of heads) assert.equal(head.querySelector(".session-group-toggle").getAttribute("aria-expanded"), "true");
  const rows = sessionRows(h.elements.sesslist);
  assert.equal(rows.length, 3);
  for (const row of rows) assert.equal(row.hidden, false);
});

test("More closes on Escape, outside pointer and outside focus without changing filters", () => {
  const h = harness();
  h.elements.sessmore.open = true;
  const key = h.dispatch(h.elements.sessmore, "keydown", { key: "Escape" });
  assert.equal(key.event.prevented, true);
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.document.activeElement, h.elements.sessmoretoggle);
  for (const event of ["pointerdown", "focusin"]) {
    h.elements.sessmore.open = true;
    for (const listener of h.documentListeners.get(event)) listener({ target: h.elements.sessmarked });
    assert.equal(h.elements.sessmore.open, true);
    for (const listener of h.documentListeners.get(event)) listener({ target: h.elements.prompt });
    assert.equal(h.elements.sessmore.open, false);
  }
  assert.deepEqual(h.requests, []);
  assert.equal(h.elements.sessfilterstatus.hidden, true);
});

for (const phone of [false, true]) test("top edit control persists the current session title (phone=" + phone + ")", async () => {
  const h = harness({ phone });
  await h.dispatch(h.elements.sessrename, "click").done;
  const editor = h.elements.sessiongroupbody.querySelector("input");
  assert.equal(editor.value, "Open session");
  editor.value = "  Current title  ";
  await h.dispatch(editor, "keydown", { key: "Enter" }).done;
  const requests = h.requests.filter((request) => request.path.endsWith("/title"));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].method, "POST");
  assert.equal(requests[0].path, "/sessions/open-session/title");
  assert.deepEqual(requests[0].body, { title: "Current title" });
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.sesstitle.textContent, "Current title");
  assert.equal(h.row("open-session").querySelector(".session-title").textContent, "Current title");
  assert.equal(h.row().querySelector(".session-title").textContent, "Target session");
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
  assert.deepEqual(h.switches, []);
});

for (const phone of [false, true]) for (const id of ["open-session", "target-session"]) {
  test("row menu renames without switching the current session (phone=" + phone + ", id=" + id + ")", async () => {
    const h = harness({ phone });
    const oldTitle = id === "open-session" ? "Open session" : "Target session";
    await h.dispatch(h.row(id).querySelector(".session-menu"), "click").done;
    const rename = h.elements.sessionactionrename;
    assert.ok(rename, "every session has a visible Rename action");
    assert.equal(rename.hidden, false);
    assert.equal(rename.textContent, "Rename...");
    await h.dispatch(rename, "click").done;
    assert.equal(h.elements.sessionactions.hidden, true);
    const editor = h.elements.sessiongroupbody.querySelector("input");
    assert.equal(editor.value, oldTitle);
    assert.equal(h.document.activeElement, editor);
    assert.equal(h.context.sessionId, "open-session");
    editor.value = "  Renamed " + id + "  ";
    await h.dispatch(h.elements.sessiongroupbody.querySelector("button"), "click").done;
    const requests = h.requests.filter((request) => request.path.endsWith("/title"));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].method, "POST");
    assert.equal(requests[0].path, "/sessions/" + id + "/title");
    assert.deepEqual(requests[0].body, { title: "Renamed " + id });
    assert.ok(h.requests.some((request) => request.path === "/sessions"), "reload titles after persistence");
    assert.equal(h.context.sessionId, "open-session");
    assert.equal(h.elements.sesstitle.textContent, id === "open-session" ? "Renamed " + id : "Open session");
    assert.equal(h.row(id).querySelector(".session-title").textContent, "Renamed " + id);
    const otherId = id === "open-session" ? "target-session" : "open-session";
    assert.equal(h.row(otherId).querySelector(".session-title").textContent, id === "open-session" ? "Target session" : "Open session");
    assert.equal(h.elements.sessiongroupdialog.hidden, true);
    assert.deepEqual(h.switches, []);
    assert.equal(h.historyLoads(), 0);
    assert.equal(h.configLoads(), 0);
    assert.equal(h.elements.prompt.value, "draft in A");
  });
}

test("row rename failure keeps both titles and the editor for retry", async () => {
  let fail = true;
  const serverItems = sessions();
  const h = harness({
    titlePost: (request) => {
      if (fail) return response({}, 500);
      serverItems[1].summary = request.body.title;
      return response({ title: request.body.title });
    },
    sessionsGet: () => response(serverItems),
  });
  await h.dispatch(h.row().querySelector(".session-menu"), "click").done;
  await h.dispatch(h.elements.sessionactionrename, "click").done;
  const editor = h.elements.sessiongroupbody.querySelector("input");
  const save = h.elements.sessiongroupbody.querySelector("button");
  editor.value = "Retried title";
  await h.dispatch(save, "click").done;
  assert.match(h.elements.sessiongrouperror.textContent, /^Rename failed: 500/);
  assert.equal(h.elements.sessiongroupdialog.hidden, false);
  assert.equal(save.disabled, false);
  assert.equal(editor.value, "Retried title");
  assert.equal(h.row().querySelector(".session-title").textContent, "Target session");
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.equal(h.context.sessionId, "open-session");
  fail = false;
  await h.dispatch(save, "click").done;
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
  assert.equal(h.requests.filter((request) => request.path.endsWith("/title")).length, 2);
  assert.equal(h.row().querySelector(".session-title").textContent, "Retried title");
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.deepEqual(h.switches, []);
});

for (const phone of [false, true]) test("select another session then rename with the top edit control (phone=" + phone + ")", async () => {
  const h = harness({ phone });
  await h.dispatch(h.row(), "click").done;
  assert.equal(h.context.sessionId, "target-session");
  await h.dispatch(h.elements.sessrename, "click").done;
  const editor = h.elements.sessiongroupbody.querySelector("input");
  assert.equal(editor.value, "Target session");
  assert.equal(h.document.activeElement, editor);
  editor.value = "Renamed target";
  await h.dispatch(h.elements.sessiongroupbody.querySelector("button"), "click").done;
  const request = h.requests.find((request) => request.path.endsWith("/title"));
  assert.equal(request.path, "/sessions/target-session/title");
  assert.deepEqual(request.body, { title: "Renamed target" });
  assert.equal(h.context.sessionId, "target-session");
  assert.equal(h.elements.sesstitle.textContent, "Renamed target");
  assert.equal(h.row("open-session").querySelector(".session-title").textContent, "Open session");
  await h.dispatch(h.row().querySelector(".mk"), "click").done;
  assert.equal(h.requests.at(-1).path, "/sessions/target-session/mark");
  await h.dispatch(h.elements.sessrename, "click").done;
  h.dispatch(h.elements.sessiongrouppanel, "keydown", { key: "Escape" });
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
});

test("overlaid actions reserve nearby title space while phone status shares their height", () => {
  const h = harness({ items: sessions(true) }), li = h.row();
  assert.match(li.title, /Target session\nID: target-session\n2 KB/);
  assert.equal(li.classList.contains("session-marked"), true);
  assert.equal(h.row("open-session").classList.contains("session-marked"), false);
  h.dispatch(li, "contextmenu");
  assert.equal(h.elements.sessionactionmeta.textContent, "ID: target-session\n2 KB\ntest-page");
  assert.match(css, /#sessions li \.session-hash \{ display:none; \}/);
  assert.match(css, /#sessions li \.mk \{[^}]*opacity:0; pointer-events:none;/);
  assert.match(css, /#sessions li\.session-marked \.mk, #sessions li:focus-within \.mk \{ opacity:1; pointer-events:auto; \}/);
  assert.match(css, /@media \(hover:hover\) \{\s*#sessions li:hover \.mk \{ opacity:1; pointer-events:auto; \}/);
  assert.match(css, /#sessions li\[data-session-id\] \{[^}]*display:flow-root;/);
  assert.match(css, /#sessions li \.session-controls \{ position:absolute; top:var\(--session-action-top\); right:var\(--session-action-right\); display:flex; flex-wrap:nowrap; align-items:center; gap:2px; \}/);
  assert.match(css, /#sessions li\[data-session-id\] \{ --session-action-size:22px; --session-action-top:8px; --session-action-right:8px;/);
  assert.match(css, /#sessions li \.session-title::before \{ content:""; float:right; width:calc\(var\(--session-action-size\) \+ var\(--session-action-size\) \+ 8px\); height:var\(--session-action-size\); \}/);
  assert.doesNotMatch(css, /#sessions li \.session-controls \{[^}]*float:/);
  assert.match(css, /#sessions li \.mk \{ width:22px; height:22px;/);
  assert.match(css, /#sessions li \.session-menu \{ width:22px; height:22px;/);
  assert.match(css, /#sessions li \.session-meta \{ clear:both; display:none; margin:0; \}/);
  assert.match(css, /#sessions li\.session-renaming \.session-controls \{ display:none; \}/);
  const desktopCSS = css.slice(0, css.indexOf("@media (max-width:700px)"));
  assert.doesNotMatch(desktopCSS, /#sessions li\[data-session-id\] \{[^}]*grid-template-columns/);
  assert.doesNotMatch(css, /#sessions[^{}]*\.session-title\s*\{[^}]*margin-right:/);
  assert.match(css, /#sessions li \.session-menu \{ width:44px; height:44px; \}/);
  assert.match(css, /#sessions li \.mk \{[^}]*width:44px; height:44px;/);
  const phoneCSS = css.slice(css.indexOf("@media (max-width:700px)"));
  assert.match(phoneCSS, /#sessions li\[data-session-id\]:not\(\[hidden\]\) \{\s*--session-action-size:44px; --session-action-top:4px; --session-action-right:6px;/);
  assert.match(phoneCSS, /#sessions li \.session-controls \{ gap:0; \}/);
  assert.doesNotMatch(phoneCSS, /#sessions[^{}]*(?:li\[data-session-id\]|\.session-controls|\.session-title)[^{}]*\{[^}]*(?:display:grid|position:static|grid-column|grid-row)/);
  assert.doesNotMatch(phoneCSS, /#sessions li \.session-title::before \{[^}]*display:none/);
  assert.match(phoneCSS, /#sessions li \.session-meta \{ clear:none; margin:0; \}/);
  assert.match(phoneCSS, /#sessions ul \{ min-height:0;[^}]*overscroll-behavior-y:contain;/);
  assert.match(phoneCSS, /#sessions \.rowbtns \{ flex:0 0 auto;[^}]*env\(safe-area-inset-bottom\)/);
  assert.match(phoneCSS, /#sessions \.rowbtns button \{[^}]*min-width:0; min-height:44px;/);
  const controls = li.querySelector(".session-controls");
  assert.deepEqual(controls.children, [li.querySelector(".mk"), li.querySelector(".session-menu")]);
  assert.deepEqual(li.children, [controls, li.querySelector(".session-title"), li.querySelector(".session-meta")]);
});

for (const phone of [false, true]) for (const marked of [false, true]) {
  for (const summary of ["feat/2026-11-17-critical-premises", "Short"]) {
    test(`session title starts beside overlaid actions without an atomic box (phone=${phone}, marked=${marked}, title=${summary})`, () => {
      const h = harness({ phone, items: [{ ...sessions()[1], summary, marked }] }), li = h.row();
      const title = li.querySelector(".session-title"), controls = li.querySelector(".session-controls");
      assert.equal(title.textContent, summary, "the full title is retained");
      assert.equal(title.children[0].textContent, summary, "no leading placeholder is added to the DOM");
      assert.equal(title.children[0].className, "session-title-token");
      assert.equal(li.querySelectorAll(".session-menu").length, 1);
      assert.equal(li.querySelector(".mk").getAttribute("aria-pressed"), String(marked));
      assert.deepEqual(controls.children, [li.querySelector(".mk"), li.querySelector(".session-menu")]);
      assert.match(css, /#sessions li \.session-title-token \{ display:inline; overflow-wrap:anywhere; \}/,
        "an identifier can break into the remaining first-line space");
      assert.doesNotMatch(css, /#sessions[^{}]*\.session-title(?:-token)?\s*\{[^}]*(?:inline-block|white-space:nowrap|text-overflow|line-clamp|overflow:hidden)/);
      assert.match(css, /#sessions li \.session-controls \{ position:absolute;/);
      assert.doesNotMatch(css, /#sessions[^{}]*\.session-controls\s*\{[^}]*position:(?:static|relative)/);
      assert.doesNotMatch(css, /#sessions[^{}]*\.session-title\s*\{[^}]*(?:padding-right|margin-right)/,
        "later title lines retain the full width after the float ends");
    });
  }
}

for (const phone of [false, true]) test("session menu opens actions with one press and restores focus (phone=" + phone + ")", async () => {
  const h = harness({ phone }), li = h.row(), menu = li.querySelector(".session-menu");
  assert.equal(menu.type, "button");
  assert.equal(menu.getAttribute("aria-label"), "Actions for Target session");
  assert.equal(menu.getAttribute("aria-haspopup"), "dialog");
  assert.equal(menu.getAttribute("aria-controls"), "sessionactionspanel");
  const down = h.dispatch(menu, "pointerdown", h.touch());
  assert.equal(down.event.stopped, true);
  h.advance(650);
  assert.equal(h.elements.sessionactions.hidden, true, "pressing the menu does not start a row hold");
  const click = h.dispatch(menu, "click");
  await click.done;
  assert.equal(click.event.stopped, true);
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(h.elements.sessionactionstitle.textContent, "Target session");
  assert.equal(h.document.activeElement, h.elements.sessionactionarchive);
  for (const id of ["sessionactionarchive", "sessionactiongroup", "sessionactiondelete"]) assert.equal(h.elements[id].hidden, false);
  await h.dispatch(h.elements.sessionactioncancel, "click").done;
  assert.equal(h.document.activeElement, menu);
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.requests, []);
});

test("empty groups stay available through Manage groups without occupying list rows", async () => {
  const groups = [{ id: "empty", name: "Empty group", order: 0, pinned: false }];
  const h = harness({ groups });
  await h.context.loadSessionGroups();
  h.context.renderSessionList(h.context.lastItems);
  assertUnassignedGroup(h.elements.sesslist, 2);
  h.elements.sessmore.open = true;
  await h.dispatch(h.elements.sessiongroupmanage, "click").done;
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.elements.sessiongroupdialog.hidden, false);
  assert.equal(h.elements.sessiongrouptitle.textContent, "Manage groups");
  const buttons = h.elements.sessiongroupbody.querySelectorAll("button");
  assert.deepEqual(buttons.map((button) => button.textContent), ["New group…", "All", "Hidden", "Empty group", "Hide from sidebar"]);
  await h.dispatch(h.elements.sessiongroupbody.querySelector(".session-group-manager-name"), "click").done;
  assert.equal(h.elements.sessiongrouptitle.textContent, "Empty group");
  assert.ok(h.elements.sessiongroupbody.querySelectorAll("button").some((button) => button.textContent === "New session in this group"));
});

test("compact group heads preserve waiting visibility and explicit collapse state", () => {
  const items = [{ ...sessions()[0], activity: "running" }, { ...sessions()[1], activity: "running", waiting: "confirm", waiting_prompt: "Approve?" }];
  const h = harness({ items }), head = h.elements.sesslist.querySelector(".session-group-head");
  const toggle = head.querySelector(".session-group-toggle");
  assert.equal(head.querySelector(".session-group-count"), null);
  assert.equal(toggle.title, "Unassigned · 2 loaded sessions");
  const waiting = head.querySelector(".waiting");
  assert.equal(waiting.textContent, "1");
  assert.equal(waiting.getAttribute("aria-label"), "1 WAITING");
  assert.match(waiting.title, /need your answer/);
  assert.equal(waiting.hidden, false);
  assert.match(toggle.getAttribute("aria-label"), /1 WAITING/);
  const running = head.querySelector(".session-group-badge");
  assert.equal(running.textContent, "1");
  assert.equal(running.getAttribute("aria-label"), "1 RUNNING");
  assert.equal(running.title, "1 loaded sessions running");
  assert.equal(running.hidden, false);
  assert.match(toggle.getAttribute("aria-label"), /1 RUNNING/);
  assert.match(css, /#sessions \.session-group-toggle \{[^}]*grid-template-columns:auto minmax\(0,1fr\) auto;/);
  assert.match(css, /#sessions \.session-group-toggle \{[^}]*min-height:28px;[^}]*font-size:11px; font-weight:500;/);
  assert.match(css, /#sessions \.session-group-name \{ grid-column:2; grid-row:1;[^}]*white-space:nowrap; color:var\(--fg\); font-size:12px; font-weight:600; line-height:18px;/);
  assert.match(css, /#sessions \.session-group-status \{ grid-column:3; grid-row:1; display:flex; flex-wrap:nowrap;/);
  assert.match(css, /#sessions \.session-group-badge \{[^}]*font-size:10px; font-weight:500; line-height:16px;/);
  h.dispatch(toggle, "click");
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.ok(sessionRows(h.elements.sesslist).every((li) => li.hidden));
  assert.equal(waiting.hidden, false);
  assert.equal(running.hidden, false);
  h.context.renderSessionList(h.context.lastItems);
  assert.equal(h.elements.sesslist.querySelector(".session-group-toggle").getAttribute("aria-expanded"), "false");
  assert.ok(sessionRows(h.elements.sesslist).every((li) => li.hidden));
  h.dispatch(h.elements.sesslist.querySelector(".session-group-toggle"), "click");
  assert.ok(sessionRows(h.elements.sesslist).every((li) => !li.hidden));
  vm.runInContext("sessionActivitySamples.clear(); paintSessionGroupActivity();", h.context);
  const refreshedHead = h.elements.sesslist.querySelector(".session-group-head");
  assert.equal(refreshedHead.querySelector(".session-group-badge").hidden, true);
  assert.equal(refreshedHead.querySelector(".waiting").hidden, true);
  assert.match(css, /#sessions \.session-group-badge\[hidden\] \{ display:none; \}/);
});

for (const phone of [false, true]) test("group padding and title toggle once while group actions stay independent (phone=" + phone + ")", async () => {
  const groups = [{ id: "work", name: "Work", order: 0, pinned: true }];
  const h = harness({ phone, groups, items: [{ ...sessions()[0], group_id: "work" }, sessions()[1]] });
  await h.context.loadSessionGroups();
  h.context.renderSessionList(h.context.lastItems);
  const heads = h.elements.sesslist.querySelectorAll(".session-group-head");
  const work = heads[0], unassigned = heads[1], toggle = work.querySelector(".session-group-toggle");
  assert.equal(work.querySelector(".session-group-name").textContent, "Work");
  assert.equal(work.querySelector(".session-group-name").classList.contains("session-group-pinned"), true);
  assert.match(css, /#sessions \.session-group-pinned::before \{ content:"★"; margin-right:4px; color:var\(--dim\); font-size:10px; font-weight:500; \}/);
  assert.match(toggle.getAttribute("aria-label"), /Pinned group/);
  const arrow = toggle.querySelector(".session-group-arrow");
  assert.equal(arrow.getAttribute("aria-hidden"), "true");
  assert.equal(arrow.textContent, "");
  assert.match(css, /#sessions \.session-group-arrow \{[^}]*grid-column:1; grid-row:1;[^}]*width:12px; height:12px; color:var\(--fg\);/);
  assert.match(css, /#sessions \.session-group-arrow::before \{ content:""; width:0; height:0; border-top:5px solid transparent; border-bottom:5px solid transparent; border-left:8px solid currentColor; \}/);
  assert.match(css, /#sessions \.session-group-toggle\[aria-expanded="true"\] \.session-group-arrow::before \{ transform:rotate\(90deg\); \}/);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  h.dispatch(work, "click");
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.equal(h.row("open-session").hidden, true);
  h.dispatch(work.querySelector(".session-group-name"), "click");
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  const menu = work.querySelector(".session-group-menu");
  const click = h.dispatch(menu, "click");
  assert.equal(click.event.stopped, true);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(h.elements.sessiongrouptitle.textContent, "Work");
  assert.equal(unassigned.querySelector(".session-group-menu"), null);
  for (const head of heads) {
    const create = head.querySelector(".session-group-new");
    assert.equal(create.textContent, "＋ New");
    assert.equal(create.title, "New session in " + head.querySelector(".session-group-name").textContent);
    assert.equal(create.getAttribute("aria-label"), create.title);
  }
  const phoneCSS = css.slice(css.indexOf("@media (max-width:700px)"));
  assert.match(phoneCSS, /#sessions \.session-group-toggle, #sessions \.session-group-new, #sessions \.session-group-menu, #sessions \.session-group-more \{ min-height:44px; \}/);
  assert.match(css, /#sessions \.session-group-more \{[^}]*min-width:38px;[^}]*font-size:11px;/);
  assert.match(phoneCSS, /#sessions \.session-group-menu \{ flex:0 0 44px; min-width:44px; padding:2px; \}/);
  assert.match(phoneCSS, /#sessions \.session-group-status \{ gap:2px; \}/);
  assert.match(phoneCSS, /#sessions \.session-group-badge \{[^}]*font-size:9px; line-height:14px;/);
  assert.match(css, /#sessions li\.session-group-head \{[^}]*border:0; border-radius:0; background:transparent; box-shadow:none;/);
  assert.match(css, /@media \(hover:hover\) and \(pointer:fine\) \{\s*#sessions \.session-group-new, #sessions \.session-group-menu \{ opacity:0; pointer-events:none; \}/);
  assert.match(css, /#sessions \.session-group-head:focus-within \.session-group-new, #sessions \.session-group-head:focus-within \.session-group-menu \{ opacity:1; pointer-events:auto; \}/);
  assert.match(css, /#sessions \.session-group-new \{[^}]*min-width:44px;[^}]*font-size:11px;/);
  assert.match(css, /#sessions li\.session-more button \{[^}]*min-height:44px;[^}]*border:0; background:transparent;/);
});

test("phone More keeps archive and group tools while the desktop-only star filter stays hidden", async () => {
  const h = harness({ phone: true });
  assert.equal(h.elements.sessmarkfilter.hidden, true);
  assert.equal(h.elements.sessactive.hidden, false);
  assert.equal(h.elements.sessiongroupmanage.hidden, false);
  h.elements.sessmore.open = true;
  await h.dispatch(h.elements.sessiongroupnew, "click").done;
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.elements.sessiongrouptitle.textContent, "New group");
  assert.equal(h.elements.sessiongroupbody.querySelector("input").getAttribute("aria-label"), "Group name");
  assert.equal(h.document.activeElement, h.elements.sessiongroupbody.querySelector("input"));
});
