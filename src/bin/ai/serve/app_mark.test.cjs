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
function between(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page functions: " + start);
  return script.slice(a, b);
}
const response = (body, status = 200) => ({ ok: status === 200, status, statusText: "Not Found", headers: { get: () => null }, json: async () => plain(body) });
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

function harness({ phone = false, sid = "open-session", items = sessions(), cachedItems = items, markError = "", confirmDelete = true, beforeMark, sessionsGet, historyGet, configGet, realNavigation = false } = {}) {
  const document = { activeElement: null };
  function node(tag = "div") {
    let text = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, hidden: false, disabled: false, title: "", scrollTop: 0,
      get className() { return [...classes].join(" "); },
      set className(value) { classes.clear(); for (const name of value.split(/\s+/).filter(Boolean)) classes.add(name); },
      classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
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
      remove() {
        if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((child) => child !== el);
        el.parentNode = null;
      },
      matches(selector) {
        return selector.split(",").some((part) => {
          const s = part.trim();
          return s.startsWith(".") ? classes.has(s.slice(1)) : tag === s;
        });
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
  const elements = { sesslist: node("ul"), sessfilter: node("input"), sesscount: node(), sesstitle: node(), sessmeta: node(), msgs: node(), prompt: node("textarea") };
  elements.prompt.value = "draft in A";
  for (const match of html.matchAll(/<(div|section|button)[^>]*id="(sessionaction[^"]*)"[^>]*>([^<]*)/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    el.textContent = match[3];
  }
  for (const id of ["sessionactionstitle", "sessionactionmark", "sessionactionfork", "sessionactiondelete", "sessionactioncancel"]) elements.sessionactionspanel.appendChild(elements[id]);
  elements.sessionactions.appendChild(elements.sessionactionsscrim);
  elements.sessionactions.appendChild(elements.sessionactionspanel);

  const serverItems = plain(items), requests = [], switches = [], confirmations = [], draftsCleared = [], messages = [], drawerChanges = [], stored = new Map();
  const media = { matches: phone, listeners: [], addEventListener(_, listener) { this.listeners.push(listener); } };
  const timers = new Map();
  let now = 1000, nextTimer = 1, historyLoads = 0, configLoads = 0;
  const context = vm.createContext({
    document, $: (id) => elements[id], window: { matchMedia: () => media },
    store: { base: "https://serve.test", token: "test-token" },
    lastItems: plain(cachedItems), sessionId: sid, pendingNew: null, lastMsgCount: 5, turnRunning: true, PAGE_ID: "test-page",
    localStorage: { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    Date: { now: () => now },
    setTimeout: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, at: now + ms }); return id; },
    clearTimeout: (id) => timers.delete(id),
    fetch: async (url, options) => {
      const request = { url, path: new URL(url).pathname, method: options.method || "GET", headers: plain(options.headers), body: options.body ? JSON.parse(options.body) : null };
      requests.push(request);
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
      return sessionsGet ? sessionsGet(request) : response(serverItems);
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
  vm.runInContext(between("function headers(", "const esc ="), context);
  vm.runInContext(between("function sessionTitle(s)", "// Phone drawer floats"), context);
  vm.runInContext(between("function renderSessionBar()", "// History is append-only"), context);
  if (realNavigation) {
    Object.assign(context, {
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
    });
    vm.runInContext(between("// BEGIN session switch", "// END session switch"), context);
    vm.runInContext(between("// History is append-only", "// Historical user inputs"), context);
    const switchSession = context.switchSession, loadHistory = context.loadHistory;
    context.switchSession = (...args) => { switches.push(args[0]); return switchSession(...args); };
    context.loadHistory = (...args) => { historyLoads += 1; return loadHistory(...args); };
  }
  vm.runInContext(between('$("sessfilter").oninput =', '$("connbtn").onclick ='), context);
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
        if (event.stopped) break;
      }
    }
    return { event, done: Promise.all(pending) };
  }
  const row = (id = "target-session") => elements.sesslist.children.find((li) => li.querySelector("small")?.textContent === id.slice(0, 8));
  function advance(ms) {
    now += ms;
    for (const [id, timer] of [...timers]) if (timer.at <= now) { timers.delete(id); timer.fn(); }
  }
  const touch = (extra = {}) => ({ pointerType: "touch", isPrimary: true, button: 0, clientX: 10, clientY: 20, ...extra });
  function hold(id) { const li = row(id); dispatch(li, "pointerdown", touch()); advance(650); dispatch(li, "pointerup", touch()); return li; }
  return { context, document, elements, requests, switches, confirmations, draftsCleared, messages, drawerChanges, stored, media, timers, row, advance, touch, hold, dispatch, historyLoads: () => historyLoads, configLoads: () => configLoads };
}

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
    assert.equal(li.children[0].textContent, "✎");
    assert.equal(li.children[1], star);
  }
});

test("a star marks its session through the authenticated API and refreshes without switching", async () => {
  const h = harness(), star = h.row().querySelector(".mk");
  const click = h.dispatch(star, "click");
  assert.equal(click.event.stopped, true);
  await click.done;
  assert.deepEqual(h.requests.map(({ path, method, body }) => ({ path, method, body })), [
    { path: "/sessions/target-session/mark", method: "POST", body: { marked: true } },
    { path: "/sessions", method: "GET", body: null },
  ]);
  assert.equal(h.requests[0].headers.Authorization, "Bearer test-token");
  assert.equal(h.requests[0].headers["Content-Type"], "application/json");
  assert.equal(h.row().querySelector(".mk").textContent, "★");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
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

test("a pending mark leaves the old star visible until the server response and refresh", async () => {
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
  assert.equal(h.elements.sessionactionmark.textContent, "标记会话");
  assert.equal(h.elements.sessionactiondelete.textContent, "删除会话");
  assert.equal(h.elements.sessionactioncancel.textContent, "取消");
  assert.equal(h.context.sessionHoldConsumed(li), true);
  await h.dispatch(li, "click").done;
  assert.deepEqual(h.switches, []);
  assert.deepEqual(h.requests, []);
  assert.deepEqual(h.confirmations, []);
});

for (const marked of [false, true]) test("the phone sheet offers " + (marked ? "unmark" : "mark") + " and toggles only the held session", async () => {
  const h = harness({ phone: true, items: sessions(marked) });
  h.hold();
  assert.equal(h.elements.sessionactionmark.textContent, marked ? "取消标记" : "标记会话");
  await h.dispatch(h.elements.sessionactionmark, "click").done;
  assert.equal(h.elements.sessionactions.hidden, true);
  assert.deepEqual(h.requests[0].body, { marked: !marked });
  assert.equal(h.requests[1].path, "/sessions");
  assert.equal(h.row().querySelector(".mk").getAttribute("aria-pressed"), String(!marked));
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.switches, []);
});

test("the sheet delete entry confirms and deletes its target while keeping the open session", async () => {
  const h = harness({ phone: true });
  h.hold();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  assert.deepEqual(h.confirmations, ['delete "Target session"?']);
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
  const h = harness({ phone: true, confirmDelete: false });
  h.hold();
  await h.dispatch(h.elements.sessionactiondelete, "click").done;
  assert.equal(h.confirmations.length, 1);
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
  h.dispatch(h.row().querySelector(".rn"), "pointerdown", h.touch());
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
  for (const titleClick of [true, false]) {
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
  assert.equal(rail.scrollTop, 0);
  assert.equal(rail.children.length, 100);
  assert.equal(h.elements.sesscount.textContent, "102 sessions · showing 100");
  assert.equal(h.row(), undefined);

  rail.scrollTop = 800;
  h.context.renderSessionList(h.context.lastItems);
  assert.equal(rail.scrollTop, 800, "a normal refresh must preserve scroll even with an active filter");
  assert.equal(rail.children.length, 100);

  filter.value = "target";
  await h.dispatch(filter, "input").done;
  assert.equal(rail.scrollTop, 0);
  assert.equal(rail.children.length, 1);
  assert.ok(h.row());
  assert.equal(h.elements.sesscount.textContent, "102 sessions · showing 1");

  filter.value = "";
  await h.dispatch(filter, "input").done;
  assert.equal(rail.scrollTop, 0);
  assert.equal(rail.children.length, 102);
  assert.equal(h.elements.sesscount.textContent, "102 sessions");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(h.requests, []);
});

test("a fresh session-list render starts at the top", () => {
  const h = harness({ cachedItems: [] }), rail = h.elements.sesslist;
  assert.equal(rail.children.length, 0);
  assert.equal(rail.scrollTop, 0);
  h.context.renderSessionList(longSessions());
  assert.equal(rail.children.length, 102);
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

test("desktop title navigation requests B history and rejects a late A response", async () => {
  const a = deferred(), b = deferred();
  const h = harness({ realNavigation: true, historyGet: (request) => request.path === "/sessions/open-session/history" ? a.promise : b.promise });
  const oldHistory = h.context.loadHistory();
  const click = h.dispatch(h.row().querySelector("span"), "click");
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
  assert.equal(h.elements.sesslist.children.length, 2);
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
  assert.equal(h.elements.sesslist.children.length, 2);
  assert.equal(h.configLoads(), 1, "rows stay visible while config is still pending");
  config.resolve(response({}));
  await boot;
});

for (const refreshAfterSwitch of [false, true]) test("a stale session list omitting B cannot steal selection (new refresh=" + refreshAfterSwitch + ")", async () => {
  const stale = deferred();
  let reads = 0;
  const h = harness({ realNavigation: true, sessionsGet: () => ++reads === 1 ? stale.promise : response(sessions()) });
  const oldList = h.context.loadSessions();
  if (refreshAfterSwitch) {
    await h.dispatch(h.row().querySelector("span"), "click").done;
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
  assert.equal(h.document.activeElement, h.elements.sessionactionmark);
  for (const id of ["sessionactiondelete", "sessionactioncancel", "sessionactionmark"]) {
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
  assert.match(phoneCSS, /#sessions li \.rn, #sessions li \.mk \{ min-width:44px; min-height:44px; \}/);
  assert.match(css, /#sessionactionspanel button \{ min-width:44px; min-height:44px;/);
  assert.match(html, /id="sessionactions" hidden/);
  assert.match(html, /id="sessionactionspanel" role="dialog" aria-modal="true" aria-labelledby="sessionactionstitle"/);
  assert.match(between("function wire()", "function registerAppServiceWorker()"), /wireSessionActions\(\);/);
});
