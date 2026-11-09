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
const response = (body, status = 200) => ({ ok: status === 200, status, statusText: "Not Found", json: async () => plain(body) });
const sessions = (marked = false) => [
  { id: "open-session", summary: "Open session", marked: false, size_bytes: 1024 },
  { id: "target-session", summary: "Target session", marked, size_bytes: 2048 },
];

function harness({ phone = false, items = sessions(), markError = "", confirmDelete = true, beforeMark } = {}) {
  const document = { activeElement: null };
  function node(tag = "div") {
    let text = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, hidden: false, disabled: false, title: "",
      get className() { return [...classes].join(" "); },
      set className(value) { classes.clear(); for (const name of value.split(/\s+/).filter(Boolean)) classes.add(name); },
      classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
      get textContent() { return text + el.children.map((child) => child.textContent).join(""); },
      set textContent(value) {
        text = String(value);
        for (const child of el.children) child.parentNode = null;
        el.children = [];
      },
      setAttribute(key, value) { el.attrs[key] = String(value); },
      getAttribute(key) { return el.attrs[key] ?? null; },
      appendChild(child) { child.parentNode = el; el.children.push(child); return child; },
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
  const elements = { sesslist: node("ul"), sesscount: node(), sesstitle: node(), sessmeta: node() };
  for (const match of html.matchAll(/<(div|section|button)[^>]*id="(sessionaction[^"]*)"[^>]*>([^<]*)/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    el.textContent = match[3];
  }
  for (const id of ["sessionactionstitle", "sessionactionmark", "sessionactiondelete", "sessionactioncancel"]) elements.sessionactionspanel.appendChild(elements[id]);
  elements.sessionactions.appendChild(elements.sessionactionsscrim);
  elements.sessionactions.appendChild(elements.sessionactionspanel);

  const serverItems = plain(items), requests = [], switches = [], confirmations = [], draftsCleared = [], messages = [];
  const media = { matches: phone, listeners: [], addEventListener(_, listener) { this.listeners.push(listener); } };
  const timers = new Map();
  let now = 1000, nextTimer = 1, historyLoads = 0, configLoads = 0;
  const context = vm.createContext({
    document, $: (id) => elements[id], window: { matchMedia: () => media },
    store: { base: "https://serve.test", token: "test-token" },
    lastItems: plain(items), sessionId: "open-session", pendingNew: null, lastMsgCount: 5, turnRunning: true, PAGE_ID: "test-page",
    Date: { now: () => now },
    setTimeout: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, at: now + ms }); return id; },
    clearTimeout: (id) => timers.delete(id),
    fetch: async (url, options) => {
      const request = { path: new URL(url).pathname, method: options.method || "GET", headers: plain(options.headers), body: options.body ? JSON.parse(options.body) : null };
      requests.push(request);
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
      return response(serverItems);
    },
    confirm: (text) => { confirmations.push(text); return confirmDelete; },
    clearDraft: (id) => draftsCleared.push(id),
    switchSession: (id) => { switches.push(id); context.sessionId = id; },
    loadHistory: async () => { historyLoads += 1; }, loadSessionConfig: async () => { configLoads += 1; },
    stopStaleFollow() {}, hideConfirm() {}, pollConfirm() {}, setDrawer() {}, resetPicks() {}, composerInUse: () => false,
    addMsg: (...args) => messages.push(args), note: (text) => messages.push(text),
  });
  vm.runInContext(between("function headers(", "const esc ="), context);
  vm.runInContext(between("function sessionTitle(s)", "// Phone drawer floats"), context);
  vm.runInContext(between("function renderSessionBar()", "// History is append-only"), context);
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
  return { context, document, elements, requests, switches, confirmations, draftsCleared, messages, media, timers, row, advance, touch, hold, dispatch, historyLoads: () => historyLoads, configLoads: () => configLoads };
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

test("desktop title clicks toggle the mark while row body clicks select the session", async () => {
  const h = harness();
  const click = h.dispatch(h.row().querySelector("span"), "click");
  assert.equal(click.event.stopped, true);
  await click.done;
  assert.equal(h.requests[0].path, "/sessions/target-session/mark");
  assert.equal(h.requests[0].method, "POST");
  assert.deepEqual(h.requests[0].body, { marked: true });
  assert.deepEqual(h.switches, []);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);

  const body = harness();
  await body.dispatch(body.row(), "click").done;
  assert.deepEqual(body.switches, ["target-session"]);
  assert.equal(body.context.sessionId, "target-session");
  assert.equal(body.historyLoads(), 1);
  assert.equal(body.configLoads(), 1);
  assert.equal(body.requests.some((request) => request.path.endsWith("/mark")), false);
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
