// Run with: node --test src/bin/ai/serve/app_changes.test.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function between(from, to, source = script) {
  const start = source.indexOf(from), end = source.indexOf(to, start);
  assert.ok(start >= 0 && end > start, "missing source slice: " + from);
  return source.slice(start, end);
}
function dom() {
  const elements = new Map();
  function node(tagName) {
    const handlers = new Map();
    const element = {
      tagName, hidden: false, children: [], attributes: {}, dataset: {}, className: "", parent: null,
      appendChild(child) {
        if (child.tagName === "#fragment") { this.append(...child.children); return child; }
        child.parent = this;
        this.children.push(child);
        return child;
      },
      append(...children) { for (const child of children) this.appendChild(child); },
      setAttribute(key, value) { this.attributes[key] = String(value); },
      getAttribute(key) { return this.attributes[key] ?? null; },
      contains(other) { return this === other || this.children.some((child) => child.contains(other)); },
      closest(selector) {
        assert.equal(selector, "[hidden]");
        for (let item = this; item; item = item.parent) if (item.hidden) return item;
        return null;
      },
      querySelectorAll(selector) {
        if (selector === ".msg.md") return [];
        assert.equal(selector, 'button, select, [tabindex="0"]');
        return this.children.flatMap((child) => [
          ...(["button", "select"].includes(child.tagName) || child.getAttribute("tabindex") === "0" ? [child] : []),
          ...child.querySelectorAll(selector),
        ]);
      },
      focus() { if (!this.closest("[hidden]")) document.activeElement = this; },
      addEventListener(type, handler) {
        if (!handlers.has(type)) handlers.set(type, []);
        handlers.get(type).push(handler);
      },
      dispatch(type, fields = {}) {
        const event = { target: this, ...fields,
          preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; },
        };
        for (let item = this; item; item = item.parent) {
          item["on" + type]?.(event);
          for (const handle of item._handlers.get(type) || []) handle(event);
          if (event.stopped) break;
        }
        return event;
      },
      _handlers: handlers,
      get textContent() { return (this.text || "") + this.children.map((child) => child.textContent).join(""); },
      set textContent(text) {
        for (const child of this.children) child.parent = null;
        this.children = [];
        this.text = String(text);
      },
      set innerHTML(_html) { assert.fail("Changes must render dynamic content as text"); },
    };
    element.classList = { toggle(name, force) {
      const classes = new Set(element.className.split(/\s+/).filter(Boolean));
      if (force) classes.add(name);
      else classes.delete(name);
      element.className = [...classes].join(" ");
    } };
    return element;
  }
  const document = node("#document");
  document.createElement = node;
  document.createDocumentFragment = () => node("#fragment");
  document.createTextNode = (text) => Object.assign(node("#text"), { text });
  // Read the small static trees so hidden ancestry, focus order, and bubbling match the markup.
  function mount(markup) {
    const stack = [document];
    for (const [token, tag, attrs, text] of markup.matchAll(/<\/?([\w-]+)([^>]*?)>|([^<]+)/g)) {
      if (!tag) { stack.at(-1).appendChild(document.createTextNode(text)); continue; }
      if (token.startsWith("</")) { assert.equal(stack.pop().tagName, tag); continue; }
      const element = node(tag);
      for (const [, key, value] of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
        if (key === "id") elements.set(value, element);
        else if (key === "hidden") element.hidden = true;
        else if (key === "class") element.className = value;
        else element.setAttribute(key, value ?? "");
      }
      stack.at(-1).appendChild(element);
      if (!token.endsWith("/>")) stack.push(element);
    }
    assert.equal(stack.length, 1);
  }
  mount(between('<div id="sessbar"', '<div id="conn">', html));
  mount(between('<div id="changes"', '<script>', html));
  for (const id of ["msgs", "info"]) elements.set(id, node("div"));
  return { document, node, $: (id) => elements.get(id) ?? null };
}

const base = "https://example.invalid", token = "mock-token";
const defaultId = "s/α+%";
const enabled = { features: { changes: true } };
const file = (id, path, overrides = {}) => ({ id, path, old_path: null, status: "modified", additions: 3, deletions: 1, binary: false, ...overrides });
const manifest = (files = [], overrides = {}) => ({ source: "working_tree", root_label: "sample repo", revision: "rev/α?= +%", files, truncated: false, reason: null, ...overrides });
const diff = (list, item, overrides = {}) => ({ revision: list.revision, file_id: item.id, path: item.path, status: item.status, format: "unified", diff: "", truncated: false, reason: null, ...overrides });
const scope = (id, label, { current = false, files = 0 } = {}) => ({ id, label, path: "/repo/" + label, current, changed: files > 0, files });
const mainScope = scope("current/α+%", "session repo", { current: true });
const scopedManifest = ({ id, label, path, current }, files = [], overrides = {}) => manifest(files, { scope: { id, label, path, current }, ...overrides });
const catalog = (...scopes) => ({ source: "git", scopes, truncated: false, reason: null });
const listPath = (sid = defaultId) => "/sessions/" + encodeURIComponent(sid) + "/changes";
const scopesPath = (sid = defaultId) => listPath(sid) + "/scopes";
const scopedListPath = (scope) => listPath() + "?scope=" + encodeURIComponent(scope.id);
const detailPath = (list, item) => listPath() + "/" + encodeURIComponent(item.id) + "?revision=" + encodeURIComponent(list.revision)
  + (list.scope ? "&scope=" + encodeURIComponent(list.scope.id) : "");
const reply = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, statusText: "HTTP error", json: async () => body });
function deferred() {
  let resolve;
  const promise = new Promise((yes) => { resolve = yes; });
  return { promise, resolve };
}
async function settle() { await new Promise((resolve) => setImmediate(resolve)); }
function harness({ phone = false, sid = defaultId, list = manifest(), respond } = {}) {
  const { document, node, $ } = dom();
  const window = node("#window"), entries = [{ base: true }], travel = [], calls = [];
  let index = 0, info = enabled, drawerOpens = 0;
  $("sessbar").onclick = () => drawerOpens++;
  const history = {
    get state() { return entries[index]; },
    pushState(state) { entries.splice(++index, entries.length, state); },
    go(delta) { travel.push(delta); },
  };
  const keyedMedia = (query) => { assert.equal(query, "(max-width: 700px)"); return { matches: phone }; };
  window.matchMedia = keyedMedia;
  const noop = () => {};
  const context = vm.createContext({
    $, document, window, history, AbortController, sessionId: sid, store: { base, token },
    fetch(url, { method = "GET", headers, signal, cache }) {
      const request = { url, path: url.slice(context.store.base.length), method, headers, signal, cache };
      calls.push(request);
      const fallback = request.path === "/info" ? info : request.path.endsWith("/changes/scopes") ? catalog() : list;
      return Promise.resolve(respond?.(request) ?? reply(fallback));
    },
    fillSelect: noop, syncModelTrigger: noop, syncPickTrigger: noop, syncOvrSummary: noop, updateAnswerReference: noop,
    pauseHistoryRetry: noop, abortEvents: noop, updateExecutionModel: noop, ensureTranscriptOwner: noop,
    syncStopButton: noop, restoreDraft: noop, resetQuestionNavigator: noop, restoreSkillPick: noop, restoreTeamPick: noop,
    localStorage: { setItem() {}, removeItem() {} }, booted: false, infoRetryTimer: null,
  });
  for (const region of [
    between("function headers(json)", "const esc ="),
    between("// BEGIN working-tree changes", "// END working-tree changes"),
    between("async function loadInfo()", "// The drawer only highlights"),
    between("// BEGIN session switch", "// END session switch"),
    script.slice(script.lastIndexOf("function isPhoneLayout()"), script.indexOf("// Dragging across the drawer")),
  ]) vm.runInContext(region, context);
  context.wireChanges();
  function flushHistory() {
    while (travel.length) {
      index += travel.shift();
      assert.ok(index >= 0 && index < entries.length, "viewer must not navigate away from the app");
      window.dispatch("popstate", { state: history.state });
    }
  }
  return {
    $, context, calls, document, window, history, flushHistory,
    rows: () => $("changes-files").children.map((item) => item.children[0]),
    drawerOpens: () => drawerOpens,
    async info(payload = enabled) { info = payload; await context.loadInfo(); },
    click: (id) => $(id).dispatch("click"),
    setScope(scope) {
      $("changes-scope").value = scope.id;
      $("changes-scope").dispatch("change");
    },
    key: (key, extra) => document.activeElement.dispatch("keydown", { key, ...extra }),
    back() { history.go(-1); flushHistory(); },
    resize(value) { phone = value; window.dispatch("resize"); },
  };
}
async function opened(options) {
  const h = harness(options);
  await h.info();
  h.click("changesbtn");
  return h;
}

test("the selected-session button requires the exact /info capability and stops drawer clicks", async () => {
  const h = harness({ sid: "" });
  assert.equal(h.$("changesbtn").hidden, true);
  await h.info();
  assert.equal(h.$("changesbtn").hidden, true);
  h.context.switchSession(defaultId, false);
  assert.equal(h.$("changesbtn").hidden, false);
  for (const capabilities of [{ changes: true }, { features: {} }, { features: { changes: false } }, { features: { changes: "true" } }]) {
    await h.info(capabilities);
    assert.equal(h.$("changesbtn").hidden, true);
  }
  await h.info();
  assert.equal(h.$("changesbtn").hidden, false);
  assert.equal(h.click("changesbtn").stopped, true);
  assert.equal(h.drawerOpens(), 0);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changesbtn").getAttribute("aria-expanded"), "true");
  assert.equal(h.document.activeElement, h.$("changes-close"));
  assert.equal(h.$("changes-list-status").textContent, "Loading changes…");
  h.context.switchSession("", false);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.$("changesbtn").hidden, true);
  h.flushHistory();
});

test("the file list and unified diff use literal text and the authenticated encoded routes", async () => {
  const hostile = "<img src=x onerror=alert(1)> & 🦊.js";
  const renamed = file("opaque/a?%+", hostile, { old_path: "old/α.js", status: "renamed" });
  const list = manifest([renamed, file("m", "m"), file("a", "a", { status: "added" }), file("d", "d", { status: "deleted" })], { root_label: "<sample> & repo" });
  const raw = "diff --git a/old b/new\n--- a/old\n+++ b/new\n@@ -1,2 +1,2 @@\n-old\n+++ literal code\n+" + hostile + "\n unchanged\n\\ No newline at end of file\n";
  const h = await opened({ list, respond: ({ path }) => path === detailPath(list, renamed) ? reply(diff(list, renamed, { diff: raw })) : undefined });
  await settle();
  assert.equal(h.$("changes-root").textContent, "<sample> & repo");
  assert.match(h.$("changes").textContent, /Working tree · uncommitted/);
  assert.equal(h.$("changes-count").textContent, "4");
  assert.deepEqual(h.rows().map((row) => row.children[0].textContent), ["R", "M", "A", "D"]);
  const row = h.rows()[0];
  assert.equal(row.children[1].textContent, "old/α.js → " + hostile);
  assert.deepEqual(row.children[1].children.map((line) => line.textContent), ["old/α.js", " → " + hostile]);
  assert.equal(row.title, row.children[1].textContent);
  assert.deepEqual(row.children[2].children.map((count) => count.textContent), ["+3", "−1"]);
  row.dispatch("click");
  assert.equal(row.getAttribute("aria-current"), "true");
  assert.equal(h.$("changes-detail-status").textContent, "Loading diff…");
  await settle();
  assert.equal(h.$("changes-diff").textContent, raw);
  assert.deepEqual(h.$("changes-diff").children.filter((line) => line.tagName === "span").map((line) => line.className),
    ["changes-line meta", "changes-line meta", "changes-line meta", "changes-line hunk", "changes-line del", "changes-line add", "changes-line add", "changes-line", "changes-line meta"]);
  assert.deepEqual(h.calls.map((call) => [call.method, call.url, call.headers.Authorization]),
    ["/info", listPath(), scopesPath(), detailPath(list, renamed)].map((route) => ["GET", base + route, "Bearer " + token]));
  assert.deepEqual(h.calls.slice(1).map((call) => call.cache), ["no-store", "no-store", "no-store"]);
});

test("rapid picks and refresh abort requests and ignore stale response bodies", async () => {
  const one = file("one", "first.txt"), two = file("two", "second.txt"), list = manifest([one, two]);
  const freshTwo = file("new-id", two.path), freshList = manifest([freshTwo], { revision: "next" });
  const slowBody = deferred(), refreshFlight = deferred();
  let fresh = false, picksOfTwo = 0;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && fresh) return reply(freshList);
    if (path === detailPath(list, one)) return { ...reply(null), json: () => slowBody.promise };
    if (path === detailPath(list, two)) return ++picksOfTwo === 1
      ? reply(diff(list, two, { diff: "@@ -1 +1 @@\n+second\n" })) : refreshFlight.promise;
    if (path === detailPath(freshList, freshTwo)) return reply(diff(freshList, freshTwo, { diff: "@@ -1 +1 @@\n+fresh\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  const oneSignal = h.calls.at(-1).signal;
  await settle();
  h.rows()[1].dispatch("click");
  assert.equal(oneSignal.aborted, true);
  await settle();
  slowBody.resolve(diff(list, one, { diff: "@@ -1 +1 @@\n+stale first\n" }));
  await settle();
  assert.match(h.$("changes-diff").textContent, /\+second/);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  h.rows()[1].dispatch("click");
  const twoSignal = h.calls.at(-1).signal;
  fresh = true;
  h.click("changes-refresh");
  assert.equal(twoSignal.aborted, true);
  await settle();
  refreshFlight.resolve(reply(diff(list, two, { diff: "@@ -1 +1 @@\n+stale second\n" })));
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [listPath(), detailPath(freshList, freshTwo), scopesPath()]);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "true");
  assert.equal(h.$("changes-diff").textContent, "@@ -1 +1 @@\n+fresh\n");
});

test("refresh does not match a lossy display path to a different file ID", async () => {
  const old = file("before", "name\uFFFD.txt"), replacement = file("after", old.path);
  const list = manifest([old]), freshList = manifest([replacement], { revision: "new" });
  let fresh = false;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && fresh) return reply(freshList);
    if (path === detailPath(list, old)) return reply(diff(list, old));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  fresh = true;
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath(), scopesPath()]);
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  assert.match(h.$("changes-detail-status").textContent, /Select a file/);
});

test("unavailable list errors stay inline and Retry can reveal the empty state", async () => {
  let unavailable = true;
  const h = await opened({ respond: ({ path }) => path === listPath() && unavailable ? reply({ error: "No repository" }, 404) : undefined });
  await settle();
  assert.match(h.$("changes-list-status").textContent, /Changes are unavailable for this session.*No repository/);
  assert.equal(h.$("changes-list-retry").hidden, false);
  h.$("changes-list-retry").focus();
  unavailable = false;
  h.click("changes-list-retry");
  await settle();
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-refresh"));
});

test("409 detail retry fetches a fresh revision rather than resending a stale file request", async () => {
  const item = file("before", "changed.txt"), freshItem = file("after", item.path);
  const list = manifest([item]), freshList = manifest([freshItem], { revision: "updated" });
  let changed = false;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === listPath() && changed) return reply(freshList);
    if (path === detailPath(list, item)) return reply({ error: "Working tree changed" }, 409);
    if (path === detailPath(freshList, freshItem)) return reply(diff(freshList, freshItem));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /Changes are unavailable for this session.*Working tree changed/);
  assert.equal(h.$("changes-detail-retry").hidden, false);
  changed = true;
  h.click("changes-detail-retry");
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [listPath(), detailPath(freshList, freshItem), scopesPath()]);
  assert.equal(h.$("changes-detail-status").textContent, "No textual differences.");
  assert.equal(h.$("changes-detail-retry").hidden, true);
});

test("binary, symlink, and truncated responses show the corresponding plain-text notes", async () => {
  const media = file("media", "banner.png", { binary: true, additions: null, deletions: null });
  const link = file("link", "latest", { status: "added", additions: null, deletions: null });
  const text = file("text", "notes.md");
  const list = manifest([media, link, text], { truncated: true, reason: "Only the first matches are shown." });
  const h = await opened({ list, respond: ({ path }) => {
    if (path === detailPath(list, media)) return reply(diff(list, media, { format: "none" }));
    if (path === detailPath(list, link)) return reply(diff(list, link, { format: "none", reason: "Symbolic link." }));
    if (path === detailPath(list, text)) return reply(diff(list, text, { diff: "@@ -1 +1 @@\n+preview", truncated: true, reason: "Size limit reached." }));
  } });
  await settle();
  assert.equal(h.$("changes-count").textContent, "3+");
  assert.match(h.$("changes-list-warning").textContent, /File list truncated.*Only the first/);
  assert.equal(h.rows()[0].children[2].textContent, "Binary");
  h.rows()[0].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /Binary file. No text diff/);
  assert.equal(h.$("changes-diff").hidden, true);
  h.rows()[1].dispatch("click");
  await settle();
  assert.match(h.$("changes-detail-status").textContent, /No text diff.*Symbolic link/);
  h.rows()[2].dispatch("click");
  await settle();
  assert.equal(h.$("changes-detail-warning").textContent, "Diff truncated. Size limit reached.");
  assert.match(h.$("changes-diff").textContent, /\+preview/);
});

test("mobile browser Back aborts the file request, restores the list, then closes", async () => {
  const item = file("one", "first.txt"), list = manifest([item]), pending = deferred();
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? pending.promise : undefined });
  await settle();
  assert.equal(h.$("changes-detail").hidden, true);
  h.rows()[0].dispatch("click");
  const detail = h.calls.at(-1);
  assert.equal(h.$("changes-listpane").hidden, true);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.document.activeElement, h.$("changes-back"));
  h.back();
  assert.equal(detail.signal.aborted, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.document.activeElement, h.rows()[0]);
  pending.resolve(reply(diff(list, item, { diff: "@@ -1 +1 @@\n+stale\n" })));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "");
  h.back();
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
});

test("mobile in-app Back followed by Close consumes only the viewer's history entries", async () => {
  const item = file("one", "first.txt"), list = manifest([item]);
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? reply(diff(list, item)) : undefined });
  await settle();
  h.rows()[0].dispatch("click");
  h.click("changes-back");
  assert.equal(h.$("changes-listpane").hidden, false);
  h.click("changes-close");
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  assert.equal(h.history.state.base, true);
  assert.equal(h.history.state.changes, undefined);
});

test("mobile refresh returns to the list when the selected file disappears", async () => {
  const item = file("one", "first.txt"), list = manifest([item]);
  let refreshed = false;
  const h = await opened({ phone: true, list, respond: ({ path }) => {
    if (path === detailPath(list, item)) return reply(diff(list, item));
    if (path === listPath() && refreshed) return reply(manifest([], { revision: "fresh" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  refreshed = true;
  h.click("changes-refresh");
  await settle();
  assert.equal(h.$("changes-listpane").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  h.flushHistory();
  assert.equal(h.history.state.changes.depth, 1);
  h.back();
  assert.equal(h.$("changes").hidden, true);
});

test("Escape from mobile detail and close from loading desktop abort and restore focus", async () => {
  const item = file("one", "first.txt"), list = manifest([item]), pending = deferred();
  const h = await opened({ phone: true, list, respond: ({ path }) => path === detailPath(list, item) ? pending.promise : undefined });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.key("Escape").defaultPrevented, true);
  assert.equal(h.calls.at(-1).signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
  h.flushHistory();
  assert.equal(h.history.state.changes, undefined);
  const laterList = deferred();
  const desktop = await opened({ respond: ({ path }) => path === listPath() ? laterList.promise : undefined });
  assert.equal(desktop.key("Tab").defaultPrevented, true);
  assert.equal(desktop.document.activeElement, desktop.$("changes-detail-scroll"));
  desktop.click("changes-close");
  assert.equal(desktop.calls.at(-1).signal.aborted, true);
  laterList.resolve(reply(list));
  await settle();
  assert.equal(desktop.rows().length, 0);
  assert.equal(desktop.document.activeElement, desktop.$("changesbtn"));
});

test("session and connection switches abort and prevent stale data or capability", async () => {
  const pending = deferred();
  const h = await opened({ respond: ({ path }) => path === listPath() ? pending.promise : undefined });
  const previousRequest = h.calls.at(-1);
  h.context.switchSession("other", false);
  assert.equal(previousRequest.signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  h.click("changesbtn");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath("other"), scopesPath("other")]);
  pending.resolve(reply(manifest([file("stale", "stale.txt")])));
  await settle();
  assert.equal(h.rows().length, 0);
  h.context.store.base = "https://older.example.invalid";
  h.window.dispatch("storage", { key: "serve.base" });
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.$("changesbtn").hidden, true);
  await h.info({});
  assert.equal(h.$("changesbtn").hidden, true);
});

test("older /info responses cannot re-enable a capability removed by the latest response", async () => {
  const pending = deferred();
  let calls = 0;
  const h = harness({ respond: ({ path }) => path === "/info" && calls++ === 0 ? pending.promise : undefined });
  const oldInfo = h.info();
  await h.info({ features: { changes: false } });
  pending.resolve(reply(enabled));
  await oldInfo;
  assert.equal(h.$("changesbtn").hidden, true);
});

test("an empty current scope automatically opens the only dirty worktree and can show current again", async () => {
  const worktree = scope("worktree/α?= +%", "<feature> & worktree", { files: 1 });
  const item = file("one", "first.txt"), list = scopedManifest(mainScope);
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [item]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath(), scopedListPath(worktree)]);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(h.rows().length, 1);
  assert.equal(h.$("changes-scope-notice").hidden, false);
  assert.equal(h.$("changes-scope-note").textContent, "No changes in session repo when opened; showing worktree <feature> & worktree.");
  const showCurrent = h.$("changes-show-current");
  assert.equal(showCurrent.hidden, false);
  showCurrent.focus();
  h.click("changes-show-current");
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(mainScope));
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-scope-notice").hidden, true);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.document.activeElement, h.$("changes-scope"));
  const beforeRefresh = h.calls.length;
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(beforeRefresh).map((call) => call.path), [scopedListPath(mainScope), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
});

test("multiple dirty worktrees are offered as keyboard-reachable empty-state choices", async () => {
  const first = scope("first", "first worktree", { files: 1 }), second = scope("second", "second worktree", { files: 2 });
  const clean = scope("clean", "clean worktree");
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog(mainScope, first, clean, second));
    if (path === scopedListPath(second)) return reply(scopedManifest(second, [file("chosen", "picked.txt")]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  const choices = h.$("changes-scope-choices");
  assert.equal(choices.hidden, false);
  assert.match(h.$("changes-list-status").textContent, /Choose a worktree with changes/);
  const buttons = choices.children.map((row) => row.children[0]);
  assert.deepEqual(buttons.map((button) => button.textContent), [first.label, second.label]);
  h.key("Tab");
  h.key("Tab");
  assert.equal(h.document.activeElement, buttons[0]);
  h.key("Tab");
  assert.equal(h.document.activeElement, buttons[1]);
  h.document.activeElement.dispatch("click");
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(second));
  assert.equal(h.$("changes-scope").value, second.id);
  assert.equal(h.rows()[0].title, "picked.txt");
  assert.equal(choices.hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-scope"));
});

test("scope changes abort list and diff requests and cannot retain a shared file ID across scopes", async () => {
  const worktree = scope("feature/α?= +%", "feature", { files: 1 });
  const item = file("same/id", "shared.txt");
  const list = scopedManifest(mainScope, [item]), worktreeList = scopedManifest(worktree, [item]);
  const staleDiff = deferred(), staleList = deferred();
  let currentDiffs = 0;
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return reply(catalog({ ...mainScope, changed: true, files: 1 }, worktree));
    if (path === scopedListPath(worktree)) return reply(worktreeList);
    if (path === scopedListPath(mainScope)) return staleList.promise;
    if (path === detailPath(list, item)) return ++currentDiffs === 1
      ? reply(diff(list, item, { diff: "+current\n" })) : staleDiff.promise;
    if (path === detailPath(worktreeList, item)) return reply(diff(worktreeList, item, { diff: "+worktree\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, detailPath(list, item));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+current\n");
  h.rows()[0].dispatch("click");
  const previousDiff = h.calls.at(-1);
  h.setScope(worktree);
  assert.equal(previousDiff.signal.aborted, true);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.equal(h.$("changes-path").textContent, "File diff");
  assert.equal(h.calls.at(-1).path, scopedListPath(worktree));
  await settle();
  assert.equal(h.rows()[0].getAttribute("aria-current"), "false");
  h.rows()[0].dispatch("click");
  assert.equal(h.calls.at(-1).path, detailPath(worktreeList, item));
  staleDiff.resolve(reply(diff(list, item, { diff: "+stale\n" })));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+worktree\n");
  h.setScope(mainScope);
  const previousList = h.calls.at(-1);
  assert.equal(previousList.path, scopedListPath(mainScope));
  h.setScope(worktree);
  assert.equal(previousList.signal.aborted, true);
  staleList.resolve(reply(scopedManifest(mainScope, [file("stale", "stale.txt")])));
  await settle();
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.deepEqual(h.rows().map((row) => row.title), [item.path]);
});

test("refresh preserves the worktree and mobile detail until the scope vanishes, then returns to current", async () => {
  const worktree = scope("alternate", "feature", { files: 1 });
  const item = file("first-id", "worktree.txt"), nextItem = file("next-id", item.path);
  const worktreeList = scopedManifest(worktree, [item]);
  const refreshed = scopedManifest(worktree, [nextItem], { revision: "next" });
  const currentItem = file("current", "current.txt");
  let vanish = false, worktreeLoads = 0;
  const h = await opened({ phone: true, list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === listPath() && vanish) return reply(scopedManifest(mainScope, [currentItem]));
    if (path === scopesPath()) return reply(vanish ? catalog(mainScope) : catalog(mainScope, worktree));
    if (path === scopedListPath(worktree)) return vanish ? reply({ error: "Unknown scope" }, 400)
      : reply(++worktreeLoads === 1 ? worktreeList : refreshed);
    if (path === detailPath(worktreeList, item)) return reply(diff(worktreeList, item));
    if (path === detailPath(refreshed, nextItem)) return reply(diff(refreshed, nextItem, { diff: "+refreshed\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [scopedListPath(worktree), detailPath(refreshed, nextItem), scopesPath()]);
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "+refreshed\n");
  vanish = true;
  h.click("changes-refresh");
  await settle();
  h.flushHistory();
  assert.deepEqual(h.calls.slice(-3).map((call) => call.path), [scopedListPath(worktree), listPath(), scopesPath()]);
  assert.equal(h.$("changes").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.rows()[0].title, currentItem.path);
  assert.match(h.$("changes-scope-note").textContent, /feature is no longer available.*session directory/);
  assert.equal(h.history.state.changes.depth, 1);
  h.back();
  assert.equal(h.$("changes").hidden, true);
  assert.equal(h.document.activeElement, h.$("changesbtn"));
});

test("the labeled scope select is in the desktop Tab loop", async () => {
  const worktree = scope("alternate", "another worktree");
  const list = scopedManifest(mainScope, [file("one", "first.txt")]);
  const h = await opened({ list, respond: ({ path }) => path === scopesPath() ? reply(catalog(mainScope, worktree)) : undefined });
  await settle();
  const picker = h.$("changes-scope");
  assert.equal(h.$("changes-scope-label").getAttribute("for"), "changes-scope");
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, picker);
  h.key("Tab");
  assert.equal(h.document.activeElement, h.rows()[0]);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, picker);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, h.$("changes-close"));
});

test("slow or failed scope discovery does not block a nonempty list or its diff", async () => {
  const item = file("one", "first.txt"), list = scopedManifest(mainScope, [item]);
  const pendingScopes = deferred();
  const h = await opened({ list, respond: ({ path }) => {
    if (path === scopesPath()) return pendingScopes.promise;
    if (path === detailPath(list, item)) return reply(diff(list, item, { diff: "+visible\n" }));
  } });
  await settle();
  const lookup = h.calls.at(-1);
  assert.equal(lookup.path, scopesPath());
  assert.equal(h.rows().length, 1);
  h.rows()[0].dispatch("click");
  await settle();
  assert.equal(lookup.signal.aborted, false);
  pendingScopes.resolve(reply({ error: "Scope discovery failed" }, 500));
  await settle();
  assert.equal(h.$("changes-diff").textContent, "+visible\n");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, true);
});

test("an empty truncated file list does not auto-select a dirty worktree", async () => {
  const worktree = scope("alternate", "another worktree", { files: 1 });
  const list = scopedManifest(mainScope, [], { truncated: true, reason: "File list limit reached." });
  const h = await opened({ list, respond: ({ path }) => path === scopesPath() ? reply(catalog(mainScope, worktree)) : undefined });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-list-status").textContent, "No files could be listed.");
  assert.equal(h.$("changes-list-warning").textContent, "File list truncated. File list limit reached.");
  assert.equal(h.$("changes-scope-notice").hidden, true);
});

test("a truncated worktree catalog never auto-selects but keeps known worktrees selectable", async () => {
  const worktree = scope("alternate", "another worktree", { files: 1 });
  const item = file("one", "worktree.txt");
  const partial = { ...catalog(mainScope, worktree), truncated: true, reason: "Worktree scan timed out." };
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => {
    if (path === scopesPath()) return reply(partial);
    if (path === scopedListPath(worktree)) return reply(scopedManifest(worktree, [item]));
  } });
  await settle();
  assert.deepEqual(h.calls.map((call) => call.path), ["/info", listPath(), scopesPath()]);
  assert.equal(h.$("changes-scope").value, mainScope.id);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(h.$("changes-scope-notice").hidden, true);
  assert.match(h.$("changes-scopes-status").textContent, /Worktree scan incomplete.*timed out/);
  h.setScope(worktree);
  await settle();
  assert.equal(h.calls.at(-1).path, scopedListPath(worktree));
  assert.equal(h.$("changes-scope").value, worktree.id);
  assert.equal(h.rows()[0].title, item.path);
});

test("only a complete refreshed catalog missing the selected worktree returns to the session directory", async () => {
  const worktree = scope("alternate", "feature", { files: 1 });
  const oldItem = file("alternate-file", "worktree.txt"), currentItem = file("current-file", "current.txt");
  const worktreeList = scopedManifest(worktree, [oldItem]);
  let currentList = scopedManifest(mainScope), nextCatalog = reply(catalog(mainScope, worktree));
  const h = await opened({ phone: true, respond: ({ path }) => {
    if (path === listPath()) return reply(currentList);
    if (path === scopesPath()) return nextCatalog;
    if (path === scopedListPath(worktree)) return reply(worktreeList);
    if (path === detailPath(worktreeList, oldItem)) return reply(diff(worktreeList, oldItem, { diff: "+old\n" }));
  } });
  await settle();
  h.rows()[0].dispatch("click");
  assert.equal(h.history.state.changes.depth, 2);
  const picker = h.$("changes-scope");
  nextCatalog = reply({ ...catalog(mainScope), truncated: true });
  h.click("changes-refresh");
  await settle();
  assert.equal(picker.value, worktree.id);
  assert.ok(picker.children.some((option) => option.value === worktree.id));
  assert.equal(h.$("changes-detail").hidden, false);
  assert.equal(h.$("changes-diff").textContent, "+old\n");
  assert.doesNotMatch(h.$("changes-scope-note").textContent, /no longer available/);

  const pendingCatalog = deferred(), beforeRefresh = h.calls.length;
  nextCatalog = pendingCatalog.promise;
  currentList = scopedManifest(mainScope, [currentItem]);
  h.click("changes-refresh");
  await settle();
  assert.equal(h.rows()[0].title, oldItem.path);
  assert.equal(h.$("changes-diff").textContent, "+old\n");
  pendingCatalog.resolve(reply(catalog(mainScope)));
  await settle();
  h.flushHistory();
  assert.deepEqual(h.calls.slice(beforeRefresh).map((call) => call.path),
    [scopedListPath(worktree), detailPath(worktreeList, oldItem), scopesPath(), listPath()]);
  assert.equal(picker.value, mainScope.id);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(h.$("changes-listpane").hidden, false);
  assert.equal(h.$("changes-detail").hidden, true);
  assert.equal(h.$("changes-diff").textContent, "");
  assert.deepEqual(h.rows().map((row) => row.title), [currentItem.path]);
  assert.match(h.$("changes-scope-note").textContent, /feature is no longer available.*session directory/);
  assert.equal(h.history.state.changes.depth, 1);
});

test("a 503 on refresh invalidates cached choices and Retry only rediscovers worktrees", async () => {
  const first = scope("first", "first worktree", { files: 1 }), second = scope("second", "second worktree", { files: 2 });
  const fresh = scope("fresh", "fresh worktree", { files: 1 });
  let nextCatalog = reply(catalog(mainScope, first, second));
  const h = await opened({ phone: true, list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? nextCatalog : undefined });
  await settle();
  const picker = h.$("changes-scope"), choices = h.$("changes-scope-choices");
  assert.equal(choices.hidden, false);
  picker.focus();
  nextCatalog = reply({ error: "Worktree discovery temporarily unavailable." }, 503);
  h.click("changes-refresh");
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [scopedListPath(mainScope), scopesPath()]);
  assert.deepEqual(picker.children.map((option) => option.value), [mainScope.id]);
  assert.equal(h.$("changes-scope-control").hidden, true);
  assert.equal(choices.hidden, true);
  assert.equal(choices.children.length, 0);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-count").textContent, "0");
  assert.equal(h.$("changes-list-retry").hidden, true);
  assert.equal(h.document.activeElement, h.$("changes-refresh"));

  const status = h.$("changes-scopes-status"), retry = h.$("changes-scopes-retry");
  assert.equal(status.getAttribute("role"), "status");
  assert.match(status.textContent, /Could not discover worktrees.*503.*temporarily unavailable/);
  assert.equal(retry.getAttribute("aria-describedby"), "changes-scopes-status");
  h.$("changes-close").focus();
  assert.equal(h.key("Tab").defaultPrevented, true);
  assert.equal(h.document.activeElement, retry);
  const pendingRetry = deferred(), beforeRetry = h.calls.length;
  nextCatalog = pendingRetry.promise;
  h.click("changes-scopes-retry");
  assert.equal(h.document.activeElement, h.$("changes-refresh"));
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-count").textContent, "0");
  pendingRetry.resolve(reply(catalog(mainScope, fresh)));
  await settle();
  assert.deepEqual(h.calls.slice(beforeRetry).map((call) => call.path), [scopesPath()]);
  assert.equal(status.hidden, true);
  assert.equal(retry.hidden, true);
  assert.equal(h.$("changes-scope-control").hidden, false);
  assert.equal(picker.value, mainScope.id);
  assert.deepEqual(picker.children.map((option) => option.value), [mainScope.id, fresh.id]);
});

test("a stale scope catalog is aborted on session switch and cannot auto-select in the new session", async () => {
  const pendingScopes = deferred();
  const worktree = scope("stale", "old worktree", { files: 1 });
  const h = await opened({ list: scopedManifest(mainScope), respond: ({ path }) => path === scopesPath() ? pendingScopes.promise : undefined });
  await settle();
  const lookup = h.calls.at(-1);
  assert.equal(lookup.path, scopesPath());
  h.context.switchSession("other", false);
  assert.equal(lookup.signal.aborted, true);
  assert.equal(h.$("changes").hidden, true);
  h.flushHistory();
  h.click("changesbtn");
  pendingScopes.resolve(reply(catalog(mainScope, worktree)));
  await settle();
  assert.deepEqual(h.calls.slice(-2).map((call) => call.path), [listPath("other"), scopesPath("other")]);
  assert.equal(h.$("changes-list-status").textContent, "No uncommitted changes");
  assert.equal(h.$("changes-scope-control").hidden, true);
});
