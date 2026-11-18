// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_session_search.test.cjs
// Exercise server-backed search through the real input handlers and pager.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const fixtureFile = path.join(__dirname, "app_mark.test.cjs");
const { harness, response, deferred, settle, sessions, longSessions, script, html } = new vm.Script(
  fs.readFileSync(fixtureFile, "utf8") + "\n;({ harness, response, deferred, settle, sessions, longSessions, script, html });",
  { filename: fixtureFile },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: fixtureFile, AbortController, URL, setImmediate,
});
const plain = (value) => JSON.parse(JSON.stringify(value));
const run = (h, source) => vm.runInContext(source, h.context);
const ids = (h) => plain(h.context.lastItems.map((item) => item.id));
const requests = (h) => h.requests.filter((request) => request.path === "/sessions");
const pager = (h) => plain(run(h, "({ total: sessionsTotal, offset: sessionsPageOffset, hasMore: sessionsHasMore, loading: sessionsLoadingMore })"));
async function input(h, value) {
  h.elements.sessfilter.value = value;
  await h.dispatch(h.elements.sessfilter, "input").done;
}
async function search(h, value) {
  await input(h, value);
  h.advance(250);
  await settle();
}
function page(items, total = items.length, size = items.length) {
  return response(items, 200, { "x-sessions-total": String(total), "x-sessions-page-size": String(size) });
}
function shortcut(h, target, extra = {}) {
  const event = { target, key: "f", shiftKey: true, ctrlKey: true, ...extra,
    prevented: false, stopped: false,
    preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; } };
  for (const listener of h.documentListeners.get("keydown") || []) listener(event);
  return event;
}

for (const modifier of ["ctrlKey", "metaKey"]) for (const key of ["f", "F"]) {
  test("session search shortcut opens from the composer and closes from the search input (" + modifier + ", " + key + ")", () => {
    const h = harness(), filter = h.elements.sessfilter;
    assert.equal(filter.hidden, true);
    assert.match(html, /<input\b[^>]*id="sessfilter"[^>]*\shidden(?:\s|>)/);
    h.elements.prompt.focus();
    const modifiers = { ctrlKey: false, metaKey: false, [modifier]: true, key };
    const open = shortcut(h, h.elements.prompt, modifiers);
    assert.equal(open.prevented, true);
    assert.equal(open.stopped, true);
    assert.equal(filter.hidden, false);
    assert.equal(h.document.activeElement, filter);
    assert.equal(h.elements.sesssearch.getAttribute("aria-expanded"), "true");
    assert.equal(h.requests.length, 0, "opening search does not request an empty result page");
    const repeat = shortcut(h, filter, { ...modifiers, repeat: true });
    assert.equal(repeat.prevented, true);
    assert.equal(filter.hidden, false, "holding the shortcut does not toggle repeatedly");
    const close = shortcut(h, filter, modifiers);
    assert.equal(close.prevented, true);
    assert.equal(close.stopped, true);
    assert.equal(filter.hidden, true);
    assert.equal(h.document.activeElement, h.elements.prompt);
    assert.equal(h.elements.sesssearch.getAttribute("aria-expanded"), "false");
  });
}

test("ordinary find keys and composing keys do not open session search", () => {
  const h = harness();
  for (const extra of [{ shiftKey: false }, { ctrlKey: false }, { key: "g" }, { isComposing: true }]) {
    const event = shortcut(h, h.elements.prompt, extra);
    assert.equal(event.prevented, false);
    assert.equal(event.stopped, false);
    assert.equal(h.elements.sessfilter.hidden, true);
  }
});

for (const phone of [false, true]) test("More offers keyboard-free search and restores menu focus on Escape (phone=" + phone + ")", async () => {
  const h = harness({ phone }), filter = h.elements.sessfilter;
  assert.match(html, /<button\b[^>]*id="sesssearch"[^>]*aria-controls="sessfilter"[^>]*>Search sessions…<\/button>/);
  assert.equal(h.elements.sessmorepanel.contains(h.elements.sesssearch), true);
  h.elements.sessmore.open = true;
  h.elements.sesssearch.focus();
  await h.dispatch(h.elements.sesssearch, "click").done;
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(filter.hidden, false);
  assert.equal(h.document.activeElement, filter);
  assert.deepEqual(plain(h.drawerChanges), phone ? [true] : []);
  const escape = h.dispatch(filter, "keydown", { key: "Escape" }).event;
  assert.equal(escape.prevented, true);
  assert.equal(escape.stopped, true);
  assert.equal(filter.hidden, true);
  assert.equal(h.document.activeElement, h.elements.sessmoretoggle);
  assert.equal(h.requests.length, 0);
});

test("empty search collapses on blur while a committed query survives Escape, shortcut and blur", async () => {
  const h = harness(), filter = h.elements.sessfilter;
  shortcut(h, h.elements.prompt);
  h.elements.prompt.focus();
  h.dispatch(filter, "blur");
  assert.equal(filter.hidden, true);
  shortcut(h, h.elements.prompt);
  await search(h, "target");
  const before = requests(h).length;
  h.dispatch(filter, "keydown", { key: "Escape" });
  shortcut(h, filter);
  assert.equal(filter.hidden, false);
  assert.equal(filter.value, "target");
  h.elements.prompt.focus();
  h.dispatch(filter, "blur");
  assert.equal(filter.hidden, false);
  assert.deepEqual(ids(h), ["target-session"]);
  assert.equal(requests(h).length, before, "visibility changes do not restart the query");
  shortcut(h, h.elements.prompt);
  assert.equal(h.document.activeElement, filter, "a shortcut from another input refocuses the existing query");
  h.elements.prompt.focus();
  await search(h, "");
  assert.equal(filter.hidden, true);
  assert.equal(requests(h).at(-1).searchParams.has("query"), false);
  assert.ok(h.row("open-session"));
});

test("clearing a focused search keeps it editable until blur and Escape does not cancel IME", async () => {
  const h = harness(), filter = h.elements.sessfilter;
  shortcut(h, h.elements.prompt);
  await search(h, "target");
  await input(h, "");
  assert.equal(filter.hidden, false);
  assert.equal(h.document.activeElement, filter);
  h.dispatch(filter, "compositionstart");
  const escape = h.dispatch(filter, "keydown", { key: "Escape", isComposing: true }).event;
  assert.equal(escape.prevented, false);
  assert.equal(filter.hidden, false);
  h.dispatch(filter, "compositionend");
  h.elements.prompt.focus();
  h.dispatch(filter, "blur");
  assert.equal(filter.hidden, true);
  h.advance(250);
  await settle();
  assert.equal(requests(h).at(-1).searchParams.has("query"), false);
});

test("search finds an off-first-page session with reordered keywords and preserves current metadata", async () => {
  const items = longSessions();
  items[0].cwd = "/tmp/current-project";
  items[1].summary = "Review Conciliator MR #79";
  const h = harness({ items });
  await h.context.loadSessions();
  assert.equal(h.row(items[1].id), undefined, "the target is outside the initially loaded page");
  const before = { title: h.elements.sesstitle.textContent, meta: h.elements.sessmeta.textContent, cwd: h.elements.sesscwd.textContent };
  await search(h, "79 Conciliator");
  assert.deepEqual(ids(h), [items[1].id]);
  assert.equal(requests(h).at(-1).searchParams.get("query"), "79 Conciliator");
  assert.equal(requests(h).at(-1).searchParams.has("include"), false);
  assert.equal(h.elements.sesscount.textContent, "1 matches · showing 1");
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.row("open-session"), undefined, "a nonmatching selection is not appended to search results");
  assert.deepEqual({ title: h.elements.sesstitle.textContent, meta: h.elements.sessmeta.textContent, cwd: h.elements.sesscwd.textContent }, before);
  assert.equal(h.elements.prompt.value, "draft in A");
  assert.deepEqual(plain(h.switches), []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
});

test("search input debounces and resets paging before requesting the latest query", async () => {
  const h = harness({ items: longSessions() });
  await h.context.loadSessions();
  await h.context.loadOlderSessions();
  await input(h, "Older");
  h.advance(200);
  await input(h, "session Older");
  assert.deepEqual(pager(h), { total: null, offset: 0, hasMore: false, loading: false });
  assert.equal(h.elements.sesscount.textContent, "searching sessions…");
  h.advance(249);
  assert.equal(requests(h).length, 2);
  h.advance(1);
  await settle();
  assert.equal(requests(h).length, 3);
  assert.equal(requests(h).at(-1).searchParams.get("query"), "session Older");
  assert.equal(requests(h).at(-1).searchParams.get("offset"), "0");
  assert.equal(requests(h).at(-1).searchParams.get("limit"), "12");
  assert.equal(h.elements.sesscount.textContent, "100 matches · showing 12 loaded");
});

test("cwd updates still reach the current session while it is absent from search results", async () => {
  const items = sessions();
  items[0].cwd = "/tmp/original-project";
  const h = harness({ items });
  await search(h, "target");
  vm.runInContext(script.slice(script.indexOf("function applySessionCwd("), script.indexOf("// BEGIN composer session adoption")), h.context);
  run(h, "applySessionCwd('open-session', '/tmp/new-project');");
  assert.equal(h.elements.sesscwd.textContent, "📁 /tmp/new-project");
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.deepEqual(ids(h), ["target-session"]);
});

test("search paging carries the query through the explicit button, scroll and wheel paths", async () => {
  const h = harness({ items: longSessions() }), rail = h.elements.sesslist;
  await search(h, "session Older");
  await h.dispatch(rail.querySelector(".session-more").querySelector("button"), "click").done;
  assert.equal(h.elements.sesscount.textContent, "100 matches · showing 24 loaded");
  rail.scrollHeight = 1000;
  rail.clientHeight = 100;
  rail.scrollTop = 900;
  h.dispatch(rail, "scroll");
  await settle();
  h.dispatch(rail, "wheel", { deltaY: 1 });
  await settle();
  assert.deepEqual(plain(requests(h).map((request) => request.searchParams.get("offset"))), ["0", "12", "24", "36"]);
  assert.ok(requests(h).every((request) => request.searchParams.get("query") === "session Older" && !request.searchParams.has("include")));
  assert.equal(h.elements.sesscount.textContent, "100 matches · showing 48 loaded");
  assert.equal(rail.scrollTop, 900);
});

for (const query of ["target", "missing"]) test("search membership never changes a lazy current session (query=" + query + ")", async () => {
  const h = harness({ sid: "lazy-session" });
  h.context.pendingNew = "lazy-session";
  h.context.pendingCwd = "/tmp/lazy-project";
  h.context.renderSessionBar();
  await search(h, query);
  assert.equal(h.context.sessionId, "lazy-session");
  assert.equal(h.context.pendingNew, "lazy-session");
  assert.equal(h.context.lastMsgCount, 5);
  assert.equal(h.context.turnRunning, true);
  assert.equal(h.elements.sesscwd.textContent, "📁 /tmp/lazy-project");
  assert.deepEqual(plain(h.switches), []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
});

test("an empty search preserves the chat and clearing restores the normal page and total", async () => {
  const h = harness({ items: longSessions() });
  await h.context.loadSessions();
  await search(h, "missing");
  assert.equal(h.elements.sesscount.textContent, "0 matches · showing 0");
  assert.equal(h.elements.sesslist.textContent, "没有匹配的会话");
  assert.equal(h.context.sessionId, "open-session");
  await search(h, " \t ");
  const request = requests(h).at(-1);
  assert.equal(request.searchParams.has("query"), false);
  assert.equal(request.searchParams.get("include"), "open-session");
  assert.equal(request.searchParams.get("offset"), "0");
  assert.equal(h.elements.sesscount.textContent, "102 sessions");
  assert.ok(h.row("open-session"));
  assert.equal(h.row("open-session").classList.contains("cur"), true);
  assert.deepEqual(plain(h.switches), []);
  assert.equal(h.elements.prompt.value, "draft in A");
});

test("clear-search refresh preserves selection even if the returned page omits it", async () => {
  const h = harness({ sessionsGet: () => page([sessions()[1]]) });
  await search(h, "target");
  await search(h, "");
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.elements.sesstitle.textContent, "Open session");
  assert.deepEqual(plain(h.switches), []);
});

for (const initialQuery of ["", "open"]) test("a late response cannot replace newer search rows or pager (previous query=" + initialQuery + ")", async () => {
  const stale = deferred();
  const h = harness({ sessionsGet: (request) => request.searchParams.get("query") === "target" ? page([sessions()[1]]) : stale.promise });
  if (initialQuery) run(h, "sessFilter = 'open';");
  const oldLoad = h.context.loadSessions();
  await input(h, "target");
  stale.resolve(page([sessions()[0]], 99, 12));
  await oldLoad;
  assert.deepEqual(ids(h), [], "old rows stay discarded during the debounce gap");
  assert.deepEqual(pager(h), { total: null, offset: 0, hasMore: false, loading: false });
  h.advance(250);
  await settle();
  assert.deepEqual(ids(h), ["target-session"]);
  assert.deepEqual(pager(h), { total: 1, offset: 1, hasMore: false, loading: false });
  assert.equal(h.context.sessionId, "open-session");
});

test("a response from before clearing the query cannot overwrite the restored list", async () => {
  const stale = deferred();
  const h = harness({ sessionsGet: (request) => request.searchParams.has("query") ? stale.promise : page(sessions()) });
  await search(h, "target");
  await search(h, "");
  stale.resolve(page([sessions()[1]], 50, 12));
  await settle();
  assert.deepEqual(ids(h), ["open-session", "target-session"]);
  assert.equal(h.elements.sesscount.textContent, "2 sessions");
  assert.deepEqual(pager(h), { total: 2, offset: 2, hasMore: false, loading: false });
});

test("a previous server's search response is discarded", async () => {
  const stale = deferred();
  const h = harness({ sessionsGet: () => stale.promise });
  await search(h, "target");
  h.context.store.base = "https://other.test";
  stale.resolve(page([sessions()[1]]));
  await settle();
  assert.deepEqual(ids(h), []);
  assert.equal(pager(h).total, null);
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(plain(h.switches), []);
});

test("old search paging cannot release the current query's loading lock", async () => {
  const oldPage = deferred(), currentPage = deferred();
  const h = harness({ sessionsGet: (request) => {
    const query = request.searchParams.get("query"), offset = request.searchParams.get("offset");
    if (offset === "12") return query === "open" ? oldPage.promise : currentPage.promise;
    return page([query === "open" ? sessions()[0] : sessions()[1]], 30, 12);
  } });
  await search(h, "open");
  const oldLoad = h.context.loadOlderSessions();
  await search(h, "target");
  const currentLoad = h.context.loadOlderSessions();
  oldPage.resolve(page([{ id: "old-open", summary: "Open obsolete" }], 99, 12));
  await oldLoad;
  assert.equal(pager(h).loading, true);
  assert.deepEqual(ids(h), ["target-session"]);
  assert.equal(pager(h).offset, 12);
  currentPage.resolve(page([{ id: "new-target", summary: "Target older" }], 30, 12));
  await currentLoad;
  assert.deepEqual(ids(h), ["target-session", "new-target"]);
  assert.deepEqual(pager(h), { total: 30, offset: 24, hasMore: true, loading: false });
});

test("starred search returns an off-page match on its first filtered page and updates totals when unmarking", async () => {
  const items = longSessions();
  items.find((item) => item.id === "older-session-80").marked = true;
  const h = harness({ items });
  await search(h, "session Older");
  await h.dispatch(h.elements.sessmarked, "click").done;
  const request = requests(h).at(-1);
  assert.equal(request.searchParams.get("query"), "session Older");
  assert.equal(request.searchParams.get("marked"), "true");
  assert.equal(request.searchParams.get("offset"), "0");
  assert.equal(request.searchParams.has("include"), false);
  assert.deepEqual(ids(h), ["older-session-80"]);
  assert.ok(h.row("older-session-80"));
  assert.equal(h.elements.sesscount.textContent, "1 matches · showing 1");
  assert.deepEqual(pager(h), { total: 1, offset: 1, hasMore: false, loading: false });
  await h.dispatch(h.row("older-session-80").querySelector(".mk"), "click").done;
  assert.equal(h.row("older-session-80"), undefined);
  assert.equal(h.elements.sesscount.textContent, "0 matches · showing 0");
  assert.equal(h.elements.sesslist.textContent, "没有匹配的星标会话");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(plain(h.switches), []);
});

for (const query of ["", "session Older"]) test("starred pagination and clearing refetch the full matching set (query=" + query + ")", async () => {
  const items = longSessions();
  for (const item of items.filter((item) => item.id.startsWith("older-session-")).slice(0, 25)) item.marked = true;
  const h = harness({ items });
  await h.context.loadSessions();
  await h.context.loadOlderSessions();
  if (query) await search(h, query);
  const before = { title: h.elements.sesstitle.textContent, meta: h.elements.sessmeta.textContent, cwd: h.elements.sesscwd.textContent };
  await h.dispatch(h.elements.sessmarked, "click").done;
  const first = requests(h).at(-1);
  assert.equal(first.searchParams.get("marked"), "true");
  assert.equal(first.searchParams.get("query"), query || null);
  assert.equal(first.searchParams.get("offset"), "0");
  assert.equal(first.searchParams.get("limit"), "12");
  assert.equal(first.searchParams.has("include"), false);
  assert.ok(ids(h).every((id) => Number(id.split("-").at(-1)) < 25), "off-page stars fill the first filtered page");
  assert.equal(ids(h).length, 12);
  assert.equal(h.row("open-session"), undefined);
  assert.deepEqual(pager(h), { total: 25, offset: 12, hasMore: true, loading: false });
  for (const offset of ["12", "24"]) {
    await h.dispatch(h.elements.sesslist.querySelector(".session-more").querySelector("button"), "click").done;
    const next = requests(h).at(-1);
    assert.equal(next.searchParams.get("offset"), offset);
    assert.equal(next.searchParams.get("marked"), "true");
    assert.equal(next.searchParams.get("query"), query || null);
    assert.equal(next.searchParams.has("include"), false);
  }
  assert.equal(ids(h).length, 25);
  assert.equal(new Set(ids(h)).size, 25);
  assert.deepEqual(pager(h), { total: 25, offset: 25, hasMore: false, loading: false });
  assert.equal(h.elements.sesscount.textContent, query ? "25 matches · showing 25" : "25 sessions · showing 25");
  assert.deepEqual({ title: h.elements.sesstitle.textContent, meta: h.elements.sessmeta.textContent, cwd: h.elements.sesscwd.textContent }, before);
  await h.dispatch(h.elements.sessall, "click").done;
  const cleared = requests(h).at(-1);
  assert.equal(cleared.searchParams.has("marked"), false);
  assert.equal(cleared.searchParams.get("offset"), "0");
  assert.equal(cleared.searchParams.get("limit"), "12");
  assert.equal(cleared.searchParams.get("query"), query || null);
  assert.equal(cleared.searchParams.get("include"), query ? null : "open-session");
  assert.equal(h.elements.sesscount.textContent, query ? "100 matches · showing 12 loaded" : "102 sessions");
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(plain(h.switches), []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
  assert.equal(h.elements.prompt.value, "draft in A");
});

for (const initialMarked of [false, true]) test("a stale marked-state response cannot overwrite a newer toggle (initial marked=" + initialMarked + ")", async () => {
  const stale = deferred(), items = sessions(true);
  const h = harness({ sessionsGet: (request) => (request.searchParams.get("marked") === "true") === initialMarked
    ? stale.promise : page(initialMarked ? items : [items[1]]) });
  let oldLoad;
  if (initialMarked) oldLoad = h.dispatch(h.elements.sessmarked, "click").done;
  else oldLoad = h.context.loadSessions();
  await h.dispatch(h.elements[initialMarked ? "sessall" : "sessmarked"], "click").done;
  const currentIds = ids(h), currentPager = pager(h), currentCount = h.elements.sesscount.textContent;
  stale.resolve(page(initialMarked ? [items[1]] : items, 99, 12));
  await oldLoad;
  assert.deepEqual(ids(h), currentIds);
  assert.deepEqual(pager(h), currentPager);
  assert.equal(h.elements.sesscount.textContent, currentCount);
  assert.equal(h.context.sessionId, "open-session");
  assert.deepEqual(plain(h.switches), []);
});

test("old unstarred paging cannot release the starred page loading lock", async () => {
  const oldPage = deferred(), starredPage = deferred(), items = sessions(true);
  const h = harness({ sessionsGet: (request) => {
    const marked = request.searchParams.get("marked") === "true";
    if (request.searchParams.get("offset") === "12") return marked ? starredPage.promise : oldPage.promise;
    return page(marked ? [items[1]] : items, marked ? 25 : 100, 12);
  } });
  await h.context.loadSessions();
  const oldLoad = h.context.loadOlderSessions();
  await h.dispatch(h.elements.sessmarked, "click").done;
  const currentLoad = h.context.loadOlderSessions();
  oldPage.resolve(page([{ id: "obsolete", summary: "Old" }], 99, 12));
  await oldLoad;
  assert.equal(pager(h).loading, true);
  assert.deepEqual(ids(h), ["target-session"]);
  assert.equal(pager(h).offset, 12);
  starredPage.resolve(page([{ id: "older-star", summary: "Older star", marked: true }], 25, 12));
  await currentLoad;
  assert.deepEqual(ids(h), ["target-session", "older-star"]);
  assert.deepEqual(pager(h), { total: 25, offset: 24, hasMore: true, loading: false });
});

test("a starred toggle consumes the pending query without a duplicate debounced request", async () => {
  const h = harness({ items: sessions(true) });
  await input(h, "target");
  await h.dispatch(h.elements.sessmarked, "click").done;
  h.advance(500);
  await settle();
  assert.equal(requests(h).length, 1);
  assert.equal(requests(h)[0].searchParams.get("query"), "target");
  assert.equal(requests(h)[0].searchParams.get("marked"), "true");
  assert.deepEqual(ids(h), ["target-session"]);
});

test("starred-only refresh keeps a lazy current session and empty result membership", async () => {
  const h = harness({ sid: "lazy-session" });
  h.context.pendingNew = "lazy-session";
  h.context.pendingCwd = "/tmp/lazy-project";
  await h.dispatch(h.elements.sessmarked, "click").done;
  await h.context.loadSessions();
  assert.equal(h.context.sessionId, "lazy-session");
  assert.equal(h.context.pendingNew, "lazy-session");
  assert.equal(h.context.turnRunning, true);
  assert.deepEqual(ids(h), []);
  assert.equal(h.elements.sesscount.textContent, "0 sessions · showing 0");
  assert.equal(h.elements.sesslist.textContent, "暂无星标会话");
  assert.deepEqual(plain(h.switches), []);
  assert.equal(h.historyLoads(), 0);
  assert.equal(h.configLoads(), 0);
});

test("toggling stars during IME composition sends only committed keywords", async () => {
  const items = sessions(true);
  items[1].summary = "中文项目审查";
  const h = harness({ items }), filter = h.elements.sessfilter;
  h.dispatch(filter, "compositionstart");
  filter.value = "中";
  h.dispatch(filter, "input", { isComposing: true });
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.equal(requests(h).length, 1);
  assert.equal(requests(h)[0].searchParams.has("query"), false);
  filter.value = "中文";
  h.dispatch(filter, "compositionend");
  h.dispatch(filter, "input");
  h.advance(250);
  await settle();
  assert.equal(requests(h).length, 2);
  assert.equal(requests(h)[1].searchParams.get("query"), "中文");
  assert.ok(requests(h).every((request) => request.searchParams.get("marked") === "true"));
  assert.deepEqual(ids(h), ["target-session"]);
});

test("IME composition sends only the committed query and avoids duplicate final input", async () => {
  const items = sessions();
  items[1].summary = "中文项目审查";
  const h = harness({ items }), filter = h.elements.sessfilter;
  h.dispatch(filter, "compositionstart");
  filter.value = "中";
  h.dispatch(filter, "input", { isComposing: true });
  h.advance(500);
  assert.equal(requests(h).length, 0);
  filter.value = "中文";
  h.dispatch(filter, "compositionend");
  h.dispatch(filter, "input");
  h.advance(250);
  await settle();
  assert.equal(requests(h).length, 1);
  assert.equal(requests(h)[0].searchParams.get("query"), "中文");
  assert.deepEqual(ids(h), ["target-session"]);
});

test("cancelling composition resumes a search whose debounce was interrupted", async () => {
  const h = harness(), filter = h.elements.sessfilter;
  await input(h, "target");
  h.dispatch(filter, "compositionstart");
  h.advance(500);
  assert.equal(requests(h).length, 0);
  h.dispatch(filter, "compositionend");
  h.advance(250);
  await settle();
  assert.equal(requests(h).length, 1);
  assert.deepEqual(ids(h), ["target-session"]);
});
