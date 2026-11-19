// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_session_view_mode.test.cjs
// Exercise view modes through the real sidebar, pager and creation functions.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function fixture(name, exports) {
  const file = path.join(__dirname, name);
  return new vm.Script(fs.readFileSync(file, "utf8") + "\n;({ " + exports + " });", { filename: file })
    .runInNewContext({
      require: (name) => name === "node:test" ? { test() {} } : require(name),
      __dirname, __filename: file, process, console, Buffer, AbortController, URL,
      URLSearchParams, TextEncoder, TextDecoder, setTimeout, clearTimeout,
      setInterval, clearInterval, setImmediate, queueMicrotask,
    });
}
const { harness, response, deferred, settle, html } = fixture("app_mark.test.cjs", "harness, response, deferred, settle, html");
const { groupCreationHarness } = fixture("app_phase4_sessions.test.cjs", "groupCreationHarness");
const run = (h, source) => vm.runInContext(source, h.context);
const plain = (value) => JSON.parse(JSON.stringify(value));
const ids = (h) => plain(h.elements.sesslist.querySelectorAll("[data-session-id]").map((row) => row.dataset.sessionId));
const heads = (h) => h.elements.sesslist.querySelectorAll(".session-group-head");
const pager = (h) => plain(run(h, "({ total: sessionsTotal, offset: sessionsPageOffset, hasMore: sessionsHasMore, loading: sessionsLoadingMore })"));
const requests = (h) => h.requests.filter((request) => request.path === "/sessions");
const groups = () => [
  { id: "work", name: "Work", order: 0, pinned: true },
  { id: "hidden", name: "Hidden work", order: 1, hidden: true },
];
const items = () => [
  { id: "newest", summary: "Newest note", group_id: null, size_bytes: 1024 },
  { id: "hidden-session", summary: "Hidden note", group_id: "hidden", size_bytes: 1024, marked: true },
  { id: "work-session", summary: "Work note", group_id: "work", size_bytes: 1024, marked: true, activity: "running" },
];
async function setup(options = {}) {
  const h = harness({ groups: groups(), items: items(), sid: "work-session", ...options });
  await h.context.loadSessionGroups();
  return h;
}
function page(items, total = items.length) {
  return response(items, 200, { "x-sessions-total": String(total), "x-sessions-page-size": String(items.length) });
}

for (const [saved, expected] of [[undefined, "group"], ["group", "group"], ["list", "list"], ["", "group"], ["LIST", "group"], ["invalid", "group"], ['"list"', "group"]]) {
  test("session view mode validates stored value: " + String(saved), async () => {
    const storage = new Map(saved === undefined ? [] : [["serve.sessionViewMode", saved]]);
    const h = await setup({ storage });
    assert.equal(run(h, "sessionViewMode"), expected);
    assert.equal(h.elements.sessviewmode.textContent, expected === "group" ? "List view" : "Group view");
    assert.equal(heads(h).length > 0, expected === "group");
  });
}

for (const phone of [false, true]) test("More switches and persists view mode immediately (phone=" + phone + ")", async () => {
  const h = await setup({ phone });
  assert.match(html, /<button id="sessviewmode" type="button">List view<\/button>/);
  assert.equal(h.elements.sessmorepanel.contains(h.elements.sessviewmode), true);
  const before = h.requests.length;
  h.elements.sesslist.scrollTop = 83;
  h.elements.sessmore.open = true;
  h.elements.sessviewmode.focus();
  await h.dispatch(h.elements.sessviewmode, "click").done;
  assert.equal(run(h, "sessionViewMode"), "list");
  assert.equal(h.stored.get("serve.sessionViewMode"), "list");
  assert.equal(h.elements.sessviewmode.textContent, "Group view");
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.document.activeElement, h.elements.sessmoretoggle);
  assert.equal(h.elements.sesslist.scrollTop, 83);
  assert.equal(h.requests.length, before, "switching reuses the global loaded page");
  assert.deepEqual(ids(h), ["newest", "hidden-session", "work-session"]);
  assert.equal(heads(h).length, 0);
  const restored = await setup({ phone, storage: h.stored });
  assert.equal(run(restored, "sessionViewMode"), "list");
  h.elements.sessmore.open = true;
  await h.dispatch(h.elements.sessviewmode, "click").done;
  assert.equal(h.stored.get("serve.sessionViewMode"), "group");
  assert.equal(h.elements.sessviewmode.textContent, "List view");
  assert.deepEqual(ids(h), ["work-session", "newest"]);
  assert.deepEqual(plain(heads(h).map((head) => head.querySelector(".session-group-name").textContent)), ["Work", "Unassigned"]);
});

test("view switching remains usable when storage reads or writes throw", async () => {
  const storage = new Map();
  storage.get = () => { throw new Error("storage unavailable"); };
  storage.set = () => { throw new Error("storage unavailable"); };
  const h = await setup({ storage });
  assert.equal(run(h, "sessionViewMode"), "group");
  await h.dispatch(h.elements.sessviewmode, "click").done;
  assert.equal(run(h, "sessionViewMode"), "list");
  assert.equal(heads(h).length, 0);
});

test("More keeps Escape, outside-pointer and focus-out closing in list mode", async () => {
  const h = await setup();
  h.context.setSessionViewMode("list");
  h.elements.sessmore.open = true;
  const escape = h.dispatch(h.elements.sessviewmode, "keydown", { key: "Escape" }).event;
  assert.equal(escape.prevented, true);
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.document.activeElement, h.elements.sessmoretoggle);
  for (const event of ["pointerdown", "focusin"]) {
    h.elements.sessmore.open = true;
    for (const listener of h.documentListeners.get(event)) listener({ target: h.elements.sessviewmode });
    assert.equal(h.elements.sessmore.open, true);
    h.elements.prompt.focus();
    for (const listener of h.documentListeners.get(event)) listener({ target: h.elements.prompt });
    assert.equal(h.elements.sessmore.open, false);
    assert.equal(h.document.activeElement, h.elements.prompt);
  }
  assert.equal(run(h, "sessionViewMode"), "list");
});

test("flat rows retain backend order, activity, actions, stars and selection", async () => {
  const h = await setup();
  h.context.setSessionViewMode("list");
  await h.context.pollSessionActivity();
  assert.deepEqual(ids(h), items().map((item) => item.id), "running is not reordered within a loaded page");
  assert.equal(h.row("work-session").classList.contains("cur"), true);
  assert.equal(h.row("work-session").classList.contains("session-running"), true);
  assert.equal(h.row("hidden-session").hidden, false);
  await h.dispatch(h.row("newest").querySelector(".mk"), "click").done;
  assert.equal(h.row("newest").classList.contains("session-marked"), true);
  assert.deepEqual(ids(h), items().map((item) => item.id));
  await h.dispatch(h.row("hidden-session").querySelector(".session-title"), "keydown", { key: "Enter" }).done;
  assert.equal(h.context.sessionId, "hidden-session");
  assert.equal(h.row("hidden-session").classList.contains("cur"), true);
  await h.dispatch(h.row("newest").querySelector(".session-menu"), "click").done;
  assert.equal(h.elements.sessionactions.hidden, false);
  assert.equal(h.elements.sessionactionstitle.textContent, "Newest note");
  assert.equal(h.context.sessionId, "hidden-session", "card actions keep selection");
});

for (const gesture of ["button", "scroll", "wheel"]) test("list pagination appends global backend pages via " + gesture, async () => {
  const ordered = Array.from({ length: 30 }, (_, index) => ({ id: "page-" + index, summary: "Note " + index, group_id: index % 2 ? "work" : "hidden", size_bytes: 1024 }));
  const h = await setup({ items: ordered.slice().reverse(), cachedItems: [], sid: "page-0" });
  h.context.setSessionViewMode("list");
  await h.context.loadSessions({ preserveSelection: true });
  assert.deepEqual(ids(h), ordered.slice(0, 12).map((item) => item.id));
  assert.deepEqual(pager(h), { total: 30, offset: 12, hasMore: true, loading: false });
  h.elements.sesslist.scrollTop = 480;
  h.elements.sesslist.clientHeight = 120;
  h.elements.sesslist.scrollHeight = 600;
  if (gesture === "button") await h.dispatch(h.elements.sesslist.querySelector(".session-more").querySelector("button"), "click").done;
  else { h.dispatch(h.elements.sesslist, gesture, { deltaY: 1 }); await settle(); }
  assert.deepEqual(ids(h), ordered.slice(0, 24).map((item) => item.id));
  assert.equal(h.elements.sesslist.scrollTop, 480);
  assert.equal(pager(h).offset, 24);
  h.context.setSessionViewMode("group");
  h.context.setSessionViewMode("list");
  assert.equal(pager(h).offset, 24, "mode changes preserve the global cursor");
  await h.context.loadOlderSessions();
  assert.deepEqual(ids(h), ordered.map((item) => item.id));
  assert.deepEqual(pager(h), { total: 30, offset: 30, hasMore: false, loading: false });
  assert.equal(heads(h).length, 0);
  assert.equal(h.elements.sesslist.querySelector(".session-more"), null);
  assert.deepEqual(plain(requests(h).map((request) => request.searchParams.get("offset"))), ["0", "12", "24"]);
  for (const request of requests(h)) assert.equal(request.searchParams.has("group"), false);
});

test("list mode preserves Starred and Archived filters and their global pages", async () => {
  const ordered = [
    ...Array.from({ length: 15 }, (_, index) => ({ id: "star-" + index, summary: "Star", marked: true, group_id: index % 2 ? "work" : "hidden" })),
    { id: "archived-star", summary: "Old star", archived: true, marked: true, group_id: "hidden" },
    { id: "archived-note", summary: "Old note", archived: true, group_id: "work" },
    { id: "note", summary: "Note", group_id: null },
  ];
  const h = await setup({ items: ordered.slice().reverse(), sid: "note" });
  h.context.setSessionViewMode("list");
  await h.dispatch(h.elements.sessmarked, "click").done;
  assert.deepEqual(ids(h), ordered.slice(0, 12).map((item) => item.id));
  await h.context.loadOlderSessions();
  assert.deepEqual(ids(h), ordered.slice(0, 15).map((item) => item.id));
  await h.dispatch(h.elements.sessarchived, "click").done;
  assert.deepEqual(ids(h), ["archived-star"]);
  assert.equal(h.elements.sessfilterstatus.textContent, "Archived ★");
  assert.equal(h.context.sessionId, "note");
  assert.equal(requests(h).at(-1).searchParams.get("marked"), "true");
  assert.equal(requests(h).at(-1).searchParams.get("archived"), "only");
  await h.dispatch(h.elements.sessall, "click").done;
  assert.deepEqual(ids(h), ["archived-star", "archived-note"]);
  await h.dispatch(h.elements.sessactive, "click").done;
  assert.equal(requests(h).at(-1).searchParams.get("archived"), "exclude");
  assert.equal(requests(h).at(-1).searchParams.has("marked"), false);
  for (const request of requests(h)) assert.equal(request.searchParams.has("group"), false);
  assert.equal(heads(h).length, 0);
});

test("list search keeps debounce, paging and filters without group tags", async () => {
  const ordered = Array.from({ length: 16 }, (_, index) => ({ id: "match-" + index, summary: "Match note " + index, marked: true, group_id: index % 2 ? "work" : "hidden" }));
  const h = await setup({ items: ordered.slice().reverse(), sid: "match-0" });
  h.context.setSessionViewMode("list");
  await h.dispatch(h.elements.sessmarked, "click").done;
  const before = requests(h).length;
  h.elements.sessfilter.value = "Match";
  h.dispatch(h.elements.sessfilter, "input");
  h.advance(100);
  h.elements.sessfilter.value = "Match note";
  h.dispatch(h.elements.sessfilter, "input");
  h.advance(249);
  assert.equal(requests(h).length, before);
  h.advance(1);
  await settle();
  assert.equal(requests(h).length, before + 1);
  assert.deepEqual(ids(h), ordered.slice(0, 12).map((item) => item.id));
  assert.equal(h.elements.sesslist.querySelectorAll(".session-group-tag").length, 0);
  assert.equal(h.elements.sesslist.querySelector(".session-more").textContent, "Load more search results");
  await h.context.loadOlderSessions();
  assert.deepEqual(ids(h), ordered.map((item) => item.id));
  assert.equal(requests(h).at(-1).searchParams.get("query"), "Match note");
  assert.equal(requests(h).at(-1).searchParams.get("marked"), "true");
  assert.equal(requests(h).at(-1).searchParams.has("group"), false);
  h.context.setSessionViewMode("group");
  assert.equal(h.elements.sesslist.querySelectorAll(".session-group-tag").length, 16);
  h.context.setSessionViewMode("list");
  assert.equal(h.elements.sesslist.querySelectorAll(".session-group-tag").length, 0);
});

for (const phone of [false, true]) test("list global plus and explicit creation both use Unassigned (phone=" + phone + ")", async () => {
  const { h, calls } = await groupCreationHarness(phone);
  h.context.setSessionViewMode("list");
  const collapsed = plain(run(h, "sessionGroupCollapsed"));
  assert.equal(h.context.currentSessionGroupId(), "work");
  assert.equal(h.elements.newbtn.title, "New session in Unassigned");
  assert.equal(h.elements.newbtn.getAttribute("aria-label"), h.elements.newbtn.title);
  await h.elements.newbtn.onclick();
  await h.context.createSessionInGroup("work");
  assert.deepEqual(plain(calls.filter((call) => call.method === "POST").map((call) => call.body)), [{ group_id: null }, { group_id: null }]);
  assert.equal(h.context.currentSessionGroupId(), null);
  assert.deepEqual(plain(run(h, "sessionGroupCollapsed")), collapsed, "list creation preserves group collapse preferences");
  assert.equal(heads(h).length, 0);
  assert.equal(h.elements.newbtn.disabled, false);
});

test("entering list clears temporary hidden pages and ignores their late response", async () => {
  const flight = deferred();
  const h = await setup({ sessionsGet: (request) => request.searchParams.has("group") ? flight.promise : page(items()) });
  await h.context.openSessionGroupManager("hidden");
  const pending = h.context.openHiddenSessionGroup(groups()[1]);
  assert.equal(run(h, "sessionGroupTemporaryOpen.size"), 1);
  h.context.setSessionViewMode("list");
  assert.equal(run(h, "sessionGroupTemporaryOpen.size"), 0);
  assert.equal(run(h, "sessionGroupPages.size"), 0);
  flight.resolve(page([{ id: "temporary-only", summary: "Temporary", group_id: "hidden" }]));
  await pending;
  assert.deepEqual(ids(h), items().map((item) => item.id));
  assert.equal(h.row("temporary-only"), undefined);
  const before = h.requests.length;
  await h.context.loadSessionGroupPage("hidden");
  assert.equal(h.requests.length, before, "list mode does not fetch a group page");
  h.context.setSessionViewMode("group");
  assert.equal(h.row("hidden-session"), undefined);
  assert.equal(h.row("temporary-only"), undefined);
});

test("list excludes loaded temporary-only rows and restores saved group collapse", async () => {
  const h = await setup();
  await h.dispatch(heads(h)[0].querySelector(".session-group-toggle"), "click").done;
  assert.equal(h.row("work-session").hidden, true);
  const saved = h.stored.get("serve.groups.collapsed.v1");
  run(h, "sessionGroupTemporaryOpen.set(sessionGroupTemporaryKey('hidden'), false); sessionGroupPages.set(sessionGroupPageKey('hidden'), { items: [{ id: 'temporary-only', group_id: 'hidden', summary: 'Temporary' }], loaded: true })");
  h.context.renderSessionList(h.context.lastItems);
  assert.ok(h.row("temporary-only"));
  h.context.setSessionViewMode("list");
  assert.deepEqual(ids(h), items().map((item) => item.id));
  assert.equal(h.row("work-session").hidden, false);
  h.context.setSessionViewMode("group");
  assert.equal(h.row("work-session").hidden, true);
  assert.equal(h.stored.get("serve.groups.collapsed.v1"), saved);
  assert.equal(h.elements.sesslist.querySelector(".session-group-new").title, "New session in Work");
});

test("explicit hidden-group Open leaves list mode before issuing a group request", async () => {
  const h = await setup();
  h.context.setSessionViewMode("list");
  await h.context.openSessionGroupManager("hidden");
  await h.context.openHiddenSessionGroup(groups()[1]);
  assert.equal(run(h, "sessionViewMode"), "group");
  assert.equal(h.stored.get("serve.sessionViewMode"), "group");
  assert.ok(h.row("hidden-session"));
  assert.ok(requests(h).some((request) => request.searchParams.get("group") === "hidden"));
});
