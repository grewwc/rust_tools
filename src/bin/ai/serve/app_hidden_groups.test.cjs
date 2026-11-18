// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_hidden_groups.test.cjs
// Exercise hidden groups through the real sidebar, manager, API and activity flow.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const fixtureFile = path.join(__dirname, "app_mark.test.cjs");
const { harness, response, deferred, html } = new vm.Script(
  fs.readFileSync(fixtureFile, "utf8") + "\n;({ harness, response, deferred, html });",
  { filename: fixtureFile },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: fixtureFile, AbortController, URL, setImmediate,
});
const run = (h, source) => vm.runInContext(source, h.context);
const plain = (value) => JSON.parse(JSON.stringify(value));
const groups = () => [
  { id: "visible", name: "Work", order: 0, pinned: false },
  { id: "hidden/group", name: "Old Work", order: 1, pinned: true, hidden: true },
  { id: "other-hidden", name: "Research", order: 2, pinned: false, hidden: true },
];
const items = () => [
  { id: "open-session", summary: "Current work", group_id: "visible", size_bytes: 1024, marked: true },
  { id: "hidden-session", summary: "Hidden work", group_id: "hidden/group", size_bytes: 1024, marked: true },
  { id: "unassigned-session", summary: "Notes", group_id: null, size_bytes: 1024 },
];
const groupNames = (h) => plain(h.elements.sesslist.querySelectorAll(".session-group-name").map((el) => el.textContent));
const managerRows = (h) => h.elements.sessiongroupbody.querySelectorAll(".session-group-manager-row");
const managerNames = (h) => plain(managerRows(h).map((row) => row.querySelector(".session-group-manager-name").textContent));
const action = (parent, label) => parent.querySelectorAll("button").find((button) => button.textContent === label);
const click = async (h, parent, label) => {
  const button = action(parent, label);
  assert.ok(button, "missing action: " + label);
  await h.dispatch(button, "click").done;
};
async function setup(options = {}) {
  const h = harness({ groups: groups(), items: items(), ...options });
  await h.context.loadSessionGroups();
  await h.context.pollSessionGroupActivity(true);
  return h;
}

for (const phone of [false, true]) test("hidden groups leave default sidebar buckets and keep Unassigned (phone=" + phone + ")", async () => {
  const h = await setup({ phone });
  assert.deepEqual(groupNames(h), ["Work", "Unassigned"]);
  assert.equal(h.row("hidden-session"), undefined);
  assert.ok(h.row("unassigned-session"));
  assert.equal(h.context.lastItems.length, 3, "visibility does not remove loaded session data");
  assert.equal(h.context.currentSessionGroupId(), "visible");
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · incomplete activity");
  assert.equal(h.elements.sessiongrouphiddenactivity.hidden, false);
  assert.equal(run(h, "sessionGroups.find(group => group.id === 'visible').hidden"), false, "old group JSON defaults to visible");
  h.context.renderSessionList([]);
  assert.deepEqual(groupNames(h), ["Unassigned"]);
});

test("hidden assignments stay out of Unassigned and preserve selected group context", async () => {
  const assigned = items().map(({ group_id, ...item }) => item);
  const h = await setup({ items: assigned, sid: "hidden-session", assignments: { "open-session": "visible", "hidden-session": "hidden/group" } });
  assert.deepEqual(groupNames(h), ["Work", "Unassigned"]);
  assert.equal(h.row("hidden-session"), undefined);
  assert.equal(h.context.currentSessionGroupId(), "hidden/group");
  assert.equal(h.context.sessionGroupName(h.context.currentSessionGroupId()), "Old Work");
});

test("Hidden groups opens the searchable Hidden manager and can switch to All", async () => {
  const h = await setup();
  assert.match(html, /<button id="sessiongrouphidden"[^>]*>Hidden groups \(0\)<\/button>/);
  h.elements.sessmore.open = true;
  await h.dispatch(h.elements.sessiongrouphidden, "click").done;
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.elements.sessiongrouptitle.textContent, "Manage groups");
  assert.deepEqual(managerNames(h), ["Old Work", "Research"]);
  assert.equal(action(h.elements.sessiongroupbody, "Hidden").getAttribute("aria-pressed"), "true");
  const search = h.elements.sessiongroupbody.querySelector("input");
  assert.equal(search.type, "search");
  assert.equal(search.getAttribute("aria-label"), "Search groups by name");
  assert.equal(h.document.activeElement, search);
  search.value = "  OLD wOrK  ";
  await h.dispatch(search, "input").done;
  assert.deepEqual(managerNames(h), ["Old Work"]);
  assert.deepEqual(plain(managerRows(h)[0].querySelectorAll("button").map((el) => el.textContent)), ["Old Work", "Show in sidebar", "Open"]);
  search.value = "none";
  await h.dispatch(search, "input").done;
  assert.match(h.elements.sessiongroupbody.textContent, /No groups match your search/);
  search.value = "";
  await h.dispatch(search, "input").done;
  await click(h, h.elements.sessiongroupbody, "All");
  assert.deepEqual(managerNames(h), ["Old Work", "Work", "Research"]);
  assert.equal(action(h.elements.sessiongroupbody, "All").getAttribute("aria-pressed"), "true");
});

test("Show in sidebar permanently clears hidden with the frozen PATCH body", async () => {
  const serverGroups = groups(), h = await setup({ groups: serverGroups });
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Show in sidebar");
  const patches = h.requests.filter((request) => request.method === "PATCH");
  assert.equal(patches.length, 1);
  assert.equal(patches[0].path, "/session-groups/hidden%2Fgroup");
  assert.deepEqual(plain(patches[0].body), { hidden: false });
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
  assert.deepEqual(groupNames(h), ["Old Work", "Work", "Unassigned"]);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (1) · incomplete activity");
  assert.ok(h.row("hidden-session"));
  const refreshed = await setup({ groups: serverGroups });
  assert.ok(refreshed.row("hidden-session"), "Show persists across page lifetimes");
});

test("Hide from sidebar is beside rename/pin/delete and preserves session data", async () => {
  const h = await setup();
  await h.context.openSessionGroupManager();
  await click(h, h.elements.sessiongroupbody, "Work");
  for (const label of ["Rename", "Pin", "Hide from sidebar", "Delete group…"]) assert.ok(action(h.elements.sessiongroupbody, label));
  await click(h, h.elements.sessiongroupbody, "Hide from sidebar");
  assert.deepEqual(plain(h.requests.find((request) => request.method === "PATCH").body), { hidden: true });
  assert.deepEqual(groupNames(h), ["Unassigned"]);
  assert.equal(h.row("open-session"), undefined);
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(h.context.currentSessionGroupId(), "visible");
  assert.equal(h.context.lastItems.find((item) => item.id === "open-session").marked, true);
});

test("All manager rows offer Hide directly and Hidden view reports an empty set", async () => {
  const h = await setup({ groups: [groups()[0]], items: [items()[0]] });
  await h.context.openSessionGroupManager("hidden");
  assert.deepEqual(managerNames(h), []);
  assert.match(h.elements.sessiongroupbody.textContent, /No hidden groups/);
  await click(h, h.elements.sessiongroupbody, "All");
  await click(h, managerRows(h)[0], "Hide from sidebar");
  assert.equal(h.row("open-session"), undefined);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (1) · incomplete activity");
});

test("failed hide keeps the group visible and presents the API error", async () => {
  const h = await setup({ groupPatch: () => response({ error: "cannot update group" }, 404) });
  await h.context.openSessionGroupManager();
  await click(h, managerRows(h).find((row) => row.querySelector(".session-group-manager-name").textContent === "Work"), "Hide from sidebar");
  assert.ok(h.row("open-session"));
  assert.equal(h.elements.sessiongroupdialog.hidden, false);
  assert.match(h.elements.sessiongrouperror.textContent, /Group update failed: 404 cannot update group/);
  assert.equal(action(h.elements.sessiongroupbody, "All").disabled, false);
});

for (const phone of [false, true]) test("Open temporarily reveals and expands a hidden group without PATCH or storage (phone=" + phone + ")", async () => {
  const h = await setup({ phone });
  const key = JSON.stringify([h.context.store.base, "hidden/group", false, "exclude"]);
  h.stored.set("serve.groups.collapsed.v1", JSON.stringify({ [key]: true }));
  run(h, "sessionGroupCollapsed = JSON.parse(localStorage.getItem(SESSION_GROUP_COLLAPSED_KEY))");
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.equal(h.requests.some((request) => request.method === "PATCH"), false);
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
  assert.equal(h.row("hidden-session").hidden, false);
  const toggle = h.elements.sesslist.querySelector(".session-group-toggle");
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(h.document.activeElement, toggle);
  await h.dispatch(toggle, "click").done;
  assert.equal(h.row("hidden-session").hidden, true);
  assert.deepEqual(JSON.parse(h.stored.get("serve.groups.collapsed.v1")), { [key]: true });
  await h.context.loadSessionGroups();
  assert.equal(h.row("hidden-session").hidden, true, "temporary collapse survives group reload");
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.equal(h.row("hidden-session").hidden, false, "Open expands again");
  if (phone) assert.deepEqual(plain(h.drawerChanges), [true, true]);
  const refreshed = await setup({ phone, storage: h.stored });
  assert.equal(refreshed.row("hidden-session"), undefined, "refresh discards temporary visibility");
});

test("Open returns from search/Starred/Archived to the default group view", async () => {
  const h = await setup();
  run(h, "sessMarkedOnly = true; sessArchivedOnly = true; sessFilter = 'nothing'; sessionsSearchPending = true; sessionsSearchTimer = setTimeout(() => {}, 250)");
  h.elements.sessfilter.value = "nothing";
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.equal(run(h, "sessMarkedOnly || sessArchivedOnly || !!sessFilter || sessionsSearchPending || sessionsSearchTimer !== null"), false);
  assert.equal(h.elements.sessfilter.value, "");
  assert.equal(h.row("hidden-session").hidden, false);
  const latest = h.requests.filter((request) => request.path === "/sessions").at(-1);
  assert.equal(latest.searchParams.get("archived"), "exclude");
  assert.equal(latest.searchParams.get("query"), null);
  assert.equal(latest.searchParams.get("marked"), null);
});

test("Open can reveal an empty hidden group and Hide revokes a previous temporary Open", async () => {
  const h = await setup();
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[1], "Open");
  assert.ok(groupNames(h).includes("Research"));
  await h.context.openSessionGroupManager();
  await click(h, h.elements.sessiongroupbody, "Research");
  await click(h, h.elements.sessiongroupbody, "Show in sidebar");
  await h.context.openSessionGroupManager();
  await click(h, h.elements.sessiongroupbody, "Research");
  await click(h, h.elements.sessiongroupbody, "Hide from sidebar");
  assert.equal(groupNames(h).includes("Research"), false);
});

test("hidden running/waiting bubbles remain visible with More closed and expire", async () => {
  const active = items();
  active[1].activity = "running";
  active.push({ id: "needs-answer", summary: "Question", group_id: "other-hidden", size_bytes: 1024, activity: "running", waiting: "input" });
  const h = await setup({ items: active });
  assert.equal(h.elements.sessmore.open, false);
  assert.equal(h.row("hidden-session"), undefined);
  assert.equal(h.row("needs-answer"), undefined);
  const button = h.elements.sessiongrouphiddenactivity;
  assert.equal(button.hidden, false);
  assert.equal(button.classList.contains("waiting"), true);
  assert.match(button.getAttribute("aria-label"), /1 RUNNING · 1 WAITING/);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · 1 RUNNING · 1 WAITING · incomplete activity");
  assert.match(button.title, /incomplete activity/);
  run(h, "mergeSessionActivities([{ id: 'hidden-session', activity: 'idle' }, { id: 'needs-answer', activity: 'running', waiting: 'confirm' }])");
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · 1 WAITING · incomplete activity");
  await h.dispatch(button, "click").done;
  assert.deepEqual(managerNames(h), ["Old Work", "Research"]);
  h.advance(7500);
  run(h, "expireSessionActivities()");
  assert.equal(button.hidden, false);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · activity unavailable");
});

test("activity polling includes hidden rows and the selected group stays hidden", async () => {
  const active = items().map((item) => ({ ...item, activity: "idle" }));
  const h = await setup({ items: active, sid: "hidden-session", activityGet: (request) => response(request.body.ids.map((id) => ({ id, activity: "running", waiting: id === "hidden-session" ? "confirm" : null }))) });
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  const poll = h.requests.find((request) => request.path === "/session-activity");
  assert.ok(poll.body.ids.includes("hidden-session"));
  assert.match(h.elements.sessiongrouphidden.textContent, /1 WAITING/);
  assert.equal(h.row("hidden-session"), undefined);
});

for (const view of ["search", "Starred", "Archived", "Starred Archived"]) test(view + " includes hidden-group sessions", async () => {
  const source = items();
  if (view.includes("Archived")) source[1].archived = true;
  const h = await setup({ items: source });
  if (view === "search") {
    run(h, "sessFilter = 'Hidden work'");
    await h.context.loadSessions({ preserveSelection: true });
  } else {
    run(h, "sessMarkedOnly = " + view.includes("Starred") + "; sessArchivedOnly = " + view.includes("Archived"));
    await h.context.loadSessions({ preserveSelection: true });
  }
  assert.ok(h.row("hidden-session"));
  assert.equal(h.row("hidden-session").hidden, false);
  if (view === "search") assert.equal(h.row("hidden-session").querySelector(".session-group-tag").textContent, "Old Work");
  else assert.ok(groupNames(h).includes("Old Work"));
  run(h, "sessMarkedOnly = false; sessArchivedOnly = false; sessFilter = ''");
  await h.context.loadSessions({ preserveSelection: true });
  assert.equal(h.row("hidden-session"), undefined, "filtered visibility does not persist in default sidebar");
});

test("temporary visibility cannot leak to another server", async () => {
  const h = await setup();
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.ok(h.row("hidden-session"));
  h.context.store.base = "https://other.test";
  await h.context.loadSessionGroups();
  assert.equal(h.row("hidden-session"), undefined);
  assert.equal(run(h, "sessionGroupTemporaryOpen.size"), 0);
});

test("a pending visibility mutation cannot update a newly selected server", async () => {
  const pending = deferred(), h = await setup({ groupPatch: () => pending.promise });
  await h.context.openSessionGroupManager();
  const row = managerRows(h).find((row) => row.querySelector(".session-group-manager-name").textContent === "Work");
  const mutation = h.dispatch(action(row, "Hide from sidebar"), "click").done;
  h.context.store.base = "https://other.test";
  await h.context.loadSessionGroups();
  pending.resolve(response({ hidden: true }));
  await mutation;
  assert.ok(h.row("open-session"));
  assert.equal(run(h, "sessionGroups.find(group => group.id === 'visible').hidden"), false);
});

const groupSnapshot = (entries = []) => ({ groups: groups().filter((group) => group.hidden).map((group) => {
  const active = entries.filter((entry) => entry.group_id === group.id);
  return {
    group_id: group.id,
    running: active.filter((entry) => !entry.waiting && entry.activity === "running").length,
    waiting: active.filter((entry) => entry.waiting).length,
    unknown: 0,
    active: active.map(({ group_id, ...entry }) => entry),
  };
}) });
const groupRequests = (h) => h.requests.filter((request) => request.path === "/session-group-activity");
const groupPageRequests = (h) => h.requests.filter((request) => request.path === "/sessions" && request.searchParams.has("group"));
const pageState = (h, id = "hidden/group") => plain(run(h, "sessionGroupPages.get(sessionGroupPageKey(" + JSON.stringify(id) + "))"));
const globalState = (h) => plain(run(h, "({items:lastItems,total:sessionsTotal,offset:sessionsPageOffset,hasMore:sessionsHasMore})"));
const settle = async () => { for (let i = 0; i < 6; ++i) await new Promise(setImmediate); };

test("full hidden activity sees off-page and archived waiting sessions without counting loaded samples twice", async () => {
  const source = items().concat([
    { id: "off-page", summary: "Off-page work", size_bytes: 1024, group_id: "hidden/group", activity: "running" },
    { id: "archived-question", summary: "Archived question", size_bytes: 1024, group_id: "other-hidden", archived: true, activity: "running", waiting: "confirm", waiting_prompt: "Proceed?" },
  ]);
  const snapshot = groupSnapshot([
    { id: "hidden-session", group_id: "hidden/group", activity: "running" },
    { id: "off-page", group_id: "hidden/group", activity: "running" },
    { id: "archived-question", group_id: "other-hidden", activity: "running", waiting: "confirm", waiting_prompt: "Proceed?" },
  ]);
  const h = await setup({ items: source, cachedItems: source.slice(0, 3), groupActivityGet: () => response(snapshot) });
  assert.equal(h.row("off-page"), undefined);
  h.context.mergeSessionActivities([{ id: "hidden-session", activity: "running" }]);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · 2 RUNNING · 1 WAITING");
  assert.equal(run(h, "sessionActivitySamples.has('off-page')"), false);
  assert.equal(run(h, "sessionActivitySamples.has('archived-question')"), false);
  await h.context.openSessionGroupManager("hidden");
  const archivedRow = managerRows(h).find((row) => row.querySelector(".session-group-manager-name").textContent === "Research");
  assert.match(archivedRow.textContent, /1 WAITING/);
  assert.equal(action(archivedRow, "WAITING · archived-question").title, "Proceed?");
  const before = globalState(h);
  await click(h, archivedRow, "WAITING · archived-question");
  assert.equal(h.context.sessionId, "archived-question");
  assert.equal(h.elements.sesstitle.textContent, "Archived question");
  assert.match(h.elements.sessmeta.textContent, /Archived/);
  const request = groupPageRequests(h).at(-1);
  assert.equal(request.searchParams.get("group"), "other-hidden");
  assert.equal(request.searchParams.get("include"), "archived-question");
  assert.equal(request.searchParams.get("archived"), "include");
  assert.equal(request.searchParams.has("group_id"), false);
  assert.deepEqual(globalState(h), before);
  assert.equal(run(h, "sessionGroupPages.size"), 0, "metadata include never advances temporary paging");
  assert.equal(h.elements.sessrename.disabled, false, "archived off-page targets can be renamed after selecting");
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "RUNNING · off-page");
  assert.equal(h.context.sessionId, "off-page");
  assert.equal(h.elements.sesstitle.textContent, "Off-page work");
  assert.deepEqual(globalState(h), before);
});

test("archived RUNNING shortcuts include and select the off-page target without changing any paging cursor", async () => {
  const target = { id: "archived-running", summary: "Archived running work", size_bytes: 1024, group_id: "hidden/group", archived: true, activity: "running" };
  const source = [target, ...Array.from({ length: 13 }, (_, i) => ({ id: "newer-" + i, summary: "Newer work " + i, size_bytes: 1024, group_id: "hidden/group" })), items()[0]];
  const h = await setup({ items: source, cachedItems: [items()[0]], groupActivityGet: () => response(groupSnapshot([target])) });
  run(h, "sessionsTotal = 80; sessionsPageOffset = 24; sessionsHasMore = true");
  assert.equal(h.row(target.id), undefined);
  assert.match(h.elements.sessiongrouphidden.textContent, /1 RUNNING/);
  await h.context.openSessionGroupManager("hidden");
  const before = globalState(h);
  await click(h, managerRows(h)[0], "RUNNING · archived-running");
  const request = groupPageRequests(h).at(-1);
  assert.equal(request.searchParams.get("group"), target.group_id);
  assert.equal(request.searchParams.get("include"), target.id);
  assert.equal(request.searchParams.get("archived"), "include");
  assert.equal(request.searchParams.get("limit"), "12");
  assert.equal(request.searchParams.get("offset"), "0");
  assert.equal(h.context.sessionId, target.id);
  assert.equal(h.context.currentSessionGroupId(), target.group_id);
  assert.equal(h.elements.sesstitle.textContent, target.summary);
  assert.match(h.elements.sessmeta.textContent, /Archived/);
  assert.equal(h.elements.sessiongroupdialog.hidden, true);
  assert.equal(h.elements.sessiongrouperror.textContent, "");
  assert.deepEqual(globalState(h), before);
  assert.equal(run(h, "sessionGroupPages.size"), 0, "shortcut metadata does not create or advance a group page");
});

test("temporary Open fetches the group's own first and second pages without changing the global cursor", async () => {
  const hidden = Array.from({ length: 25 }, (_, i) => ({ id: "group-" + i, summary: "Group session " + i, group_id: "hidden/group", size_bytes: 1024 }));
  const source = [items()[0], ...hidden];
  const h = await setup({ items: source, cachedItems: [source[0]], groupActivityGet: () => response(groupSnapshot()) });
  run(h, "sessionsTotal = 80; sessionsPageOffset = 12; sessionsHasMore = true");
  const before = globalState(h), storageBefore = plain([...h.stored]);
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.ok(h.row("group-24"));
  assert.equal(pageState(h).items.length, 12);
  assert.equal(pageState(h).total, 25);
  assert.equal(pageState(h).offset, 12);
  assert.equal(pageState(h).hasMore, true);
  const first = groupPageRequests(h)[0];
  assert.equal(first.searchParams.get("group"), "hidden/group");
  assert.equal(first.searchParams.get("group_id"), null);
  assert.equal(first.searchParams.get("newest"), "true");
  assert.equal(first.searchParams.get("archived"), "exclude");
  assert.equal(first.searchParams.get("limit"), "12");
  assert.equal(first.searchParams.get("offset"), "0");
  await click(h, h.elements.sesslist, "Load more in Old Work");
  assert.ok(h.row("group-1"));
  assert.equal(pageState(h).items.length, 24);
  assert.equal(pageState(h).offset, 24);
  assert.equal(groupPageRequests(h)[1].searchParams.get("offset"), "12");
  assert.deepEqual(globalState(h), before);
  assert.deepEqual(plain([...h.stored]), storageBefore);
  assert.equal(h.requests.some((request) => request.method === "PATCH"), false);
  assert.equal(h.requests.filter((request) => request.path === "/sessions" && !request.searchParams.has("group")).length, 0);
  await click(h, h.elements.sesslist, "Load more in Old Work");
  assert.equal(pageState(h).offset, 25);
  assert.equal(pageState(h).hasMore, false);
  run(h, "resetTemporarySessionGroups(); renderSessionList(lastItems)");
  assert.equal(h.row("group-24"), undefined);
  assert.deepEqual(globalState(h), before);
  assert.match(html, /\$\("refresh"\)\.onclick = async \(\) => \{ resetTemporarySessionGroups\(\);/);
});

test("temporary rows merge by ID with the global page and Unassigned uses the group query contract", async () => {
  const h = await setup();
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.equal(h.elements.sesslist.querySelectorAll("[data-session-id]").filter((row) => row.dataset.sessionId === "hidden-session").length, 1);
  run(h, "sessionGroupDialogBase = store.base");
  await h.context.openHiddenSessionGroup({ id: null, name: "Unassigned" });
  assert.equal(groupPageRequests(h).at(-1).searchParams.get("group"), "unassigned");
  assert.ok(groupNames(h).includes("Unassigned"));
  assert.ok(h.row("unassigned-session"));
});

test("failed Open keeps good group data and offers retry rather than a successful empty group", async () => {
  let mode = "ok";
  const h = await setup({ cachedItems: [items()[0]], sessionsGet: (request) => mode === "error"
    ? response({ error: "temporary failure" }, 503)
    : response([items()[1]], 200, { "x-sessions-total": "1", "x-sessions-page-size": "1" }) });
  await h.context.openSessionGroupManager("hidden");
  await click(h, managerRows(h)[0], "Open");
  assert.ok(h.row("hidden-session"));
  mode = "error";
  await h.context.loadSessionGroupPage("hidden/group");
  assert.ok(h.row("hidden-session"));
  assert.match(h.elements.sesslist.textContent, /Group sessions unavailable: 503 temporary failure/);
  assert.ok(action(h.elements.sesslist, "Retry group sessions"));
  mode = "ok";
  await click(h, h.elements.sesslist, "Retry group sessions");
  assert.equal(pageState(h).error, "");
  assert.equal(pageState(h).offset, 1);
  run(h, "resetTemporarySessionGroups()");
  mode = "error";
  run(h, "sessionGroupDialogBase = store.base");
  await h.context.openHiddenSessionGroup(groups()[1]);
  assert.equal(pageState(h).loaded, false);
  assert.match(h.elements.sesslist.textContent, /Group sessions unavailable/);
  assert.ok(action(h.elements.sesslist, "Retry group sessions"));
});

for (const change of ["server", "filter", "group", "refresh"]) test("temporary group paging discards a pending response after " + change + " changes", async () => {
  const pending = deferred();
  const h = await setup({ cachedItems: [items()[0]], sessionsGet: () => pending.promise });
  run(h, "sessionGroupDialogBase = store.base");
  const opened = h.context.openHiddenSessionGroup(groups()[1]);
  if (change === "server") { h.context.store.base = "https://other.test"; await h.context.loadSessionGroups(); }
  if (change === "filter") { run(h, "sessFilter = 'different'; resetSessionSearch()"); }
  if (change === "group") {
    run(h, "++sessionGroupPageEpoch; sessionGroupTemporaryOpen.set(sessionGroupTemporaryKey('other-hidden'), false)");
  }
  if (change === "refresh") run(h, "resetTemporarySessionGroups()");
  pending.resolve(response([items()[1]], 200, { "x-sessions-total": "30", "x-sessions-page-size": "12" }));
  await opened;
  assert.equal(h.row("hidden-session"), undefined);
  assert.equal(run(h, "[...sessionGroupPages.values()].some(page => page.loaded)"), false);
});

for (const status of [401, 403, 500, 503, "parse", "shape", "timeout"]) test("hidden activity " + status + " failures show unavailable and never fall back to a fake zero", async () => {
  let fail = false, timedOut = false;
  const pending = deferred();
  const h = await setup({ groupActivityGet: () => {
    if (!fail) return response(groupSnapshot([{ id: "off-page", group_id: "hidden/group", activity: "running" }]));
    if (status === "parse") return { ok: true, json: async () => { throw new SyntaxError("bad JSON"); } };
    if (status === "shape") return response({ groups: [] });
    if (status === "timeout") return timedOut ? response({ error: "unavailable" }, 503) : pending.promise;
    return response({ error: "unavailable" }, status);
  } });
  assert.match(h.elements.sessiongrouphidden.textContent, /1 RUNNING/);
  fail = true;
  const poll = h.context.pollSessionGroupActivity(true);
  if (status === "timeout") { h.advance(5000); pending.resolve(response(groupSnapshot())); }
  await poll;
  timedOut = true;
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · activity unavailable");
  assert.equal(h.elements.sessiongrouphiddenactivity.hidden, false);
  await h.context.openSessionGroupManager("hidden");
  assert.match(managerRows(h)[0].textContent, /activity unavailable/);
});

test("full group snapshot expires at 7.5 seconds and 404 fallback is explicitly incomplete", async () => {
  let legacy = false;
  const h = await setup({ groupActivityGet: () => legacy ? response({ error: "missing" }, 404) : response(groupSnapshot()) });
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2)");
  h.advance(7499);
  h.context.expireSessionActivities();
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2)");
  h.advance(1);
  h.context.expireSessionActivities();
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · activity unavailable");
  legacy = true;
  h.context.mergeSessionActivities([{ id: "hidden-session", activity: "running" }]);
  await h.context.pollSessionGroupActivity(true);
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2) · 1 RUNNING · incomplete activity");
});

test("group polling runs beside loaded activity every 2.5 seconds and probes on tab return and network recovery", async () => {
  const h = await setup({ cachedItems: [], groupActivityGet: () => response(groupSnapshot()) });
  assert.ok(groupRequests(h).length > 0, "group loading probes immediately even without loaded IDs");
  h.requests.length = 0;
  h.document.visibilityState = "visible";
  h.context.wireSessionActivityPoll();
  await settle();
  assert.equal(groupRequests(h).length, 1);
  assert.equal(h.requests.some((request) => request.path === "/session-activity"), false, "unsupported loaded activity does not gate group activity");
  h.advance(2500); await settle();
  assert.equal(groupRequests(h).length, 2);
  h.document.visibilityState = "hidden";
  for (const listener of h.documentListeners.get("visibilitychange")) listener();
  h.advance(5000); await settle();
  assert.equal(groupRequests(h).length, 2);
  h.document.visibilityState = "visible";
  for (const listener of h.documentListeners.get("visibilitychange")) listener();
  await settle();
  assert.equal(groupRequests(h).length, 3);
  for (const listener of h.windowListeners.get("online")) listener();
  await settle();
  assert.equal(groupRequests(h).length, 4);
});

test("group and loaded activity have independent no-overlap guards and stale server replies cannot commit", async () => {
  let pendingMode = false;
  const groupPending = deferred(), loadedPending = deferred();
  const h = await setup({ items: items().map((item) => ({ ...item, activity: "idle" })),
    groupActivityGet: () => pendingMode ? groupPending.promise : response(groupSnapshot()),
    activityGet: () => loadedPending.promise,
  });
  h.document.visibilityState = "visible";
  h.requests.length = 0; pendingMode = true;
  const groupPoll = h.context.pollSessionGroupActivity(), loadedPoll = h.context.pollSessionActivity();
  h.context.pollSessionGroupActivity(); h.context.pollSessionActivity();
  assert.equal(groupRequests(h).length, 1);
  assert.equal(h.requests.filter((request) => request.path === "/session-activity").length, 1);
  h.context.store.base = "https://other.test";
  pendingMode = false;
  await h.context.loadSessionGroups(); await settle();
  groupPending.resolve(response(groupSnapshot([{ id: "old-server-active", group_id: "hidden/group", activity: "running" }])));
  loadedPending.resolve(response([]));
  await Promise.all([groupPoll, loadedPoll]); await settle();
  assert.equal(h.elements.sessiongrouphidden.textContent, "Hidden groups (2)");
  assert.equal(run(h, "hiddenSessionGroupActivity('hidden/group').active.length"), 0);
  assert.equal(run(h, "sessionActivitySamples.has('old-server-active')"), false);
});
