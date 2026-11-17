// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_session_activity.test.cjs
// Reuse the sidebar fixture to verify live status without replacing list rows.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const fixtureFile = path.join(__dirname, "app_mark.test.cjs");
const { harness, response, deferred, settle, sessions, css } = new vm.Script(
  fs.readFileSync(fixtureFile, "utf8") + "\n;({ harness, response, deferred, settle, sessions, css });",
  { filename: fixtureFile },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: fixtureFile, AbortController, URL, setImmediate,
});
const plain = (value) => JSON.parse(JSON.stringify(value));
const run = (h, source) => vm.runInContext(source, h.context);
const listed = (activity = "idle") => sessions().map((item) => ({ ...item, activity }));
const row = (h, id) => h.elements.sesslist.children.find((node) => node.dataset.sessionId === id);
const badge = (h, id) => row(h, id)?.querySelector(".session-running-badge");
const dot = (h, id) => row(h, id)?.querySelector(".session-running-dot");
const running = (h, id) => Boolean(badge(h, id));
const samples = (h) => plain(run(h, "[...sessionActivitySamples].map(([id, sample]) => [id, sample.activity])"));
const activityRequests = (h) => h.requests.filter((request) => request.path === "/session-activity");
function sameRows(h, before) {
  assert.equal(h.elements.sesslist.children.length, before.length);
  for (const [index, element] of before.entries()) assert.equal(h.elements.sesslist.children[index], element);
}

const sendFixture = path.join(__dirname, "app_phase2_baseline.test.cjs");
const { sendHarness } = new vm.Script(
  fs.readFileSync(sendFixture, "utf8") + "\n;({ sendHarness });", { filename: sendFixture },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: sendFixture, console, Buffer, process,
  TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
  setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
});

for (const phone of [false, true]) test("first paint shows a labeled running row keyed by full IDs (phone=" + phone + ")", () => {
  const items = [
    { id: "samehead-alpha", summary: "这是一个很长的中文会话标题，用来确认运行标签不会改变完整标题内容", marked: true, activity: "running" },
    { id: "samehead-beta", summary: "Second", activity: "idle" },
    { id: "unknown", summary: "Unknown", activity: "unknown" },
    { id: "legacy", summary: "Legacy" },
  ];
  const h = harness({ phone, items, sid: items[0].id });
  assert.equal(running(h, items[0].id), true);
  for (const item of items) {
    const active = item.activity === "running", element = row(h, item.id), indicator = badge(h, item.id);
    assert.equal(element.classList.contains("session-running"), active);
    assert.equal(element.querySelectorAll(".session-running-badge").length, active ? 1 : 0);
    assert.equal(element.querySelector(".session-hash").textContent, item.id.slice(0, 8));
    if (active) {
      assert.equal(indicator.textContent, "RUNNING");
      assert.equal(indicator.title, "RUNNING");
      assert.equal(indicator.getAttribute("aria-label"), "RUNNING");
      assert.equal(indicator.getAttribute("role"), "img");
      assert.equal(dot(h, item.id).getAttribute("aria-hidden"), "true");
      assert.equal(element.classList.contains("cur"), true, "running composes with selection");
    } else {
      assert.equal(indicator, null);
      assert.equal(dot(h, item.id), null);
      assert.equal(element.querySelector("small").textContent, item.id.slice(0, 8));
    }
    assert.equal(row(h, item.id).querySelector("span").textContent, item.summary);
  }
  assert.equal(running(h, items[1].id), false, "equal short IDs must not share status");
  assert.equal(running(h, "unknown"), false);
  assert.equal(running(h, "legacy"), false);
  assert.equal(h.elements.sesscount.textContent, "4 sessions");
  assert.deepEqual(plain(h.requests), []);
});

for (const phone of [false, true]) test("running badges are added once and removed on idle or unknown after repaint (phone=" + phone + ")", () => {
  const h = harness({ phone, items: listed() }), id = "target-session";
  const before = [...h.elements.sesslist.children], target = row(h, id);
  const controls = [target.querySelector(".session-title"), target.querySelector(".mk")];
  assert.equal(badge(h, id), null);
  let indicator;
  for (let i = 0; i < 3; i++) {
    h.context.mergeSessionActivities([{ id, activity: "running" }]);
    run(h, "paintSessionActivity('target-session');");
    indicator ??= badge(h, id);
    assert.equal(badge(h, id), indicator, "repeated observations reuse the badge");
    assert.equal(target.querySelectorAll(".session-running-badge").length, 1);
    assert.equal(target.querySelectorAll(".session-running-dot").length, 1);
    assert.equal(target.classList.contains("session-running"), true);
    assert.equal(target.classList.contains("cur"), false, "running does not select the row");
  }
  sameRows(h, before);
  assert.deepEqual([target.querySelector(".session-title"), target.querySelector(".mk")], controls);
  h.context.mergeSessionActivities([{ id, activity: "idle" }]);
  run(h, "paintSessionActivity('target-session');");
  assert.equal(badge(h, id), null);
  assert.equal(dot(h, id), null);
  assert.equal(target.classList.contains("session-running"), false);
  assert.equal(target.querySelector("small").textContent, id.slice(0, 8));
  run(h, "lastItems[1].activity = 'running'; renderSessionList(lastItems);");
  assert.equal(badge(h, id), null, "a stale list cannot restore a finished session's badge");
  assert.equal(row(h, id).classList.contains("session-running"), false);

  h.context.mergeSessionActivities([{ id, activity: "running" }]);
  for (let i = 0; i < 3; i++) {
    h.context.renderSessionList(h.context.lastItems);
    run(h, "paintSessionActivity('target-session');");
    assert.equal(row(h, id).querySelectorAll(".session-running-badge").length, 1);
    assert.equal(badge(h, id).textContent, "RUNNING");
  }
  h.context.mergeSessionActivities([{ id, activity: "unknown" }]);
  h.context.renderSessionList(h.context.lastItems);
  assert.equal(badge(h, id), null);
  assert.equal(row(h, id).classList.contains("session-running"), false);
});

test("a session parked on a question paints CONFIRM from the list payload", () => {
  const items = [
    { id: "parked", summary: "Parked", activity: "running", waiting: "confirm", waiting_prompt: "Approve the merge?" },
    { id: "busy", summary: "Busy", activity: "running" },
  ];
  const h = harness({ items, sid: "parked" });
  assert.equal(badge(h, "parked").textContent, "CONFIRM");
  assert.equal(badge(h, "parked").dataset.kind, "confirm");
  assert.equal(badge(h, "parked").title, "CONFIRM: Approve the merge?");
  assert.equal(row(h, "parked").classList.contains("session-waiting"), true);
  assert.equal(badge(h, "busy").textContent, "RUNNING", "a running session without a question keeps the running badge");
  assert.equal(row(h, "busy").classList.contains("session-waiting"), false);
  assert.match(css, /#sessions li\.session-waiting:not\(\.cur\) \{ box-shadow:inset 2px 0 0 var\(--mark\); \}/);
  assert.match(css, /#sessions li \.session-waiting-badge \{ background:var\(--session-waiting-bg\); color:var\(--session-waiting-fg\); box-shadow:inset 0 0 0 1px var\(--session-waiting-edge\); \}/);
});

test("answering the question downgrades the badge instead of leaving it parked", () => {
  const h = harness({ items: listed("running") }), id = "target-session";
  h.context.mergeSessionActivities([
    { id, activity: "running", waiting: "input", waiting_prompt: "paste the token:" },
  ]);
  run(h, "paintSessionActivity('target-session');");
  assert.equal(badge(h, id).textContent, "INPUT");
  assert.equal(badge(h, id).title, "INPUT: paste the token:");

  // The answer's next sample still runs the turn but carries no question.
  h.context.mergeSessionActivities([{ id, activity: "running" }]);
  run(h, "paintSessionActivity('target-session');");
  assert.equal(badge(h, id).textContent, "RUNNING");
  assert.equal(badge(h, id).dataset.kind, "running");
  assert.equal(row(h, id).classList.contains("session-waiting"), false);

  // An idle sample drops both badges.
  h.context.mergeSessionActivities([{ id, activity: "idle" }]);
  run(h, "paintSessionActivity('target-session');");
  assert.equal(badge(h, id), null);
});

test("activity merges preserve row identity, scroll, selection, filters and the rename editor", () => {
  const h = harness({ items: listed("running") });
  const before = [...h.elements.sesslist.children], count = h.elements.sesscount.textContent;
  h.elements.sesslist.scrollTop = 235;
  h.elements.sessfilter.value = "draft filter";
  const title = row(h, "target-session").querySelector("span"), click = title.onclick;
  h.context.startRename(row(h, "target-session"), h.context.lastItems[1]);
  const editor = row(h, "target-session").querySelector(".rnm");
  assert.equal(h.document.activeElement, editor);
  h.context.mergeSessionActivities([
    { id: "open-session", activity: "unknown" },
    { id: "target-session", activity: "idle" },
    { id: "not-loaded", activity: "running" },
  ]);
  sameRows(h, before);
  assert.equal(h.elements.sesslist.scrollTop, 235);
  assert.equal(h.elements.sesscount.textContent, count);
  assert.equal(h.elements.sessfilter.value, "draft filter");
  assert.equal(h.context.sessionId, "open-session");
  assert.equal(row(h, "open-session").classList.contains("cur"), true);
  assert.equal(editor.parentNode, row(h, "target-session"));
  assert.equal(h.document.activeElement, editor);
  assert.equal(title.onclick, click);
  assert.equal(running(h, "open-session"), false);
  assert.equal(running(h, "target-session"), false);
  assert.deepEqual(samples(h), [["open-session", "unknown"], ["target-session", "idle"]]);
  h.context.startRename(row(h, "target-session"), h.context.lastItems[1]);
  assert.equal(editor.parentNode, row(h, "target-session"), "the dot is never mistaken for a renameable title");

  run(h, "sessFilter = 'Target'; renderSessionList(lastItems);");
  const filtered = row(h, "target-session");
  h.context.mergeSessionActivities([{ id: "target-session", activity: "running" }]);
  assert.equal(row(h, "target-session"), filtered);
  assert.equal(h.elements.sesslist.children.length, 1);
  assert.equal(h.elements.sesscount.textContent, "2 sessions · showing 1");
  assert.equal(running(h, "target-session"), true);
});

test("visible polling requests only loaded full IDs and updates dots in place", async () => {
  const h = harness({
    items: listed(),
    activityGet: () => response([
      { id: "open-session", activity: "running" },
      { id: "unrequested", activity: "running" },
    ]),
  });
  const before = [...h.elements.sesslist.children];
  h.document.visibilityState = "visible";
  h.elements.sesslist.scrollTop = 123;
  await h.context.pollSessionActivity();
  assert.equal(activityRequests(h).length, 1);
  const request = activityRequests(h)[0];
  assert.equal(request.method, "POST");
  assert.deepEqual(plain(request.body), { ids: ["open-session", "target-session"] });
  assert.equal(request.headers.Authorization, "Bearer test-token");
  sameRows(h, before);
  assert.equal(h.elements.sesslist.scrollTop, 123);
  assert.equal(running(h, "open-session"), true);
  assert.equal(running(h, "target-session"), false, "an omitted status is unknown");
  assert.deepEqual(samples(h), [["open-session", "running"], ["target-session", "unknown"]]);
  h.document.visibilityState = "hidden";
  await h.context.pollSessionActivity();
  assert.equal(activityRequests(h).length, 1);
});

test("polls run every 2.5 seconds while visible and refresh immediately on return", async () => {
  const h = harness({ items: listed("running") });
  h.document.visibilityState = "visible";
  const before = [...h.elements.sesslist.children];
  h.context.wireSessionActivityPoll();
  await settle();
  assert.equal(activityRequests(h).length, 1);
  h.advance(2499);
  await settle();
  assert.equal(activityRequests(h).length, 1);
  h.advance(1);
  await settle();
  assert.equal(activityRequests(h).length, 2);
  h.document.visibilityState = "hidden";
  for (const fn of h.documentListeners.get("visibilitychange")) fn();
  await settle();
  assert.equal(running(h, "open-session"), false, "hidden samples become unknown");
  h.advance(10000);
  await settle();
  assert.equal(activityRequests(h).length, 2);
  h.document.visibilityState = "visible";
  for (const fn of h.documentListeners.get("visibilitychange")) fn();
  await settle();
  assert.equal(activityRequests(h).length, 3);
  assert.equal(running(h, "open-session"), true);
  sameRows(h, before);
});

test("failed or malformed samples clear running to unknown and allow recovery", async () => {
  let mode = "error";
  const h = harness({
    items: listed("running"),
    activityGet: () => mode === "error" ? response({ error: "offline" }, 503)
      : mode === "malformed" ? response({ sessions: [] })
      : response([{ id: "open-session", activity: "running" }, { id: "target-session", activity: "idle" }]),
  });
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), false);
  assert.deepEqual(samples(h), [["open-session", "unknown"], ["target-session", "unknown"]]);
  mode = "good";
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), true);
  mode = "malformed";
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), false);
  assert.deepEqual(samples(h), [["open-session", "unknown"], ["target-session", "unknown"]]);
  assert.equal(h.elements.sesscount.textContent, "2 sessions");
});

test("authoritative observations reject an older idle sample and reconcile on the next poll", async () => {
  const stale = deferred();
  let reads = 0;
  const h = harness({
    items: listed(), realNavigation: true,
    activityGet: () => ++reads === 1 ? stale.promise
      : response([{ id: "open-session", activity: "idle" }, { id: "target-session", activity: "idle" }]),
  });
  h.document.visibilityState = "visible";
  const pending = h.context.pollSessionActivity();
  h.context.updateExecutionModel({ kind: "begin" });
  const owner = h.context.updateExecutionModel({ kind: "activity" });
  h.context.updateExecutionModel({ kind: "live", entry: {} });
  h.context.mergeSessionActivities([{ id: "open-session", activity: "running" }]);
  assert.equal(running(h, "open-session"), true);
  stale.resolve(response([{ id: "open-session", activity: "idle" }, { id: "target-session", activity: "idle" }]));
  await pending;
  assert.equal(running(h, "open-session"), true);
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), false, "a newer backend sample reconciles the earlier observation");
  h.context.updateExecutionModel({ kind: "live", entry: {} });
  h.context.mergeSessionActivities([{ id: "open-session", activity: "running" }]);
  assert.equal(running(h, "open-session"), true);
  h.context.updateExecutionModel({ kind: "finish", owner, reason: "ended", answer: true });
  await settle();
  assert.equal(running(h, "open-session"), false, "the completion poll's idle sample clears the dot");
  assert.equal(activityRequests(h).length, 3, "completion immediately samples the backend");
});

test("a lost local stream becomes unknown when its fresh status sample fails", async () => {
  const h = harness({ items: listed(), realNavigation: true, activityGet: () => response({ error: "offline" }, 503) });
  h.context.updateExecutionModel({ kind: "begin" });
  h.context.updateExecutionModel({ kind: "live", entry: {} });
  h.context.mergeSessionActivities([{ id: "open-session", activity: "running" }]);
  assert.equal(running(h, "open-session"), true);
  h.context.updateExecutionModel({ kind: "live", entry: null });
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), false);
  assert.equal(samples(h).find(([id]) => id === "open-session")[1], "unknown");
});

test("list loading and paging refresh only the newly loaded set", async () => {
  const items = Array.from({ length: 30 }, (_, index) => ({
    id: "listed-" + index, summary: "Session " + index, activity: index % 2 ? "running" : "idle",
  }));
  const h = harness({ items, cachedItems: [], sid: "" });
  h.document.visibilityState = "visible";
  await h.context.loadSessions();
  await settle();
  assert.equal(h.context.lastItems.length, 12);
  assert.deepEqual(plain(activityRequests(h)[0].body.ids), plain(h.context.lastItems.map((item) => item.id)));
  for (const item of h.context.lastItems) assert.equal(running(h, item.id), item.activity === "running");
  await h.context.loadOlderSessions();
  await settle();
  assert.equal(h.context.lastItems.length, 24);
  assert.deepEqual(plain(activityRequests(h).at(-1).body.ids), plain(h.context.lastItems.map((item) => item.id)));
  assert.equal(h.elements.sesscount.textContent, "30 sessions");
});

test("bulk polling splits large loaded sets into bounded requests without loading history", async () => {
  const items = Array.from({ length: 300 }, (_, index) => ({ id: "full-id-" + index, activity: "idle" }));
  const h = harness({ items });
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  assert.deepEqual(plain(activityRequests(h).map((request) => request.body.ids.length)), [256, 44]);
  assert.deepEqual(plain(activityRequests(h).flatMap((request) => request.body.ids)), items.map((item) => item.id));
  assert.equal(h.requests.length, 2);
  assert.equal(h.historyLoads(), 0);
});

test("removed IDs and responses from a previous server cannot restore stale activity", async () => {
  const stale = deferred();
  const h = harness({ items: listed(), activityGet: () => stale.promise });
  h.document.visibilityState = "visible";
  const pending = h.context.pollSessionActivity();
  h.context.lastItems = [h.context.lastItems[1]];
  h.context.renderSessionList(h.context.lastItems);
  stale.resolve(response([{ id: "open-session", activity: "running" }, { id: "target-session", activity: "running" }]));
  await pending;
  assert.deepEqual(samples(h), [["target-session", "running"]]);
  assert.equal(row(h, "open-session"), undefined);

  const oldServer = deferred();
  const other = harness({ items: listed(), activityGet: () => oldServer.promise });
  other.document.visibilityState = "visible";
  const oldPoll = other.context.pollSessionActivity();
  other.context.store.base = "https://other.test";
  oldServer.resolve(response([{ id: "open-session", activity: "running" }]));
  await oldPoll;
  assert.equal(running(other, "open-session"), false);
  assert.equal(activityRequests(other).length, 1);
});

test("a response received after its request timeout remains unknown", async () => {
  const slow = deferred();
  const h = harness({ items: listed("running"), activityGet: () => slow.promise });
  h.document.visibilityState = "visible";
  const pending = h.context.pollSessionActivity();
  h.advance(5001);
  slow.resolve(response([{ id: "open-session", activity: "running" }, { id: "target-session", activity: "running" }]));
  await pending;
  assert.equal(running(h, "open-session"), false);
  assert.deepEqual(samples(h), [["open-session", "unknown"], ["target-session", "unknown"]]);
});

test("later chunks drop IDs unloaded while a previous chunk was in flight", async () => {
  const items = Array.from({ length: 300 }, (_, index) => ({ id: "chunk-id-" + index, activity: "idle" }));
  let h;
  h = harness({
    items,
    activityGet: (request) => {
      h.context.lastItems = h.context.lastItems.slice(0, 256);
      h.context.renderSessionList(h.context.lastItems);
      return response(request.body.ids.map((id) => ({ id, activity: "idle" })));
    },
  });
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  assert.equal(activityRequests(h).length, 1);
  assert.equal(samples(h).length, 256);
  assert.equal(row(h, "chunk-id-299"), undefined);
});

test("expired samples become unknown and an older daemon leaves status unclaimed", async () => {
  const h = harness({ items: listed("running") });
  h.advance(8000);
  h.context.expireSessionActivities();
  assert.equal(running(h, "open-session"), false);
  assert.deepEqual(samples(h), [["open-session", "unknown"], ["target-session", "unknown"]]);
  const legacy = harness();
  legacy.document.visibilityState = "visible";
  await legacy.context.pollSessionActivity();
  assert.deepEqual(plain(legacy.requests), []);
  assert.deepEqual(samples(legacy), [["open-session", "unknown"], ["target-session", "unknown"]]);
});

test("running rail and compact label use theme tokens without overriding selection", () => {
  const rail = "#sessions li.session-running {";
  assert.match(css, /#sessions li\.session-running \{ background:var\(--tint-2\); box-shadow:inset 2px 0 0 var\(--acc-line\); \}/);
  assert.ok(css.indexOf(rail) < css.indexOf("#sessions li.cur {"), "selection keeps its stronger rail and wash");
  assert.match(css, /#sessions li\.cur \{ background:var\(--session-selected-bg\); box-shadow:inset 3px 0 0 var\(--acc\); \}/);
  assert.match(css, /#sessions li\.cur\[data-session-id\] \{ background:var\(--session-selected-bg\); border-color:var\(--session-selected-edge\); \}/);
  assert.match(css, /#sessions li\.cur, #sessions li\.cur:hover \{\s*background:var\(--session-selected-bg\); box-shadow:inset 3px 0 0 var\(--acc\);/);
  assert.doesNotMatch(css, /#sessions li\.cur\[data-session-id\] \{[^}]*border-color:var\(--acc-line\)/);
  assert.match(css, /#sessions li \.session-running-badge \{[^}]*flex:0 0 auto;[^}]*border-radius:999px; background:var\(--session-running-bg\); color:var\(--session-running-fg\); box-shadow:inset 0 0 0 1px var\(--session-running-edge\); font-size:11px;[^}]*white-space:nowrap;/);
  assert.match(css, /#sessions li \.session-running-label \{ letter-spacing:\.04em; \}/);
});

test("all themes keep selection distinct and balance status fills", (t) => {
  const desktopStart = css.indexOf("@media (min-width:701px)");
  const desktopEnd = css.indexOf("@media (max-width:700px)", desktopStart);
  const shared = css.slice(0, desktopStart);
  const desktop = css.slice(desktopStart, desktopEnd);
  const tokens = (source, theme) => {
    const selector = theme ? ':root\\[data-theme="' + theme + '"\\]' : ':root';
    const body = source.match(new RegExp(selector + ' \\{([^}]+)\\}'))?.[1];
    assert.ok(body, "theme palette exists: " + (theme || "dark"));
    return Object.fromEntries([...body.matchAll(/--([\w-]+):([^;]+);/g)].map((match) => [match[1], match[2].trim()]));
  };
  const rgb = (hex) => {
    assert.match(hex, /^#[\da-f]{6}$/i, "status colors are opaque");
    return hex.slice(1).match(/../g).map((channel) => parseInt(channel, 16));
  };
  const luminance = (channels) => {
    const [r, g, b] = channels.map((channel) => {
      const s = channel / 255;
      return s <= .04045 ? s / 12.92 : ((s + .055) / 1.055) ** 2.4;
    });
    return .2126 * r + .7152 * g + .0722 * b;
  };
  const contrast = (a, b) => {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + .05) / (low + .05);
  };
  for (const theme of ["dark", "light", "warm"]) for (const phone of [true, false]) {
    const palette = { ...tokens(shared), ...(theme !== "dark" ? tokens(shared, theme) : {}),
      ...(!phone ? tokens(desktop) : {}), ...(!phone && theme !== "dark" ? tokens(desktop, theme) : {}) };
    const panel = rgb(palette.panel);
    const selected = rgb(palette["session-selected-bg"]);
    const selectedEdge = rgb(palette["session-selected-edge"]);
    assert.ok(contrast(selectedEdge, panel) >= 3 && contrast(selectedEdge, selected) >= 3,
      "selected card boundary separates from both the panel and its fill");
    for (const name of ["fg", "dim"]) assert.ok(contrast(rgb(palette[name]), selected) >= 4.5,
      "selected title and metadata remain readable");
    t.diagnostic(`${theme} ${phone ? "phone" : "desktop"} selected: edge/panel ${contrast(selectedEdge, panel).toFixed(2)}:1; edge/fill ${contrast(selectedEdge, selected).toFixed(2)}:1; title/fill ${contrast(rgb(palette.fg), selected).toFixed(2)}:1; metadata/fill ${contrast(rgb(palette.dim), selected).toFixed(2)}:1`);
    const surfaces = [panel, rgb(palette["surface-2"]), rgb(palette["surface-hover"]), selected];
    for (const name of ["tint-2", "tint-4"]) {
      const rgba = palette[name].match(/^rgba\((\d+),(\d+),(\d+),([\d.]+)\)$/);
      assert.ok(rgba, "card tint is defined");
      const alpha = Number(rgba[4]);
      surfaces.push(panel.map((channel, i) => Number(rgba[i + 1]) * alpha + channel * (1 - alpha)));
    }
    for (const state of ["running", "waiting"]) {
      const fill = rgb(palette["session-" + state + "-bg"]);
      const edge = rgb(palette["session-" + state + "-edge"]);
      const textRatio = contrast(rgb(palette["session-" + state + "-fg"]), fill);
      const fillRatios = surfaces.map((surface) => contrast(fill, surface));
      const edgeRatio = Math.min(...surfaces.map((surface) => contrast(edge, surface)));
      assert.ok(textRatio >= 4.5, "small status text meets 4.5:1");
      // The outline carries recognition while the fill stays close to the card.
      assert.ok(Math.max(...fillRatios) <= 2.5, "status fill avoids a dominant solid block");
      assert.ok(edgeRatio >= 3, "inset edge separates from every card state and group header");
      if (theme === "dark") assert.ok(luminance(fill) > luminance(panel) && contrast(fill, panel) >= 1.5,
        "dark status fill is visibly lighter than the surrounding panel");
      assert.ok(state === "running" ? fill[1] > fill[0] && fill[1] > fill[2] : fill[0] > fill[1] && fill[1] > fill[2], "green and amber remain distinct");
      t.diagnostic(`${theme} ${phone ? "phone" : "desktop"} ${state}: text/fill ${textRatio.toFixed(2)}:1; fill/surface ${Math.min(...fillRatios).toFixed(2)}-${Math.max(...fillRatios).toFixed(2)}:1; minimum edge/surface ${edgeRatio.toFixed(2)}:1`);
    }
  }
  for (const [selector, state] of [["#sessions .session-group-badge", "running"], ["#sessions .session-group-badge.waiting", "waiting"]]) {
    const body = css.match(new RegExp(selector.replaceAll(".", "\\.") + " \\{([^}]+)\\}"))?.[1];
    assert.ok(body, "group status style exists");
    for (const [property, suffix] of [["background", "bg"], ["color", "fg"], ["box-shadow:inset 0 0 0 1px", "edge"]]) {
      assert.ok(body.includes(property + (property.startsWith("box-shadow") ? " " : ":") + `var(--session-${state}-${suffix})`), "group badge shares the status palette");
    }
  }
});

test("titles wrap fully at a moderate weight and waiting labels stay visible", () => {
  for (const phone of [false, true]) {
    const titles = ["Short title", "发布 test_aeolus_coding_node 卡片",
      "为 agent webui 增加会话分组/话题", "Review MR !79 的 Codebase 授权入口缺失",
      "feat/2026-11-17-critical-premises", "Long identifier " + "x".repeat(160)];
    const items = titles.map((summary, index) => ({ ...sessions()[0], id: "title-" + index, summary }));
    const h = harness({ phone, items });
    for (const item of items) assert.equal(row(h, item.id).querySelector(".session-title").textContent, item.summary);
    const tokens = (index) => Array.from(row(h, items[index].id).querySelectorAll(".session-title-token"), (token) => token.textContent);
    assert.deepEqual(tokens(1), ["test_aeolus_coding_node"]);
    assert.deepEqual(tokens(4), ["feat/2026-11-17-critical-premises"]);
    assert.deepEqual(tokens(5), ["Long", "identifier", "x".repeat(160)]);
  }
  assert.match(css, /#sessions li \.session-title \{ display:block; white-space:normal; overflow-wrap:break-word; word-break:normal;[^}]*color:var\(--fg\); font-size:13px; font-weight:500; line-height:22px;/);
  assert.match(css, /#sessions li \.session-title-token \{ display:inline-block; max-width:100%; vertical-align:top; overflow-wrap:anywhere; \}/);
  assert.doesNotMatch(css, /#sessions[^{}]*\.session-title\s*\{[^}]*(?:line-clamp|[;\s](?:min-|max-)?height\s*:|overflow\s*:\s*hidden|text-overflow\s*:)/);
  assert.doesNotMatch(css, /#sessions li\[data-session-id\]\s*\{[^}]*[;\s](?:min-|max-)?height\s*:/);
  assert.doesNotMatch(css, /#sessions li\.cur span\[role="button"\] \{[^}]*font-weight/);
  assert.match(css, /#sessions li\.session-running \.session-meta \{ display:flex; align-items:center; flex-wrap:wrap; gap:4px 8px; min-width:0; container:session-meta \/ inline-size; \}/);
  assert.doesNotMatch(css, /#sessions[^{}]*(?:\.session-running|\.session-waiting|\.cur)[^{}]*\.session-title\s*\{/);
  assert.match(css, /#sessions li\.session-running \.session-hash \{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; \}/);
  assert.match(css, /#sessions li \.session-hash \{ display:none; \}/);
  assert.match(css, /@container session-meta \(max-width:72px\) \{\s*#sessions li \.session-running-label:not\(\.session-waiting-label\) \{ display:none; \}/);
  assert.doesNotMatch(css, /@media[^{}]*\{[^}]*\.session-running-label/);
});

test("only the six-pixel dot breathes and reduced motion disables its animation", () => {
  assert.match(css, /#sessions li \.session-running-dot \{[^}]*flex:0 0 6px; width:6px; height:6px;[^}]*background:currentColor;[^}]*animation:session-running-pulse 2\.8s ease-in-out infinite;/);
  assert.doesNotMatch(css, /#sessions li \.session-running-badge \{[^}]*animation:/);
  assert.match(css, /@keyframes session-running-pulse/);
  assert.match(css, /@media \(prefers-reduced-motion:reduce\) \{ #sessions li \.session-running-dot \{ animation:none; \} \}/);
});

test("pending and accepted local submissions cannot synthesize a running dot", async () => {
  const h = harness({ items: listed(), realNavigation: true });
  h.context.updateExecutionModel({ kind: "begin" });
  h.context.updateExecutionModel({ kind: "live", entry: { userInput: { settled: false } } });
  const pending = { requests: h.requests.length, backend_fixture: "idle", dot: running(h, "open-session"),
    sample: samples(h).find(([id]) => id === "open-session")[1] };
  assert.equal(pending.dot, false, JSON.stringify(pending));
  assert.equal(pending.sample, "idle");
  assert.equal(pending.requests, 0);
  h.context.executionState().liveEntry.userInput.settled = true;
  assert.equal(running(h, "open-session"), false, "HTTP acceptance alone cannot prove that a child is running");
  h.document.visibilityState = "visible";
  await h.context.pollSessionActivity();
  assert.equal(running(h, "open-session"), false, "an idle server sample stays idle after submission");
});

test("real submission and acceptance sample activity without promoting an idle session", async () => {
  const h = sendHarness(), c = h.c;
  let backend = "idle", reads = 0;
  c.lastItems = c.lastItems.map((item) => ({ ...item, activity: backend }));
  c.renderSessionList(c.lastItems);
  const fetch = c.fetch;
  c.fetch = async (url, options) => {
    const pathname = new URL(url).pathname;
    if (pathname === "/session-activity") {
      reads++;
      const { ids } = JSON.parse(options.body);
      return response(ids.map((id) => ({ id, activity: backend })));
    }
    if (pathname === "/sessions") return response(c.lastItems.map((item) => ({ ...item, activity: backend })), 200, {
      "x-sessions-total": String(c.lastItems.length), "x-sessions-page-size": String(c.lastItems.length),
    });
    return fetch(url, options);
  };
  const isRunning = () => Boolean(h.elements.sesslist.children.find((item) => item.dataset.sessionId === c.sessionId)
    .querySelector(".session-running-badge"));
  await h.start();
  assert.equal(h.requests.length, 1, "the actual turn POST was submitted");
  assert.ok(reads >= 1, "submission immediately requests authoritative activity");
  assert.equal(isRunning(), false);
  await h.accept();
  assert.ok(reads >= 2, "acceptance reconciles activity again");
  assert.equal(isRunning(), false, "accepted HTTP headers still do not prove a child is registered");
  backend = "running";
  await c.pollSessionActivity();
  assert.equal(isRunning(), true);
  const beforeDone = reads;
  await h.frame("done", {});
  assert.ok(reads > beforeDone, "completion immediately refreshes authoritative activity");
  assert.equal(isRunning(), true, "done does not clear a still-registered child");
  backend = "idle";
  await c.pollSessionActivity();
  assert.equal(isRunning(), false);
  await h.dispose();
});

test("an unloaded and reloaded ID rejects an earlier running sample", async () => {
  const stale = deferred();
  const item = { id: "reloaded-session", summary: "Reloaded", activity: "idle" };
  const h = harness({ items: [item], sid: item.id, activityGet: () => stale.promise });
  h.context.mergeSessionActivities([{ id: item.id, activity: "idle" }]);
  const oldRevision = run(h, "sessionActivityRevisions.get('reloaded-session')");
  h.document.visibilityState = "visible";
  const pending = h.context.pollSessionActivity();
  h.context.lastItems = [];
  h.context.renderSessionList([]);
  h.context.lastItems = [item];
  h.context.renderSessionList(h.context.lastItems);
  h.context.mergeSessionActivities([{ id: item.id, activity: "idle" }]);
  const reloadedRevision = run(h, "sessionActivityRevisions.get('reloaded-session')");
  assert.equal(running(h, item.id), false);
  stale.resolve(response([{ id: item.id, activity: "running" }]));
  await pending;
  const finalRevision = run(h, "sessionActivityRevisions.get('reloaded-session')");
  assert.equal(running(h, item.id), false, JSON.stringify({ oldRevision, reloadedRevision, finalRevision, dot: running(h, item.id) }));
  assert.notEqual(oldRevision, reloadedRevision, "generations must not repeat after reloading an ID");
});

test("done cannot produce an idle-then-running flip while the child is still registered", async () => {
  const held = deferred();
  const h = harness({
    items: listed("running"), realNavigation: true,
    activityGet: () => held.promise,
  });
  h.context.updateExecutionModel({ kind: "begin" });
  const owner = h.context.updateExecutionModel({ kind: "activity" });
  h.context.updateExecutionModel({ kind: "live", entry: {} });
  h.document.visibilityState = "visible";
  const sequence = [running(h, "open-session")];
  h.context.updateExecutionModel({ kind: "finish", owner, reason: "ended", answer: true });
  sequence.push(running(h, "open-session"));
  held.resolve(response(listed("running").map(({ id, activity }) => ({ id, activity }))));
  await settle();
  sequence.push(running(h, "open-session"));
  assert.deepEqual(sequence, [true, true, true], "done precedes child reaping, so it cannot clear the running dot");
  h.context.mergeSessionActivities([{ id: "open-session", activity: "idle" }]);
  assert.equal(running(h, "open-session"), false, "the first authoritative idle sample clears the dot");
});
