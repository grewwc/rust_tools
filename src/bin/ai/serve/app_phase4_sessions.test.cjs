// Run with: env NODE_OPTIONS= node --test --test-reporter=tap src/bin/ai/serve/app_phase4_sessions.test.cjs
// Characterize session calculations through the existing page and DOM fixture.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const appFile = process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html");
const html = fs.readFileSync(appFile, "utf8");
const fixtureFile = path.join(__dirname, "app_desktop.test.cjs");
const fixtureSource = fs.readFileSync(fixtureFile, "utf8");
const fixtures = new Map();
const plain = (value) => JSON.parse(JSON.stringify(value));

function harness(options = {}, source = html) {
  if (!fixtures.has(source)) {
    const fixtureFs = { ...fs, readFileSync(file, ...args) {
      if (String(file) === appFile || String(file) === path.join(__dirname, "app.html")) {
        return args[0] ? source : Buffer.from(source);
      }
      return fs.readFileSync(file, ...args);
    } };
    fixtures.set(source, new vm.Script(fixtureSource + "\n;({ harness });", { filename: fixtureFile })
      .runInNewContext({
        require: (name) => name === "node:test" ? { test() {} }
          : name === "node:fs" ? fixtureFs : require(name),
        __dirname, __filename: fixtureFile, process, console, Buffer,
        TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
        setTimeout, clearTimeout, setInterval, clearInterval, setImmediate, queueMicrotask,
      }));
  }
  return fixtures.get(source).harness(options);
}

function setPager(h, state = {}) {
  h.context.phase4Pager = { total: null, offset: 0, hasMore: false, ...state };
  vm.runInContext("sessionsTotal = phase4Pager.total; sessionsPageOffset = phase4Pager.offset; sessionsHasMore = phase4Pager.hasMore;", h.context);
}

function setFilter(h, query) {
  h.context.phase4Filter = query;
  vm.runInContext("sessFilter = phase4Filter;", h.context);
}

function pageState(h) {
  return vm.runInContext("({ total: sessionsTotal, offset: sessionsPageOffset, hasMore: sessionsHasMore, loading: sessionsLoadingMore, items: lastItems })", h.context);
}

function rendered(h) {
  return {
    titles: Array.from(h.elements.sesslist.children, (row) => row.querySelector("span")?.textContent).filter((title) => title !== undefined),
    count: h.elements.sesscount.textContent,
    tail: Array.from(h.elements.sesslist.children).filter((row) => !row.querySelector("span")).map((row) => row.textContent),
    scrollTop: h.elements.sesslist.scrollTop,
  };
}

function installResponse(h, value, headers = {}) {
  const calls = [];
  h.context.api = async (url) => {
    calls.push(url);
    return { json: async () => value, headers: { get: (name) => headers[name] ?? null } };
  };
  return calls;
}

const listed = [
  { id: "Alpha-ID", summary: "FIRST Summary", first_user_prompt: "hidden prompt", title: "ignored title" },
  { id: "beta-id", summary: "", first_user_prompt: "Second Prompt" },
  { id: "fallback-123", summary: null, first_user_prompt: "" },
  { id: "unicode-id", summary: "中文 [literal] École" },
];

test("empty star-filter pages expose older sessions and keep filtering appended results", async () => {
  const h = harness();
  setPager(h, { total: 3, offset: 2, hasMore: true });
  h.dispatch(h.elements.sessmarked, "click");
  assert.deepEqual(rendered(h).tail, ["暂无星标会话（已加载范围内）", "加载更多会话"]);
  const calls = installResponse(h, [{ id: "older-star", summary: "Older star", marked: true }], {
    "x-sessions-total": "3", "x-sessions-page-size": "1",
  });
  const more = h.elements.sesslist.querySelector("button");
  await h.dispatch(more, "click").done;
  assert.deepEqual(calls, ["/sessions?newest=true&limit=12&offset=2"]);
  assert.deepEqual(rendered(h).titles, ["Older star"]);
  assert.deepEqual(rendered(h).tail, []);
  assert.equal(rendered(h).count, "3 sessions · showing 1");
  assert.equal(h.context.sessionId, "open-session");
});

test("blank and whitespace session filters preserve input order and title fallback rules", () => {
  for (const query of ["", " \t\n "]) {
    const h = harness();
    setFilter(h, query);
    h.context.renderSessionList(listed);
    assert.deepEqual(rendered(h).titles, ["FIRST Summary", "Second Prompt", "new session · fallback", "中文 [literal] École"]);
    assert.equal(rendered(h).count, "4 sessions");
    assert.deepEqual(plain(h.requests), []);
  }
});

test("session filters trim and fold case across displayed titles and complete IDs", () => {
  for (const [query, expected] of [
    [" first SUMMARY ", ["FIRST Summary"]], ["ALPHA-id", ["FIRST Summary"]],
    ["second prompt", ["Second Prompt"]], ["fallback-123", ["new session · fallback"]],
    ["new session", ["new session · fallback"]], ["中文", ["中文 [literal] École"]],
    ["[literal]", ["中文 [literal] École"]], ["éCOLE", ["中文 [literal] École"]],
    ["hidden prompt", []], ["ignored title", []], [".*", []],
  ]) {
    const h = harness();
    setFilter(h, query);
    h.context.renderSessionList(listed);
    assert.deepEqual(rendered(h).titles, expected, query);
    assert.equal(rendered(h).count, "4 sessions · showing " + expected.length, query);
  }
});

test("filtered session counts distinguish partially loaded totals and exclude the error row", () => {
  for (const [total, expected] of [[10, "10 sessions · showing 1 loaded"], [4, "4 sessions · showing 1"],
    [2, "2 sessions · showing 1"], [null, "4 sessions · showing 1"], ["10", "4 sessions · showing 1"]]) {
    const h = harness();
    setPager(h, { total });
    setFilter(h, "beta");
    h.context.renderSessionList(listed, "sessions refresh failed: token");
    assert.equal(rendered(h).count, expected);
    assert.deepEqual(rendered(h).tail, ["sessions refresh failed: token"]);
  }
});

test("numeric session totals retain zero negative fractional and non-finite count text", () => {
  for (const total of [0, -3, 2.5, NaN, Infinity, -Infinity]) {
    for (const query of ["", "no match"]) {
      const h = harness();
      setPager(h, { total });
      setFilter(h, query);
      h.context.renderSessionList(listed);
      assert.equal(rendered(h).count, String(total) + " sessions" + (query ? " · showing 0" + (listed.length < total ? " loaded" : "") : ""));
    }
  }
});

test("empty session lists retain filtered count suffixes and truthy error text verbatim", () => {
  for (const [query, total, count] of [["", null, "0 sessions"], ["missing", null, "0 sessions · showing 0"],
    ["missing", 3, "3 sessions · showing 0 loaded"]]) {
    for (const error of [undefined, "", "failure\n<raw>", 7]) {
      const h = harness();
      setPager(h, { total });
      setFilter(h, query);
      h.context.renderSessionList([], error);
      assert.equal(rendered(h).count, count);
      assert.deepEqual(rendered(h).tail, error ? [String(error)] : []);
    }
  }
});

test("malformed session filters fail before replacing existing list content", () => {
  for (const query of [null, undefined, 7, {}, { trim: 3 }]) {
    const h = harness(), before = rendered(h);
    setFilter(h, query);
    assert.throws(() => h.context.renderSessionList(listed), { name: "TypeError" });
    assert.deepEqual(rendered(h), before);
  }
  const h = harness();
  setFilter(h, 7);
  assert.throws(() => h.context.renderSessionList(listed), { name: "TypeError", message: "sessFilter.trim is not a function" });
});

test("malformed filtered lists preserve their native failure text and existing rows", () => {
  const h = harness(), before = rendered(h);
  setFilter(h, "x");
  assert.throws(() => h.context.renderSessionList({}), { name: "TypeError", message: "items.filter is not a function" });
  assert.deepEqual(rendered(h), before);
});

test("session filtering and rendering accept frozen data without mutating it", () => {
  const items = Object.freeze(listed.map((item) => Object.freeze({ ...item })));
  const before = plain(items), h = harness();
  setFilter(h, "id");
  h.context.renderSessionList(items);
  assert.deepEqual(plain(items), before);
  assert.deepEqual(rendered(h).titles, ["FIRST Summary", "Second Prompt", "中文 [literal] École"]);
});

test("session rendering preserves scroll writes at the existing render options", () => {
  for (const [options, expected] of [[{}, 91], [{ preserveScroll: false }, 0], [{ followNewest: true }, 0]]) {
    const h = harness();
    h.elements.sesslist.scrollTop = 91;
    h.context.renderSessionList(listed, undefined, options);
    assert.equal(h.elements.sesslist.scrollTop, expected);
  }
});

test("bounded response projection preserves field order array identity and uncapped headers", async () => {
  const h = harness(), items = Object.freeze([{ id: "selected" }]);
  const calls = installResponse(h, items, { "x-sessions-total": "2.5", "x-sessions-page-size": "42" });
  const page = await h.context.fetchSessionPage(-2, 12, true);
  assert.deepEqual(Object.keys(page), ["items", "total", "pageSize"]);
  assert.equal(page.items, items);
  assert.equal(page.total, 2.5);
  assert.equal(page.pageSize, 42);
  assert.deepEqual(calls, ["/sessions?newest=true&limit=12&offset=-2&include=open-session"]);
});

test("session page headers retain native Number conversion and finite fallback behavior", async () => {
  for (const [header, expected] of [[null, null], [undefined, null], ["", 0], [" ", 0], ["0", 0],
    ["-2", -2], ["2.5", 2.5], ["1e2", 100], ["0x10", 16], [true, 1], [false, 0],
    ["NaN", null], ["Infinity", null], ["-Infinity", null], ["bad", null]]) {
    const h = harness();
    installResponse(h, [{ id: "a" }, { id: "b" }], { "x-sessions-total": header, "x-sessions-page-size": header });
    const page = await h.context.fetchSessionPage(0, 1);
    assert.equal(page.total, expected, String(header));
    assert.equal(page.pageSize, expected === null ? 1 : expected, String(header));
  }
});

test("non-array and missing session bodies normalize to an empty page", async () => {
  for (const value of [undefined, null, false, 0, "bad", {}, { items: [{ id: "a" }] }]) {
    const h = harness();
    installResponse(h, value);
    const page = await h.context.fetchSessionPage(0, 12);
    assert.deepEqual(plain(page), { items: [], total: null, pageSize: 0 });
  }
});

test("missing and unusable header accessors keep the page length fallback", async () => {
  for (const headers of [undefined, null, {}, { get: 7 }]) {
    const h = harness();
    h.context.api = async () => ({ headers, json: async () => listed });
    assert.deepEqual(plain(await h.context.fetchSessionPage(0, 2)), { items: plain(listed), total: null, pageSize: 2 });
  }
});

test("page-size fallback retains limit coercion and boundary behavior", async () => {
  for (const [limit, expected] of [[0, 0], [-2, -2], [2.5, 2.5], [Infinity, 4], [-Infinity, -Infinity],
    [NaN, NaN], [undefined, NaN], [null, 0], ["2", 2], ["bad", NaN]]) {
    const h = harness();
    installResponse(h, listed);
    assert.equal((await h.context.fetchSessionPage(0, limit)).pageSize, expected);
  }
});

test("header reads precede numeric conversion and preserve thrown error details", async () => {
  const h = harness(), effects = [];
  h.context.api = async () => ({ json: async () => listed, headers: { get(name) {
    effects.push(name);
    return { valueOf() { effects.push("number:" + name); throw new RangeError("controlled header failure"); } };
  } } });
  await assert.rejects(h.context.fetchSessionPage(0, 12), { name: "RangeError", message: "controlled header failure" });
  assert.deepEqual(effects, ["x-sessions-total", "x-sessions-page-size", "number:x-sessions-total"]);
});

test("refresh requests preserve the loaded offset as the minimum page limit", async () => {
  for (const [offset, limit] of [[0, 12], [-1, 12], [2.5, 12], [12, 12], [13, 13], [24, 24],
    [undefined, 12], [null, 12], [NaN, 12], [-Infinity, 12], [Infinity, Infinity], ["18", 18], ["bad", NaN]]) {
    const h = harness();
    setPager(h, { offset });
    const calls = installResponse(h, [{ id: "open-session", summary: "Open" }], { "x-sessions-total": "9", "x-sessions-page-size": "1" });
    await h.context.loadSessions();
    assert.deepEqual(calls, ["/sessions?newest=true&limit=" + limit + "&offset=0&include=open-session"]);
    assert.equal(pageState(h).offset, 1);
    assert.equal(pageState(h).hasMore, true);
  }
});

test("bounded refresh retains exact offset-total comparisons including empty pages", async () => {
  for (const [total, pageSize, expected] of [[0, 0, false], [1, 0, true], [12, 12, false], [13, 12, true],
    [-2, -3, true], [-2, 0, false], [2.5, 2, true], [2.5, 2.5, false], [1, 99, false]]) {
    const h = harness({ sid: "" });
    installResponse(h, [], { "x-sessions-total": String(total), "x-sessions-page-size": String(pageSize) });
    await h.context.loadSessions();
    const state = pageState(h);
    assert.equal(state.total, total);
    assert.equal(state.offset, pageSize);
    assert.equal(state.hasMore, expected);
    assert.equal(rendered(h).count, total + " sessions");
  }
});

test("legacy refresh reverses the full response and disables incremental paging", async () => {
  const h = harness(), legacy = [{ id: "first", summary: "Oldest" }, { id: "open-session", summary: "Newest" }];
  const calls = [];
  h.context.api = async (url) => {
    calls.push(url);
    return { json: async () => url === "/sessions" ? legacy : [{ id: "ignored-bounded" }] };
  };
  await h.context.loadSessions();
  assert.deepEqual(calls, ["/sessions?newest=true&limit=12&offset=0&include=open-session", "/sessions"]);
  assert.deepEqual(rendered(h).titles, ["Newest", "Oldest"]);
  assert.equal(pageState(h).total, 2);
  assert.equal(pageState(h).offset, 2);
  assert.equal(pageState(h).hasMore, false);
});

test("legacy refresh mutates the original response array in place", async () => {
  const h = harness(), legacy = [{ id: "first", summary: "Oldest" },
    { id: "middle", summary: "Middle" }, { id: "open-session", summary: "Newest" }];
  const original = legacy.slice();
  installResponse(h, legacy);
  await h.context.loadSessions();
  assert.deepEqual(legacy.map((item) => item.id), ["open-session", "middle", "first"]);
  assert.equal(legacy[0], original[2]);
  assert.equal(legacy[1], original[1]);
  assert.equal(legacy[2], original[0]);
});

test("legacy refresh commits the exact response array object", async () => {
  for (const legacy of [[], [{ id: "open-session" }], [{ id: "first" }, { id: "open-session" }]]) {
    const h = harness();
    installResponse(h, legacy);
    await h.context.loadSessions();
    assert.equal(pageState(h).items, legacy);
    assert.equal(h.context.lastItems, legacy);
  }
});

test("legacy refresh surfaces the native TypeError from reversing a frozen response", async () => {
  const h = harness(), legacy = [{ id: "first" }, { id: "open-session" }];
  let failure;
  Object.defineProperty(legacy, "reverse", { value() {
    try { return Array.prototype.reverse.call(this); }
    catch (error) { failure = { name: error.name, message: error.message }; throw error; }
  } });
  Object.freeze(legacy);
  installResponse(h, legacy);
  await h.context.loadSessions();
  assert.deepEqual(failure, { name: "TypeError",
    message: "Cannot assign to read only property '0' of object '[object Array]'" });
  assert.deepEqual(rendered(h).tail, ["sessions refresh failed: " + failure.message]);
});

test("legacy refresh preserves the previous list and pager when the response is frozen", async () => {
  const h = harness(), items = h.context.lastItems;
  setPager(h, { total: 31, offset: 24, hasMore: true });
  h.context.renderSessionList(items);
  h.elements.sesslist.scrollTop = 87;
  const before = rendered(h), selected = h.context.sessionId;
  const legacy = Object.freeze([{ id: "first", summary: "Oldest" }, { id: "open-session", summary: "Newest" }]);
  installResponse(h, legacy);
  await h.context.loadSessions();
  const state = pageState(h);
  assert.equal(state.items, items);
  assert.equal(h.context.lastItems, items);
  assert.equal(state.total, 31);
  assert.equal(state.offset, 24);
  assert.equal(state.hasMore, true);
  assert.equal(state.loading, false);
  assert.equal(h.context.sessionId, selected);
  assert.deepEqual(plain(legacy), [{ id: "first", summary: "Oldest" }, { id: "open-session", summary: "Newest" }]);
  assert.deepEqual(rendered(h), { ...before,
    tail: ["sessions refresh failed: Cannot assign to read only property '0' of object '[object Array]'"] });
});

test("legacy refresh retains the native order of indexed array writes", async () => {
  const h = harness(), writes = [];
  const legacy = new Proxy([{ id: "first" }, { id: "second" }, { id: "middle" },
    { id: "fourth" }, { id: "open-session" }], { set(target, key, value, receiver) {
    writes.push([key, value.id]);
    return Reflect.set(target, key, value, receiver);
  } });
  installResponse(h, legacy);
  await h.context.loadSessions();
  assert.deepEqual(writes, [["0", "open-session"], ["4", "first"], ["1", "fourth"], ["3", "second"]]);
  assert.equal(pageState(h).items, legacy);
});

test("malformed legacy full responses normalize to an empty committed list", async () => {
  for (const legacy of [null, undefined, {}, "bad", false, 7]) {
    const h = harness();
    h.context.api = async (url) => ({ json: async () => url === "/sessions" ? legacy : [] });
    await h.context.loadSessions();
    assert.deepEqual(plain(pageState(h).items), []);
    assert.equal(pageState(h).total, 0);
    assert.equal(pageState(h).offset, 0);
    assert.equal(pageState(h).hasMore, false);
    assert.equal(h.context.sessionId, "");
  }
});

test("failed bounded or legacy refresh preserves the last good list and pager", async () => {
  for (const stage of ["body", "headers", "legacy"]) {
    const h = harness(), before = h.context.lastItems;
    setPager(h, { total: 31, offset: 24, hasMore: true });
    h.context.api = async (url) => ({
      json: async () => { if (stage === "body" || stage === "legacy" && url === "/sessions") throw new Error(stage + " failure"); return []; },
      headers: { get() { if (stage === "headers") throw new Error(stage + " failure"); return null; } },
    });
    await h.context.loadSessions();
    assert.equal(h.context.lastItems, before);
    assert.equal(pageState(h).total, 31);
    assert.equal(pageState(h).offset, 24);
    assert.equal(pageState(h).hasMore, true);
    assert.deepEqual(rendered(h).tail, ["sessions refresh failed: " + stage + " failure"]);
  }
});

test("older-page merge keeps loaded records first and retains duplicates within the new page", async () => {
  const h = harness();
  const loaded = Object.freeze([{ id: "open-session", summary: "Loaded" }, { id: "kept", summary: "Existing" }].map(Object.freeze));
  const older = Object.freeze([{ id: "open-session", summary: "Discarded" }, { id: "new", summary: "First duplicate" },
    { id: "new", summary: "Second duplicate" }, { id: "kept", summary: "Discarded again" }, { id: "tail", summary: "Tail" }].map(Object.freeze));
  h.context.lastItems = loaded;
  h.context.renderSessionList(loaded);
  h.elements.sesslist.scrollTop = 143;
  setPager(h, { total: 30, offset: 12, hasMore: true });
  installResponse(h, older, { "x-sessions-total": "30", "x-sessions-page-size": "5" });
  await h.context.loadOlderSessions();
  assert.deepEqual(rendered(h).titles, ["Loaded", "Existing", "First duplicate", "Second duplicate", "Tail"]);
  assert.equal(h.context.lastItems[0], loaded[0]);
  assert.equal(h.context.lastItems[2], older[1]);
  assert.equal(h.context.lastItems[3], older[2]);
  assert.equal(pageState(h).offset, 17);
  assert.equal(rendered(h).scrollTop, 143);
  assert.equal(loaded.length, 2);
  assert.equal(older.length, 5);
});

test("an all-known older page advances pagination without rerendering or replacing the list", async () => {
  const h = harness(), items = h.context.lastItems, row = h.elements.sesslist.children[0];
  setPager(h, { total: 40, offset: 12, hasMore: true });
  h.elements.sesslist.scrollTop = 77;
  installResponse(h, items, { "x-sessions-total": "40", "x-sessions-page-size": "12" });
  await h.context.loadOlderSessions();
  assert.equal(h.context.lastItems, items);
  assert.equal(h.elements.sesslist.children[0], row);
  assert.equal(h.elements.sesslist.scrollTop, 77);
  assert.equal(pageState(h).offset, 24);
  assert.equal(pageState(h).hasMore, true);
});

test("older-page offsets retain addition semantics and exact numeric boundaries", async () => {
  for (const [offset, pageSize, expectedOffset, total, hasMore] of [[0, 12, 12, 12, false], [12, 12, 24, 25, true],
    [-2, 1, -1, 0, true], [2.5, 0.5, 3, 3, false], ["7", 2, "72", 73, true],
    [null, 2, 2, 3, true], [NaN, 2, NaN, 3, false], [Infinity, 2, Infinity, 3, false]]) {
    const h = harness();
    setPager(h, { total, offset, hasMore: true });
    h.context.fetchSessionPage = async (actualOffset, limit) => {
      assert.equal(actualOffset, offset);
      assert.equal(limit, 12);
      return { items: [], total, pageSize };
    };
    await h.context.loadOlderSessions();
    assert.equal(pageState(h).offset, expectedOffset);
    assert.equal(pageState(h).hasMore, hasMore);
  }
});

test("unknown older-page totals retain strict full-page detection", async () => {
  for (const [pageSize, expected] of [[12, true], ["12", false], [11, false], [0, false], [13, false], [NaN, false]]) {
    const h = harness();
    setPager(h, { total: null, offset: 0, hasMore: true });
    h.context.fetchSessionPage = async () => ({ items: [], total: null, pageSize });
    await h.context.loadOlderSessions();
    assert.equal(pageState(h).total, null);
    assert.equal(pageState(h).hasMore, expected);
  }
});

test("older pages preserve an existing numeric total when a response omits it", async () => {
  const h = harness();
  setPager(h, { total: 25, offset: 12, hasMore: true });
  h.context.fetchSessionPage = async () => ({ items: [], pageSize: 12 });
  await h.context.loadOlderSessions();
  assert.equal(pageState(h).total, 25);
  assert.equal(pageState(h).offset, 24);
  assert.equal(pageState(h).hasMore, true);
});

test("a malformed older page reports its native error after committing the existing pager calculations", async () => {
  const h = harness(), items = h.context.lastItems;
  setPager(h, { total: 30, offset: 12, hasMore: true });
  h.context.fetchSessionPage = async () => ({ items: {}, total: 18, pageSize: 6 });
  await h.context.loadOlderSessions();
  assert.equal(h.context.lastItems, items);
  assert.equal(pageState(h).total, 18);
  assert.equal(pageState(h).offset, 18);
  assert.equal(pageState(h).hasMore, false);
  assert.deepEqual(rendered(h).tail, ["sessions refresh failed: page.items.filter is not a function"]);
});

test("HEAD parity: the shared older pager lock blocks a replacement load and finally releases", async () => {
  const h = harness(), pending = [];
  setPager(h, { total: 40, offset: 12, hasMore: true });
  h.context.fetchSessionPage = () => new Promise((resolve) => pending.push(resolve));
  const oldLoad = h.context.loadOlderSessions();
  h.context.switchSession("target-session", false);
  const currentLoad = h.context.loadOlderSessions();
  h.elements.sesslist.scrollTop = 119;
  const items = h.context.lastItems;
  assert.equal(pending.length, 1, "HEAD's shared loading flag prevents a second fetch");
  assert.equal(pageState(h).loading, true);
  pending[0]({ items: [{ id: "old", summary: "Obsolete" }], total: 99, pageSize: 12 });
  await oldLoad; await currentLoad;
  assert.equal(h.context.lastItems, items);
  assert.equal(h.elements.sesslist.scrollTop, 119);
  assert.equal(pageState(h).total, 40);
  assert.equal(pageState(h).offset, 12);
  assert.equal(pageState(h).loading, false);
});

test("a failed older page keeps the last list and releases only its own pager lock", async () => {
  const h = harness(), items = h.context.lastItems;
  setPager(h, { total: 30, offset: 12, hasMore: true });
  h.context.fetchSessionPage = async () => { throw new Error("older failure"); };
  await h.context.loadOlderSessions();
  assert.equal(h.context.lastItems, items);
  assert.equal(pageState(h).offset, 12);
  assert.equal(pageState(h).loading, false);
  assert.deepEqual(rendered(h).tail, ["sessions refresh failed: older failure"]);
});
