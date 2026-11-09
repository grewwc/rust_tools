// Run with: node --test src/bin/ai/serve/app_history_paging.test.cjs
// The transcript is fetched as a tail and grown on demand; this pins the growth
// rule and the scroll compensation without rebuilding the whole page harness.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const page = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const start = page.indexOf("async function loadOlderHistory()");
const end = page.indexOf('if ($("msgs")', start);
assert.ok(start > 0 && end > start, "loadOlderHistory must stay extractable");

test("a grown history fetch remains expanded when rendered", () => {
  const render = page.slice(page.indexOf("async function loadHistory("), start);
  assert.match(render, /const expandedWindow =/);
  assert.match(render, /historyRowsOf\(msgs, histTotal,[^;]*\|\| expandedWindow\)/);
});

function harness({ windowSession = "s", shown = 200, total = 900, loading = 0, pid = "s" } = {}) {
  const root = { scrollTop: 40, scrollHeight: 1000, clientHeight: 400 };
  const calls = [];
  const context = vm.createContext({
    sessionId: pid,
    lastMsgCount: shown,
    canonicalHistoryTotal: total,
    historyLoadState: () => ({ loading }),
    $: () => root,
    loadHistory: async () => {
      calls.push({ limit: root._histWin });
      root.scrollHeight += 1000;
      root.scrollTop = 0;
    },
    Promise,
    Math,
    Number,
  });
  vm.runInContext(page.slice(start, end), context);
  root._histWinSession = windowSession;
  root._histWin = shown;
  root._histOlder = false;
  return { root, calls, context };
}

test("scrolling up grows the fetched tail and keeps the reading position", async () => {
  const { root, calls, context } = harness({ shown: 200, total: 900 });
  await context.loadOlderHistory();
  assert.equal(calls.length, 1);
  // max(shown + 50, ceil(shown * 2.5)) capped by the session total.
  assert.equal(calls[0].limit, 500);
  assert.equal(root.scrollTop, 40 + 1000, "prepended height is compensated");
  assert.equal(root._histOlder, false);
});

test("a fully loaded session, a foreign window and a running load are left alone", async () => {
  for (const setup of [
    { shown: 900, total: 900 },
    { shown: 200, total: 900, windowSession: "other" },
    { shown: 200, total: 900, loading: 7 },
  ]) {
    const { calls, context } = harness(setup);
    await context.loadOlderHistory();
    assert.equal(calls.length, 0);
  }
});

test("repeated scrolls keep growing until the session is fully mounted", async () => {
  const { root, calls, context } = harness({ shown: 200, total: 3000 });
  for (let i = 0; i < 3; i++) {
    await context.loadOlderHistory();
    context.lastMsgCount = root._histWin;
  }
  assert.deepEqual(calls.map((call) => call.limit), [500, 1250, 3000]);
});
