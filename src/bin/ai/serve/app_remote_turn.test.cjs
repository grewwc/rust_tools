// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_remote_turn.test.cjs
// Remote owner notices use current observations and the activity sample TTL.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

const fixtureFile = path.join(__dirname, "app_mark.test.cjs");
const { harness: sidebarHarness, response, deferred, settle } = new vm.Script(
  fs.readFileSync(fixtureFile, "utf8") + "\n;({ harness, response, deferred, settle });",
  { filename: fixtureFile },
).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: fixtureFile, AbortController, URL, setImmediate,
});
function between(start, end) {
  const from = script.indexOf(start), to = script.indexOf(end, from);
  assert.ok(from >= 0 && to > from, "missing page boundary: " + start);
  return script.slice(from, to);
}
const remoteSample = (remote = "4242") => ({ pending: null, running: false, turn: 0, remote });
const remoteText = (pid = "") => "正在另一个进程中运行" + (pid ? "（pid " + pid + "）" : "")
  + "；本轮结束后结果会自动出现在这里。";

function harness(confirm, { realCadence = false, legacyActivity = false } = {}) {
  const h = sidebarHarness({
    sid: "s1", realNavigation: true,
    items: [{ id: "s1", summary: "Session", ...(legacyActivity ? {} : { activity: "idle" }) }],
  });
  const { context, elements, document } = h;
  const calls = { arm: 0, history: 0, sessions: 0, attach: 0, confirm: [] };
  for (const id of ["send", "sidenote", "composer", "prompt"]) {
    const node = elements[id] = document.createElement(id === "composer" ? "div" : id === "prompt" ? "textarea" : "button");
    node.classList.toggle = (name, active) => active ? node.classList.add(name) : node.classList.remove(name);
  }
  elements.prompt.value = "follow-up draft";
  Object.assign(context, {
    pendingImages: [], promptComposing: false, MAX_LIVE_TURNS: 4,
    confirmBusy: false,
    confirmPollSeq: 0,
    confirmPollTimer: null,
    runEpoch: 0,
    turnRunning: false,
    turnRunningAt: 0,
    turnRunningPid: 0,
    endedTurnPid: 0,
    eventStreamEnded: false,
    eventStreamFailed: false,
    eventFailures: 0,
    eventActive: false,
    eventSession: "",
    staleSession: "",
    remoteTurnPid: "",
    remoteTurnAt: 0,
    RUNNING_SAMPLE_TTL_MS: 60000,
    confirmShown: null,
    NATIVE_PREFIX: "__native__:",
    sideNoteBusy: false,
    ensureConfirmPoll: () => { },
    armHistoryFollow: () => { calls.arm++; context.staleSession = context.sessionId; },
    attachEvents: () => { calls.attach++; },
    abortEvents: () => { },
    loadSessions: async () => { calls.sessions++; },
    refreshSideNotes: () => { },
    showConfirm: () => { },
    hideConfirm: () => { },
    handleNativeRequest: () => { },
    holdLayout: (_key, fn) => fn(),
  });
  const loadHistory = context.loadHistory, fetch = context.fetch;
  context.loadHistory = async (...args) => { calls.history++; return loadHistory(...args); };
  context.fetch = async (url, options = {}) => {
    if (!new URL(url).pathname.endsWith("/confirm")) return fetch(url, options);
    calls.confirm.push(context.Date.now());
    return typeof confirm === "function" ? confirm() : response(confirm);
  };
  for (const [start, end] of [
    ["// BEGIN remote turn note", "// END remote turn note"],
    ["async function pollConfirm()", "// Live streams, replay streams"],
    ["function isRemoteTurn()", "// Transient note through"],
    ["async function stopTurn()", "function parseBlock(raw)"],
    ["async function send()", "// END composer send"],
  ]) vm.runInContext(between(start, end), context);
  if (realCadence) {
    const intervals = new Map();
    let nextInterval = 0;
    context.setInterval = (fn, delay) => {
      const id = ++nextInterval;
      const tick = () => {
        if (!intervals.has(id)) return;
        fn();
        if (intervals.has(id)) intervals.set(id, context.setTimeout(tick, delay));
      };
      intervals.set(id, context.setTimeout(tick, delay));
      return id;
    };
    context.clearInterval = (id) => { context.clearTimeout(intervals.get(id)); intervals.delete(id); };
    vm.runInContext(between("function ensureConfirmPoll()", "// A pick is the moment"), context);
  }
  document.visibilityState = "visible";
  return {
    ...h, calls, setConfirm: (value) => { confirm = value; },
    notices: () => Array.from(elements.msgs.querySelectorAll(".remote-turn-note")),
    ttl: vm.runInContext("SESSION_ACTIVITY_MAX_AGE_MS", context),
    async advanceCadence(ms) {
      const target = context.Date.now() + ms;
      while (true) {
        const next = Math.min(...[...h.timers.values()].map((timer) => timer.at));
        if (next > target) break;
        h.advance(Math.max(0, next - context.Date.now()));
        await settle();
      }
      h.advance(target - context.Date.now());
      await settle();
    },
  };
}

test("a remote turn blocks sending and follows history without attaching a stream", async () => {
  const h = harness(remoteSample()), { context, calls } = h;
  const running = await context.pollConfirm();
  assert.equal(running, true);
  assert.equal(context.remoteTurnPid, "4242");
  assert.equal(context.isRemoteTurn(), true);
  assert.equal(context.isTurnRunning(), true);
  assert.equal(calls.attach, 0, "there is no local event log to attach");
  assert.equal(calls.arm, 1, "history follow must keep the reply visible");
  assert.equal(h.notices().length, 1);
  const node = h.notices()[0];
  assert.equal(node.textContent, "正在另一个进程中运行（pid 4242）；本轮结束后结果会自动出现在这里。");
  assert.equal(node.className, "msg asst pending remote-turn-note");
  assert.equal(node.getAttribute("role"), "note");
  assert.equal(node.getAttribute("aria-label"), "其他进程中的运行状态");
  assert.equal(node.getAttribute("aria-live"), "off");
  assert.equal(h.elements.send.getAttribute("aria-label"), "running in another process");
  assert.equal(h.elements.send.disabled, true);
});

test("a remote owner without a supplied pid omits the entire parenthetical", async () => {
  const h = harness(remoteSample(true));
  assert.equal(await h.context.pollConfirm(), true);
  assert.equal(h.context.remoteTurnPid, "");
  assert.equal(h.context.isRemoteTurn(), true);
  assert.equal(h.notices().length, 1);
  assert.equal(h.notices()[0].textContent, "正在另一个进程中运行；本轮结束后结果会自动出现在这里。");
  h.advance(h.ttl);
  assert.equal(h.notices().length, 0);
});

for (const remote of ["", null, undefined]) test("a finished remote turn clears its notice (owner=" + remote + ")", async () => {
  const h = harness(remoteSample()), { context, calls } = h;
  await context.pollConfirm();
  h.setConfirm(remoteSample(remote));
  if (remote === undefined) h.setConfirm({ pending: null, running: false, turn: 0 });
  const running = await context.pollConfirm();
  assert.equal(running, false);
  assert.equal(context.isRemoteTurn(), false);
  // History follow keeps the transcript refreshing until the reply lands;
  // only the remote owner marker must be cleared here.
  assert.equal(context.remoteTurnPid, "");
  assert.equal(h.notices().length, 0);
  assert.equal(calls.history, 1);
  assert.equal(calls.sessions, 1);
});

test("a locally served turn still attaches its event stream", async () => {
  const { context, calls } = harness({ pending: null, running: true, turn: 99, remote: "" });
  const running = await context.pollConfirm();
  assert.equal(running, true);
  assert.equal(context.isRemoteTurn(), false);
  assert.equal(calls.attach, 1);
});

test("an authoritative idle activity sample clears the owner and history cannot restore it", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  assert.equal(h.notices()[0].textContent, remoteText("4242"));
  c.mergeSessionActivities([{ id: "s1", activity: "idle" }]);
  assert.equal(h.notices().length, 0);
  assert.equal(c.isRemoteTurn(), false);
  c.mergeSessionActivities([{ id: "s1", activity: "running" }]);
  await c.loadHistory({ sessionId: "s1", messages: [{ role: "assistant", content: "Completed answer." }], histTotal: 1 });
  c.renderSessionList(c.lastItems);
  c.syncStopButton();
  assert.equal(h.notices().length, 0, "history and generic running cannot synthesize an owner notice");
  assert.equal(h.elements.msgs.textContent, "Completed answer.");
});

test("a local running confirmation takes over even if it includes the old remote owner", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  h.setConfirm({ ...remoteSample(), running: true, turn: 99 });
  await c.pollConfirm();
  assert.equal(h.notices().length, 0);
  assert.equal(c.remoteTurnPid, "");
  assert.equal(c.isRemoteTurn(), false);
  assert.equal(h.calls.attach, 1);
  h.setConfirm({ ...remoteSample(""), running: false });
  await c.pollConfirm();
  assert.equal(h.notices().length, 0);
});

test("a local stream clears the remote sample before and after replay", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  c.updateExecutionModel({ kind: "live", entry: {} });
  assert.equal(h.notices().length, 0);
  assert.equal(c.isRemoteTurn(), false);
  c.updateExecutionModel({ kind: "live", entry: null });
  c.updateExecutionModel({ kind: "history", summaries: [] });
  c.syncStopButton();
  assert.equal(h.notices().length, 0);
});

test("repeated confirms, send and stop reuse one muted notice and update the supplied pid", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  const node = h.notices()[0];
  for (let i = 0; i < 3; i++) {
    await c.pollConfirm();
    await c.send();
    await c.stopTurn();
    c.updateExecutionModel({ kind: "history", summaries: [] });
    c.syncStopButton();
    assert.deepEqual(h.notices(), [node]);
    assert.equal(node.textContent, remoteText("4242"));
  }
  h.setConfirm(remoteSample("96706"));
  await c.pollConfirm();
  assert.deepEqual(h.notices(), [node]);
  assert.equal(node.textContent, "正在另一个进程中运行（pid 96706）；本轮结束后结果会自动出现在这里。");
  h.setConfirm(remoteSample(true));
  await c.pollConfirm();
  assert.deepEqual(h.notices(), [node]);
  assert.equal(node.textContent, remoteText());
});

test("history rerender retires the notice and only a fresh confirm can recreate it", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  const node = h.notices()[0];
  const history = { sessionId: "s1", messages: [{ role: "assistant", content: "Persisted answer." }], histTotal: 1 };
  await c.loadHistory(history);
  assert.equal(node.parentNode, null);
  assert.equal(h.notices().length, 0);
  for (let i = 0; i < 2; i++) { await c.loadHistory(history); c.syncStopButton(); }
  assert.equal(h.notices().length, 0, "replay cannot synthesize a notice from retained remote state");
  await c.pollConfirm();
  assert.equal(h.notices().length, 1);
  assert.equal(h.notices()[0].textContent, remoteText("4242"));
  h.setConfirm(remoteSample(""));
  await c.pollConfirm();
  await c.loadHistory(history);
  c.syncStopButton();
  assert.equal(h.notices().length, 0, "a cleared notice stays cleared after history replay");
});

test("a remote notice expires at the activity freshness boundary without another response", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  const node = h.notices()[0];
  h.advance(h.ttl - 1);
  assert.deepEqual(h.notices(), [node]);
  assert.equal(node.textContent, remoteText("4242"));
  h.advance(1);
  assert.equal(h.notices().length, 0);
  assert.equal(c.isRemoteTurn(), false);
  c.syncStopButton();
  c.updateExecutionModel({ kind: "history", summaries: [] });
  assert.equal(h.notices().length, 0);
});

test("a fresh confirm renews expiry without the old timer removing the current notice", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  const node = h.notices()[0];
  h.advance(5000);
  await c.pollConfirm();
  h.advance(h.ttl - 5000);
  assert.deepEqual(h.notices(), [node]);
  assert.equal(node.textContent, remoteText("4242"));
  h.advance(4999);
  assert.deepEqual(h.notices(), [node]);
  h.advance(1);
  assert.equal(h.notices().length, 0);
});

test("an older remote confirm cannot resurrect a notice cleared by a newer activity sample", async () => {
  const h = harness(remoteSample()), c = h.context;
  await c.pollConfirm();
  const slow = deferred();
  h.setConfirm(() => slow.promise);
  const pending = c.pollConfirm();
  await settle();
  c.mergeSessionActivities([{ id: "s1", activity: "idle" }]);
  assert.equal(h.notices().length, 0);
  slow.resolve(response(remoteSample()));
  assert.equal(await pending, null);
  c.syncStopButton();
  assert.equal(h.notices().length, 0);
  h.setConfirm(remoteSample("96706"));
  await c.pollConfirm();
  assert.equal(h.notices().length, 1, "a genuinely newer owner can start another turn");
  assert.equal(h.notices()[0].textContent, remoteText("96706"));
});

test("unknown activity and session or server changes clear the visible remote owner", async () => {
  for (const transition of ["unknown", "session", "server"]) {
    const h = harness(remoteSample()), c = h.context;
    await c.pollConfirm();
    if (transition === "unknown") c.mergeSessionActivities([{ id: "s1", activity: "unknown" }]);
    if (transition === "session") c.switchSession("s2", false);
    if (transition === "server") { c.store.base = "https://other.test"; c.sessionActivityScope(); }
    assert.equal(h.notices().length, 0, transition);
    h.advance(h.ttl);
    c.syncStopButton();
    assert.equal(h.notices().length, 0, transition + " cannot be undone by the old expiry timer");
  }
});

for (const activityPolling of [false, true]) test("an 8s-delayed initial confirm cannot override newer two-second idle samples (activity polling=" + activityPolling + ")", async () => {
  const slow = deferred();
  let reads = 0;
  const h = harness(() => ++reads === 1 ? slow.promise : response(remoteSample(null)),
    { realCadence: true, legacyActivity: true }), c = h.context;
  const startedAt = c.Date.now();
  const revision = c.snapshotSessionActivityRevisions().get("s1");
  const pending = c.pollConfirm();
  if (activityPolling) c.wireSessionActivityPoll();
  await h.advanceCadence(8000);
  const earlyRequests = [startedAt, ...[2000, 4000, 6000, 8000].map((ms) => startedAt + ms)];
  assert.deepEqual(h.calls.confirm, earlyRequests, "idle ticks sample every two seconds");
  assert.equal(h.notices().length, 0);
  if (activityPolling) assert.notEqual(c.snapshotSessionActivityRevisions().get("s1"), revision,
    "expiry must invalidate the old request even when a legacy sample was already unknown");
  slow.resolve(response(remoteSample("30904")));
  assert.equal(await pending, null, "request age cannot become fresh on receipt");
  assert.equal(h.notices().length, 0);
  assert.equal(c.remoteTurnPid, "");
  h.setConfirm(remoteSample("96706"));
  await h.advanceCadence(2000);
  assert.deepEqual(h.calls.confirm, [...earlyRequests, startedAt + 10000]);
  assert.equal(h.notices().length, 1);
  assert.equal(h.notices()[0].textContent, "正在另一个进程中运行（pid 96706）；本轮结束后结果会自动出现在这里。");
  c.stopConfirmPoll();
});

test("activity expiry invalidates a younger in-flight remote confirm", async () => {
  const slow = deferred();
  const h = harness(() => slow.promise, { legacyActivity: true }), c = h.context;
  h.advance(h.ttl - 1);
  const revision = c.snapshotSessionActivityRevisions().get("s1");
  const pending = c.pollConfirm();
  await settle();
  h.advance(1);
  c.expireSessionActivities();
  assert.notEqual(c.snapshotSessionActivityRevisions().get("s1"), revision);
  slow.resolve(response(remoteSample("30904")));
  assert.equal(await pending, null, "expiry invalidates a response that is only one millisecond old");
  assert.equal(h.notices().length, 0);
  h.setConfirm(remoteSample("96706"));
  await c.pollConfirm();
  assert.equal(h.notices().length, 1);
  assert.equal(h.notices()[0].textContent, remoteText("96706"));
});

test("a remote sample's remaining freshness starts before its fetch", async () => {
  const slow = deferred();
  const h = harness(() => slow.promise), c = h.context, startedAt = c.Date.now();
  const pending = c.pollConfirm();
  h.advance(2500);
  slow.resolve(response(remoteSample()));
  await pending;
  assert.equal(c.remoteTurnAt, startedAt);
  assert.equal(h.notices().length, 1);
  assert.equal(h.notices()[0].textContent, remoteText("4242"));
  h.advance(h.ttl - 2500 - 1);
  assert.equal(h.notices().length, 1);
  h.advance(1);
  assert.equal(h.notices().length, 0, "receipt time does not extend the ownership sample");
});
