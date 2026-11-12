// Run with: node --test src/bin/ai/serve/app_remote_turn.test.cjs
// A turn owned by another process (a terminal or `/bg` session) has no local
// event stream. `/confirm` reports its owner pid, so the page must report the
// turn as running, keep history follow alive, and refresh when it ends without
// attaching an SSE stream that would immediately finish.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness(confirm) {
  const calls = { arm: 0, history: 0, sessions: 0, attach: 0, cards: [] };
  const context = {
    sessionId: "s1",
    confirmBusy: false,
    confirmPollSeq: 0,
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
    turns: new Map(),
    staleSession: "",
    remoteTurnPid: "",
    remoteTurnAt: 0,
    REMOTE_RUNNING_TTL_MS: 30000,
    RUNNING_SAMPLE_TTL_MS: 60000,
    confirmShown: null,
    NATIVE_PREFIX: "__native__:",
    sideNoteBusy: false,
    api: async () => ({ ok: true, json: async () => confirm }),
    ensureConfirmPoll: () => { },
    armHistoryFollow: () => { calls.arm++; context.staleSession = context.sessionId; },
    attachEvents: () => { calls.attach++; },
    abortEvents: () => { },
    loadHistory: async () => { calls.history++; },
    loadSessions: async () => { calls.sessions++; },
    refreshSideNotes: () => { },
    showConfirm: () => { },
    hideConfirm: () => { },
    handleNativeRequest: () => { },
    addMsg: (role, text, cls) => { calls.cards.push({ role, text, cls }); return {}; },
    holdLayout: (_key, fn) => fn(),
    $: (id) => {
      if (id === "msgs") return { querySelector: () => null };
      return {
        textContent: "", title: "", disabled: false, tabIndex: 0,
        classList: { toggle() { } }, setAttribute() { },
      };
    },
    document: { visibilityState: "visible", querySelectorAll: () => [] },
  };
  vm.createContext(context);
  vm.runInContext(script.slice(script.indexOf("// BEGIN execution model"), script.indexOf("// END execution model")), context);
  const start = script.indexOf("async function pollConfirm()");
  const end = script.indexOf("function note(msg)");
  vm.runInContext(script.slice(start, end), context);
  return { context, calls };
}

test("a remote turn blocks sending and follows history without attaching a stream", async () => {
  const { context, calls } = harness({ pending: null, running: false, turn: 0, remote: "4242" });
  const running = await context.pollConfirm();
  assert.equal(running, true);
  assert.equal(context.remoteTurnPid, "4242");
  assert.equal(context.isRemoteTurn(), true);
  assert.equal(context.isTurnRunning(), true);
  assert.equal(calls.attach, 0, "there is no local event log to attach");
  assert.equal(calls.arm, 1, "history follow must keep the reply visible");
  assert.equal(calls.cards.length, 1);
  assert.match(calls.cards[0].cls, /remote-turn-note/);
});

test("a finished remote turn refreshes history and clears the state", async () => {
  const { context, calls } = harness({ pending: null, running: false, turn: 0, remote: "4242" });
  await context.pollConfirm();
  context.api = async () => ({ ok: true, json: async () => ({ pending: null, running: false, turn: 0, remote: "" }) });
  const running = await context.pollConfirm();
  assert.equal(running, false);
  assert.equal(context.isRemoteTurn(), false);
  // History follow keeps the transcript refreshing until the reply lands;
  // only the remote owner marker must be cleared here.
  assert.equal(context.remoteTurnPid, "");
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
