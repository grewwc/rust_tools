// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_composer_running.test.cjs
// Exercise sparse, healthy streams through the real send and stop handlers.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

test("fresh idle /confirm clears a stop left by a historical active task", async () => {
  const h = sendHarness(), c = h.c;
  loadConfirm(c);
  c.turnRunningAt = 0;
  c.updateExecutionModel({ kind: "history", summaries: [
    { taskStates: [["worker", true]], busy: true, resetTasks: false },
  ] });
  c.commitExecutionView();
  c.syncStopButton();
  assert.equal(c.canStop(), true, "the historical task needs a stop before the server sample");
  c.api = async () => ({ json: async () => ({ pending: null, running: false, turn: 0, remote: null }) });
  await c.pollConfirm();
  const sampled = c.executionState();
  assert.equal(sampled.running, false);
  assert.ok(sampled.idleSample?.at > 0, "an idle poll records explicit server evidence");
  assert.equal(sampled.idleSample.generation, sampled.sampleGeneration);
  assert.equal(c.canStop(), false, "fresh idle evidence overrides the stale task snapshot");
  assert.equal(c.canSend(), true);
  assert.equal(h.elements.send.textContent, "➤");
  assert.equal(c.displayedExecutionState().taskStates.get("worker"), true, "canonical history remains intact");
  const now = c.Date.now();
  c.Date = { now: () => now + c.RUNNING_SAMPLE_TTL_MS };
  assert.equal(c.canStop(), false, "expiry must not resurrect historical active tasks after authoritative idle");
  assert.equal(h.elements.send.textContent, "➤");
  assert.equal(Boolean(h.elements.sesslist.children.find((item) => item.dataset.sessionId === c.sessionId)
    .querySelector(".session-running-badge")), false, "both projections remain idle after expiry");
  await h.dispose();
});

for (const phone of [false, true]) for (const sample of ["positive", "previous-turn idle"]) {
  test(`parked answerless done retains stop after ${sample} sample (phone=${phone})`, async () => {
    const h = sendHarness({ phone }), c = h.c;
    loadConfirm(c);
    const api = c.api;
    let body = { running: false, remote: null, pending: null };
    c.api = (path, ...args) => path.endsWith("/confirm")
      ? Promise.resolve({ json: async () => body }) : api(path, ...args);
    if (sample === "previous-turn idle") {
      await c.pollConfirm();
      assert.ok(c.executionState().idleSample);
    }
    await h.start(); await h.accept();
    assert.equal(c.executionState().idleSample, null, "begin invalidates the previous turn's idle evidence");
    if (sample === "positive") {
      body = { running: true, turn: 42, pending: null };
      await c.pollConfirm();
      assert.ok(c.executionState().runningAt > 0);
    }
    await h.frame("subagent_progress", structuredSubagents([structuredTask("parked-worker", 5000)]));
    await h.finish(() => { h.reader.frame('event: done\ndata: {}\n\n'); h.reader.eof(); });
    const state = h.snapshot();
    assert.equal(state.live, false);
    assert.equal(state.ended, true, JSON.stringify(state));
    assert.equal(c.executionState().running, false, "stream cleanup clears only positive running state");
    assert.equal(c.executionState().idleSample, null);
    assert.equal(c.displayedExecutionState().taskStates.get("parked-worker"), true);
    assert.equal(state.canStop, true);
    assert.equal(state.canSend, false);
    assert.equal(state.send, "⏹");
    assert.equal(state.stopClass, true);
    await h.dispose();
  });
}

for (const phone of [false, true]) for (const sample of [
  "idle while POST pending",
  "pending POST poll delivered after acceptance",
  "idle during accepted stream",
  "live poll delivered after cleanup",
]) test(`parked turn rejects ${sample} (phone=${phone})`, async () => {
  const h = sendHarness({ phone }), c = h.c;
  loadConfirm(c);
  const idle = { running: false, remote: null, pending: null, turn: 0 };
  const api = c.api;
  let respond;
  const delayed = sample.includes("delivered");
  c.api = (path, ...args) => path.endsWith("/confirm")
    ? Promise.resolve({ json: () => delayed
      ? new Promise((resolve) => { respond = resolve; }) : Promise.resolve(idle) })
    : api(path, ...args);
  await h.start();
  if (sample === "idle during accepted stream" || sample === "live poll delivered after cleanup") await h.accept();
  const pending = c.pollConfirm();
  await settle();
  if (!delayed) {
    assert.equal(await pending, false);
    assert.equal(c.executionState().idleSample, null, "a live owner cannot supply durable idle evidence");
  }
  if (sample === "idle while POST pending" || sample === "pending POST poll delivered after acceptance") await h.accept();
  if (sample === "pending POST poll delivered after acceptance") {
    respond(idle);
    assert.equal(await pending, null, "HTTP acceptance retires queries made before the turn was accepted");
    assert.equal(c.executionState().idleSample, null);
  }
  await h.frame("subagent_progress", structuredSubagents([structuredTask("parked-worker", 5000)]));
  await h.finish(() => { h.reader.frame('event: done\ndata: {}\n\n'); h.reader.eof(); });
  if (sample === "live poll delivered after cleanup") {
    respond(idle);
    assert.equal(await pending, null, "a poll from the retired live owner cannot settle its parked tasks");
  }
  assert.equal(c.executionState().idleSample, null);
  assert.equal(c.displayedExecutionState().taskStates.get("parked-worker"), true);
  assert.equal(c.canStop(), true);
  assert.equal(c.canSend(), false);
  assert.equal(h.elements.send.textContent, "⏹");

  // A new idle query after cleanup can still retire the stale displayed task.
  c.api = (path, ...args) => path.endsWith("/confirm")
    ? Promise.resolve({ json: async () => idle }) : api(path, ...args);
  assert.equal(await c.pollConfirm(), false);
  assert.ok(c.executionState().idleSample);
  assert.equal(c.canStop(), false);
  assert.equal(c.canSend(), true);
  assert.equal(h.elements.send.textContent, "➤");
  assert.equal(c.displayedExecutionState().taskStates.get("parked-worker"), true);
  await h.dispose();
});

test("HTTP acceptance for a replaced live entry cannot retire its replacement's samples", async () => {
  const h = sendHarness(), c = h.c;
  await h.start();
  const original = h.entry();
  c.updateExecutionModel({ kind: "live", entry: { box: c.document.createElement("div") } });
  const generation = c.executionState().sampleGeneration;
  c.updateExecutionModel({ kind: "accepted", entry: original });
  assert.equal(c.executionState().sampleGeneration, generation);
  await h.accept();
  assert.equal(c.executionState().sampleGeneration, generation, "the late POST belongs to the replaced entry");
  await h.finish(() => { h.reader.frame('event: done\ndata: {}\n\n'); h.reader.eof(); });
  await h.dispose();
});

for (const event of [
  { kind: "begin" },
  { kind: "select" },
  { kind: "field", name: "turnRunning", value: true },
  { kind: "field", name: "turnRunningPid", value: 42 },
  { kind: "field", name: "remoteTurnPid", value: "42" },
  { kind: "remote", active: true, pid: "42", at: 1000 },
]) test(`idle evidence expires on ${event.name || event.kind} observations`, async () => {
  const h = sendHarness(), c = h.c;
  loadConfirm(c);
  c.api = async () => ({ json: async () => ({ running: false, pending: null }) });
  await c.pollConfirm();
  assert.ok(c.executionState().idleSample);
  c.updateExecutionModel(event);
  assert.equal(c.executionState().idleSample, null);

  // A response started under the old owner cannot restore its idle evidence.
  c.updateExecutionModel({ kind: "select" });
  let respond;
  c.api = () => new Promise((resolve) => { respond = resolve; });
  const pending = c.pollConfirm();
  c.updateExecutionModel(event);
  respond({ json: async () => ({ running: false, pending: null }) });
  assert.equal(await pending, null);
  assert.equal(c.executionState().idleSample, null);
  await h.dispose();
});

function loadFixtures(file, names) {
  const filename = path.join(__dirname, file);
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : require(name),
    __dirname, __filename: filename, console, Buffer, process,
    TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
    setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
  });
}
const { sendHarness, records, settle } = loadFixtures("app_phase2_baseline.test.cjs", ["sendHarness", "records", "settle"]);
const { script, structuredTask, structuredSubagents, descendants } = loadFixtures("app_stream.test.cjs", ["script", "structuredTask", "structuredSubagents", "descendants"]);
const { streamFramerHarness } = loadFixtures("app_phase3_stream_framer.test.cjs", ["streamFramerHarness"]);

for (const status of ["running", "ready", "starting", "processing output"]) {
  test(`new active ${status} report supersedes idle evidence and an older idle request`, async () => {
    const h = sendHarness(), c = h.c;
    loadConfirm(c);
    const idle = { running: false, pending: null, remote: null };
    c.api = async () => ({ json: async () => idle });
    await c.pollConfirm();
    assert.ok(c.executionState().idleSample);
    let respond;
    c.api = () => new Promise((resolve) => { respond = resolve; });
    const pending = c.pollConfirm();
    const owner = c.updateExecutionModel({ kind: "activity" });
    const report = structuredSubagents([structuredTask("fresh-worker", 0, { status, progress: { phase: "starting" } })]).task_report;
    c.updateExecutionModel({ kind: "task", owner, slot: "subagents", summary: c.projectTaskReport(report) });
    assert.equal(c.executionState().idleSample, null, "live task evidence retires the older idle observation");
    assert.equal(h.elements.send.textContent, "⏹");
    assert.equal(c.canSend(), false);
    assert.equal(report.tasks[0].status, status, "the producer's detailed status stays intact");
    respond({ json: async () => idle });
    assert.equal(await pending, null, "the request predates the active task");
    assert.equal(c.canStop(), true);
    await h.dispose();
  });
}

test("idle confirmation cadence discovers a turn on the first two-second tick", async () => {
  const h = sendHarness(), c = h.c;
  let reads = 0;
  c.isTurnRunning = () => false;
  c.pollConfirm = async () => { reads++; return true; };
  c.ensureConfirmPoll();
  const timer = [...h.timers.values()].find((item) => item.type === "interval" && item.delay === 2000);
  assert.ok(timer);
  await timer.fn();
  assert.equal(reads, 1, "idle sessions must not skip the first four ticks");
  h.document.visibilityState = "hidden";
  await timer.fn();
  assert.equal(reads, 1, "hidden tabs still do not poll");
  await h.dispose();
});

test("confirmation samples update the session badge without waiting for the bulk poll", async () => {
  const h = sendHarness(), c = h.c;
  loadConfirm(c);
  c.lastItems = c.lastItems.map((item) => ({ ...item, activity: "idle" }));
  c.renderSessionList(c.lastItems);
  const row = h.elements.sesslist.children.find((item) => item.dataset.sessionId === c.sessionId);
  let body = { running: true, turn: 42, pending: null, remote: null };
  c.api = async () => ({ json: async () => body });
  await c.pollConfirm();
  assert.equal(row.querySelector(".session-running-label")?.textContent, "RUNNING");
  body = { running: false, pending: null, remote: null };
  await c.pollConfirm();
  assert.equal(row.querySelector(".session-running-badge"), null);
  await h.dispose();
});

function assertRunningAgreement(h, expected, checkpoint) {
  const c = h.c;
  const row = h.elements.sesslist.children.find((item) => item.dataset.sessionId === c.sessionId);
  assert.equal(Boolean(row.querySelector(".session-running-badge")), expected, checkpoint + ": sidebar badge");
  assert.equal(c.isTurnRunning(), expected, checkpoint + ": shared running verdict");
  assert.equal(c.canStop(), expected, checkpoint + ": composer stop permission");
  assert.equal(h.elements.send.textContent, expected ? "⏹" : "➤", checkpoint + ": composer glyph");
  assert.equal(h.elements.send.classList.contains("stop"), expected, checkpoint + ": composer class");
}
function agreementHarness() {
  const h = sendHarness();
  loadConfirm(h.c);
  h.c.attachEvents = () => {};
  h.c.lastItems = h.c.lastItems.map((item) => ({ ...item, activity: "idle" }));
  h.c.renderSessionList(h.c.lastItems);
  h.elements.prompt.value = "follow-up draft";
  h.c.syncStopButton(); // mount controls before exercising automatic updates
  return h;
}

function hiddenGroupAgreementHarness(snapshot) {
  const h = agreementHarness(), c = h.c;
  for (const id of ["sessiongrouphidden", "sessiongrouphiddenactivity"]) {
    h.elements[id] = h.document.createElement("button");
  }
  c.hiddenGroupSnapshot = snapshot;
  vm.runInContext(`
    sessionGroupsBase = store.base;
    sessionGroups = [{ id: "hidden", name: "Hidden work", order: 0, hidden: true }];
    lastItems = lastItems.map(item => ({ ...item, group_id: item.id === sessionId ? "hidden" : null }));
    sessionGroupTemporaryOpen.set(sessionGroupTemporaryKey("hidden"), false);
    sessionGroupActivitySnapshots.set(store.base, {
      at: Date.now(), status: "complete", groups: new Map([["hidden", hiddenGroupSnapshot]])
    });
    renderSessionList(lastItems);
  `, c);
  return h;
}

function assertHiddenGroupAgreement(h, running, waiting, checkpoint) {
  const header = vm.runInContext('sessionGroupHeaders.get("hidden")', h.c);
  const activity = h.c.hiddenSessionGroupActivity("hidden");
  assert.equal(activity.running, running, checkpoint + ": hidden group running count");
  assert.equal(activity.waiting, waiting, checkpoint + ": hidden group waiting count");
  assert.equal(header.running.textContent, running ? String(running) : "", checkpoint + ": group badge count");
  assert.equal(header.running.hidden, !running, checkpoint + ": group running badge visibility");
  assert.equal(header.waiting.hidden, !waiting, checkpoint + ": group waiting badge visibility");
  assert.equal(header.status.hidden, !running && !waiting, checkpoint + ": group status visibility");
  assert.equal(h.elements.sessiongrouphiddenactivity.hidden, !running && !waiting, checkpoint + ": hidden activity shortcut");
}

test("unified running: complete idle hidden snapshot merges later loaded activity on every control", async () => {
  const snapshot = { running: 0, waiting: 0, unknown: 0, active: [] };
  const h = hiddenGroupAgreementHarness(snapshot), c = h.c;
  assertRunningAgreement(h, false, "initial idle");
  assertHiddenGroupAgreement(h, 0, 0, "initial idle");
  c.mergeSessionActivities([{ id: c.sessionId, activity: "running" }]);
  assertRunningAgreement(h, true, "loaded session starts after snapshot");
  assertHiddenGroupAgreement(h, 1, 0, "loaded session starts after snapshot");
  c.renderSessionList(c.lastItems);
  assertRunningAgreement(h, true, "list repaint");
  assertHiddenGroupAgreement(h, 1, 0, "list repaint");
  c.mergeSessionActivities([{ id: c.sessionId, activity: "running", waiting: "input", waiting_prompt: "Answer?" }]);
  assertRunningAgreement(h, true, "loaded session waits");
  assertHiddenGroupAgreement(h, 0, 1, "loaded session waits");
  c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
  assertRunningAgreement(h, false, "loaded session finishes");
  assertHiddenGroupAgreement(h, 0, 0, "loaded session finishes");
  assert.deepEqual(snapshot, { running: 0, waiting: 0, unknown: 0, active: [] }, "the source snapshot stays intact");
  await h.dispose();
});

test("unified running: hidden snapshot retains unloaded activity while replacing loaded entries", async () => {
  const snapshot = { running: 2, waiting: 1, unknown: 1, active: [
    { id: "open-session", activity: "running", waiting: null },
    { id: "unloaded-running", activity: "running", waiting: null },
    { id: "unloaded-waiting", activity: "running", waiting: "confirm", waiting_prompt: "Proceed?" },
  ] };
  const original = JSON.stringify(snapshot);
  const h = hiddenGroupAgreementHarness(snapshot), c = h.c;
  c.mergeSessionActivities([{ id: c.sessionId, activity: "running" }]);
  assertRunningAgreement(h, true, "loaded entry counted once");
  assertHiddenGroupAgreement(h, 2, 1, "loaded entry counted once");
  c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
  assertRunningAgreement(h, false, "loaded entry settles");
  assertHiddenGroupAgreement(h, 1, 1, "unloaded entries retained");
  const merged = c.hiddenSessionGroupActivity("hidden");
  assert.deepEqual(Array.from(merged.active, item => item.id), ["unloaded-running", "unloaded-waiting"]);
  assert.equal(merged.unknown, 1, "unidentified snapshot unknowns remain intact");
  assert.equal(merged.unavailable, true);
  assert.equal(JSON.stringify(snapshot), original, "the source snapshot stays intact");
  await h.dispose();
});

test("unified running: held terminal lease without PID stays running across bulk and confirm polls", async () => {
  const h = agreementHarness(), c = h.c;
  c.api = async () => ({ json: async () => ({ pending: null, running: false, turn: 0, remote: true }) });
  for (let tick = 0; tick < 3; ++tick) {
    c.mergeSessionActivities([{ id: c.sessionId, activity: "running" }]);
    assertRunningAgreement(h, true, "held lease bulk tick " + tick);
    assert.equal(await c.pollConfirm(), true);
    assertRunningAgreement(h, true, "held lease confirm tick " + tick);
    assert.equal(c.isRemoteTurn(), true);
    assert.equal(c.executionState().remotePid, "", "PID remains optional diagnostic data");
  }
  c.api = async () => ({ json: async () => ({ pending: null, running: false, turn: 0, remote: null }) });
  await c.pollConfirm();
  assertRunningAgreement(h, false, "released lease confirm");
  c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
  assertRunningAgreement(h, false, "released lease bulk");
  await h.dispose();
});

for (const status of ["running", "ready", "starting", "processing output"]) {
  test(`unified running: fresh bulk idle settles active ${status} on both controls`, async () => {
    const h = agreementHarness(), c = h.c;
    const owner = c.updateExecutionModel({ kind: "activity" });
    const report = structuredSubagents([structuredTask("worker", 0, { status, progress: { phase: "starting" } })]).task_report;
    c.updateExecutionModel({ kind: "task", owner, slot: "subagents", summary: c.projectTaskReport(report) });
    assertRunningAgreement(h, true, "live task requires no poll tick");
    c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
    assertRunningAgreement(h, false, "new authoritative bulk idle");
    assert.equal(c.displayedExecutionState().taskStates.get("worker"), true, "task detail remains available");
    assert.equal(report.tasks[0].status, status);
    const now = c.Date.now();
    c.Date = { now: () => now + 7500 };
    c.expireSessionActivities();
    assertRunningAgreement(h, false, "expiry cannot restore historical activity");
    await h.dispose();
  });
}

for (const remote of [null, "4242"]) {
  test(`unified running: delayed ${remote ? "terminal" : "serve"} running confirm cannot undo newer bulk idle`, async () => {
    const h = agreementHarness(), c = h.c;
    let respond;
    c.api = () => new Promise((resolve) => { respond = resolve; });
    const pending = c.pollConfirm();
    c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
    assertRunningAgreement(h, false, "new bulk idle before confirm response");
    respond({ json: async () => ({ running: !remote, turn: remote ? 0 : 42, remote, pending: null }) });
    assert.equal(await pending, null, "the common gate rejects the older response before transport writes");
    assertRunningAgreement(h, false, "older running response rejected");
    assert.equal(c.executionState().remoteActive, false);
    assert.equal(c.executionState().running, false);
    await h.dispose();
  });
}

test("unified running: terminal idle overrides active historical tasks without deleting their status", async () => {
  const h = agreementHarness(), c = h.c;
  c.updateExecutionModel({ kind: "history", summaries: [
    { taskStates: [["historical-worker", true]], busy: true, resetTasks: false },
  ] });
  c.commitExecutionView();
  c.api = async () => ({ json: async () => ({ running: false, remote: "4242", pending: null }) });
  await c.pollConfirm();
  assertRunningAgreement(h, true, "fresh terminal lease");
  assert.equal(c.isRemoteTurn(), true);
  c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
  assertRunningAgreement(h, false, "latest bulk idle retires terminal and historical task evidence");
  assert.equal(c.isRemoteTurn(), false);
  c.commitExecutionView();
  c.renderSessionList(c.lastItems);
  assertRunningAgreement(h, false, "history repaint cannot promote active history");
  assert.equal(c.displayedExecutionState().taskStates.get("historical-worker"), true);
  await h.dispose();
});

for (const remote of [null, "4242"]) test(`unified running: ${remote ? "terminal" : "serve"} sample expires on both controls at 7500ms`, async () => {
  const h = agreementHarness(), c = h.c;
  c.api = async () => ({ json: async () => ({ running: !remote, turn: remote ? 0 : 42, remote, pending: null }) });
  await c.pollConfirm();
  assertRunningAgreement(h, true, "fresh sample");
  const now = c.Date.now();
  c.Date = { now: () => now + 7499 };
  c.expireSessionActivities();
  assertRunningAgreement(h, true, "one millisecond before common expiry");
  c.Date = { now: () => now + 7500 };
  c.expireSessionActivities();
  assertRunningAgreement(h, false, "common expiry boundary");
  await h.dispose();
});

test("unified running: selection and history rebuild cannot revive settled worker status", async () => {
  const h = agreementHarness(), c = h.c;
  const history = [{ taskStates: [["settled-worker", true]], busy: true, resetTasks: false }];
  c.updateExecutionModel({ kind: "history", summaries: history });
  c.commitExecutionView();
  c.mergeSessionActivities([{ id: c.sessionId, activity: "idle" }]);
  assertRunningAgreement(h, false, "latest idle");
  c.updateExecutionModel({ kind: "select" });
  c.updateExecutionModel({ kind: "history", summaries: history });
  c.commitExecutionView();
  c.renderSessionList(c.lastItems);
  assertRunningAgreement(h, false, "selection preserves the settled observation");
  const owner = c.updateExecutionModel({ kind: "activity" });
  c.updateExecutionModel({ kind: "task", owner, slot: "subagents",
    summary: { taskStates: [["new-worker", true]], busy: true, resetTasks: true } });
  assertRunningAgreement(h, true, "a later live producer can still start activity");
  await h.dispose();
});

test("unified running: clock-driven composer refresh repaints the expired badge without a bulk poll", async () => {
  const h = agreementHarness(), c = h.c;
  c.api = async () => ({ json: async () => ({ running: true, turn: 42, pending: null }) });
  await c.pollConfirm();
  assertRunningAgreement(h, true, "fresh serve sample");
  const now = c.Date.now();
  c.Date = { now: () => now + 7500 };
  c.syncStopButton();
  assertRunningAgreement(h, false, "the same clock boundary repaints both controls");
  await h.dispose();
});

test("unified running: persisted selection binds before the sidebar observation maps initialize", () => {
  const context = vm.createContext({ sessionId: "opened-at-boot", store: { base: "https://server.test" },
    sessionActivityReady: undefined, Date, paintSessionActivity() { assert.fail("sidebar is not initialized"); },
    refreshComposerControls() {},
  });
  const observerFrom = script.indexOf("function observeSessionActivityExecution(");
  const observerTo = script.indexOf("function refreshSessionActivity(", observerFrom);
  vm.runInContext(script.slice(observerFrom, observerTo), context);
  const from = script.indexOf("// BEGIN execution model"), to = script.indexOf("// END execution model", from);
  vm.runInContext(script.slice(from, to), context);
  assert.equal(context.executionState().sessionId, "opened-at-boot");
  assert.equal(context.executionState().live, false);
});

test("unified running: first idle two-second tick updates both controls and hidden pages pause", async () => {
  const h = agreementHarness(), c = h.c;
  let reads = 0;
  c.api = async () => { reads++; return { json: async () => ({ running: true, turn: 42, remote: null, pending: null }) }; };
  c.ensureConfirmPoll();
  const timer = [...h.timers.values()].find((item) => item.type === "interval" && item.delay === 2000);
  assert.ok(timer);
  await timer.fn();
  assert.equal(reads, 1);
  assertRunningAgreement(h, true, "first two-second idle tick");
  h.document.visibilityState = "hidden";
  await timer.fn();
  assert.equal(reads, 1, "hidden pages do not sample");
  h.document.visibilityState = "visible";
  c.api = async () => { reads++; return { json: async () => ({ running: false, remote: null, pending: null }) }; };
  await timer.fn();
  assert.equal(reads, 2);
  assertRunningAgreement(h, false, "first visible idle response");
  await h.dispose();
});

function loadConfirm(c) {
  const from = script.indexOf("async function pollConfirm()");
  const to = script.indexOf("// Live streams, replay streams", from);
  assert.ok(from >= 0 && to > from);
  vm.runInContext(script.slice(from, to), c);
  c.confirmBusy = false;
  c.armHistoryFollow = () => {};
  c.refreshSideNotes = () => {};
}

function assertStop(h, checkpoint) {
  const state = h.snapshot();
  assert.equal(state.send, "⏹", checkpoint + ": running turn renders stop");
  assert.equal(state.stopClass, true, checkpoint + ": stop class agrees with glyph");
  assert.equal(h.elements.send.getAttribute("aria-label"), "stop this turn");
  assert.equal(state.canStop, true);
  assert.equal(state.canSend, false);
  assert.equal(state.live, true);
  assert.equal(state.ended, false);
  assert.equal(state.failed, false);
}

for (const width of [390, 1280]) for (const subagent of [false, true]) for (const terminal of ["done", "error"]) {
  test("healthy sparse replay: " + width + "px, " + (subagent ? "active subagent" : "main turn only") + ", " + terminal,
    { timeout: 5000 }, async () => {
      const h = streamFramerHarness("replay", { phone: width <= 700 }), c = h.c;
      const controls = sendHarness({ phone: width <= 700 }).elements, lookup = c.$;
      controls.prompt.value = "follow-up draft";
      c.$ = (id) => ["send", "sidenote", "composer", "prompt"].includes(id) ? controls[id] : lookup(id);
      c.promptComposing = false; c.pendingImages = []; c.MAX_LIVE_TURNS = 4;
      c.holdLayout = (_, apply) => apply();
      c.window.innerWidth = width;
      c.window.matchMedia = (query) => ({
        matches: query.includes("max-width") ? width <= 700 : query.includes("min-width") ? width >= 701 : false,
        addEventListener() {},
      });
      const from = script.indexOf("function isRemoteTurn()"), to = script.indexOf("// Transient note through", from);
      vm.runInContext(script.slice(from, to), c);
      await h.start(); await h.accept();
      await h.chunk('event: thinking_done\ndata: {}\n\n');
      if (subagent) await h.chunk("event: subagent_progress\ndata: " + JSON.stringify(structuredSubagents([
        structuredTask("sparse-worker", 2000, { status: "ready", progress: { phase: "starting" } }),
      ])) + "\n\n");
      for (let step = 0; step < 5; step++) {
        h.tick(15000);
        await h.chunk(": keepalive\n\n");
      }
      h.tick(4000);
      // Run the actual phase interval after its server sample has expired.
      for (const timer of [...h.timers.values()]) if (timer.type === "interval" && timer.delay === 1000) timer.fn();
      const phase = descendants(h.view().box).find((node) => node.classList.contains("pending") && !node.hidden);
      assert.match(phase.textContent, /Processing….*1:19/);
      assert.equal(c.executionState().live, false, "replay uses its own stream owner");
      assert.equal(c.executionState().attached, true);
      c.syncStopButton();
      assert.equal(controls.send.textContent, "⏹", "Processing at 1:19: running replay renders stop");
      assert.equal(controls.send.classList.contains("stop"), true);
      assert.equal(c.canSend(), false);
      if (terminal === "done") await h.chunk('event: replace\ndata: {"text":"Final answer."}\n\n');
      await h.chunk("event: " + terminal + "\ndata: " + JSON.stringify(terminal === "error" ? { error: "confirmed failure" } : {}) + "\n\n");
      await h.finish();
      assert.equal(controls.send.textContent, "➤", "terminal replay returns to send");
      assert.equal(c.canSend(), true);
      await h.dispose();
    });
}

for (const width of [390, 1280]) for (const subagent of [false, true]) for (const terminal of ["done", "error"]) {
  const name = "healthy sparse stream: " + width + "px, " + (subagent ? "active subagent" : "main turn only") + ", " + terminal;
  test(name, { timeout: 5000 }, async () => {
    const h = sendHarness({ phone: width <= 700 }), c = h.c;
    let now = 1000;
    c.Date = { now: () => now };
    c.window.innerWidth = width;
    c.window.matchMedia = (query) => ({
      matches: query.includes("max-width") ? width <= 700 : query.includes("min-width") ? width >= 701 : false,
      addEventListener() {},
    });
    await h.start();
    assertStop(h, "headers pending");
    await h.accept();
    await h.frame("thinking_done", {});
    if (subagent) {
      const report = structuredSubagents([structuredTask("sparse-worker", 2000, {
        status: "ready", description: "Fix mobile formula clipping", progress: {
          phase: "starting", stale_for_ms: 2930, latest_activity: "starting",
        },
      })], "task_wait still running. Reason parked waited 0s");
      await h.frame("tool", { id: "wait", name: "task_wait", phase: "finished", ok: true,
        output: report.text, task_report: { ...report.task_report, operation: "task_wait", snapshot: false } });
      assert.deepEqual(JSON.parse(JSON.stringify(h.snapshot().tasks)), [["sparse-worker", true]]);
      assert.ok(records(h.liveBox()).some((node) => node.dataset.subagents === "true"));
    }
    // Heartbeats keep the transport healthy without adding any execution event.
    for (let elapsed = 15000; elapsed <= 75000; elapsed += 15000) {
      now = 1000 + elapsed;
      h.reader.frame(": keepalive\n\n");
      await settle();
    }
    now = 80000;
    for (const timer of [...h.timers.values()]) if (timer.type === "interval" && timer.delay === 1000) timer.fn();
    const phase = descendants(h.liveBox()).find((node) => node.classList.contains("pending") && !node.hidden);
    assert.ok(phase, "the running phase remains mounted");
    assert.match(phase.textContent, /Processing….*1:19/, "the real phase timer advances during sparse progress");
    assertStop(h, "Processing at 1:19");

    const interrupts = [], fetch = c.fetch;
    c.fetch = async (url, options = {}) => {
      if (new URL(url).pathname.endsWith("/interrupt")) {
        interrupts.push({ path: new URL(url).pathname, method: options.method });
        return { ok: true, json: async () => ({ interrupted: true }) };
      }
      return fetch(url, options);
    };
    const from = script.indexOf("async function stopTurn()"), to = script.indexOf("function parseBlock(raw)", from);
    assert.ok(from >= 0 && to > from);
    vm.runInContext(script.slice(from, to), c);
    h.elements.send.onclick();
    await settle();
    assert.deepEqual(interrupts, [{ path: "/sessions/open-session/interrupt", method: "POST" }], "stop clicks reach the interrupt endpoint");
    assertStop(h, "interrupt acknowledgement before terminal event");

    if (terminal === "done") await h.frame("replace", { text: "Final answer." });
    await h.finish(() => {
      h.reader.frame("event: " + terminal + "\ndata: " + JSON.stringify(terminal === "error" ? { error: "confirmed failure" } : {}) + "\n\n");
      h.reader.eof();
    });
    assert.equal(h.snapshot().send, "➤", "terminal evidence returns the composer to send");
    assert.equal(h.snapshot().stopClass, false);
    assert.equal(h.snapshot().canSend, true);
    assert.equal(h.snapshot().failed, terminal === "error");
    assert.equal(descendants(h.liveBox()).some((node) => node.classList.contains("pending") && !node.hidden), false);
    await h.dispose();
  });
}
