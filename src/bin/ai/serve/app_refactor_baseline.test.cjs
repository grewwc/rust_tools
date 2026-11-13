// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_refactor_baseline.test.cjs
// Characterize current behavior, including DOM-order and editor-refresh quirks.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function loadFixtures(file, names) {
  const filename = path.join(__dirname, file);
  // Reuse the existing helpers without registering their test cases twice.
  // Fixture assertions and the product functions they load still execute normally.
  return new vm.Script(fs.readFileSync(filename, "utf8") +
    "\n;({" + names.join(",") + "});", { filename }).runInNewContext({
    require: (name) => name === "node:test" ? { test() {} } : require(name),
    __dirname, __filename: filename, console, process, Buffer,
    TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
    setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
  });
}

const stream = loadFixtures("app_stream.test.cjs", [
  "streamHarness", "replayHarness", "retainedHistoryHarness", "runningComposer",
  "descendants", "nodeList", "structuredTask", "structuredSubagents", "rateNodes", "script",
]);
const desktop = loadFixtures("app_desktop.test.cjs", ["harness"]);
const copy = loadFixtures("app_copy.test.cjs", ["harness"]);
const phase2 = loadFixtures("app_phase2_baseline.test.cjs", ["sendHarness"]);
const toolArguments = JSON.stringify({ path: "README.md" });
const trace = [
  ["thinking", { text: "Inspect README." }],
  ["thinking_done", {}],
  ["tool", { id: "read-1", name: "read_file", phase: "started", arguments: toolArguments }],
  ["tool", { id: "read-1", name: "read_file", phase: "finished", ok: true, output: "Fixture contents." }],
  ["delta", { delta: "Answer." }],
];
const historyMessages = [
  { role: "assistant", content: "", reasoning_content: "Inspect README.", tool_calls: [
    { id: "read-1", type: "function", function: { name: "read_file", arguments: toolArguments } },
  ] },
  { role: "tool", tool_call_id: "read-1", name: "read_file", content: "Fixture contents." },
  { role: "assistant", content: "Answer." },
];

function driveTrace(harness) {
  for (const [kind, value] of trace) harness.emit(kind, value);
}

function installTranscriptQuery(document, root) {
  // The desktop fixture returns no task nodes and the stream fixture has no
  // document query API. Resolve selectors against their actual DOM tree instead
  // of returning a prearranged card array to the execution predicate.
  document.querySelectorAll = (selector) => {
    const selectors = selector.split(",").map((part) => {
      const scoped = /^#msgs\s+(.+)$/.exec(part.trim());
      assert.ok(scoped, "the fixture query is scoped to the transcript");
      return scoped[1];
    });
    return stream.nodeList(Array.from(stream.descendants(root)).filter((node) => selectors.some((part) => {
      const attribute = /^\[data-([\w-]+)\]$/.exec(part);
      if (!attribute) return node.matches?.(part) === true;
      const key = attribute[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      return Object.hasOwn(node.dataset || {}, key);
    })));
  };
}

function taskCard(context, root, id, active) {
  const before = new Set(Array.from(stream.descendants(root)));
  const activity = context.createLiveActivity(root, null);
  activity.subagents(JSON.stringify(stream.structuredSubagents([
    stream.structuredTask(id, 5000, { status: active ? "running" : "completed", active, delivered: !active }),
  ])));
  const card = Array.from(stream.descendants(root)).find((node) => !before.has(node)
    && node.dataset?.taskStates !== undefined);
  assert.ok(card, "the real activity renderer creates task metadata");
  assert.deepEqual(JSON.parse(card.dataset.taskStates), [[id, active]]);
  assert.equal(card.dataset.taskReset, "true");
  return card;
}

test("task snapshot gates keep the Phase 0 natural-order result through any card placement", () => {
  const h = stream.streamHarness({ realThinking: true });
  driveTrace(h);
  h.emit("done", {});
  const c = h.context, root = h.box, composer = stream.runningComposer(c);
  installTranscriptQuery(c.document, root);
  Object.assign(c, {
    turns: new Map(), eventActive: false, eventSession: "", staleSession: "",
    turnRunning: false, turnRunningAt: 0, eventStreamEnded: true, eventStreamFailed: false,
  });
  const running = taskCard(c, root, "worker-a", true);
  const completed = taskCard(c, root, "worker-b", false);
  const metadata = JSON.stringify([running.dataset, completed.dataset]);
  const ledger = JSON.stringify([...c.executionState().taskStates]);
  c.document.querySelectorAll = () => assert.fail("execution gates cannot query the rendered task cards");
  const send = { running: false, glyph: "➤", title: "send" };
  const gate = () => {
    c.syncStopButton();
    assert.equal(JSON.stringify([running.dataset, completed.dataset]), metadata);
    assert.equal(JSON.stringify([...c.executionState().taskStates]), ledger);
    assert.equal(c.canSend(), true);
    assert.equal(c.canStop(), false);
    return { running: c.isTurnRunning(), glyph: composer.send.textContent, title: composer.send.title };
  };

  // The real producer order is running then completed: the later snapshot
  // replaces the earlier task state, including when their IDs differ.
  root.insertBefore(running, null);
  root.insertBefore(completed, null);
  assert.deepEqual(gate(), send);
  root.insertBefore(completed, null);
  root.insertBefore(running, null);
  assert.deepEqual(gate(), send);
  assert.equal(c.canSend(), true);
  assert.equal(c.canStop(), false);

  const group = Array.from(stream.descendants(root)).find((node) => node.classList?.contains("execution-items"));
  assert.ok(group, "the trace created a real completed-execution group");
  group.appendChild(running);
  assert.equal(running.parentNode, group);
  assert.ok(Array.from(stream.descendants(root)).includes(running), "grouped cards remain inside the scanned transcript");
  // Regrouping is a presentation change; it cannot replace the producer order.
  assert.deepEqual(gate(), send);
  root.insertBefore(running, null);
  assert.equal(running.parentNode, root);
  assert.deepEqual(gate(), send);
  // Exercise both task permutations at every existing transcript boundary.
  for (const before of root.children.slice()) {
    root.insertBefore(running, before);
    root.insertBefore(completed, before);
    assert.deepEqual(gate(), send);
    root.insertBefore(completed, before);
    root.insertBefore(running, before);
    assert.deepEqual(gate(), send);
  }
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function loadSessionSwitch(c) {
  Object.assign(c, {
    window: { ...c.window, matchMedia: () => ({ matches: false }) },
    localStorage: { setItem() {}, removeItem() {} },
    persistDraft() {}, restoreDraft() {}, resetQuestionNavigator() {},
    restoreSkillPick() {}, restoreTeamPick() {},
  });
  const start = stream.script.indexOf("// BEGIN session switch");
  const end = stream.script.indexOf("// END session switch");
  assert.ok(start > 0 && end > start);
  vm.runInContext(stream.script.slice(start, end), c);
}

function allowHistoryFailure(c) {
  const errors = [];
  c.console = { error(...args) { errors.push(args); } };
  c.addMsg = (role, text, cls = "asst err", parent = c.$("msgs")) => {
    const row = c.document.createElement("div");
    row.className = "msg " + cls; row.textContent = text;
    return parent.appendChild(row);
  };
  return errors;
}

function historySnapshot(harness) {
  const c = harness.c;
  return JSON.stringify({
    sessionId: c.sessionId, historySession: c.historySession, historyKeys: c.historyKeys,
    canonicalHistoryTotal: c.canonicalHistoryTotal, lastMsgCount: c.lastMsgCount,
    historyLoadSeq: c.historyLoadSeq, text: harness.root.textContent,
    nodes: Array.from(stream.descendants(harness.root)).map((node) => ({
      className: node.className, text: node.textContent, dataset: node.dataset,
    })),
  });
}

test("a late session A history body leaves session B transcript and mounted state unchanged", async () => {
  const h = stream.retainedHistoryHarness([]), c = h.c;
  const bodyA = deferred(), readingA = deferred(), requests = [];
  const messagesA = [{ role: "assistant", content: "Late answer from A." }];
  const messagesB = [{ role: "assistant", content: "Visible answer from B." }];
  c.api = async (url) => {
    requests.push(url);
    const forA = url.startsWith("/sessions/s/history?");
    assert.ok(forA || url.startsWith("/sessions/session-b/history?"));
    return {
      headers: { get: () => String(forA ? messagesA.length : messagesB.length) },
      json: () => {
        if (forA) { readingA.resolve(); return bodyA.promise; }
        return Promise.resolve(messagesB);
      },
    };
  };
  const pendingA = c.loadHistory();
  await readingA.promise;
  // This fixture exposes selected-session state, rather than the navigation UI.
  // Both the old response and B's replacement still use the real history loader.
  c.sessionId = "session-b";
  await c.loadHistory();
  assert.equal(h.root.textContent, "Visible answer from B.");
  assert.equal(c.historySession, "session-b");
  const before = historySnapshot(h), children = Array.from(h.root.children);
  bodyA.resolve(messagesA);
  await pendingA;
  assert.equal(historySnapshot(h), before);
  assert.equal(h.root.children.length, children.length);
  children.forEach((node, index) => assert.equal(h.root.children[index], node));
  assert.equal(requests.length, 2);
  assert.doesNotMatch(h.root.textContent, /Late answer from A/);
});

test("session navigation retires the previous transcript gate before replacement history commits", async () => {
  const report = stream.structuredSubagents([stream.structuredTask("worker-a", 5000)]).task_report;
  const h = stream.retainedHistoryHarness([
    { role: "user", content: "Start worker A." },
    { role: "tool", tool_call_id: "spawn-1", name: "task_spawn", content: "worker started", task_report: report },
  ]), c = h.c;
  Object.assign(c, {
    turns: new Map(), turnRunning: false, turnRunningAt: 0,
    remoteTurnPid: "", remoteTurnAt: 0, eventActive: false, eventSession: "",
    staleSession: "", eventStreamEnded: true, eventStreamFailed: false,
  });
  const composer = stream.runningComposer(c);
  await c.loadHistory();
  c.syncStopButton();
  assert.equal(c.displayedExecutionState().sessionId, "s");
  assert.equal(composer.send.title, "stop this turn");

  loadSessionSwitch(c);
  c.switchSession("session-b", false);
  c.stopStaleFollow();

  const body = deferred();
  c.api = async () => ({ headers: { get: () => "1" }, json: () => body.promise });
  const pending = c.loadHistory();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(c.sessionId, "session-b");
  assert.equal(c.displayedExecutionState().sessionId, "session-b");
  assert.equal(c.isTurnRunning(), false);
  assert.equal(composer.send.title, "send");
  assert.equal(h.root.children.length, 0, "A's rows cannot remain visible during B's fetch");

  body.resolve([{ role: "assistant", content: "Visible B answer." }]);
  await pending;
  assert.equal(c.displayedExecutionState().sessionId, "session-b");
  assert.equal(c.isTurnRunning(), false);
  assert.equal(composer.send.title, "send");

  // A background turn can finish while its original transcript is detached.
  // Its removal must refresh the composer without waiting for drawer requests.
  const live = phase2.sendHarness();
  await live.start(); await live.accept();
  await live.frame("subagent_progress", stream.structuredSubagents([stream.structuredTask("background-parked", 5000)]));
  live.c.switchSession("session-b", false);
  live.c.stopStaleFollow();
  live.c.loadInfo = () => {};
  live.c.loadSessions = async () => {};
  await live.frame("done", {});
  await live.finish(() => {});
  assert.equal(live.liveBox().parentNode, null);
  assert.equal(live.c.displayedExecutionState().sources.size, 0);
  assert.equal(live.c.canSend(), true);
  assert.equal(live.elements.send.title, "send");
  await live.dispose();
});

test("displayed sources: an ended parked task without a live entry becomes idle after ABA and unchanged history", async () => {
  const h = stream.retainedHistoryHarness([{ role: "assistant", content: "Earlier A answer." }]), c = h.c;
  const composer = stream.runningComposer(c);
  loadSessionSwitch(c);
  c.turnRunning = false;
  await c.loadHistory();
  const box = c.document.createElement("div");
  box.className = "turn"; h.root.appendChild(box);
  const activity = c.createLiveActivity(box, null);
  activity.subagents(stream.structuredSubagents([stream.structuredTask("parked-a", 5000)]));
  activity.finish("ended");
  c.syncStopButton();
  assert.equal(composer.send.title, "stop this turn");

  c.switchSession("session-b", false);
  c.switchSession("s", false);
  await c.loadHistory();
  c.stopStaleFollow();
  c.syncStopButton();
  assert.equal(box.parentNode, null, "navigation retires an unregistered parked producer from the transcript");
  assert.equal(box.querySelectorAll(".task-output").length, 1);
  assert.equal(c.executionState().activities.size, 0, "selection retired the session activity and its mount");
  // HEAD 18e9fbc3: arbitrate.cjs minimal head exits 0 with send/true/false.
  // The original baseline exits 1 on HEAD at this title assertion (actual send).
  // Evidence: /tmp/app-phase2-aba-fix-mea4vtqy/before/arbitration-runs.json.
  assert.equal(composer.send.title, "send");
  assert.equal(c.canSend(), true);
  assert.equal(c.canStop(), false);
  assert.equal(c.displayedExecutionState().sources.size, 1, "only the selected canonical history is mounted");
  assert.equal(JSON.stringify([...c.displayedExecutionState().taskStates]), JSON.stringify([]));
  activity.subagents(stream.structuredSubagents([stream.structuredTask("parked-a", 5000)]));
  // HEAD ignores task/finish observations once select removed the activity.
  // Evidence: /tmp/phase2-independent-final-ocu_yehx/minimal-retired-update.cjs.
  assert.equal(c.canStop(), false, "a same-turn report cannot reactivate an activity absent from the registry");
  assert.equal(composer.send.title, "send");
  activity.subagents(stream.structuredSubagents([stream.structuredTask("parked-a", 5000, { status: "completed", active: false })]));
  assert.equal(c.canSend(), true, "rendering the detached card does not restore its retired model observation");
  activity.remove(); box.remove();
  assert.equal(c.displayedExecutionState().sources.size, 1, "removing the detached activity cannot unregister canonical history");
});

for (const live of [false, true]) test("displayed sources: ABA retires parked observations only when live entry is absent (live=" + live + ")", async () => {
  const h = stream.retainedHistoryHarness([{ role: "assistant", content: "Earlier A answer." }]), c = h.c;
  const composer = stream.runningComposer(c);
  loadSessionSwitch(c); c.turnRunning = false;
  await c.loadHistory();
  const canonical = [...h.root.children], box = c.document.createElement("div");
  box.className = "turn"; h.root.appendChild(box);
  const activity = c.createLiveActivity(box, null);
  activity.subagents(stream.structuredSubagents([stream.structuredTask("parked-control", 5000)]));
  activity.finish("ended");
  if (live) c.turns.set("s", { box });
  c.switchSession("session-b", false); c.switchSession("s", false);
  assert.equal(composer.send.title, live ? "stop this turn" : "send", "the selected live owner alone retains its running gate");
  await c.loadHistory(); c.turns.delete("s"); c.stopStaleFollow(); c.syncStopButton();
  assert.equal(composer.send.title, live ? "stop this turn" : "send");
  assert.equal(c.canSend(), !live);
  assert.equal(c.canStop(), live);
  assert.equal(box.parentNode, live ? h.root : null);
  assert.equal(box.querySelectorAll(".task-output").length, 1);
  assert.equal(h.historyCalls.length, 2);
  canonical.forEach((node, index) => {
    assert.notEqual(h.root.children[index], node, "navigation rebuilds the selected canonical rows");
    assert.equal(h.root.children[index].textContent, node.textContent);
  });
  activity.remove(); box.remove();
});

test("displayed sources: a new same-turn report is retired again on the next ABA without a live entry", async () => {
  const h = stream.retainedHistoryHarness([{ role: "assistant", content: "Earlier A answer." }]), c = h.c;
  const composer = stream.runningComposer(c);
  loadSessionSwitch(c); c.turnRunning = false;
  await c.loadHistory();
  const box = c.document.createElement("div"); box.className = "turn"; h.root.appendChild(box);
  const activity = c.createLiveActivity(box, null);
  activity.subagents(stream.structuredSubagents([stream.structuredTask("first-observation", 5000)]));
  activity.finish("ended");
  for (const id of ["fresh-observation", "next-observation"]) {
    c.switchSession("session-b", false); c.switchSession("s", false);
    await c.loadHistory(); c.stopStaleFollow(); c.syncStopButton();
    assert.equal(composer.send.title, "send");
    assert.equal(c.canSend(), true);
    assert.equal(c.executionState().activities.size, 0);
    activity.subagents(stream.structuredSubagents([stream.structuredTask(id, 5000)]));
    assert.equal(composer.send.title, "send");
    assert.equal(JSON.stringify([...c.displayedExecutionState().taskStates]), JSON.stringify([]));
  }
  activity.remove(); box.remove();
});

for (const kind of ["task", "finish"]) test("displayed sources: retired same-turn " + kind + " cannot mutate model slots or summaries", async () => {
  const h = stream.retainedHistoryHarness([{ role: "assistant", content: "Earlier A answer." }]), c = h.c;
  const composer = stream.runningComposer(c);
  loadSessionSwitch(c); c.turnRunning = false;
  await c.loadHistory();
  const box = c.document.createElement("div"); box.className = "turn"; h.root.appendChild(box);
  let owner;
  const update = c.updateExecutionModel;
  c.updateExecutionModel = (event, ...args) => {
    const value = update(event, ...args);
    if (event.kind === "activity") owner = value;
    return value;
  };
  const activity = c.createLiveActivity(box, null);
  c.updateExecutionModel = update;
  activity.subagents(stream.structuredSubagents([stream.structuredTask("before-retirement", 5000)]));
  activity.finish("ended");
  c.switchSession("session-b", false); c.switchSession("s", false);
  await c.loadHistory(); c.stopStaleFollow(); c.syncStopButton();
  const fingerprint = () => JSON.stringify({
    slots: [...owner.source.slots], generation: owner.source.activityGeneration,
    ended: c.executionState().ended, failed: c.executionState().failed,
    tasks: [...c.executionState().taskStates], displayed: [...c.displayedExecutionState().taskStates],
    busy: c.displayedExecutionState().taskBusy,
  });
  const before = fingerprint();
  if (kind === "task") activity.subagents(stream.structuredSubagents([stream.structuredTask("after-retirement", 5000)]));
  else activity.finish("failed", true);
  assert.equal(fingerprint(), before, "HEAD returns before changing a retired activity's observations");
  assert.equal(composer.send.title, "send");
  assert.equal(composer.send.textContent, "➤");
  assert.equal(c.canSend(), true);
  assert.equal(c.canStop(), false);
  assert.equal(box.parentNode, null, "a retired producer remains detached across navigation");
  const card = Array.from(stream.descendants(box)).find((node) => node.dataset?.subagents === "true");
  assert.ok(card, "the existing card still renders");
  if (kind === "task") assert.equal(card.dataset.taskStates, JSON.stringify([["after-retirement", true]]));
  else assert.equal(card.dataset.taskBusy, "false");
  const sourceCount = c.displayedExecutionState().sources.size;
  activity.remove(); box.remove();
  assert.equal(c.displayedExecutionState().sources.size, sourceCount, "removing a detached source preserves the selected transcript");
});

test("displayed sources: a late running report from an older turn keeps the completed current turn idle", async () => {
  const h = stream.retainedHistoryHarness([{ role: "assistant", content: "Committed history." }]), c = h.c;
  const composer = stream.runningComposer(c);
  c.turnRunning = false;
  await c.loadHistory();
  const createActivity = () => {
    const box = c.document.createElement("div");
    box.className = "turn"; h.root.appendChild(box);
    return { box, activity: c.createLiveActivity(box, null) };
  };
  c.updateExecutionModel({ kind: "begin" });
  const old = createActivity();
  old.activity.subagents(stream.structuredSubagents([stream.structuredTask("old-task", 5000,
    { status: "completed", active: false, delivered: true })]));
  old.activity.finish("ended");
  c.syncStopButton();
  assert.equal(composer.send.title, "send");

  c.updateExecutionModel({ kind: "begin" });
  const current = createActivity();
  current.activity.finish("ended");
  c.syncStopButton();
  assert.equal(composer.send.title, "send");
  const state = c.executionState(), displayed = c.displayedExecutionState();
  const before = JSON.stringify({ turn: state.turn, ended: state.ended, failed: state.failed,
    tasks: [...displayed.taskStates], busy: displayed.taskBusy });

  old.activity.subagents(stream.structuredSubagents([stream.structuredTask("old-task", 5000)]));
  c.syncStopButton();
  assert.equal(c.canStop(), false, "a superseded turn cannot reactivate the idle gate");
  assert.equal(c.canSend(), true);
  assert.equal(composer.send.title, "send");
  assert.equal(old.box.parentNode, h.root, "the old producer remains mounted");
  assert.equal(current.box.parentNode, h.root);
  const nextState = c.executionState(), nextDisplayed = c.displayedExecutionState();
  assert.equal(JSON.stringify({ turn: nextState.turn, ended: nextState.ended, failed: nextState.failed,
    tasks: [...nextDisplayed.taskStates], busy: nextDisplayed.taskBusy }), before);
});

test("HEAD parity: a same-session rebuild failure retains the displayed active task snapshot", async () => {
  const report = stream.structuredSubagents([stream.structuredTask("old-active", 5000)]).task_report;
  const h = stream.retainedHistoryHarness([
    { role: "user", content: "Start the old task." },
    { role: "tool", name: "task_spawn", content: "old task started", task_report: report },
  ]), c = h.c, composer = stream.runningComposer(c);
  c.turnRunning = false;
  await c.loadHistory();
  c.stopStaleFollow();
  assert.equal(composer.send.title, "stop this turn");
  const errors = allowHistoryFailure(c);
  c.renderHistoryRow = () => { throw new Error("controlled same-session render failure"); };
  await c.loadHistory({ sessionId: "s", messages: [{ role: "assistant", content: "Replacement answer." }], histTotal: 1 });
  assert.equal(errors.length, 1);
  assert.match(h.root.textContent, /controlled same-session render failure/);
  assert.equal(h.root.querySelectorAll(".task-output").length, 0);
  assert.equal(c.historySession, "");
  assert.equal(composer.send.title, "stop this turn");
  assert.equal(c.canSend(), false);
  assert.equal(c.canStop(), true);
  assert.equal(c.displayedExecutionState().sources.size, 0);
  assert.equal(c.displayedExecutionState().taskBusy, true);
  assert.equal(JSON.stringify([...c.displayedExecutionState().taskStates]), JSON.stringify([["old-active", true]]));
});

test("navigation permits an ABA replay retry only in A's own transcript", async () => {
  const h = stream.retainedHistoryHarness([]), c = h.c;
  const composer = stream.runningComposer(c);
  loadSessionSwitch(c);
  allowHistoryFailure(c);
  const firstReplay = c.attachEvents();
  await h.settle();
  h.fire(25000);
  await firstReplay;
  assert.ok([...h.timers.values()].some((timer) => timer.ms === 2000), "A scheduled its replay retry");
  const queuedRetry = [...h.timers.values()].find((timer) => timer.ms === 2000);

  c.switchSession("session-b", false);
  const report = stream.structuredSubagents([stream.structuredTask("mounted-b", 5000)]).task_report;
  await c.loadHistory({ sessionId: "session-b", messages: [
    { role: "user", content: "Run B's task." },
    { role: "tool", name: "task_spawn", content: "B task started", task_report: report },
  ], histTotal: 2 });
  c.stopStaleFollow();
  const canonicalB = [...h.root.children];
  assert.equal(composer.send.title, "stop this turn");

  c.switchSession("s", false);
  const body = deferred();
  c.api = async () => body.promise;
  const pendingHistory = c.loadHistory();
  await h.settle();
  c.turnRunning = true; c.turnRunningAt = c.Date.now();
  const attach = c.attachEvents;
  let retryAttachment;
  c.attachEvents = () => { retryAttachment = attach(); return retryAttachment; };
  queuedRetry.fn();
  await h.settle();
  assert.ok(c.recoveredView, "HEAD's sid/epoch retry accepts the return to A");
  const currentReplay = retryAttachment;
  assert.ok(currentReplay);
  const retryBox = c.recoveredView.box;
  assert.equal(retryBox.parentNode, h.root, "A's current replay mounts in A's selected transcript");
  for (const row of canonicalB) assert.equal(row.parentNode, null, "B's rows are detached before A's history completes");
  body.reject(new Error("controlled A history failure"));
  await pendingHistory;
  assert.equal(retryBox.parentNode, h.root, "the history failure reattaches A's replay box");

  c.turnRunning = false;
  c.staleSession = "";
  h.fire(25000);
  await currentReplay;
  for (let i = 0; i < 4; i++) await h.settle();
  c.pauseHistoryRetry("s");
  assert.equal(c.eventActive, false);
  assert.equal(c.recoveredView, null);
  assert.equal(retryBox.parentNode, null);
  for (const row of canonicalB) assert.equal(stream.descendants(h.root).includes(row), false);
  assert.equal(h.root.querySelectorAll(".task-output").length, 0, "B's task observations cannot remain in A's transcript");
  c.syncStopButton();
  assert.equal(composer.send.title, "send");
  assert.equal(c.canSend(), true);
  assert.equal(c.canStop(), false);
  assert.equal(c.displayedExecutionState().sessionId, "s", "reattachment keeps the selected transcript owner");
  assert.equal(c.displayedExecutionState().sources.size, 0, "the replay removes only its own source");
  assert.equal(JSON.stringify([...c.displayedExecutionState().taskStates]), JSON.stringify([]));
});

test("quirk: a successful session-list refresh detaches the active rename editor", async () => {
  const h = desktop.harness();
  const inputApi = copy.harness().document.createElement("input");
  const createElement = h.document.createElement;
  // Reuse the existing copy fixture's selection API; the desktop fixture lacks select().
  h.document.createElement = (tag) => {
    const node = createElement(tag);
    if (tag === "input") {
      node.select = inputApi.select;
      node.setSelectionRange = inputApi.setSelectionRange;
    }
    return node;
  };
  const row = h.row("target-session");
  const session = h.context.lastItems.find((item) => item.id === "target-session");
  h.context.startRename(row, session);
  const input = row.querySelector("input");
  assert.equal(input.value, "Target session");
  assert.equal(h.document.activeElement, input);
  assert.equal(input.selectionStart, 0);
  assert.equal(input.selectionEnd, input.value.length);
  input.value = "Edited title";
  input.setSelectionRange(2, 6);
  const editing = { value: input.value, start: input.selectionStart, end: input.selectionEnd };

  // Drive the reachable refresh flow, which calls renderSessionList after its GET.
  await h.context.loadSessions();
  assert.equal(row.parentNode, null);
  assert.notEqual(h.row("target-session"), row);
  assert.equal(h.elements.sesslist.querySelectorAll("input").length, 0);
  assert.equal(h.title("target-session").textContent, "Target session");
  assert.deepEqual({ value: input.value, start: input.selectionStart, end: input.selectionEnd }, editing);
  assert.deepEqual(Array.from(h.requests, (request) => ({ path: request.path, method: request.method })), [
    { path: "/sessions", method: "GET" },
  ]);
  // Detached-node focus pointers and native blur delivery are not modeled here;
  // the observable assertion is editor removal, not a browser focus destination.
});

function answerText(node) {
  if (typeof node.innerHTML === "string" && node.innerHTML) return node.innerHTML;
  let text = node.textContent;
  for (const rate of Array.from(stream.rateNodes(node))) {
    if (text.endsWith(rate.textContent)) text = text.slice(0, -rate.textContent.length);
  }
  return text;
}

function transcriptProjection(root) {
  const projection = [];
  for (const node of Array.from(stream.descendants(root))) {
    if (!node.classList?.contains("msg") || node.hidden
      || node.classList.contains("execution-group") || node.classList.contains("pending")) continue;
    if (node.summary) {
      const thinking = node.dataset.thinkingPhase === "true" || /^thinking(?: ·|$)/.test(node.summary.textContent);
      // Normalize grouping, rate labels, icons and the live argument preview into
      // logical records. The raw argument-preview difference is pinned separately.
      projection.push({
        role: thinking ? "thinking" : "tool",
        label: thinking ? "thinking" : node.summary.textContent.replace(/^[◌!✓•]\s*/, ""),
        text: thinking ? node.body.textContent : node.body.textContent.replace(/^Arguments\n[^\n]*\n\n/, ""),
        state: node.dataset.executionState, error: node.dataset.executionException === "true",
      });
    } else {
      const text = answerText(node);
      if (text) projection.push({ role: "assistant", text });
    }
  }
  return projection;
}

test("history, replay and live project the same completed trace; tool argument previews differ today", async () => {
  const live = stream.streamHarness({ realThinking: true });
  driveTrace(live);
  live.emit("done", {});
  assert.equal(live.context.finished, true);
  const replay = stream.replayHarness({ realThinking: true });
  driveTrace(replay);
  // This is the same completion hook called by attachEvents on its done frame.
  replay.view.finish();
  const history = stream.retainedHistoryHarness(historyMessages);
  await history.c.loadHistory();
  assert.equal(history.c.canonicalHistoryTotal, historyMessages.length);
  const expected = [
    { role: "thinking", label: "thinking", text: "Inspect README.", state: "completed", error: false },
    { role: "tool", label: "read_file README.md", text: "Fixture contents.", state: "completed", error: false },
    { role: "assistant", text: "Answer." },
  ];
  const roots = [live.box, replay.view.box, history.root];
  for (const root of roots) {
    assert.deepEqual(transcriptProjection(root), expected);
    assert.ok(Array.from(stream.descendants(root)).some((node) => node.dataset?.taskReset === "true"),
      "each completed path exposes the terminal answer marker");
  }
  const toolBody = (root) => Array.from(stream.descendants(root))
    .find((node) => node.summary?.textContent.includes("read_file"))?.body.textContent;
  assert.equal(toolBody(live.box), "Arguments\n" + toolArguments + "\n\nFixture contents.");
  assert.equal(toolBody(replay.view.box), toolBody(live.box));
  assert.equal(toolBody(history.root), "Fixture contents.");
});

// Phase 4 pins caller observations before extracting construction and navigation.
const phase4Html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const phase4Plain = (value) => JSON.parse(JSON.stringify(value));
function phase4Section(script, start, end) {
  const first = script.indexOf(start), last = script.indexOf(end, first);
  assert.ok(first >= 0 && last > first, "missing Phase 4 fixture section: " + start);
  return script.slice(first, last);
}
function phase4EffectPoint(effect) {
  if (effect[0] === "dataset") return "dataset:" + effect[2];
  if (effect[0] === "attribute") return "attribute:" + effect[2];
  return effect[0];
}
function phase4Dom(spec) {
  const effects = [], created = [], injected = new Error("fixture failure: " + spec.failAt);
  const record = (...effect) => {
    effects.push(phase4Plain(effect));
    if (spec.failAt && phase4EffectPoint(effect) === spec.failAt) throw injected;
  };
  function node(id, tagName = "div") {
    let className = "", innerHTML = "";
    const element = {
      id, tagName, children: [], attrs: {}, parentNode: null,
      value: "unsent draft", selectionStart: 2, selectionEnd: 3,
      get className() { return className; },
      set className(value) { record("class", id, value); className = String(value); },
      get innerHTML() { return innerHTML; },
      set innerHTML(value) { record("html", id, value); innerHTML = String(value); },
      appendChild(child) {
        record("append", id, child.id);
        this.children.push(child); child.parentNode = this;
        return child;
      },
      setAttribute(key, value) { record("attribute", id, key, value); this.attrs[key] = String(value); },
      removeAttribute(key) { record("remove-attribute", id, key); delete this.attrs[key]; },
      scrollIntoView(options) { record("scroll", id, options); },
    };
    element.dataset = new Proxy({}, {
      set(data, key, value) {
        record("dataset", id, key, String(value));
        data[key] = String(value);
        return true;
      },
    });
    return element;
  }
  const snapshot = (element) => ({
    id: element.id, tagName: element.tagName, className: element.className,
    innerHTML: element.innerHTML, dataset: { ...element.dataset }, attrs: { ...element.attrs },
    value: element.value, selectionStart: element.selectionStart, selectionEnd: element.selectionEnd,
    parent: element.parentNode?.id || null, children: element.children.map(snapshot),
  });
  function outcome(run) {
    try {
      const value = run();
      return value?.id ? { type: "return", node: value.id }
        : value === undefined ? { type: "return", value: "undefined" } : { type: "return", value };
    } catch (error) {
      // Only the expected fixture exception or characterized input error is observed.
      // An unrelated product/fixture error still fails both tests and differential runs.
      if (error !== injected && (!spec.error || error.name !== spec.error[0] || error.message !== spec.error[1])) throw error;
      return { type: "throw", name: error.name, message: error.message };
    }
  }
  return { record, node, snapshot, outcome, effects, created };
}
function phase4PlainMessageObservation(html, spec) {
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const h = phase4Dom(spec), root = h.node("root"), parent = h.node("parent");
  const prior = h.node("prior");
  root.children.push(prior); prior.parentNode = root;
  const c = vm.createContext({
    document: { createElement(tag) {
      h.record("create", tag);
      const element = h.node("created-" + (h.created.length + 1), tag);
      h.created.push(element); return element;
    } },
    $(id) { h.record("lookup", id); assert.equal(id, "msgs"); return root; },
    shouldFollowOutput(node) { h.record("follow-query", node.id); return spec.follow; },
    followOutput(follow, node) { h.record("follow-output", follow, node.id); },
    updateQuestionButton() { h.record("questions"); },
  });
  const escape = script.indexOf("const esc =");
  assert.ok(escape >= 0);
  vm.runInContext(script.slice(escape, script.indexOf("\n", escape)), c);
  vm.runInContext(phase4Section(script, "function addMsg(", "// Assistant replies render as markdown"), c);
  const mount = spec.parent === "explicit" ? parent : spec.parent === "false" ? false : undefined;
  const outcome = h.outcome(() => c.addMsg(spec.role, spec.text, spec.cls, mount));
  return phase4Plain({
    outcome, effects: h.effects, root: h.snapshot(root), parent: h.snapshot(parent),
    created: h.created.map(h.snapshot),
  });
}
const phase4EmptyMessage = "<span style='color:var(--dim)'>(empty)</span>";
const phase4MessageCases = [
  { name: "escaped user text and question metadata", role: "user", text: 'a & <b> "quoted"\n中🙂', cls: "user",
    markup: "a &amp; &lt;b&gt; &quot;quoted&quot;\n中🙂" },
  { name: "plain system notice", role: "system", text: "(paused)", cls: "asst system", markup: "(paused)" },
  { name: "escaped error notice", role: "err", text: 'failed: <timeout> & "offline"', cls: "asst err",
    markup: "failed: &lt;timeout&gt; &amp; &quot;offline&quot;" },
  { name: "plain progress notice", role: "assistant", text: "◌ waiting", cls: "asst pending", markup: "◌ waiting" },
  { name: "empty user placeholder", role: "user", text: "", cls: "user", markup: phase4EmptyMessage },
  { name: "empty error placeholder", role: "err", text: "", cls: "asst err", markup: phase4EmptyMessage },
  { name: "apostrophes stay literal", role: "user", text: "'single'", cls: "user", markup: "'single'" },
  { name: "missing role and class are not defaulted", text: "notice", markup: "notice" },
  { name: "null class is not defaulted", role: "system", text: "notice", cls: null, markup: "notice" },
  { name: "null input throws after metadata", role: "user", text: null, cls: "user",
    error: ["TypeError", "Cannot read properties of null (reading 'replace')"] },
  { name: "undefined input throws after metadata", role: "err", cls: "asst err",
    error: ["TypeError", "Cannot read properties of undefined (reading 'replace')"] },
  { name: "false input is not normalized", role: "user", text: false, cls: "user",
    error: ["TypeError", "s.replace is not a function"] },
  { name: "zero input is not normalized", role: "err", text: 0, cls: "asst err",
    error: ["TypeError", "s.replace is not a function"] },
];
function phase4MessagePrefix(spec) {
  const effects = spec.parent === "explicit" ? [] : [["lookup", "msgs"], ["follow-query", "root"]];
  effects.push(["create", "div"], ["class", "created-1", "msg " + spec.cls],
    ["dataset", "created-1", "role", String(spec.role)]);
  if (spec.role === "user") effects.push(["dataset", "created-1", "question", String(spec.text)]);
  return effects;
}
for (const item of phase4MessageCases) for (const parent of ["global", "explicit"]) {
  test("Phase 4 plain message characterization: " + item.name + " / " + parent, () => {
    const spec = { ...item, parent, follow: false };
    const observed = phase4PlainMessageObservation(phase4Html, spec);
    const expected = phase4MessagePrefix(spec);
    assert.equal(observed.created.length, 1);
    const row = observed.created[0];
    assert.equal(row.className, "msg " + spec.cls);
    assert.deepEqual(row.dataset, spec.role === "user"
      ? { role: "user", question: String(spec.text) } : { role: String(spec.role) });
    if (spec.error) {
      assert.deepEqual(observed.outcome, { type: "throw", name: spec.error[0], message: spec.error[1] });
      assert.equal(row.parent, null);
      assert.equal(row.innerHTML, "");
      assert.equal(observed.root.children.length, 1);
      assert.equal(observed.parent.children.length, 0);
    } else {
      const destination = parent === "explicit" ? "parent" : "root";
      expected.push(["html", "created-1", spec.markup], ["append", destination, "created-1"]);
      if (parent !== "explicit") expected.push(["follow-output", false, "root"], ["questions"]);
      assert.deepEqual(observed.outcome, { type: "return", node: "created-1" });
      assert.equal(row.innerHTML, spec.markup);
      assert.equal(row.parent, destination);
      assert.deepEqual(observed.root.children.map((node) => node.id),
        parent === "explicit" ? ["prior"] : ["prior", "created-1"]);
      assert.deepEqual(observed.parent.children.map((node) => node.id),
        parent === "explicit" ? ["created-1"] : []);
    }
    assert.deepEqual(observed.effects, expected);
  });
}
const phase4MessageFailurePoints = ["lookup", "follow-query", "create", "class",
  "dataset:role", "dataset:question", "html", "append", "follow-output", "questions"];
for (const failAt of phase4MessageFailurePoints) {
  test("Phase 4 plain message propagates " + failAt + " failure without later effects", () => {
    const spec = { ...phase4MessageCases[0], parent: "global", follow: true };
    const complete = phase4PlainMessageObservation(phase4Html, spec);
    const failed = phase4PlainMessageObservation(phase4Html, { ...spec, failAt });
    const stop = complete.effects.findIndex((effect) => phase4EffectPoint(effect) === failAt);
    assert.ok(stop >= 0);
    assert.deepEqual(failed.outcome, { type: "throw", name: "Error", message: "fixture failure: " + failAt });
    assert.deepEqual(failed.effects, complete.effects.slice(0, stop + 1));
    assert.deepEqual(failed.root.children.map((node) => node.id),
      ["follow-output", "questions"].includes(failAt) ? ["prior", "created-1"] : ["prior"]);
    assert.equal(failed.parent.children.length, 0);
  });
}
function phase4NavigationObservation(html, spec) {
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const h = phase4Dom(spec), command = spec.kind === "command";
  const input = h.node(command ? "prompt" : "skillsearch", "input"), menu = h.node("menu");
  const options = h.node("options"), other = h.node("other");
  input.attrs["aria-activedescendant"] = "before-active";
  menu.hidden = !!spec.hidden;
  for (let i = 0; i < spec.count; i++) {
    const row = h.node((command ? "cmdopt-" : "skill-option-") + i, "button");
    row.attrs["data-active"] = "before";
    row.attrs["aria-selected"] = "keep-" + i;
    options.appendChild(row);
  }
  h.effects.length = 0;
  const elements = command ? { cmdmenu: menu, cmdoptions: options, prompt: input }
    : { skillmenu: menu, skilloptions: options, skillsearch: input };
  const document = { activeElement: spec.unfocused ? other : input };
  const c = vm.createContext({
    document, $(id) { h.record("lookup", id); return elements[id]; },
    promptComposing: command && !!spec.composing, skillComposing: !command && !!spec.composing,
    commandActive: spec.active, skillActive: spec.active,
    commandMatches: Array.from({ length: spec.count }, (_, i) => ({ name: "command-" + i })),
    skillMatches: Array.from({ length: spec.count }, (_, i) => ({ name: "skill-" + i })),
    closeSkillMenu() { h.record("close-skill"); },
    chooseSkill(name) { h.record("choose-skill", name); },
  });
  // Load the completion dependencies as well as the navigation slice. A VM
  // without matchMedia still exercises the original navigation observations.
  if (command && script.includes("const desktopCompletion =")) {
    vm.runInContext(phase4Section(script, "const desktopCompletion =", "function normalizeCommands("), c);
  }
  vm.runInContext(phase4Section(script, command ? "function setCommandActive(" : "function setSkillActive(",
    command ? "async function loadInfo()" : "function validateSkillInput()"), c);
  const setter = command ? "setCommandActive" : "setSkillActive", original = c[setter];
  c[setter] = (target) => { h.record("activate", target); return original(target); };
  const event = { key: spec.key, isComposing: !!spec.isComposing, keyCode: spec.keyCode || 0,
    preventDefault() { h.record("prevent", spec.key); } };
  const outcome = h.outcome(() => (command ? c.commandKeydown : c.skillKeydown)(event));
  return phase4Plain({
    outcome, effects: h.effects, active: command ? c.commandActive : c.skillActive,
    input: h.snapshot(input), options: h.snapshot(options), focus: document.activeElement.id,
  });
}
const phase4NavigationCases = [
  { kind: "command", name: "no active row / down", count: 3, active: -1, key: "ArrowDown", target: 0, next: 0 },
  { kind: "command", name: "no active row / up", count: 3, active: -1, key: "ArrowUp", target: 2, next: 2 },
  { kind: "command", name: "first row / up wraps in setter", count: 3, active: 0, key: "ArrowUp", target: -1, next: 2 },
  { kind: "command", name: "last row / down wraps in setter", count: 3, active: 2, key: "ArrowDown", target: 3, next: 0 },
  { kind: "command", name: "out of range keeps raw target", count: 3, active: 7, key: "ArrowDown", target: 8, next: 2 },
  { kind: "command", name: "string target keeps addition coercion", count: 3, active: "1", key: "ArrowDown", target: "11", next: 2 },
  { kind: "command", name: "one row / up", count: 1, active: 0, key: "ArrowUp", target: -1, next: 0 },
  { kind: "command", name: "one row / down", count: 1, active: 0, key: "ArrowDown", target: 1, next: 0 },
  { kind: "command", name: "empty catalog does not prevent", count: 0, active: -1, key: "ArrowDown" },
  { kind: "command", name: "Home remains unsupported", count: 3, active: 1, key: "Home" },
  { kind: "command", name: "End remains unsupported", count: 3, active: 1, key: "End" },
  { kind: "command", name: "hidden menu does not prevent", count: 3, active: 1, key: "ArrowDown", hidden: true },
  { kind: "skill", name: "no active row / down", count: 3, active: -1, key: "ArrowDown", target: 0, next: 0 },
  { kind: "skill", name: "no active row / up", count: 3, active: -1, key: "ArrowUp", target: 2, next: 2 },
  { kind: "skill", name: "first row / up wraps before setter", count: 3, active: 0, key: "ArrowUp", target: 2, next: 2 },
  { kind: "skill", name: "last row / down wraps before setter", count: 3, active: 2, key: "ArrowDown", target: 0, next: 0 },
  { kind: "skill", name: "Home selects first target", count: 3, active: 1, key: "Home", target: 0, next: 0 },
  { kind: "skill", name: "End selects last target", count: 3, active: 1, key: "End", target: 2, next: 2 },
  { kind: "skill", name: "out of range target is wrapped", count: 3, active: 7, key: "ArrowDown", target: 2, next: 2 },
  { kind: "skill", name: "string target keeps addition coercion", count: 3, active: "1", key: "ArrowDown", target: 2, next: 2 },
  { kind: "skill", name: "one row / up", count: 1, active: 0, key: "ArrowUp", target: 0, next: 0 },
  { kind: "skill", name: "empty catalog still prevents", count: 0, active: -1, key: "ArrowDown", prevent: true },
  { kind: "skill", name: "hidden menu does not prevent", count: 3, active: 1, key: "ArrowDown", hidden: true },
  { kind: "skill", name: "unfocused search does not prevent", count: 3, active: 1, key: "ArrowDown", unfocused: true },
];
for (const item of phase4NavigationCases) {
  test("Phase 4 navigation characterization: " + item.kind + " / " + item.name, () => {
    const observed = phase4NavigationObservation(phase4Html, item);
    const activated = Object.hasOwn(item, "target"), command = item.kind === "command";
    assert.deepEqual(observed.outcome, { type: "return", value: command ? activated : "undefined" });
    assert.equal(observed.active, activated ? item.next : item.active);
    assert.deepEqual(observed.effects.filter((effect) => effect[0] === "activate"),
      activated ? [["activate", item.target]] : []);
    assert.deepEqual(observed.effects.filter((effect) => effect[0] === "prevent"),
      activated || item.prevent ? [["prevent", item.key]] : []);
    assert.equal(observed.input.value, "unsent draft");
    assert.equal(observed.input.selectionStart, 2);
    assert.equal(observed.input.selectionEnd, 3);
    assert.equal(observed.focus, item.unfocused ? "other" : observed.input.id);
    assert.equal(observed.input.attrs["aria-activedescendant"], activated
      ? (command ? "cmdopt-" : "skill-option-") + item.next : "before-active");
    const attributes = [];
    for (let i = 0; i < item.count; i++) {
      const row = observed.options.children[i], active = i === item.next ? "true" : "false";
      assert.equal(row.attrs["data-active"], activated ? active : "before");
      assert.equal(row.attrs["aria-selected"], activated && command ? active : "keep-" + i);
      if (activated) {
        if (command) attributes.push(["attribute", row.id, "aria-selected", active]);
        attributes.push(["attribute", row.id, "data-active", active]);
      }
    }
    if (activated) {
      attributes.push(["attribute", observed.input.id, "aria-activedescendant", observed.input.attrs["aria-activedescendant"]]);
      assert.deepEqual(observed.effects.filter((effect) => effect[0] === "attribute"), attributes);
      assert.deepEqual(observed.effects.at(-1), ["scroll", observed.input.attrs["aria-activedescendant"], { block: "nearest" }]);
      const preventing = observed.effects.findIndex((effect) => effect[0] === "prevent");
      const activating = observed.effects.findIndex((effect) => effect[0] === "activate");
      assert.ok(preventing < activating, "prevent the event before updating the active row");
    }
  });
}
for (const kind of ["command", "skill"]) for (const guard of ["composing", "isComposing", "keyCode"]) {
  test("Phase 4 navigation preserves " + kind + " " + guard + " guard", () => {
    const spec = { kind, count: 3, active: 1, key: "ArrowDown", [guard]: guard === "keyCode" ? 229 : true };
    const observed = phase4NavigationObservation(phase4Html, spec);
    assert.deepEqual(observed.outcome, { type: "return", value: kind === "command" ? false : "undefined" });
    assert.equal(observed.active, 1);
    assert.deepEqual(observed.effects, []);
    assert.equal(observed.input.attrs["aria-activedescendant"], "before-active");
  });
}
for (const kind of ["command", "skill"]) for (const failAt of ["prevent", "activate", "attribute:data-active", "scroll"]) {
  test("Phase 4 navigation propagates " + kind + " " + failAt + " failure in order", () => {
    const spec = { kind, count: 3, active: 0, key: "ArrowUp" };
    const complete = phase4NavigationObservation(phase4Html, spec);
    const failed = phase4NavigationObservation(phase4Html, { ...spec, failAt });
    const stop = complete.effects.findIndex((effect) => phase4EffectPoint(effect) === failAt);
    assert.ok(stop >= 0);
    assert.deepEqual(failed.outcome, { type: "throw", name: "Error", message: "fixture failure: " + failAt });
    assert.deepEqual(failed.effects, complete.effects.slice(0, stop + 1));
    assert.equal(failed.active, ["prevent", "activate"].includes(failAt) ? 0 : 2);
  });
}
