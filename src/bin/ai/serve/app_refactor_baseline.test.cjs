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
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
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

test("session navigation keeps the mounted transcript gate until replacement history commits", async () => {
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

  Object.assign(c, {
    window: { matchMedia: () => ({ matches: false }) },
    localStorage: { setItem() {}, removeItem() {} },
    persistDraft() {}, restoreDraft() {}, resetQuestionNavigator() {},
    restoreSkillPick() {}, restoreTeamPick() {},
  });
  const start = stream.script.indexOf("// BEGIN session switch");
  const end = stream.script.indexOf("// END session switch");
  assert.ok(start > 0 && end > start);
  vm.runInContext(stream.script.slice(start, end), c);
  c.switchSession("session-b", false);
  c.stopStaleFollow();

  const body = deferred();
  c.api = async () => ({ headers: { get: () => "1" }, json: () => body.promise });
  const pending = c.loadHistory();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(c.sessionId, "session-b");
  assert.equal(c.displayedExecutionState().sessionId, "s");
  assert.equal(c.isTurnRunning(), true);
  assert.equal(composer.send.title, "stop this turn");

  body.resolve([{ role: "assistant", content: "Visible B answer." }]);
  await pending;
  assert.equal(c.displayedExecutionState().sessionId, "session-b");
  assert.equal(c.isTurnRunning(), false);
  assert.equal(composer.send.title, "send");
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
