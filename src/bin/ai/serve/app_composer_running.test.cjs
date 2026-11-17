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
  assert.equal(c.canStop(), true, "expired idle evidence no longer overrides the task snapshot");
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
      c.$ = (id) => ["send", "sidenote", "composer"].includes(id) ? controls[id] : lookup(id);
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
