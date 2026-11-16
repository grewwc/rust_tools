// Run with: env NODE_OPTIONS= node --test src/bin/ai/serve/app_plan_card.test.cjs
// Exercise the shared renderer through live, replay and canonical adapters.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];

function source(first, last) {
  const start = script.indexOf(first), end = script.indexOf(last, start);
  assert.ok(start >= 0 && end > start, "missing page source: " + first);
  return script.slice(start, end);
}
function harness({ phone = false, storage = new Map() } = {}) {
  function element(tag = "div") {
    let text = "";
    const listeners = new Map();
    const el = {
      tagName: tag.toUpperCase(), children: [], parentNode: null, className: "", dataset: {}, style: {}, hidden: false, open: false,
      get innerHTML() { throw new Error("plan content must use textContent"); },
      set innerHTML(_) { throw new Error("plan content must use textContent"); },
      get isConnected() {
        for (let node = this; node; node = node.parentNode) if (node === root) return true;
        return false;
      },
      get textContent() { return text + this.children.map((child) => child.textContent).join(""); },
      set textContent(value) { text = String(value); this.children.forEach((child) => { child.parentNode = null; }); this.children = []; },
      get lastElementChild() { return this.children.at(-1) || null; },
      get nextElementSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; },
      appendChild(child) { return this.insertBefore(child, null); },
      insertBefore(child, anchor) {
        if (child === anchor) return child;
        for (let ancestor = this; ancestor; ancestor = ancestor.parentNode) assert.notEqual(ancestor, child, "DOM placement cannot create a cycle");
        assert.ok(!anchor || anchor.parentNode === this, "insertion anchor belongs to this owner");
        child.remove();
        const index = anchor ? this.children.indexOf(anchor) : this.children.length;
        this.children.splice(index, 0, child); child.parentNode = this; return child;
      },
      remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this); this.parentNode = null; },
      matches(selector) {
        if (selector.includes(",")) return selector.split(",").some((part) => this.matches(part.trim()));
        const tag = /^[\w-]+/.exec(selector)?.[0];
        return (!tag || this.tagName === tag.toUpperCase()) && [...selector.matchAll(/\.([\w-]+)/g)].every((match) => this.classList.contains(match[1]));
      },
      querySelectorAll(selector) { return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
      closest(selector) { for (let node = this; node; node = node.parentNode) if (node.matches(selector)) return node; return null; },
      addEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).concat(fn)); },
      fire(type) { for (const fn of listeners.get(type) || []) fn({ target: this }); },
    };
    el.classList = {
      contains: (name) => el.className.split(/\s+/).includes(name),
      add: (name) => { if (!el.classList.contains(name)) el.className = (el.className + " " + name).trim(); },
      remove: (name) => { el.className = el.className.split(/\s+/).filter((part) => part !== name).join(" "); },
    };
    return el;
  }
  const root = element(), context = vm.createContext({
    document: { createElement: element, hidden: false, body: root, addEventListener() {} },
    window: { addEventListener() {}, matchMedia: () => ({ matches: !phone }) }, MutationObserver: class { observe() {} },
    $: () => root, sessionId: "s", isPhoneLayout: () => phone,
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    textOf: (text) => typeof text === "string" ? text : "", isRuntimeSyntheticUser: () => false,
    Date: { now: () => 1000 }, setInterval: () => 1, clearInterval() {},
    createTurnRateLine: () => () => {}, createOutputRateCounter: () => ({ text: () => "", counts: () => "", reset() {} }),
    createThinkingRateRecorder: () => ({ start() {}, close() {} }), statusRateText: (text) => text,
    closeThinkingPhase() {}, sealAnswerSegment() {}, mutateTranscript: (fn) => fn(), schedulePaint() {},
  });
  for (const [first, last] of [
    ["// BEGIN execution model", "// END execution model"], ["let lastFoldTap", "// The agent sends a phone action"],
    ["function toolArgsOf(", "const RATE_STORAGE ="], ["function addFold(", "// The three override picks"],
    ["function historyRowsOf(", "// A session the server reports as running"], ["function recoveredProgressView(", "function showEventWarning("],
  ]) vm.runInContext(source(first, last), context);
  vm.runInContext(source("  // BEGIN live event view", "  // END live event view")
    + '\nfunction liveTool(payload) { dispatchExecutionEvent("tool", payload, liveView, "live"); }', context);
  const messages = [];
  function emit(mode, name, args, output, extra = {}) {
    const argumentsText = typeof args === "string" ? args : JSON.stringify(args);
    const event = { id: "plan-call", name, arguments: argumentsText, output, phase: "finished", ok: true, ...extra };
    if (mode === "live") {
      context.activity ||= context.createLiveActivity(root, null); context.liveTool(JSON.stringify(event));
    } else if (mode === "replay") {
      context.view ||= context.recoveredProgressView("s"); context.handleReplayEvent("s", "tool", JSON.stringify(event), context.view);
    } else {
      messages.push({ role: "assistant", content: "", tool_calls: [{ id: event.id, function: { name, arguments: argumentsText } }] },
        { role: "tool", tool_call_id: event.id, content: output, ok: event.ok,
          output_truncated: event.output_truncated, arguments_truncated: event.arguments_truncated });
      root.textContent = "";
      const canonical = JSON.parse(JSON.stringify(messages));
      for (const row of context.historyRowsOf(canonical, canonical.length, true).rows) context.renderHistoryRow(row, root);
    }
    return root.querySelectorAll(".plan-card").at(-1);
  }
  return { root, context, emit, storage };
}
function row(step, action, status = "pending", extra = {}) { return { step, action, status, tool: "read_file", reason: "Reason for " + step, ...extra }; }
function headline(step) {
  const note = step.note?.trim();
  const suffix = step.status === "pending" ? note ? " (" + note + ")" : ""
    : " (" + step.status + (note ? ", note: " + note : "") + ")";
  return (step.parallelizable ? "  || " : "") + "Step " + step.step + ". [" + (step.tool ?? "unspecified") + "]"
    + (step.delegate ? " [delegate]" : "") + " " + step.action + suffix + "\n" + (step.reason ? "  Reason: " + step.reason + "\n" : "");
}
function progress(total, extra = {}) {
  const counts = { done: 0, running: 0, failed: 0, skipped: 0, ...extra };
  return "Progress: " + counts.done + "/" + total + " steps done"
    + ["running", "failed", "skipped"].filter((status) => counts[status]).map((status) => ", " + counts[status] + " " + status).join("") + ".\n";
}
function fullPlan(steps, summary = "Review the change") {
  const counts = Object.fromEntries(["done", "running", "failed", "skipped"].map((status) => [status, steps.filter((step) => step.status === status).length]));
  return (summary ? "Plan: " + summary + "\n" : "") + (Object.values(counts).some(Boolean) ? progress(steps.length, counts) : "")
    + "\n" + steps.map(headline).join("") + "\n---\n" + steps.length + " step(s) planned. Proceed to execute.\n";
}
function inputOf(steps, summary = "Review the change") { return { summary, steps: steps.map(({ status, note, ...step }) => step) }; }
function delta(steps, changed, summary = "Review the change") {
  const counts = Object.fromEntries(["done", "running", "failed", "skipped"].map((status) => [status, steps.filter((step) => step.status === status).length]));
  const failed = steps.filter((step) => step.status === "failed"), remaining = steps.filter((step) => ["pending", "running"].includes(step.status));
  return (summary ? "Plan: " + summary + "\n" : "") + (changed && ["done", "skipped"].includes(changed.status) ? headline(changed) : "")
    + progress(steps.length, counts) + (failed.length ? "Failed steps:\n" + failed.map(headline).join("") : "")
    + (remaining.length ? "Remaining steps:\n" + remaining.map(headline).join("") : "Remaining steps: none.\n");
}
function assertRaw(card, output, open = false) {
  const details = card.parentNode.querySelector(".plan-raw");
  assert.equal(details.querySelector("summary").textContent, "Raw");
  assert.equal(details.open, open);
  assert.ok(details.querySelector(".prog").textContent.endsWith(output));
}

for (const phone of [false, true]) for (const mode of ["live", "replay", "history"]) {
  const label = mode + " phone=" + phone;
  test("full plan highlights every running row and keeps exact mixed counts, " + label, () => {
    const h = harness({ phone }), steps = [row(0, "Read context", "done"), row(8, "Inspect files", "running"), row(3, "Review tests", "running", { parallelizable: true }),
      row(8, "Check failure", "failed"), row(42, "Optional check", "skipped"), row(2, "Write report")];
    const output = fullPlan(steps), card = h.emit(mode, "plan", inputOf(steps), output);
    assert.equal(card.querySelector(".plan-count").textContent, "1/6");
    assert.equal(card.dataset.state, "failed");
    const rows = card.querySelectorAll(".plan-step");
    assert.deepEqual(rows.map((line) => line.dataset.state), steps.map((step) => step.status));
    assert.equal(rows.filter((line) => line.dataset.state === "running").length, 2);
    for (const [index, line] of rows.entries()) {
      assert.equal(line.querySelector(".plan-step-action").textContent, steps[index].action);
      assert.equal(Boolean(line.querySelector(".plan-step-reason")), ["running", "failed"].includes(steps[index].status));
      if (line.dataset.state === "running") assert.equal(line.querySelector(".work-status").textContent, "running");
    }
    const segments = card.querySelectorAll(".plan-progress-segment");
    assert.deepEqual(segments.map((segment) => segment.dataset.state), ["done", "running", "failed", "skipped"]);
    for (const [index, expected] of [100 / 6, 100 / 3, 100 / 6, 100 / 6].entries()) {
      assert.ok(Math.abs(Number.parseFloat(segments[index].style.width) - expected) < 1e-10, "progress segment must match the result count");
    }
    assert.equal(card.querySelector(".plan-counts").textContent, "1 done · 2 running · 1 failed · 1 skipped · 1 pending");
    assertRaw(card, output);
  });
  test("all-pending full plan uses array length and keeps status-looking action prose, " + label, () => {
    const h = harness({ phone }), steps = [row(9, "Read the file (running)"), row(9, "A second step"), row(0, "Finish")];
    const card = h.emit(mode, "plan", inputOf(steps), fullPlan(steps));
    assert.equal(card.dataset.state, "pending"); assert.equal(card.querySelector(".plan-count").textContent, "0/3");
    assert.ok(card.querySelectorAll(".plan-step").every((line) => line.dataset.state === "pending"));
    assert.equal(card.querySelector(".plan-step-action").textContent, steps[0].action);
    assert.equal(card.querySelectorAll(".plan-progress-segment").length, 0);
  });
  test("running delta promotes the changed step and never invents completed rows or pending counts, " + label, () => {
    const h = harness({ phone }), changed = row(3, "Inspect the implementation", "running", { note: "Use the full source (including notes)" });
    const steps = [row(1, "An omitted completion", "done"), row(2, "Another omitted completion", "done"), changed, row(4, "Write report")];
    const output = delta(steps, changed), card = h.emit(mode, "plan_update", { step: 3, status: "running", note: changed.note }, output);
    assert.equal(card.dataset.state, "running"); assert.equal(card.querySelector(".plan-count").textContent, "2/4");
    assert.equal(card.querySelector(".plan-hero").dataset.state, "running");
    assert.equal(card.querySelector(".plan-hero").querySelector(".plan-step-note").textContent, changed.note);
    assert.deepEqual(card.querySelectorAll(".plan-step").map((line) => line.dataset.step), ["3", "4"]);
    assert.equal(card.querySelector(".plan-counts").textContent, "2 done · 1 running");
    assert.ok(!card.textContent.includes("omitted completion")); assertRaw(card, output);
  });
  test("failed delta keeps failed and remaining work distinct with a transition warning, " + label, () => {
    const h = harness({ phone }), changed = row(1, "Retry the read", "running"), failed = row(2, "Check the failed command", "failed", { note: "Exit code 1" });
    const steps = [changed, failed, row(3, "Report")], warning = "Warning: step 1 was marked done, and is now running — this overrides a terminal status.";
    const card = h.emit(mode, "plan_update", { step: 1, status: "running" }, warning + "\n" + delta(steps, changed));
    assert.equal(card.dataset.state, "failed"); assert.equal(card.querySelector(".plan-warning").textContent, warning);
    assert.equal(card.querySelectorAll(".plan-group").length, 2);
    const failure = card.querySelectorAll(".plan-group").find((group) => group.dataset.state === "failed");
    assert.equal(failure.querySelector(".plan-step").dataset.state, "failed");
    assert.equal(failure.querySelector(".plan-step-reason").textContent, "Reason: " + failed.reason);
    assert.equal(card.querySelectorAll(".plan-step").filter((line) => line.dataset.step === "1").length, 1);
  });
  test("none remaining with failed work never claims success, " + label, () => {
    const h = harness({ phone }), changed = row(1, "Completed read", "done"), failed = row(2, "Still failed", "failed");
    const card = h.emit(mode, "plan_update", { step: 1, status: "done" }, delta([changed, failed], changed));
    assert.equal(card.dataset.state, "failed"); assert.equal(card.querySelector(".plan-count").textContent, "1/2");
    assert.equal(card.querySelector(".plan-message").textContent, "No remaining steps · failed steps need attention.");
    assert.equal(card.querySelector(".plan-hero").dataset.state, "done");
    assert.equal(card.querySelectorAll(".plan-step").filter((line) => line.dataset.state === "running").length, 0);
  });
  test("a changed failed step is a failed hero even with no remaining work, " + label, () => {
    const h = harness({ phone }), changed = row(1, "Read failed", "failed", { note: "Permission denied" });
    const card = h.emit(mode, "plan_update", { step: 1, status: "failed", note: changed.note }, delta([changed], changed));
    assert.equal(card.dataset.state, "failed"); assert.equal(card.querySelector(".plan-hero").dataset.state, "failed");
    assert.equal(card.querySelector(".plan-hero").querySelector(".plan-step-note").textContent, changed.note);
    assert.equal(card.querySelector(".plan-message").textContent, "No remaining steps · failed steps need attention.");
    assert.equal(card.querySelectorAll(".plan-step").length, 1);
  });
  test("all-complete delta displays a done hero without inventing old completions, " + label, () => {
    const h = harness({ phone }), changed = row(2, "Finish report", "done", { note: "Verified" });
    const card = h.emit(mode, "plan_update", { step: 2, status: "done", note: "Verified" }, delta([row(1, "Old completion", "done"), changed], changed));
    assert.equal(card.dataset.state, "done"); assert.equal(card.querySelector(".plan-count").textContent, "2/2");
    assert.equal(card.querySelectorAll(".plan-step").length, 1); assertRaw(card, delta([row(1, "Old completion", "done"), changed], changed));
  });
  for (const output of ["Error: bad request", "No active plan in this session. Call `plan` first.", "Step 7 not found in the active plan.\n" + fullPlan([row(1, "Done", "done")]), "Unknown result format", "Progress: 9/2 steps done.\nRemaining steps: none."]) {
    test("plan fallback retains visible raw bytes and no speculative progress, " + label + " " + output.split("\n")[0], () => {
      const h = harness({ phone }), card = h.emit(mode, "plan_update", { step: 7, status: "done" }, output);
      assert.ok(["attention", "failed"].includes(card.dataset.state));
      assert.equal(card.querySelector(".plan-count"), null); assert.equal(card.querySelector(".plan-progress"), null);
      assert.equal(card.querySelectorAll(".plan-step").length, 0); assertRaw(card, output, true);
    });
  }
  for (const flag of ["output_truncated", "arguments_truncated"]) test("truncated flags suppress totals and completion claims, " + label + " " + flag, () => {
    const h = harness({ phone }), steps = [row(1, "Done", "done")], output = fullPlan(steps);
    const card = h.emit(mode, "plan", inputOf(steps), output, { [flag]: true });
    assert.equal(card.querySelector(".plan-count"), null); assert.equal(card.querySelector(".plan-progress"), null);
    assert.ok(card.querySelector(".plan-hint").textContent.includes("truncated")); assert.notEqual(card.dataset.state, "done");
  });
  test("long CJK actions and multiline field-looking prose stay exact and cannot inject HTML, " + label, () => {
    const h = harness({ phone }), action = "审查文档读取流程并核对当前进度".repeat(180)
      + '\nStep 2. [read_file] Literal inside the action (running)\nReason: Literal reason\n<img src=x onerror=alert(1)><script>throw 1</script>';
    const steps = [row(1, action, "running", { reason: "检查原始内容\nStep 2. [none] Literal reason line", note: "运行中\nRemaining steps: none." }), row(2, "Finish")];
    const output = fullPlan(steps), card = h.emit(mode, "plan", inputOf(steps), output);
    assert.equal(card.querySelector(".plan-count").textContent, "0/2");
    assert.equal(card.querySelector(".plan-step-action").textContent, action);
    assert.equal(card.querySelectorAll(".plan-step").length, 2);
    assert.equal(h.root.querySelectorAll("img, script").length, 0); assertRaw(card, output);
    assert.equal(Boolean(card.querySelector(".plan-hint")), false, "full history is parsed before the ordinary output cap");
  });
  test("exact delta notes protect embedded structural markers, " + label, () => {
    const h = harness({ phone }), changed = row(1, "Review the source", "running", { note: "第一行\nStep 5. [none] literal\nProgress: 9/9 steps done.\nRemaining steps: none." });
    const card = h.emit(mode, "plan_update", { step: 1, status: "running", note: changed.note }, delta([changed, row(2, "Report")], changed));
    assert.equal(card.dataset.state, "running"); assert.equal(card.querySelector(".plan-step-note").textContent, changed.note);
    assert.equal(card.querySelectorAll(".plan-step").length, 2);
  });
  test("ambiguous multiline delta falls back without applying requested state, " + label, () => {
    const h = harness({ phone }), changed = row(1, "First line\nSecond line", "running");
    const output = delta([changed], changed), card = h.emit(mode, "plan_update", { step: 1, status: "running" }, output);
    assert.equal(card.querySelector(".plan-count"), null); assert.equal(card.querySelectorAll(".plan-step").length, 0);
    assertRaw(card, output, true);
  });
  test("another step's multiline note cannot swap the running highlight onto literal action text, " + label, () => {
    for (const continuation of ["Reason: y", "  Reason: y"]) {
      const h = harness({ phone }), args = { step: 3, status: "done", note: "" };
      const output = [
        "Step 3. [none] Finish (done)",
        "Progress: 1/3 steps done, 1 running.",
        "Remaining steps:",
        "Step 1. [none] Read (running)",
        "Step 2. [none] Inspect (running, note: x",
        continuation + ")",
        "",
      ].join("\n");
      const result = h.context.projectPlanTool({ function: { name: "plan_update", arguments: JSON.stringify(args) } }, output);
      assert.equal(result.fallback, true, "an unknown multiline note must not downgrade its running row to pending");
      assert.equal(result.counts, null, "matching aggregate counts cannot validate a swapped status");
      const card = h.emit(mode, "plan_update", args, output);
      assert.equal(card.querySelectorAll(".plan-step").length, 0, "raw fallback must not retain a running highlight on step 1");
      assert.equal(card.querySelector(".plan-count"), null); assertRaw(card, output, true);
    }
  });
  test("an unindented Reason action continuation is ambiguous rather than a reason field, " + label, () => {
    const h = harness({ phone }), args = { step: 2, status: "done", note: "" };
    const output = [
      "Step 2. [none] Finish (done)",
      "Progress: 1/2 steps done.",
      "Remaining steps:",
      "Step 1. [none] Inspect",
      "Reason: literal action continuation",
      "",
    ].join("\n");
    const result = h.context.projectPlanTool({ function: { name: "plan_update", arguments: JSON.stringify(args) } }, output);
    assert.equal(result.fallback, true, "a literal continuation cannot become producer metadata");
    const card = h.emit(mode, "plan_update", args, output);
    assert.equal(card.querySelectorAll(".plan-step").length, 0); assertRaw(card, output, true);
  });
  test("a status on an indented action continuation cannot validate a swapped running row, " + label, () => {
    const h = harness({ phone }), args = { step: 3, status: "done", note: "" };
    const output = [
      "Step 3. [none] Finish (done)",
      "Progress: 1/3 steps done, 1 running.",
      "Remaining steps:",
      "Step 1. [none] Read (running)",
      "Step 2. [none] Inspect",
      "  Reason: literal action continuation (running)",
      "",
    ].join("\n");
    const result = h.context.projectPlanTool({ function: { name: "plan_update", arguments: JSON.stringify(args) } }, output);
    assert.equal(result.fallback, true, "a continuation's status suffix leaves the action boundary ambiguous");
    assert.equal(result.counts, null);
    const card = h.emit(mode, "plan_update", args, output);
    assert.equal(card.querySelectorAll(".plan-step").length, 0); assertRaw(card, output, true);
  });
  test("exact plan arguments still preserve status-looking actions and multiline note/reason prose, " + label, () => {
    const h = harness({ phone }), steps = [
      row(1, "Read (running)", "pending", { tool: "none", reason: "" }),
      row(2, "Inspect\nReason: literal action continuation", "running", { tool: "none", reason: "Actual reason", note: "x\nReason: y" }),
      row(3, "Finish", "done", { tool: "none", reason: "" }),
    ];
    const output = fullPlan(steps), card = h.emit(mode, "plan", inputOf(steps), output);
    const rows = card.querySelectorAll(".plan-step");
    assert.deepEqual(rows.map((line) => line.dataset.state), ["pending", "running", "done"]);
    assert.equal(rows[0].querySelector(".plan-step-action").textContent, "Read (running)");
    assert.equal(rows[1].querySelector(".plan-step-action").textContent, steps[1].action);
    assert.equal(rows[1].querySelector(".plan-step-note").textContent, steps[1].note);
    assert.equal(rows[1].querySelector(".plan-step-reason").textContent, "Reason: Actual reason");
    assert.equal(card.querySelector(".plan-count").textContent, "1/3"); assertRaw(card, output);
  });
}

for (const mode of ["live", "replay"]) test("started plans prepare quietly and finished results retain fold/raw nodes, " + mode, () => {
  const h = harness(), steps = [row(1, "Inspect")], args = inputOf(steps);
  const card = h.emit(mode, "plan", args, "", { phase: "started" }), outer = card.parentNode, raw = outer.querySelector(".plan-raw"), body = raw.querySelector(".prog");
  assert.equal(card.dataset.state, "preparing"); assert.equal(card.querySelector(".plan-step").dataset.state, "pending");
  outer.open = true; outer.fire("toggle"); raw.open = true; raw.fire("toggle");
  const finished = h.emit(mode, "plan", args, fullPlan(steps));
  assert.equal(finished, card); assert.equal(card.parentNode, outer); assert.equal(outer.open, true);
  assert.equal(outer.querySelector(".plan-raw"), raw); assert.equal(raw.querySelector(".prog"), body); assert.equal(raw.open, true);
});
for (const mode of ["live", "replay"]) test("started updates do not apply a requested status before the finished result, " + mode, () => {
  const h = harness(), args = { step: 1, status: "done", note: "Verified" };
  const card = h.emit(mode, "plan_update", args, "", { phase: "started" });
  assert.equal(card.dataset.state, "preparing"); assert.equal(card.querySelectorAll(".plan-step").length, 0);
  assert.equal(card.querySelector(".plan-count"), null);
  const changed = row(1, "Finish", "done", { note: "Verified" });
  h.emit(mode, "plan_update", args, delta([changed], changed));
  assert.equal(card.dataset.state, "done"); assert.equal(card.querySelector(".plan-hero").dataset.state, "done");
});
test("interrupted live updates cannot leave a preparing or running plan preview behind", () => {
  const h = harness(), card = h.emit("live", "plan_update", { step: 1, status: "running" }, "", { phase: "started" });
  h.context.activity.finish("interrupted");
  assert.equal(card.dataset.state, "attention"); assert.ok(card.textContent.includes("interrupted"));
  assert.equal(card.querySelector(".plan-progress"), null); assert.equal(card.parentNode.querySelector(".plan-raw").open, true);
});
test("plan raw disclosure choices survive canonical reconstruction", () => {
  const steps = [row(1, "Inspect")], h = harness(), args = inputOf(steps), output = fullPlan(steps);
  const card = h.emit("history", "plan", args, output), raw = card.parentNode.querySelector(".plan-raw");
  card.parentNode.open = true; card.parentNode.fire("toggle"); raw.open = true; raw.fire("toggle");
  const cold = harness({ storage: new Map(h.storage) }), restored = cold.emit("history", "plan", args, output);
  assert.equal(restored.parentNode.open, true); assert.equal(restored.parentNode.querySelector(".plan-raw").open, true);
});
test("malformed plan arguments and unknown progress labels produce safe raw fallbacks", () => {
  const h = harness();
  for (const argumentsText of ["{", "null", "[]", '{"steps":[null]}']) {
    const call = { function: { name: "plan", arguments: argumentsText } };
    assert.doesNotThrow(() => h.context.projectPlanTool(call, "Unrecognized output"));
    const result = h.context.projectPlanTool(call, "Unrecognized output");
    assert.equal(result.counts, null); assert.equal(result.fallback, true);
  }
  const result = h.context.projectPlanTool({ function: { name: "plan_update", arguments: "{}" } }, "Progress: 0/2 steps done, 2 pending.\nRemaining steps: none.");
  assert.equal(result.counts, null); assert.equal(result.fallback, true);
});
test("explicitly capped history never claims a complete total", () => {
  const h = harness(), steps = [row(1, "Done", "done")];
  const result = h.context.projectPlanTool({ function: { name: "plan", arguments: JSON.stringify(inputOf(steps)) } }, fullPlan(steps), { historyCapped: true });
  assert.equal(result.truncated, true); assert.equal(result.counts, null); assert.notEqual(result.state, "done");
});
test("contradictory changed-step counts and empty groups stay raw without false status", () => {
  const h = harness(), args = { step: 1, status: "done" };
  for (const output of [
    "Step 1. [none] Done (done)\nProgress: 0/1 steps done.\nRemaining steps:\nStep 2. [none] Pending\n",
    "Step 1. [none] Done (done)\nProgress: 1/1 steps done.\nFailed steps:\nRemaining steps: none.\n",
    "Step 1. [none] Done (done)\nProgress: 1/1 steps done.\nRemaining steps:\n",
  ]) {
    const card = h.emit("live", "plan_update", args, output);
    assert.equal(card.dataset.state, "attention"); assert.equal(card.querySelector(".plan-count"), null);
    assert.equal(card.querySelectorAll(".plan-step").length, 0); assertRaw(card, output, true);
  }
});
test("plan styling uses theme tokens, clear running accents, compact progress and reduced motion", () => {
  assert.match(css, /\.plan-step\[data-state="running"\]\s*\{[^}]*var\(--acc\) 9%[^}]*border-inline-start-color:var\(--acc\)/);
  assert.match(css, /\.plan-step\[data-state="running"\] \.plan-step-action[^}]*font-weight:600/);
  assert.match(css, /\.plan-progress\s*\{[^}]*height:4px[^}]*border-radius:999px/);
  assert.match(css, /\.plan-step-action\s*\{[^}]*white-space:pre-wrap[^}]*overflow-wrap:anywhere/);
  assert.match(css, /\.plan-step\s*\{[^}]*minmax\(0,1fr\)/);
  assert.match(css, /@media \(prefers-reduced-motion:reduce\)\s*\{[^}]*\.plan-progress-segment\[data-state="running"\][^}]*animation:none/);
});
