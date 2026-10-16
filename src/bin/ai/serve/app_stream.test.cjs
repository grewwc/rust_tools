// Run with: node --test src/bin/ai/serve/app_stream.test.cjs
// Execute the embedded stream state and dispatch, rather than a second renderer.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "app.html"), "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];

function streamHarness() {
  const box = {
    children: [],
    insertBefore(node, before) {
      this.children = this.children.filter((child) => child !== node);
      this.children.splice(this.children.indexOf(before), 0, node);
    },
  };
  function node() {
    const classes = new Set();
    const item = { dataset: {}, textContent: "", innerHTML: "", classList: {
      add: (name) => classes.add(name), remove: (name) => classes.delete(name),
    }, remove() { box.children = box.children.filter((child) => child !== item); } };
    return item;
  }
  const context = vm.createContext({
    box, sid: "s", sessionId: "s", finished: false, turnError: false,
    setTimeout: () => 1, clearTimeout() {},
    shouldFollowOutput: () => false, followOutput() {},
    renderMarkdown: (text) => text, decorateMath() {},
    showConfirm() {}, hideConfirm() {}, confirmShown: null,
    addMsgMd() { const item = node(); box.children.push(item); return item; },
    addFold(title, text) {
      const item = node(), summary = { textContent: title }, details = { open: false };
      const body = { textContent: text };
      item.summary = summary; item.body = body;
      item.querySelector = (selector) => selector === "summary" ? summary : details;
      box.children.push(item);
      return { box: item, body, set(label, value) { summary.textContent = label; body.textContent = value; } };
    },
  });
  vm.runInContext(script.slice(script.indexOf("function toolArgsOf("), script.indexOf("const FOLD_ROLE_LABELS")), context);
  vm.runInContext(script.slice(script.indexOf("function createThinkingFold("), script.indexOf("// The three override picks")), context);
  const start = script.indexOf('  const bubble = addMsgMd("", box);');
  vm.runInContext(script.slice(start, script.indexOf("  let turnStarted = false;", start)), context);
  const dispatchStart = script.indexOf('        if (kind === "done")', start);
  const dispatchEnd = script.indexOf("      if (finished) break;", dispatchStart);
  // Remove the SSE block loop's closing brace and supply a loop for done's break.
  const dispatch = script.slice(dispatchStart, dispatchEnd).replace(/\s*}\s*$/, "");
  return {
    box,
    emit(kind, value) {
      context.kind = kind;
      context.payload = typeof value === "string" ? value : JSON.stringify(value ?? {});
      vm.runInContext("for (;;) { " + dispatch + "\nbreak; } flushPaint();", context);
    },
    titles: () => box.children.filter((item) => item.summary).map((item) => item.summary.textContent),
    bodies: () => box.children.filter((item) => item.body).map((item) => item.body.textContent),
    pending: () => vm.runInContext("bubble.textContent", context),
    fail: () => vm.runInContext('fail("turn failed")', context),
    reply: () => vm.runInContext("bubble.innerHTML", context),
  };
}

test("live tools separate real thinking phases, while logs cannot split tokens", () => {
  const h = streamHarness();
  h.emit("thinking_start");
  h.emit("thinking", { text: "The" });
  h.emit("progress", "request diagnostic");
  h.emit("thinking", { text: " user asks" });
  h.emit("thinking_done");
  h.emit("answer_pending");
  h.emit("tool", { id: "t1", name: "web_search", phase: "started", arguments: '{"query":"DeepSeek"}' });
  assert.match(h.pending(), /Running web_search/);
  assert.match(h.titles().join("\n"), /web_search DeepSeek/);
  h.emit("answer_pending");
  assert.match(h.pending(), /Running web_search/);
  h.emit("tool", { id: "t1", name: "web_search", phase: "finished", output: "search results" });
  h.emit("thinking_start");
  h.emit("thinking", { text: "Next thought" });
  h.emit("progress", "another diagnostic");
  assert.deepEqual(h.titles().filter((title) => title.startsWith("thinking")), ["thinking", "thinking…"]);
  assert.equal(h.titles().filter((title) => title.startsWith("activity log")).length, 1);
  assert.ok(h.bodies().includes("The user asks"));
  assert.ok(h.bodies().some((text) => text.includes("search results")));
  assert.ok(h.titles().findIndex((title) => title.includes("web_search")) < h.titles().lastIndexOf("thinking…"));
  h.emit("delta", { delta: "Final answer" });
  h.emit("done");
  assert.equal(h.reply(), "Final answer");
});

test("parallel tool completion and confirmation preserve the remaining active status", () => {
  const h = streamHarness();
  for (const id of ["a", "b"]) h.emit("tool", { id, name: "read_file", phase: "started", arguments: JSON.stringify({ file_path: id }) });
  h.emit("confirm_request", { id: "confirm", prompt: "approve?" });
  assert.match(h.pending(), /Waiting for confirmation/);
  h.emit("confirm_done", { id: "confirm" });
  h.emit("tool", { id: "b", name: "read_file", phase: "finished", output: "B" });
  assert.match(h.pending(), /Running read_file…/);
  h.emit("tool", { id: "b", name: "read_file", phase: "finished", output: "B" });
  assert.equal(h.titles().length, 2);
  h.fail();
  assert.ok(h.titles().some((title) => title.includes("interrupted")));
  assert.ok(h.bodies().some((text) => text.includes("B")));
});

test("subagent snapshots stay distinct from logs and empty thinking markers", () => {
  const h = streamHarness();
  h.emit("thinking_start"); h.emit("thinking_done");
  h.emit("subagent_progress", "worker: running");
  h.emit("subagent_progress", "worker: finished");
  assert.equal(h.titles().length, 1);
  assert.match(h.titles()[0], /subagents/);
  assert.equal(h.bodies()[0], "worker: finished");
});

test("EOF without the done event follows server history instead of declaring completion", () => {
  assert.match(script, /if \(!finished && !turnError\) \{[\s\S]*?stream ended before completion[\s\S]*?goStale/);
});

test("reconnect does not bind input listeners twice and diagnostics never contain typed text", () => {
  const listeners = [];
  const prompt = { value: "中文 secret", style: { height: "44px" }, addEventListener: (type, fn) => listeners.push({ type, fn }) };
  const elements = { prompt, base: {}, token: {}, inputdiagnostics: {}, inputtrace: { focus() {}, select() {} } };
  const context = vm.createContext({
    $: (id) => elements[id], localStorage: { getItem: () => "" }, showStored() {}, syncStopButton() {},
    document: { activeElement: prompt, addEventListener() {} },
    window: { innerHeight: 600, addEventListener() {} }, navigator: { userAgent: "WebView-test" },
  });
  const start = script.indexOf("function createInputTrace(");
  const end = script.indexOf('  $("send").onclick', start);
  vm.runInContext(script.slice(start, end) + "}\nwire(); wire();", context);
  assert.equal(listeners.filter((item) => item.type === "input").length, 1);
  const input = listeners.find((item) => item.type === "input").fn;
  for (let i = 0; i < 100; i++) input({ type: "input", data: "secret", key: "中", isComposing: true, inputType: "insertCompositionText" });
  elements.inputdiagnostics.onclick();
  const report = JSON.parse(elements.inputtrace.value);
  assert.equal(report.events.length, 80);
  assert.equal(report.events[0].composing, true);
  assert.equal(report.events[0].focused, true);
  assert.equal(report.events[0].length, prompt.value.length);
  assert.doesNotMatch(elements.inputtrace.value, /secret|中文/);
});

test("failed cached tool result is not displayed as success and truncated previews are labeled", () => {
  const h = streamHarness();
  h.emit("tool", { id: "t", name: "read_file", phase: "started", arguments: "{", arguments_truncated: true });
  h.emit("tool", { id: "t", name: "read_file", phase: "finished", ok: false, cached: true, output: "failed", output_truncated: true });
  assert.match(h.titles()[0], /^! read_file · cached$/);
  assert.match(h.bodies()[0], /full output in history/);
  assert.doesNotMatch(h.pending(), /Running/);
});
