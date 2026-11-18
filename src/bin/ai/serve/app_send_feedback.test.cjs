// Run with: env -u NODE_OPTIONS node --test src/bin/ai/serve/app_send_feedback.test.cjs
// Exercise the real click, preparation and stream path with controlled responses.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const filename = path.join(__dirname, "app_phase2_baseline.test.cjs");
const { sendHarness, settle } = new vm.Script(fs.readFileSync(filename, "utf8")
  + "\n;({ sendHarness, settle });", { filename }).runInNewContext({
  require: (name) => name === "node:test" ? { test() {} } : require(name),
  __dirname, __filename: filename, console, Buffer, process,
  TextEncoder, TextDecoder, AbortController, URL, URLSearchParams,
  setTimeout, clearTimeout, setInterval, clearInterval, setImmediate,
});
const html = fs.readFileSync(process.env.ASERVE_TEST_APP_HTML || path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function harness() {
  const h = sendHarness();
  h.elements.prompt.value = "未完成直接给我结果了？为什么不继续？";
  h.c.window.matchMedia = (query) => ({ matches: query.includes("min-width:701px"), addEventListener() {} });
  const start = script.indexOf("async function stopTurn()"), end = script.indexOf("function parseBlock(raw)", start);
  assert.ok(start >= 0 && end > start, "the stop branch is loaded from the actual page");
  vm.runInContext(script.slice(start, end), h.c);
  h.c.syncStopButton();
  return h;
}
function assertControl(h, blocked = !h.c.canSend()) {
  const button = h.elements.send;
  assert.equal(button.disabled, blocked, "the visible button follows the authoritative admission predicate");
  assert.equal(button.getAttribute("aria-disabled"), String(blocked));
  if (blocked) assert.equal(button.title, button.getAttribute("aria-label"), "the blocked control describes its action");
}
function assertRejected(h, draft, { blocked = !h.c.canSend(), requests = 0 } = {}) {
  assert.equal(h.elements.prompt.value, draft, "a rejected attempt preserves the draft");
  assert.equal(h.requests.length, requests, "a rejected attempt cannot start a model turn");
  assertControl(h, blocked);
}
async function finish(h) {
  await h.accept();
  await h.finish(() => { h.reader.frame('event: done\ndata: {}\n\n'); h.reader.eof(); });
  await h.dispose();
}

test("an idle send click starts the ordinary turn and clears its draft", async () => {
  const h = harness(), draft = h.elements.prompt.value;
  h.elements.omodel.value = "selected-model";
  const submitted = h.elements.send.onclick();
  await settle();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].body.prompt, draft);
  assert.equal(h.requests[0].body.model, "selected-model");
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.c.executionState().live, true);
  await h.accept();
  h.reader.frame('event: done\ndata: {}\n\n');
  h.reader.eof();
  await submitted;
  await h.dispose();
});

test("the live-turn limit disables sending before and after a guarded click", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  for (let i = 0; i < c.MAX_LIVE_TURNS; i++) c.turns.set("other-" + i, {});
  c.syncStopButton();
  assert.equal(h.elements.send.textContent, "➤", "other sessions do not change this session's send glyph");
  assertControl(h, true);
  await h.elements.send.onclick();
  await settle();
  assertRejected(h, draft);
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), false);
  c.turns.clear();
  c.syncStopButton();
  assertControl(h, false);
  await h.start(draft);
  assert.equal(h.requests[0].body.prompt, draft);
  await finish(h);
});

test("a second click cannot clear the first attempt's preparation lock", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  c.sendBusy = true;
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(c.sendBusy, true, "the active attempt still owns its lock");
  c.sendBusy = false;
  assertControl(h, false);
  await h.dispose();
});

for (const blocked of ["remote", "stopping", "interrupt rejection"]) test("the send button's " + blocked + " stop branch preserves the draft", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  if (blocked === "remote") c.updateExecutionModel({ kind: "remote", active: true, pid: "4242", at: c.Date.now() });
  else { c.turnRunning = true; c.turnRunningAt = c.Date.now(); }
  if (blocked === "stopping") c.stopBusy = true;
  if (blocked === "interrupt rejection") c.api = async () => { throw new Error("interrupt rejected"); };
  c.syncStopButton();
  assert.equal(c.canStop(), true);
  await h.elements.send.onclick();
  assertRejected(h, draft,
    { blocked: blocked !== "interrupt rejection" });
  if (blocked === "stopping") assert.equal(c.stopBusy, true, "the active interrupt retains its own lock");
  else assert.equal(c.stopBusy, false);
  assert.equal(c.canStop(), true, "a refused action cannot discard evidence of a running turn");
  await h.dispose();
});

for (const stage of ["command", "session config"]) test("a " + stage + " preparation exception preserves the input and retry succeeds", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  const images = c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
  const name = stage === "command" ? "runCdCommand" : "ensureTeamConfig", original = c[name];
  c[name] = async () => { throw new Error("preparation disconnected"); };
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(c.pendingImages, images);
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), true);
  c[name] = original;
  await h.start(draft);
  assert.match(h.requests[0].body.prompt, /\[\[image:draft\.png\]\]/);
  assert.deepEqual(JSON.parse(JSON.stringify(h.requests[0].body.images)), images);
  await finish(h);
});

test("a refused mode save preserves the draft until config is saved", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  const team = { pending: true, saving: null, errorSource: "" };
  c.teamPickState = () => team;
  c.ensureTeamConfig = async () => { team.errorSource = "save"; return false; };
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), false);
  team.errorSource = "";
  c.ensureTeamConfig = async () => true;
  c.syncStopButton();
  assertControl(h, false);
  await h.start(draft);
  await finish(h);
});

for (const change of ["draft", "session", "server"]) test("a " + change + " change during preparation preserves the input and its owner", async () => {
  const h = harness(), c = h.c;
  let resolve;
  c.ensureTeamConfig = () => new Promise((done) => { resolve = done; });
  const pending = h.elements.send.onclick();
  await settle();
  assertControl(h, true);
  if (change === "session") c.switchSession("next-session", false);
  if (change === "server") { c.store.base = "https://other-server.test"; c.syncStopButton(); }
  h.elements.prompt.value = "updated draft";
  resolve(true);
  await pending;
  if (change === "draft") assertRejected(h, "updated draft");
  else {
    assert.equal(h.elements.prompt.value, "updated draft");
    assert.equal(h.requests.length, 0);
    assertControl(h, false);
  }
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), true);
  await h.dispose();
});

test("an unexpected selector exception is caught by the click handler before losing attachments", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  const images = c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
  delete h.elements.omodel;
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(c.pendingImages, images);
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), true);
  await h.dispose();
});

for (const blocked of ["empty", "composition", "skill composition"]) test("a " + blocked + " send attempt preserves the draft", async () => {
  const h = harness(), c = h.c;
  if (blocked === "empty") h.elements.prompt.value = " \t\n ";
  if (blocked === "composition") c.promptComposing = true;
  if (blocked === "skill composition") { c.skillInputReady = () => false; c.validateSkillInput = () => false; }
  const draft = h.elements.prompt.value;
  c.syncStopButton();
  assertControl(h, true);
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(c.sendBusy, false);
  assert.equal(c.canSend(), false);
  await h.dispose();
});

test("live turns on other servers count toward the same visible limit", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  for (let i = 0; i < 5; i++) c.updateExecutionModel({ kind: "live", entry: {} }, "other-" + i, "https://server-" + i + ".test");
  assert.equal(c.executionState().liveCount, 5);
  assertControl(h, true);
  await h.elements.send.onclick();
  assertRejected(h, draft);
  for (let i = 0; i < 2; i++) c.updateExecutionModel({ kind: "live", entry: null }, "other-" + i, "https://server-" + i + ".test");
  assertControl(h, false);
  await h.start(draft);
  await finish(h);
});

for (const changed of ["limit", "local turn", "remote turn", "composition", "skill composition"]) {
  test("preparation rechecks " + changed + " after waiting for config", async () => {
    const h = harness(), c = h.c, draft = h.elements.prompt.value;
    let resolve;
    c.ensureTeamConfig = () => new Promise((done) => { resolve = done; });
    const sending = h.elements.send.onclick();
    await settle();
    assertControl(h, true);
    if (changed === "limit") for (let i = 0; i < 4; i++) c.updateExecutionModel({ kind: "live", entry: {} }, "other-" + i);
    if (changed === "local turn") { c.turnRunning = true; c.turnRunningAt = c.Date.now(); }
    if (changed === "remote turn") c.updateExecutionModel({ kind: "remote", active: true, pid: "4242", at: c.Date.now() });
    if (changed === "composition") c.promptComposing = true;
    if (changed === "skill composition") c.skillInputReady = () => false;
    resolve(true);
    await sending;
    assertRejected(h, draft, { blocked: changed !== "local turn" });
    assert.equal(c.sendBusy, false);
    if (changed === "local turn") assert.equal(h.elements.send.textContent, "⏹", "a local turn retains its stop action");
    await h.dispose();
  });
}

for (const mismatch of ["session", "server"]) test("an inconsistent " + mismatch + " preparation owner preserves the draft", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value, prepare = c.prepareTurnSession;
  c.prepareTurnSession = (sid, base, ...rest) => prepare(mismatch === "session" ? "obsolete" : sid,
    mismatch === "server" ? "https://obsolete.test" : base, ...rest);
  await h.elements.send.onclick();
  assertRejected(h, draft);
  await h.dispose();
});

test("an empty preparation result preserves the draft", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  c.prepareTurnSession = async () => "";
  await h.elements.send.onclick();
  assertRejected(h, draft);
  await h.dispose();
});

test("a transient blocker preserves the draft until a later send", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value, original = c.canSend;
  let blocked = true;
  c.canSend = () => {
    const allowed = !blocked;
    blocked = false;
    return allowed;
  };
  await h.elements.send.onclick();
  assertRejected(h, draft, { blocked: false });
  c.canSend = original;
  await h.start(draft);
  await finish(h);
});

function loadCommandBridges(h) {
  for (const name of ["composer cd command", "composer export/import commands"]) {
    const start = script.indexOf("// BEGIN " + name), end = script.indexOf("// END " + name, start);
    assert.ok(start >= 0 && end > start);
    vm.runInContext(script.slice(start, end), h.c);
  }
  h.c.shortLine = (text, length) => text.slice(0, length);
  h.c.addFold = () => ({ box: h.document.createElement("div") });
}

for (const command of ["ordinary turn", "/cd /tmp", "/git status"]) {
  test("failed session adoption retains " + command, async () => {
    const h = harness(), c = h.c;
    loadCommandBridges(h);
    c.switchSession("", false);
    h.elements.prompt.value = command;
    let resolve;
    c.api = () => new Promise((done) => { resolve = done; });
    c.adoptComposerSession = () => false;
    const sending = h.elements.send.onclick();
    await settle();
    assertControl(h, true);
    resolve({ json: async () => ({ id: "created", cwd: "/srv" }) });
    await sending;
    assertRejected(h, command);
    assert.equal(c.sessionId, "");
    await h.dispose();
  });
}

test("a selector error after adopting a lazy session preserves the original draft", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  c.switchSession("", false);
  h.elements.prompt.value = draft;
  c.api = async () => ({ json: async () => ({ id: "created", cwd: "/srv" }) });
  delete h.elements.omodel;
  await h.elements.send.onclick();
  assert.equal(c.sessionId, "created");
  assertRejected(h, draft);
  await h.dispose();
});

test("a partial session adoption exception preserves the draft and attachments", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  c.switchSession("", false);
  h.elements.prompt.value = draft;
  const images = c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
  c.api = async () => ({ json: async () => ({ id: "created", cwd: "/srv" }) });
  c.carryTeamPick = () => { throw new Error("adoption configuration failed"); };
  await h.elements.send.onclick();
  assert.equal(c.sessionId, "created");
  assertRejected(h, draft);
  assert.equal(c.pendingImages, images);
  assert.equal(c.sendBusy, false);
  await h.dispose();
});

for (const command of ["/cd /tmp", "/git status", "/import", "/model selected-model"]) {
  test("the live-turn cap preserves local command handling for " + command, async () => {
    const h = harness(), c = h.c, calls = [];
    loadCommandBridges(h);
    h.elements.omodel.options = [{ value: "selected-model" }];
    h.elements.omodel.value = "selected-model";
    c.pickTransferArchive = async () => null;
    c.api = async (path) => {
      calls.push(path);
      return { ok: true, json: async () => ({ stdout: "done", cwd: "/tmp", output: "done" }) };
    };
    for (let i = 0; i < 4; i++) c.updateExecutionModel({ kind: "live", entry: {} }, "other-" + i);
    h.elements.prompt.value = command;
    c.syncStopButton();
    assert.equal(c.canSend(), true, "a command that does not start a model turn needs no live slot");
    assertControl(h, false);
    await h.elements.send.onclick();
    assert.equal(h.requests.length, 0);
    assert.equal(c.sendBusy, false);
    if (command.startsWith("/cd")) assert.ok(calls.some((path) => path.endsWith("/cd")));
    if (command.startsWith("/git")) assert.ok(calls.some((path) => path.endsWith("/checkpoints")));
    await h.dispose();
  });
}

for (const command of ["/cd /tmp", "/model selected-model\nquestion"]) {
  test("a command that would start a turn respects the live cap: " + JSON.stringify(command), async () => {
    const h = harness(), c = h.c;
    loadCommandBridges(h);
    h.elements.omodel.options = [{ value: "selected-model" }];
    if (command.startsWith("/cd")) c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
    for (let i = 0; i < 4; i++) c.updateExecutionModel({ kind: "live", entry: {} }, "other-" + i);
    h.elements.prompt.value = command;
    c.syncStopButton();
    assertControl(h, true);
    await h.elements.send.onclick();
    assertRejected(h, command);
    await h.dispose();
  });
}

for (const command of ["/import as=", "/export last", "/export"]) test("unusable transfer command " + command + " retains the input", async () => {
  const h = harness(), c = h.c;
  loadCommandBridges(h);
  if (command === "/export") c.switchSession("", false);
  h.elements.prompt.value = command;
  await h.elements.send.onclick();
  assert.equal(h.elements.prompt.value, command);
  assert.equal(h.requests.length, 0);
  assertControl(h, false);
  await h.dispose();
});

test("whitespace without attachments is blocked, while an image-only draft still sends", async () => {
  const h = harness(), c = h.c;
  h.elements.prompt.value = " \t\n ";
  c.syncStopButton();
  assertControl(h, true);
  c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
  c.syncStopButton();
  assertControl(h, false);
  await h.start("");
  assert.equal(h.requests[0].body.prompt, "[[image:draft.png]]");
  assert.equal(h.requests[0].body.images.length, 1);
  await finish(h);
});

function loadSideNotes(h) {
  const start = script.indexOf("async function sendSideNote()"), end = script.indexOf("let sideNotesSeq =", start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(script.slice(start, end), h.c);
  vm.runInContext('$("sidenote").onclick = sendSideNote;', h.c);
}
for (const blocked of ["busy", "empty", "missing session", "composition"]) test("a " + blocked + " side-note attempt preserves the draft", async () => {
  const h = harness(), c = h.c;
  loadSideNotes(h);
  c.turnRunning = true; c.turnRunningAt = c.Date.now();
  if (blocked === "busy") c.sideNoteBusy = true;
  if (blocked === "empty") h.elements.prompt.value = " \t\n ";
  if (blocked === "missing session") { c.switchSession("", false); h.elements.prompt.value = "note"; }
  if (blocked === "composition") c.promptComposing = true;
  c.syncStopButton();
  assert.equal(h.elements.sidenote.disabled, true);
  const draft = h.elements.prompt.value;
  await h.elements.sidenote.onclick();
  assertRejected(h, draft, { blocked: false });
  await h.dispose();
});

for (const outcome of ["queued", "rejected", "session switched"]) test("a side note scopes its " + outcome + " result", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  loadSideNotes(h);
  c.turnRunning = true; c.turnRunningAt = c.Date.now();
  let resolve, reject;
  c.api = () => new Promise((yes, no) => { resolve = yes; reject = no; });
  const sending = h.elements.sidenote.onclick();
  assert.equal(h.elements.sidenote.disabled, true);
  assertControl(h, false);
  if (outcome === "session switched") { c.switchSession("next-session", false); h.elements.prompt.value = "next draft"; }
  if (outcome === "rejected") reject(new Error("offline"));
  else resolve({ json: async () => ({ notes: [] }) });
  await sending;
  assert.equal(c.sideNoteBusy, false);
  if (outcome === "session switched") {
    assert.equal(h.elements.prompt.value, "next draft");
  } else if (outcome === "rejected") assertRejected(h, draft, { blocked: false });
  else {
    assert.equal(h.elements.prompt.value, "");
  }
  await h.dispose();
});

for (const [name, draft] of [
  ["plain text", '  先检查 <img src=x onerror=alert(1)> & "引号"\n\t保留草稿  '],
  ["long Unicode text", "第一行\n\t" + "🧭".repeat(600) + "\nHIDDEN_TAIL"],
]) test("the side-note queue preserves the complete " + name, async (t) => {
  const h = harness(), c = h.c;
  t.after(async () => { c.renderSideNotes([]); await h.dispose(); });
  loadSideNotes(h);
  h.elements.prompt.value = draft;
  h.sideNotes.push({ from: "user", content: draft.trim() });
  await h.elements.sidenote.onclick();
  await settle();
  const content = h.elements.sidequeuelist.querySelector(".side-note-text");
  assert.equal(h.elements.sidequeue.hidden, false);
  assert.equal(content.textContent, draft.trim(), "the queue retains the full note");
  assert.equal(content.parentNode.title, draft.trim(), "the tooltip retains the same full note");
  assert.equal(content.children.length, 0, "note markup cannot create elements in the queue");
});

test("refreshing and draining a visible queue preserves composer focus", async (t) => {
  const h = harness(), c = h.c;
  t.after(async () => { c.renderSideNotes([]); await h.dispose(); });
  const notes = [{ from: "user", content: "keep this note readable" }];
  h.elements.prompt.focus();
  c.renderSideNotes(notes);
  assert.equal(h.elements.sidequeuelist.querySelector(".side-note-text").textContent, notes[0].content);
  assert.equal(h.document.activeElement, h.elements.prompt, "refreshing the queue preserves composer focus");
  c.renderSideNotes(notes.concat({ from: "agent", content: "another note" }));
  c.renderSideNotes([]);
  assert.equal(h.elements.sidequeue.hidden, true);
});

test("a blocked main-composer click preserves an existing side-note queue", async (t) => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  t.after(async () => { c.renderSideNotes([]); await h.dispose(); });
  c.renderSideNotes([{ from: "user", content: "keep this queued note" }]);
  c.promptComposing = true;
  c.syncStopButton();
  await h.elements.send.onclick();
  assertRejected(h, draft);
  assert.equal(h.elements.sidequeuelist.querySelector(".side-note-text").textContent, "keep this queued note");
});

test("a stop click blocks duplicate interrupts and keeps the draft", async () => {
  const h = harness(), c = h.c, draft = h.elements.prompt.value;
  c.turnRunning = true; c.turnRunningAt = c.Date.now();
  let resolve;
  c.api = () => new Promise((done) => { resolve = done; });
  c.syncStopButton();
  assertControl(h, false);
  const stopping = h.elements.send.onclick();
  assertControl(h, true);
  await h.elements.send.onclick();
  assertRejected(h, draft, { blocked: true });
  resolve({ json: async () => ({ interrupted: true }) });
  await stopping;
  assert.equal(h.elements.prompt.value, draft);
  assertControl(h, false);
  await h.dispose();
});

for (const draft of ["/audit", "/audit -f", "/audit --deep", "/audit 仔细检查这段修改", "/audit -f --deep 检查边界情况\n保留这行"]) {
  test("ordinary sends preserve the turn prompt " + JSON.stringify(draft), async () => {
    const h = harness();
    h.elements.omodel.value = "selected-model";
    await h.start(draft);
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0].body.prompt, draft);
    assert.equal(h.requests[0].body.model, "selected-model", "the override selector owns the request's model");
    await finish(h);
  });
}
