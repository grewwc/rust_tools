// Run with: node --test src/bin/ai/serve/app_git.test.cjs
// Execute the real composer commands, API and send path with controlled HTTP
// responses. Checkpoint commands must never become model turns.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const plain = (value) => JSON.parse(JSON.stringify(value));
function between(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page functions: " + start);
  return script.slice(a, b);
}
const marked = (name) => between("// BEGIN " + name, "// END " + name);
const settle = () => new Promise((resolve) => setImmediate(resolve));
const response = (body, status = 200) => ({
  ok: status >= 200 && status < 300, status, statusText: "Unavailable",
  json: async () => plain(body),
});
const checkpoint = (output = "On branch main\n", confirmation = null) => ({
  output, confirmation, checkpoints: ["before"], branches: ["main"], checkpoint_status: "main@before",
});

function harness({ sid = "s1", images = [], respond } = {}) {
  const document = { activeElement: null, querySelectorAll: () => [] };
  function node(tag = "div") {
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, style: {},
      hidden: false, disabled: false, value: "", textContent: "",
      classList: {
        add: (...names) => names.forEach((name) => classes.add(name)),
        remove: (...names) => names.forEach((name) => classes.delete(name)),
        contains: (name) => classes.has(name),
      },
      set className(value) { classes.clear(); value.split(/\s+/).forEach((name) => classes.add(name)); },
      appendChild(child) { child.remove(); child.parentNode = el; el.children.push(child); return child; },
      insertBefore(child, before) {
        child.remove(); child.parentNode = el;
        el.children.splice(before ? el.children.indexOf(before) : el.children.length, 0, child);
        return child;
      },
      remove() {
        if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((child) => child !== el);
        el.parentNode = null;
      },
      focus() { document.activeElement = el; },
      addEventListener(type, listener) { el["on" + type] = listener; },
      querySelectorAll() { return []; },
      setAttribute(name, value) { el.attrs[name] = String(value); },
      removeAttribute(name) { delete el.attrs[name]; },
    };
    return el;
  }
  document.createElement = node;
  const elements = {};
  for (const id of ["prompt", "msgs", "omodel", "oagent", "oeffort", "gitconfirmwrap", "gitconfirmpanel", "gitconfirmtext", "gitcancel", "gitaccept", "cmdmenu", "cmdoptions", "cmdempty"]) {
    elements[id] = node();
  }
  elements.gitconfirmwrap.hidden = true;
  elements.cmdmenu.hidden = true;
  elements.cmdempty.hidden = true;
  elements.cmdmenu.appendChild(elements.cmdoptions);
  elements.cmdmenu.appendChild(elements.cmdempty);
  elements.gitconfirmwrap.appendChild(elements.gitconfirmpanel);
  for (const id of ["gitconfirmtext", "gitcancel", "gitaccept"]) elements.gitconfirmpanel.appendChild(elements[id]);
  elements.prompt.focus();
  const requests = [], messages = [], clears = [], carried = [], notes = [], stored = new Map(), drafts = new Map();
  const context = vm.createContext({
    document, $: (id) => elements[id], TextDecoder,
    sessionId: sid, pendingNew: null, store: { base: "https://a.test", token: "test-token" },
    localStorage: { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    fetch: async (url, opts = {}) => {
      const request = { path: new URL(url).pathname, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null, headers: opts.headers };
      requests.push(request);
      if (respond) {
        const result = await respond(request);
        if (result) return result;
      }
      if (request.path === "/sessions") return response({ id: "fresh" });
      if (request.path.endsWith("/checkpoints")) return response(checkpoint());
      if (request.path.endsWith("/cd")) return response({ stdout: "Working directory: /tmp\n", stderr: "" });
      assert.ok(request.path.endsWith("/turns/stream"), "unexpected request: " + request.path);
      let read = false;
      return { ...response({}), body: { getReader: () => ({
        read: async () => read ? { done: true } : (read = true, { done: false, value: Buffer.from("event: done\n\n") }),
        cancel() {},
      }) } };
    },
    addMsg: (role, text, cls) => { messages.push({ role, text, cls }); return node(); },
    setPromptValue: (value) => { elements.prompt.value = value; },
    clearDraft: (id) => { clears.push(id); drafts.delete(id); },
    carrySkillPick: (from, to) => carried.push({ from, to }), carryTeamPick() {},
    note: (text) => notes.push(text), shortLine: (text, length) => text.slice(0, length),
    scrollToLatest() {}, syncStopButton() {}, stopStaleFollow() {}, retryHistory() {},
    persistPicks() {}, ensureTeamConfig: async () => true, isTurnRunning: () => false,
    validateSkillInput: () => true, captureSkillPick: () => ({ skills: [] }), consumeSkillPick() {},
    sendBusy: false, promptComposing: false, pendingImages: images,
    turns: new Map(), MAX_LIVE_TURNS: 4, historySession: sid, canonicalHistoryTotal: 0,
    pendingUserInputs: new Map(), questionCache: null, runEpoch: 0, turnRunning: false, confirmPollSeq: 0,
    eventActive: false, eventSession: "", eventStreamEnded: false, eventStreamFailed: false,
    turnRunningPid: 0, endedTurnPid: 0,
    staleSession: "", staleDone: false, lastHistoryEndsUnanswered: false,
    LIVE_STALL_MS: 60000, STREAM_SILENCE_MS: 60000,
    isPhoneLayout: () => false,
    addLiveUserBubble: () => { const row = node(), bubble = node(); row.appendChild(bubble); elements.msgs.appendChild(row); return bubble; },
    addMsgMd: (_, parent) => parent.appendChild(node()),
    createLiveActivity: () => ({ label: () => "Connecting", setPhase() {}, finish() {}, anchor() {} }),
    shouldFollowOutput: () => false, followOutput() {}, decorateAndLoad() {},
    formatElapsed: () => "0s", replyPaintDelay: () => 0, renderMarkdown: (text) => text,
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    loadInfo: async () => {}, loadSessions: async () => {}, loadSessionConfig: async () => {},
  });
  for (const code of [
    between("// BEGIN execution model", "// END execution model"),
    between("function canSend()", "// The one send/stop button"),
    between("function headers(", "const esc ="),
    between("// Slash-command completion.", "async function loadInfo()"),
    marked("composer cd command"), marked("composer git command"), marked("turn session preparation"),
    between("function parseBlock(raw)", "// `/cd` changes"),
    between("const RATE_STORAGE =", "function addFold("),
    between("function addFold(", "// Create a fold only"),
    between("function setAnswerRate(", "// Only completed answer bubbles"),
    marked("composer send"),
  ]) vm.runInContext(code, context);
  const folds = () => elements.msgs.children.filter((el) => el.classList.contains("tool-activity"));
  const key = (value, extra = {}) => {
    const ev = { key: value, prevented: false, stopped: false,
      preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...extra };
    elements.gitconfirmpanel.onkeydown(ev);
    return ev;
  };
  const send = (text) => {
    elements.prompt.value = text;
    drafts.set(context.sessionId, text);
    return context.send();
  };
  return { context, document, elements, requests, messages, clears, carried, notes, stored, drafts, folds, key, send };
}

for (const command of ["/git tree", "/git status"]) test(command + " posts a checkpoint command and renders all output without a turn", async () => {
  const output = "* <before> (main)\n" + "    checkpoint <older>\n".repeat(150) + "* <older> (side)\n";
  const h = harness({ respond: () => response(checkpoint(output)) });
  await h.send(command);
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].path, "/sessions/s1/checkpoints");
  assert.equal(h.requests[0].method, "POST");
  assert.deepEqual(h.requests[0].body, { command, confirmed: false });
  assert.equal(h.requests[0].headers.Authorization, "Bearer test-token");
  assert.deepEqual(h.messages, [{ role: "user", text: command, cls: "user rewindable" }]);
  const fold = h.folds()[0];
  assert.equal(fold.children[0].children[1].textContent, output);
  assert.equal(fold.children[0].children[0].textContent, "activity log (1) · * <older> (side)");
  assert.equal(fold.classList.contains("git-output"), true);
  assert.equal(h.elements.prompt.value, "");
  assert.equal(h.elements.prompt.style.height, "auto");
  assert.equal(h.drafts.has("s1"), false);
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.context.sendBusy, false);
});

test("bare /git and :git use the checkpoint endpoint", async () => {
  for (const text of ["/git", "  :git list  "]) {
    const h = harness();
    await h.send(text);
    assert.deepEqual(h.requests.map((r) => r.body), [{ command: text.trim(), confirmed: false }]);
    assert.equal(h.requests[0].path, "/sessions/s1/checkpoints");
  }
});

test("confirmation displays server text and resends the same command only on Confirm", async () => {
  const prompt = "Discard 3 turns?\nThis restores '<before>'. (y/n): ";
  const command = '/git checkout --discard "before"';
  const h = harness({ respond: (request) => response(request.body.confirmed ? checkpoint("Checked out 'before'.\n") : checkpoint("Comparing live history.\n", prompt)) });
  const sending = h.send(command);
  await settle();
  assert.equal(h.requests.length, 1);
  assert.equal(h.elements.gitconfirmwrap.hidden, false);
  assert.equal(h.elements.gitconfirmtext.textContent, prompt);
  assert.equal(h.document.activeElement, h.elements.gitaccept);
  assert.equal(h.context.sendBusy, true);
  await h.context.send();
  assert.equal(h.requests.length, 1, "the modal command prevents a concurrent send");
  h.elements.gitaccept.onclick();
  await sending;
  assert.deepEqual(h.requests.map((r) => r.body), [{ command, confirmed: false }, { command, confirmed: true }]);
  assert.ok(h.requests.every((r) => r.path === "/sessions/s1/checkpoints"));
  assert.equal(h.folds().at(-1).children[0].children[1].textContent, "Checked out 'before'.\n");
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.document.activeElement, h.elements.prompt);
  assert.equal(h.context.sendBusy, false);
});

for (const cancel of ["button", "Escape", "scrim"]) test("canceling via " + cancel + " does not resend", async () => {
  const h = harness({ respond: () => response(checkpoint("", "Clear all checkpoints?")) });
  const sending = h.send("/git clear");
  await settle();
  if (cancel === "button") h.elements.gitcancel.onclick();
  else if (cancel === "Escape") {
    const ev = h.key("Escape");
    assert.equal(ev.prevented, true);
    assert.equal(ev.stopped, true);
  } else h.elements.gitconfirmwrap.onclick({ target: h.elements.gitconfirmwrap });
  await sending;
  assert.equal(h.requests.length, 1);
  assert.ok(h.messages.some((m) => m.role === "asst" && m.text === "Canceled: nothing was changed."));
  assert.equal(h.folds().length, 0);
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.document.activeElement, h.elements.prompt);
  assert.equal(h.context.sendBusy, false);
});

test("the dialog traps Tab, ignores IME and repeated Enter, and confirms with Enter", async () => {
  const h = harness({ respond: (r) => response(checkpoint(r.body.confirmed ? "Cleared.\n" : "", r.body.confirmed ? null : "Clear?")) });
  const sending = h.send("/git clear");
  await settle();
  assert.equal(h.key("Tab").prevented, true);
  assert.equal(h.document.activeElement, h.elements.gitcancel);
  h.key("Tab", { shiftKey: true });
  assert.equal(h.document.activeElement, h.elements.gitaccept);
  h.key("Enter", { isComposing: true });
  h.key("Enter", { keyCode: 229 });
  h.key("Enter", { repeat: true });
  assert.equal(h.elements.gitconfirmwrap.hidden, false);
  assert.equal(h.requests.length, 1);
  const ev = h.key("Enter");
  assert.equal(ev.prevented, true);
  assert.equal(ev.stopped, true);
  await sending;
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[1].body.confirmed, true);
  assert.equal(h.document.activeElement, h.elements.prompt);
});

test("Enter on Cancel cancels instead of confirming", async () => {
  const h = harness({ respond: () => response(checkpoint("", "Clear all checkpoints?")) });
  const sending = h.send("/git clear");
  await settle();
  h.key("Tab");
  assert.equal(h.document.activeElement, h.elements.gitcancel);
  h.key("Enter");
  await sending;
  assert.equal(h.requests.length, 1, "Cancel must not resend the command");
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.messages.at(-1).text, "Canceled: nothing was changed.");
});

test("a first /git creates and safely adopts a lazy session", async () => {
  const h = harness({ sid: "" });
  await h.send("/git list");
  assert.deepEqual(h.requests.map((r) => r.path), ["/sessions", "/sessions/fresh/checkpoints"]);
  assert.equal(h.context.sessionId, "fresh");
  assert.equal(h.context.pendingNew, "fresh");
  assert.equal(h.stored.get("serve.session"), "fresh");
  assert.deepEqual(h.carried, [{ from: "", to: "fresh" }]);
  assert.ok(h.clears.includes("fresh"));
  assert.equal(h.elements.prompt.value, "");
});

test("a session switch during lazy creation does not adopt or execute the old command", async () => {
  let finishCreate;
  const h = harness({ sid: "", respond: () => new Promise((resolve) => { finishCreate = resolve; }) });
  const sending = h.send("/git list");
  await settle();
  h.context.sessionId = "other";
  h.elements.prompt.value = "other draft";
  finishCreate(response({ id: "fresh" }));
  await sending;
  assert.equal(h.context.sessionId, "other");
  assert.equal(h.elements.prompt.value, "other draft");
  assert.equal(h.requests.length, 1);
  assert.deepEqual(h.messages, []);
});

for (const stage of ["probe", "confirmed"]) test("server errors during " + stage + " are visible assistant errors", async () => {
  const h = harness({ respond: (r) => stage === "confirmed" && !r.body.confirmed
    ? response(checkpoint("", "Discard history?"))
    : response({ error: "checkpoint helper failed" }, 500) });
  const sending = h.send("/git checkout --discard before");
  if (stage === "confirmed") { await settle(); h.elements.gitaccept.onclick(); }
  await sending;
  const err = h.messages.find((m) => m.cls === "asst err");
  assert.equal(err.role, "err");
  assert.match(err.text, /500 checkpoint helper failed/);
  assert.equal(h.folds().length, 0);
  assert.equal(h.elements.gitconfirmwrap.hidden, true);
  assert.equal(h.context.sendBusy, false);
});

test("failed session creation preserves the composer and reports the server error", async () => {
  const h = harness({ sid: "", respond: () => response({ error: "cannot create session" }, 503) });
  await h.send("/git list");
  assert.equal(h.requests.length, 1);
  assert.equal(h.context.sessionId, "");
  assert.equal(h.elements.prompt.value, "/git list");
  assert.ok(h.messages.some((m) => m.cls === "asst err" && /cannot create session/.test(m.text)));
  assert.equal(h.context.sendBusy, false);
});

test("plain text, multiline input and longer command names still send normal turns", async () => {
  for (const text of ["  hello  ", "/git list\nand explain it", ":git list\r\nthen help", "/git list\n", "/github", ":gitx", "/git-status"]) {
    const h = harness();
    await h.send(text);
    assert.equal(h.requests.length, 1, JSON.stringify(text));
    assert.equal(h.requests[0].path, "/sessions/s1/turns/stream");
    assert.equal(h.requests[0].body.prompt, text);
    assert.equal(h.requests[0].body.confirm, true);
    assert.equal(h.messages.some((m) => m.cls === "asst err"), false);
    assert.equal(h.elements.gitconfirmwrap.hidden, true);
  }
});

test("/cd still uses its own session endpoint and transcript fold", async () => {
  const h = harness();
  await h.send("/cd /tmp");
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].path, "/sessions/s1/cd");
  assert.deepEqual(h.requests[0].body, { command: "/cd /tmp" });
  assert.equal(h.folds()[0].children[0].children[1].textContent, "Working directory: /tmp");
  assert.equal(h.folds()[0].classList.contains("git-output"), false);
});

test("a single-line /git stays local with pending images and preserves the attachments", async () => {
  const images = [{ filename: "photo.png", data: "image" }];
  const h = harness({ images });
  await h.send("/git list");
  assert.equal(h.requests[0].path, "/sessions/s1/checkpoints");
  assert.equal(h.requests.length, 1);
  assert.deepEqual(plain(h.context.pendingImages), images);
});

test("navigation during a confirmation cannot resend against another session or server", async () => {
  for (const navigate of [(ctx) => { ctx.sessionId = "other"; }, (ctx) => { ctx.store.base = "https://other.test"; }]) {
    const h = harness({ respond: () => response(checkpoint("", "Discard?")) });
    const sending = h.send("/git checkout --discard before");
    await settle();
    navigate(h.context);
    h.elements.gitaccept.onclick();
    await sending;
    assert.equal(h.requests.length, 1);
    assert.equal(h.context.sendBusy, false);
  }
});

test("the checkpoint dialog is an in-app accessible surface with no native confirmation", () => {
  assert.match(html, /id="gitconfirmwrap" hidden/);
  assert.match(html, /id="gitconfirmpanel" role="alertdialog" aria-modal="true" aria-labelledby="gitconfirmtitle" aria-describedby="gitconfirmtext"/);
  assert.doesNotMatch(marked("composer git command"), /\b(?:window\.)?confirm\s*\(/);
});
