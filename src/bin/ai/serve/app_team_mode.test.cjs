// Run with: node --test src/bin/ai/serve/app_team_mode.test.cjs
// Execute the marked page functions, including the actual send path and API,
// against a small DOM and controllable session-config responses.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "app.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const plain = (value) => JSON.parse(JSON.stringify(value));
function marked(name) {
  const start = "// BEGIN " + name, end = "// END " + name;
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing marked page functions: " + name);
  return script.slice(a + start.length, b);
}
function existingFunction(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a, "missing page function: " + start);
  return script.slice(a, b);
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const NORMAL = { model: "", agent: "", reasoning_effort: "", default_skills: ["other-skill"], subagent_backend: "", subagent_model: "", output: "Sticky agent-team: active" };
// A kernel session stores the backend explicitly; an empty value means "never
// chosen", which the page reads as its TraeCLI default.
const KERNEL = { ...NORMAL, default_skills: ["other-skill", "AGENT-TEAM"], subagent_backend: "kernel", output: "Sticky agent-team: off" };
const TRAECLI = { ...KERNEL, subagent_backend: "traecli", subagent_model: "worker-model" };
const response = (cfg) => ({ ok: true, json: async () => plain(cfg) });
const turnResponse = (read = async () => ({ done: false, value: Buffer.from("event: done\n\n") })) => ({ ok: true, body: { getReader: () => ({ read }) } });
const settle = () => new Promise((resolve) => setImmediate(resolve));
function nodeList(nodes = []) {
  const list = {
    ...nodes, length: nodes.length,
    item(index) { return this[index] ?? null; },
    forEach(callback, thisArg) { nodes.forEach((node, index) => callback.call(thisArg, node, index, list)); },
    [Symbol.iterator]() { return nodes[Symbol.iterator](); },
  };
  return list;
}

function harness({ sid = "alpha", phone = true, cfg = NORMAL, configPost, configGet, createSession, skills = [], subagentModels, turnPost } = {}) {
  const document = { activeElement: null, querySelectorAll: () => nodeList() };
  function node(tag = "div") {
    let text = "";
    const classes = new Set();
    const el = {
      tagName: tag, children: [], parentNode: null, attrs: {}, dataset: {}, style: {}, value: "", title: "",
      hidden: false, disabled: false, tabIndex: tag === "button" || tag === "input" ? 0 : -1,
      classList: { add: (...names) => names.forEach((n) => classes.add(n)), remove: (...names) => names.forEach((n) => classes.delete(n)), toggle: (name, on) => on ? classes.add(name) : classes.delete(name) },
      set textContent(value) { text = String(value); el.children = []; },
      get textContent() { return text; },
      setAttribute(key, value) { el.attrs[key] = String(value); },
      getAttribute(key) { return el.attrs[key] ?? null; },
      appendChild(child) { child.parentNode = el; el.children.push(child); return child; },
      remove() { if (el.parentNode) el.parentNode.children = el.parentNode.children.filter((n) => n !== el); },
      focus(options) { if (!el.disabled) { document.activeElement = el; el.focusOptions = options; } },
      click() { if (!el.disabled) return el.onclick?.(); },
      contains(target) { for (let cur = target; cur; cur = cur.parentNode) if (cur === el) return true; return false; },
      closest(selector) {
        for (let cur = el; cur; cur = cur.parentNode) {
          if (selector.startsWith("#") && cur.id === selector.slice(1)) return cur;
          if (selector === "[hidden]" && cur.hidden) return cur;
          if (selector === '[role="radiogroup"]' && cur.attrs.role === "radiogroup") return cur;
        }
        return null;
      },
      querySelectorAll(selector) {
        const found = [];
        const walk = (parent) => parent.children.forEach((child) => {
          if ((selector === "button, input" && ["button", "input"].includes(child.tagName)) || (selector === '[role="radio"]' && child.attrs.role === "radio")) found.push(child);
          walk(child);
        });
        walk(el);
        return nodeList(found);
      },
    };
    return el;
  }
  document.createElement = node;
  const elements = {};
  for (const match of html.matchAll(/<(div|section|button|span|input|p|label)[^>]*id="(team[^"]*)"[^>]*>/g)) {
    const el = elements[match[2]] = node(match[1]);
    el.id = match[2];
    el.hidden = /\shidden(?:\s|>)/.test(match[0]);
    const role = /role="([^"]+)"/.exec(match[0]);
    if (role) el.attrs.role = role[1];
  }
  const attach = (id, ...children) => children.forEach((child) => elements[id].appendChild(elements[child]));
  attach("teampick", "teampickscrim", "teampickpanel");
  attach("teampickpanel", "teamclose", "teambody", "teamactions");
  attach("teambody", "teammodes", "teamadvanced", "teamerror");
  attach("teammodes", "teamnormal", "teamcollab");
  attach("teamadvanced", "teambackends", "teamworkerwrap");
  attach("teambackends", "teamkernel", "teamtraecli");
  attach("teamworkerwrap", "teamworkerlabel", "teamworkerbtn", "teamworkerlist", "teamworkerhint");
  attach("teamworkerbtn", "teamworkervalue");
  elements.teamworkerbtn.appendChild(node("span"));
  attach("teamworkerlist", "teamworkersearch", "teamworkeroptions", "teamworkerstatus", "teamworkerretry");
  attach("teamactions", "teamcancel", "teamapply");
  attach("teamcontrols", "teamtrigger");
  attach("teamtrigger", "teamlabel", "teamstatus");
  for (const [id, value] of [["teamnormal", "normal"], ["teamcollab", "team"]]) elements[id].dataset.mode = value;
  for (const [id, value] of [["teamkernel", "kernel"], ["teamtraecli", "traecli"]]) elements[id].dataset.backend = value;
  for (const [id, values] of [["omodel", ["", "main-model"]], ["oagent", ["", "build"]], ["oeffort", ["", "high"]]]) {
    const el = elements[id] = node("select");
    el.options = values.map((value) => ({ value }));
  }
  elements.prompt = node("textarea");
  elements.prompt.value = "一起完成这个任务";
  elements.msgs = node();
  const requests = [], notes = [], messages = [], consumed = [], stored = new Map(), listeners = new Map();
  const configs = new Map([[sid || "fresh", plain(cfg)]]);
  const media = { matches: phone, addEventListener: (_, fn) => { media.change = fn; } };
  const history = { pushes: 0, backs: 0, pushState() { this.pushes++; }, back() { this.backs++; queueMicrotask(() => listeners.get("popstate")?.()); } };
  const context = vm.createContext({
    $: (id) => elements[id], document, history, TextDecoder,
    window: { matchMedia: () => media, addEventListener: (name, fn) => listeners.set(name, fn) },
    isPhoneLayout: () => media.matches, PHONE_LAYOUT_QUERY: "(max-width:700px)",
    store: { base: "https://a.test" }, sessionId: sid, pendingNew: null,
    localStorage: { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) },
    headers: () => ({ "Content-Type": "application/json" }),
    note: (text) => notes.push(text), holdLayout: (_, apply) => apply(),
    syncModelTrigger() {}, syncOvrSummary() {}, syncPickTrigger() {},
    retryHistory() {}, pauseHistoryRetry() {},
    persistDraft() { stored.set("draft." + context.sessionId, elements.prompt.value); },
    restoreDraft() { elements.prompt.value = stored.get("draft." + context.sessionId) || ""; },
    restoreSkillPick() {}, carrySkillPick() {}, resetQuestionNavigator() {},
    sendBusy: false, promptComposing: false, pendingImages: [],
    turns: new Map(), MAX_LIVE_TURNS: 4, historySession: sid, canonicalHistoryTotal: 0,
    pendingUserInputs: new Map(), questionCache: null, runEpoch: 0, turnRunning: false,
    turnRunningAt: 0, RUNNING_SAMPLE_TTL_MS: 60000,
    eventActive: false, eventSession: "", eventStreamEnded: false, eventStreamFailed: false, staleSession: "", staleDone: false,
    confirmPollSeq: 0, lastHistoryEndsUnanswered: false, LIVE_STALL_MS: 60000,
    validateSkillInput: () => true,
    captureSkillPick: () => ({ skills }), consumeSkillPick: (pick) => consumed.push(pick),
    addMsg: (role, text) => { messages.push({ role, text }); return node(); },
    addLiveUserBubble: () => { const parent = node(), bubble = node(); parent.appendChild(bubble); return bubble; },
    addMsgMd: (_, parent) => parent.appendChild(node()),
    createLiveActivity: () => ({ label: () => "Connecting", setPhase() {}, finish() {} }),
    setPromptValue: (value) => { elements.prompt.value = value; }, clearDraft() {},
    syncStopButton() {}, scrollToLatest() {}, stopStaleFollow() {}, abortEvents() {},
    shouldFollowOutput: () => false, followOutput() {}, decorateAndLoad() {},
    formatElapsed: () => "0s", replyPaintDelay: () => 0, renderMarkdown: (s) => s,
    parseBlock: () => ({ kind: "done", payload: "" }),
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    loadInfo: async () => {}, loadSessions: async () => {},
    fetch: async (url, opts = {}) => {
      const request = { url, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : undefined };
      requests.push(request);
      const pathname = new URL(url).pathname;
      if (pathname === "/sessions") return createSession ? createSession(request) : response({ id: "fresh" });
      if (pathname === "/subagent-models") return subagentModels ? subagentModels(request) : response({ models: ["worker-model", "other-model"] });
      if (pathname.endsWith("/config")) {
        const id = pathname.split("/")[2], current = configs.get(id) || plain(NORMAL);
        if (request.method === "GET") return configGet ? configGet(request) : response(current);
        if (request.body.subagents && configPost) return configPost(request);
        let next = { ...current, ...request.body };
        if (request.body.subagents) {
          const sub = request.body.subagents;
          // The server stores the chosen backend explicitly (kernel included)
          // and reports TraeCLI for a session that never chose one.
          next = sub.action === "reset" ? { ...current, default_skills: NORMAL.default_skills, subagent_backend: "traecli", subagent_model: "" }
            : { ...current, default_skills: KERNEL.default_skills, subagent_backend: sub.backend, subagent_model: sub.model || current.subagent_model };
        }
        configs.set(id, next);
        return response(next);
      }
      if (pathname.endsWith("/turns/stream")) return turnPost ? turnPost(request) : turnResponse();
      throw new Error("unexpected request: " + pathname);
    },
  });
  vm.runInContext(existingFunction("async function api(", "const esc ="), context);
  vm.runInContext(marked("mobile team mode"), context);
  vm.runInContext(marked("session config hydration"), context);
  vm.runInContext(existingFunction("async function persistPicks()", "// A changed pick belongs"), context);
  vm.runInContext(marked("session switch"), context);
  vm.runInContext(marked("composer cd command"), context);
  vm.runInContext(marked("composer git command"), context);
  vm.runInContext(marked("turn session preparation"), context);
  vm.runInContext(existingFunction("function isTurnRunning()", "// The one send/stop button"), context);
  vm.runInContext(existingFunction("const RATE_STORAGE =", "function addFold("), context);
  vm.runInContext(existingFunction("function setAnswerRate(", "// Only completed answer bubbles"), context);
  vm.runInContext(marked("composer send"), context);
  context.wireTeamMode();
  return {
    context, elements, requests, notes, messages, consumed, configs, media, history, listeners, stored,
    state: (id = context.sessionId, base = context.store.base) => context.teamPickState(id, base),
    teamPosts: () => requests.filter((r) => r.method === "POST" && r.body?.subagents),
    turns: () => requests.filter((r) => r.url.endsWith("/turns/stream")),
    async choose(mode, backend = "kernel", model = "") {
      context.openTeamSheet();
      context.changeTeamDraft("mode", mode);
      context.changeTeamDraft("backend", backend);
      context.setTeamWorkerModel(model);
      return context.applyTeamDraft();
    },
  };
}

test("normal, Kernel and TraeCLI apply only the supported subagent payloads", async () => {
  for (const [mode, backend, model, expected] of [
    ["normal", "traecli", "ignored", { subagents: { action: "reset" } }],
    ["team", "kernel", "ignored", { subagents: { action: "use", backend: "kernel" } }],
    ["team", "traecli", " worker-model ", { subagents: { action: "use", backend: "traecli", model: "worker-model" } }],
    ["team", "traecli", "  ", { subagents: { action: "use", backend: "traecli" } }],
  ]) {
    // Start from the state the case actually changes: a no-op draft would not
    // post anything, and the new default already selects TraeCLI.
    const h = harness({ cfg: mode === "normal" ? KERNEL : NORMAL });
    await h.context.loadSessionConfig();
    assert.equal(await h.choose(mode, backend, model), true);
    assert.deepEqual(h.teamPosts().map((r) => r.body), [expected]);
    assert.equal(h.state().pending, false);
  }
});

test("GET config restores sticky team state independently of backend and output text", async () => {
  for (const [cfg, mode, backend] of [[NORMAL, "normal", "traecli"], [KERNEL, "team", "kernel"], [{ ...KERNEL, subagent_backend: "kernel" }, "team", "kernel"], [TRAECLI, "team", "traecli"], [{ ...TRAECLI, default_skills: [] }, "normal", "traecli"]]) {
    const h = harness({ cfg: { ...cfg, model: "main-model", agent: "build", reasoning_effort: "high" } });
    await h.context.loadSessionConfig();
    assert.equal(h.state().choice.mode, mode);
    assert.equal(h.state().choice.backend, backend);
    assert.equal(h.state().choice.model, cfg.subagent_model);
    assert.equal(h.elements.omodel.value, "main-model");
    assert.equal(h.elements.oagent.value, "build");
    assert.equal(h.elements.oeffort.value, "high");
    assert.equal(h.teamPosts().length, 0);
  }
});

for (const phone of [true, false]) test("team-only config refresh updates team state without replacing main model, agent or effort picks (" + (phone ? "phone" : "desktop") + ")", async () => {
  const h = harness({ phone, cfg: TRAECLI });
  h.elements.omodel.value = "main-model";
  h.elements.oagent.value = "build";
  h.elements.oeffort.value = "high";
  await h.context.loadSessionConfig({ teamOnly: true });
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.elements.omodel.value, "main-model");
  assert.equal(h.elements.oagent.value, "build");
  assert.equal(h.elements.oeffort.value, "high");
});

for (const order of ["team-first", "full-first"]) test("switching sessions restores full config despite the open team sheet (" + order + ")", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const beta = { ...KERNEL, model: "", agent: "", reasoning_effort: "" };
  const fullRead = deferred(), teamRead = deferred(), events = [];
  let betaReads = 0;
  const h = harness({ cfg: alpha, configGet: (request) => {
    if (request.url.endsWith("/alpha/config")) return response(alpha);
    assert.equal(request.url, "https://a.test/sessions/beta/config");
    if (++betaReads === 1) { events.push("full requested"); return fullRead.promise; }
    if (betaReads === 2) { events.push("team requested"); return teamRead.promise; }
    return response(beta);
  } });
  const picks = () => [h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value];
  h.configs.set("beta", plain(beta));
  await h.context.loadSessionConfig();
  assert.deepEqual(picks(), ["main-model", "build", "high"]);
  h.context.switchSession("beta");
  const restoring = h.context.loadSessionConfig().then(() => events.push("full completed"));
  await settle();
  assert.equal(betaReads, 1);
  h.context.openTeamSheet();
  await settle();
  assert.equal(betaReads, 2);
  assert.deepEqual(events, ["full requested", "team requested"]);
  if (order === "team-first") {
    teamRead.resolve(response({ ...beta, model: "main-model", agent: "build", reasoning_effort: "high" }));
    await settle();
    events.push("team completed");
    assert.deepEqual(picks(), ["main-model", "build", "high"]);
    assert.equal(h.state().choice.backend, "kernel");
    fullRead.resolve(response(beta));
    await restoring;
  } else {
    fullRead.resolve(response(beta));
    await restoring;
    assert.deepEqual(picks(), ["", "", ""]);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
    teamRead.resolve(response({ ...beta, model: "main-model", agent: "build", reasoning_effort: "high" }));
    await settle();
    events.push("team completed");
  }
  assert.deepEqual(events, ["full requested", "team requested", ...(order === "team-first" ? ["team completed", "full completed"] : ["full completed", "team completed"])]);
  assert.deepEqual(picks(), ["", "", ""]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
  h.context.closeTeamSheet();
  h.elements.prompt.value = "beta ordinary message";
  await h.context.send();
  await settle();
  assert.equal(h.turns().length, 1);
  assert.equal(h.turns()[0].url, "https://a.test/sessions/beta/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "beta ordinary message", confirm: true });
  assert.equal(h.teamPosts().length, 0);
});

test("a failed team-only read cannot cancel a pending full restore", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const beta = { ...KERNEL, model: "beta-model", agent: "beta-agent", reasoning_effort: "low" };
  const fullRead = deferred();
  let betaReads = 0;
  const h = harness({ cfg: alpha, configGet: (request) => {
    if (request.url.endsWith("/alpha/config")) return response(alpha);
    if (++betaReads === 1) return fullRead.promise;
    if (betaReads === 2) return { ok: false, status: 503, json: async () => ({ error: "team read unavailable" }) };
    return response(beta);
  } });
  for (const [id, value] of [["omodel", "beta-model"], ["oagent", "beta-agent"], ["oeffort", "low"]]) h.elements[id].options.push({ value });
  await h.context.loadSessionConfig();
  h.context.switchSession("beta");
  const restoring = h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  assert.equal(betaReads, 2);
  assert.ok(h.notes.some((text) => text.includes("team read unavailable")));
  fullRead.resolve(response(beta));
  await restoring;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["beta-model", "beta-agent", "low"]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(beta)));
  h.context.closeTeamSheet();
  h.elements.prompt.value = "beta configured message";
  await h.context.send();
  await settle();
  assert.deepEqual(h.turns()[0].body, { prompt: "beta configured message", confirm: true, model: "beta-model", agent: "beta-agent", reasoning_effort: "low" });
});

test("late full and team reads from the previous session cannot overwrite the current session or turn", async () => {
  const alpha = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const fullRead = deferred(), teamRead = deferred();
  let alphaReads = 0;
  const h = harness({ configGet: (request) => {
    if (request.url.endsWith("/beta/config")) return response(NORMAL);
    return ++alphaReads === 1 ? fullRead.promise : teamRead.promise;
  } });
  const reading = h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  assert.equal(alphaReads, 2);
  h.context.switchSession("beta");
  await h.context.loadSessionConfig();
  teamRead.resolve(response(alpha));
  fullRead.resolve(response(alpha));
  await reading;
  await settle();
  assert.equal(h.context.sessionId, "beta");
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["", "", ""]);
  assert.equal(h.state().choice.mode, "normal");
  h.elements.prompt.value = "current beta message";
  await h.context.send();
  await settle();
  assert.equal(h.turns()[0].url, "https://a.test/sessions/beta/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "current beta message", confirm: true });
});

test("an older full read cannot roll back the team state restored after /reset", async () => {
  const before = { ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" };
  const after = { ...NORMAL, model: "main-model", agent: "build", reasoning_effort: "high" };
  const fullRead = deferred();
  let reads = 0;
  const h = harness({ cfg: before, configGet: () => ++reads === 1 ? fullRead.promise : response(after), turnPost: () => turnResponse() });
  h.context.applySessionConfig(before);
  const reading = h.context.loadSessionConfig();
  h.elements.prompt.value = "/reset";
  await h.context.send();
  await settle();
  assert.equal(reads, 2);
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.state().choice.mode, "normal");
  fullRead.resolve(response(before));
  await reading;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["main-model", "build", "high"]);
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  assert.equal(h.elements.teamlabel.textContent, "普通对话");
});

test("a superseded full read cannot overwrite a newer full restore", async () => {
  const oldRead = deferred(), newRead = deferred();
  let reads = 0;
  const h = harness({ configGet: () => ++reads === 1 ? oldRead.promise : newRead.promise });
  const first = h.context.loadSessionConfig(), second = h.context.loadSessionConfig();
  newRead.resolve(response(NORMAL));
  await second;
  oldRead.resolve(response({ ...TRAECLI, model: "main-model", agent: "build", reasoning_effort: "high" }));
  await first;
  assert.deepEqual([h.elements.omodel.value, h.elements.oagent.value, h.elements.oeffort.value], ["", "", ""]);
  assert.equal(h.state().choice.mode, "normal");
});

test("first send creates a session and waits for config success before the actual turn POST", async () => {
  const save = deferred(), h = harness({ sid: "", configPost: () => save.promise, skills: ["review"] });
  h.elements.omodel.value = "main-model";
  await h.choose("team", "traecli", "worker-model");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.elements.teamstatus.textContent, "待保存");
  const sending = h.context.send();
  await settle();
  assert.equal(h.context.sessionId, "fresh");
  assert.equal(h.context.pendingNew, "fresh");
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.state().pending, true);
  assert.equal(h.state().confirmed, null);
  assert.equal(h.elements.teamstatus.textContent, "保存中");
  assert.equal(h.turns().length, 0);
  assert.equal(h.elements.prompt.value, "一起完成这个任务");
  h.configs.set("fresh", { ...TRAECLI, model: "main-model" });
  save.resolve(response({ ...TRAECLI, model: "main-model" }));
  await sending;
  assert.equal(h.turns().length, 1);
  assert.equal(h.turns()[0].url, "https://a.test/sessions/fresh/turns/stream");
  assert.deepEqual(h.turns()[0].body, { prompt: "一起完成这个任务", confirm: true, skills: ["review"], model: "main-model" });
  assert.equal(h.state().pending, false);
  assert.equal(h.consumed.length, 1);
});

for (const state of ["replay", "server", "stale", "live"]) test(`button and keyboard sends share the running guard (${state})`, async () => {
  const h = harness(), c = h.context;
  let aborted = 0, stopped = 0, keydown;
  Object.assign(c, { eventAbort: { abort() { aborted++; } }, recoveredView: null,
    stopTurn() { stopped++; },
  });
  vm.runInContext(existingFunction("function abortEvents()", "async function pollConfirm()"), c);
  if (state === "replay") Object.assign(c, { eventActive: true, eventSession: "alpha", staleDone: true });
  if (state === "server") Object.assign(c, { turnRunning: true, turnRunningAt: Date.now() });
  if (state === "stale") Object.assign(c, { staleSession: "alpha", staleDone: true });
  if (state === "live") c.turns.set("alpha", {});
  const images = c.pendingImages = [{ filename: "draft.png", data_base64: "image" }];
  const draft = h.elements.prompt.value;
  h.elements.send = {};
  h.elements.prompt.addEventListener = (_, listener) => { keydown = listener; };
  vm.runInContext(existingFunction('$("send").onclick =', '$("sidenote").onclick ='), c);
  vm.runInContext(existingFunction('$("prompt").addEventListener("keydown"', "const toggleSessions ="), c);
  h.elements.send.onclick();
  assert.equal(stopped, 1, "the button must stop the active turn");
  await c.send();
  for (const modifier of ["ctrlKey", "metaKey"]) {
    let prevented = false;
    keydown({ key: "Enter", [modifier]: true, preventDefault() { prevented = true; } });
    await settle();
    assert.equal(prevented, true);
  }
  assert.equal(h.turns().length, 0);
  assert.equal(aborted, 0, "blocked submission must preserve the active event feed");
  assert.equal(h.elements.prompt.value, draft);
  assert.equal(c.pendingImages, images);
  c.turns.clear();
  Object.assign(c, { eventActive: false, eventSession: "", staleSession: "", eventStreamEnded: true });
  keydown({ key: "Enter", ctrlKey: true, preventDefault() {} });
  await settle();
  assert.equal(h.turns().length, 1, "keyboard submission unlocks after true completion");
});

test("a replay that starts during async send setup prevents the turn POST and preserves the draft", async () => {
  const h = harness(), c = h.context, ready = deferred();
  const draft = h.elements.prompt.value;
  c.ensureTeamConfig = () => ready.promise;
  let aborted = false;
  c.abortEvents = () => { aborted = true; };
  const sending = c.send();
  await settle();
  Object.assign(c, { eventActive: true, eventSession: "alpha", staleDone: true });
  ready.resolve(true);
  await sending;
  assert.equal(h.turns().length, 0);
  assert.equal(h.elements.prompt.value, draft);
  assert.equal(aborted, false);
});

test("failed config blocks the first turn, retains selection, text, images and skills, and can retry", async () => {
  let fail = true;
  const h = harness({ sid: "", skills: ["review"], configPost: () => fail ? { ok: false, status: 503, json: async () => ({ error: "temporarily offline" }) } : response(KERNEL) });
  const images = [{ filename: "photo.png", data: "data:image/png;base64,test" }];
  h.context.pendingImages = images;
  await h.choose("team", "kernel", "never-send-to-kernel");
  await h.context.send();
  assert.equal(h.turns().length, 0);
  assert.equal(h.state().pending, true);
  assert.equal(h.state().choice.backend, "kernel");
  assert.equal(h.state().choice.model, "never-send-to-kernel");
  assert.equal(h.elements.prompt.value, "一起完成这个任务");
  assert.equal(h.context.pendingImages, images);
  assert.equal(h.consumed.length, 0);
  assert.equal(h.context.sendBusy, false);
  assert.equal(h.elements.teamtrigger.dataset.state, "error");
  assert.ok(h.notes.some((text) => /503 temporarily offline/.test(text)));
  h.context.openTeamSheet();
  assert.equal(h.elements["team" + h.state().choice.backend].attrs["aria-checked"], "true");
  h.context.closeTeamSheet();
  fail = false;
  h.configs.set("fresh", KERNEL);
  await h.context.send();
  assert.equal(h.turns().length, 1);
  assert.equal(h.teamPosts().length, 2);
  assert.equal("model" in h.teamPosts()[1].body.subagents, false);
});

test("cancel, Escape and browser back discard sheet edits without persisting", async () => {
  for (const action of ["cancel", "escape", "back"]) {
    const h = harness();
    await h.context.loadSessionConfig();
    h.context.openTeamSheet();
    h.context.changeTeamDraft("mode", "team");
    h.context.changeTeamDraft("backend", "traecli");
    h.context.setTeamWorkerModel("draft-worker");
    if (action === "cancel") h.elements.teamcancel.click();
    if (action === "escape") h.context.teamSheetKeydown({ key: "Escape", preventDefault() {} });
    if (action === "back") h.listeners.get("popstate")();
    await settle();
    assert.equal(h.teamPosts().length, 0);
    assert.deepEqual(plain(h.state().choice), { mode: "normal", backend: "traecli", model: "" });
    assert.equal(h.elements.teampick.hidden, true);
    assert.equal(h.elements.teamtrigger.attrs["aria-expanded"], "false");
    assert.equal(h.context.document.activeElement, h.elements.teamtrigger);
  }
});

test("saving can be dismissed, and send waits for the same request without duplicate POSTs", async () => {
  const save = deferred(), h = harness({ configPost: () => save.promise });
  await h.context.loadSessionConfig();
  const applying = h.choose("team", "kernel");
  await settle();
  assert.equal(h.elements.teamapply.disabled, true);
  assert.equal(h.elements.teamcancel.textContent, "收起");
  assert.equal(h.state().confirmed.mode, "normal");
  h.elements.teamcancel.click();
  const sending = h.context.send();
  await settle();
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.turns().length, 0);
  h.configs.set("alpha", KERNEL);
  save.resolve(response(KERNEL));
  assert.equal(await applying, true);
  await sending;
  assert.equal(h.turns().length, 1);
});

test("repeated send taps share one preparation and cannot create duplicate sessions or turns", async () => {
  const save = deferred(), h = harness({ sid: "", configPost: () => save.promise });
  await h.choose("team", "kernel");
  const first = h.context.send(), second = h.context.send();
  await settle();
  assert.equal(h.requests.filter((r) => r.url === "https://a.test/sessions").length, 1);
  assert.equal(h.teamPosts().length, 1);
  assert.equal(h.turns().length, 0);
  h.configs.set("fresh", KERNEL);
  save.resolve(response(KERNEL));
  await Promise.all([first, second]);
  assert.equal(h.turns().length, 1);
});

test("an open sheet preserves edits and displays a failed save without acknowledging it", async () => {
  const h = harness({ configPost: () => { throw new Error("network disconnected"); } });
  await h.context.loadSessionConfig();
  assert.equal(await h.choose("team", "traecli", "worker-model"), false);
  assert.equal(h.elements.teampick.hidden, false);
  assert.equal(h.elements.teamworkervalue.textContent, "worker-model");
  assert.equal(h.elements.teamerror.hidden, false);
  assert.match(h.elements.teamerror.textContent, /network disconnected/);
  assert.equal(h.elements.teamapply.disabled, false);
  assert.equal(h.state().confirmed.mode, "normal");
  assert.equal(h.state().pending, true);
});

test("a successful HTTP status with mismatched structured config cannot submit a turn", async () => {
  const h = harness({ sid: "", configPost: () => response(NORMAL) });
  await h.choose("team", "traecli", "worker-model");
  await h.context.send();
  assert.equal(h.turns().length, 0);
  assert.equal(h.state().pending, true);
  assert.match(h.state().error, /尚未确认/);
});

test("late GETs cannot overwrite pending intent or an acknowledged team save", async () => {
  const oldRead = deferred(), h = harness({ configGet: () => oldRead.promise });
  const reading = h.context.loadSessionConfig();
  await h.choose("team", "traecli", "worker-model");
  oldRead.resolve(response(NORMAL));
  await reading;
  assert.equal(h.state().choice.mode, "team");
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.state().pending, false);
  assert.equal(h.elements.teamlabel.textContent, "subagent");
  assert.equal(h.elements.teamtrigger.attrs["aria-label"], "subagent · TraeCLI，选择对话模式");
});

test("session switches restore each selection and never inherit it into a new session", async () => {
  const h = harness({ cfg: TRAECLI });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.changeTeamDraft("mode", "normal");
  h.context.switchSession("fresh");
  assert.equal(h.elements.teampick.hidden, true);
  await h.context.loadSessionConfig();
  assert.equal(h.state().choice.mode, "normal");
  h.context.switchSession("alpha");
  assert.equal(h.state().choice.backend, "traecli");
  assert.equal(h.elements.teamlabel.textContent, "subagent");
  assert.equal(h.teamPosts().length, 0);
});

test("navigation during first-session creation cannot overwrite the selected session or submit", async () => {
  const created = deferred(), h = harness({ sid: "", createSession: () => created.promise });
  await h.choose("team", "kernel");
  const sending = h.context.send();
  await settle();
  h.context.switchSession("beta");
  created.resolve(response({ id: "fresh" }));
  await sending;
  assert.equal(h.context.sessionId, "beta");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.turns().length, 0);
  assert.equal(h.state("").pending, true);
});

test("session or server changes during save prevent the old input from becoming a turn", async () => {
  for (const change of ["session", "server"]) {
    const save = deferred(), h = harness({ configPost: () => save.promise });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", "traecli", "worker-model");
    await settle();
    h.context.closeTeamSheet();
    const sending = h.context.send();
    await settle();
    if (change === "session") h.context.switchSession("beta");
    else { h.context.store.base = "https://other.test"; h.context.restoreTeamPick(); }
    save.resolve(response(TRAECLI));
    await applying;
    await sending;
    assert.equal(h.turns().length, 0);
    assert.equal(h.state("alpha", "https://a.test").confirmed.backend, "traecli");
    assert.equal(h.state().choice.mode, "normal");
    assert.notEqual(h.elements.teamlabel.textContent, "subagent");
  }
});

test("input edits and active composition during config saving preserve the new draft", async () => {
  for (const composing of [false, true]) {
    const save = deferred(), h = harness({ sid: "", configPost: () => save.promise });
    await h.choose("team", "kernel");
    const sending = h.context.send();
    await settle();
    if (composing) h.context.promptComposing = true;
    else h.elements.prompt.value = "修改后的问题";
    save.resolve(response(KERNEL));
    await sending;
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, composing ? "一起完成这个任务" : "修改后的问题");
  }
});

test("keyboard navigation uses real backend radios, traps focus and respects search IME composition", async () => {
  const h = harness();
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.teamSheetKeydown({ key: "ArrowDown", preventDefault() {} });
  assert.equal(h.elements.teamcollab.attrs["aria-checked"], "true");
  h.elements.teamkernel.focus();
  h.context.teamSheetKeydown({ key: "ArrowRight", preventDefault() {} });
  assert.equal(h.elements.teamtraecli.attrs["aria-checked"], "true");
  assert.equal(h.elements.teamworkerwrap.hidden, false);
  h.elements.teamapply.focus();
  h.context.teamSheetKeydown({ key: "Tab", preventDefault() {} });
  assert.equal(h.context.document.activeElement, h.elements.teamclose);
  h.elements.teamworkersearch.oncompositionstart();
  h.context.teamSheetKeydown({ key: "Escape", preventDefault() { assert.fail("IME key must not dismiss the sheet"); } });
  assert.equal(h.elements.teampick.hidden, false);
  h.elements.teamworkersearch.oncompositionend();
});

test("worker model stays separate from the main model and blank preserves the saved worker", async () => {
  const h = harness({ cfg: { ...TRAECLI, model: "main-model" } });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  assert.match(h.elements.teamworkerhint.textContent, /主模型.*model.*worker-model/);
  h.elements.teamworkerbtn.click();
  await settle();
  const rows = h.elements.teamworkeroptions.children;
  const fallback = rows.find((row) => row.dataset.model === "");
  assert.ok(fallback, "picker needs the session-default row");
  fallback.click();
  assert.equal(h.elements.omodel.value, "main-model");
  assert.equal(await h.context.applyTeamDraft(), true);
  assert.deepEqual(h.teamPosts()[0].body, { subagents: { action: "use", backend: "traecli" } });
  assert.equal(h.state().choice.model, "worker-model");
});

test("worker model opening and selection work with browser NodeList snapshots", async () => {
  const h = harness({ cfg: TRAECLI });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  const snapshot = h.elements.teamworkeroptions.querySelectorAll('[role="radio"]');
  assert.equal(Array.isArray(snapshot), false);
  assert.equal(snapshot.find, undefined);
  assert.equal(snapshot.map, undefined);
  assert.equal(snapshot.item(0), snapshot[0]);
  assert.doesNotThrow(() => h.elements.teamworkerbtn.click());
  const rows = Array.from(h.elements.teamworkeroptions.querySelectorAll('[role="radio"]'));
  assert.equal(h.context.document.activeElement, rows.find((row) => row.dataset.model === "worker-model"));
  assert.equal(h.context.document.activeElement.focusOptions.preventScroll, true);
  assert.equal(h.elements.teamworkerbtn.attrs["aria-expanded"], "true");
  assert.doesNotThrow(() => rows.find((row) => row.dataset.model === "other-model").click());
  assert.equal(h.elements.teamworkervalue.textContent, "other-model");
  assert.equal(h.elements.teamworkerlist.hidden, true);
  assert.doesNotThrow(() => h.elements.teamworkerbtn.click());
  assert.equal(h.context.document.activeElement.dataset.model, "other-model");
});

test("the subagent model is a list picker fed by /subagent-models and filtered by its search field", async () => {
  const h = harness({ subagentModels: () => response({ models: ["alpha-model", "beta-model"] }) });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.elements.teamcollab.click();
  h.elements.teamtraecli.click();
  await settle();
  assert.ok(h.requests.some((request) => request.url.endsWith("/subagent-models") && request.method === "GET"), "the picker must ask the server for the model list");
  assert.equal(h.elements.teamworkervalue.textContent, "会话默认");
  h.elements.teamworkerbtn.click();
  assert.equal(h.elements.teamworkerlist.hidden, false);
  assert.deepEqual(h.elements.teamworkeroptions.children.map((row) => row.dataset.model), ["", "alpha-model", "beta-model"]);
  h.elements.teamworkersearch.value = "beta";
  h.elements.teamworkersearch.oninput();
  assert.deepEqual(h.elements.teamworkeroptions.children.map((row) => row.dataset.model), ["beta-model"]);
  h.elements.teamworkeroptions.children[0].click();
  assert.equal(h.elements.teamworkervalue.textContent, "beta-model");
  assert.equal(h.elements.teamworkerlist.hidden, true);
  assert.equal(await h.context.applyTeamDraft(), true);
  assert.deepEqual(h.teamPosts()[0].body, { subagents: { action: "use", backend: "traecli", model: "beta-model" } });
  assert.equal(h.state().choice.model, "beta-model");
});

test("the open list floats over the sheet and closes without changing the draft", async () => {
  const h = harness({ subagentModels: () => response({ models: ["alpha-model", "beta-model"] }) });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.elements.teamcollab.click();
  h.elements.teamtraecli.click();
  await settle();
  // Layout boxes are stub data: the trigger sits in the lower half, so the list
  // may grow past the sheet panel top and is bounded by the viewport only.
  let triggerTop = 420;
  h.context.window.innerHeight = 800;
  h.elements.teamworkerbtn.getBoundingClientRect = () => ({ top: triggerTop, bottom: triggerTop + 44, left: 12, width: 340, height: 44 });
  h.elements.teamworkerwrap.getBoundingClientRect = () => ({ top: triggerTop - 34, bottom: triggerTop + 78, left: 12, width: 340, height: 112 });
  h.elements.teampickpanel.getBoundingClientRect = () => ({ top: 60, bottom: 900, left: 0, width: 393, height: 840 });
  h.elements.teamworkerbtn.click();
  assert.equal(h.elements.teamworkerlist.hidden, false);
  assert.equal(h.elements.teamworkerlist.style.maxHeight, "422px", "the list reaches from the sheet's top edge down to the card bottom");
  assert.equal(h.elements.teamworkeroptions.style.maxHeight, "358px", "model rows scroll below the search field");
  assert.equal(h.elements.teamworkerlist.style.bottom, "412px", "the list bottom lands on the card's bottom edge");
  for (const [top, expected] of [[660, 662], [296, 298], [190, 240]]) {
    triggerTop = top;
    h.context.placeWorkerList();
    assert.equal(h.elements.teamworkerlist.style.maxHeight, expected + "px", "the viewport cap and preferred floor must fit the measured space");
    assert.equal(h.elements.teamworkeroptions.style.maxHeight, Math.max(0, expected - 64) + "px");
  }
  h.context.teamSheetKeydown({ key: "Escape", preventDefault() {} });
  assert.equal(h.elements.teamworkerlist.hidden, true);
  assert.equal(h.elements.teampick.hidden, false, "Escape must close the list before the sheet");
  h.elements.teamworkerbtn.click();
  h.elements.teampickpanel.onclick({ target: { closest: () => null } });
  assert.equal(h.elements.teamworkerlist.hidden, true);
  assert.equal(h.elements.teamworkervalue.textContent, "会话默认", "dismissing the list must not pick a model");
  assert.equal(h.elements.teamcollab.attrs["aria-checked"], "true");
  assert.equal(h.teamPosts().length, 0);
});

test("worker list ignores child taps throughout the trigger card", async () => {
  const h = harness({ cfg: TRAECLI });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  await settle();
  h.elements.teamworkerbtn.click();
  const choice = plain(h.state().choice);
  for (const target of [
    ...h.elements.teamworkerbtn.children,
    h.elements.teamworkerlabel, h.elements.teamworkerhint, h.elements.teamworkerwrap,
    h.elements.teamworkeroptions.children[0].children[0].children[0],
  ]) {
    h.elements.teampickpanel.onclick({ target });
    assert.equal(h.elements.teamworkerlist.hidden, false, "an inside tap must leave the worker list open");
  }
  h.elements.teampickpanel.onclick({ target: h.elements.teamadvanced });
  assert.equal(h.elements.teamworkerlist.hidden, true);
  assert.equal(h.elements.teampick.hidden, false);
  assert.deepEqual(plain(h.state().choice), choice);
  assert.equal(h.teamPosts().length, 0);
});

test("the trigger sits in the metadata line without toggling it", () => {
  const h = harness();
  let prevented = false;
  h.elements.teamtrigger.onclick({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true, "the summary must not also collapse the override panel");
  assert.equal(h.elements.teampick.hidden, false);
});

test("a failed model-list fetch keeps the session-default row and offers a retry", async () => {
  let fail = true;
  const h = harness({ subagentModels: () => {
    if (fail) throw new Error("503 temporarily offline");
    return response({ models: ["alpha-model"] });
  } });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.elements.teamcollab.click();
  h.elements.teamtraecli.click();
  await settle();
  h.elements.teamworkerbtn.click();
  await settle();
  assert.match(h.elements.teamworkerstatus.textContent, /503 temporarily offline/);
  assert.equal(h.elements.teamworkerretry.hidden, false);
  assert.deepEqual(h.elements.teamworkeroptions.children.map((row) => row.dataset.model), [""]);
  fail = false;
  h.elements.teamworkerretry.click();
  await settle();
  assert.equal(h.elements.teamworkerretry.hidden, true);
  assert.deepEqual(h.elements.teamworkeroptions.children.map((row) => row.dataset.model), ["", "alpha-model"]);
});

test("empty worker discovery is retried when the model list opens again", async () => {
  let calls = 0;
  const h = harness({ cfg: TRAECLI, subagentModels: () => response({ models: ++calls === 1 ? [] : ["alpha-model"] }) });
  await h.context.loadSessionConfig(); h.context.openTeamSheet();
  await settle();
  assert.equal(calls, 1);
  assert.match(h.elements.teamworkerstatus.textContent, /未返回模型列表/);
  h.elements.teamworkerbtn.click();
  await settle();
  assert.equal(calls, 2);
  assert.ok(h.elements.teamworkeroptions.children.some((row) => row.dataset.model === "alpha-model"));
  h.context.setWorkerListOpen(false); h.context.setWorkerListOpen(true);
  await settle();
  assert.equal(calls, 2, "nonempty worker discovery is cached");
});

test("cold worker openings and retries share one pending discovery", async () => {
  const discovery = deferred();
  let calls = 0;
  const h = harness({ cfg: TRAECLI, subagentModels: () => { calls++; return discovery.promise; } });
  await h.context.loadSessionConfig(); h.context.openTeamSheet();
  const first = h.context.loadWorkerModels(), second = h.context.loadWorkerModels(true);
  assert.equal(calls, 1);
  assert.equal(first, second);
  let completed = false;
  first.then(() => { completed = true; });
  await settle();
  assert.equal(completed, false, "every caller waits for the same discovery");
  discovery.resolve(response({ models: ["alpha-model"] }));
  await Promise.all([first, second]);
  assert.ok(h.elements.teamworkeroptions.children.some((row) => row.dataset.model === "alpha-model"));
});

test("worker discovery cannot publish an old server's result over a newer request", async () => {
  const old = deferred();
  const h = harness({ cfg: TRAECLI, subagentModels: (request) => request.url.startsWith("https://a.test")
    ? old.promise : response({ models: ["new-model"] }) });
  await h.context.loadSessionConfig(); h.context.openTeamSheet();
  const first = h.context.loadWorkerModels();
  h.context.store.base = "https://b.test";
  await h.context.loadWorkerModels();
  old.resolve(response({ models: ["old-model"] })); await first;
  h.context.setWorkerListOpen(true);
  assert.deepEqual(h.elements.teamworkeroptions.children.map((row) => row.dataset.model), ["", "new-model", "worker-model"]);
  assert.equal(h.requests.filter((request) => request.url.endsWith("/subagent-models")).length, 2);
});

test("send captures a transcript anchor and links successive pending inputs despite a stale count", async () => {
  const h = harness();
  await h.context.loadSessionConfig();
  const preceding = h.context.document.createElement("div");
  preceding.dataset.canonicalIdentity = '["assistant","finished answer",false,"",null]';
  preceding.dataset.canonicalIndex = "7";
  h.elements.msgs.appendChild(preceding);
  h.context.canonicalHistoryTotal = 1;
  await h.context.send();
  const first = h.context.pendingUserInputs.get("alpha")[0];
  assert.equal(first.minIndex, 1);
  assert.equal(first.afterHistoryKey, preceding.dataset.canonicalIdentity);
  assert.equal(first.afterHistoryIndex, 7);
  assert.equal(first.afterInput, null);
  h.elements.prompt.value = "second prompt";
  await h.context.send();
  const second = h.context.pendingUserInputs.get("alpha")[1];
  assert.equal(second.afterInput, first);
  assert.equal(second.afterHistoryKey, preceding.dataset.canonicalIdentity);
  assert.equal(h.turns().length, 2);
});

// The Rust driver owns command parsing. The turn fixtures below supply its
// persisted result; the assertions exercise the actual client send and GET path.
test("mobile /use and /reset preserve the complete input and restore authoritative config", async () => {
  for (const [input, before, after] of [
    ["/use traecli worker-model", NORMAL, TRAECLI],
    ["/reset", TRAECLI, NORMAL],
    [" \r\n :use kernel\r\n\r\n 第一行\r\n第二行  \r\n", TRAECLI, { ...KERNEL, subagent_model: "worker-model" }],
    ["/use", TRAECLI, TRAECLI],
    ["/use xxx", KERNEL, KERNEL],
    ["/reset extra", TRAECLI, TRAECLI],
    ["/useful kernel", KERNEL, KERNEL],
    ["/resetting", TRAECLI, TRAECLI],
    ["/reset/other", TRAECLI, TRAECLI],
  ]) {
    const h = harness({ cfg: before, turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      assert.equal("subagents" in request.body, false);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    h.elements.prompt.value = input;
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal("subagents" in h.turns()[0].body, false);
    assert.equal(h.teamPosts().length, 0);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    assert.equal(h.elements.teamlabel.textContent, h.context.teamChoiceLabel(h.state().choice));
  }
});

test("commands wait for an earlier UI save, then win without a later replay of the old selection", async () => {
  for (const [input, backend, earlier, after] of [
    ["/use traecli worker-model", "kernel", KERNEL, TRAECLI],
    ["/reset", "traecli", TRAECLI, NORMAL],
  ]) {
    const save = deferred();
    let turnCount = 0;
    const h = harness({ configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.body.prompt, turnCount++ ? "后续消息" : input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", backend, backend === "traecli" ? "worker-model" : "");
    await settle();
    h.context.closeTeamSheet();
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    assert.equal(h.teamPosts().length, 1);
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, input);
    h.configs.set("alpha", plain(earlier));
    save.resolve(response(earlier));
    await Promise.all([applying, sending]);
    await settle();
    const writeIndex = h.requests.findIndex((r) => r.body?.subagents);
    const turnIndex = h.requests.findIndex((r) => r.url.endsWith("/turns/stream"));
    assert.ok(writeIndex < turnIndex);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.ok(h.requests.slice(turnIndex + 1).some((r) => r.method === "GET" && r.url.endsWith("/config")));
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    h.elements.prompt.value = "后续消息";
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 2);
    assert.equal(h.teamPosts().length, 1);
    assert.equal(h.state().pending, false);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a first /use or /reset carries the UI intent to the new session and keeps the raw command last", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    const save = deferred();
    const h = harness({ sid: "", configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.url, "https://a.test/sessions/fresh/turns/stream");
      assert.equal(request.body.prompt, input);
      h.configs.set("fresh", plain(after));
      return turnResponse();
    } });
    await h.choose("team", "kernel");
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    assert.equal(h.context.sessionId, "fresh");
    assert.equal(h.context.pendingNew, "fresh");
    assert.equal(h.teamPosts()[0].url, "https://a.test/sessions/fresh/config");
    assert.equal(h.turns().length, 0);
    h.configs.set("fresh", plain(KERNEL));
    save.resolve(response(KERNEL));
    await sending;
    await settle();
    const writesAndTurns = h.requests.filter((r) => r.body?.subagents || r.url.endsWith("/turns/stream"));
    assert.deepEqual(writesAndTurns.map((r) => r.url), ["https://a.test/sessions/fresh/config", "https://a.test/sessions/fresh/turns/stream"]);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 1);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a failed mode save retains /use and /reset and retries before submitting either command", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    let fail = true;
    const h = harness({ sid: "", configPost: () => {
      if (fail) return { ok: false, status: 503, json: async () => ({ error: "save unavailable" }) };
      h.configs.set("fresh", plain(KERNEL));
      return response(KERNEL);
    }, turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      h.configs.set("fresh", plain(after));
      return turnResponse();
    } });
    await h.choose("team", "kernel");
    h.elements.prompt.value = input;
    await h.context.send();
    assert.equal(h.turns().length, 0);
    assert.equal(h.elements.prompt.value, input);
    assert.equal(h.state().pending, true);
    assert.ok(h.notes.some((text) => text.includes("save unavailable")));
    fail = false;
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 2);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    assert.equal(h.state().pending, false);
  }
});

test("a GET started before the command cannot replace its newer authoritative result", async () => {
  for (const [input, before, after] of [["/use traecli worker-model", NORMAL, TRAECLI], ["/reset", TRAECLI, NORMAL]]) {
    const oldRead = deferred();
    let reads = 0;
    const h = harness({ cfg: before, configGet: () => ++reads === 1 ? oldRead.promise : response(after), turnPost: (request) => {
      assert.equal(request.body.prompt, input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    h.context.applySessionConfig(before);
    const reading = h.context.loadSessionConfig({ teamOnly: true });
    h.elements.prompt.value = input;
    await h.context.send();
    await settle();
    assert.equal(reads, 2);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
    oldRead.resolve(response(before));
    await reading;
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("a UI application made after command submission is a newer intent and wins when its save completes", async () => {
  const finished = deferred(), save = deferred();
  const h = harness({ cfg: TRAECLI, configPost: () => save.promise, turnPost: (request) => {
    assert.equal(request.body.prompt, "/reset");
    return turnResponse(() => finished.promise);
  } });
  await h.context.loadSessionConfig();
  h.elements.prompt.value = "/reset";
  const sending = h.context.send();
  await settle();
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.context.turns.size, 1);
  const applying = h.choose("team", "kernel");
  await settle();
  const turnIndex = h.requests.findIndex((r) => r.url.endsWith("/turns/stream"));
  const writeIndex = h.requests.findIndex((r) => r.body?.subagents);
  assert.ok(turnIndex < writeIndex);
  h.configs.set("alpha", plain(NORMAL));
  finished.resolve({ done: false, value: Buffer.from("event: done\n\n") });
  await sending;
  await settle();
  assert.equal(h.state().pending, true);
  assert.equal(h.state().choice.backend, "kernel");
  assert.equal(h.elements.teamstatus.textContent, "保存中");
  h.configs.set("alpha", plain(KERNEL));
  save.resolve(response(KERNEL));
  await applying;
  assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(KERNEL)));
  assert.equal(h.state().pending, false);
});

test("cancelled UI edits leave /reset on the existing turn path", async () => {
  const h = harness({ cfg: TRAECLI, turnPost: (request) => {
    assert.equal(request.body.prompt, "/reset");
    h.configs.set("alpha", plain(NORMAL));
    return turnResponse();
  } });
  await h.context.loadSessionConfig();
  h.context.openTeamSheet();
  h.context.changeTeamDraft("mode", "normal");
  h.context.closeTeamSheet();
  assert.equal(h.state().choice.backend, "traecli");
  h.elements.prompt.value = "/reset";
  await h.context.send();
  await settle();
  assert.equal(h.turns()[0].body.prompt, "/reset");
  assert.equal(h.teamPosts().length, 0);
  assert.equal(h.state().choice.mode, "normal");
});

test("switching sessions during a mode save keeps the command draft and cannot submit it in the new session", async () => {
  for (const [input, after] of [["/use traecli worker-model", TRAECLI], ["/reset", NORMAL]]) {
    const save = deferred();
    const h = harness({ configPost: () => save.promise, turnPost: (request) => {
      assert.equal(request.url, "https://a.test/sessions/alpha/turns/stream");
      assert.equal(request.body.prompt, input);
      h.configs.set("alpha", plain(after));
      return turnResponse();
    } });
    await h.context.loadSessionConfig();
    const applying = h.choose("team", "kernel");
    await settle();
    h.context.closeTeamSheet();
    h.elements.prompt.value = input;
    const sending = h.context.send();
    await settle();
    h.context.switchSession("beta");
    h.configs.set("alpha", plain(KERNEL));
    save.resolve(response(KERNEL));
    await Promise.all([applying, sending]);
    assert.equal(h.turns().length, 0);
    assert.equal(h.stored.get("draft.alpha"), input);
    h.context.switchSession("alpha");
    assert.equal(h.elements.prompt.value, input);
    await h.context.send();
    await settle();
    assert.equal(h.turns().length, 1);
    assert.equal(h.turns()[0].body.prompt, input);
    assert.equal(h.teamPosts().length, 1);
    assert.deepEqual(plain(h.state().choice), plain(h.context.teamChoiceFromConfig(after)));
  }
});

test("desktop opens the shared team sheet and can select agent-team", async () => {
  const h = harness({ phone: false });
  h.context.openTeamSheet();
  assert.equal(h.elements.teampick.hidden, false);
  assert.equal(h.history.pushes, 1);
  assert.equal(await h.choose("team", "kernel"), true);
  assert.deepEqual(h.teamPosts().map((r) => r.body), [{ subagents: { action: "use", backend: "kernel" } }]);
  assert.equal(h.elements.teampick.hidden, true);
  await h.context.send();
  assert.equal(h.turns().length, 1);
});

test("the layout hosts the team entry and the accessible sheet uses the existing theme and viewport", () => {
  const phoneStart = css.indexOf("@media (max-width:700px)"), desktop = css.slice(0, phoneStart), phoneCss = css.slice(phoneStart);
  assert.match(desktop, /#teamcontrols \{ display:none; \}/);
  assert.match(desktop, /#teampick\[hidden\] \{ display:none; \}/);
  assert.match(desktop, /#teampick:not\(\[hidden\]\) \{ display:block; \}/);
  assert.match(phoneCss, /#ovrsummaryline #teamcontrols \{ display:flex;/);
  assert.match(phoneCss, /#ovrwrap > summary \{ display:flex; align-items:center;/);
  assert.match(phoneCss, /#ovrsummary \{ flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;/);
  assert.match(desktop, /#ovrsummaryline #teamcontrols \{ flex:0 0 auto; padding:0; margin-left:auto; \}/);
  const desktopRow = /@media \(min-width:701px\) \{([\s\S]*?)\n\}/.exec(desktop);
  assert.ok(desktopRow);
  assert.match(desktopRow[1], /#ovrwrap \{[^}]*display:grid; grid-template-columns:minmax\(0,1fr\) auto;[^}]*gap:4px 8px; padding:10px 24px 4px;/);
  assert.match(desktopRow[1], /#ovrwrap > summary \{ display:flex; align-items:center; grid-column:2; grid-row:1; padding:0; list-style:none; \}/);
  assert.match(desktopRow[1], /#ovrsummary \{ display:none; \}/);
  assert.match(desktopRow[1], /#ovr \{[^}]*display:grid;[^}]*grid-column:1; grid-row:1;[^}]*min-width:0;/);
  assert.match(desktopRow[1], /#teamcontrols \{ display:flex; \}/);
  assert.match(desktopRow[1], /#teamtrigger \{ min-height:36px;/);
  assert.doesNotMatch(css, /#teamcontrols \{(?:[^}]*;)?\s*(?:width:100%|display:block)/);
  assert.match(desktop, /#teamtrigger \{[^}]*gap:5px; min-height:26px;[^}]*padding:0 9px; border-radius:999px;[^}]*font-size:12px;/);
  assert.match(desktop, /\.teamicon \{[^}]*width:13px; height:13px;/);
  assert.match(desktop, /\.teamicon svg \{ width:13px; height:13px;/);
  assert.match(desktop, /\.teamchev \{[^}]*width:14px; height:14px;/);
  assert.match(desktop, /#teambackends \.teamchoice \{[^}]*min-height:60px;/);
  assert.match(desktop, /#teampickpanel \{[^}]*max-height:calc\(var\(--app-height, 100vh\) \* \.88\);[^}]*env\(safe-area-inset-bottom\)[^}]*background:var\(--panel\)/);
  assert.match(desktop, /#teambody \{[^}]*min-height:0; overflow-y:auto; overscroll-behavior:contain; touch-action:pan-y;/);
  assert.match(html, /id="teampickpanel" role="dialog" aria-modal="true" aria-labelledby="teamtitle" aria-describedby="teamnote"/);
  assert.match(html, /id="teamworkerbtn"[^>]*aria-describedby="teamworkerhint"/);
  assert.match(html, /id="teamworkeroptions" role="radiogroup"/);
  assert.match(desktop, /#teamworkeroptions \.teamchoice \{[^}]*min-height:44px;/);
  assert.match(desktop, /#teamworkersearch \{[^}]*min-height:38px;/);
  assert.match(desktop, /\.teamchoice \{[^}]*width:100%;/);
  assert.match(desktop, /#teammodes \{[^}]*grid-template-columns:repeat\(2,minmax\(0,1fr\)\);/);
  assert.match(desktop, /#teammodes \.teamchoicetext small \{ display:none; \}/);
  assert.match(desktop, /#teamworkerlist \{[^}]*position:absolute;[^}]*left:12px; right:12px;/);
  assert.match(desktop, /#teamworkerlist \{[^}]*max-height:calc\(var\(--app-height, 100vh\) \* \.96\);/);
  assert.match(desktop, /#teamworkerlist \{[^}]*box-shadow:/);
  assert.match(desktop, /#teamworkerlist\[hidden\] \{ display:none; \}/);
  const workerOptionsCss = /#teamworkeroptions \{([^}]*)\}/.exec(desktop)[1];
  assert.match(workerOptionsCss, /min-height:0;/);
  assert.match(workerOptionsCss, /overflow-y:auto; overscroll-behavior:contain; touch-action:pan-y;/);
  assert.doesNotMatch(workerOptionsCss, /max-height:/);
  assert.match(css, /prefers-reduced-motion:reduce\) \{ #teampickpanel, #teampickscrim \{ animation:none; \}/);
  const metadata = /<summary id="ovrsummaryline">([\s\S]*?)<\/summary>/.exec(html)[1];
  assert.match(metadata, /<span id="ovrsummary">[^<]*<\/span>/);
  assert.match(metadata, /<span id="teamcontrols">\s*<button id="teamtrigger" type="button" aria-haspopup="dialog" aria-controls="teampickpanel" aria-expanded="false">/);
  assert.match(metadata, /<span id="teamlabel">/);
  assert.match(metadata, /<span id="teamstatus" role="status" aria-live="polite">/);
  assert.equal((html.match(/id="teamcontrols"/g) || []).length, 1);
  assert.ok(html.indexOf('<summary id="ovrsummaryline">') < html.indexOf('<div id="composer">'));
});

// A config read that failed has no other exit than a later read, so the chip
// used to stay on its retry state until the session changed. The page schedules
// its own bounded retry; these cases drive that timer instead of sleeping.
function recorder(h) {
  const timers = new Map();
  let next = 1;
  h.elements.teamstatus.textContent = "";
  h.context.document.visibilityState = "visible";
  h.context.setTimeout = (fn, ms) => { const id = next++; timers.set(id, { fn, ms }); return id; };
  h.context.clearTimeout = (id) => { timers.delete(id); };
  return timers;
}

test("a failed config read retries itself and clears the mode chip", async () => {
  let reads = 0, fail = true;
  const h = harness({ configGet: () => {
    reads++;
    if (fail) return Promise.reject(new Error("network down"));
    return response(NORMAL);
  } });
  const timers = recorder(h);
  const state = h.context.teamPickState();

  await h.context.loadSessionConfig();
  await settle();
  assert.equal(reads, 1);
  assert.equal(h.elements.teamstatus.textContent, "重试");
  assert.equal(h.elements.teamtrigger.dataset.state, "error");
  const id = state.configRetryTimer;
  assert.ok(id !== null && id !== undefined, "a failed read schedules its own retry");
  assert.equal(timers.get(id).ms, 1000);

  fail = false;
  timers.get(id).fn();
  await settle();
  await settle();
  assert.equal(reads, 2);
  assert.equal(state.configRetryTimer, null);
  assert.equal(h.elements.teamstatus.textContent, "");
  assert.equal(h.elements.teamtrigger.dataset.state, "ready");
});

test("a hidden or offline config failure waits for the next read instead of retrying", async () => {
  let fail = true;
  const h = harness({ configGet: () => fail ? Promise.reject(new Error("network down")) : response(NORMAL) });
  const timers = recorder(h);
  const state = h.context.teamPickState();
  h.context.document.visibilityState = "hidden";
  h.context.navigator = { onLine: false };

  await h.context.loadSessionConfig();
  await settle();
  assert.equal(h.elements.teamstatus.textContent, "重试");
  assert.equal(state.configRetryTimer, null);
  assert.equal(timers.size, 0);

  fail = false;
  h.context.document.visibilityState = "visible";
  h.context.navigator = { onLine: true };
  await h.context.loadSessionConfig({ teamOnly: true });
  await settle();
  assert.equal(h.elements.teamstatus.textContent, "");
});

test("a failed mode save keeps the user's retry and schedules no read", async () => {
  const h = harness({ configPost: () => Promise.reject(new Error("save down")) });
  const timers = recorder(h);
  const state = h.context.teamPickState();
  state.pending = true;
  state.choice = { mode: "team", backend: "kernel", model: "" };

  await h.context.ensureTeamConfig("alpha");
  await settle();
  assert.equal(state.errorSource, "save");
  assert.equal(h.elements.teamstatus.textContent, "重试");
  assert.equal(state.configRetryTimer, null);
  assert.equal(timers.size, 0);
});
